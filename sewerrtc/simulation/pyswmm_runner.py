from __future__ import annotations

import json
from contextlib import nullcontext
import hashlib
import os
import time
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd

from sewerrtc.control.horizon_action_features import build_action_feature_map
from sewerrtc.graph.graph_builder import khop_nodes
from .action_policies import (
    GenericActionPolicy,
    PolicyContext,
    attach_reference_nodes,
    phase_from_time,
)
from .kpi_metrics import compute_kpis
from .v42_hydraulic_recorder import record_v42_hydraulic_targets
from .runtime_contracts import (
    analyze_recovery,
    checkpoint_targets,
    controller_memory_payload,
    parse_swmm_time_options,
    sha256_file,
    try_save_hotstart,
    utc_now,
    write_csv,
    write_json,
)


def _as_float(value, default: float = 0.0) -> float:
    try:
        arr = np.asarray(value, dtype=float).reshape(-1)
        if arr.size:
            finite = arr[np.isfinite(arr)]
            if finite.size:
                return float(finite[0])
        return float(default)
    except Exception:
        try:
            return float(value)
        except Exception:
            return float(default)


def _checkpoint_file_stem(checkpoint_id: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in checkpoint_id)
    if len(safe) <= 48:
        return safe
    digest = hashlib.sha256(checkpoint_id.encode("utf-8")).hexdigest()[:16]
    return f"cp_{digest}"


def _phase_reliable_action_filter(
    table: pd.DataFrame | None,
    *,
    pattern: str,
    phase: str,
    allow_tfv_noninferior: bool = False,
    require_pfv_improvement: bool = False,
    pfv_tolerance_abs: float = 100.0,
    pfv_tolerance_frac: float = 0.005,
    tfv_tolerance_abs: float = 0.0,
    tfv_tolerance_frac: float = 0.0,
    peak_tolerance_abs: float = 0.75,
    peak_tolerance_frac: float = 0.005,
    elapsed_min: float | None = None,
    evidence_time_tolerance_min: float = 2.5,
) -> tuple[list[str], dict[str, list[str]], str, float | None]:
    """Return locally verified actuator directions, or deny interventions.

    The input is produced from exact no-control replay counterfactuals.  This
    is deliberately stricter than the generic candidate generator: missing
    pattern-phase evidence means no active override, never reopening the full
    asset pool.
    """
    if table is None or table.empty:
        return [], {}, "phase_reliability_unavailable", None
    work = table.copy()
    if "pattern" not in work or "phase" not in work:
        return [], {}, "phase_reliability_missing_columns", None
    work = work[
        work["pattern"].astype(str).eq(str(pattern))
        & work["phase"].astype(str).eq(str(phase))
    ].copy()
    if elapsed_min is not None and "elapsed_min" in work.columns:
        times = pd.to_numeric(work["elapsed_min"], errors="coerce")
        work = work[times.sub(float(elapsed_min)).abs().le(max(0.0, float(evidence_time_tolerance_min)))].copy()
    if work.empty:
        return [], {}, "phase_reliability_no_local_evidence", None
    state_threshold: float | None = None
    if "effect_TFV_H" in work:
        for col in ("effect_PFV_H", "effect_TFV_H", "effect_peak_TFV_rate_H", "reference_PFV_H", "reference_peak_TFV_rate_H"):
            work[col] = pd.to_numeric(work.get(col, 0.0), errors="coerce").fillna(0.0)
        pfv_limit = np.maximum(
            float(pfv_tolerance_abs),
            float(pfv_tolerance_frac) * work["reference_PFV_H"].clip(lower=0.0),
        )
        tfv_limit = np.maximum(
            float(tfv_tolerance_abs),
            float(tfv_tolerance_frac) * work["reference_TFV_H"].clip(lower=0.0),
        ) if "reference_TFV_H" in work else float(tfv_tolerance_abs)
        peak_limit = np.maximum(
            float(peak_tolerance_abs),
            float(peak_tolerance_frac) * work["reference_peak_TFV_rate_H"].clip(lower=0.0),
        )
        # The exact replay table is a pre-filter.  Let the online horizon gate
        # decide whether a TFV-neutral action is useful for PFV repair.
        tfv_ok = (
            work["effect_TFV_H"] <= tfv_limit
            if bool(allow_tfv_noninferior)
            else work["effect_TFV_H"] < 0.0
        )
        pfv_ok = (
            work["effect_PFV_H"] < 0.0
            if bool(require_pfv_improvement)
            else work["effect_PFV_H"] <= pfv_limit
        )
        work = work[pfv_ok & tfv_ok & (work["effect_peak_TFV_rate_H"] <= peak_limit)]
        if "priority_depth_mean" in work and not work.empty:
            state_threshold = float(pd.to_numeric(work["priority_depth_mean"], errors="coerce").dropna().min())
    else:
        for col in ("repair_safe_frac", "pfv_noninferior_frac", "tfv_improved_frac", "peak_safe_frac", "rows"):
            work[col] = pd.to_numeric(work.get(col, 0.0), errors="coerce").fillna(0.0)
        work = work[
            work["repair_safe_frac"].ge(1.0 - 1.0e-9)
            & work["pfv_noninferior_frac"].ge(1.0 - 1.0e-9)
            & work["tfv_improved_frac"].ge(1.0 - 1.0e-9)
            & work["peak_safe_frac"].ge(1.0 - 1.0e-9)
            & work["rows"].ge(1)
        ]
    directions: dict[str, list[str]] = {}
    for row in work.itertuples(index=False):
        aid = str(getattr(row, "actuator_id", "")).strip()
        direction = str(getattr(row, "action_direction", "")).strip().lower()
        if aid and direction in {"increase", "decrease"}:
            directions.setdefault(aid, [])
            if direction not in directions[aid]:
                directions[aid].append(direction)
    return list(directions), directions, "phase_reliability_exact_local", state_threshold


def _phase_reliable_action_delta_limits(
    table: pd.DataFrame | None,
    *,
    pattern: str,
    phase: str,
    allow_tfv_noninferior: bool = False,
    require_pfv_improvement: bool = False,
    pfv_tolerance_abs: float = 100.0,
    pfv_tolerance_frac: float = 0.005,
    tfv_tolerance_abs: float = 0.0,
    tfv_tolerance_frac: float = 0.0,
    peak_tolerance_abs: float = 0.75,
    peak_tolerance_frac: float = 0.005,
    elapsed_min: float | None = None,
    evidence_time_tolerance_min: float = 2.5,
) -> dict[str, dict[str, float]]:
    """Return safe tested amplitudes, preserving the direction semantics."""
    if table is None or table.empty or "action_delta" not in table:
        return {}
    work = table[
        table.get("pattern", pd.Series(dtype=str)).astype(str).eq(str(pattern))
        & table.get("phase", pd.Series(dtype=str)).astype(str).eq(str(phase))
    ].copy()
    if elapsed_min is not None and "elapsed_min" in work.columns:
        times = pd.to_numeric(work["elapsed_min"], errors="coerce")
        work = work[times.sub(float(elapsed_min)).abs().le(max(0.0, float(evidence_time_tolerance_min)))].copy()
    required = {"pattern", "phase", "actuator_id", "action_direction", "action_delta", "effect_PFV_H", "effect_TFV_H", "effect_peak_TFV_rate_H"}
    if work.empty or not required.issubset(work.columns):
        return {}
    for col in required - {"actuator_id", "action_direction"}:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    work = work.dropna(subset=["action_delta", "effect_PFV_H", "effect_TFV_H", "effect_peak_TFV_rate_H"])
    if work.empty:
        return {}
    ref_pfv = pd.to_numeric(work.get("reference_PFV_H", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    ref_tfv = pd.to_numeric(work.get("reference_TFV_H", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    ref_peak = pd.to_numeric(work.get("reference_peak_TFV_rate_H", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    pfv_limit = np.maximum(float(pfv_tolerance_abs), float(pfv_tolerance_frac) * ref_pfv)
    tfv_limit = np.maximum(float(tfv_tolerance_abs), float(tfv_tolerance_frac) * ref_tfv)
    peak_limit = np.maximum(float(peak_tolerance_abs), float(peak_tolerance_frac) * ref_peak)
    tfv_ok = work["effect_TFV_H"] <= tfv_limit if allow_tfv_noninferior else work["effect_TFV_H"] < 0.0
    pfv_ok = work["effect_PFV_H"] < 0.0 if require_pfv_improvement else work["effect_PFV_H"] <= pfv_limit
    safe = work[pfv_ok & tfv_ok & (work["effect_peak_TFV_rate_H"] <= peak_limit)]
    limits: dict[str, dict[str, float]] = {}
    for row in safe.itertuples(index=False):
        aid = str(row.actuator_id).strip()
        direction = str(row.action_direction).strip().lower()
        if aid and direction in {"increase", "decrease"}:
            limits.setdefault(aid, {})[direction] = max(
                float(limits.get(aid, {}).get(direction, 0.0)),
                abs(float(row.action_delta)),
            )
    return limits


def _phase_reliable_verified_effects(
    table: pd.DataFrame | None,
    *,
    event_id: str,
    pattern: str,
    phase: str,
    allow_tfv_noninferior: bool = False,
    require_pfv_improvement: bool = False,
    pfv_tolerance_abs: float = 100.0,
    pfv_tolerance_frac: float = 0.005,
    tfv_tolerance_abs: float = 0.0,
    tfv_tolerance_frac: float = 0.0,
    peak_tolerance_abs: float = 0.75,
    peak_tolerance_frac: float = 0.005,
    elapsed_min: float | None = None,
    evidence_time_tolerance_min: float = 2.5,
) -> dict[str, dict[str, list[dict[str, float]]]]:
    """Return event-specific safe exact effects for empirical replay gates."""
    if table is None or table.empty or "event_id" not in table.columns:
        return {}
    work = table[table["event_id"].astype(str).eq(str(event_id))].copy()
    if work.empty:
        return {}
    limits = _phase_reliable_action_delta_limits(
        work,
        pattern=pattern,
        phase=phase,
        allow_tfv_noninferior=allow_tfv_noninferior,
        require_pfv_improvement=require_pfv_improvement,
        pfv_tolerance_abs=pfv_tolerance_abs,
        pfv_tolerance_frac=pfv_tolerance_frac,
        tfv_tolerance_abs=tfv_tolerance_abs,
        tfv_tolerance_frac=tfv_tolerance_frac,
        peak_tolerance_abs=peak_tolerance_abs,
        peak_tolerance_frac=peak_tolerance_frac,
        elapsed_min=elapsed_min,
        evidence_time_tolerance_min=evidence_time_tolerance_min,
    )
    if not limits:
        return {}
    work = work[
        work.get("pattern", pd.Series(dtype=str)).astype(str).eq(str(pattern))
        & work.get("phase", pd.Series(dtype=str)).astype(str).eq(str(phase))
    ].copy()
    if elapsed_min is not None and "elapsed_min" in work.columns:
        times = pd.to_numeric(work["elapsed_min"], errors="coerce")
        work = work[times.sub(float(elapsed_min)).abs().le(max(0.0, float(evidence_time_tolerance_min)))].copy()
    required = {"pattern", "phase", "actuator_id", "action_direction", "action_delta", "effect_PFV_H", "effect_TFV_H", "effect_peak_TFV_rate_H"}
    if work.empty or not required.issubset(work.columns):
        return {}
    for col in required - {"actuator_id", "action_direction"}:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    ref_pfv = pd.to_numeric(work.get("reference_PFV_H", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    ref_tfv = pd.to_numeric(work.get("reference_TFV_H", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    ref_peak = pd.to_numeric(work.get("reference_peak_TFV_rate_H", 0.0), errors="coerce").fillna(0.0).clip(lower=0.0)
    pfv_limit = np.maximum(float(pfv_tolerance_abs), float(pfv_tolerance_frac) * ref_pfv)
    tfv_limit = np.maximum(float(tfv_tolerance_abs), float(tfv_tolerance_frac) * ref_tfv)
    peak_limit = np.maximum(float(peak_tolerance_abs), float(peak_tolerance_frac) * ref_peak)
    tfv_ok = work["effect_TFV_H"] <= tfv_limit if allow_tfv_noninferior else work["effect_TFV_H"] < 0.0
    pfv_ok = work["effect_PFV_H"] < 0.0 if require_pfv_improvement else work["effect_PFV_H"] <= pfv_limit
    safe = work[pfv_ok & tfv_ok & (work["effect_peak_TFV_rate_H"] <= peak_limit)].copy()
    result: dict[str, dict[str, list[dict[str, float]]]] = {}
    for row in safe.itertuples(index=False):
        aid = str(row.actuator_id).strip()
        direction = str(row.action_direction).strip().lower()
        if aid not in limits or direction not in limits[aid]:
            continue
        if abs(abs(float(row.action_delta)) - float(limits[aid][direction])) > 1.0e-6:
            continue
        result.setdefault(aid, {}).setdefault(direction, []).append(
            {
                "delta_abs": abs(float(row.action_delta)),
                "effect_PFV_H": float(row.effect_PFV_H),
                "effect_TFV_H": float(row.effect_TFV_H),
                "effect_peak_TFV_rate_H": float(row.effect_peak_TFV_rate_H),
            }
        )
    return result


def _zero_effect_for_reference_sequences(
    effects: pd.DataFrame,
    sequences: list[np.ndarray],
    contexts: list[dict],
) -> pd.DataFrame:
    """Enforce the candidate-minus-reference identity at deployment time.

    The effect head is learned from finite data and can retain a small bias for
    a no-control sequence.  A sequence identical to its reference is exactly
    a zero intervention by definition, so allowing that bias into the online
    reference corrupts every subsequent safety comparison.
    """
    if effects is None or effects.empty:
        return effects
    out = effects.copy()
    effect_cols = [col for col in out.columns if str(col).startswith("pred_")]
    for i, (sequence, context) in enumerate(zip(sequences, contexts)):
        reference = context.get("reference_action_sequence")
        if reference is None or i >= len(out):
            continue
        candidate_arr = np.asarray(sequence, dtype=float)
        reference_arr = np.asarray(reference, dtype=float)
        if candidate_arr.shape == reference_arr.shape and np.allclose(candidate_arr, reference_arr, rtol=0.0, atol=1.0e-6):
            out.iloc[i, out.columns.get_indexer(effect_cols)] = 0.0
    return out


def _make_horizon_surrogate_predictor(
    model_path: str | Path,
    horizon_steps: int,
    priority_indices: list[int],
    actuators: pd.DataFrame | None = None,
    priority_to_actuators: pd.DataFrame | None = None,
    device: str = "cpu",
):
    from sewerrtc.models.temporal_graph_surrogate import _feature_matrix, load_horizon_surrogate

    model = load_horizon_surrogate(model_path)
    h = max(1, int(horizon_steps))
    priority_indices = list(priority_indices or [])
    predict_device = "cuda" if str(device).lower() == "cuda" else "cpu"
    action_ids = (
        actuators["actuator_id"].astype(str).tolist()
        if actuators is not None and not actuators.empty and "actuator_id" in actuators
        else []
    )

    torch_model = None
    torch_dev = None
    if hasattr(model, "_build_model") and getattr(model, "state_dict", None) is not None:
        try:
            import torch

            torch_model, torch_dev = model._build_model(predict_device)
        except Exception:
            torch_model = None
            torch_dev = None

    def _predict_frame(frame: pd.DataFrame) -> pd.DataFrame:
        if torch_model is not None and torch_dev is not None:
            import torch

            x = _feature_matrix(frame, model.feature_columns)
            xs = (x - model.x_mean) / model.x_std
            outs = []
            with torch.no_grad():
                for start in range(0, len(xs), 4096):
                    batch = torch.tensor(xs[start : start + 4096], dtype=torch.float32, device=torch_dev)
                    outs.append(torch_model(batch).detach().cpu().numpy())
            ys = np.concatenate(outs, axis=0) if outs else np.zeros((0, len(model.target_columns)), dtype=float)
            y = ys * model.y_std + model.y_mean
            if hasattr(model, "_inverse_y"):
                y = model._inverse_y(y)
            out = pd.DataFrame(y, columns=[f"pred_{c}" for c in model.target_columns], index=frame.index)
            return out.clip(lower=0.0)
        try:
            return model.predict(frame, device=predict_device)
        except TypeError:
            return model.predict(frame)

    def _row_for_sequence(sequence: np.ndarray, context: dict) -> dict[str, float]:
        state = np.asarray(context.get("reconstructed_state", []), dtype=float).reshape(-1)
        rainfall = np.asarray(context.get("rainfall_window", []), dtype=float).reshape(-1)
        seq = np.asarray(sequence, dtype=float)
        if rainfall.size == 0:
            rainfall = np.zeros(h, dtype=float)
        if rainfall.size < h:
            rainfall = np.pad(rainfall, (0, h - rainfall.size), mode="edge")
        current_depths = state[np.isfinite(state)] if state.size else np.asarray([], dtype=float)
        if priority_indices and state.size:
            valid_priority = [i for i in priority_indices if 0 <= i < state.size]
            priority_depths = state[valid_priority] if valid_priority else current_depths
        else:
            priority_depths = current_depths
        action = seq[0] if seq.ndim == 2 and seq.shape[0] else np.asarray([], dtype=float)
        current_action = np.asarray(context.get("current_action", action), dtype=float).reshape(-1)
        reference_action_sequence = np.asarray(
            context.get("reference_action_sequence", current_action), dtype=float
        )
        row = {
            "current_depth_mean": float(np.mean(current_depths)) if current_depths.size else 0.0,
            "current_depth_p95": float(np.quantile(current_depths, 0.95)) if current_depths.size else 0.0,
            "current_depth_max": float(np.max(current_depths)) if current_depths.size else 0.0,
            "priority_depth_mean": float(np.mean(priority_depths)) if priority_depths.size else 0.0,
            "priority_depth_max": float(np.max(priority_depths)) if priority_depths.size else 0.0,
            "priority_depth_trend": 0.0,
            "rain_now": float(rainfall[0]) if rainfall.size else 0.0,
            "rain_forecast_mean": float(np.mean(rainfall[:h])) if rainfall.size else 0.0,
            "rain_forecast_max": float(np.max(rainfall[:h])) if rainfall.size else 0.0,
        }
        ids = action_ids if action_ids else [f"actuator_{i}" for i in range(int(action.size))]
        row.update(
            build_action_feature_map(
                ids,
                action,
                sequence=seq,
                reference_action=reference_action_sequence,
                actuators=actuators,
                priority_to_actuators=priority_to_actuators,
            )
        )
        return row

    def _pred_row_to_horizon(pred: pd.Series, effect: pd.Series | None = None) -> dict[str, np.ndarray]:
        pfv_total = float(pred.get("pred_PFV_H", 0.0) or 0.0)
        tfv_total = float(pred.get("pred_TFV_H", 0.0) or 0.0)
        peak = float(pred.get("pred_peak_TFV_rate_H", 0.0) or 0.0)
        if effect is not None:
            pfv_total += float(effect.get("pred_PFV_H", 0.0) or 0.0)
            tfv_total += float(effect.get("pred_TFV_H", 0.0) or 0.0)
            peak += float(effect.get("pred_peak_TFV_rate_H", 0.0) or 0.0)
        pfv_total = max(0.0, pfv_total)
        tfv_total = max(0.0, tfv_total)
        peak = max(0.0, peak)
        # Candidate-vs-reference gating is a paired effect decision. Use the
        # effect-head calibration margin when available; adding the absolute
        # target error here would make every candidate look unsafe even when
        # the relative action effect is well resolved.
        margins = (
            getattr(model, "effect_calibration_margins", {}) or {}
            if effect is not None
            else getattr(model, "calibration_margins", {}) or {}
        )
        pfv_margin = max(0.0, float(margins.get("PFV_H", 0.0)))
        tfv_margin = max(0.0, float(margins.get("TFV_H", 0.0)))
        peak_margin = max(0.0, float(margins.get("peak_TFV_rate_H", 0.0)))
        return {
            "pfv": np.full(h, pfv_total / h, dtype=float),
            "tfv": np.full(h, tfv_total / h, dtype=float),
            "peak_tfv_rate": np.full(h, peak, dtype=float),
            "pfv_upper": np.full(h, (pfv_total + pfv_margin) / h, dtype=float),
            "tfv_upper": np.full(h, (tfv_total + tfv_margin) / h, dtype=float),
            "peak_tfv_rate_upper": np.full(h, peak + peak_margin, dtype=float),
            "uncertainty_margin": np.asarray([pfv_margin, tfv_margin, peak_margin], dtype=float),
        }

    def _predict_many(sequences: list[np.ndarray], contexts: list[dict]) -> list[dict[str, np.ndarray]]:
        rows = [_row_for_sequence(sequence, context) for sequence, context in zip(sequences, contexts)]
        frame = pd.DataFrame([{col: row.get(col, 0.0) for col in model.feature_columns} for row in rows])
        preds = _predict_frame(frame)
        effects = None
        predict_effect = getattr(model, "predict_effect", None)
        if callable(predict_effect):
            try:
                effects = predict_effect(frame, device=predict_device)
            except TypeError:
                effects = predict_effect(frame)
        if effects is not None:
            effects = _zero_effect_for_reference_sequences(effects, sequences, contexts)
            base_rows = []
            use_external_reference = all(
                context.get("reference_pfv") is not None
                and context.get("reference_tfv") is not None
                and context.get("reference_peak") is not None
                for context in contexts
            )
            if use_external_reference:
                for context in contexts:
                    base_rows.append(
                        {
                            "pred_PFV_H": float(np.sum(np.maximum(0.0, np.asarray(context["reference_pfv"], dtype=float)))),
                            "pred_TFV_H": float(np.sum(np.maximum(0.0, np.asarray(context["reference_tfv"], dtype=float)))),
                            "pred_peak_TFV_rate_H": float(np.max(np.maximum(0.0, np.asarray(context["reference_peak"], dtype=float)))),
                        }
                    )
                preds = pd.DataFrame(base_rows)
                for target in model.target_columns:
                    col = f"pred_{target}"
                    if col not in preds:
                        preds[col] = 0.0
            else:
                default_rows = []
                for context in contexts:
                    current = np.asarray(context.get("current_action", []), dtype=np.float32).reshape(-1)
                    default_seq = np.asarray(
                        context.get("reference_action_sequence", np.repeat(current[None, :], h, axis=0)),
                        dtype=np.float32,
                    )
                    default_rows.append(_row_for_sequence(default_seq, context))
                default_frame = pd.DataFrame(
                    [{col: row.get(col, 0.0) for col in model.feature_columns} for row in default_rows]
                )
                preds = _predict_frame(default_frame)
        return [
            _pred_row_to_horizon(preds.iloc[i], effects.iloc[i] if effects is not None else None)
            for i in range(len(preds))
        ]

    def _predict(sequence: np.ndarray, context: dict) -> dict[str, np.ndarray]:
        return _predict_many([sequence], [context])[0]

    _predict.predict_many = _predict_many  # type: ignore[attr-defined]
    _predict.device = predict_device  # type: ignore[attr-defined]
    return _predict


def _make_horizon_ridge_predictor(
    model_path: str | Path,
    horizon_steps: int,
    priority_indices: list[int],
    actuators: pd.DataFrame | None = None,
    priority_to_actuators: pd.DataFrame | None = None,
):
    return _make_horizon_surrogate_predictor(
        model_path,
        horizon_steps,
        priority_indices,
        actuators=actuators,
        priority_to_actuators=priority_to_actuators,
    )


def _make_action_effect_ensemble_horizon_predictor(model_path: str | Path, horizon_steps: int):
    """Adapt the Project6 V3 action-effect ensemble to the closed-loop scorer.

    The formal closed loop must use the frozen Round0/1/2 action-effect model,
    but the legacy closed-loop shell expects a horizon predictor object.  This
    adapter converts candidate-vs-reference action deltas into the same compact
    eight-feature vector used by ``prompt3.action_effect_mpc`` training.
    """
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Project6 V3 action-effect ensemble missing: {path}")
    model = np.load(path, allow_pickle=False)
    weights = np.asarray(model["weights"], dtype=float)
    feature_mean = np.asarray(model["feature_mean"], dtype=float)
    feature_scale = np.maximum(np.asarray(model["feature_scale"], dtype=float), 1.0e-6)
    labels = [str(x) for x in np.asarray(model["labels"]).tolist()]
    expected = ["delta_PFV_vs_internal", "delta_TFV_vs_fallback", "delta_peak_vs_fallback"]
    if labels != expected:
        raise ValueError(f"Unexpected action-effect labels in {path}: {labels}")
    if weights.ndim != 3 or weights.shape[1:] != (9, 3):
        raise ValueError(f"Unexpected action-effect weight shape in {path}: {weights.shape}")
    if feature_mean.shape != feature_scale.shape or feature_mean.shape != (weights.shape[0], 8):
        raise ValueError(
            f"Unexpected action-effect feature stats in {path}: "
            f"mean={feature_mean.shape} scale={feature_scale.shape} members={weights.shape[0]}"
        )
    h = max(1, int(horizon_steps))

    def _feature_for(sequence: np.ndarray, context: dict) -> np.ndarray:
        seq = np.asarray(sequence, dtype=float)
        action = seq[0] if seq.ndim == 2 and seq.shape[0] else np.asarray([], dtype=float)
        reference = np.asarray(context.get("reference_action_sequence", action), dtype=float)
        reference_first = reference[0] if reference.ndim == 2 and reference.shape[0] else np.asarray(context.get("current_action", action), dtype=float)
        action = action.reshape(-1)
        reference_first = reference_first.reshape(-1)
        n = min(action.size, reference_first.size)
        delta = action[:n] - reference_first[:n] if n else np.asarray([], dtype=float)
        active = np.abs(delta) > 1.0e-6
        concurrency = int(active.sum())
        label = str(context.get("label", "")).lower()
        positive = bool((delta[active] > 0).any()) if concurrency else False
        negative = bool((delta[active] < 0).any()) if concurrency else False
        max_abs = float(np.max(np.abs(delta[active]))) if concurrency else 0.0
        if max_abs >= 0.90:
            magnitude = 1.0
        elif max_abs >= 0.50:
            magnitude = 0.75
        elif max_abs >= 0.20:
            magnitude = 0.50
        elif max_abs > 0.0:
            magnitude = 0.25
        else:
            magnitude = 0.0
        phase = str(context.get("phase", ""))
        phase_hash = (int(hashlib.sha1(phase.encode("utf-8")).hexdigest()[:4], 16) % 1000) / 1000.0
        return np.asarray(
            [
                float(concurrency),
                float(concurrency),
                1.0 if positive or "increase" in label else 0.0,
                1.0 if negative or "decrease" in label else 0.0,
                1.0 if "off_to_on" in label or "off->on" in label else 0.0,
                1.0 if "on_to_off" in label or "on->off" in label else 0.0,
                magnitude,
                phase_hash,
            ],
            dtype=float,
        )

    def _predict_many(sequences: list[np.ndarray], contexts: list[dict]) -> list[dict[str, np.ndarray]]:
        out: list[dict[str, np.ndarray]] = []
        for seq, context in zip(sequences, contexts):
            x = _feature_for(seq, context)
            member_preds = []
            for member_idx in range(weights.shape[0]):
                xn = (x - feature_mean[member_idx]) / feature_scale[member_idx]
                design = np.concatenate(([1.0], xn))
                member_preds.append(design @ weights[member_idx])
            pred = np.asarray(member_preds, dtype=float)
            mean = pred.mean(axis=0)
            std = pred.std(axis=0)
            ref_pfv = np.asarray(context.get("reference_pfv", []), dtype=float).reshape(-1)
            ref_tfv = np.asarray(context.get("reference_tfv", []), dtype=float).reshape(-1)
            ref_peak = np.asarray(context.get("reference_peak", []), dtype=float).reshape(-1)
            base_pfv = float(np.sum(np.maximum(0.0, ref_pfv))) if ref_pfv.size else 0.0
            base_tfv = float(np.sum(np.maximum(0.0, ref_tfv))) if ref_tfv.size else 0.0
            base_peak = float(np.max(np.maximum(0.0, ref_peak))) if ref_peak.size else 0.0
            pfv_total = max(0.0, base_pfv + float(mean[0]))
            tfv_total = max(0.0, base_tfv + float(mean[1]))
            peak_value = max(0.0, base_peak + float(mean[2]))
            z = 1.645
            out.append(
                {
                    "pfv": np.full(h, pfv_total / h, dtype=float),
                    "tfv": np.full(h, tfv_total / h, dtype=float),
                    "peak_tfv_rate": np.full(h, peak_value, dtype=float),
                    "pfv_upper": np.full(h, (pfv_total + z * max(0.0, float(std[0]))) / h, dtype=float),
                    "tfv_upper": np.full(h, (tfv_total + z * max(0.0, float(std[1]))) / h, dtype=float),
                    "peak_tfv_rate_upper": np.full(h, peak_value + z * max(0.0, float(std[2])), dtype=float),
                    "uncertainty_margin": z * np.maximum(0.0, std),
                }
            )
        return out

    def _predict(sequence: np.ndarray, context: dict) -> dict[str, np.ndarray]:
        return _predict_many([sequence], [context])[0]

    _predict.predict_many = _predict_many  # type: ignore[attr-defined]
    _predict.device = "cpu"  # type: ignore[attr-defined]
    _predict.source_model_path = str(path)  # type: ignore[attr-defined]
    return _predict



def _make_dual_reference_action_effect_predictor(model_path: str | Path, horizon_steps: int):
    """Load the causal V4 reference-envelope + action-residual ensemble.

    The predictor never reads a pre-run future hydraulic trajectory.  Baseline
    envelopes are forecast from the current reconstructed state and rainfall
    window, then metric-aligned residual heads are added to form candidate
    predictions.
    """
    from sewerrtc.prompt3.action_effect_v4 import (
        ACTION_FEATURE_NAMES,
        CONTEXT_FEATURE_NAMES,
        REFERENCE_LABELS,
        RESIDUAL_LABELS,
        runtime_action_features,
        runtime_context_features,
    )

    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Project6 V4 action-effect ensemble missing: {path}")
    model = np.load(path, allow_pickle=False)
    ref_weights = np.asarray(model["reference_weights"], dtype=float)
    ref_mean = np.asarray(model["reference_feature_mean"], dtype=float)
    ref_scale = np.maximum(np.asarray(model["reference_feature_scale"], dtype=float), 1.0e-6)
    res_weights = np.asarray(model["residual_weights"], dtype=float)
    res_mean = np.asarray(model["residual_feature_mean"], dtype=float)
    res_scale = np.maximum(np.asarray(model["residual_feature_scale"], dtype=float), 1.0e-6)
    ref_labels = [str(x) for x in np.asarray(model["reference_labels"]).tolist()]
    res_labels = [str(x) for x in np.asarray(model["residual_labels"]).tolist()]
    ctx_names = [str(x) for x in np.asarray(model["context_feature_names"]).tolist()]
    act_names = [str(x) for x in np.asarray(model["action_feature_names"]).tolist()]
    ref_conformal = np.maximum(np.asarray(model["reference_conformal"], dtype=float), 0.0)
    res_conformal = np.maximum(np.asarray(model["residual_conformal"], dtype=float), 0.0)
    quantile = float(np.asarray(model.get("quantile", np.asarray([0.95]))).reshape(-1)[0])
    if ref_labels != list(REFERENCE_LABELS) or res_labels != list(RESIDUAL_LABELS):
        raise ValueError(f"Unexpected V4 labels in {path}: reference={ref_labels}, residual={res_labels}")
    if ctx_names != list(CONTEXT_FEATURE_NAMES) or act_names != list(ACTION_FEATURE_NAMES):
        raise ValueError(f"Unexpected V4 feature contract in {path}")
    if ref_weights.ndim != 3 or ref_weights.shape[1:] != (len(CONTEXT_FEATURE_NAMES) + 1, len(REFERENCE_LABELS)):
        raise ValueError(f"Unexpected V4 reference weight shape: {ref_weights.shape}")
    if res_weights.ndim != 3 or res_weights.shape[1:] != (len(CONTEXT_FEATURE_NAMES) + len(ACTION_FEATURE_NAMES) + 1, len(RESIDUAL_LABELS)):
        raise ValueError(f"Unexpected V4 residual weight shape: {res_weights.shape}")
    h = max(1, int(horizon_steps))
    ref_index = {label: i for i, label in enumerate(ref_labels)}
    res_index = {label: i for i, label in enumerate(res_labels)}

    def _member_predict(weights: np.ndarray, mean: np.ndarray, scale: np.ndarray, x: np.ndarray) -> np.ndarray:
        out = []
        for member in range(weights.shape[0]):
            xn = (x - mean[member]) / scale[member]
            out.append(np.concatenate(([1.0], xn)) @ weights[member])
        return np.asarray(out, dtype=float)

    def _upper(member_values: np.ndarray, conformal: float) -> float:
        return max(0.0, float(np.quantile(member_values, quantile)) + max(0.0, float(conformal)))

    def _reference_forecast(context: dict) -> dict[str, Any]:
        x_ctx = runtime_context_features(context)
        members = np.maximum(0.0, _member_predict(ref_weights, ref_mean, ref_scale, x_ctx))
        mean = np.maximum(0.0, members.mean(axis=0))
        upper = np.asarray([_upper(members[:, i], ref_conformal[i]) for i in range(len(ref_labels))])
        return {
            "member_values": members,
            "mean_values": mean,
            "upper_values": upper,
            "no_control_pfv": np.full(h, mean[ref_index["no_control_PFV_H120"]] / h),
            "no_control_pfv_upper": np.full(h, upper[ref_index["no_control_PFV_H120"]] / h),
            "passive_pfv": np.full(h, mean[ref_index["passive_PFV_H120"]] / h),
            "passive_pfv_upper": np.full(h, upper[ref_index["passive_PFV_H120"]] / h),
            "internal_pfv": np.full(h, mean[ref_index["internal_PFV_H120"]] / h),
            "internal_pfv_upper": np.full(h, upper[ref_index["internal_PFV_H120"]] / h),
            "internal_tfv": np.full(h, mean[ref_index["internal_TFV_H120"]] / h),
            "internal_tfv_upper": np.full(h, upper[ref_index["internal_TFV_H120"]] / h),
            "internal_peak": np.full(h, mean[ref_index["internal_peak_H120"]]),
            "internal_peak_upper": np.full(h, upper[ref_index["internal_peak_H120"]]),
            "no_control_event_pfv": float(mean[ref_index["no_control_PFV_full"]]),
            "no_control_event_pfv_upper": float(upper[ref_index["no_control_PFV_full"]]),
            "passive_event_pfv": float(mean[ref_index["passive_PFV_full"]]),
            "passive_event_pfv_upper": float(upper[ref_index["passive_PFV_full"]]),
            "internal_event_pfv": float(mean[ref_index["internal_PFV_full"]]),
            "internal_event_pfv_upper": float(upper[ref_index["internal_PFV_full"]]),
            "reference_uncertainty_margin": upper - mean,
            "source": "causal_reference_envelope_model",
        }

    def _predict_many(sequences: list[np.ndarray], contexts: list[dict]) -> list[dict[str, np.ndarray]]:
        output: list[dict[str, np.ndarray]] = []
        for seq, context in zip(sequences, contexts):
            reference = context.get("causal_reference_forecast")
            if not isinstance(reference, dict):
                reference = _reference_forecast(context)
            x_ctx = runtime_context_features(context)
            x_act = runtime_action_features(seq, context)
            x = np.concatenate([x_ctx, x_act])
            res_members = _member_predict(res_weights, res_mean, res_scale, x)
            ref_members = np.asarray(reference["member_values"], dtype=float)
            m = min(len(ref_members), len(res_members))
            ref_members, res_members = ref_members[:m], res_members[:m]
            cand_nc = np.maximum(0.0, ref_members[:, ref_index["no_control_PFV_H120"]] + res_members[:, res_index["delta_PFV_H120_vs_no_control"]])
            cand_pa = np.maximum(0.0, ref_members[:, ref_index["passive_PFV_H120"]] + res_members[:, res_index["delta_PFV_H120_vs_passive"]])
            cand_pfv = np.maximum(cand_nc, cand_pa)
            cand_tfv = np.maximum(0.0, ref_members[:, ref_index["internal_TFV_H120"]] + res_members[:, res_index["delta_TFV_H120_vs_internal"]])
            cand_peak = np.maximum(0.0, ref_members[:, ref_index["internal_peak_H120"]] + res_members[:, res_index["delta_peak_H120_vs_internal"]])
            cand_event_nc = np.maximum(0.0, ref_members[:, ref_index["no_control_PFV_full"]] + res_members[:, res_index["delta_PFV_full_vs_no_control"]])
            cand_event_pa = np.maximum(0.0, ref_members[:, ref_index["passive_PFV_full"]] + res_members[:, res_index["delta_PFV_full_vs_passive"]])
            cand_event = np.maximum(cand_event_nc, cand_event_pa)
            pfv_mean, tfv_mean, peak_mean, event_mean = map(float, (cand_pfv.mean(), cand_tfv.mean(), cand_peak.mean(), cand_event.mean()))
            pfv_extra = max(res_conformal[res_index["delta_PFV_H120_vs_no_control"]], res_conformal[res_index["delta_PFV_H120_vs_passive"]])
            event_extra = max(res_conformal[res_index["delta_PFV_full_vs_no_control"]], res_conformal[res_index["delta_PFV_full_vs_passive"]])
            pfv_upper = _upper(cand_pfv, pfv_extra)
            tfv_upper = _upper(cand_tfv, res_conformal[res_index["delta_TFV_H120_vs_internal"]])
            peak_upper = _upper(cand_peak, res_conformal[res_index["delta_peak_H120_vs_internal"]])
            event_upper = _upper(cand_event, event_extra)
            output.append({
                "pfv": np.full(h, pfv_mean / h),
                "tfv": np.full(h, tfv_mean / h),
                "peak_tfv_rate": np.full(h, peak_mean),
                "pfv_upper": np.full(h, pfv_upper / h),
                "tfv_upper": np.full(h, tfv_upper / h),
                "peak_tfv_rate_upper": np.full(h, peak_upper),
                "event_pfv": np.asarray([event_mean]),
                "event_pfv_upper": np.asarray([event_upper]),
                "pfv_from_no_control": np.full(h, float(cand_nc.mean()) / h),
                "pfv_from_passive": np.full(h, float(cand_pa.mean()) / h),
                "uncertainty_margin": np.asarray([pfv_upper - pfv_mean, tfv_upper - tfv_mean, peak_upper - peak_mean, event_upper - event_mean]),
                "online_future_hydraulics_used": np.asarray([0.0]),
            })
        return output

    def _predict(sequence: np.ndarray, context: dict) -> dict[str, np.ndarray]:
        return _predict_many([sequence], [context])[0]

    _predict.predict_many = _predict_many  # type: ignore[attr-defined]
    _predict.reference_forecast = _reference_forecast  # type: ignore[attr-defined]
    _predict.device = "cpu"  # type: ignore[attr-defined]
    _predict.source_model_path = str(path)  # type: ignore[attr-defined]
    _predict.reference_contract = "causal PFV:no_control+passive; causal TFV+Peak:internal"  # type: ignore[attr-defined]
    _predict.online_future_hydraulics_forbidden = True  # type: ignore[attr-defined]
    return _predict


def _load_rainfall_forecast(rainfall_csv: str | Path | None) -> pd.DataFrame:
    if not rainfall_csv:
        return pd.DataFrame()
    path = Path(rainfall_csv)
    if not path.exists():
        return pd.DataFrame()
    try:
        rain = pd.read_csv(path)
    except Exception:
        return pd.DataFrame()
    if "elapsed_min" not in rain or "intensity_mm_h" not in rain:
        return pd.DataFrame()
    out = rain[["elapsed_min", "intensity_mm_h"]].copy()
    out["elapsed_min"] = pd.to_numeric(out["elapsed_min"], errors="coerce")
    out["intensity_mm_h"] = pd.to_numeric(out["intensity_mm_h"], errors="coerce").fillna(0.0)
    out = out.dropna(subset=["elapsed_min"]).sort_values("elapsed_min")
    return out


def _rainfall_window_at(
    rainfall_forecast: pd.DataFrame,
    elapsed_min: float,
    horizon_steps: int,
    step_min: float,
    fallback_rain: float,
) -> np.ndarray:
    h = max(1, int(horizon_steps))
    if rainfall_forecast is None or rainfall_forecast.empty:
        return np.full(h, float(fallback_rain), dtype=np.float32)
    times = rainfall_forecast["elapsed_min"].to_numpy(dtype=float)
    values = rainfall_forecast["intensity_mm_h"].to_numpy(dtype=float)
    if times.size == 0:
        return np.full(h, float(fallback_rain), dtype=np.float32)
    targets = float(elapsed_min) + np.arange(h, dtype=float) * float(step_min)
    window = np.interp(targets, times, values, left=float(values[0]), right=0.0)
    if window.size:
        window[0] = float(fallback_rain)
    return np.nan_to_num(window, nan=float(fallback_rain)).astype(np.float32)


def _reference_horizon_arrays_from_detail(
    detail: pd.DataFrame,
    *,
    elapsed_min: float,
    horizon_steps: int,
    dt_sec: int,
    priority_nodes: list[str],
) -> dict[str, np.ndarray]:
    h = max(1, int(horizon_steps))
    zeros = {
        "pfv": np.zeros(h, dtype=np.float32),
        "tfv": np.zeros(h, dtype=np.float32),
        "peak_tfv_rate": np.zeros(h, dtype=np.float32),
    }
    if detail is None or detail.empty or "elapsed_min" not in detail:
        return zeros
    frame = detail.copy()
    frame["elapsed_min"] = pd.to_numeric(frame["elapsed_min"], errors="coerce")
    frame = frame.dropna(subset=["elapsed_min"]).sort_values("elapsed_min").reset_index(drop=True)
    if frame.empty:
        return zeros
    flood_cols = [c for c in frame.columns if str(c).startswith("flood:")]
    priority_flood_cols = [f"flood:{n}" for n in priority_nodes if f"flood:{n}" in frame.columns]
    if not flood_cols:
        return zeros
    flood_rate = frame[flood_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1).to_numpy(float)
    if priority_flood_cols:
        pfv_rate = frame[priority_flood_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1).to_numpy(float)
    else:
        pfv_rate = np.zeros(len(frame), dtype=float)
    times = frame["elapsed_min"].to_numpy(float)
    start_idx = int(np.searchsorted(times, float(elapsed_min), side="left"))
    if start_idx >= len(frame):
        start_idx = len(frame) - 1
    indices = np.arange(start_idx + 1, start_idx + 1 + h, dtype=int)
    valid = indices < len(frame)
    clipped = np.clip(indices, 0, len(frame) - 1)
    pfv = pfv_rate[clipped] * float(dt_sec)
    tfv = flood_rate[clipped] * float(dt_sec)
    peak = flood_rate[clipped]
    pfv[~valid] = 0.0
    tfv[~valid] = 0.0
    peak[~valid] = 0.0
    return {
        "pfv": np.nan_to_num(pfv, nan=0.0).astype(np.float32),
        "tfv": np.nan_to_num(tfv, nan=0.0).astype(np.float32),
        "peak_tfv_rate": np.nan_to_num(peak, nan=0.0).astype(np.float32),
    }


def _constraint_reference_window(
    detail: pd.DataFrame | None,
    *,
    elapsed_min: float,
    horizon_steps: int,
    dt_sec: int,
    priority_nodes: list[str],
) -> tuple[dict[str, np.ndarray] | None, str]:
    """Use an offline twin only when one was explicitly supplied.

    An absent no-control detail belongs to online operation.  Treating it as
    a zero-risk horizon makes every non-hold candidate infeasible and is not a
    valid no-control reference.
    """
    if detail is None or detail.empty:
        return None, "online_predicted_no_control_sequence"
    return (
        _reference_horizon_arrays_from_detail(
            detail,
            elapsed_min=elapsed_min,
            horizon_steps=horizon_steps,
            dt_sec=dt_sec,
            priority_nodes=priority_nodes,
        ),
        "offline_true_no_control_twin_horizon",
    )


def _strict_reference_horizon_arrays(
    reference_details: list[pd.DataFrame],
    *,
    elapsed_min: float,
    horizon_steps: int,
    dt_sec: int,
    priority_nodes: list[str],
    fallback_pfv: float,
    fallback_tfv: float,
    fallback_peak: float,
) -> dict[str, np.ndarray]:
    h = max(1, int(horizon_steps))
    arrays = [
        _reference_horizon_arrays_from_detail(
            detail,
            elapsed_min=elapsed_min,
            horizon_steps=h,
            dt_sec=dt_sec,
            priority_nodes=priority_nodes,
        )
        for detail in reference_details
        if detail is not None and not detail.empty
    ]
    if not arrays:
        return {
            "pfv": np.full(h, float(fallback_pfv) / h, dtype=np.float32),
            "tfv": np.full(h, float(fallback_tfv) / h, dtype=np.float32),
            "peak_tfv_rate": np.full(h, float(fallback_peak), dtype=np.float32),
        }
    return {
        "pfv": np.min(np.vstack([a["pfv"] for a in arrays]), axis=0).astype(np.float32),
        "tfv": np.min(np.vstack([a["tfv"] for a in arrays]), axis=0).astype(np.float32),
        "peak_tfv_rate": np.min(np.vstack([a["peak_tfv_rate"] for a in arrays]), axis=0).astype(np.float32),
    }


def _initial_action_from_links(link_objs: dict[str, object], actuator_ids: list[str]) -> np.ndarray:
    action = np.ones(len(actuator_ids), dtype=np.float32)
    for i, aid in enumerate(actuator_ids):
        obj = link_objs.get(aid)
        if obj is None:
            continue
        value = np.nan
        for attr in ("current_setting", "target_setting"):
            try:
                value = _as_float(getattr(obj, attr), np.nan)
            except Exception:
                value = np.nan
            if np.isfinite(value):
                break
        action[i] = _as_float(np.clip(value, 0.0, 1.0), 1.0) if np.isfinite(value) else 1.0
    return action


def _observed_action_from_links(
    link_objs: dict[str, object],
    actuator_ids: list[str],
    actuators: pd.DataFrame,
) -> np.ndarray:
    """Read current physical settings without turning an unavailable pump on."""
    type_by_id = (
        actuators.set_index("actuator_id")["link_type"].astype(str).str.lower().to_dict()
        if "actuator_id" in actuators and "link_type" in actuators
        else {}
    )
    values: list[float] = []
    for aid in actuator_ids:
        fallback = 0.0 if type_by_id.get(str(aid), "") == "pump" else 1.0
        obj = link_objs.get(aid)
        value = np.nan
        if obj is not None:
            try:
                value = _as_float(getattr(obj, "current_setting"), np.nan)
            except Exception:
                value = np.nan
        values.append(float(np.clip(value, 0.0, 1.0)) if np.isfinite(value) else fallback)
    return np.asarray(values, dtype=np.float32)


def _enforce_actuator_semantics(
    action: np.ndarray,
    actuator_ids: list[str],
    actuators: pd.DataFrame,
    pump_control_mode: str = "continuous",
    variable_speed_pump_ids: Iterable[str] | None = None,
) -> np.ndarray:
    """Project a requested action into the declared physical control space.

    SWMM accepts fractional pump settings, but a pump represented by an OFF/ON
    curve in the INP is not a variable-speed pump.  Keeping such values in a
    training trajectory creates a train/serve and engineering-semantics error.
    """
    out = np.asarray(action, dtype=np.float32).reshape(-1).copy()
    if out.size != len(actuator_ids):
        out = np.resize(out, len(actuator_ids)) if out.size else np.ones(len(actuator_ids), dtype=np.float32)
    out = np.clip(np.nan_to_num(out, nan=1.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    mode = str(pump_control_mode or "continuous").strip().lower()
    # A VFD declaration is asset-specific.  Do not let an otherwise binary
    # pump inherit continuous semantics merely because another station has a
    # variable-speed drive.
    vfd_ids = {str(value) for value in (variable_speed_pump_ids or [])}
    if mode == "variable_speed":
        # Retain the legacy all-pump behavior only when no explicit registry
        # is supplied.  New configurations should use binary_unless_verified
        # together with variable_speed_pump_ids.
        if not vfd_ids:
            return out.astype(np.float32)
    elif mode not in {"binary", "binary_unless_verified", "on_off"}:
        return out.astype(np.float32)
    types = (
        actuators.set_index("actuator_id")["link_type"].astype(str).str.lower().to_dict()
        if "actuator_id" in actuators and "link_type" in actuators
        else {}
    )
    for i, aid in enumerate(actuator_ids):
        if types.get(str(aid), "") == "pump" and str(aid) not in vfd_ids:
            out[i] = 1.0 if float(out[i]) >= 0.5 else 0.0
    return out.astype(np.float32)


def _node_max_depth(obj) -> float:
    for attr in ("max_depth", "full_depth", "sur_depth"):
        try:
            val = getattr(obj, attr)
            if callable(val):
                val = val()
            f = _as_float(val, np.nan)
            if np.isfinite(f) and f > 1e-6:
                return f
        except Exception:
            continue
    return np.nan


def _require_pyswmm():
    try:
        from pyswmm import Links, Nodes, RainGages, Simulation
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "PySWMM is required for real trajectory generation. Install with `pip install pyswmm` "
            "inside the selected environment, then rerun."
        ) from exc
    return Simulation, Nodes, Links, RainGages


def _ids_from_container(container, attr: str) -> list[str]:
    ids = []
    try:
        raw = list(container)
    except Exception:
        return ids
    for item in raw:
        if isinstance(item, str):
            ids.append(item)
        elif hasattr(item, attr):
            ids.append(str(getattr(item, attr)))
        elif hasattr(item, f"{attr}id"):
            ids.append(str(getattr(item, f"{attr}id")))
        else:
            ids.append(str(item))
    return ids


def _get_existing_links(links, ids: list[str]) -> tuple[list[str], dict]:
    ok, objs = [], {}
    for lid in ids:
        try:
            objs[lid] = links[lid]
            ok.append(lid)
        except Exception:
            continue
    return ok, objs


def _load_nominal_action_table(nominal_detail_csv: str | Path | None, actuator_ids: list[str]) -> dict[float, np.ndarray]:
    if not nominal_detail_csv:
        return {}
    path = Path(nominal_detail_csv)
    if not path.exists():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if "elapsed_min" not in df:
        return {}
    action_cols = [f"a:{aid}" for aid in actuator_ids]
    for c in action_cols:
        if c not in df:
            df[c] = 1.0
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(1.0).clip(0.0, 1.0)
    table = {}
    for _, row in df.iterrows():
        key = round(float(row["elapsed_min"]), 6)
        table[key] = row[action_cols].to_numpy(dtype=np.float32)
    return table


def _load_full_prefix_table(
    baseline_detail_csv: str | Path | None,
    eng36_ids: list[str],
    extra_prefix_link_ids: list[str],
) -> tuple[dict[float, np.ndarray], dict[float, np.ndarray], list[str], list[str]]:
    """Load full prefix schedule covering Eng36 + native control links.

    Returns
    -------
    eng36_table : dict[float, ndarray]
        Eng36 actuator actions (from ``a:`` columns).
    extra_table : dict[float, ndarray]
        Extra prefix link settings (from ``setting:`` columns).
    eng36_found : list[str]
        Eng36 IDs actually found in the CSV.
    extra_found : list[str]
        Extra link IDs actually found in the CSV.
    """
    if not baseline_detail_csv:
        return {}, {}, [], []
    path = Path(baseline_detail_csv)
    if not path.exists():
        return {}, {}, [], []
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}, {}, [], []
    if "elapsed_min" not in df:
        return {}, {}, [], []

    # Eng36 a: columns
    eng36_cols = [f"a:{aid}" for aid in eng36_ids]
    eng36_found = []
    for c in eng36_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(1.0).clip(0.0, 1.0)
            eng36_found.append(c.split(":", 1)[1])
        else:
            df[c] = 1.0
            eng36_found.append(c.split(":", 1)[1])

    # Extra prefix link setting: columns
    extra_cols = [f"setting:{lid}" for lid in extra_prefix_link_ids]
    extra_found = []
    for c in extra_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(1.0).clip(0.0, 1.0)
            extra_found.append(c.split(":", 1)[1])
        else:
            df[c] = 1.0
            extra_found.append(c.split(":", 1)[1])

    eng36_table = {}
    extra_table = {}
    for _, row in df.iterrows():
        key = round(float(row["elapsed_min"]), 6)
        eng36_table[key] = row[eng36_cols].to_numpy(dtype=np.float32)
        extra_table[key] = row[extra_cols].to_numpy(dtype=np.float32)

    return eng36_table, extra_table, eng36_found, extra_found


def _nominal_action_at(table: dict[float, np.ndarray], elapsed_min: float) -> np.ndarray | None:
    if not table:
        return None
    key = round(float(elapsed_min), 6)
    if key in table:
        return table[key]
    keys = np.asarray(list(table.keys()), dtype=float)
    if keys.size == 0:
        return None
    j = int(np.argmin(np.abs(keys - float(elapsed_min))))
    if abs(float(keys[j]) - float(elapsed_min)) <= 1e-3:
        return table[float(keys[j])]
    return None


def _reference_action_sequence_from_table(
    table: dict[float, np.ndarray],
    *,
    elapsed_min: float,
    horizon_steps: int,
    dt_sec: int,
    fallback_action: np.ndarray,
) -> np.ndarray:
    """Return absolute No-control settings on the same rolling horizon."""
    fallback = np.asarray(fallback_action, dtype=np.float32).reshape(-1)
    rows = []
    step_min = float(dt_sec) / 60.0
    previous = fallback.copy()
    for k in range(max(1, int(horizon_steps))):
        value = _nominal_action_at(table, float(elapsed_min) + k * step_min)
        if value is not None:
            previous = np.asarray(value, dtype=np.float32).reshape(-1)
        rows.append(previous.copy())
    return np.vstack(rows).astype(np.float32)


def run_swmm_trajectory(
    inp_path: str | Path,
    policy_id: str,
    actuators: pd.DataFrame,
    priority_nodes: list[str],
    out_detail_csv: str | Path,
    event_id: str,
    duration_min: int,
    control_step_sec: int = 300,
    seed: int = 2026,
    max_steps: int = 0,
    simulation_duration_min: Optional[int] = None,
    recession_min: Optional[int] = None,
    pump_control_mode: str = "continuous",
    variable_speed_pump_ids: Iterable[str] | None = None,
    raw_joint_model_path: str | Path | None = None,
    temporal_joint_config: dict | None = None,
    trajectory_id: str | None = None,
    runtime_output_root: str | Path | None = None,
    extra_recording_link_ids: list[str] | None = None,
) -> dict:
    Simulation, Nodes, Links, RainGages = _require_pyswmm()
    inp_path = Path(inp_path)
    out_detail_csv = Path(out_detail_csv)
    out_detail_csv.parent.mkdir(parents=True, exist_ok=True)
    # Attach real from_node/to_node topology from the INP so filling-degree
    # policies (Auto-RBC, EFD) resolve genuine reference-node depths. Without
    # this the Project6 actuator table (which only carries storage topology for
    # a handful of retrofit assets) leaves most facilities with no reference
    # node, collapsing both rule baselines onto identical zero-depth behaviour.
    actuators = attach_reference_nodes(actuators, inp_path)
    policy = GenericActionPolicy(policy_id, actuators, seed=seed)
    trajectory_id = trajectory_id or f"{event_id}__{policy_id}"
    runtime_root = Path(runtime_output_root) if runtime_output_root is not None else out_detail_csv.parent.parent
    simulation_end_min = int(simulation_duration_min) if simulation_duration_min is not None else int(duration_min + (recession_min or 0))
    target_checkpoints = checkpoint_targets(int(duration_min), simulation_end_min, step_min=float(control_step_sec) / 60.0)
    pending_checkpoints = {float(item["elapsed_min"]): dict(item) for item in target_checkpoints}
    checkpoint_rows: list[dict[str, Any]] = []
    recovery_contract_path = runtime_root / "recovery" / str(event_id) / f"{trajectory_id}__recovery.json"
    checkpoint_manifest_stem = _checkpoint_file_stem(f"{trajectory_id}__checkpoint_manifest")
    checkpoint_manifest_path = runtime_root / "checkpoints" / str(event_id) / f"{checkpoint_manifest_stem}.csv"
    recovery_contract_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    t0 = time.time()
    # run_swmm_trajectory is the baseline trajectory runner (no_control,
    # internal_rules, passive). It never runs the V4 dual-reference
    # controller, so the shadow simulation is not needed here.
    is_v4_dual_reference_controller = False
    internal_shadow_inp_path = None
    internal_shadow_iter = None
    internal_shadow_link_objs: dict = {}
    shadow_context = (
        Simulation(str(internal_shadow_inp_path))
        if is_v4_dual_reference_controller and internal_shadow_inp_path
        else nullcontext(None)
    )
    with Simulation(str(inp_path)) as sim, shadow_context as internal_shadow_sim:
        sim.step_advance(int(control_step_sec))
        if internal_shadow_sim is not None:
            internal_shadow_sim.step_advance(int(control_step_sec))
        nodes = Nodes(sim)
        links = Links(sim)
        gages = RainGages(sim)
        node_ids = _ids_from_container(nodes, "nodeid")
        node_objs = {nid: nodes[nid] for nid in node_ids}
        node_max_depths = {nid: _node_max_depth(obj) for nid, obj in node_objs.items()}
        all_requested_actuator_ids = actuators["actuator_id"].astype(str).tolist()
        actuator_ids, link_objs = _get_existing_links(links, all_requested_actuator_ids)
        rain_ids = _ids_from_container(gages, "raingageid")
        rain_obj = gages[rain_ids[0]] if rain_ids else None
        prev_action = np.ones(len(actuator_ids), dtype=np.float32)
        override_controls = bool(getattr(policy, "enforces_targets", False)) or policy_id not in (
            "internal_rules",
            "no_control",
            "native",
        )
        step_i = 0
        for _ in sim:
            if internal_shadow_iter is not None:
                try:
                    next(internal_shadow_iter)
                except StopIteration as exc:
                    raise RuntimeError("V4 causal Internal shadow ended before Proposed branch") from exc
            elapsed_min = (sim.current_time - sim.start_time).total_seconds() / 60.0
            internal_shadow_action = (
                _observed_action_from_links(internal_shadow_link_objs, actuator_ids, actuators)
                if internal_shadow_link_objs
                else None
            )
            rain = _as_float(rain_obj.rainfall, 0.0) if rain_obj is not None else 0.0
            phase = phase_from_time(elapsed_min, duration_min)
            node_depths = {}
            for nid, obj in node_objs.items():
                try:
                    node_depths[nid] = _as_float(obj.depth, 0.0)
                except Exception:
                    node_depths[nid] = 0.0
            ctx = PolicyContext(
                elapsed_min,
                duration_min,
                rain,
                phase,
                prev_action,
                node_depths=node_depths,
                node_max_depths=node_max_depths,
            )
            action = policy.action(ctx)[: len(actuator_ids)]
            if override_controls:
                action = _enforce_actuator_semantics(
                    action, actuator_ids, actuators, pump_control_mode, variable_speed_pump_ids
                )
                for j, aid in enumerate(actuator_ids):
                    try:
                        link_objs[aid].target_setting = _as_float(np.clip(action[j], 0.0, 1.0), 1.0)
                    except Exception:
                        pass
            else:
                current = []
                for aid in actuator_ids:
                    try:
                        current.append(_as_float(link_objs[aid].current_setting, 1.0))
                    except Exception:
                        current.append(1.0)
                action = np.asarray(current, dtype=np.float32)
                action = _enforce_actuator_semantics(
                    action, actuator_ids, actuators, pump_control_mode, variable_speed_pump_ids
                )
            row = {
                "event_id": event_id,
                "policy_id": policy_id,
                "elapsed_min": elapsed_min,
                "datetime": str(sim.current_time),
                "rainfall_mm_h": rain,
                "phase": phase,
            }
            for nid, obj in node_objs.items():
                try:
                    row[f"h:{nid}"] = _as_float(obj.depth, np.nan)
                    row[f"head:{nid}"] = _as_float(getattr(obj, "head", np.nan), np.nan)
                    row[f"storage_volume:{nid}"] = _as_float(getattr(obj, "volume", np.nan), np.nan)
                    row[f"flood:{nid}"] = _as_float(obj.flooding, 0.0)
                except Exception:
                    row[f"h:{nid}"] = np.nan
                    row[f"head:{nid}"] = np.nan
                    row[f"storage_volume:{nid}"] = np.nan
                    row[f"flood:{nid}"] = 0.0
            for j, aid in enumerate(actuator_ids):
                row[f"a:{aid}"] = _as_float(action[j], 1.0)
                try:
                    row[f"flow:{aid}"] = _as_float(link_objs[aid].flow, np.nan)
                    row[f"setting:{aid}"] = _as_float(link_objs[aid].current_setting, np.nan)
                except Exception:
                    row[f"flow:{aid}"] = np.nan
                    row[f"setting:{aid}"] = np.nan
            for aid in all_requested_actuator_ids:
                row.setdefault(f"a:{aid}", np.nan)
                row.setdefault(f"flow:{aid}", np.nan)
                row.setdefault(f"setting:{aid}", np.nan)
            # Extra recording links (native control links for V3+ prefix)
            if extra_recording_link_ids:
                for elid in extra_recording_link_ids:
                    try:
                        row[f"setting:{elid}"] = _as_float(links[elid].current_setting, np.nan)
                    except Exception:
                        row[f"setting:{elid}"] = np.nan
            records.append(row)
            for target_elapsed in sorted(list(pending_checkpoints)):
                if elapsed_min + 1.0e-6 >= target_elapsed:
                    meta = pending_checkpoints.pop(target_elapsed)
                    phase_id = str(meta.get("phase", phase))
                    checkpoint_id = f"{trajectory_id}__{phase_id}__{int(round(elapsed_min)):04d}m"
                    checkpoint_file_stem = _checkpoint_file_stem(checkpoint_id)
                    hotstart_path = runtime_root / "checkpoints" / str(event_id) / "hotstart" / f"{checkpoint_file_stem}.hsf"
                    memory_path = runtime_root / "checkpoints" / str(event_id) / "controller_memory" / f"{checkpoint_file_stem}.json"
                    hotstart = try_save_hotstart(sim, hotstart_path)
                    memory_payload = controller_memory_payload(
                        trajectory_id=trajectory_id,
                        event_id=event_id,
                        policy_id=policy_id,
                        elapsed_min=float(elapsed_min),
                        row=row,
                        actuator_ids=all_requested_actuator_ids,
                        phase=phase_id,
                    )
                    write_json(memory_path, memory_payload)
                    checkpoint_rows.append(
                        {
                            "checkpoint_id": checkpoint_id,
                            "checkpoint_file_stem": checkpoint_file_stem,
                            "trajectory_id": trajectory_id,
                            "event_id": event_id,
                            "policy_id": policy_id,
                            "detail_file": str(out_detail_csv),
                            "phase": phase_id,
                            "checkpoint_elapsed_min": float(elapsed_min),
                            "hotstart_path": hotstart.get("path", ""),
                            "hotstart_sha256": hotstart.get("sha256") or "",
                            "hotstart_save_status": hotstart.get("status", ""),
                            "controller_memory_path": str(memory_path),
                            "controller_memory_sha256": sha256_file(memory_path) or "",
                            "network_path": str(inp_path),
                            "network_sha256": sha256_file(inp_path) or "",
                            "rainfall_state_hash": sha256_file(inp_path) or "",
                            "future_120min_available": str(float(elapsed_min) + 120.0 <= simulation_end_min).lower(),
                            "history_60min_available": str(float(elapsed_min) >= 60.0).lower(),
                            "runtime_clone_eligible": str(bool(hotstart.get("sha256")) and float(elapsed_min) >= 60.0 and float(elapsed_min) + 120.0 <= simulation_end_min).lower(),
                        }
                    )
            prev_action = action.copy()
            step_i += 1
            if max_steps and step_i >= max_steps:
                sim.terminate_simulation()
                break
    detail = pd.DataFrame(records)
    detail.to_csv(out_detail_csv, index=False)
    recovery = analyze_recovery(
        detail,
        event_id=event_id,
        policy_id=policy_id,
        trajectory_id=trajectory_id,
        duration_min=int(duration_min),
        minimum_tail_min=int(recession_min) if recession_min is not None else 180,
        priority_nodes=priority_nodes,
    )
    write_json(recovery_contract_path, recovery)
    write_csv(checkpoint_manifest_path, checkpoint_rows)
    kpis = compute_kpis(detail, priority_nodes, dt_sec=control_step_sec)
    kpis.update(
        {
            "event_id": event_id,
            "policy_id": policy_id,
            "duration_min": duration_min,
            "rain_duration_min": duration_min,
            "recession_min": int(recession_min) if recession_min is not None else None,
            "simulation_duration_min": int(simulation_duration_min)
            if simulation_duration_min is not None
            else (float(detail["elapsed_min"].max()) + control_step_sec / 60.0 if not detail.empty else duration_min),
            "detail_file": str(out_detail_csv),
            "rows": len(detail),
            "wall_time_sec": time.time() - t0,
            "trajectory_id": trajectory_id,
            "recovery_contract_path": str(recovery_contract_path),
            "recovery_contract_sha256": sha256_file(recovery_contract_path),
            "recovery_criteria_met": recovery.get("recovery_criteria_met"),
            "recovery_censored": recovery.get("recovery_censored"),
            "actual_tail_min": recovery.get("actual_tail_min"),
            "tail_termination_reason": recovery.get("tail_termination_reason"),
            "checkpoint_manifest_file": str(checkpoint_manifest_path),
            "checkpoint_manifest_sha256": sha256_file(checkpoint_manifest_path),
            "checkpoint_count": len(checkpoint_rows),
            "runtime_clone_eligible_checkpoint_count": sum(1 for item in checkpoint_rows if str(item.get("runtime_clone_eligible", "")).lower() == "true"),
            "swmm_time_options": json.dumps(parse_swmm_time_options(inp_path), sort_keys=True),
        }
    )
    return kpis


def run_swmm_mpc_closed_loop(
    inp_path: str | Path,
    actuators: pd.DataFrame,
    priority_nodes: list[str],
    out_detail_csv: str | Path,
    out_history_csv: str | Path,
    event_id: str,
    duration_min: int,
    gat_model_path: str | Path | None,
    surrogate_model_path: str | Path | None,
    sensor_node_ids: list[str],
    node_order: list[str],
    control_step_sec: int = 300,
    device: str = "cpu",
    max_steps: int = 0,
    nominal_detail_csv: str | Path | None = None,
    no_control_detail_csv: str | Path | None = None,
    passive_detail_csv: str | Path | None = None,
    internal_shadow_inp_path: str | Path | None = None,
    residual_value_path: str | Path | None = None,
    residual_pfv_prob_min: float = 0.60,
    residual_safe_prob_min: float = 0.70,
    residual_nonzero_prob_min: float = 0.45,
    residual_peak_prob_min: float = 0.60,
    max_candidate_delta: float = 0.08,
    topk_log_count: int = 8,
    max_candidate_count: int = 96,
    candidate_hold_steps: tuple[int, ...] | list[int] | str | None = None,
    allowed_candidate_templates: set[str] | list[str] | tuple[str, ...] | str | None = None,
    blocked_candidate_templates: set[str] | list[str] | tuple[str, ...] | str | None = None,
    allowed_candidate_scopes_by_template: dict[str, set[str] | list[str] | tuple[str, ...] | str] | None = None,
    priority_khop: int = 3,
    empirical_guard_path: str | Path | None = None,
    empirical_guard_unknown_allow: bool = True,
    boost_safe_prob_extra: float = 0.12,
    boost_peak_prob_extra: float = 0.10,
    protective_safe_prob_relief: float = 0.05,
    release_peak_hold_max: int = 1,
    low_risk_pfv_threshold: float = 1000.0,
    high_risk_pfv_threshold: float = 20000.0,
    release_recession_pfv_min: float = 500.0,
    release_recession_priority_depth_min: float = 1.0,
    strict_guard_return_period_max: int = 15,
    strict_guard_patterns: str = "chicago_late,block,double_peak",
    strict_guard_prob_extra: float = 0.10,
    horizon_smooth_weight: float = 0.05,
    horizon_violation_penalty: float = 1.0e6,
    proposed_controller: str = "native_shield",
    horizon_steps: int = 6,
    priority_to_actuators_csv: str | Path | None = None,
    horizon_surrogate_model_path: str | Path | None = None,
    rainfall_csv: str | Path | None = None,
    generic_default_policy_id: str = "hold_previous_or_all_open_safe",
    min_pfv_improvement_abs: float = 1.0,
    min_pfv_improvement_frac: float = 0.0,
    max_candidate_sequences: int = 512,
    candidate_group_limit: int = 12,
    tfv_tolerance_abs: float = 0.0,
    tfv_tolerance_frac: float = 0.0,
    peak_tolerance_abs: float = 0.0,
    peak_tolerance_frac: float = 0.0,
    pfv_tolerance_abs: float = 0.0,
    pfv_tolerance_frac: float = 0.0,
    tfv_required_reduction_abs: float = 0.0,
    tfv_required_reduction_frac: float = 0.0,
    tfv_required_reduction_dry_multiplier: float = 1.0,
    tfv_hard_constraint: bool = True,
    dry_rain_threshold: float = 0.10,
    peak_weight: float = 1.0,
    pfv_weight: float = 1.0,
    adaptive_delta_enabled: bool = False,
    low_risk_max_candidate_delta: float = 0.08,
    high_risk_max_candidate_delta: float = 0.03,
    pfv_high_risk_horizon_threshold: float = 1000.0,
    pfv_low_risk_horizon_threshold: float = 100.0,
    max_first_step_delta: float = 1.0,
    per_actuator_max_delta: dict[str, float] | None = None,
    min_hold_steps_by_actuator: dict[str, int] | None = None,
    objective_mode: str = "pfv_first",
    allowed_actuator_ids: set[str] | list[str] | tuple[str, ...] | str | None = None,
    blocked_actuator_ids: set[str] | list[str] | tuple[str, ...] | str | None = None,
    allowed_action_directions: dict[str, set[str] | list[str] | tuple[str, ...]] | None = None,
    phase_reliability_csv: str | Path | None = None,
    phase_reliability_allow_tfv_noninferior: bool = False,
    phase_reliability_require_pfv_improvement: bool = False,
    phase_reliability_pfv_tolerance_abs: float = 100.0,
    phase_reliability_pfv_tolerance_frac: float = 0.005,
    phase_reliability_tfv_tolerance_abs: float = 0.0,
    phase_reliability_tfv_tolerance_frac: float = 0.0,
    phase_reliability_peak_tolerance_abs: float = 0.75,
    phase_reliability_peak_tolerance_frac: float = 0.005,
    phase_reliability_pulse_steps: int = 2,
    phase_reliability_max_overrides: int = 0,
    phase_reliability_candidate_group_limit: int = 4,
    empirical_single_action_gate: bool = False,
    empirical_hold_steps: int = 2,
    phase_reliability_fallback_to_surrogate: bool = False,
    phase_reliability_evidence_time_tolerance_min: float = 2.5,
    action_effect_model_path: str | Path | None = None,
    pump_control_mode: str = "continuous",
    variable_speed_pump_ids: Iterable[str] | None = None,
    raw_joint_model_path: str | Path | None = None,
    temporal_joint_config: dict | None = None,
    v4_quantile: float = 0.95,
    v4_pfv_abs_margin_m3: float = 0.0,
    v4_pfv_rel_margin: float = 0.0,
    v4_max_k: int = 8,
    v4_readback_tolerance: float = 1.0e-4,
    v4_action_deadband: float = 0.02,
    v4_adaptive_k_values: Iterable[int] | None = None,
    v4_changed_facility_penalty: float = 1.0,
    v4_variation_penalty: float = 1.0,
    v4_reversal_penalty: float = 5.0,
    v4_minimum_material_benefit: float = 0.0,
    v4_minimum_benefit_cost_ratio: float = 1.5,
) -> dict:
    import torch

    from sewerrtc.control.generic_gat_mpc import GenericGATMPCController
    from sewerrtc.control.mpc_controller import PFVFirstMPC
    from sewerrtc.control.temporal_joint_36_controller import TemporalJoint36Controller
    from sewerrtc.control.temporal_joint_candidate_search import TemporalJointCandidateConfig
    from sewerrtc.control.temporal_joint_safety import JointSafetyConfig
    from sewerrtc.models.gat_reconstructor import SparseGATReconstructor
    from sewerrtc.models.raw_joint_online_predictor import RawJointOnlinePredictor
    from sewerrtc.control.no_control_reference_predictor import constant_default_action_sequence
    from sewerrtc.control.dual_reference_v4 import (
        DualReferenceLimits, EventQuantilePfvBudget, HydraulicPhase,
        adaptive_k, classify_phase, choose_phase_aware_fallback, enforce_final_readback,
    )

    Simulation, Nodes, Links, RainGages = _require_pyswmm()
    inp_path = Path(inp_path)
    out_detail_csv = Path(out_detail_csv)
    out_history_csv = Path(out_history_csv)
    out_detail_csv.parent.mkdir(parents=True, exist_ok=True)
    out_history_csv.parent.mkdir(parents=True, exist_ok=True)
    dev = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
    node_order = list(node_order)
    node_index = {n: i for i, n in enumerate(node_order)}
    priority_idx = [node_index[n] for n in priority_nodes if n in node_index]
    priority_upstream_nodes: set[str] = set()
    priority_downstream_nodes: set[str] = set()
    try:
        if surrogate_model_path:
            audit_dir = Path(surrogate_model_path).resolve().parents[1] / "audit"
            link_table_path = audit_dir / "link_table.csv"
            if link_table_path.exists():
                links_for_scope = pd.read_csv(link_table_path)
                priority_upstream_nodes = khop_nodes(
                    links_for_scope,
                    priority_nodes,
                    k=int(priority_khop),
                    direction="upstream",
                )
                priority_downstream_nodes = khop_nodes(
                    links_for_scope,
                    priority_nodes,
                    k=int(priority_khop),
                    direction="downstream",
                )
    except Exception:
        priority_upstream_nodes = set()
        priority_downstream_nodes = set()
    gat = None
    gat_node_static = None
    gat_edge_index = None
    gat_ckpt = None
    if gat_model_path and Path(gat_model_path).exists():
        gat_ckpt = torch.load(gat_model_path, map_location=dev, weights_only=False)
        if "node_ids" in gat_ckpt:
            node_order = [str(x) for x in gat_ckpt["node_ids"]]
        gat_node_static = torch.tensor(gat_ckpt["node_static"], dtype=torch.float32, device=dev)
        gat_edge_index = torch.tensor(gat_ckpt["edge_index"], dtype=torch.long, device=dev)
        gat = SparseGATReconstructor(
            int(gat_ckpt.get("n_nodes", len(node_order))),
            int(gat_ckpt.get("static_dim", gat_node_static.shape[1])),
            int(gat_ckpt.get("hidden_dim", 256)),
            int(gat_ckpt.get("gat_heads", 4)),
        ).to(dev)
        gat.load_state_dict(gat_ckpt["model"])
        gat.eval()
    # A checkpoint may define a different canonical order than the audit CSV.
    # Every downstream index must be rebuilt from that canonical order.
    node_index = {n: i for i, n in enumerate(node_order)}
    priority_idx = [node_index[n] for n in priority_nodes if n in node_index]
    sensor_idx = [node_index[n] for n in sensor_node_ids if n in node_index]
    proposed_controller = str(proposed_controller or "native_shield")
    is_v3_action_effect_controller = proposed_controller == "proposed_pfvfirst_dualfallback_v3"
    is_v4_dual_reference_controller = proposed_controller == "proposed_dual_reference_v4"
    is_generic_like_controller = proposed_controller in {"generic_gat_mpc", "proposed_pfvfirst_dualfallback_v3", "proposed_dual_reference_v4"}
    nominal_kpis = {}
    nominal_detail_df = pd.DataFrame()
    if nominal_detail_csv and Path(nominal_detail_csv).exists():
        try:
            nominal_detail_df = pd.read_csv(nominal_detail_csv)
            nominal_kpis = compute_kpis(nominal_detail_df, priority_nodes, dt_sec=control_step_sec)
        except Exception:
            nominal_kpis = {}
            nominal_detail_df = pd.DataFrame()
    no_control_kpis = {}
    no_control_detail_df = pd.DataFrame()
    if no_control_detail_csv and Path(no_control_detail_csv).exists():
        try:
            no_control_detail_df = pd.read_csv(no_control_detail_csv)
            no_control_kpis = compute_kpis(no_control_detail_df, priority_nodes, dt_sec=control_step_sec)
        except Exception:
            no_control_kpis = {}
            no_control_detail_df = pd.DataFrame()
    passive_kpis = {}
    passive_detail_df = pd.DataFrame()
    if passive_detail_csv and Path(passive_detail_csv).exists():
        try:
            passive_detail_df = pd.read_csv(passive_detail_csv)
            passive_kpis = compute_kpis(passive_detail_df, priority_nodes, dt_sec=control_step_sec)
        except Exception:
            passive_kpis = {}
            passive_detail_df = pd.DataFrame()
    if is_generic_like_controller:
        reference_kpis = [k for k in (nominal_kpis, no_control_kpis, passive_kpis) if k]
    else:
        reference_kpis = [nominal_kpis] if nominal_kpis else []
    reference_mode_parts: list[str] = []
    if nominal_kpis:
        reference_mode_parts.append("internal")
    if no_control_kpis:
        reference_mode_parts.append("no_control")
    if passive_kpis:
        reference_mode_parts.append("passive")
    if reference_mode_parts:
        constraint_reference_mode = "rolling_horizon_" + "_and_".join(reference_mode_parts)
    else:
        constraint_reference_mode = "fallback_scalar_reference"

    def _strict_reference(metric: str, default: float) -> float:
        vals = []
        for kpi in reference_kpis:
            val = _as_float(kpi.get(metric, np.nan), np.nan)
            if np.isfinite(val):
                vals.append(float(val))
        if not vals:
            return float(default)
        return float(min(vals))

    baseline_tfv_reference = _strict_reference("TFV", 1.0)
    baseline_peak_reference = _strict_reference("peak_TFV_rate", 1.0)
    baseline_pfv_reference = _strict_reference("PFV", 0.0)
    rolling_reference_details = []
    if is_generic_like_controller:
        for detail_frame in (nominal_detail_df, no_control_detail_df):
            if detail_frame is not None and not detail_frame.empty:
                rolling_reference_details.append(detail_frame)
    priority_to_actuators = None
    if priority_to_actuators_csv and Path(priority_to_actuators_csv).exists():
        try:
            priority_to_actuators = pd.read_csv(priority_to_actuators_csv)
        except Exception:
            priority_to_actuators = None
    rainfall_forecast = _load_rainfall_forecast(rainfall_csv)
    phase_reliability = None
    loaded_phase_reliability_pulse_steps = 0
    loaded_phase_reliability_max_overrides = 0
    if phase_reliability_csv and Path(phase_reliability_csv).exists():
        try:
            phase_reliability = pd.read_csv(phase_reliability_csv)
            # Exact local data generated under a former configuration may
            # contain fractional settings for ordinary OFF/ON pumps.  Those
            # rows are not executable evidence after physical semantics are
            # corrected, so exclude them before they can authorize a replay
            # action.  Explicit VFDs remain eligible.
            if phase_reliability is not None and "actuator_id" in phase_reliability.columns:
                mode = str(pump_control_mode or "continuous").strip().lower()
                vfd_ids = {str(x) for x in (variable_speed_pump_ids or [])}
                if mode in {"binary", "binary_unless_verified", "on_off"} or (
                    mode == "variable_speed" and vfd_ids
                ):
                    type_by_id = (
                        actuators.set_index("actuator_id")["link_type"].astype(str).str.lower().to_dict()
                        if "actuator_id" in actuators and "link_type" in actuators
                        else {}
                    )
                    binary_pumps = {
                        aid for aid, link_type in type_by_id.items()
                        if link_type == "pump" and aid not in vfd_ids
                    }
                    phase_reliability = phase_reliability[
                        ~phase_reliability["actuator_id"].astype(str).isin(binary_pumps)
                    ].copy()
            loaded_phase_reliability_pulse_steps = max(1, int(phase_reliability_pulse_steps))
            # A non-positive value means unlimited verified overrides. The
            # horizon safety gate still runs at every control instant.
            loaded_phase_reliability_max_overrides = max(0, int(phase_reliability_max_overrides))
        except Exception:
            phase_reliability = None
    if is_generic_like_controller:
        horizon_predictor = None
        horizon_predictor_source = "proxy_hydraulic_heuristic"
        if is_v4_dual_reference_controller:
            if not action_effect_model_path:
                raise FileNotFoundError("proposed_dual_reference_v4 requires action_effect_model_path")
            action_effect_path = Path(action_effect_model_path)
            if not action_effect_path.exists():
                raise FileNotFoundError(f"Project6 V4 dual-reference ensemble not found: {action_effect_path}")
            horizon_predictor = _make_dual_reference_action_effect_predictor(action_effect_path, int(horizon_steps))
            horizon_predictor_source = f"project6_v4_dual_reference_ensemble:{action_effect_path}"
        elif is_v3_action_effect_controller:
            if not action_effect_model_path:
                raise FileNotFoundError("proposed_pfvfirst_dualfallback_v3 requires action_effect_model_path")
            action_effect_path = Path(action_effect_model_path)
            if not action_effect_path.exists():
                raise FileNotFoundError(f"Project6 V3 action-effect ensemble not found: {action_effect_path}")
            horizon_predictor = _make_action_effect_ensemble_horizon_predictor(
                action_effect_path,
                int(horizon_steps),
            )
            horizon_predictor_source = f"project6_v3_action_effect_ensemble:{action_effect_path}"
        elif horizon_surrogate_model_path:
            horizon_model_path = Path(horizon_surrogate_model_path)
        elif surrogate_model_path:
            model_dir = Path(surrogate_model_path).resolve().parent
            temporal_candidate = model_dir / "horizon_temporal_gnn.pt"
            ridge_candidate = model_dir / "horizon_ridge_surrogate.npz"
            horizon_model_path = temporal_candidate if temporal_candidate.exists() else ridge_candidate
        else:
            temporal_candidate = Path("horizon_temporal_gnn.pt")
            horizon_model_path = temporal_candidate if temporal_candidate.exists() else Path("horizon_ridge_surrogate.npz")
        if not is_v3_action_effect_controller and not is_v4_dual_reference_controller and horizon_model_path.exists():
            horizon_predictor = _make_horizon_surrogate_predictor(
                horizon_model_path,
                int(horizon_steps),
                priority_idx,
                actuators=actuators,
                priority_to_actuators=priority_to_actuators,
                device=str(device),
            )
            horizon_device = getattr(horizon_predictor, "device", str(device))
            horizon_predictor_source = f"horizon_surrogate:{horizon_model_path};device={horizon_device}"
        mpc = GenericGATMPCController(
            actuators,
            horizon_steps=int(horizon_steps),
            max_candidate_delta=float(max_candidate_delta),
            priority_to_actuators=priority_to_actuators,
            horizon_predictor=horizon_predictor,
            predictor_source=horizon_predictor_source,
            smooth_weight=float(horizon_smooth_weight),
            violation_penalty=float(horizon_violation_penalty),
            min_pfv_improvement_abs=float(min_pfv_improvement_abs),
            min_pfv_improvement_frac=float(min_pfv_improvement_frac),
            max_candidate_sequences=int(max_candidate_sequences),
            candidate_group_limit=int(candidate_group_limit),
            tfv_tolerance_abs=float(tfv_tolerance_abs),
            tfv_tolerance_frac=float(tfv_tolerance_frac),
            peak_tolerance_abs=float(peak_tolerance_abs),
            peak_tolerance_frac=float(peak_tolerance_frac),
            pfv_tolerance_abs=float(pfv_tolerance_abs),
            pfv_tolerance_frac=float(pfv_tolerance_frac),
            tfv_required_reduction_abs=float(tfv_required_reduction_abs),
            tfv_required_reduction_frac=float(tfv_required_reduction_frac),
            tfv_required_reduction_dry_multiplier=float(tfv_required_reduction_dry_multiplier),
            tfv_hard_constraint=bool(tfv_hard_constraint),
            dry_rain_threshold=float(dry_rain_threshold),
            peak_weight=float(peak_weight),
            pfv_weight=float(pfv_weight),
            adaptive_delta_enabled=bool(adaptive_delta_enabled),
            low_risk_max_candidate_delta=float(low_risk_max_candidate_delta),
            high_risk_max_candidate_delta=float(high_risk_max_candidate_delta),
            pfv_high_risk_horizon_threshold=float(pfv_high_risk_horizon_threshold),
            pfv_low_risk_horizon_threshold=float(pfv_low_risk_horizon_threshold),
            max_first_step_delta=float(max_first_step_delta),
            per_actuator_max_delta=dict(per_actuator_max_delta or {}),
            min_hold_steps_by_actuator=dict(min_hold_steps_by_actuator or {}),
            objective_mode=str(objective_mode),
            allowed_actuator_ids=allowed_actuator_ids,
            blocked_actuator_ids=blocked_actuator_ids,
            allowed_action_directions=allowed_action_directions,
            empirical_single_action_gate=bool(empirical_single_action_gate),
            empirical_hold_steps=max(1, int(empirical_hold_steps)),
            pump_control_mode=str(pump_control_mode),
            variable_speed_pump_ids=variable_speed_pump_ids,
        )
        proposed_policy_id = (
            "proposed_pfvfirst_dualfallback_v3"
            if is_v3_action_effect_controller
            else "proposed_gat_mpc"
        )
        generic_default_policy_name = str(generic_default_policy_id or "hold_previous_or_all_open_safe")
        generic_default_policy = (
            None
            if generic_default_policy_name in {"", "hold_previous_or_all_open_safe", "hold_previous"}
            else GenericActionPolicy(generic_default_policy_name, actuators)
        )
    elif proposed_controller in {"temporal_joint_36", "hierarchical_core26_residual10"}:
        settings = dict(temporal_joint_config or {})
        if len(actuators) != 36:
            raise ValueError(f"temporal_joint_36 requires exactly 36 canonical actuators, got {len(actuators)}")
        if not raw_joint_model_path:
            raise FileNotFoundError("temporal_joint_36 requires a verified --raw-joint-model checkpoint")
        action_ids = actuators["actuator_id"].astype(str).tolist()
        predictor = RawJointOnlinePredictor(
            raw_joint_model_path,
            canonical_action_ids=action_ids,
            device=str(device),
            batch_size=int(settings.get("predict_batch_size", 256)),
        )
        candidate_settings = dict(settings.get("candidate_search", {}) or {})
        safety_settings = dict(settings.get("safety", {}) or {})
        hierarchical_settings = dict(settings.get("hierarchical", {}) or {})
        temporal_control_step_min = max(1.0e-6, float(control_step_sec) / 60.0)
        prediction_horizon_steps = int(
            settings.get(
                "prediction_horizon_steps",
                round(float(settings.get("prediction_horizon_min", 120.0)) / temporal_control_step_min),
            )
        )
        prediction_horizon_steps = max(int(horizon_steps), prediction_horizon_steps)
        terminal_return_steps = int(
            settings.get(
                "terminal_return_steps",
                round(float(settings.get("terminal_return_min", 30.0)) / temporal_control_step_min),
            )
        )
        engineering_templates = list(candidate_settings.get("engineering_templates", []) or [])
        template_path_value = str(candidate_settings.get("engineering_template_path", "") or "")
        if template_path_value:
            template_path = Path(template_path_value)
            if template_path.exists():
                loaded_templates = json.loads(template_path.read_text(encoding="utf-8"))
                if isinstance(loaded_templates, dict):
                    loaded_templates = loaded_templates.get("templates", [])
                engineering_templates.extend(list(loaded_templates or []))
        candidate_config = TemporalJointCandidateConfig(
            horizon_steps=int(horizon_steps),
            max_candidates=int(candidate_settings.get("max_candidates", max_candidate_sequences)),
            max_simultaneous_changes=int(candidate_settings.get("max_simultaneous_changes", 6)),
            max_change_points=int(candidate_settings.get("max_change_points", 2)),
            continuous_max_delta=float(candidate_settings.get("continuous_max_delta", max_candidate_delta)),
            continuous_delta_levels=tuple(candidate_settings.get("continuous_delta_levels", [])),
            binary_pump_ids=tuple(candidate_settings.get("binary_pump_ids", ["ADD301.2", "ADD301.3"])),
            binary_pump_min_dwell_steps=int(candidate_settings.get("binary_pump_min_dwell_steps", 2)),
            max_pump_switches_per_event=int(candidate_settings.get("max_pump_switches_per_event", 6)),
            storage_interlock=bool(candidate_settings.get("storage_interlock", True)),
            max_storage_actuators=int(candidate_settings.get("max_storage_actuators", 4)),
            allowed_candidate_ids=tuple(candidate_settings.get("allowed_candidate_ids", [])),
            engineering_templates=tuple(engineering_templates),
        )
        safety_config = JointSafetyConfig(
            pfv_abs_margin_m3=float(safety_settings.get("pfv_abs_margin_m3", 100.0)),
            pfv_rel_margin=float(safety_settings.get("pfv_rel_margin", 0.005)),
            event_pfv_budget_enabled=bool(safety_settings.get("event_pfv_budget_enabled", False)),
            event_pfv_abs_margin_m3=float(safety_settings.get("event_pfv_abs_margin_m3", 100.0)),
            event_pfv_rel_margin=float(safety_settings.get("event_pfv_rel_margin", 0.005)),
            peak_margin=float(safety_settings.get("peak_margin", 0.0)),
            uncertainty_z=float(safety_settings.get("uncertainty_z", 1.645)),
            min_tfv_lcb_reduction=float(safety_settings.get("min_tfv_lcb_reduction", 0.0)),
            min_pfv_noninferiority_probability=float(safety_settings.get("min_pfv_noninferiority_probability", 0.0)),
            min_tfv_improvement_probability=float(safety_settings.get("min_tfv_improvement_probability", 0.0)),
            min_peak_safe_probability=float(safety_settings.get("min_peak_safe_probability", 0.0)),
            use_classifier_thresholds=bool(safety_settings.get("use_classifier_thresholds", True)),
            tfv_reduction_weight=float(safety_settings.get("tfv_reduction_weight", 1.0)),
            peak_reduction_weight=float(safety_settings.get("peak_reduction_weight", 0.0)),
            action_l1_penalty=float(safety_settings.get("action_l1_penalty", 0.0)),
            simultaneous_action_penalty=float(safety_settings.get("simultaneous_action_penalty", 0.0)),
            pump_switch_penalty=float(safety_settings.get("pump_switch_penalty", 0.0)),
            engineering_template_bonus=float(safety_settings.get("engineering_template_bonus", 0.0)),
        )
        legacy_predictor = None
        residual_enabled = True
        residual_status_reason = "hierarchical_disabled"
        residual_actuator_ids: list[str] = []
        if bool(hierarchical_settings.get("enabled", False)):
            legacy_model_path = Path(str(hierarchical_settings.get("legacy_model_path", "") or ""))
            if not legacy_model_path.exists():
                raise FileNotFoundError(f"hierarchical Tier 1 model not found: {legacy_model_path}")
            legacy_predictor = _make_horizon_surrogate_predictor(
                legacy_model_path,
                int(prediction_horizon_steps),
                priority_idx,
                actuators=actuators,
                priority_to_actuators=priority_to_actuators,
                device=str(device),
            )
            residual_actuator_ids = [str(value) for value in hierarchical_settings.get("residual_actuator_ids", [])]
            residual_status_reason = "enabled_by_configuration"
            if bool(hierarchical_settings.get("require_residual_validation", True)):
                report_path = Path(str(hierarchical_settings.get("residual_validation_report", "") or ""))
                if report_path.exists():
                    report = json.loads(report_path.read_text(encoding="utf-8"))
                    residual_enabled = bool(report.get("validation_gate_passed", False)) and bool(
                        (report.get("rolling_horizon_smoke_eligibility", {}) or {}).get("passed", False)
                    )
                    residual_status_reason = "validated" if residual_enabled else "validation_gate_false"
                else:
                    residual_enabled = False
                    residual_status_reason = "validation_report_missing"
        mpc = TemporalJoint36Controller(
            actuators=actuators,
            predictor=predictor,
            candidate_config=candidate_config,
            safety_config=safety_config,
            legacy_groups=settings.get("legacy_groups", []),
            paired_groups=settings.get("paired_groups", []),
            legacy_predictor=legacy_predictor,
            residual_actuator_ids=residual_actuator_ids,
            residual_enabled=residual_enabled,
            residual_status_reason=residual_status_reason,
            event_id=str(event_id),
            deployment_allowed_patterns=(
                (settings.get("deployment_reliability", {}) or {}).get("allowed_patterns", [])
            ),
            deployment_return_period_min=int(
                (settings.get("deployment_reliability", {}) or {}).get("return_period_min", 0)
            ),
            deployment_return_period_max=int(
                (settings.get("deployment_reliability", {}) or {}).get("return_period_max", 0)
            ),
            deployment_evidence_csv=str(
                (settings.get("deployment_reliability", {}) or {}).get("evidence_csv", "") or ""
            ),
            deployment_time_tolerance_min=float(
                (settings.get("deployment_reliability", {}) or {}).get("time_tolerance_min", 0.1)
            ),
            deployment_require_evidence=bool(
                (settings.get("deployment_reliability", {}) or {}).get("require_evidence", False)
            ),
            deployment_stratum_rules=(
                (settings.get("deployment_reliability", {}) or {}).get("stratum_rules", [])
            ),
            template_reliability_csv=str(
                (settings.get("template_reliability", {}) or {}).get("rules_csv", "") or ""
            ),
            template_reliability_default_allow=bool(
                (settings.get("template_reliability", {}) or {}).get("default_allow", True)
            ),
            template_reliability_stress_return_period_min=int(
                (settings.get("template_reliability", {}) or {}).get("stress_return_period_min", 75)
            ),
            template_reliability_stress_default_allow=bool(
                (settings.get("template_reliability", {}) or {}).get("stress_default_allow", False)
            ),
            template_reliability_block_strong_patterns=(
                (settings.get("template_reliability", {}) or {}).get("block_strong_patterns", [])
            ),
            prediction_horizon_steps=int(prediction_horizon_steps),
            terminal_return_steps=int(terminal_return_steps),
        )
        proposed_policy_id = (
            "retrofit_hierarchical36"
            if proposed_controller == "hierarchical_core26_residual10"
            else "proposed_hierarchical_v8_residual_36"
            if legacy_predictor is not None
            else "proposed_temporal_joint_36"
        )
        generic_default_policy_name = "no_control_online_predicted"
        generic_default_policy = None
    else:
        mpc = PFVFirstMPC(
            surrogate_model_path,
            actuators,
            len(node_order),
            device=device,
            priority_node_indices=priority_idx,
            residual_value_path=residual_value_path,
            residual_pfv_prob_min=float(residual_pfv_prob_min),
            residual_safe_prob_min=float(residual_safe_prob_min),
            residual_nonzero_prob_min=float(residual_nonzero_prob_min),
            residual_peak_prob_min=float(residual_peak_prob_min),
            max_candidate_delta=float(max_candidate_delta),
            topk_log_count=int(topk_log_count),
            max_candidate_count=int(max_candidate_count),
            candidate_hold_steps=candidate_hold_steps,
            allowed_candidate_templates=allowed_candidate_templates,
            blocked_candidate_templates=blocked_candidate_templates,
            allowed_candidate_scopes_by_template=allowed_candidate_scopes_by_template,
            priority_upstream_nodes=priority_upstream_nodes,
            priority_downstream_nodes=priority_downstream_nodes,
            empirical_guard_path=empirical_guard_path,
            empirical_guard_unknown_allow=bool(empirical_guard_unknown_allow),
            boost_safe_prob_extra=float(boost_safe_prob_extra),
            boost_peak_prob_extra=float(boost_peak_prob_extra),
            protective_safe_prob_relief=float(protective_safe_prob_relief),
            release_peak_hold_max=int(release_peak_hold_max),
            event_id=str(event_id),
            nominal_pfv_reference=baseline_pfv_reference,
            low_risk_pfv_threshold=float(low_risk_pfv_threshold),
            high_risk_pfv_threshold=float(high_risk_pfv_threshold),
            release_recession_pfv_min=float(release_recession_pfv_min),
            release_recession_priority_depth_min=float(release_recession_priority_depth_min),
            strict_guard_return_period_max=int(strict_guard_return_period_max),
            strict_guard_patterns=str(strict_guard_patterns),
            strict_guard_prob_extra=float(strict_guard_prob_extra),
            horizon_smooth_weight=float(horizon_smooth_weight),
            horizon_violation_penalty=float(horizon_violation_penalty),
        )
        proposed_policy_id = "proposed_pfv_first_mpc"
        generic_default_policy_name = ""
        generic_default_policy = None
    details, history = [], []
    t0 = time.time()
    with Simulation(str(inp_path)) as sim:
        sim.step_advance(int(control_step_sec))
        nodes = Nodes(sim)
        links = Links(sim)
        gages = RainGages(sim)
        node_objs = {}
        for nid in node_order:
            try:
                node_objs[nid] = nodes[nid]
            except Exception:
                continue
        node_max_depths = {nid: _node_max_depth(obj) for nid, obj in node_objs.items()}
        actuator_ids, link_objs = _get_existing_links(links, actuators["actuator_id"].astype(str).tolist())
        internal_shadow_link_objs: dict[str, Any] = {}
        internal_shadow_iter = None
        if internal_shadow_sim is not None:
            internal_shadow_links = Links(internal_shadow_sim)
            shadow_ids, internal_shadow_link_objs = _get_existing_links(
                internal_shadow_links, actuators["actuator_id"].astype(str).tolist()
            )
            if list(shadow_ids) != list(actuator_ids):
                raise RuntimeError(
                    "V4 causal Internal shadow actuator order mismatch: "
                    f"proposed={actuator_ids} internal_shadow={shadow_ids}"
                )
            internal_shadow_iter = iter(internal_shadow_sim)
        nominal_table = _load_nominal_action_table(nominal_detail_csv, actuator_ids)
        no_control_action_table = _load_nominal_action_table(no_control_detail_csv, actuator_ids)
        passive_action_table = _load_nominal_action_table(passive_detail_csv, actuator_ids)
        rain_ids = _ids_from_container(gages, "raingageid")
        rain_obj = gages[rain_ids[0]] if rain_ids else None
        step_i = 0
        previous_action = _observed_action_from_links(link_objs, actuator_ids, actuators)
        # The passive anchor is frozen from the initial executable-passive
        # setting only. Future baseline hydraulic states/actions are never read
        # by the online V4 controller.
        no_control_default_action = previous_action.copy()
        passive_anchor_action = _nominal_action_at(passive_action_table, 0.0) if passive_action_table else None
        if passive_anchor_action is None:
            passive_anchor_action = no_control_default_action.copy()
        passive_anchor_action = _enforce_actuator_semantics(
            passive_anchor_action[: len(previous_action)], actuator_ids, actuators,
            pump_control_mode, variable_speed_pump_ids,
        )
        v4_limits = DualReferenceLimits(
            pfv_abs_margin_m3=float(v4_pfv_abs_margin_m3),
            pfv_rel_margin=float(v4_pfv_rel_margin),
            quantile=float(v4_quantile),
            readback_tolerance=float(v4_readback_tolerance),
            action_deadband=float(v4_action_deadband),
            max_k=int(v4_max_k),
        )
        # The event cap is initialized by the causal reference model at the
        # first decision and may only tighten. Offline baseline KPI files are
        # retained for post-event evaluation but are forbidden online inputs.
        v4_event_budget = EventQuantilePfvBudget(
            abs_margin_m3=float(v4_pfv_abs_margin_m3),
            rel_margin=float(v4_pfv_rel_margin),
        )
        v4_observed_pfv_m3 = 0.0
        v4_pending_command: np.ndarray | None = None
        v4_pending_anchor: np.ndarray | None = None
        v4_pending_fallback_id = ""
        v4_force_fallback_reason = ""
        has_executed_control_action = False
        empirical_pulse_action: np.ndarray | None = None
        empirical_pulse_remaining = 0
        empirical_overrides_by_phase: dict[str, int] = {}
        for _ in sim:
            elapsed_min = (sim.current_time - sim.start_time).total_seconds() / 60.0
            rain = _as_float(rain_obj.rainfall, 0.0) if rain_obj is not None else 0.0
            phase = phase_from_time(elapsed_min, duration_min)
            true_state = np.zeros(len(node_order), dtype=np.float32)
            for nid, obj in node_objs.items():
                true_state[node_index[nid]] = _as_float(obj.depth, 0.0)
            node_depths = {nid: float(true_state[node_index[nid]]) for nid in node_objs if nid in node_index}
            if is_v4_dual_reference_controller:
                priority_rate = 0.0
                for nid in priority_nodes:
                    obj = node_objs.get(nid)
                    if obj is None:
                        continue
                    try:
                        priority_rate += max(0.0, _as_float(obj.flooding, 0.0))
                    except Exception:
                        continue
                v4_observed_pfv_m3 += priority_rate * float(control_step_sec)
                v4_event_budget.update_observed(v4_observed_pfv_m3)
            if gat is not None and sensor_idx:
                sparse = np.zeros_like(true_state)
                mask = np.zeros_like(true_state)
                sparse[sensor_idx] = true_state[sensor_idx]
                mask[sensor_idx] = 1.0
                with torch.no_grad():
                    recon = gat(
                        torch.tensor(sparse[None, :], dtype=torch.float32, device=dev),
                        torch.tensor(mask[None, :], dtype=torch.float32, device=dev),
                        torch.tensor([[rain]], dtype=torch.float32, device=dev),
                        gat_node_static,
                        gat_edge_index,
                    ).cpu().numpy()[0]
            else:
                recon = true_state
            nominal_action = _nominal_action_at(nominal_table, elapsed_min)
            if proposed_controller in {"temporal_joint_36", "hierarchical_core26_residual10"}:
                h = int(horizon_steps)
                long_h = int(getattr(mpc, "prediction_horizon_steps", h))
                # Formal deployment cannot index the future offline twin. The
                # known passive/default setting is rolled forward from the
                # current reconstructed hydraulic state. It must not inherit
                # actions previously executed by Proposed.
                reference_action_sequence = constant_default_action_sequence(
                    no_control_default_action, h
                )
                reference_action_sequence_long = constant_default_action_sequence(
                    no_control_default_action, long_h
                )
                rainfall_window = _rainfall_window_at(
                    rainfall_forecast,
                    elapsed_min,
                    h,
                    float(control_step_sec) / 60.0,
                    rain,
                )[:, None]
                rainfall_long_window = _rainfall_window_at(
                    rainfall_forecast,
                    elapsed_min,
                    long_h,
                    float(control_step_sec) / 60.0,
                    rain,
                )[:, None]
                action, info = mpc.choose(
                    reconstructed_state=recon,
                    rainfall_window=rainfall_window,
                    reference_action_sequence=reference_action_sequence,
                    rainfall_long_window=rainfall_long_window,
                    reference_action_sequence_long=reference_action_sequence_long,
                    phase=str(phase),
                    elapsed_min=float(elapsed_min),
                )
                info["constraint_reference_mode"] = "online_predicted_no_control_reference"
                info["offline_no_control_used_by_controller"] = False
                info["no_control_reference_detail_csv"] = str(no_control_detail_csv or "")
                info["passive_reference_detail_csv"] = str(passive_detail_csv or "")
                if is_v4_dual_reference_controller:
                    info["event_pfv_budget_remaining_m3"] = float(v4_event_budget.remaining_m3)
                    info["event_pfv_cap_m3"] = float(v4_event_budget.event_cap_m3)
                    info["observed_proposed_pfv_m3"] = float(v4_observed_pfv_m3)
                    info["pfv_reference_contract"] = "causal_quantile_min(No-control,Passive)"
                    info["tfv_peak_reference_contract"] = "causal_quantile_Internal"
                    info["online_future_hydraulics_used"] = False
                    if v4_actual_readback_audit is not None:
                        info["previous_step_actual_readback_audit"] = json.dumps(v4_actual_readback_audit, ensure_ascii=False)
            elif is_generic_like_controller:
                phase_allowed_ids, phase_directions, phase_filter_source, phase_depth_threshold = _phase_reliable_action_filter(
                    phase_reliability,
                    pattern=str(event_id).split("_", 2)[2] if str(event_id).count("_") >= 2 else "",
                    phase=str(phase),
                    allow_tfv_noninferior=bool(phase_reliability_allow_tfv_noninferior),
                    require_pfv_improvement=bool(phase_reliability_require_pfv_improvement),
                    pfv_tolerance_abs=float(phase_reliability_pfv_tolerance_abs),
                    pfv_tolerance_frac=float(phase_reliability_pfv_tolerance_frac),
                    tfv_tolerance_abs=float(phase_reliability_tfv_tolerance_abs),
                    tfv_tolerance_frac=float(phase_reliability_tfv_tolerance_frac),
                    peak_tolerance_abs=float(phase_reliability_peak_tolerance_abs),
                    peak_tolerance_frac=float(phase_reliability_peak_tolerance_frac),
                    elapsed_min=float(elapsed_min),
                    evidence_time_tolerance_min=float(phase_reliability_evidence_time_tolerance_min),
                )
                phase_delta_limits = _phase_reliable_action_delta_limits(
                    phase_reliability,
                    pattern=str(event_id).split("_", 2)[2] if str(event_id).count("_") >= 2 else "",
                    phase=str(phase),
                    allow_tfv_noninferior=bool(phase_reliability_allow_tfv_noninferior),
                    require_pfv_improvement=bool(phase_reliability_require_pfv_improvement),
                    pfv_tolerance_abs=float(phase_reliability_pfv_tolerance_abs),
                    pfv_tolerance_frac=float(phase_reliability_pfv_tolerance_frac),
                    tfv_tolerance_abs=float(phase_reliability_tfv_tolerance_abs),
                    tfv_tolerance_frac=float(phase_reliability_tfv_tolerance_frac),
                    peak_tolerance_abs=float(phase_reliability_peak_tolerance_abs),
                    peak_tolerance_frac=float(phase_reliability_peak_tolerance_frac),
                    elapsed_min=float(elapsed_min),
                    evidence_time_tolerance_min=float(phase_reliability_evidence_time_tolerance_min),
                )
                phase_verified_effects = _phase_reliable_verified_effects(
                    phase_reliability,
                    event_id=str(event_id),
                    pattern=str(event_id).split("_", 2)[2] if str(event_id).count("_") >= 2 else "",
                    phase=str(phase),
                    allow_tfv_noninferior=bool(phase_reliability_allow_tfv_noninferior),
                    require_pfv_improvement=bool(phase_reliability_require_pfv_improvement),
                    pfv_tolerance_abs=float(phase_reliability_pfv_tolerance_abs),
                    pfv_tolerance_frac=float(phase_reliability_pfv_tolerance_frac),
                    tfv_tolerance_abs=float(phase_reliability_tfv_tolerance_abs),
                    tfv_tolerance_frac=float(phase_reliability_tfv_tolerance_frac),
                    peak_tolerance_abs=float(phase_reliability_peak_tolerance_abs),
                    peak_tolerance_frac=float(phase_reliability_peak_tolerance_frac),
                    elapsed_min=float(elapsed_min),
                    evidence_time_tolerance_min=float(phase_reliability_evidence_time_tolerance_min),
                )
                phase_priority_depth = float(np.mean(recon[priority_idx])) if priority_idx else 0.0
                if phase_depth_threshold is not None and phase_priority_depth + 1.0e-6 < phase_depth_threshold:
                    phase_allowed_ids, phase_directions = [], {}
                    phase_delta_limits = {}
                    phase_verified_effects = {}
                    phase_filter_source = "phase_reliability_state_below_verified_window"
                if (
                    bool(phase_reliability_fallback_to_surrogate)
                    and phase_filter_source in {
                        "phase_reliability_no_local_evidence",
                        "phase_reliability_state_below_verified_window",
                    }
                ):
                    # Missing local ablation evidence is not proof that every
                    # action is unsafe. Remove only the empirical pre-filter;
                    # the learned horizon uncertainty gate remains mandatory.
                    phase_allowed_ids = None
                    phase_directions = None
                    phase_delta_limits = None
                    phase_verified_effects = None
                    phase_filter_source = "phase_reliability_surrogate_only"
                if phase_reliability is not None:
                    # Single-actuator exact replay data establishes local
                    # effects, not the interaction of arbitrary combinations.
                    # Do not reconstruct an untested joint action by combining
                    # individually safe directions online.
                    raw_exact_effects = "effect_TFV_H" in phase_reliability.columns
                    mpc.set_runtime_action_filter(
                        phase_allowed_ids,
                        phase_directions,
                        candidate_group_limit=(
                            max(1, int(phase_reliability_candidate_group_limit))
                            if raw_exact_effects
                            else None
                        ),
                        action_delta_limits=phase_delta_limits,
                        verified_action_effects=phase_verified_effects,
                    )
                h = int(getattr(mpc, "config").horizon_steps)
                observed_action = _observed_action_from_links(link_objs, actuator_ids, actuators)
                v4_actual_readback_audit = None
                if is_v4_dual_reference_controller and v4_pending_command is not None:
                    v4_actual_readback_audit = enforce_final_readback(
                        requested=v4_pending_command,
                        projected=v4_pending_command,
                        readback=observed_action,
                        anchor=(v4_pending_anchor if v4_pending_anchor is not None else v4_pending_command),
                        actuator_ids=actuator_ids,
                        binary_pump_ids={"ADD301.2", "ADD301.3"},
                        max_k=int(v4_max_k),
                        tolerance=float(v4_readback_tolerance),
                        deadband=float(v4_action_deadband),
                    )
                    if not bool(v4_actual_readback_audit.get("passed")):
                        v4_force_fallback_reason = "previous_step_actual_readback_mismatch"
                    else:
                        v4_force_fallback_reason = ""
                    v4_pending_command = None
                    v4_pending_anchor = None
                if generic_default_policy_name == "no_control":
                    # A no-control fallback must replay the observed passive
                    # SWMM setting, not write an all-open action vector.
                    default_action = observed_action
                elif generic_default_policy is not None:
                    default_ctx = PolicyContext(
                        elapsed_min,
                        duration_min,
                        rain,
                        phase,
                        previous_action,
                        node_depths=node_depths,
                        node_max_depths=node_max_depths,
                    )
                    default_action = generic_default_policy.action(default_ctx)[: len(previous_action)]
                else:
                    default_action = previous_action
                if generic_default_policy_name == "no_control" and no_control_action_table:
                    replay = _nominal_action_at(no_control_action_table, elapsed_min)
                    if replay is not None:
                        default_action = replay[: len(previous_action)]
                rainfall_window = _rainfall_window_at(
                    rainfall_forecast,
                    elapsed_min,
                    h,
                    float(control_step_sec) / 60.0,
                    rain,
                )
                v4_extra_context = None
                if is_v4_dual_reference_controller:
                    reference_forecast_fn = getattr(horizon_predictor, "reference_forecast", None)
                    if not callable(reference_forecast_fn):
                        raise RuntimeError("V4 causal reference forecast is not available")
                    priority_ratio_values = []
                    for nid in priority_nodes:
                        if nid not in node_index:
                            continue
                        depth = float(recon[node_index[nid]])
                        max_depth = max(1.0e-6, float(node_max_depths.get(nid, 1.0)))
                        priority_ratio_values.append(depth / max_depth)
                    priority_depth_ratio = float(max(priority_ratio_values)) if priority_ratio_values else 0.0
                    v4_phase = classify_phase(
                        current_rainfall=float(rain),
                        future_rainfall=rainfall_window,
                        priority_depth_ratio=priority_depth_ratio,
                        elapsed_since_rain_end_min=(max(0.0, float(elapsed_min) - float(duration_min)) if elapsed_min > duration_min else None),
                    )
                    causal_reference_context = {
                        "elapsed_min": float(elapsed_min),
                        "phase": v4_phase.value,
                        "reconstructed_state": np.asarray(recon, dtype=float),
                        "rainfall_window": np.asarray(rainfall_window, dtype=float),
                        "current_action": np.asarray(observed_action, dtype=float),
                    }
                    causal_reference = reference_forecast_fn(causal_reference_context)
                    event_cap = v4_event_budget.freeze_or_tighten(
                        float(causal_reference["no_control_event_pfv_upper"]),
                        float(causal_reference["passive_event_pfv_upper"]),
                    )
                    safety_pfv = np.minimum(
                        np.asarray(causal_reference["no_control_pfv_upper"], dtype=float),
                        np.asarray(causal_reference["passive_pfv_upper"], dtype=float),
                    )
                    safety_cap_horizon = float(np.sum(np.maximum(0.0, safety_pfv))) + max(
                        float(v4_pfv_abs_margin_m3),
                        float(v4_pfv_rel_margin) * float(np.sum(np.maximum(0.0, safety_pfv))),
                    )
                    fallback_decision = choose_phase_aware_fallback(
                        phase=v4_phase,
                        pfv_budget_remaining_m3=v4_event_budget.remaining_m3,
                        internal_predicted_pfv_quantile_m3=float(causal_reference["internal_event_pfv_upper"]),
                        pfv_cap_m3=float(event_cap),
                        internal_legal=internal_shadow_action is not None,
                        passive_legal=True,
                    )
                    if v4_force_fallback_reason:
                        fallback_decision = choose_phase_aware_fallback(
                            phase=HydraulicPhase.RISING,
                            pfv_budget_remaining_m3=-1.0,
                            internal_predicted_pfv_quantile_m3=float("inf"),
                            pfv_cap_m3=float(event_cap),
                            internal_legal=False,
                            passive_legal=True,
                        )
                    fallback_action = (
                        np.asarray(internal_shadow_action, dtype=float).copy()
                        if fallback_decision.fallback_id == "internal_rules" and internal_shadow_action is not None
                        else np.asarray(passive_anchor_action, dtype=float).copy()
                    )
                    reference_action_sequence = np.repeat(fallback_action[None, :], h, axis=0).astype(np.float32)
                    default_action = reference_action_sequence[0].copy()
                    reference_window = {
                        "pfv": safety_pfv,
                        "tfv": np.asarray(causal_reference["internal_tfv_upper"], dtype=float),
                        "peak_tfv_rate": np.asarray(causal_reference["internal_peak_upper"], dtype=float),
                    }
                    reference_mode = "v4_causal_dual_reference_model_no_future_swmm"
                    uncertainty_margin = np.asarray(causal_reference.get("reference_uncertainty_margin", []), dtype=float)
                    uncertainty_scale = max(1.0, float(event_cap), float(np.max(np.asarray(causal_reference.get("upper_values", [0.0]), dtype=float))))
                    uncertainty_score = float(np.clip(np.max(uncertainty_margin) / uncertainty_scale, 0.0, 1.0)) if uncertainty_margin.size else 1.0
                    pfv_headroom_fraction = float(np.clip(v4_event_budget.remaining_m3 / max(event_cap, 1.0e-9), -1.0, 1.0))
                    v4_k_limit = adaptive_k(
                        phase=v4_phase,
                        pfv_headroom_fraction=pfv_headroom_fraction,
                        uncertainty_score=uncertainty_score,
                        allowed_values=(tuple(v4_adaptive_k_values) if v4_adaptive_k_values is not None else (0, 2, 4, 6, 8)),
                    )
                    v4_extra_context = {
                        **causal_reference_context,
                        "causal_reference_forecast": causal_reference,
                        "selected_fallback_id": fallback_decision.fallback_id,
                        "selected_fallback_reason": fallback_decision.reason,
                        "event_pfv_budget_remaining_m3": v4_event_budget.remaining_m3,
                        "event_pfv_cap_m3": event_cap,
                        "horizon_pfv_cap_m3": safety_cap_horizon,
                        "adaptive_k_limit": int(v4_k_limit),
                        "pfv_headroom_fraction": pfv_headroom_fraction,
                        "reference_uncertainty_score": uncertainty_score,
                        "action_setting_deadband": float(v4_action_deadband),
                        "changed_facility_penalty": float(v4_changed_facility_penalty),
                        "variation_penalty": float(v4_variation_penalty),
                        "reversal_penalty": float(v4_reversal_penalty),
                        "minimum_material_benefit": float(v4_minimum_material_benefit),
                        "minimum_benefit_cost_ratio": float(v4_minimum_benefit_cost_ratio),
                        "action_semantics": "dual_reference_residual_first_step_then_reoptimize",
                    }
                else:
                    reference_action_sequence = _reference_action_sequence_from_table(
                        no_control_action_table,
                        elapsed_min=elapsed_min,
                        horizon_steps=h,
                        dt_sec=control_step_sec,
                        fallback_action=default_action,
                    )
                    default_action = reference_action_sequence[0]
                    reference_window, reference_mode = _constraint_reference_window(
                        no_control_detail_df,
                        elapsed_min=elapsed_min,
                        horizon_steps=h,
                        dt_sec=control_step_sec,
                        priority_nodes=priority_nodes,
                    )
                phase_key = f"{str(event_id).split('_', 2)[2] if str(event_id).count('_') >= 2 else ''}|{phase}"
                if empirical_pulse_remaining > 0 and empirical_pulse_action is not None:
                    action = empirical_pulse_action.copy()
                    empirical_pulse_remaining -= 1
                    info = {
                        "policy_id": proposed_policy_id,
                        "fallback_to_default": False,
                        "intervention_reason": "empirical_verified_pulse_hold",
                        "selected_sequence_label": "empirical_verified_pulse_hold",
                        "selected_gate_pass": True,
                    }
                elif (
                    phase_reliability is not None
                    and loaded_phase_reliability_max_overrides > 0
                    and empirical_overrides_by_phase.get(phase_key, 0) >= loaded_phase_reliability_max_overrides
                ):
                    action = default_action.copy()
                    info = {
                        "policy_id": proposed_policy_id,
                        "fallback_to_default": True,
                        "intervention_reason": "phase_reliability_override_budget_exhausted",
                        "selected_sequence_label": "hold_native",
                        "selected_gate_pass": False,
                    }
                else:
                    action, info = mpc.choose(
                        reconstructed_state=recon,
                        rainfall_window=rainfall_window,
                        current_action=default_action,
                        reference_pfv=reference_window["pfv"] if reference_window is not None else None,
                        reference_tfv=reference_window["tfv"] if reference_window is not None else None,
                        reference_peak=reference_window["peak_tfv_rate"] if reference_window is not None else None,
                        reference_action_sequence=reference_action_sequence,
                        last_executed_action=(
                            previous_action if has_executed_control_action else default_action
                        ),
                        elapsed_min=elapsed_min,
                        phase=str(phase),
                        empirical_local_verified=(phase_filter_source == "phase_reliability_exact_local"),
                        extra_predictor_context=v4_extra_context,
                    )
                    if is_v4_dual_reference_controller:
                        info["policy_id"] = proposed_policy_id
                        info["formal_controller"] = "proposed_dual_reference_v4"
                        info["value_model_source"] = horizon_predictor_source
                        info["hydraulic_evidence_source"] = "authoritative_swmm"
                        info["selected_fallback"] = str((v4_extra_context or {}).get("selected_fallback_id", ""))
                        info["fallback_selection_reason"] = str((v4_extra_context or {}).get("selected_fallback_reason", ""))
                        if bool(info.get("online_future_hydraulics_used", False)):
                            raise RuntimeError("V4 online controller attempted to use future SWMM hydraulic truth")
                        predicted_event_pfv = float(info.get("selected_event_pfv_upper", np.inf))
                        event_allowed = np.isfinite(predicted_event_pfv) and v4_event_budget.allows(predicted_event_pfv)
                        if v4_force_fallback_reason or not event_allowed:
                            action = default_action.copy()
                            info["selected_gate_pass"] = False
                            info["fallback_to_default"] = True
                            info["intervention_reason"] = v4_force_fallback_reason or "event_pfv_quantile_budget_exhausted"
                            info["selected_sequence_label"] = "phase_frozen_fallback"
                        else:
                            v4_event_budget.set_inflight(predicted_event_pfv)
                        if bool(info.get("fallback_to_default", False)) and str(info.get("selected_fallback", "")) == "internal_rules":
                            info["internal_fallback_action_source"] = "synchronized_causal_native_rule_shadow_current_setting"
                    if is_v3_action_effect_controller:
                        info["policy_id"] = proposed_policy_id
                        info["formal_controller"] = "proposed_pfvfirst_dualfallback_v3"
                        info["value_model_source"] = horizon_predictor_source
                        info["hydraulic_evidence_source"] = "authoritative_swmm"
                    if (
                        phase_reliability is not None
                        and bool(info.get("selected_gate_pass", False))
                        and str(info.get("selected_sequence_label", "")) != "hold_native"
                    ):
                        empirical_pulse_action = np.asarray(action, dtype=float).copy()
                        empirical_pulse_remaining = max(0, loaded_phase_reliability_pulse_steps - 1)
                        empirical_overrides_by_phase[phase_key] = empirical_overrides_by_phase.get(phase_key, 0) + 1
                info["generic_default_policy_id"] = generic_default_policy_name
                info["constraint_reference_PFV"] = float(info.get("selected_reference_pfv_horizon", np.nan))
                info["constraint_reference_TFV"] = float(info.get("selected_reference_tfv_horizon", np.nan))
                info["constraint_reference_peak_TFV_rate"] = float(info.get("selected_reference_peak_tfv_rate", np.nan))
                info["constraint_reference_mode"] = reference_mode
                info["phase_reliability_source"] = phase_filter_source
                info["phase_reliability_allowed_actuators"] = ",".join(phase_allowed_ids or [])
                info["phase_reliability_priority_depth"] = phase_priority_depth
                info["phase_reliability_min_priority_depth"] = phase_depth_threshold
                info["phase_reliability_pulse_remaining"] = int(empirical_pulse_remaining)
                info["phase_reliability_overrides_used"] = int(empirical_overrides_by_phase.get(phase_key, 0))
                info["internal_reference_detail_csv"] = str(nominal_detail_csv or "")
                info["no_control_reference_detail_csv"] = str(no_control_detail_csv or "")
                info["passive_reference_detail_csv"] = str(passive_detail_csv or "")
                if is_v4_dual_reference_controller:
                    info["event_pfv_budget_remaining_m3"] = float(v4_event_budget.remaining_m3)
                    info["pfv_reference_contract"] = "min(No-control,Passive)"
                    info["tfv_peak_reference_contract"] = "Internal"
            else:
                action, info = mpc.choose(
                    recon,
                    rain,
                    phase,
                    baseline_tfv_rate=baseline_tfv_reference,
                    baseline_peak=baseline_peak_reference,
                    nominal_action=nominal_action,
                    elapsed_min=elapsed_min,
                )
            skip_action_write = bool(info.get("skip_action_write", False))
            if skip_action_write:
                # True no-control fallback: do not overwrite SWMM link settings.
                action = _observed_action_from_links(link_objs, actuator_ids, actuators)
            else:
                action = _enforce_actuator_semantics(
                    action, actuator_ids, actuators, pump_control_mode, variable_speed_pump_ids
                )
                write_errors: list[str] = []
                target_readback = np.full(len(action), np.nan, dtype=float)
                for j, aid in enumerate(actuator_ids[: len(action)]):
                    try:
                        value = _as_float(np.clip(action[j], 0.0, 1.0), 1.0)
                        link_objs[aid].target_setting = value
                        target_readback[j] = _as_float(getattr(link_objs[aid], "target_setting", np.nan), np.nan)
                    except Exception as exc:
                        write_errors.append(f"{aid}:{type(exc).__name__}:{exc}")
                if is_v4_dual_reference_controller:
                    anchor_for_audit = np.asarray(default_action, dtype=float)
                    audit = enforce_final_readback(
                        requested=action, projected=action, readback=target_readback,
                        anchor=anchor_for_audit, actuator_ids=actuator_ids,
                        binary_pump_ids={"ADD301.2", "ADD301.3"},
                        max_k=int(v4_max_k), tolerance=float(v4_readback_tolerance),
                        deadband=float(v4_action_deadband),
                    )
                    if write_errors:
                        audit["passed"] = False
                        audit.setdefault("reasons", []).append("swmm_action_write_exception")
                        audit["write_errors"] = write_errors
                    if not bool(audit.get("passed")):
                        # Hard safety takeover: execute the fallback frozen before
                        # candidate scoring and verify the rollback as well.
                        fallback_action = _enforce_actuator_semantics(
                            anchor_for_audit, actuator_ids, actuators, pump_control_mode, variable_speed_pump_ids
                        )
                        fallback_readback = np.full(len(fallback_action), np.nan, dtype=float)
                        fallback_errors: list[str] = []
                        for j, aid in enumerate(actuator_ids[: len(fallback_action)]):
                            try:
                                value = _as_float(np.clip(fallback_action[j], 0.0, 1.0), 1.0)
                                link_objs[aid].target_setting = value
                                fallback_readback[j] = _as_float(getattr(link_objs[aid], "target_setting", np.nan), np.nan)
                            except Exception as exc:
                                fallback_errors.append(f"{aid}:{type(exc).__name__}:{exc}")
                        rollback = enforce_final_readback(
                            requested=fallback_action, projected=fallback_action, readback=fallback_readback,
                            anchor=fallback_action, actuator_ids=actuator_ids,
                            binary_pump_ids={"ADD301.2", "ADD301.3"}, max_k=0,
                            tolerance=float(v4_readback_tolerance), deadband=float(v4_action_deadband),
                        )
                        if fallback_errors or not bool(rollback.get("passed")):
                            raise RuntimeError(
                                "V4 SWMM write/readback hard constraint failed and fallback rollback failed: "
                                + json.dumps({"candidate_audit": audit, "rollback": rollback, "errors": fallback_errors}, ensure_ascii=False)
                            )
                        action = fallback_action
                        info["fallback_to_default"] = True
                        info["intervention_reason"] = "readback_hard_constraint_takeover"
                        info["readback_candidate_audit"] = json.dumps(audit, ensure_ascii=False)
                        info["readback_rollback_pass"] = True
                    else:
                        info["readback_candidate_audit"] = json.dumps(audit, ensure_ascii=False)
                        info["readback_rollback_pass"] = False
                    info["write_readback_match"] = bool(audit.get("passed"))
                    # The immediate target readback is necessary but not
                    # sufficient. The next control instant audits the actual
                    # SWMM current_setting before another candidate is allowed.
                    v4_pending_command = np.asarray(action, dtype=float).copy()
                    v4_pending_anchor = np.asarray(default_action, dtype=float).copy()
                    v4_pending_fallback_id = str(info.get("selected_fallback", ""))
                    if bool(audit.get("passed")) and not v4_force_fallback_reason:
                        v4_force_fallback_reason = ""
            if is_v4_dual_reference_controller and skip_action_write:
                v4_pending_command = None
                v4_pending_anchor = None
                v4_pending_fallback_id = "internal_rules"
            previous_action = np.asarray(action, dtype=np.float32).copy()
            has_executed_control_action = True
            row = {
                "event_id": event_id,
                "policy_id": proposed_policy_id,
                "elapsed_min": elapsed_min,
                "datetime": str(sim.current_time),
                "rainfall_mm_h": rain,
                "phase": phase,
            }
            for nid, obj in node_objs.items():
                row[f"h:{nid}"] = _as_float(obj.depth, np.nan)
                row[f"flood:{nid}"] = _as_float(obj.flooding, 0.0)
            for j, aid in enumerate(actuator_ids[: len(action)]):
                row[f"a:{aid}"] = _as_float(action[j], 1.0)
                try:
                    row[f"flow:{aid}"] = _as_float(link_objs[aid].flow, np.nan)
                    row[f"setting:{aid}"] = _as_float(link_objs[aid].current_setting, np.nan)
                except Exception:
                    row[f"flow:{aid}"] = np.nan
                    row[f"setting:{aid}"] = np.nan
            details.append(row)
            history.append({"event_id": event_id, "elapsed_min": elapsed_min, "phase": phase, "rainfall_mm_h": rain, **info})
            step_i += 1
            if max_steps and step_i >= max_steps:
                sim.terminate_simulation()
                break
    detail = pd.DataFrame(details)
    hist = pd.DataFrame(history)
    detail.to_csv(out_detail_csv, index=False)
    hist.to_csv(out_history_csv, index=False)
    kpis = compute_kpis(detail, priority_nodes, dt_sec=control_step_sec)
    kpis.update(
        {
            "event_id": event_id,
            "policy_id": proposed_policy_id,
            "duration_min": duration_min,
            "detail_file": str(out_detail_csv),
            "history_file": str(out_history_csv),
            "rows": len(detail),
            "fallback_rate": float(
                hist.get("fallback_to_nominal", hist.get("fallback_to_default", pd.Series([True]))).mean()
            )
            if not hist.empty
            else 1.0,
            "wall_time_sec": time.time() - t0,
            "proposed_controller": proposed_controller,
            "v4_internal_shadow_inp_path": str(internal_shadow_inp_path or ""),
            "v4_internal_shadow_current_action_only": bool(is_v4_dual_reference_controller and internal_shadow_inp_path),
        }
    )
    return kpis


def run_swmm_residual_override_trajectory(
    inp_path: str | Path,
    actuators: pd.DataFrame,
    priority_nodes: list[str],
    out_detail_csv: str | Path,
    event_id: str,
    duration_min: int,
    override_start_min: float,
    override_steps: int,
    override_deltas: dict[str, float],
    control_step_sec: int = 300,
    seed: int = 2026,
    max_steps: int = 0,
    simulation_duration_min: Optional[int] = None,
    recession_min: Optional[int] = None,
) -> dict:
    """Run native SWMM controls with a short residual action override.

    This preserves the INP's internal rules as the base policy and perturbs only
    selected actuators during a short time window, producing paired labels for
    residual action-value learning.
    """
    Simulation, Nodes, Links, RainGages = _require_pyswmm()
    inp_path = Path(inp_path)
    out_detail_csv = Path(out_detail_csv)
    out_detail_csv.parent.mkdir(parents=True, exist_ok=True)
    records = []
    t0 = time.time()
    override_end = float(override_start_min) + max(1, int(override_steps)) * int(control_step_sec) / 60.0
    with Simulation(str(inp_path)) as sim:
        sim.step_advance(int(control_step_sec))
        nodes = Nodes(sim)
        links = Links(sim)
        gages = RainGages(sim)
        node_ids = _ids_from_container(nodes, "nodeid")
        node_objs = {nid: nodes[nid] for nid in node_ids}
        actuator_ids, link_objs = _get_existing_links(links, actuators["actuator_id"].astype(str).tolist())
        rain_ids = _ids_from_container(gages, "raingageid")
        rain_obj = gages[rain_ids[0]] if rain_ids else None
        step_i = 0
        for _ in sim:
            elapsed_min = (sim.current_time - sim.start_time).total_seconds() / 60.0
            rain = _as_float(rain_obj.rainfall, 0.0) if rain_obj is not None else 0.0
            phase = phase_from_time(elapsed_min, duration_min)
            native_action = []
            for aid in actuator_ids:
                try:
                    native_action.append(_as_float(link_objs[aid].current_setting, 1.0))
                except Exception:
                    native_action.append(1.0)
            action = np.asarray(native_action, dtype=np.float32)
            override_active = float(override_start_min) <= elapsed_min < override_end
            if override_active:
                for j, aid in enumerate(actuator_ids):
                    if aid in override_deltas:
                        action[j] = float(np.clip(action[j] + float(override_deltas[aid]), 0.0, 1.0))
                        try:
                            link_objs[aid].target_setting = _as_float(action[j], 1.0)
                        except Exception:
                            pass
            row = {
                "event_id": event_id,
                "policy_id": "internal_residual_override",
                "elapsed_min": elapsed_min,
                "datetime": str(sim.current_time),
                "rainfall_mm_h": rain,
                "phase": phase,
                "override_active": bool(override_active),
                "override_start_min": float(override_start_min),
                "override_end_min": float(override_end),
            }
            for nid, obj in node_objs.items():
                try:
                    row[f"h:{nid}"] = _as_float(obj.depth, np.nan)
                    row[f"flood:{nid}"] = _as_float(obj.flooding, 0.0)
                except Exception:
                    row[f"h:{nid}"] = np.nan
                    row[f"flood:{nid}"] = 0.0
            for j, aid in enumerate(actuator_ids):
                row[f"a:{aid}"] = _as_float(action[j], 1.0)
                row[f"native_a:{aid}"] = _as_float(native_action[j], 1.0)
                row[f"delta_a:{aid}"] = float(override_deltas.get(aid, 0.0)) if override_active else 0.0
                try:
                    row[f"flow:{aid}"] = _as_float(link_objs[aid].flow, np.nan)
                    row[f"setting:{aid}"] = _as_float(link_objs[aid].current_setting, np.nan)
                except Exception:
                    row[f"flow:{aid}"] = np.nan
                    row[f"setting:{aid}"] = np.nan
            records.append(row)
            step_i += 1
            if max_steps and step_i >= max_steps:
                sim.terminate_simulation()
                break
    detail = pd.DataFrame(records)
    detail.to_csv(out_detail_csv, index=False)
    kpis = compute_kpis(detail, priority_nodes, dt_sec=control_step_sec)
    kpis.update(
        {
            "event_id": event_id,
            "policy_id": "internal_residual_override",
            "duration_min": duration_min,
            "rain_duration_min": duration_min,
            "recession_min": int(recession_min) if recession_min is not None else None,
            "simulation_duration_min": int(simulation_duration_min)
            if simulation_duration_min is not None
            else (float(detail["elapsed_min"].max()) + control_step_sec / 60.0 if not detail.empty else duration_min),
            "detail_file": str(out_detail_csv),
            "rows": len(detail),
            "wall_time_sec": time.time() - t0,
            "override_start_min": float(override_start_min),
            "override_steps": int(override_steps),
            "override_actuator_count": int(len(override_deltas)),
        }
    )
    return kpis


def _causal_override_sequence_step(
    *,
    elapsed_min: float,
    override_start_min: float,
    control_step_sec: int,
) -> int:
    """Map an off-grid checkpoint to consecutive causal control tokens."""
    if int(control_step_sec) <= 0:
        raise ValueError("control_step_sec must be positive")
    elapsed_steps = (
        (float(elapsed_min) - float(override_start_min))
        * 60.0
        / float(control_step_sec)
    )
    return max(0, int(np.floor(elapsed_steps + 1.0e-9)))


def run_swmm_no_control_action_ablation(
    inp_path: str | Path,
    actuators: pd.DataFrame,
    priority_nodes: list[str],
    no_control_detail_csv: str | Path,
    out_detail_csv: str | Path,
    event_id: str,
    duration_min: int,
    override_start_min: float,
    override_steps: int,
    actuator_id: str,
    action_delta: float,
    control_step_sec: int = 300,
    max_steps: int = 0,
    target_setting: float | None = None,
    override_targets: dict[str, float] | None = None,
    override_delta_sequence: dict[str, list[float]] | None = None,
    override_target_sequence: dict[str, list[float]] | None = None,
    post_override_nominal_detail_csv: str | Path | None = None,
    policy_id: str = "no_control_single_actuator_ablation",
    cleanup_swmm_artifacts: bool = False,
    storage_node_ids: list[str] | None = None,
    outfall_node_ids: list[str] | None = None,
) -> dict:
    """Replay No-control and branch one absolute actuator at one time.

    The candidate and reference simulations have identical forcing and action
    history before ``override_start_min``. This is the counterfactual contract
    required by the horizon effect head; generic policy trajectories do not
    satisfy it because their states have already diverged before a sampled row.
    """
    Simulation, Nodes, Links, RainGages = _require_pyswmm()
    inp_path = Path(inp_path)
    out_detail_csv = Path(out_detail_csv)
    out_detail_csv.parent.mkdir(parents=True, exist_ok=True)
    reference_detail = pd.read_csv(no_control_detail_csv)
    all_requested_ids = actuators["actuator_id"].astype(str).tolist()
    reference_table = _load_nominal_action_table(no_control_detail_csv, all_requested_ids)
    post_override_table = _load_nominal_action_table(post_override_nominal_detail_csv, all_requested_ids)
    override_end = float(override_start_min) + max(1, int(override_steps)) * float(control_step_sec) / 60.0
    records = []
    t0 = time.time()
    with Simulation(str(inp_path)) as sim:
        sim.step_advance(int(control_step_sec))
        nodes = Nodes(sim)
        links = Links(sim)
        gages = RainGages(sim)
        node_ids = _ids_from_container(nodes, "nodeid")
        node_objs = {nid: nodes[nid] for nid in node_ids}
        actuator_ids, link_objs = _get_existing_links(links, all_requested_ids)
        targets = {str(actuator_id): target_setting}
        if override_targets:
            targets.update({str(key): float(value) for key, value in override_targets.items()})
        sequence_ids = set((override_delta_sequence or {})) | set((override_target_sequence or {}))
        for aid in sequence_ids:
            targets.setdefault(str(aid), None)
        missing_targets = sorted(set(targets) - set(link_objs))
        if missing_targets:
            raise KeyError(f"Ablation actuator is not present in SWMM links: {missing_targets[:5]}")
        rain_ids = _ids_from_container(gages, "raingageid")
        rain_obj = gages[rain_ids[0]] if rain_ids else None
        previous = _initial_action_from_links(link_objs, actuator_ids)
        step_i = 0
        for _ in sim:
            elapsed_min = (sim.current_time - sim.start_time).total_seconds() / 60.0
            active = float(override_start_min) <= elapsed_min < override_end
            replay_table = (
                reference_table
                if elapsed_min < float(override_end) or not post_override_table
                else post_override_table
            )
            replay = _nominal_action_at(replay_table, elapsed_min)
            if replay is not None:
                previous = np.asarray(replay[: len(actuator_ids)], dtype=np.float32)
            reference_action = previous.copy()
            action = reference_action.copy()
            if active:
                sequence_step = _causal_override_sequence_step(
                    elapsed_min=elapsed_min,
                    override_start_min=float(override_start_min),
                    control_step_sec=int(control_step_sec),
                )
                for aid, requested_target in targets.items():
                    j = actuator_ids.index(aid)
                    delta_values = list((override_delta_sequence or {}).get(aid, []))
                    target_values = list((override_target_sequence or {}).get(aid, []))
                    step_delta = delta_values[min(sequence_step, len(delta_values) - 1)] if delta_values else float(action_delta)
                    step_target = target_values[min(sequence_step, len(target_values) - 1)] if target_values else requested_target
                    action[j] = float(
                        np.clip(
                            reference_action[j] + float(step_delta)
                            if step_target is None
                            else float(step_target),
                            0.0,
                            1.0,
                        )
                    )
            for j, aid in enumerate(actuator_ids):
                try:
                    link_objs[aid].target_setting = _as_float(action[j], 1.0)
                except Exception:
                    pass
            rain = _as_float(rain_obj.rainfall, 0.0) if rain_obj is not None else 0.0
            row = {
                "event_id": event_id,
                "policy_id": str(policy_id),
                "elapsed_min": elapsed_min,
                "datetime": str(sim.current_time),
                "rainfall_mm_h": rain,
                "phase": phase_from_time(elapsed_min, duration_min),
                "override_active": bool(active),
                "override_actuator_id": ",".join(sorted(targets)),
                "override_delta": float(action_delta),
            }
            for nid, obj in node_objs.items():
                row[f"h:{nid}"] = _as_float(obj.depth, np.nan)
                row[f"flood:{nid}"] = _as_float(obj.flooding, 0.0)
            for j, aid in enumerate(actuator_ids):
                row[f"a:{aid}"] = _as_float(action[j], 1.0)
                row[f"reference_a:{aid}"] = _as_float(reference_action[j], 1.0)
                try:
                    row[f"flow:{aid}"] = _as_float(link_objs[aid].flow, np.nan)
                    row[f"setting:{aid}"] = _as_float(link_objs[aid].current_setting, np.nan)
                except Exception:
                    row[f"flow:{aid}"] = np.nan
                    row[f"setting:{aid}"] = np.nan
            record_v42_hydraulic_targets(
                row=row,
                node_objects=node_objs,
                facility_link_objects=link_objs,
                graph_node_ids=node_ids,
                storage_node_ids=storage_node_ids or [],
                facility_ids=actuator_ids,
                outfall_node_ids=outfall_node_ids or [],
            )
            records.append(row)
            previous = reference_action
            step_i += 1
            if max_steps and step_i >= int(max_steps):
                sim.terminate_simulation()
                break
    detail = pd.DataFrame(records)
    detail.to_csv(out_detail_csv, index=False)
    kpis = compute_kpis(detail, priority_nodes, dt_sec=control_step_sec)
    kpis.update(
        {
            "event_id": str(event_id),
            "policy_id": str(policy_id),
            "detail_file": str(out_detail_csv),
            "reference_detail_file": str(no_control_detail_csv),
            "actuator_id": str(actuator_id),
            "action_delta": float(action_delta),
            "target_setting": float(target_setting) if target_setting is not None else np.nan,
            "override_targets": json.dumps(targets, sort_keys=True),
            "override_delta_sequence": json.dumps(override_delta_sequence or {}, sort_keys=True),
            "override_target_sequence": json.dumps(override_target_sequence or {}, sort_keys=True),
            "actuator_ids": ",".join(sorted(targets)),
            "action_direction": "increase" if float(action_delta) > 0 else "decrease",
            "override_start_min": float(override_start_min),
            "override_steps": int(override_steps),
            "rows": int(len(detail)),
            "wall_time_sec": float(time.time() - t0),
            "reference_rows": int(len(reference_detail)),
        }
    )
    if cleanup_swmm_artifacts:
        # The case input is intentionally retained for provenance. SWMM's
        # binary output and report are regenerated artifacts and can consume
        # several GB during a parallel targeted scan.
        for suffix in (".out", ".rpt"):
            try:
                inp_path.with_suffix(suffix).unlink(missing_ok=True)
            except OSError:
                pass
    return kpis


def run_swmm_dynamic_internal(
    inp_path: str | Path,
    actuators: pd.DataFrame,
    priority_nodes: list[str],
    internal_baseline_detail_csv: str | Path,
    out_detail_csv: str | Path,
    event_id: str,
    duration_min: int,
    override_start_min: float,
    control_step_sec: int = 300,
    max_steps: int = 0,
    policy_id: str = "dynamic_internal_rules",
    cleanup_swmm_artifacts: bool = False,
    extra_prefix_link_ids: list[str] | None = None,
    extra_prefix_table: dict[float, np.ndarray] | None = None,
    prefix_inp_path: str | Path | None = None,
    hotstart_dir: str | Path | None = None,
) -> dict:
    """Run the Dynamic Internal reference branch.

    V3 two-phase mode (when *prefix_inp_path* is provided):
      Phase 1: Run prefix with no-controls INP (exact shared prefix).
      At checkpoint: save hotstart .hsf.
      Phase 2: Open with-controls INP, load hotstart, native rules govern.

    Legacy mode (no *prefix_inp_path*): single simulation with controls INP.
    """
    Simulation, Nodes, Links, RainGages = _require_pyswmm()
    inp_path = Path(inp_path)
    out_detail_csv = Path(out_detail_csv)
    out_detail_csv.parent.mkdir(parents=True, exist_ok=True)

    baseline_detail = pd.read_csv(internal_baseline_detail_csv)
    all_requested_ids = actuators["actuator_id"].astype(str).tolist()
    baseline_table = _load_nominal_action_table(internal_baseline_detail_csv, all_requested_ids)

    _extra_ids = list(extra_prefix_link_ids or [])
    _extra_table = extra_prefix_table or {}
    _use_two_phase = prefix_inp_path is not None
    _prefix_inp = Path(prefix_inp_path) if _use_two_phase else None
    _hs_dir = Path(hotstart_dir) if hotstart_dir else out_detail_csv.parent

    records: list[dict] = []
    t0 = time.time()
    hotstart_used = False
    save_hotstart_count = 0

    if _use_two_phase:
        # ================================================================
        # PHASE 1: prefix replay with no-controls INP (shared prefix)
        # ================================================================
        hsf_path = _hs_dir / "_v3_dynamic_internal_hotstart.hsf"
        hsf_path.parent.mkdir(parents=True, exist_ok=True)

        with Simulation(str(_prefix_inp)) as sim:
            sim.step_advance(int(control_step_sec))
            nodes = Nodes(sim)
            links = Links(sim)
            gages = RainGages(sim)
            node_ids = _ids_from_container(nodes, "nodeid")
            node_objs = {nid: nodes[nid] for nid in node_ids}
            actuator_ids, link_objs = _get_existing_links(links, all_requested_ids)
            rain_ids = _ids_from_container(gages, "raingageid")
            rain_obj = gages[rain_ids[0]] if rain_ids else None
            _extra_objs: dict = {}
            if _extra_ids:
                for lid in _extra_ids:
                    try:
                        _extra_objs[lid] = links[lid]
                    except Exception:
                        pass

            for _ in sim:
                elapsed_min = (sim.current_time - sim.start_time).total_seconds() / 60.0
                if elapsed_min >= float(override_start_min) + 1e-6:
                    # Save hotstart and break
                    save_fn = getattr(sim, "save_hotstart", None) or getattr(sim, "save_hot_start", None)
                    if save_fn:
                        try:
                            save_fn(str(hsf_path))
                            save_hotstart_count += 1
                        except Exception:
                            pass
                    break

                # Prefix replay: write Eng36 + extra link settings
                replay = _nominal_action_at(baseline_table, elapsed_min)
                if replay is not None:
                    action = np.asarray(replay[:len(actuator_ids)], dtype=np.float32)
                    for j, aid in enumerate(actuator_ids):
                        try:
                            link_objs[aid].target_setting = _as_float(action[j], 1.0)
                        except Exception:
                            pass
                if _extra_table:
                    extra_replay = _nominal_action_at(_extra_table, elapsed_min)
                    if extra_replay is not None:
                        for j, lid in enumerate(_extra_ids):
                            if lid in _extra_objs:
                                try:
                                    _extra_objs[lid].target_setting = _as_float(
                                        np.clip(extra_replay[j], 0.0, 1.0), 1.0)
                                except Exception:
                                    pass

                rain = _as_float(rain_obj.rainfall, 0.0) if rain_obj is not None else 0.0
                row: dict = {
                    "event_id": event_id,
                    "policy_id": str(policy_id),
                    "elapsed_min": elapsed_min,
                    "datetime": str(sim.current_time),
                    "rainfall_mm_h": rain,
                    "phase": phase_from_time(elapsed_min, duration_min),
                    "override_active": True,
                    "override_actuator_id": ",".join(sorted(actuator_ids)),
                    "override_delta": 0.0,
                    "policy_phase": "prefix_replay",
                    "write_source": "prefix_replay",
                }
                for nid, obj in node_objs.items():
                    row[f"h:{nid}"] = _as_float(obj.depth, np.nan)
                    row[f"head:{nid}"] = _as_float(getattr(obj, "head", np.nan), np.nan)
                    row[f"storage_volume:{nid}"] = _as_float(getattr(obj, "volume", np.nan), np.nan)
                    row[f"flood:{nid}"] = _as_float(obj.flooding, 0.0)
                for j, aid in enumerate(actuator_ids):
                    try:
                        row[f"a:{aid}"] = _as_float(link_objs[aid].current_setting, 1.0)
                        row[f"setting:{aid}"] = _as_float(link_objs[aid].current_setting, np.nan)
                        row[f"flow:{aid}"] = _as_float(link_objs[aid].flow, np.nan)
                    except Exception:
                        row[f"a:{aid}"] = np.nan
                        row[f"setting:{aid}"] = np.nan
                        row[f"flow:{aid}"] = np.nan
                    row[f"reference_a:{aid}"] = row[f"a:{aid}"]
                # V3: record extra prefix link settings
                for lid in _extra_ids:
                    if lid in _extra_objs:
                        try:
                            row[f"setting:{lid}"] = _as_float(_extra_objs[lid].current_setting, np.nan)
                        except Exception:
                            row[f"setting:{lid}"] = np.nan
                records.append(row)

        # ================================================================
        # PHASE 2: native rules with with-controls INP + hotstart
        # ================================================================
        if hsf_path.exists():
            hotstart_used = True
        with Simulation(str(inp_path)) as sim:
            sim.step_advance(int(control_step_sec))
            # Load hotstart if available
            if hsf_path.exists():
                load_fn = (getattr(sim, "use_hotstart", None) or getattr(sim, "use_hot_start", None)
                           or getattr(sim, "load_hotstart", None) or getattr(sim, "load_hot_start", None))
                if load_fn:
                    try:
                        load_fn(str(hsf_path))
                    except Exception:
                        pass
            nodes = Nodes(sim)
            links = Links(sim)
            gages = RainGages(sim)
            node_ids = _ids_from_container(nodes, "nodeid")
            node_objs = {nid: nodes[nid] for nid in node_ids}
            actuator_ids, link_objs = _get_existing_links(links, all_requested_ids)
            rain_ids = _ids_from_container(gages, "raingageid")
            rain_obj = gages[rain_ids[0]] if rain_ids else None

            for _ in sim:
                elapsed_min = (sim.current_time - sim.start_time).total_seconds() / 60.0
                # Skip prefix range rows (already recorded in Phase 1)
                if elapsed_min < float(override_start_min) - 1e-6:
                    continue
                # Do NOT override -- native [CONTROLS] govern
                rain = _as_float(rain_obj.rainfall, 0.0) if rain_obj is not None else 0.0
                row = {
                    "event_id": event_id,
                    "policy_id": str(policy_id),
                    "elapsed_min": elapsed_min,
                    "datetime": str(sim.current_time),
                    "rainfall_mm_h": rain,
                    "phase": phase_from_time(elapsed_min, duration_min),
                    "override_active": False,
                    "override_actuator_id": ",".join(sorted(actuator_ids)),
                    "override_delta": 0.0,
                    "policy_phase": "native_rules",
                    "write_source": "native_rules",
                }
                for nid, obj in node_objs.items():
                    row[f"h:{nid}"] = _as_float(obj.depth, np.nan)
                    row[f"head:{nid}"] = _as_float(getattr(obj, "head", np.nan), np.nan)
                    row[f"storage_volume:{nid}"] = _as_float(getattr(obj, "volume", np.nan), np.nan)
                    row[f"flood:{nid}"] = _as_float(obj.flooding, 0.0)
                for j, aid in enumerate(actuator_ids):
                    try:
                        row[f"a:{aid}"] = _as_float(link_objs[aid].current_setting, 1.0)
                        row[f"setting:{aid}"] = _as_float(link_objs[aid].current_setting, np.nan)
                        row[f"flow:{aid}"] = _as_float(link_objs[aid].flow, np.nan)
                    except Exception:
                        row[f"a:{aid}"] = np.nan
                        row[f"setting:{aid}"] = np.nan
                        row[f"flow:{aid}"] = np.nan
                    row[f"reference_a:{aid}"] = row[f"a:{aid}"]
                records.append(row)
                if max_steps and len(records) >= int(max_steps):
                    sim.terminate_simulation()
                    break

        # Cleanup hotstart file
        try:
            hsf_path.unlink(missing_ok=True)
        except OSError:
            pass

    else:
        # ================================================================
        # LEGACY: single simulation with controls INP
        # ================================================================
        with Simulation(str(inp_path)) as sim:
            sim.step_advance(int(control_step_sec))
            nodes = Nodes(sim)
            links = Links(sim)
            gages = RainGages(sim)
            node_ids = _ids_from_container(nodes, "nodeid")
            node_objs = {nid: nodes[nid] for nid in node_ids}
            actuator_ids, link_objs = _get_existing_links(links, all_requested_ids)
            rain_ids = _ids_from_container(gages, "raingageid")
            rain_obj = gages[rain_ids[0]] if rain_ids else None
            _extra_objs = {}
            if _extra_ids:
                for lid in _extra_ids:
                    try:
                        _extra_objs[lid] = links[lid]
                    except Exception:
                        pass

            for _ in sim:
                elapsed_min = (sim.current_time - sim.start_time).total_seconds() / 60.0
                in_prefix = elapsed_min < float(override_start_min)
                policy_phase = "prefix_replay" if in_prefix else "native_rules"
                requested = None
                if in_prefix:
                    replay = _nominal_action_at(baseline_table, elapsed_min)
                    if replay is not None:
                        action = np.asarray(replay[: len(actuator_ids)], dtype=np.float32)
                        requested = action.copy()
                        for j, aid in enumerate(actuator_ids):
                            try:
                                link_objs[aid].target_setting = _as_float(action[j], 1.0)
                            except Exception:
                                pass
                    if _extra_table:
                        extra_replay = _nominal_action_at(_extra_table, elapsed_min)
                        if extra_replay is not None:
                            for j, lid in enumerate(_extra_ids):
                                if lid in _extra_objs:
                                    try:
                                        _extra_objs[lid].target_setting = _as_float(
                                            np.clip(extra_replay[j], 0.0, 1.0), 1.0)
                                    except Exception:
                                        pass

                rain = _as_float(rain_obj.rainfall, 0.0) if rain_obj is not None else 0.0
                write_source = "prefix_replay" if in_prefix else "native_rules"
                row = {
                    "event_id": event_id,
                    "policy_id": str(policy_id),
                    "elapsed_min": elapsed_min,
                    "datetime": str(sim.current_time),
                    "rainfall_mm_h": rain,
                    "phase": phase_from_time(elapsed_min, duration_min),
                    "override_active": bool(in_prefix),
                    "override_actuator_id": ",".join(sorted(actuator_ids)),
                    "override_delta": 0.0,
                    "policy_phase": policy_phase,
                    "write_source": write_source,
                }
                for nid, obj in node_objs.items():
                    row[f"h:{nid}"] = _as_float(obj.depth, np.nan)
                    row[f"head:{nid}"] = _as_float(getattr(obj, "head", np.nan), np.nan)
                    row[f"storage_volume:{nid}"] = _as_float(getattr(obj, "volume", np.nan), np.nan)
                    row[f"flood:{nid}"] = _as_float(obj.flooding, 0.0)
                for j, aid in enumerate(actuator_ids):
                    try:
                        if requested is not None:
                            row[f"requested_setting:{aid}"] = float(requested[j])
                        else:
                            row[f"requested_setting:{aid}"] = np.nan
                        row[f"a:{aid}"] = _as_float(link_objs[aid].current_setting, 1.0)
                        row[f"setting:{aid}"] = _as_float(link_objs[aid].current_setting, np.nan)
                        row[f"flow:{aid}"] = _as_float(link_objs[aid].flow, np.nan)
                    except Exception:
                        row[f"requested_setting:{aid}"] = np.nan
                        row[f"a:{aid}"] = np.nan
                        row[f"setting:{aid}"] = np.nan
                        row[f"flow:{aid}"] = np.nan
                    row[f"reference_a:{aid}"] = row[f"a:{aid}"]
                records.append(row)
                if max_steps and len(records) >= int(max_steps):
                    sim.terminate_simulation()
                    break

    detail = pd.DataFrame(records)
    detail.to_csv(out_detail_csv, index=False)
    kpis = compute_kpis(detail, priority_nodes, dt_sec=control_step_sec)
    kpis.update(
        {
            "event_id": str(event_id),
            "policy_id": str(policy_id),
            "detail_file": str(out_detail_csv),
            "reference_detail_file": str(internal_baseline_detail_csv),
            "override_start_min": float(override_start_min),
            "rows": int(len(detail)),
            "wall_time_sec": float(time.time() - t0),
            "prefix_rows": int(len(detail[detail["policy_phase"] == "prefix_replay"])),
            "native_rows": int(len(detail[detail["policy_phase"] == "native_rules"])),
            "hotstart_used": hotstart_used,
            "use_hotstart_call_count": 1 if hotstart_used else 0,
            "save_hotstart_call_count": save_hotstart_count,
        }
    )
    if cleanup_swmm_artifacts:
        for suffix in (".out", ".rpt"):
            try:
                inp_path.with_suffix(suffix).unlink(missing_ok=True)
            except OSError:
                pass
    return kpis


# ---------------------------------------------------------------------------
# Physical network SHA (ignores CONTROLS / RAINGAGES / TIMESERIES / dates)
# ---------------------------------------------------------------------------

def physical_network_sha256(inp_path: str | Path) -> str:
    """SHA-256 of the INP physical network, ignoring event-specific sections.

    Strips [CONTROLS], [RAINGAGES], [TIMESERIES] and OPTIONS date/time keys
    so that with-controls and no-controls INP variants can be compared.
    """
    inp_path = Path(inp_path)
    text = inp_path.read_text(encoding="utf-8", errors="replace")
    lines: list[str] = []
    skip_section = False
    strip_sections = {"[CONTROLS]", "[RAINGAGES]", "[TIMESERIES]"}
    strip_options = {"START_DATE", "START_TIME", "END_DATE", "END_TIME",
                     "REPORT_START_DATE", "REPORT_START_TIME"}
    in_options = False
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        upper = stripped.upper()
        # Detect section headers
        if upper.startswith("[") and upper.endswith("]"):
            in_options = upper == "[OPTIONS]"
            skip_section = upper in strip_sections
            if not skip_section:
                lines.append(stripped)
            continue
        if skip_section:
            continue
        if in_options:
            # Remove date/time keys from OPTIONS
            key = stripped.split()[0].upper() if stripped.split() else ""
            if key in strip_options:
                continue
        lines.append(stripped)
    canonical = "\n".join(lines) + "\n"
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# External-override runners used by V4 same-state branches
# ---------------------------------------------------------------------------


def select_post_action(
    post_schedule: np.ndarray,
    elapsed_min: float,
    override_start_min: float,
    decision_interval_sec: int = 600,
) -> np.ndarray:
    """Select the active control step from a one- or two-dimensional schedule."""
    schedule = np.asarray(post_schedule, dtype=np.float64)
    if schedule.ndim == 1:
        return schedule.copy()
    if schedule.ndim != 2 or schedule.shape[0] < 1:
        raise ValueError("post_schedule must be a non-empty 1D or 2D array")
    if int(decision_interval_sec) <= 0:
        raise ValueError("decision_interval_sec must be positive")
    elapsed_after_override_sec = max(
        0.0, (float(elapsed_min) - float(override_start_min)) * 60.0
    )
    step = int(elapsed_after_override_sec // int(decision_interval_sec))
    return schedule[min(step, schedule.shape[0] - 1)].copy()


def managed_setting_write_required(
    in_prefix: bool,
    prefix_schedule_is_none: bool,
    post_control_mode: str,
) -> bool:
    """Whether Python may write managed settings for the current routing step."""
    if post_control_mode not in {"external_override", "native_rules"}:
        raise ValueError(
            "post_control_mode must be external_override or native_rules"
        )
    if bool(in_prefix):
        return not bool(prefix_schedule_is_none)
    return post_control_mode == "external_override"


def run_swmm_fixed_action(
    inp_path: str | Path,
    actuators: pd.DataFrame,
    priority_nodes: list[str],
    out_detail_csv: str | Path,
    event_id: str,
    duration_min: int,
    prefix_schedule: dict[float, np.ndarray] | None,
    override_start_min: float,
    post_action: np.ndarray | dict[str, float],
    control_step_sec: int = 300,
    decision_interval_sec: int = 600,
    stop_after_override_min: float | None = None,
    prefix_history_min: float | None = None,
    max_steps: int = 0,
    policy_id: str = "fixed_action",
    simulation_duration_min: int | None = None,
    cleanup_swmm_artifacts: bool = False,
    extra_prefix_link_ids: list[str] | None = None,
    extra_prefix_table: dict[float, np.ndarray] | None = None,
    record_node_ids: Iterable[str] | None = None,
    hydraulic_summary_start_min: float | None = None,
    out_action_trace_csv: str | Path | None = None,
    post_control_mode: str = "external_override",
) -> dict:
    """Run a fixed-action branch with external override at every step.

    Parameters
    ----------
    prefix_schedule : dict[float, ndarray] or None
        Mapping elapsed_min -> action vector for prefix replay.  When ``None``,
        no managed setting is written before the checkpoint and native SWMM
        controls advance deterministically from the event start.
    override_start_min : float
        Checkpoint time.  Before this, replay prefix.  After, write post_action.
    post_action : ndarray or dict
        Fixed action vector (or {actuator_id: setting}) for post-checkpoint.

    Records at every step:
      - requested_setting: the value we *intended* to write
      - target_setting: same as requested (written before step)
      - current_setting: read from Link *after* SWMM routing step
      - flow: read from Link after step
      - write_source: "prefix_replay" | "external_override"

    Hotstart is NEVER used.
    """
    if post_control_mode not in {"external_override", "native_rules"}:
        raise ValueError(
            "post_control_mode must be external_override or native_rules"
        )
    Simulation, Nodes, Links, RainGages = _require_pyswmm()
    inp_path = Path(inp_path)
    out_detail_csv = Path(out_detail_csv)
    out_detail_csv.parent.mkdir(parents=True, exist_ok=True)
    action_trace_path = (
        Path(out_action_trace_csv) if out_action_trace_csv is not None else None
    )
    if action_trace_path is not None:
        action_trace_path.parent.mkdir(parents=True, exist_ok=True)

    all_requested_ids = actuators["actuator_id"].astype(str).tolist()
    n_act = len(all_requested_ids)

    # Build post-action vector or schedule.
    if isinstance(post_action, dict):
        post_vec = np.ones(n_act, dtype=np.float64)
        for aid, val in post_action.items():
            if aid in all_requested_ids:
                post_vec[all_requested_ids.index(aid)] = float(val)
        post_schedule = post_vec
    else:
        post_array = np.asarray(post_action, dtype=np.float64)
        if post_array.ndim == 1:
            post_vec = post_array.ravel()
            if post_vec.size < n_act:
                padded = np.ones(n_act, dtype=np.float64)
                padded[:post_vec.size] = post_vec
                post_vec = padded
            post_schedule = post_vec
        elif post_array.ndim == 2:
            if post_array.shape[1] > n_act:
                post_array = post_array[:, :n_act]
            elif post_array.shape[1] < n_act:
                padded = np.ones((post_array.shape[0], n_act), dtype=np.float64)
                padded[:, :post_array.shape[1]] = post_array
                post_array = padded
            post_schedule = post_array
        else:
            raise ValueError("post_action must be a vector, schedule, or mapping")

    records: list[dict] = []
    t0 = time.time()
    use_hotstart_count = 0
    save_hotstart_count = 0
    action_trace_records: list[dict] = []

    with Simulation(str(inp_path)) as sim:
        sim.step_advance(int(control_step_sec))
        nodes = Nodes(sim)
        links = Links(sim)
        gages = RainGages(sim)
        node_ids = _ids_from_container(nodes, "nodeid")
        if record_node_ids is not None:
            requested_nodes = {str(node_id) for node_id in record_node_ids}
            node_ids = [node_id for node_id in node_ids if node_id in requested_nodes]
        node_objs = {nid: nodes[nid] for nid in node_ids}
        all_link_ids = _ids_from_container(links, "linkid")
        all_link_objs = {link_id: links[link_id] for link_id in all_link_ids}
        link_full_depths = {
            link_id: max(
                _as_float(getattr(obj, "full_depth", 0.0), 0.0), 1.0e-6
            )
            for link_id, obj in all_link_objs.items()
        }
        actuator_ids, link_objs = _get_existing_links(links, all_requested_ids)
        rain_ids = _ids_from_container(gages, "raingageid")
        rain_obj = gages[rain_ids[0]] if rain_ids else None

        # Extra prefix link objects (native control links for V3+ shared prefix)
        _extra_ids = list(extra_prefix_link_ids or [])
        _extra_table = extra_prefix_table or {}
        _extra_objs: dict = {}
        if _extra_ids:
            for lid in _extra_ids:
                try:
                    _extra_objs[lid] = links[lid]
                except Exception:
                    pass

        step_i = 0
        for _ in sim:
            elapsed_min = (sim.current_time - sim.start_time).total_seconds() / 60.0
            in_prefix = elapsed_min < float(override_start_min)
            native_prefix = in_prefix and prefix_schedule is None
            native_post = (
                not in_prefix and post_control_mode == "native_rules"
            )
            write_source = (
                "native_rules"
                if native_prefix
                else (
                    "prefix_replay"
                    if in_prefix
                    else (
                        "native_rule_override"
                        if native_post
                        else "external_override"
                    )
                )
            )

            # Determine requested action
            if in_prefix:
                if native_prefix:
                    requested = np.asarray(
                        [
                            _as_float(link_objs[aid].current_setting, 1.0)
                            for aid in actuator_ids
                        ],
                        dtype=np.float64,
                    )
                else:
                    replay = _nominal_action_at(prefix_schedule, elapsed_min)
                    if replay is not None:
                        requested = np.asarray(
                            replay[:len(actuator_ids)], dtype=np.float64
                        )
                    else:
                        requested = np.ones(len(actuator_ids), dtype=np.float64)
            else:
                requested = select_post_action(
                    post_schedule,
                    elapsed_min=elapsed_min,
                    override_start_min=override_start_min,
                    decision_interval_sec=decision_interval_sec,
                )[:len(actuator_ids)]

            # Write target_setting BEFORE routing step
            if managed_setting_write_required(
                in_prefix=in_prefix,
                prefix_schedule_is_none=prefix_schedule is None,
                post_control_mode=post_control_mode,
            ):
                for j, aid in enumerate(actuator_ids):
                    try:
                        link_objs[aid].target_setting = float(
                            np.clip(requested[j], 0.0, 1.0)
                        )
                    except Exception:
                        pass
            # V3: also write extra prefix link settings during prefix
            if in_prefix and _extra_table:
                extra_replay = _nominal_action_at(_extra_table, elapsed_min)
                if extra_replay is not None:
                    for j, lid in enumerate(_extra_ids):
                        if lid in _extra_objs:
                            try:
                                _extra_objs[lid].target_setting = _as_float(
                                    np.clip(extra_replay[j], 0.0, 1.0), 1.0)
                            except Exception:
                                pass

            rain = _as_float(rain_obj.rainfall, 0.0) if rain_obj is not None else 0.0
            if action_trace_path is not None:
                trace_row: dict = {
                    "elapsed_min": elapsed_min,
                    "write_source": write_source,
                }
                for j, aid in enumerate(actuator_ids):
                    trace_row[f"requested_setting:{aid}"] = float(requested[j])
                    trace_row[f"a:{aid}"] = _as_float(
                        link_objs[aid].current_setting, np.nan
                    )
                action_trace_records.append(trace_row)
            retain_row = (
                prefix_history_min is None
                or not in_prefix
                or elapsed_min
                >= float(override_start_min) - float(prefix_history_min)
            )
            if not retain_row:
                step_i += 1
                if max_steps and step_i >= int(max_steps):
                    sim.terminate_simulation()
                    break
                continue
            row: dict = {
                "event_id": event_id,
                "policy_id": str(policy_id),
                "elapsed_min": elapsed_min,
                "datetime": str(sim.current_time),
                "rainfall_mm_h": rain,
                "phase": phase_from_time(elapsed_min, duration_min),
                "override_active": True,
                "write_source": write_source,
            }
            # Node observations (after step)
            for nid, obj in node_objs.items():
                row[f"h:{nid}"] = _as_float(obj.depth, np.nan)
                row[f"head:{nid}"] = _as_float(getattr(obj, "head", np.nan), np.nan)
                row[f"storage_volume:{nid}"] = _as_float(getattr(obj, "volume", np.nan), np.nan)
                row[f"flood:{nid}"] = _as_float(obj.flooding, 0.0)
            if (
                hydraulic_summary_start_min is None
                or elapsed_min >= float(hydraulic_summary_start_min)
            ):
                node_volumes = [
                    _as_float(getattr(obj, "volume", 0.0), 0.0)
                    for obj in node_objs.values()
                ]
                flood_rates = [
                    _as_float(getattr(obj, "flooding", 0.0), 0.0)
                    for obj in node_objs.values()
                ]
                fullness = np.asarray(
                    [
                        _as_float(getattr(obj, "depth", 0.0), 0.0)
                        / link_full_depths[link_id]
                        for link_id, obj in all_link_objs.items()
                    ],
                    dtype=float,
                )
                excess = np.maximum(fullness - 1.0, 0.0)
                row["system_stored_volume_m3"] = float(np.sum(node_volumes))
                row["tfv_rate_m3s"] = float(np.sum(flood_rates))
                row["excess_fullness_mean"] = float(np.mean(excess))
                row["excess_fullness_p95"] = float(np.quantile(excess, 0.95))
                row["excess_fullness_fraction"] = float(np.mean(excess > 0.05))
            # Link observations (after step -- readback from SWMM)
            for j, aid in enumerate(actuator_ids):
                try:
                    row[f"requested_setting:{aid}"] = float(requested[j])
                    row[f"target_setting:{aid}"] = float(
                        np.clip(requested[j], 0.0, 1.0)
                    )
                    row[f"a:{aid}"] = _as_float(link_objs[aid].current_setting, 1.0)
                    row[f"setting:{aid}"] = _as_float(link_objs[aid].current_setting, np.nan)
                    row[f"actual_setting:{aid}"] = _as_float(
                        link_objs[aid].current_setting, np.nan
                    )
                    row[f"readback_setting:{aid}"] = _as_float(
                        link_objs[aid].current_setting, np.nan
                    )
                    row[f"flow:{aid}"] = _as_float(link_objs[aid].flow, np.nan)
                except Exception:
                    row[f"requested_setting:{aid}"] = float(requested[j])
                    row[f"target_setting:{aid}"] = float(
                        np.clip(requested[j], 0.0, 1.0)
                    )
                    row[f"a:{aid}"] = np.nan
                    row[f"setting:{aid}"] = np.nan
                    row[f"actual_setting:{aid}"] = np.nan
                    row[f"readback_setting:{aid}"] = np.nan
                    row[f"flow:{aid}"] = np.nan
            # V3: record extra prefix link settings
            for lid in _extra_ids:
                if lid in _extra_objs:
                    try:
                        row[f"setting:{lid}"] = _as_float(_extra_objs[lid].current_setting, np.nan)
                    except Exception:
                        row[f"setting:{lid}"] = np.nan
            records.append(row)

            step_i += 1
            if max_steps and step_i >= int(max_steps):
                sim.terminate_simulation()
                break
            if (
                stop_after_override_min is not None
                and elapsed_min
                >= float(override_start_min) + float(stop_after_override_min)
            ):
                sim.terminate_simulation()
                break

    detail = pd.DataFrame(records)
    detail_tmp = out_detail_csv.with_name(f"{out_detail_csv.name}.tmp")
    detail.to_csv(detail_tmp, index=False)
    os.replace(detail_tmp, out_detail_csv)
    if action_trace_path is not None:
        trace_tmp = action_trace_path.with_name(f"{action_trace_path.name}.tmp")
        pd.DataFrame(action_trace_records).to_csv(trace_tmp, index=False)
        os.replace(trace_tmp, action_trace_path)
    kpis = compute_kpis(detail, priority_nodes, dt_sec=control_step_sec)
    kpis.update({
        "event_id": str(event_id),
        "policy_id": str(policy_id),
        "detail_file": str(out_detail_csv),
        "override_start_min": float(override_start_min),
        "decision_interval_sec": int(decision_interval_sec),
        "post_control_mode": str(post_control_mode),
        "stop_after_override_min": (
            float(stop_after_override_min)
            if stop_after_override_min is not None
            else None
        ),
        "prefix_history_min": (
            float(prefix_history_min) if prefix_history_min is not None else None
        ),
        "rows": int(len(detail)),
        "wall_time_sec": float(time.time() - t0),
        "prefix_rows": int(
            len(
                detail[
                    detail["write_source"].isin(
                        ["prefix_replay", "native_rules"]
                    )
                ]
            )
        ),
        "post_rows": int(
            len(
                detail[
                    detail["write_source"].isin(
                        ["external_override", "native_rule_override"]
                    )
                ]
            )
        ),
        "hotstart_used": False,
        "use_hotstart_call_count": use_hotstart_count,
        "save_hotstart_call_count": save_hotstart_count,
        "action_trace_file": (
            str(action_trace_path) if action_trace_path is not None else None
        ),
    })
    if cleanup_swmm_artifacts:
        for suffix in (".out", ".rpt"):
            try:
                inp_path.with_suffix(suffix).unlink(missing_ok=True)
            except OSError:
                pass
    return kpis
