from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config
from sewerrtc.models.temporal_graph_surrogate import TARGET_COLUMNS, load_horizon_surrogate, regression_report


def _configured_dir(cfg: dict, key: str, default: str) -> Path:
    raw = (cfg.get("outputs", {}) or {}).get(key, default)
    path = Path(raw)
    if not path.is_absolute():
        path = cfg_path(cfg, "project_root") / path
    return path


def _read_dataset(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        try:
            return pd.read_parquet(path)
        except ImportError:
            csv_path = path.with_suffix(".csv")
            if csv_path.exists():
                return pd.read_csv(csv_path)
            raise
    return pd.read_csv(path)


def _numeric_pfv(values) -> pd.Series:
    return pd.to_numeric(pd.Series(values), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _resolve_high_risk_threshold(
    true_pfv,
    pred_pfv,
    requested_threshold: float,
    quantile: float = 0.95,
    min_true_count: int = 1,
) -> dict:
    true = _numeric_pfv(true_pfv)
    pred = _numeric_pfv(pred_pfv)
    requested = float(requested_threshold)
    q = min(1.0, max(0.0, float(quantile)))
    min_count = max(1, int(min_true_count))

    requested_true_count = int((true >= requested).sum())
    requested_pred_count = int((pred >= requested).sum())
    if requested_true_count >= min_count:
        return {
            "threshold": requested,
            "mode": "absolute",
            "quantile": q,
            "min_true_count": min_count,
            "requested_threshold": requested,
            "requested_true_count": requested_true_count,
            "requested_pred_count": requested_pred_count,
            "true_count": requested_true_count,
            "pred_count": requested_pred_count,
        }

    positive_true = true[true > 0.0]
    if positive_true.empty:
        return {
            "threshold": requested,
            "mode": "no_positive_pfv",
            "quantile": q,
            "min_true_count": min_count,
            "requested_threshold": requested,
            "requested_true_count": requested_true_count,
            "requested_pred_count": requested_pred_count,
            "true_count": 0,
            "pred_count": requested_pred_count,
        }

    threshold = float(positive_true.quantile(q))
    if not np.isfinite(threshold):
        threshold = float(positive_true.max())
    adaptive_true_count = int((true >= threshold).sum())
    if adaptive_true_count < min_count and len(positive_true) >= min_count:
        threshold = float(positive_true.sort_values(ascending=False).iloc[min_count - 1])
        adaptive_true_count = int((true >= threshold).sum())
    adaptive_pred_count = int((pred >= threshold).sum())
    return {
        "threshold": threshold,
        "mode": "adaptive_quantile",
        "quantile": q,
        "min_true_count": min_count,
        "requested_threshold": requested,
        "requested_true_count": requested_true_count,
        "requested_pred_count": requested_pred_count,
        "true_count": adaptive_true_count,
        "pred_count": adaptive_pred_count,
    }


def _paired_direction_report(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    high_risk_threshold: float,
    reference_policy: str = "no_control",
    validation_gate: dict | None = None,
) -> tuple[pd.DataFrame, dict]:
    work = frame.reset_index(drop=True).copy()
    pred = predictions.reset_index(drop=True)
    for target in ("PFV_H", "TFV_H", "peak_TFV_rate_H"):
        work[f"pred_{target}"] = pd.to_numeric(pred[f"pred_{target}"], errors="coerce").fillna(0.0)
    keys = [c for c in ("event_id", "row_index", "elapsed_min") if c in work]
    if "event_id" not in keys or not keys:
        return pd.DataFrame(), {"paired_rows": 0, "reason": "missing_pair_keys"}
    ref = work[work["policy_id"].astype(str).eq(reference_policy)].copy()
    candidates = work[~work["policy_id"].astype(str).eq(reference_policy)].copy()
    if ref.empty or candidates.empty:
        return pd.DataFrame(), {"paired_rows": 0, "reason": "missing_reference_or_candidates"}
    targets = ("PFV_H", "TFV_H", "peak_TFV_rate_H")
    keep = keys + [*targets, *(f"pred_{target}" for target in targets)]
    # ``frame`` already carries paired effect labels called ``reference_*``.
    # Do not reuse those names here: merging the no-control rows would cause
    # pandas to append ``_x/_y`` suffixes and silently break the direction
    # audit. These fields are deliberately local to this audit table.
    rename = {
        target: f"paired_reference_true_{target}"
        for target in targets
    } | {
        f"pred_{target}": f"paired_reference_pred_{target}"
        for target in targets
    }
    ref = ref[keep].drop_duplicates(keys).rename(columns=rename)
    paired = candidates.merge(ref, on=keys, how="inner")
    paired = paired[
        pd.to_numeric(paired["paired_reference_true_PFV_H"], errors="coerce") >= float(high_risk_threshold)
    ].copy()
    gate_cfg = validation_gate or {}
    tolerances = {
        "PFV_H": (float(gate_cfg.get("pfv_direction_tolerance_abs", 100.0)),
                  float(gate_cfg.get("pfv_direction_tolerance_frac", 0.005))),
        "TFV_H": (float(gate_cfg.get("tfv_direction_tolerance_abs", 100.0)),
                  float(gate_cfg.get("tfv_direction_tolerance_frac", 0.005))),
        "peak_TFV_rate_H": (float(gate_cfg.get("peak_direction_tolerance_abs", 0.5)),
                             float(gate_cfg.get("peak_direction_tolerance_frac", 0.01))),
    }
    for target in targets:
        paired[f"true_delta_{target}"] = pd.to_numeric(paired[target], errors="coerce") - pd.to_numeric(
            paired[f"paired_reference_true_{target}"], errors="coerce"
        )
        paired[f"pred_delta_{target}"] = pd.to_numeric(paired[f"pred_{target}"], errors="coerce") - pd.to_numeric(
            paired[f"paired_reference_pred_{target}"], errors="coerce"
        )
        abs_tol, frac_tol = tolerances[target]
        paired[f"direction_tolerance_{target}"] = np.maximum(
            abs_tol,
            frac_tol * pd.to_numeric(paired[f"paired_reference_true_{target}"], errors="coerce").abs(),
        )
        true_sign = np.sign(
            paired[f"true_delta_{target}"].where(
                paired[f"true_delta_{target}"].abs() > paired[f"direction_tolerance_{target}"], 0.0
            )
        )
        pred_sign = np.sign(
            paired[f"pred_delta_{target}"].where(
                paired[f"pred_delta_{target}"].abs() > paired[f"direction_tolerance_{target}"], 0.0
            )
        )
        paired[f"direction_correct_{target}"] = true_sign == pred_sign
    pfv_tol = paired["direction_tolerance_PFV_H"]
    tfv_tol = paired["direction_tolerance_TFV_H"]
    peak_tol = paired["direction_tolerance_peak_TFV_rate_H"]
    pred_safe = (
        paired["pred_delta_PFV_H"].le(pfv_tol)
        & paired["pred_delta_TFV_H"].le(tfv_tol)
        & paired["pred_delta_peak_TFV_rate_H"].le(peak_tol)
    )
    true_safe = (
        paired["true_delta_PFV_H"].le(pfv_tol)
        & paired["true_delta_TFV_H"].le(tfv_tol)
        & paired["true_delta_peak_TFV_rate_H"].le(peak_tol)
    )
    paired["predicted_joint_safe"] = pred_safe
    paired["true_joint_safe"] = true_safe
    predicted_safe_count = int(pred_safe.sum())
    summary = {
        "paired_rows": int(len(paired)),
        "paired_events": int(paired["event_id"].nunique()) if "event_id" in paired else 0,
        "PFV_direction_accuracy": float(paired["direction_correct_PFV_H"].mean()) if len(paired) else None,
        "TFV_direction_accuracy": float(paired["direction_correct_TFV_H"].mean()) if len(paired) else None,
        "peak_direction_accuracy": float(paired["direction_correct_peak_TFV_rate_H"].mean()) if len(paired) else None,
        "predicted_joint_safe_count": predicted_safe_count,
        "joint_safe_precision": float((pred_safe & true_safe).sum() / predicted_safe_count) if predicted_safe_count else None,
        "direction_tolerances": {
            "PFV_H": {"abs": tolerances["PFV_H"][0], "fraction": tolerances["PFV_H"][1]},
            "TFV_H": {"abs": tolerances["TFV_H"][0], "fraction": tolerances["TFV_H"][1]},
            "peak_TFV_rate_H": {"abs": tolerances["peak_TFV_rate_H"][0], "fraction": tolerances["peak_TFV_rate_H"][1]},
        },
    }
    return paired, summary


def _combine_effect_predictions(
    frame: pd.DataFrame,
    absolute: pd.DataFrame,
    effect: pd.DataFrame | None,
    reference_policy: str = "no_control",
) -> pd.DataFrame:
    if effect is None or not {"event_id", "row_index", "policy_id"}.issubset(frame.columns):
        return absolute
    keys = ["event_id", "row_index"]
    out = absolute.reset_index(drop=True).copy()
    eff = effect.reset_index(drop=True)
    work = frame.reset_index(drop=True)
    ref_idx = work[work["policy_id"].astype(str).eq(reference_policy)].copy()
    ref_idx["_row"] = np.arange(len(work))[work["policy_id"].astype(str).eq(reference_policy)]
    ref_map = {
        tuple(row[k] for k in keys): int(row["_row"])
        for _, row in ref_idx.iterrows()
    }
    for target in TARGET_COLUMNS:
        col = f"pred_{target}"
        if col not in out or col not in eff:
            continue
        values = out[col].to_numpy(float).copy()
        effects = pd.to_numeric(eff[col], errors="coerce").fillna(0.0).to_numpy(float)
        for i, row in work.iterrows():
            base_i = ref_map.get(tuple(row[k] for k in keys), i)
            values[i] = values[base_i] + effects[i]
        out[col] = values
    return out


def _exact_effect_direction_report(
    frame: pd.DataFrame,
    predictions: pd.DataFrame,
    validation_gate: dict,
) -> tuple[pd.DataFrame, dict]:
    work = frame.reset_index(drop=True).copy()
    pred = predictions.reset_index(drop=True)
    tolerances = {
        "PFV_H": (float(validation_gate.get("pfv_direction_tolerance_abs", 100.0)), float(validation_gate.get("pfv_direction_tolerance_frac", 0.005))),
        "TFV_H": (float(validation_gate.get("tfv_direction_tolerance_abs", 100.0)), float(validation_gate.get("tfv_direction_tolerance_frac", 0.005))),
        "peak_TFV_rate_H": (float(validation_gate.get("peak_direction_tolerance_abs", 0.5)), float(validation_gate.get("peak_direction_tolerance_frac", 0.01))),
    }
    for target, (abs_tol, frac_tol) in tolerances.items():
        work[f"true_delta_{target}"] = pd.to_numeric(work[f"effect_{target}"], errors="coerce").fillna(0.0)
        work[f"pred_delta_{target}"] = pd.to_numeric(pred[f"pred_{target}"], errors="coerce").fillna(0.0)
        work[f"direction_tolerance_{target}"] = np.maximum(
            abs_tol,
            frac_tol * pd.to_numeric(work[f"reference_{target}"], errors="coerce").fillna(0.0).abs(),
        )
        tol = work[f"direction_tolerance_{target}"]
        true_sign = np.sign(work[f"true_delta_{target}"].where(work[f"true_delta_{target}"].abs() > tol, 0.0))
        pred_sign = np.sign(work[f"pred_delta_{target}"].where(work[f"pred_delta_{target}"].abs() > tol, 0.0))
        work[f"direction_correct_{target}"] = true_sign == pred_sign
    pred_safe = (
        work["pred_delta_PFV_H"].le(work["direction_tolerance_PFV_H"])
        & work["pred_delta_TFV_H"].le(0.0)
        & work["pred_delta_peak_TFV_rate_H"].le(work["direction_tolerance_peak_TFV_rate_H"])
    )
    true_safe = (
        work["true_delta_PFV_H"].le(work["direction_tolerance_PFV_H"])
        & work["true_delta_TFV_H"].le(0.0)
        & work["true_delta_peak_TFV_rate_H"].le(work["direction_tolerance_peak_TFV_rate_H"])
    )
    work["predicted_joint_safe"] = pred_safe
    work["true_joint_safe"] = true_safe
    predicted_safe_count = int(pred_safe.sum())
    return work, {
        "paired_rows": int(len(work)),
        "paired_events": int(work["event_id"].nunique()) if "event_id" in work else 0,
        "PFV_direction_accuracy": float(work["direction_correct_PFV_H"].mean()) if len(work) else None,
        "TFV_direction_accuracy": float(work["direction_correct_TFV_H"].mean()) if len(work) else None,
        "peak_direction_accuracy": float(work["direction_correct_peak_TFV_rate_H"].mean()) if len(work) else None,
        "predicted_joint_safe_count": predicted_safe_count,
        "joint_safe_precision": float((pred_safe & true_safe).sum() / predicted_safe_count) if predicted_safe_count else None,
        "effect_label_mode": "exact_no_control_replay_counterfactual",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wuhan.yaml")
    ap.add_argument("--dataset", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--exact-effect-dataset", default="")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--high-risk-pfv", type=float, default=None)
    ap.add_argument("--high-risk-quantile", type=float, default=None)
    ap.add_argument("--high-risk-min-count", type=int, default=None)
    ap.add_argument("--all-events-diagnostic", action="store_true")
    ap.add_argument("--fail-on-quality", action="store_true")
    ap.add_argument("--allow-quality-fail", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = cfg_path(cfg, "project_root")
    hcfg = cfg.get("horizon_surrogate", {}) or {}
    dataset_path = Path(args.dataset) if args.dataset else root / hcfg.get("output_dataset", "data/surrogate/horizon_mpc_dataset.parquet")
    if not dataset_path.exists():
        fallback = root / hcfg.get("fallback_output_dataset", "data/surrogate/horizon_mpc_dataset.csv")
        if fallback.exists():
            dataset_path = fallback
    if args.model:
        model_path = Path(args.model)
    else:
        model_dir = _configured_dir(cfg, "models", "outputs/models_paired_no_controls")
        temporal_path = model_dir / "horizon_temporal_gnn.pt"
        ridge_path = model_dir / "horizon_ridge_surrogate.npz"
        model_kind = str(hcfg.get("model_kind", hcfg.get("formal_model_kind", "ridge_baseline")))
        if model_kind == "temporal_gnn" and temporal_path.exists():
            model_path = temporal_path
        elif temporal_path.exists() and not ridge_path.exists():
            model_path = temporal_path
        else:
            model_path = ridge_path
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing horizon dataset: {dataset_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Missing horizon surrogate model: {model_path}")
    df = _read_dataset(dataset_path)
    validation_scope = "all_events_diagnostic"
    if not args.all_events_diagnostic:
        val_events_path = _configured_dir(cfg, "surrogate", "outputs/surrogate") / "horizon_surrogate_val_events.csv"
        if not val_events_path.exists():
            raise FileNotFoundError(f"Missing event-held-out validation list: {val_events_path}")
        val_events = set(pd.read_csv(val_events_path)["event_id"].dropna().astype(str))
        df = df[df["event_id"].astype(str).isin(val_events)].copy()
        if df.empty:
            raise ValueError("No horizon samples match the held-out validation event list")
        validation_scope = "event_id_held_out"
    model = load_horizon_surrogate(model_path)
    pred = model.predict(df)
    effect_pred = None
    if callable(getattr(model, "predict_effect", None)):
        effect_pred = model.predict_effect(df)
        pred = _combine_effect_predictions(df, pred, effect_pred)
    report = regression_report(df[TARGET_COLUMNS], pred, TARGET_COLUMNS)
    requested_threshold = float(
        args.high_risk_pfv
        if args.high_risk_pfv is not None
        else hcfg.get("high_risk_pfv_threshold", hcfg.get("high_risk_threshold_PFV_H", 1000.0))
    )
    threshold_quantile = float(
        args.high_risk_quantile
        if args.high_risk_quantile is not None
        else hcfg.get("high_risk_threshold_quantile", 0.95)
    )
    min_true_count = int(
        args.high_risk_min_count
        if args.high_risk_min_count is not None
        else hcfg.get("high_risk_min_true_count", 30)
    )
    true_pfv = _numeric_pfv(df["PFV_H"])
    pred_pfv = _numeric_pfv(pred["pred_PFV_H"])
    threshold_info = _resolve_high_risk_threshold(
        true_pfv,
        pred_pfv,
        requested_threshold=requested_threshold,
        quantile=threshold_quantile,
        min_true_count=min_true_count,
    )
    effective_threshold = float(threshold_info["threshold"])
    true_high = true_pfv >= effective_threshold
    pred_high = pred_pfv >= effective_threshold
    tp = int((true_high & pred_high).sum())
    fn = int((true_high & ~pred_high).sum())
    fp = int((~true_high & pred_high).sum())
    true_pos_count = int(true_high.sum())
    pred_pos_count = int(pred_high.sum())
    unsafe_recall = None if true_pos_count == 0 else float(tp / max(1, tp + fn))
    high_risk_precision = None if pred_pos_count == 0 else float(tp / max(1, tp + fp))
    gate_cfg = hcfg.get("validation_gate", {}) or {}
    paired_direction, direction_summary = _paired_direction_report(
        df,
        pred,
        effective_threshold,
        validation_gate=gate_cfg,
    )
    exact_direction = pd.DataFrame()
    exact_direction_summary = None
    exact_effect_path = Path(args.exact_effect_dataset) if args.exact_effect_dataset else None
    if exact_effect_path is not None and exact_effect_path.exists():
        exact = _read_dataset(exact_effect_path)
        if not args.all_events_diagnostic and "event_id" in exact:
            exact_val = exact[exact["event_id"].astype(str).isin(val_events)].copy()
            if not exact_val.empty:
                exact = exact_val
        exact_pred = model.predict_effect(exact)
        exact_direction, exact_direction_summary = _exact_effect_direction_report(exact, exact_pred, gate_cfg)
        direction_summary_for_gate = exact_direction_summary
    else:
        direction_summary_for_gate = direction_summary
    quality_reasons = []
    checks = (
        ("paired_rows", int(gate_cfg.get("min_paired_rows", 100)), "min"),
        ("PFV_direction_accuracy", float(gate_cfg.get("min_pfv_direction_accuracy", 0.70)), "min"),
        ("TFV_direction_accuracy", float(gate_cfg.get("min_tfv_direction_accuracy", 0.70)), "min"),
        ("peak_direction_accuracy", float(gate_cfg.get("min_peak_direction_accuracy", 0.80)), "min"),
        ("joint_safe_precision", float(gate_cfg.get("min_joint_safe_precision", 0.80)), "min"),
    )
    for key, threshold, _ in checks:
        value = direction_summary_for_gate.get(key)
        if value is None or not np.isfinite(float(value)) or float(value) < float(threshold):
            quality_reasons.append(f"{key}={value} < {threshold}")
    quality_gate = {"passed": not quality_reasons, "reasons": quality_reasons}
    summary = {
        "dataset": str(dataset_path),
        "model": str(model_path),
        "samples": int(len(df)),
        "validation_scope": validation_scope,
        "high_risk_threshold_PFV_H": effective_threshold,
        "high_risk_threshold_mode": threshold_info["mode"],
        "high_risk_threshold_quantile": float(threshold_info["quantile"]),
        "high_risk_min_true_count": int(threshold_info["min_true_count"]),
        "requested_high_risk_threshold_PFV_H": float(threshold_info["requested_threshold"]),
        "requested_true_high_risk_count": int(threshold_info["requested_true_count"]),
        "requested_pred_high_risk_count": int(threshold_info["requested_pred_count"]),
        "true_high_risk_count": true_pos_count,
        "pred_high_risk_count": pred_pos_count,
        "high_risk_binary_recall": unsafe_recall,
        "high_risk_binary_precision": high_risk_precision,
        "can_output": ["PFV_H", "TFV_H", "peak_TFV_rate_H"],
        "high_risk_paired_direction": direction_summary,
        "exact_counterfactual_direction": exact_direction_summary,
        "quality_gate_basis": "exact_no_control_replay_counterfactual" if exact_direction_summary is not None else "paired_policy_trajectory_approximation",
        "quality_gate": quality_gate,
    }
    out_dir = ensure_dir(Path(args.out_dir) if args.out_dir else _configured_dir(cfg, "surrogate", "outputs/surrogate"))
    report.to_csv(out_dir / "horizon_surrogate_full_validation.csv", index=False)
    paired_direction.to_csv(out_dir / "horizon_surrogate_high_risk_direction_audit.csv", index=False)
    if not exact_direction.empty:
        exact_direction.to_csv(out_dir / "horizon_surrogate_exact_effect_direction_audit.csv", index=False)
    (out_dir / "horizon_surrogate_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(report.to_string(index=False))
    enforce_quality = bool(args.fail_on_quality or hcfg.get("enforce_validation_gate", False)) and not args.allow_quality_fail
    if enforce_quality and not quality_gate["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
