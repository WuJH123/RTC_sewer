from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


RESIDUAL_ACTUATORS = (
    "RTC_IN_01", "RTC_OUT_01", "RTC_IN_02", "RTC_OUT_02", "RTC_IN_03", "RTC_OUT_03",
    "HS2512760.1", "gbz1.8", "ADD301.2", "ADD301.3",
)
CONTINUOUS_RESIDUAL_ACTUATORS = RESIDUAL_ACTUATORS[:8]
BINARY_RESIDUAL_PUMPS = RESIDUAL_ACTUATORS[8:]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def freeze_dataset_manifest(dataset_path: Path, *, intended_rows: int) -> dict[str, object]:
    data = np.load(dataset_path, allow_pickle=True)
    rows = int(len(data["event_ids"]))
    if rows != int(intended_rows):
        raise ValueError(f"frozen base row count changed: {rows} != {intended_rows}")
    if tuple(data["candidate_action_seq"].shape[1:]) != (6, 36):
        raise ValueError("frozen base does not use [N,6,36] actions")
    return {
        "dataset": str(dataset_path.resolve()),
        "sha256": file_sha256(dataset_path),
        "rows": rows,
        "events": int(len(set(data["event_ids"].astype(str)))),
        "original_split_rows": {
            str(key): int(value)
            for key, value in zip(*np.unique(data["split"].astype(str), return_counts=True))
        },
        "frozen": True,
        "mutated": False,
        "intended_use": "v7_development_warm_start_only",
    }


def _stable_hash(value: object, seed: int) -> str:
    return hashlib.sha256(f"{seed}|{value}".encode()).hexdigest()


def select_fresh_event_roles(
    rainfall: pd.DataFrame,
    *,
    excluded_events: set[str],
    fit_events: int,
    calibration_events: int,
    validation_events: int,
    seed: int,
) -> pd.DataFrame:
    required = int(fit_events) + int(calibration_events) + int(validation_events)
    frame = rainfall.copy()
    frame["event_id"] = frame["event_id"].astype(str)
    frame = frame[~frame["event_id"].isin({str(value) for value in excluded_events})].copy()
    if len(frame) < required:
        raise ValueError(f"insufficient fresh events: {len(frame)}/{required}")
    if "pattern" not in frame:
        frame["pattern"] = frame["event_id"].str.split("_", n=2).str[-1]
    if "rain_id" not in frame:
        frame["rain_id"] = frame["event_id"].str.split("_", n=1).str[0]
    frame["_order"] = frame["event_id"].map(lambda value: _stable_hash(value, seed))
    frame = frame.sort_values(["_order", "rain_id", "pattern"]).reset_index(drop=True)

    selected: list[int] = []
    pattern_count: dict[str, int] = {}
    rain_count: dict[str, int] = {}
    remaining = set(frame.index)
    while len(selected) < required:
        best = min(
            remaining,
            key=lambda index: (
                pattern_count.get(str(frame.at[index, "pattern"]), 0),
                rain_count.get(str(frame.at[index, "rain_id"]), 0),
                str(frame.at[index, "_order"]),
            ),
        )
        remaining.remove(best)
        selected.append(best)
        pattern = str(frame.at[best, "pattern"])
        rain_id = str(frame.at[best, "rain_id"])
        pattern_count[pattern] = pattern_count.get(pattern, 0) + 1
        rain_count[rain_id] = rain_count.get(rain_id, 0) + 1
    chosen = frame.loc[selected].reset_index(drop=True)
    role_order = (
        ["fit"] * int(fit_events)
        + ["calibration"] * int(calibration_events)
        + ["locked_validation"] * int(validation_events)
    )
    chosen["role"] = role_order
    chosen["split"] = np.where(chosen["role"].eq("locked_validation"), "validation", "train")
    return chosen.drop(columns=["_order"])


def select_safe_tier1_bases(
    candidates: pd.DataFrame,
    *,
    pfv_abs_margin_m3: float,
    pfv_rel_margin: float,
) -> pd.DataFrame:
    frame = candidates.copy()
    reference_pfv = (
        pd.to_numeric(frame["reference_PFV"], errors="coerce").fillna(0.0)
        if "reference_PFV" in frame
        else pd.Series(np.zeros(len(frame)), index=frame.index)
    )
    margin = np.maximum(float(pfv_abs_margin_m3), reference_pfv.to_numpy(float) * float(pfv_rel_margin))
    frame["tier1_safe"] = (
        pd.to_numeric(frame["delta_PFV"], errors="coerce").to_numpy(float) <= margin
    ) & (pd.to_numeric(frame["delta_peak"], errors="coerce").to_numpy(float) <= 0.0)
    safe = frame[frame["tier1_safe"]].copy()
    if safe.empty:
        return safe
    sort_columns = ["event_id", "phase", "delta_TFV", "delta_PFV", "delta_peak"]
    return (
        safe.sort_values(sort_columns)
        .groupby(["event_id", "phase"], as_index=False, sort=False)
        .head(1)
        .reset_index(drop=True)
    )


def select_deployment_tier1_bases(
    candidates: pd.DataFrame,
    *,
    pfv_abs_margin_m3: float,
    pfv_rel_margin: float,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Select holdout bases using a phase policy learned from fit events only."""
    required = {"event_id", "phase", "role", "tier1_mode", "delta_PFV", "delta_TFV", "delta_peak"}
    missing = required - set(candidates.columns)
    if missing:
        raise KeyError(f"Tier 1 deployment selection lacks columns: {sorted(missing)}")
    frame = candidates.copy()
    reference_pfv = (
        pd.to_numeric(frame["reference_PFV"], errors="coerce").fillna(0.0)
        if "reference_PFV" in frame
        else pd.Series(np.zeros(len(frame)), index=frame.index)
    )
    margin = np.maximum(float(pfv_abs_margin_m3), reference_pfv.to_numpy(float) * float(pfv_rel_margin))
    frame["tier1_safe"] = (
        pd.to_numeric(frame["delta_PFV"], errors="coerce").to_numpy(float) <= margin
    ) & (pd.to_numeric(frame["delta_peak"], errors="coerce").to_numpy(float) <= 0.0)
    fit = frame[frame["role"].astype(str).eq("fit")].copy()
    if fit.empty:
        raise ValueError("fit events are required to learn the Tier 1 deployment policy")
    mode_score = (
        fit.groupby(["phase", "tier1_mode"], as_index=False)
        .agg(
            safe_fraction=("tier1_safe", "mean"),
            median_delta_TFV=("delta_TFV", "median"),
            independent_events=("event_id", "nunique"),
        )
        .sort_values(
            ["phase", "safe_fraction", "median_delta_TFV", "independent_events", "tier1_mode"],
            ascending=[True, False, True, False, True],
        )
    )
    best = mode_score.groupby("phase", as_index=False, sort=False).head(1)
    phase_policy = dict(zip(best["phase"].astype(str), best["tier1_mode"].astype(str)))
    expected_phases = set(frame["phase"].astype(str))
    if set(phase_policy) != expected_phases:
        raise ValueError(f"fit data do not cover all deployment phases: {sorted(expected_phases - set(phase_policy))}")

    fit_selected = select_safe_tier1_bases(
        fit,
        pfv_abs_margin_m3=pfv_abs_margin_m3,
        pfv_rel_margin=pfv_rel_margin,
    )
    fit_selected["selection_basis"] = "fit_same_context_true_safety"
    holdout = frame[~frame["role"].astype(str).eq("fit")].copy()
    holdout["deployment_mode"] = holdout["phase"].astype(str).map(phase_policy)
    holdout = holdout[holdout["tier1_mode"].astype(str).eq(holdout["deployment_mode"].astype(str))].copy()
    holdout = holdout.sort_values(["event_id", "phase", "tier1_mode"]).groupby(
        ["event_id", "phase"], as_index=False, sort=False
    ).head(1)
    holdout["selection_basis"] = "fit_only_phase_policy"
    selected = pd.concat([fit_selected, holdout], ignore_index=True)
    return selected, phase_policy


def _phase_profile(magnitude: float, phase: str) -> list[float]:
    active = 3 if str(phase) in {"peak", "recession"} else 2
    return [float(magnitude)] * active + [0.0] * (6 - active)


def _pump_toggle_profile(reference: np.ndarray, action_index: int, phase: str) -> list[float]:
    base = (np.asarray(reference[:, action_index]) >= 0.5).astype(np.float32)
    active = 3 if str(phase) in {"peak", "recession"} else 2
    target = 0.0 if base[0] >= 0.5 else 1.0
    profile = base.copy()
    profile[:active] = target
    return profile.astype(float).tolist()


def build_residual_specifications(
    *,
    action_ids: list[str],
    no_control_reference: np.ndarray,
    tier1_signed_profiles: dict[str, list[float]],
    phase: str,
    magnitude: float,
) -> list[dict[str, Any]]:
    missing = [actuator_id for actuator_id in RESIDUAL_ACTUATORS if actuator_id not in action_ids]
    if missing:
        raise ValueError(f"missing residual actuators from canonical action order: {missing}")
    base = {str(key): list(map(float, value)) for key, value in tier1_signed_profiles.items()}
    specs: list[dict[str, Any]] = []

    def append(
        mode: str,
        *,
        signed: dict[str, list[float]] | None = None,
        targets: dict[str, list[float]] | None = None,
    ) -> None:
        residual_ids = sorted(set((signed or {})) | set((targets or {})))
        spec: dict[str, Any] = {
            "family": "tier2_residual_v7_deployment",
            "kind": "legacy_plus_new_residual",
            "mode": mode,
            "actuators": sorted(set(base) | set(residual_ids)),
            "residual_actuators": residual_ids,
            "signed_profiles": {**base, **(signed or {})},
            "horizon_steps": 6,
            "sequence_semantics": "relative_to_same_state_no_control_reference",
            "online_candidate_eligible": True,
            "tier": 2,
        }
        if targets:
            spec["target_profiles"] = targets
        specs.append(spec)

    for actuator_id in CONTINUOUS_RESIDUAL_ACTUATORS:
        action_index = action_ids.index(actuator_id)
        setting = float(np.median(no_control_reference[:3, action_index]))
        if setting >= 0.98:
            contrasts = (-abs(float(magnitude)), -min(0.20, 2.0 * abs(float(magnitude))))
        elif setting <= 0.02:
            contrasts = (abs(float(magnitude)), min(0.20, 2.0 * abs(float(magnitude))))
        else:
            contrasts = (-abs(float(magnitude)), abs(float(magnitude)))
        for delta in contrasts:
            append(
                f"single_{actuator_id}_{delta:+.2f}_{phase}",
                signed={actuator_id: _phase_profile(delta, phase)},
            )
    for pump_id in BINARY_RESIDUAL_PUMPS:
        append(
            f"binary_toggle_{pump_id}_{phase}",
            targets={pump_id: _pump_toggle_profile(no_control_reference, action_ids.index(pump_id), phase)},
        )

    inlet, outlet = "RTC_IN_02", "RTC_OUT_02"
    inlet_sign = -1.0 if float(np.median(no_control_reference[:3, action_ids.index(inlet)])) >= 0.5 else 1.0
    outlet_sign = -1.0 if float(np.median(no_control_reference[3:, action_ids.index(outlet)])) >= 0.5 else 1.0
    append(
        f"storage_interlock_{inlet}_{outlet}_{phase}",
        signed={
            inlet: [inlet_sign * abs(float(magnitude))] * 3 + [0.0] * 3,
            outlet: [0.0] * 3 + [outlet_sign * abs(float(magnitude))] * 3,
        },
    )
    append(
        f"binary_pair_ADD301.2_ADD301.3_{phase}",
        targets={
            pump_id: _pump_toggle_profile(no_control_reference, action_ids.index(pump_id), phase)
            for pump_id in BINARY_RESIDUAL_PUMPS
        },
    )
    return specs
