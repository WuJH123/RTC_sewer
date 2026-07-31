from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd


RESIDUAL_ACTUATORS_V8 = (
    "RTC_IN_01", "RTC_OUT_01", "RTC_IN_02", "RTC_OUT_02", "RTC_IN_03", "RTC_OUT_03",
    "HS2512760.1", "gbz1.8", "ADD301.2", "ADD301.3",
)
CONTINUOUS_RESIDUAL_ACTUATORS_V8 = RESIDUAL_ACTUATORS_V8[:8]
BINARY_RESIDUAL_PUMPS_V8 = RESIDUAL_ACTUATORS_V8[8:]
BOUNDARY_REGULATORS_V8 = (
    "ADD424.1", "ADD424.2", "ADD424.3", "cc006.1", "dwxh.2", "Zhongyi-2.2",
    "RTC_IN_02", "RTC_OUT_02", "RTC_OUT_01", "RTC_OUT_03",
)


def stable_hash(payload: object, *, seed: int = 0) -> str:
    encoded = json.dumps(
        {"seed": int(seed), "payload": payload},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def allocate_v8_case_budget(
    *,
    total_cases: int,
    train_boundary_cases: int,
    calibration_cases: int,
    validation_cases: int,
) -> dict[str, int]:
    total = int(total_cases)
    allocation = {
        "fit_boundary": int(train_boundary_cases),
        "calibration_boundary": int(calibration_cases),
        "locked_validation_boundary": int(validation_cases),
    }
    deployment = total - sum(allocation.values())
    if deployment <= 0:
        raise ValueError("total_cases must leave positive fit_deployment capacity")
    return {"fit_deployment": deployment, **allocation}


def _normalise_rainfall_table(rainfall: pd.DataFrame) -> pd.DataFrame:
    frame = rainfall.copy()
    frame["event_id"] = frame["event_id"].astype(str)
    if "rain_id" not in frame:
        frame["rain_id"] = frame["event_id"].str.split("_", n=1).str[0]
    if "pattern" not in frame:
        frame["pattern"] = frame["event_id"].str.split("_", n=2).str[-1]
    if "duration_min" not in frame:
        frame["duration_min"] = (
            frame["event_id"].str.extract(r"_D([0-9]+)_", expand=False).fillna("0").astype(float)
        )
    return frame


def select_v8_event_roles(
    rainfall: pd.DataFrame,
    *,
    excluded_events: set[str],
    fit_events: int,
    calibration_events: int,
    validation_events: int,
    seed: int,
) -> pd.DataFrame:
    """Select fresh event roles while balancing return periods and rain patterns."""
    required = int(fit_events) + int(calibration_events) + int(validation_events)
    frame = _normalise_rainfall_table(rainfall)
    frame = frame[~frame["event_id"].isin({str(event) for event in excluded_events})].copy()
    if len(frame) < required:
        raise ValueError(f"insufficient fresh events: {len(frame)}/{required}")

    selected_indices: list[int] = []
    counts: dict[tuple[str, str], int] = {}
    rain_counts: dict[str, int] = {}
    pattern_counts: dict[str, int] = {}
    remaining = set(frame.index)
    while len(selected_indices) < required:
        best = min(
            remaining,
            key=lambda index: (
                counts.get((str(frame.at[index, "rain_id"]), str(frame.at[index, "pattern"])), 0),
                rain_counts.get(str(frame.at[index, "rain_id"]), 0),
                pattern_counts.get(str(frame.at[index, "pattern"]), 0),
                stable_hash(str(frame.at[index, "event_id"]), seed=seed),
            ),
        )
        selected_indices.append(best)
        remaining.remove(best)
        rain_id = str(frame.at[best, "rain_id"])
        pattern = str(frame.at[best, "pattern"])
        counts[(rain_id, pattern)] = counts.get((rain_id, pattern), 0) + 1
        rain_counts[rain_id] = rain_counts.get(rain_id, 0) + 1
        pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

    roles = ["fit"] * int(fit_events) + ["calibration"] * int(calibration_events) + ["locked_validation"] * int(validation_events)
    out = frame.loc[selected_indices].reset_index(drop=True)
    out["role"] = roles
    out["split"] = np.where(out["role"].eq("locked_validation"), "validation", "train")
    return out


def phase_start_min(duration_min: float, phase: str) -> float:
    phase = str(phase)
    if phase == "rising":
        return 0.25 * float(duration_min)
    if phase == "peak":
        return 0.55 * float(duration_min)
    if phase == "recession":
        return float(duration_min) + 30.0
    raise ValueError(f"unsupported phase: {phase}")


def temporal_delta_profile(delta: float, phase: str, *, horizon_steps: int = 6, variant: str = "hold") -> list[float]:
    profile = np.zeros(int(horizon_steps), dtype=np.float32)
    value = float(delta)
    if variant == "delayed":
        start = 2 if phase == "rising" else 1
        stop = int(horizon_steps) - 1
    elif variant == "late":
        start = max(1, int(horizon_steps) // 2)
        stop = int(horizon_steps)
    else:
        start = 1 if phase == "rising" else 0
        stop = int(horizon_steps) if phase == "recession" else int(horizon_steps) - 1
    if stop <= start:
        stop = min(int(horizon_steps), start + 1)
    profile[start:stop] = value
    return profile.astype(float).tolist()


def binary_toggle_profile(reference_values: np.ndarray, phase: str) -> list[float]:
    base = (np.asarray(reference_values, dtype=np.float32) >= 0.5).astype(np.float32)
    target = 0.0 if base[0] >= 0.5 else 1.0
    out = base.copy()
    if phase == "rising":
        out[1:5] = target
    elif phase == "peak":
        out[:5] = target
    else:
        out[:3] = target
    return out.astype(float).tolist()


def build_v8_deployment_residual_specifications(
    *,
    action_ids: Sequence[str],
    reference_action_seq: np.ndarray,
    tier1_profiles: dict[str, list[float]],
    phase: str,
    magnitudes: Sequence[float] = (0.05, 0.10, 0.20),
) -> list[dict[str, object]]:
    """Build online-eligible Tier1 plus new-asset residual candidates."""
    action_ids = [str(value) for value in action_ids]
    missing = [actuator for actuator in RESIDUAL_ACTUATORS_V8 if actuator not in action_ids]
    if missing:
        raise ValueError(f"missing v8 residual actuators from canonical order: {missing}")
    reference = np.asarray(reference_action_seq, dtype=np.float32)
    base = {str(key): list(map(float, value)) for key, value in tier1_profiles.items()}
    specs: list[dict[str, object]] = []

    def append(mode: str, *, signed: dict[str, list[float]] | None = None, targets: dict[str, list[float]] | None = None) -> None:
        residual_ids = sorted(set((signed or {}).keys()) | set((targets or {}).keys()))
        specs.append({
            "family": "tier2_residual_v8_deployment",
            "kind": "legacy_plus_new_residual",
            "mode": mode,
            "actuators": sorted(set(base) | set(residual_ids)),
            "residual_actuators": residual_ids,
            "signed_profiles": {**base, **(signed or {})},
            "target_profiles": targets or {},
            "horizon_steps": 6,
            "sequence_semantics": "relative_to_same_state_no_control_reference",
            "intended_evidence_role": "deployment_boundary",
            "online_candidate_eligible": True,
            "tier": 2,
        })

    for actuator in CONTINUOUS_RESIDUAL_ACTUATORS_V8:
        idx = action_ids.index(actuator)
        setting = float(np.median(reference[:, idx]))
        directions = (-1.0, 1.0)
        if setting <= 0.02:
            directions = (1.0,)
        elif setting >= 0.98:
            directions = (-1.0,)
        for magnitude in magnitudes:
            for direction in directions:
                for variant in ("hold", "delayed"):
                    delta = float(direction) * abs(float(magnitude))
                    append(
                        f"{actuator}_{delta:+.2f}_{phase}_{variant}",
                        signed={actuator: temporal_delta_profile(delta, phase, variant=variant)},
                    )

    for pump in BINARY_RESIDUAL_PUMPS_V8:
        append(
            f"{pump}_binary_toggle_{phase}",
            targets={pump: binary_toggle_profile(reference[:, action_ids.index(pump)], phase)},
        )

    if {"RTC_IN_02", "RTC_OUT_02"}.issubset(action_ids):
        append(
            f"storage_interlock_RTC02_{phase}",
            signed={
                "RTC_IN_02": temporal_delta_profile(-0.10, phase, variant="hold"),
                "RTC_OUT_02": temporal_delta_profile(0.10, phase, variant="late"),
            },
        )
    return specs


def _available(items: Sequence[str], action_ids: set[str], *, limit: int | None = None) -> list[str]:
    out = [str(item) for item in items if str(item) in action_ids]
    return out if limit is None else out[: int(limit)]


def build_v8_boundary_specifications(
    *,
    phase: str,
    action_ids: Sequence[str],
) -> list[dict[str, object]]:
    """Build mixed deployment-boundary and offline rejection candidates.

    Deployment-boundary rows are eligible for online use after passing the
    effect gate. Offline rows exist only to teach the model the unsafe side of
    the PFV/peak boundary.
    """
    action_set = {str(value) for value in action_ids}
    priority = _available(("ADD424.1", "ADD424.2", "ADD424.3", "cc006.1", "dwxh.2", "Zhongyi-2.2"), action_set)
    outlet = _available(("RTC_OUT_01", "RTC_OUT_02", "RTC_OUT_03", "cc006.1", "dwxh.2", "ADD424.1", "ADD424.2", "ADD424.3"), action_set)
    mixed = _available(("ADD424.1", "ADD424.3", "cc006.1", "RTC_OUT_01", "RTC_OUT_02", "RTC_IN_02", "dwxh.2", "Zhongyi-2.2"), action_set)
    pump_storage = _available(("ADD301.2", "ADD301.3", "RTC_OUT_01", "RTC_OUT_02", "RTC_OUT_03", "cc006.1", "dwxh.2"), action_set)
    specs: list[dict[str, object]] = []

    def continuous_group(
        *,
        family: str,
        mode: str,
        actuators: list[str],
        magnitude: float,
        variant: str,
        online: bool,
        role: str,
    ) -> None:
        if not actuators:
            return
        specs.append({
            "family": family,
            "kind": "strong_counterfactual" if not online else "legacy_plus_new_residual",
            "mode": mode,
            "actuators": actuators,
            "signed_profiles": {
                actuator: temporal_delta_profile(-abs(float(magnitude)), phase, variant=variant)
                for actuator in actuators
            },
            "horizon_steps": 6,
            "sequence_semantics": "relative_to_same_state_no_control_reference",
            "intended_evidence_role": role,
            "online_candidate_eligible": bool(online),
            "stress_magnitude": float(magnitude),
            "stress_profile": variant,
        })

    continuous_group(
        family="tier2_residual_v8_boundary",
        mode=f"deployment_priority_restrict_0p20_{phase}",
        actuators=priority[:4],
        magnitude=0.20,
        variant="hold",
        online=True,
        role="deployment_boundary",
    )
    continuous_group(
        family="tier2_residual_v8_boundary",
        mode=f"deployment_outlet_restrict_0p20_{phase}",
        actuators=outlet[:4],
        magnitude=0.20,
        variant="delayed",
        online=True,
        role="deployment_boundary",
    )
    offline_groups = {
        "priority": priority[:8],
        "outlet": outlet[:8],
        "mixed": mixed[:8],
    }
    for group_name, group in offline_groups.items():
        for magnitude in (0.35, 0.50, 0.70, 0.85, 1.00):
            for variant in ("hold", "delayed", "late"):
                continuous_group(
                    family="tier2_residual_v8_safety_boundary",
                    mode=f"offline_{group_name}_close_{str(magnitude).replace('.', 'p')}_{phase}_{variant}",
                    actuators=group,
                    magnitude=magnitude,
                    variant=variant,
                    online=False,
                    role="offline_safety_rejection_only",
                )
    pumps = [actuator for actuator in ("ADD301.2", "ADD301.3") if actuator in action_set]
    continuous = [actuator for actuator in pump_storage if actuator not in pumps]
    if pumps and continuous:
        for magnitude in (0.50, 0.85, 1.00):
            for variant in ("hold", "delayed"):
                pump_target = (
                    [1.0, 1.0, 1.0, 1.0, 0.0, 0.0]
                    if phase == "recession"
                    else [0.0, 1.0, 1.0, 1.0, 1.0, 0.0]
                )
                specs.append({
                    "family": "tier2_residual_v8_safety_boundary",
                    "kind": "strong_counterfactual",
                    "mode": f"offline_pump_storage_stress_{str(magnitude).replace('.', 'p')}_{phase}_{variant}",
                    "actuators": pumps + continuous[:6],
                    "signed_profiles": {
                        actuator: temporal_delta_profile(-magnitude, phase, variant=variant)
                        for actuator in continuous[:6]
                    },
                    "target_profiles": {pump: pump_target for pump in pumps},
                    "horizon_steps": 6,
                    "sequence_semantics": "relative_to_same_state_no_control_reference",
                    "intended_evidence_role": "offline_safety_rejection_only",
                    "online_candidate_eligible": False,
                    "stress_magnitude": magnitude,
                    "stress_profile": variant,
                })
    return specs


def summarize_v8_manifest_preflight(
    manifest: pd.DataFrame,
    *,
    target_cases: int,
    min_locked_validation_cases: int,
    min_locked_validation_events: int,
) -> dict[str, object]:
    candidates = manifest[manifest["branch"].astype(str).eq("B")].copy() if len(manifest) else manifest.copy()
    locked = candidates[candidates["event_role"].astype(str).eq("locked_validation")].copy() if len(candidates) else candidates
    train_events = set(candidates.loc[candidates.get("split", pd.Series(dtype=str)).astype(str).eq("train"), "event_id"].astype(str)) if len(candidates) else set()
    validation_events = set(candidates.loc[candidates.get("split", pd.Series(dtype=str)).astype(str).eq("validation"), "event_id"].astype(str)) if len(candidates) else set()
    noop_fraction = float(candidates["is_noop"].astype(bool).mean()) if len(candidates) and "is_noop" in candidates else 1.0
    shapes_ok = True
    if len(candidates) and "materialized_candidate_action_sequence" in candidates:
        def _shape_ok(value: object) -> bool:
            try:
                arr = np.asarray(json.loads(value) if isinstance(value, str) else value)
                return arr.shape == (6, 36)
            except Exception:
                return False
        shapes_ok = bool(candidates["materialized_candidate_action_sequence"].map(_shape_ok).all())
    checks = {
        "target_case_count": int(len(candidates)) == int(target_cases),
        "no_op_rate_le_5pct": noop_fraction <= 0.05,
        "event_group_split_disjoint": not bool(train_events & validation_events),
        "locked_validation_case_support": int(len(locked)) >= int(min_locked_validation_cases),
        "locked_validation_event_support": int(locked["event_id"].astype(str).nunique()) >= int(min_locked_validation_events) if len(locked) else False,
        "canonical_shape_H36": shapes_ok,
    }
    role_counts = candidates["event_role"].value_counts().astype(int).to_dict() if len(candidates) and "event_role" in candidates else {}
    evidence_counts = (
        candidates["intended_evidence_role"].value_counts().astype(int).to_dict()
        if len(candidates) and "intended_evidence_role" in candidates
        else {}
    )
    return {
        "passed": bool(all(checks.values())),
        "checks": {key: bool(value) for key, value in checks.items()},
        "planned_case_count": int(len(candidates)),
        "target_cases": int(target_cases),
        "role_counts": role_counts,
        "evidence_role_counts": evidence_counts,
        "locked_validation_cases": int(len(locked)),
        "locked_validation_event_coverage": int(locked["event_id"].astype(str).nunique()) if len(locked) else 0,
        "train_validation_event_overlap": sorted(train_events & validation_events),
        "noop_fraction": noop_fraction,
    }
