from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TemporalJointCandidateConfig:
    horizon_steps: int = 6
    max_candidates: int = 512
    max_simultaneous_changes: int = 6
    max_change_points: int = 2
    continuous_max_delta: float = 0.10
    continuous_delta_levels: tuple[float, ...] = ()
    binary_pump_ids: tuple[str, ...] = ("ADD301.2", "ADD301.3")
    binary_pump_min_dwell_steps: int = 2
    max_pump_switches_per_event: int = 6
    storage_interlock: bool = True
    max_storage_actuators: int = 4
    allowed_candidate_ids: tuple[str, ...] = ()
    engineering_templates: tuple[dict[str, object], ...] = ()


def _column(frame: pd.DataFrame, name: str, default: object = "") -> pd.Series:
    if name in frame:
        return frame[name]
    return pd.Series([default] * len(frame), index=frame.index)


def _profiles(horizon: int, magnitude: float, phase: str = "unknown") -> list[tuple[str, np.ndarray]]:
    h = max(1, int(horizon))
    split = max(1, h // 2)
    early = np.zeros(h, dtype=np.float32)
    early[:split] = magnitude
    pulse = np.zeros(h, dtype=np.float32)
    pulse_lo = max(0, h // 3)
    pulse_hi = max(pulse_lo + 1, min(h, 2 * h // 3 + 1))
    pulse[pulse_lo:pulse_hi] = magnitude
    delayed = np.zeros(h, dtype=np.float32)
    delayed[split:] = magnitude
    ramp = np.full(h, magnitude, dtype=np.float32)
    ramp[0] = 0.5 * magnitude
    ramp[-max(1, h // 3):] = 0.0
    restore = np.zeros(h, dtype=np.float32)
    restore[:split] = magnitude
    profiles = {
        "ramp": ramp,
        "early_hold": early,
        "early_then_restore": early,
        "hold_then_restore": early,
        "pulse": pulse,
        "release_pulse": pulse,
        "delayed_hold": delayed,
        "ramp_restore": restore,
    }
    order = {
        "rising": ("delayed_hold", "ramp", "early_hold", "pulse", "ramp_restore"),
        "peak": ("early_hold", "pulse", "ramp_restore", "ramp", "delayed_hold"),
        "recession": ("ramp_restore", "pulse", "early_hold", "delayed_hold", "ramp"),
    }.get(str(phase).lower(), ("ramp", "early_hold", "pulse", "delayed_hold", "ramp_restore"))
    return [(name, profiles[name]) for name in order]


def _storage_groups(actuators: pd.DataFrame) -> dict[str, list[int]]:
    roles = _column(actuators, "storage_control_type").astype(str).str.lower()
    explicit = _column(actuators, "retrofit_storage_group").astype(str)
    groups: dict[str, list[int]] = {}
    for index, (role, group) in enumerate(zip(roles, explicit)):
        if not role.startswith("storage_"):
            continue
        key = group.strip() or f"storage_unmapped_{index}"
        groups.setdefault(key, []).append(index)
    return groups


def validate_candidate_sequence(
    candidate_action_seq: np.ndarray,
    reference_action_seq: np.ndarray,
    actuators: pd.DataFrame,
    config: TemporalJointCandidateConfig,
) -> dict[str, int | bool | str]:
    candidate = np.asarray(candidate_action_seq, dtype=np.float32)
    reference = np.asarray(reference_action_seq, dtype=np.float32)
    expected = (int(config.horizon_steps), len(actuators))
    if candidate.shape != expected or reference.shape != expected:
        return {"valid": False, "reason": f"shape_mismatch:{candidate.shape}:{reference.shape}"}
    if not np.isfinite(candidate).all() or np.any(candidate < 0.0) or np.any(candidate > 1.0):
        return {"valid": False, "reason": "setting_bounds"}
    ids = actuators["actuator_id"].astype(str).tolist()
    residual = candidate - reference
    changed = np.abs(residual) > 1.0e-7
    simultaneous = changed.sum(axis=1)
    max_simultaneous = int(simultaneous.max(initial=0))
    if max_simultaneous > int(config.max_simultaneous_changes):
        return {"valid": False, "reason": "simultaneous_limit", "max_simultaneous_changes": max_simultaneous}

    pump_dwell_violations = 0
    for pump_id in config.binary_pump_ids:
        if pump_id not in ids:
            continue
        values = candidate[:, ids.index(pump_id)]
        if not np.all(np.isin(values, np.asarray([0.0, 1.0], dtype=np.float32))):
            return {"valid": False, "reason": f"fractional_binary_pump:{pump_id}"}
        changes = np.flatnonzero(np.abs(np.diff(values)) > 1.0e-7) + 1
        boundaries = np.concatenate(([0], changes, [len(values)]))
        if any(
            int(boundaries[i + 1] - boundaries[i]) < int(config.binary_pump_min_dwell_steps)
            for i in range(1, len(boundaries) - 2)
        ):
            pump_dwell_violations += 1

    storage_interlock_violations = 0
    if config.storage_interlock:
        roles = _column(actuators, "storage_control_type").astype(str).str.lower().tolist()
        for indices in _storage_groups(actuators).values():
            inlet = [i for i in indices if roles[i] == "storage_inlet"]
            outlet = [i for i in indices if roles[i] == "storage_outlet"]
            if inlet and outlet:
                both = changed[:, inlet].any(axis=1) & changed[:, outlet].any(axis=1)
                storage_interlock_violations += int(both.sum())
    storage_indices = [
        i for i, role in enumerate(_column(actuators, "storage_control_type").astype(str).str.lower())
        if role.startswith("storage_")
    ]
    max_storage = int(changed[:, storage_indices].sum(axis=1).max(initial=0)) if storage_indices else 0
    change_points = int(np.max(np.sum(np.abs(np.diff(candidate, axis=0)) > 1.0e-7, axis=0), initial=0))
    valid = (
        pump_dwell_violations == 0
        and storage_interlock_violations == 0
        and max_storage <= int(config.max_storage_actuators)
        and change_points <= int(config.max_change_points)
    )
    reason = "ok" if valid else (
        "pump_dwell" if pump_dwell_violations else
        "storage_interlock" if storage_interlock_violations else
        "storage_simultaneous_limit" if max_storage > int(config.max_storage_actuators) else
        "change_point_limit"
    )
    return {
        "valid": valid,
        "reason": reason,
        "max_simultaneous_changes": max_simultaneous,
        "pump_dwell_violations": pump_dwell_violations,
        "storage_interlock_violations": storage_interlock_violations,
        "max_storage_actuators": max_storage,
        "max_change_points": change_points,
    }


def generate_temporal_joint_candidates(
    *,
    reference_action_seq: np.ndarray,
    actuators: pd.DataFrame,
    legacy_groups: Sequence[Sequence[str]] = (),
    paired_groups: Sequence[Sequence[str]] = (),
    phase: str,
    config: TemporalJointCandidateConfig,
) -> list[dict[str, object]]:
    reference = np.asarray(reference_action_seq, dtype=np.float32)
    expected = (int(config.horizon_steps), len(actuators))
    if reference.shape != expected:
        raise ValueError(f"reference_action_seq must be {expected}, got {reference.shape}")
    ids = actuators["actuator_id"].astype(str).tolist()
    id_to_index = {actuator_id: index for index, actuator_id in enumerate(ids)}
    allowed_ids = set(config.allowed_candidate_ids)
    legacy_mask = _column(actuators, "is_legacy_v8", False).astype(bool).to_numpy()
    pump_ids = set(config.binary_pump_ids)
    delta_levels = sorted({
        abs(float(value))
        for value in (config.continuous_delta_levels or (config.continuous_max_delta,))
        if 0.0 < abs(float(value)) <= abs(float(config.continuous_max_delta)) + 1.0e-9
    })
    if not delta_levels:
        delta_levels = [abs(float(config.continuous_max_delta))]
    candidates: list[dict[str, object]] = []
    signatures: set[bytes] = set()

    def append(label: str, tier: int, sequence: np.ndarray, targets: Iterable[str], rationale: str) -> None:
        if len(candidates) >= int(config.max_candidates):
            return
        seq = np.asarray(sequence, dtype=np.float32)
        report = validate_candidate_sequence(seq, reference, actuators, config)
        if not report.get("valid"):
            return
        signature = np.round(seq, 6).tobytes()
        if signature in signatures:
            return
        signatures.add(signature)
        candidates.append(
            {
                "label": label,
                "tier": int(tier),
                "candidate_action_seq": seq,
                "reference_action_seq": reference.copy(),
                "residual_action_seq": seq - reference,
                "target_actuators": list(targets),
                "phase": str(phase),
                "physical_rationale": rationale,
                "validation": report,
            }
        )

    append("reference_no_control", 0, reference, (), "Online predicted No-control/default sequence.")

    for template_no, template in enumerate(config.engineering_templates):
        targets = [
            str(item) for item in template.get("actuators", [])
            if str(item) in id_to_index and (not allowed_ids or str(item) in allowed_ids)
        ]
        if not targets:
            continue
        phase_filter = {str(item).lower() for item in template.get("phases", [])}
        if phase_filter and str(phase).lower() not in phase_filter:
            continue
        indices = [id_to_index[item] for item in targets]
        kind = str(template.get("kind", "continuous_profile"))
        label = str(template.get("label", f"engineered_template_{template_no}"))
        tier = int(template.get("tier", 1))
        seq = reference.copy()
        if kind == "binary_pump":
            start = max(0, min(int(config.horizon_steps) - 1, int(template.get("start_step", 0))))
            for index in indices:
                target = float(template.get("target_setting", 0.0 if reference[start, index] >= 0.5 else 1.0))
                seq[start:, index] = 1.0 if target >= 0.5 else 0.0
        else:
            magnitude = abs(float(template.get("magnitude", config.continuous_max_delta)))
            magnitude = min(magnitude, abs(float(config.continuous_max_delta)))
            direction = -1.0 if float(template.get("direction", -1.0)) < 0.0 else 1.0
            profile_name = str(template.get("profile", "ramp"))
            profiles = {name: values for name, values in _profiles(config.horizon_steps, direction * magnitude, phase)}
            profile = profiles.get(profile_name, next(iter(profiles.values())))
            seq[:, indices] = np.clip(seq[:, indices] + profile[:, None], 0.0, 1.0)
        append(
            f"engineered_{label}", tier, seq, targets,
            str(template.get("rationale", "Frozen engineering action template extracted from prior closed-loop evidence.")),
        )

    # Reserve the front of the finite beam for proven group semantics. If all
    # single-actuator profiles are emitted first, a large legacy registry can
    # consume the cap before Tier 1 and Tier 3 are ever considered.
    for group_no, group in enumerate(legacy_groups):
        targets = [item for item in group if item in id_to_index and (not allowed_ids or item in allowed_ids)]
        if not targets:
            continue
        indices = [id_to_index[item] for item in targets]
        for magnitude in delta_levels:
            for direction in (-1.0, 1.0):
                seq = reference.copy()
                profile = _profiles(config.horizon_steps, direction * magnitude, phase)[1][1]
                seq[:, indices] = np.clip(seq[:, indices] + profile[:, None], 0.0, 1.0)
                append(
                    f"tier1_legacy_group_{group_no}_{magnitude:.2f}_{direction:+.0f}", 1, seq, targets,
                    "Validated 26-facility legacy group retained independently of new-asset evidence.",
                )

    for group_no, group in enumerate(paired_groups):
        targets = [item for item in group if item in id_to_index and (not allowed_ids or item in allowed_ids)]
        if len(targets) < 2:
            continue
        indices = [id_to_index[item] for item in targets]
        for magnitude in delta_levels:
            for direction in (-1.0, 1.0):
                seq = reference.copy()
                profile = _profiles(config.horizon_steps, direction * magnitude, phase)[2][1]
                seq[:, indices] = np.clip(seq[:, indices] + profile[:, None], 0.0, 1.0)
                append(
                    f"tier3_paired_group_{group_no}_{magnitude:.2f}_{direction:+.0f}", 3, seq, targets,
                    "Paired action retained only when supplied by paired-evidence configuration.",
                )

    for index, actuator_id in enumerate(ids):
        if allowed_ids and actuator_id not in allowed_ids:
            continue
        if actuator_id in pump_ids:
            for start in sorted({0, max(0, int(config.horizon_steps) // 2)}):
                seq = reference.copy()
                target = 0.0 if reference[start, index] >= 0.5 else 1.0
                seq[start:, index] = target
                tier = 1 if legacy_mask[index] else 2
                append(
                    f"tier{tier}_binary_pump_{actuator_id}_from_h{start}", tier, seq, (actuator_id,),
                    "Binary pump switch with an explicit future change point and dwell projection.",
                )
            continue
        for magnitude in delta_levels:
            for direction in (-1.0, 1.0):
                for profile_name, profile in _profiles(config.horizon_steps, direction * magnitude, phase):
                    seq = reference.copy()
                    seq[:, index] = np.clip(seq[:, index] + profile, 0.0, 1.0)
                    tier = 1 if legacy_mask[index] else 2
                    append(
                        f"tier{tier}_{profile_name}_{actuator_id}_{magnitude:.2f}_{direction:+.0f}", tier, seq, (actuator_id,),
                        "Independent bounded actuator profile around the online No-control sequence.",
                    )

    return candidates
