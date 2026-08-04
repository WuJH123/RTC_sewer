from __future__ import annotations

import numpy as np
import pandas as pd

BINARY_ACTUATOR_IDS = {"ADD301.2", "ADD301.3"}


def generate_action_sequences(
    native_action: np.ndarray,
    actuators: pd.DataFrame,
    horizon_steps: int,
    max_delta: float = 0.08,
    include_hold: bool = True,
    priority_to_actuators: pd.DataFrame | None = None,
    max_sequences: int = 0,
    group_limit: int = 12,
    allowed_actuator_ids: set[str] | list[str] | tuple[str, ...] | str | None = None,
    blocked_actuator_ids: set[str] | list[str] | tuple[str, ...] | str | None = None,
    reference_sequence: np.ndarray | None = None,
    allowed_action_directions: dict[str, set[str] | list[str] | tuple[str, ...]] | None = None,
    empirical_hold_steps: int = 0,
    allowed_action_delta_limits: dict[str, dict[str, float]] | None = None,
) -> list[dict]:
    native = np.asarray(native_action, dtype=float).reshape(-1)
    horizon_steps = max(1, int(horizon_steps))
    max_delta = abs(float(max_delta))
    empirical_hold_steps = max(0, int(empirical_hold_steps))
    group_limit = max(1, int(group_limit))
    if reference_sequence is None:
        reference = np.tile(native[None, :], (horizon_steps, 1))
    else:
        reference = np.asarray(reference_sequence, dtype=float)
        if reference.ndim == 1:
            reference = np.tile(reference.reshape(1, -1), (horizon_steps, 1))
        if reference.shape != (horizon_steps, native.size):
            reference = np.resize(reference, (horizon_steps, native.size))
        reference = np.clip(np.nan_to_num(reference, nan=1.0), 0.0, 1.0)
        native = reference[0].copy()
    sequences: list[dict] = []

    def _append_sequence(label: str, seq: np.ndarray, rationale: str, targets: list[str] | str = "") -> None:
        target_text = ",".join(targets) if isinstance(targets, list) else str(targets or "")
        absolute = np.asarray(seq, dtype=np.float32)
        residual = absolute - reference.astype(np.float32)
        if allowed_action_directions:
            target_ids = [x.strip() for x in target_text.split(",") if x.strip()]
            for aid in target_ids:
                if aid not in id_to_idx:
                    continue
                idx = id_to_idx[aid]
                permitted = {str(x).strip().lower() for x in allowed_action_directions.get(aid, ())}
                if permitted:
                    if np.any(residual[:, idx] > 1.0e-7) and "increase" not in permitted:
                        return
                    if np.any(residual[:, idx] < -1.0e-7) and "decrease" not in permitted:
                        return
                if allowed_action_delta_limits and aid in allowed_action_delta_limits:
                    limits = allowed_action_delta_limits.get(aid, {}) or {}
                    positive = float(limits.get("increase", np.inf))
                    negative = float(limits.get("decrease", np.inf))
                    if np.any(residual[:, idx] > positive + 1.0e-7):
                        return
                    if np.any(residual[:, idx] < -negative - 1.0e-7):
                        return
        sequences.append(
            {
                "label": label,
                # ``sequence`` remains the execution-compatible absolute
                # setting for legacy callers. The explicit fields prevent
                # training/inference from confusing absolute settings with
                # residual changes around the No-control twin.
                "sequence": absolute,
                "absolute_sequence": absolute,
                "residual_sequence": residual,
                "reference_sequence": reference.astype(np.float32),
                "action_semantics": "absolute_from_no_control_reference",
                "target_actuators": target_text,
                "physical_rationale": rationale,
            }
        )

    if include_hold:
        _append_sequence(
            "hold_native",
            reference.copy(),
            "Fallback to the current bounded action sequence.",
        )
    if actuators.empty:
        return sequences
    ids = actuators["actuator_id"].astype(str).tolist()
    types = actuators.get("asset_role", actuators.get("link_type", pd.Series(["unknown"] * len(actuators)))).astype(str).str.lower()
    roles = actuators.get("storage_control_type", pd.Series([""] * len(actuators))).astype(str).str.lower()
    id_to_idx = {aid: i for i, aid in enumerate(ids)}

    def _parse_id_filter(value) -> set[str]:
        if value is None:
            return set()
        if isinstance(value, str):
            return {x.strip() for x in value.replace(";", ",").split(",") if x.strip()}
        return {str(x).strip() for x in value if str(x).strip()}

    allowed_ids = _parse_id_filter(allowed_actuator_ids)
    blocked_ids = _parse_id_filter(blocked_actuator_ids)

    def _asset_allowed(aid: str) -> bool:
        text = str(aid).strip()
        if allowed_ids and text not in allowed_ids:
            return False
        if text in blocked_ids:
            return False
        return True

    influence_rows = []
    if priority_to_actuators is not None and not priority_to_actuators.empty:
        for _, row in priority_to_actuators.iterrows():
            aid = str(row.get("actuator_id", "")).strip()
            if aid in id_to_idx and _asset_allowed(aid):
                influence_rows.append(row)

    def _append_single_asset_sequence(label: str, idx: int, first_delta: float, second_delta: float | None, rationale: str) -> None:
        seq = reference.copy()
        split = max(1, horizon_steps // 2)
        seq[:split, idx] = np.clip(seq[:split, idx] + float(first_delta), 0.0, 1.0)
        if second_delta is None:
            seq[split:, idx] = seq[split - 1, idx]
        else:
            seq[split:, idx] = np.clip(reference[split:, idx] + float(second_delta), 0.0, 1.0)
        _append_sequence(label, seq, rationale, ids[idx])

    def _delta_profile(kind: str, magnitude: float) -> np.ndarray:
        magnitude = float(magnitude)
        h = horizon_steps
        if kind == "ramp":
            if h <= 1:
                return np.asarray([magnitude], dtype=float)
            ramp = np.linspace(0.25 * magnitude, magnitude, h, dtype=float)
            restore = max(1, h // 3)
            ramp[-restore:] = np.linspace(0.5 * magnitude, 0.0, restore, dtype=float)
            return ramp
        if kind == "pulse":
            profile = np.zeros(h, dtype=float)
            lo = max(0, h // 3)
            hi = max(lo + 1, min(h, 2 * h // 3 + 1))
            profile[lo:hi] = magnitude
            return profile
        if kind == "early_then_restore":
            profile = np.zeros(h, dtype=float)
            split = max(1, h // 2)
            profile[:split] = magnitude
            return profile
        if kind == "through_horizon":
            return np.full(h, magnitude, dtype=float)
        if kind == "retain_then_release":
            profile = np.full(h, -abs(magnitude), dtype=float)
            late = max(1, h // 3)
            profile[-late:] = 0.5 * abs(magnitude)
            return profile
        return np.full(h, magnitude, dtype=float)

    def _append_binary_toggle_sequence(priority_node: str, aid: str, idx: int) -> None:
        """Generate a real 0->1 or 1->0 transition, never a fractional pump delta."""
        current_value = float(reference[0, idx])
        target = 0.0 if current_value >= 0.5 else 1.0
        prefix = min(3, horizon_steps)
        seq = reference.copy()
        seq[:prefix, idx] = target
        _append_sequence(
            f"binary_toggle|priority={priority_node}|actuator={aid}|target={int(target)}",
            seq,
            "Explicit binary pump toggle over the executable H3 prefix.",
            aid,
        )

    def _append_profile_sequence(label: str, indices: list[int], profile: np.ndarray, rationale: str) -> None:
        if not indices:
            return
        selected = []
        for idx in indices:
            if 0 <= int(idx) < len(ids) and int(idx) not in selected and _asset_allowed(ids[int(idx)]):
                selected.append(int(idx))
        if not selected:
            return
        seq = reference.copy()
        deltas = np.asarray(profile, dtype=float).reshape(-1)
        if deltas.size < horizon_steps:
            deltas = np.pad(deltas, (0, horizon_steps - deltas.size), mode="edge")
        deltas = deltas[:horizon_steps]
        for t in range(horizon_steps):
            seq[t, selected] = np.clip(reference[t, selected] + deltas[t], 0.0, 1.0)
        _append_sequence(label, seq, rationale, [ids[i] for i in selected])

    grouped_by_priority: dict[str, list[int]] = {}
    grouped_by_priority_role: dict[str, dict[str, list[int]]] = {}

    # Ensure max_sequences cannot hide the only physically valid binary move.
    for row in influence_rows:
        aid = str(row.get("actuator_id", "")).strip()
        if aid in BINARY_ACTUATOR_IDS and aid in id_to_idx:
            _append_binary_toggle_sequence(
                str(row.get("priority_node", "priority")), aid, id_to_idx[aid]
            )

    def _role_family(role_text: str) -> str:
        role = str(role_text or "").lower()
        if "storage" in role or "inlet" in role or "outlet" in role:
            return "storage"
        if "pump" in role:
            return "pump"
        if "orifice" in role or "weir" in role or "regulator" in role:
            return "regulator"
        return "other"

    for row in influence_rows:
        aid = str(row.get("actuator_id", "")).strip()
        idx = id_to_idx[aid]
        role_text = str(row.get("asset_role", roles.iloc[idx] if idx < len(roles) else types.iloc[idx])).lower()
        priority_node = str(row.get("priority_node", "priority"))
        distance = int(row.get("influence_path_length", 999) or 999)
        grouped_by_priority.setdefault(priority_node, []).append(idx)
        family = _role_family(role_text)
        grouped_by_priority_role.setdefault(priority_node, {}).setdefault(family, []).append(idx)
        if aid in BINARY_ACTUATOR_IDS:
            _append_binary_toggle_sequence(priority_node, aid, idx)
            continue
        # Exact local replay data normally represent a relative setting change
        # held for a fixed number of control intervals and then restored. Keep
        # that temporal meaning explicit. These candidates are the only ones
        # eligible for the empirical point gate in the controller.
        if empirical_hold_steps > 0 and allowed_action_directions and aid in allowed_action_directions:
            permitted = {str(value).strip().lower() for value in allowed_action_directions.get(aid, ())}
            hold = min(empirical_hold_steps, horizon_steps)
            for direction, sign in (("decrease", -1.0), ("increase", 1.0)):
                if direction not in permitted:
                    continue
                profile = np.zeros(horizon_steps, dtype=float)
                local_limit = float(
                    (allowed_action_delta_limits or {})
                    .get(aid, {})
                    .get(direction, max_delta)
                )
                if not np.isfinite(local_limit) or local_limit <= 0.0:
                    continue
                # Use the empirically observed safe amplitude when available;
                # a smaller generic amplitude can have a different hydraulic
                # regime and is not interchangeable with the tested action.
                profile[:hold] = sign * local_limit
                _append_profile_sequence(
                    f"empirical_relative_hold|priority={priority_node}|actuator={aid}|d={sign * max_delta:.3f}|hold={hold}",
                    [idx],
                    profile,
                    "Relative single-actuator hold matching the exact local No-control replay semantics.",
                )
        if "inlet" in role_text:
            _append_single_asset_sequence(
                f"restrict_then_release|priority={priority_node}|actuator={aid}|d={-max_delta:.3f}",
                idx,
                -max_delta,
                0.0,
                "Temporarily restrict an influential storage inlet near the priority zone, then return to the current setting.",
            )
            _append_profile_sequence(
                f"storage_inlet_ramp_restrict_restore|priority={priority_node}|actuator={aid}|d={-max_delta:.3f}",
                [idx],
                _delta_profile("ramp", -max_delta),
                "Gradually restrict a storage inlet before restoring it late in the horizon.",
            )
            _append_profile_sequence(
                f"storage_inlet_pulse_restrict|priority={priority_node}|actuator={aid}|d={-max_delta:.3f}",
                [idx],
                _delta_profile("pulse", -max_delta),
                "Apply a short storage-inlet restriction pulse around the predicted risk window.",
            )
        if "outlet" in role_text or "storage" in role_text:
            _append_single_asset_sequence(
                f"retain_through_peak|priority={priority_node}|actuator={aid}|d={-max_delta:.3f}",
                idx,
                -max_delta,
                None,
                "Retain storage in the influence domain through the prediction horizon.",
            )
            _append_single_asset_sequence(
                f"release_if_safe|priority={priority_node}|actuator={aid}|d={max_delta:.3f}",
                idx,
                max_delta,
                0.0,
                "Release an influential storage outlet only if the horizon safety objective accepts the sequence.",
            )
            _append_profile_sequence(
                f"storage_outlet_retain_then_release_ramp|priority={priority_node}|actuator={aid}|d={max_delta:.3f}",
                [idx],
                _delta_profile("retain_then_release", max_delta),
                "Retain storage through the peak and release mildly during recession if the horizon objective accepts it.",
            )
        if "pump" in role_text:
            _append_single_asset_sequence(
                f"pump_throttle_if_peak_risk|priority={priority_node}|actuator={aid}|d={-0.5 * max_delta:.3f}",
                idx,
                -0.5 * max_delta,
                None,
                "Throttle an influential pump during high priority-zone risk.",
            )
            _append_single_asset_sequence(
                f"pump_boost_if_safe|priority={priority_node}|actuator={aid}|d={max_delta:.3f}",
                idx,
                max_delta,
                0.0,
                "Boost an influential pump only if the horizon safety objective accepts the sequence.",
            )
            _append_profile_sequence(
                f"pump_ramp_boost_if_safe|priority={priority_node}|actuator={aid}|d={max_delta:.3f}",
                [idx],
                _delta_profile("ramp", max_delta),
                "Ramp a pump upward only when the predicted TFV and peak constraints remain safe.",
            )
            _append_profile_sequence(
                f"pump_pulse_throttle_if_peak_risk|priority={priority_node}|actuator={aid}|d={-0.5 * max_delta:.3f}",
                [idx],
                _delta_profile("pulse", -0.5 * max_delta),
                "Use a short pump-throttling pulse during the highest predicted peak-risk part of the horizon.",
            )
        if "orifice" in role_text or "weir" in role_text or "regulator" in role_text:
            _append_single_asset_sequence(
                f"regulator_restrict_then_restore|priority={priority_node}|actuator={aid}|d={-max_delta:.3f}",
                idx,
                -max_delta,
                0.0,
                "Temporarily restrict an influential regulator and restore it within the prediction horizon.",
            )
            _append_single_asset_sequence(
                f"regulator_release_if_safe|priority={priority_node}|actuator={aid}|d={max_delta:.3f}",
                idx,
                max_delta,
                0.0,
                "Temporarily release an influential regulator only if PFV and system safety constraints pass.",
            )
            _append_profile_sequence(
                f"regulator_ramp_restrict_restore|priority={priority_node}|actuator={aid}|d={-max_delta:.3f}",
                [idx],
                _delta_profile("ramp", -max_delta),
                "Gradually restrict an influential ordinary orifice/weir and restore late in the horizon.",
            )
            _append_profile_sequence(
                f"regulator_pulse_release_if_safe|priority={priority_node}|actuator={aid}|d={max_delta:.3f}",
                [idx],
                _delta_profile("pulse", max_delta),
                "Apply a short regulator release pulse only when the predicted safety constraints pass.",
            )
        if distance <= 1 and "inlet" not in role_text:
            _append_single_asset_sequence(
                f"local_retain|priority={priority_node}|actuator={aid}|d={-0.5 * max_delta:.3f}",
                idx,
                -0.5 * max_delta,
                None,
                "Apply a small local retain action for a directly connected control asset.",
            )

    for priority_node, role_map in grouped_by_priority_role.items():
        for family, indices in role_map.items():
            unique_indices = []
            for idx in indices:
                if idx not in unique_indices:
                    unique_indices.append(idx)
            selected = unique_indices[: min(group_limit, len(unique_indices))]
            if len(selected) < 2:
                continue
            if family == "regulator":
                _append_profile_sequence(
                    f"priority_group_regulator_restrict_then_restore|priority={priority_node}|n={len(selected)}|d={-max_delta:.3f}",
                    selected,
                    _delta_profile("early_then_restore", -max_delta),
                    "Coordinated short-horizon restriction over ordinary orifices/weirs influencing the priority group.",
                )
                _append_profile_sequence(
                    f"priority_group_regulator_pulse_release_if_safe|priority={priority_node}|n={len(selected)}|d={max_delta:.3f}",
                    selected,
                    _delta_profile("pulse", max_delta),
                    "Coordinated release pulse over ordinary orifices/weirs, accepted only by the safety objective.",
                )
            elif family == "storage":
                _append_profile_sequence(
                    f"priority_group_storage_retain_then_release|priority={priority_node}|n={len(selected)}|d={max_delta:.3f}",
                    selected,
                    _delta_profile("retain_then_release", max_delta),
                    "Coordinated storage retain-then-release profile over storage inlet/outlet controls.",
                )
            elif family == "pump":
                _append_profile_sequence(
                    f"priority_group_pump_ramp_boost_if_safe|priority={priority_node}|n={len(selected)}|d={max_delta:.3f}",
                    selected,
                    _delta_profile("ramp", max_delta),
                    "Coordinated pump ramp profile, accepted only when horizon TFV and peak constraints pass.",
                )

    for priority_node, raw_indices in grouped_by_priority.items():
        unique_indices = []
        for idx in raw_indices:
            if idx not in unique_indices:
                unique_indices.append(idx)
        if len(unique_indices) < 2:
            continue
        selected = unique_indices[: min(group_limit, len(unique_indices))]
        for sign, name in [(-1.0, "priority_group_restrict"), (1.0, "priority_group_release")]:
            seq = reference.copy()
            split = max(1, horizon_steps // 2)
            seq[:split, selected] = np.clip(seq[:split, selected] + sign * max_delta, 0.0, 1.0)
            seq[split:, selected] = reference[split:, selected]
            target_ids = [ids[i] for i in selected]
            _append_sequence(
                f"{name}|priority={priority_node}|n={len(selected)}|d={sign * max_delta:.3f}",
                seq,
                "Coordinated small-amplitude action over the priority influence-domain assets.",
                target_ids,
            )

        # Direction-aware mixed-facility candidates approximate the otherwise
        # intractable 3^109 joint action space. Each facility keeps the sign
        # proven safe by its No-control replay ablation, while rolling windows
        # and group sizes provide diverse simultaneous interventions.
        if allowed_action_directions:
            ordered = [i for i in unique_indices if ids[i] in allowed_action_directions]
            for size in (2, 4, 8, group_limit):
                size = min(int(size), len(ordered), group_limit)
                if size < 2:
                    continue
                for offset in range(0, len(ordered), size):
                    selected_mixed = ordered[offset : offset + size]
                    if len(selected_mixed) < 2:
                        continue
                    seq = reference.copy()
                    split = max(1, horizon_steps // 2)
                    for idx in selected_mixed:
                        aid = ids[idx]
                        permitted = {str(x).lower() for x in allowed_action_directions.get(aid, ())}
                        if permitted == {"increase"}:
                            delta = max_delta
                        elif permitted == {"decrease"}:
                            delta = -max_delta
                        elif "increase" in permitted and "decrease" in permitted:
                            role = _role_family(str(types.iloc[idx]))
                            delta = max_delta if role == "pump" else -max_delta
                        else:
                            continue
                        seq[:split, idx] = np.clip(reference[:split, idx] + delta, 0.0, 1.0)
                    target_ids = [ids[i] for i in selected_mixed]
                    _append_sequence(
                        f"direction_safe_joint_repair|priority={priority_node}|n={len(selected_mixed)}|offset={offset}",
                        seq,
                        "Joint rolling-horizon repair using only actuator directions supported by exact No-control replay ablations.",
                        target_ids,
                    )

    if influence_rows:
        return _dedupe_and_cap_sequences(sequences, int(max_sequences))

    for sign, name in [(-1.0, "retain_or_throttle"), (1.0, "release_or_boost")]:
        seq = reference.copy()
        mask = types.str.contains("pump|storage|orifice|weir", regex=True, na=False).to_numpy()
        mask = np.asarray([bool(m) and _asset_allowed(aid) for aid, m in zip(ids, mask)], dtype=bool)
        seq[:, mask] = np.clip(seq[:, mask] + sign * float(max_delta), 0.0, 1.0)
        _append_sequence(
            f"{name}_all_control_assets_d={sign * float(max_delta):.3f}",
            seq,
            "Uniform bounded exploratory sequence over controllable pumps/storage regulators.",
            [aid for aid, m in zip(ids, mask) if m],
        )
    return _dedupe_and_cap_sequences(sequences, int(max_sequences))


def _dedupe_and_cap_sequences(sequences: list[dict], max_sequences: int = 0) -> list[dict]:
    unique: list[dict] = []
    seen: set[tuple[tuple[int, ...], bytes]] = set()
    for item in sequences:
        sequence = np.asarray(item.get("sequence", []), dtype=np.float32)
        rounded = np.round(sequence, decimals=6).astype(np.float32, copy=False)
        key = (tuple(rounded.shape), rounded.tobytes())
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    if int(max_sequences) <= 0 or len(unique) <= int(max_sequences):
        return unique

    def _priority(label: str) -> tuple[int, str]:
        text = str(label)
        if text == "hold_native":
            return (0, text)
        if "priority_group_regulator" in text or "priority_group_storage" in text or "priority_group_pump" in text:
            return (1, text)
        if (
            "restrict_then_release" in text
            or "retain_through_peak" in text
            or "regulator_restrict_then_restore" in text
            or "regulator_release_if_safe" in text
        ):
            return (2, text)
        if "ramp" in text:
            return (3, text)
        if "pulse" in text:
            return (4, text)
        return (5, text)

    return sorted(unique, key=lambda item: _priority(str(item.get("label", ""))))[: int(max_sequences)]
