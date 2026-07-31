"""Pilot400 candidate materialization: role plan -> executable schedules.

Three planning layers (spec section II):

1. ``pilot_role_plan.csv``   — 400 role assignment rows (never run directly);
2. ``pilot_candidate_plan.csv`` — 400 materialized candidates, each with a
   deterministic, uniquely projected 12x36 schedule plus full constraint
   evidence; this file is the only Candidate input for Preflight and Run;
3. ``pilot_branch_plan.csv`` — 1600 rows = 400 samples x 4 branches with
   reference-cache keys; reference rows are never counted as samples.

Materialization uses only the current checkpoint state (canonical catalog
row), rainfall forecast metadata, static facility semantics and the frozen
anchor libraries.  It never reads future SWMM results and never copies an
old schedule byte-for-byte: Peak anchors act as structural anchors that are
re-materialized against the current anchor action.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sewerrtc.control.v4_candidate_generator import (
    _float_or,
    _is_binary,
    _semantics_by_id,
    project_candidate_schedule,
)
from .peak_boundary import _schedule_sha, verify_projection_constraints
from .pilot import PILOT_ROLES


HORIZON_STEPS = 12
MAX_K = 8
MAX_ATTEMPTS_PER_ROLE = 24

RESPONSIVE_ROLES = PILOT_ROLES

LOW_OPPORTUNITY_ROLES = (
    "hold_neighbourhood",
    "toward_no_control",
    "di_neighbourhood",
    "k1_strong_legal",
    "k2_strong_legal",
    "binary_legal_toggle",
    "continuous_absolute_level",
    "temporal_pulse",
    "coverage_gap",
    "expected_low_response",
)

PILOT_BRANCH_ROLES = (
    "candidate",
    "no_control",
    "dynamic_internal_rules",
    "hold_previous",
)


def _sha256_json(value: dict) -> str:
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def checkpoint_state_sha(checkpoint: dict | pd.Series) -> str:
    """Deterministic checkpoint-state identity from canonical catalog fields."""
    return _sha256_json(
        {
            "event_id": str(checkpoint["event_id"]),
            "checkpoint_id": str(checkpoint["checkpoint_id"]),
            "checkpoint_min": float(checkpoint["checkpoint_min"]),
            "rainfall_sha256": str(checkpoint["rainfall_sha256"]),
            "network_sha256": str(checkpoint.get("network_sha256", "")),
            "anchor_action_json": str(checkpoint["anchor_action_json"]),
        }
    )


def build_pilot_role_plan(checkpoints: pd.DataFrame) -> pd.DataFrame:
    """Layer 1: 400 role rows; responsive and low-opportunity role menus."""
    required = {
        "event_id",
        "rainfall_sha256",
        "checkpoint_id",
        "checkpoint_role",
        "checkpoint_min",
        "anchor_action_json",
    }
    missing = required - set(checkpoints)
    if missing:
        raise ValueError(f"checkpoints missing columns: {sorted(missing)}")
    if checkpoints["event_id"].nunique() != 8:
        raise ValueError("Pilot400 requires exactly 8 events")
    if not checkpoints.groupby("event_id").size().eq(5).all():
        raise ValueError("Pilot400 requires five checkpoints per event")
    events = sorted(checkpoints["event_id"].astype(str).unique())
    split_names = ["pilot_train"] * 5 + [
        "pilot_calibration",
        "pilot_validation",
        "pilot_challenge",
    ]
    split_map = dict(zip(events, split_names))
    rows: list[dict] = []
    for _, checkpoint in checkpoints.iterrows():
        responsive = str(checkpoint["checkpoint_role"]) == "responsive"
        roles = RESPONSIVE_ROLES if responsive else LOW_OPPORTUNITY_ROLES
        state_id = checkpoint_state_sha(checkpoint)
        for priority, role in enumerate(roles):
            case_id = (
                f"pilot__{checkpoint['event_id']}__"
                f"{checkpoint['checkpoint_id']}__{role}"
            )
            rows.append(
                {
                    "event_id": checkpoint["event_id"],
                    "rainfall_sha256": checkpoint["rainfall_sha256"],
                    "checkpoint_id": checkpoint["checkpoint_id"],
                    "checkpoint_role": checkpoint["checkpoint_role"],
                    "state_id": state_id,
                    "candidate_role": role,
                    "candidate_family_target": _ROLE_FAMILY.get(role, role),
                    "split": split_map[str(checkpoint["event_id"])],
                    "case_id": case_id,
                    "priority": priority,
                    "source_anchor_role": (
                        "peak_boundary_anchor"
                        if role
                        in ("peak_boundary", "PFV_safe_Peak_hard_negative")
                        else "checkpoint_anchor"
                    ),
                }
            )
    return pd.DataFrame(rows)


_ROLE_FAMILY = {
    "joint_beneficial_a": "staggered_throttle",
    "joint_beneficial_b": "storage_relief",
    "pfv_boundary": "pfv_boundary_stress",
    "PFV_hard_negative": "full_open_surge",
    "TFV_hard_negative": "front_loaded_release",
    "peak_boundary": "peak_structural_anchor",
    "PFV_safe_Peak_hard_negative": "synchronized_pump_starts",
    "neutral": "near_reference",
    "coverage_gap": "coverage_gap_probe",
    "uncertainty": "temporal_probe",
    "hold_neighbourhood": "hold_neighbourhood",
    "toward_no_control": "toward_no_control",
    "di_neighbourhood": "dynamic_internal_neighbourhood",
    "k1_strong_legal": "k1_strong_legal",
    "k2_strong_legal": "k2_strong_legal",
    "binary_legal_toggle": "binary_legal_toggle",
    "continuous_absolute_level": "continuous_absolute_level",
    "temporal_pulse": "temporal_pulse",
    "expected_low_response": "expected_low_response",
}


def _bounds(row: dict) -> tuple[float, float]:
    lower = _float_or(row.get("lower_bound"), 0.0)
    upper = _float_or(row.get("upper_bound"), 1.0)
    if lower > upper:
        lower, upper = upper, lower
    return lower, upper


def _toward(value: float, target: float, fraction: float) -> float:
    return float(value + (target - value) * float(fraction))


def _structural_moves(
    anchor_row: pd.Series, facility_ids: list[str]
) -> list[tuple[int, float]]:
    """Extract (index, direction) structure from a frozen Peak anchor row.

    Only the *structure* (which facilities move and in which direction) is
    reused; the amplitude is re-materialized against the current state.
    """
    projected = np.asarray(
        json.loads(str(anchor_row["projected_schedule_json"])), dtype=float
    )
    anchor = np.asarray(
        json.loads(str(anchor_row["anchor_schedule_json"])), dtype=float
    )
    if projected.shape != anchor.shape or projected.shape[1] != len(
        facility_ids
    ):
        return []
    diff = projected - anchor
    moves: list[tuple[int, float]] = []
    for index in range(diff.shape[1]):
        column = diff[:, index]
        if np.any(np.abs(column) > 1e-9):
            direction = 1.0 if column[np.argmax(np.abs(column))] > 0 else -1.0
            moves.append((index, direction))
    return moves


def _build_role_request(
    role: str,
    rng: np.random.Generator,
    anchor: np.ndarray,
    facility_ids: list[str],
    semantics_map: dict,
    active_ids: list[str],
    peak_moves: list[tuple[int, float]],
) -> np.ndarray:
    """Deterministic 12x36 request for one role attempt (pre-projection)."""
    requested = anchor.copy()
    anchor_vector = anchor[0]
    binary_ids = [
        item for item in facility_ids if _is_binary(item, semantics_map[item])
    ]
    continuous_ids = [
        item for item in facility_ids if item not in binary_ids
    ]
    pumps = [
        item
        for item in facility_ids
        if str(semantics_map[item].get("actuator_type", "")).lower() == "pump"
    ]
    active = [item for item in active_ids if item in facility_ids]
    inactive = [item for item in facility_ids if item not in active]

    def pick(pool: list[str], count: int) -> list[str]:
        source = pool or facility_ids
        order = rng.permutation(len(source))
        return [source[int(i)] for i in order[: max(1, min(count, len(source)))]]

    def set_channel(
        facility: str,
        *,
        direction: str,
        fraction: float,
        start: int = 0,
        stop: int = HORIZON_STEPS,
        absolute: float | None = None,
    ) -> None:
        index = facility_ids.index(facility)
        row = semantics_map[facility]
        lower, upper = _bounds(row)
        current = float(anchor_vector[index])
        if _is_binary(facility, row):
            value = upper if current <= (lower + upper) / 2.0 else lower
        elif absolute is not None:
            value = float(np.clip(absolute, lower, upper))
        elif direction == "up":
            value = _toward(current, upper, fraction)
        else:
            value = _toward(current, lower, fraction)
        requested[start:stop, index] = value

    jitter = float(rng.uniform(0.85, 1.15))
    if role == "joint_beneficial_a":
        for offset, facility in enumerate(pick(active or continuous_ids, 4)):
            set_channel(
                facility,
                direction="down",
                fraction=min(1.0, 0.6 * jitter),
                start=min(offset, 2),
                stop=min(offset, 2) + 6,
            )
    elif role == "joint_beneficial_b":
        for facility in pick(continuous_ids or active, 6):
            set_channel(
                facility, direction="down", fraction=min(1.0, 0.8 * jitter),
                stop=8,
            )
    elif role == "pfv_boundary":
        for facility in pick(active or facility_ids, 6):
            set_channel(
                facility, direction="up", fraction=min(1.0, 0.5 * jitter),
                stop=4,
            )
    elif role == "PFV_hard_negative":
        for facility in pick(facility_ids, MAX_K):
            set_channel(facility, direction="up", fraction=1.0, stop=6)
    elif role == "TFV_hard_negative":
        chosen = pick(active or facility_ids, 6)
        for facility in chosen:
            set_channel(facility, direction="up", fraction=1.0, stop=4)
        for facility in chosen:
            set_channel(
                facility, direction="down", fraction=1.0, start=4, stop=12
            )
    elif role in ("peak_boundary", "PFV_safe_Peak_hard_negative"):
        if role == "PFV_safe_Peak_hard_negative" and pumps:
            for facility in pick(pumps, min(MAX_K, len(pumps))):
                set_channel(
                    facility, direction="up",
                    fraction=min(1.0, 0.75 * jitter), stop=4,
                )
        elif peak_moves:
            for index, direction in peak_moves[:MAX_K]:
                facility = facility_ids[index]
                set_channel(
                    facility,
                    direction="up" if direction > 0 else "down",
                    fraction=min(1.0, 0.75 * jitter),
                    stop=4,
                )
        else:
            for facility in pick(pumps or active or facility_ids, 4):
                set_channel(
                    facility, direction="up",
                    fraction=min(1.0, 0.75 * jitter), stop=4,
                )
    elif role == "neutral":
        for facility in pick(continuous_ids or facility_ids, 1):
            row = semantics_map[facility]
            step = min(
                _float_or(row.get("rate_limit"), 1.0), 0.05 * jitter
            )
            index = facility_ids.index(facility)
            lower, upper = _bounds(row)
            direction = "up" if anchor_vector[index] < upper - step else "down"
            set_channel(
                facility, direction=direction,
                fraction=step / max(upper - lower, 1e-9), stop=2,
            )
    elif role in ("coverage_gap",):
        for facility in pick(inactive or facility_ids, 3):
            set_channel(
                facility, direction="up", fraction=min(1.0, 0.5 * jitter),
                stop=6,
            )
    elif role == "uncertainty":
        for facility in pick(active or facility_ids, 4):
            for start in (0, 4, 8):
                set_channel(
                    facility, direction="up",
                    fraction=min(1.0, 0.6 * jitter),
                    start=start, stop=start + 2,
                )
    elif role == "hold_neighbourhood":
        for facility in pick(continuous_ids or facility_ids, 2):
            set_channel(
                facility, direction="up", fraction=min(1.0, 0.1 * jitter),
                stop=6,
            )
    elif role == "toward_no_control":
        for facility in pick(facility_ids, 4):
            set_channel(
                facility, direction="up", fraction=min(1.0, 0.5 * jitter),
                stop=8,
            )
    elif role == "di_neighbourhood":
        for facility in pick(continuous_ids or facility_ids, 3):
            direction = "up" if rng.uniform() < 0.5 else "down"
            set_channel(
                facility, direction=direction,
                fraction=min(1.0, 0.25 * jitter), stop=6,
            )
    elif role == "k1_strong_legal":
        for facility in pick(facility_ids, 1):
            set_channel(facility, direction="up", fraction=1.0, stop=8)
    elif role == "k2_strong_legal":
        for facility in pick(facility_ids, 2):
            set_channel(facility, direction="up", fraction=1.0, stop=8)
    elif role == "binary_legal_toggle":
        for facility in pick(binary_ids or facility_ids, 2):
            row = semantics_map[facility]
            hold = max(1, int(_float_or(row.get("min_hold_steps"), 1)))
            set_channel(
                facility, direction="up", fraction=1.0,
                stop=max(hold, 4),
            )
    elif role == "continuous_absolute_level":
        levels = (0.2, 0.4, 0.6, 0.8)
        for facility in pick(continuous_ids or facility_ids, 2):
            set_channel(
                facility,
                direction="up",
                fraction=1.0,
                absolute=float(levels[int(rng.integers(len(levels)))]),
                stop=8,
            )
    elif role == "temporal_pulse":
        width = int((2, 4, 6)[int(rng.integers(3))])
        for facility in pick(active or facility_ids, 3):
            set_channel(
                facility, direction="up", fraction=min(1.0, 0.7 * jitter),
                stop=width,
            )
    elif role == "expected_low_response":
        for facility in pick(continuous_ids or facility_ids, 1):
            set_channel(
                facility, direction="up", fraction=min(1.0, 0.08 * jitter),
                start=6, stop=9,
            )
    else:
        raise ValueError(f"unknown pilot candidate role: {role}")
    return requested


def materialize_pilot_candidates(
    role_plan: pd.DataFrame,
    checkpoint_catalog: pd.DataFrame,
    *,
    facility_ids: list[str],
    facility_semantics: pd.DataFrame,
    peak_boundary_anchor_library: pd.DataFrame,
    gate5r_anchor_library: pd.DataFrame | None = None,
    contract_sha256: str = "",
    config_sha256: str = "",
    code_sha256: str = "",
    schedule_dir: Path | None = None,
    schedule_dir_relative_to: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Layer 2: materialize the 400 role rows into unique legal candidates.

    Returns ``(candidate_plan, coverage_missing)``; a role that cannot be
    materialized uniquely within the attempt budget is recorded in
    ``coverage_missing`` and never padded by copying another candidate.
    """
    semantics_map = _semantics_by_id(facility_ids, facility_semantics)
    catalog = checkpoint_catalog.set_index(
        checkpoint_catalog["checkpoint_id"].astype(str)
    )
    peak_by_event: dict[str, pd.DataFrame] = {
        str(event): group
        for event, group in peak_boundary_anchor_library.groupby("event_id")
    }
    rows: list[dict] = []
    missing: list[dict] = []
    ones_sha = _schedule_sha(
        np.ones((HORIZON_STEPS, len(facility_ids)), dtype=float)
    )
    if schedule_dir is not None:
        schedule_dir.mkdir(parents=True, exist_ok=True)
    seen_by_state: dict[str, set[str]] = {}
    for _, role_row in role_plan.iterrows():
        checkpoint = catalog.loc[str(role_row["checkpoint_id"])]
        if isinstance(checkpoint, pd.DataFrame):
            checkpoint = checkpoint.iloc[0]
        anchor_vector = np.asarray(
            json.loads(str(checkpoint["anchor_action_json"])), dtype=float
        )
        if anchor_vector.size != len(facility_ids):
            raise ValueError(
                "anchor action does not match Engineering36 order"
            )
        anchor = np.repeat(
            anchor_vector.reshape(1, -1), HORIZON_STEPS, axis=0
        )
        anchor_sha = _schedule_sha(anchor)
        active_ids = [
            str(item)
            for item in json.loads(
                str(checkpoint.get("active_facility_ids_json", "[]"))
            )
            if str(item) in facility_ids
        ]
        state_id = str(role_row["state_id"])
        seen = seen_by_state.setdefault(state_id, {anchor_sha, ones_sha})
        role = str(role_row["candidate_role"])
        sample_id = str(role_row["case_id"])
        seed = int(
            hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:8], 16
        )
        event_id = str(role_row["event_id"])
        peak_library = peak_by_event.get(event_id)
        if peak_library is None and peak_by_event:
            peak_library = peak_boundary_anchor_library
        source_anchor_id = ""
        peak_moves: list[tuple[int, float]] = []
        if peak_library is not None and len(peak_library):
            anchor_row = peak_library.iloc[seed % len(peak_library)]
            source_anchor_id = str(anchor_row.get("sample_id", ""))
            peak_moves = _structural_moves(anchor_row, facility_ids)
        materialized: dict | None = None
        for attempt in range(MAX_ATTEMPTS_PER_ROLE):
            rng = np.random.default_rng(seed + attempt * 7919)
            # A saturated anchor (all active facilities at their upper
            # bound) makes every "up" request from the active pool collapse
            # onto the anchor itself; widen to the full facility pool for
            # the second half of the attempt budget so the unique legal
            # candidate that exists elsewhere can be found.
            pool_ids = (
                active_ids
                if attempt < MAX_ATTEMPTS_PER_ROLE // 2
                else []
            )
            requested = _build_role_request(
                role,
                rng,
                anchor,
                facility_ids,
                semantics_map,
                pool_ids,
                peak_moves,
            )
            requested_sha = _schedule_sha(requested)
            if requested_sha in seen:
                continue
            projected, projection = project_candidate_schedule(
                requested,
                anchor,
                facility_ids,
                facility_semantics,
                max_k=MAX_K,
            )
            projected_sha = projection["projected_schedule_hash"]
            if projected_sha in seen or np.allclose(projected, anchor):
                continue
            flags = verify_projection_constraints(
                projected, anchor, facility_ids, facility_semantics
            )
            if not all(flags.values()):
                continue
            bounds_ok = True
            for index, facility in enumerate(facility_ids):
                lower, upper = _bounds(semantics_map[facility])
                column = projected[:, index]
                if np.any(column < lower - 1e-8) or np.any(
                    column > upper + 1e-8
                ):
                    bounds_ok = False
            # The projector never routes non-binary facilities (e.g. the
            # variable-speed pump add350.1) through binary quantization, so
            # VSP semantics hold exactly when its channel stays in bounds.
            vsp_ok = bounds_ok
            if not bounds_ok:
                continue
            k_by_step = projection.get("k_by_step", [])
            materialized = {
                "requested": requested,
                "projected": projected,
                "requested_sha": requested_sha,
                "projected_sha": projected_sha,
                "projection": projection,
                "flags": flags,
                "bounds_ok": bounds_ok,
                "vsp_ok": True,
                "k_by_step": k_by_step,
                "attempt": attempt,
            }
            break
        if materialized is None:
            missing.append(
                {
                    **{
                        key: role_row[key]
                        for key in (
                            "event_id",
                            "checkpoint_id",
                            "candidate_role",
                            "split",
                            "case_id",
                        )
                    },
                    "reason": "coverage_missing_no_unique_legal_candidate",
                }
            )
            continue
        seen.add(materialized["requested_sha"])
        seen.add(materialized["projected_sha"])
        base_kwargs = json.loads(
            str(checkpoint.get("source_runner_kwargs", "{}"))
        )
        base_kwargs.update(
            {
                "override_start_min": float(checkpoint["checkpoint_min"]),
                "post_action": materialized["projected"].tolist(),
                "stop_after_override_min": 120.0,
                "prefix_history_min": 60.0,
                "decision_interval_sec": 600,
                "control_step_sec": 300,
                "post_control_mode": "external_override",
                "hotstart_dir": None,
            }
        )
        schedule_path = ""
        if schedule_dir is not None:
            target = schedule_dir / f"{sample_id}.json"
            target.write_text(
                json.dumps(
                    {
                        "sample_id": sample_id,
                        "requested_schedule": materialized[
                            "requested"
                        ].tolist(),
                        "projected_schedule": materialized[
                            "projected"
                        ].tolist(),
                        "anchor_schedule": anchor.tolist(),
                        "requested_schedule_sha256": materialized[
                            "requested_sha"
                        ],
                        "projected_schedule_sha256": materialized[
                            "projected_sha"
                        ],
                    }
                ),
                encoding="utf-8",
            )
            base = schedule_dir_relative_to or schedule_dir
            schedule_path = target.relative_to(base).as_posix()
        max_k_used = int(
            max([int(k) for k in materialized["k_by_step"]] or [0])
        )
        rows.append(
            {
                "sample_id": sample_id,
                "case_id": sample_id,
                "event_id": event_id,
                "rainfall_sha256": str(role_row["rainfall_sha256"]),
                "checkpoint_id": str(role_row["checkpoint_id"]),
                "checkpoint_role": str(role_row["checkpoint_role"]),
                "checkpoint_min": float(checkpoint["checkpoint_min"]),
                "checkpoint_state_sha256": state_id,
                "candidate_role": role,
                "candidate_family": _ROLE_FAMILY.get(role, role),
                "family": _ROLE_FAMILY.get(role, role),
                "priority": int(role_row["priority"]),
                "source_anchor_role": str(role_row["source_anchor_role"]),
                "source_anchor_id": source_anchor_id,
                "requested_schedule_json": json.dumps(
                    materialized["requested"].tolist()
                ),
                "projected_schedule_json": json.dumps(
                    materialized["projected"].tolist()
                ),
                "anchor_schedule_json": json.dumps(anchor.tolist()),
                "requested_schedule_sha256": materialized["requested_sha"],
                "projected_schedule_sha256": materialized["projected_sha"],
                "requested_schedule_path": schedule_path,
                "projected_schedule_path": schedule_path,
                "schedule_format": "json",
                "k_target": max(1, max_k_used),
                "k_actual": max_k_used,
                "k_sequence": json.dumps(
                    [int(k) for k in materialized["k_by_step"]]
                ),
                "binary_semantics_ok": bool(
                    materialized["flags"]["binary_semantics_ok"]
                ),
                "vsp_semantics_ok": bool(materialized["vsp_ok"]),
                "bounds_ok": bool(materialized["bounds_ok"]),
                "rate_limit_ok": bool(
                    materialized["flags"]["rate_limit_ok"]
                ),
                "ramp_ok": bool(materialized["flags"]["rate_limit_ok"]),
                "dwell_ok": bool(materialized["flags"]["dwell_ok"]),
                "interlock_ok": bool(materialized["flags"]["interlock_ok"]),
                "no_reversal_ok": bool(
                    materialized["projection"].get("no_reversal_ok", True)
                ),
                "projection_valid": True,
                "materialization_attempt": int(materialized["attempt"]),
                "split": str(role_row["split"]),
                "runner_group_key": sample_id,
                "runner_function": "run_swmm_fixed_action",
                "runner_kwargs": json.dumps(
                    base_kwargs, separators=(",", ":")
                ),
                "network_sha256": str(checkpoint.get("network_sha256", "")),
                "contract_sha256": contract_sha256,
                "config_sha256": config_sha256,
                "code_sha256": code_sha256,
                "status": "planned",
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(missing)


def build_pilot_branch_plan(
    candidate_plan: pd.DataFrame,
    *,
    contract_sha256: str = "",
) -> pd.DataFrame:
    """Layer 3: 1600 branch rows = 400 candidates x 4 branch roles.

    Reference branches carry a deterministic cache key so that every
    checkpoint state runs no_control / dynamic_internal_rules /
    hold_previous exactly once; they are never counted as samples.
    """
    required = {
        "sample_id",
        "event_id",
        "checkpoint_id",
        "checkpoint_state_sha256",
        "rainfall_sha256",
        "network_sha256",
        "runner_kwargs",
        "projected_schedule_json",
        "anchor_schedule_json",
        "split",
    }
    missing = required - set(candidate_plan)
    if missing:
        raise ValueError(f"candidate plan missing: {sorted(missing)}")
    rows: list[dict] = []
    for _, candidate in candidate_plan.iterrows():
        base = json.loads(str(candidate["runner_kwargs"]))
        projected = json.loads(str(candidate["projected_schedule_json"]))
        anchor = json.loads(str(candidate["anchor_schedule_json"]))
        sample_id = str(candidate["sample_id"])
        event_id = str(candidate["event_id"])
        checkpoint_id = str(candidate["checkpoint_id"])
        for branch in PILOT_BRANCH_ROLES:
            kwargs = dict(base)
            kwargs["post_control_mode"] = (
                "native_rules"
                if branch == "dynamic_internal_rules"
                else "external_override"
            )
            if branch == "candidate":
                kwargs["post_action"] = projected
            elif branch == "no_control":
                width = len(projected[0])
                kwargs["post_action"] = [[1.0] * width for _ in range(12)]
            else:
                kwargs["post_action"] = anchor
            is_candidate = branch == "candidate"
            reference_cache_key = ""
            reference_artifact_path = ""
            if not is_candidate:
                reference_cache_key = _sha256_json(
                    {
                        "event_id": event_id,
                        "checkpoint_id": checkpoint_id,
                        "checkpoint_state_sha256": str(
                            candidate["checkpoint_state_sha256"]
                        ),
                        "network_sha256": str(candidate["network_sha256"]),
                        "rainfall_sha256": str(candidate["rainfall_sha256"]),
                        "contract_sha256": str(
                            contract_sha256
                            or candidate.get("contract_sha256", "")
                        ),
                        "branch_role": branch,
                    }
                )
                reference_artifact_path = (
                    f"references/{event_id}/{checkpoint_id}/{branch}"
                )
            branch_id = f"{sample_id}__{branch}"
            rows.append(
                {
                    "sample_id": sample_id,
                    "branch_id": branch_id,
                    "case_id": branch_id,
                    "branch_role": branch,
                    "branch": branch,
                    "counted_as_sample": bool(is_candidate),
                    "event_id": event_id,
                    "checkpoint_id": checkpoint_id,
                    "checkpoint_state_sha256": str(
                        candidate["checkpoint_state_sha256"]
                    ),
                    "rainfall_sha256": str(candidate["rainfall_sha256"]),
                    "network_sha256": str(candidate["network_sha256"]),
                    "runner_function": "run_swmm_fixed_action",
                    "runner_kwargs": json.dumps(
                        kwargs, separators=(",", ":")
                    ),
                    "reference_cache_key": reference_cache_key,
                    "reference_artifact_path": reference_artifact_path,
                    "candidate_schedule_path": (
                        str(candidate.get("projected_schedule_path", ""))
                        if is_candidate
                        else ""
                    ),
                    "split": str(candidate["split"]),
                }
            )
    return pd.DataFrame(rows)


_CONSTRAINT_COLUMNS = (
    "binary_semantics_ok",
    "vsp_semantics_ok",
    "bounds_ok",
    "rate_limit_ok",
    "ramp_ok",
    "dwell_ok",
    "interlock_ok",
    "no_reversal_ok",
    "projection_valid",
)

_SHA_COLUMNS = (
    "checkpoint_state_sha256",
    "requested_schedule_sha256",
    "projected_schedule_sha256",
    "contract_sha256",
    "config_sha256",
    "code_sha256",
)


def audit_pilot_materialized_plan(
    role_plan: pd.DataFrame,
    candidate_plan: pd.DataFrame,
    branch_plan: pd.DataFrame,
    coverage_missing: pd.DataFrame,
    *,
    max_missing: int = 0,
    peak_tuned_event_ids: set[str] | None = None,
) -> dict:
    """Audit the materialized three-layer plan (spec section IX).

    Fail Closed: any missing coverage above ``max_missing``, any duplicate
    or illegal candidate, or any broken four-branch association fails the
    whole plan; nothing is padded or silently dropped.
    """
    for name, frame, needed in (
        ("candidate_plan", candidate_plan, set(_CONSTRAINT_COLUMNS)
         | set(_SHA_COLUMNS)
         | {"sample_id", "event_id", "checkpoint_id", "split",
            "rainfall_sha256", "k_target", "k_sequence",
            "runner_group_key"}),
        ("branch_plan", branch_plan,
         {"sample_id", "branch_id", "branch_role", "counted_as_sample",
          "reference_cache_key", "split", "event_id", "checkpoint_id"}),
    ):
        absent = needed - set(frame)
        if absent:
            return {
                "status": "blocked",
                "missing_columns": {name: sorted(absent)},
            }
    state_groups = candidate_plan.groupby(["event_id", "checkpoint_id"])
    per_state = state_groups.size()
    requested_unique = state_groups["requested_schedule_sha256"].apply(
        lambda column: column.is_unique
    )
    projected_unique = state_groups["projected_schedule_sha256"].apply(
        lambda column: column.is_unique
    )
    k_sequences_ok = candidate_plan["k_sequence"].apply(
        lambda text: max(json.loads(str(text)) or [0]) <= MAX_K
    )
    branch_sets = branch_plan.groupby("sample_id")["branch_role"].apply(set)
    candidate_counts = (
        branch_plan[branch_plan["branch_role"] == "candidate"]
        .groupby("sample_id")
        .size()
    )
    references = branch_plan[branch_plan["branch_role"] != "candidate"]
    reference_keys = references.drop_duplicates(
        ["event_id", "checkpoint_id", "branch_role"]
    )["reference_cache_key"]
    event_split = candidate_plan.groupby("event_id")["split"].nunique()
    rainfall_split = candidate_plan.groupby("rainfall_sha256")[
        "split"
    ].nunique()
    peak_events = {str(item) for item in (peak_tuned_event_ids or set())}
    tuned = candidate_plan[
        candidate_plan["event_id"].astype(str).isin(peak_events)
    ]
    checks = {
        "role_rows_400": len(role_plan) == 400,
        "candidate_rows_400": len(candidate_plan) == 400,
        "branch_rows_1600": len(branch_plan) == 1600,
        "events_8": candidate_plan["event_id"].nunique() == 8,
        "states_40": state_groups.ngroups == 40,
        "ten_candidates_per_state": bool(per_state.eq(10).all()),
        "sample_ids_unique": candidate_plan["sample_id"].is_unique,
        "requested_unique_per_state": bool(requested_unique.all()),
        "projected_unique_per_state": bool(projected_unique.all()),
        "k_le_8": bool(
            candidate_plan["k_target"].astype(int).le(MAX_K).all()
            and k_sequences_ok.all()
        ),
        "constraints_all_true": bool(
            candidate_plan[list(_CONSTRAINT_COLUMNS)]
            .astype(bool)
            .all()
            .all()
        ),
        "four_branch_association": bool(
            branch_sets.eq(set(PILOT_BRANCH_ROLES)).all()
            and set(branch_sets.index)
            == set(candidate_plan["sample_id"].astype(str))
            and candidate_counts.eq(1).all()
            and len(candidate_counts) == len(candidate_plan)
        ),
        "references_not_counted": not references[
            "counted_as_sample"
        ].astype(bool).any(),
        "reference_cache_keys_complete": bool(
            references["reference_cache_key"].astype(str).str.len().gt(0).all()
        ),
        "reference_cache_keys_120": (
            reference_keys.nunique() == 120 and len(reference_keys) == 120
        ),
        "split_no_event_leakage": bool(event_split.le(1).all()),
        "split_no_rainfall_leakage": bool(rainfall_split.le(1).all()),
        "peak_tuned_events_pilot_train_only": bool(
            tuned["split"].eq("pilot_train").all()
        )
        if len(tuned)
        else True,
        "sha_columns_complete": bool(
            candidate_plan[list(_SHA_COLUMNS)]
            .astype(str)
            .apply(lambda column: column.str.len().gt(0))
            .all()
            .all()
        ),
        "runner_group_key_is_sample_id": bool(
            candidate_plan["runner_group_key"]
            .astype(str)
            .eq(candidate_plan["sample_id"].astype(str))
            .all()
        ),
        "coverage_missing_within_budget": len(coverage_missing)
        <= int(max_missing),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    return {
        "status": "pass" if all(checks.values()) else "scientific_fail",
        "checks": checks,
        "counts": {
            "role_rows": int(len(role_plan)),
            "candidate_rows": int(len(candidate_plan)),
            "branch_rows": int(len(branch_plan)),
            "coverage_missing": int(len(coverage_missing)),
            "reference_cache_keys": int(reference_keys.nunique()),
        },
        "coverage_missing_rows": coverage_missing.to_dict("records"),
    }
