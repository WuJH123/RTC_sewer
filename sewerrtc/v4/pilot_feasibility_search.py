"""Gate P3 Exact-SWMM feasibility search planning (spec section V).

Round A: up to 16 candidates from 12 frozen families for each of the 23
joint-missing responsive states, plus 4 replay/probe rows for each of the
9 positive-control states.  Round B: up to 16 more candidates only for
near-boundary states, total per-state budget 32.

All three references are reused from the frozen Pilot cache; a plan row
never schedules a new reference run.  Candidate generation reads only the
frozen v2 dataset, the v1 candidate plan (state identity + runner kwargs)
and the reference schedules -- never candidate-branch SWMM outcomes of the
rows being planned (Round B reads only the completed Round A map, which is
a development diagnostic by contract).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sewerrtc.control.v4_candidate_generator import (
    _is_binary,
    _semantics_by_id,
    project_candidate_schedule,
)
from .peak_boundary import _schedule_sha, verify_projection_constraints
from .pilot_candidates import HORIZON_STEPS, MAX_K, _bounds
from .pilot_extension import (
    _bool,
    _candidate_plan_row,
    _identity_matches_v1,
    _ranked_moved_facilities,
    _state_seen_shas,
    di_schedule_from_reference,
)

FEASIBILITY_FAMILIES = (
    "replay_compatible_oracle_seed",
    "toward_no_control",
    "toward_dynamic_internal",
    "toward_hold",
    "coordinate_top1_top2",
    "leave_one_out",
    "storage_headroom",
    "downstream_capacity",
    "temporal_2step_4step",
    "staggered_release",
    "beam_sparse_k2_k4",
    "boundary_interpolation",
)

POSITIVE_CONTROL_ROLES = (
    "positive_control_replay",
    "positive_control_neighborhood",
    "hard_negative_probe",
    "neutral_probe",
)

FEASIBILITY_PHASE = "feasibility_p3"
P3_CONTRACT_VERSION = "project6_v4_pilot_feasibility_gate_p3"
ROUND_A = "round_a"
ROUND_B = "round_b"

ROUND_A_MAX_PER_MISSING_STATE = 16
ROUND_B_MAX_PER_STATE = 16
TOTAL_BUDGET_PER_MISSING_STATE = 32
MAX_TOTAL_CANDIDATE_RUNS = 772

_DELTA_PFV = "delta_pfv_h120_vs_no_control"
_DELTA_TFV = "delta_tfv_h120_vs_dynamic_internal"
_DELTA_PEAK = "delta_peak_h120_vs_dynamic_internal"
_STATE_KEYS = ["event_id", "checkpoint_id"]


def load_seed_schedules(
    replay_plan: pd.DataFrame,
    *,
    event_id: str,
    checkpoint_min: float,
    facility_ids: list[str],
    anchor: np.ndarray,
    max_seeds: int = 3,
) -> list[np.ndarray]:
    """Legacy oracle seed schedules aligned to the H120 decision window.

    Seeds are search shapes only (``label_use_forbidden`` by contract):
    same-event constrained seeds first, then global constrained seeds.
    Facilities absent from a seed file keep the anchor value; rows are
    matched on episode minutes with hold-last fallback.  Unparseable
    seeds are skipped fail-safe.
    """
    if replay_plan is None or replay_plan.empty:
        return []
    frame = replay_plan.copy()
    frame["__same_event"] = (
        frame["event_id"].astype(str) == str(event_id)
    )
    constrained = frame[
        frame["constraint_mode"].astype(str) == "constrained"
    ]
    ordered = constrained.sort_values(
        ["__same_event", "seed_id"], ascending=[False, True]
    )
    seeds: list[np.ndarray] = []
    for _, row in ordered.iterrows():
        if len(seeds) >= int(max_seeds):
            break
        path = Path(str(row.get("schedule_path_resolved", "")))
        if not path.exists():
            continue
        try:
            table = pd.read_csv(path)
        except (OSError, ValueError):
            continue
        time_column = next(
            (c for c in table.columns if "simtime" in c.lower()), None
        )
        if time_column is None:
            continue
        minutes = (
            pd.to_numeric(table[time_column], errors="coerce") * 60.0
        ).round(1)
        schedule = anchor.copy()
        usable = False
        for index, facility in enumerate(facility_ids):
            if facility not in table.columns:
                continue
            values = pd.to_numeric(table[facility], errors="coerce")
            for step in range(HORIZON_STEPS):
                desired = float(checkpoint_min) + step * 10.0
                mask = minutes <= desired + 0.5
                if not mask.any():
                    continue
                value = values[mask].iloc[-1]
                if np.isfinite(value):
                    schedule[step, index] = float(value)
                    usable = True
        if usable:
            seeds.append(schedule)
    return seeds


def _interp(base: np.ndarray, target: np.ndarray, weight: float) -> np.ndarray:
    return base + float(weight) * (target - base)


def _feasibility_family_requests(
    family: str,
    ctx: dict,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Deterministic pre-projection request variants for one P3 family."""
    anchor: np.ndarray = ctx["anchor"]
    di: np.ndarray = ctx["di"]
    nc: np.ndarray = ctx["nc"]
    facility_ids: list[str] = ctx["facility_ids"]
    semantics_map: dict = ctx["semantics_map"]
    active_ids: list[str] = ctx["active_ids"]
    best_safe: np.ndarray | None = ctx.get("best_pfv_safe")
    best_tfv: np.ndarray | None = ctx.get("best_tfv_schedule")
    tfv_bases: list[np.ndarray] = ctx.get("tfv_degraded_bases", [])
    seeds: list[np.ndarray] = ctx.get("seed_schedules", [])
    base = best_safe if best_safe is not None else di
    requests: list[np.ndarray] = []
    pool = [
        facility_ids.index(item)
        for item in (active_ids or facility_ids)
        if item in facility_ids
    ] or list(range(len(facility_ids)))
    ranked = [
        index
        for index in _ranked_moved_facilities(anchor, di)
        if index in pool
    ] or pool
    if family == "replay_compatible_oracle_seed":
        requests.extend(seed.copy() for seed in seeds)
    elif family == "toward_no_control":
        for weight in (1.0, 0.5, 0.25):
            for steps in (HORIZON_STEPS, 4):
                requested = base.copy()
                requested[:steps, :] = _interp(
                    base[:steps, :], nc[:steps, :], weight
                )
                requests.append(requested)
    elif family == "toward_dynamic_internal":
        origin = best_safe if best_safe is not None else anchor
        for top_n in (1, 2, 4):
            for weight in (1.0, 0.5):
                chosen = _ranked_moved_facilities(origin, di)[:top_n]
                if not chosen:
                    continue
                requested = origin.copy()
                for index in chosen:
                    requested[:, index] = _interp(
                        origin[:, index], di[:, index], weight
                    )
                requests.append(requested)
    elif family == "toward_hold":
        for hold_steps in (2, 4):
            requested = base.copy()
            requested[:hold_steps, :] = anchor[:hold_steps, :]
            requests.append(requested)
        requests.append(_interp(base, anchor, 0.5))
    elif family == "coordinate_top1_top2":
        for chosen in ([ranked[0]] if ranked else [],
                       ranked[:2],
                       ranked[1:2]):
            if not chosen:
                continue
            requested = anchor.copy()
            for index in chosen:
                requested[:, index] = di[:, index]
            requests.append(requested)
    elif family == "leave_one_out":
        bases = tfv_bases or [di]
        for source in bases:
            for index in _ranked_moved_facilities(source, anchor)[:6]:
                requested = source.copy()
                requested[:, index] = anchor[:, index]
                requests.append(requested)
    elif family == "storage_headroom":
        inlets = ctx.get("storage_inlet_indices", [])
        outlets = ctx.get("storage_outlet_indices", [])
        if inlets or outlets:
            for fill_steps in (4, 6):
                requested = anchor.copy()
                for index in inlets:
                    _, upper = _bounds(
                        semantics_map[facility_ids[index]]
                    )
                    requested[:fill_steps, index] = upper
                for index in outlets:
                    lower, upper = _bounds(
                        semantics_map[facility_ids[index]]
                    )
                    requested[:fill_steps, index] = lower
                    requested[fill_steps:, index] = upper
                requests.append(requested)
            requested = anchor.copy()
            for index in inlets:
                _, upper = _bounds(semantics_map[facility_ids[index]])
                requested[:, index] = upper
            requests.append(requested)
    elif family == "downstream_capacity":
        regulators = ctx.get("downstream_regulator_indices", [])
        targets = regulators or ranked[:2]
        for fraction in (0.5, 0.25):
            for steps in (4, 8):
                requested = base.copy()
                for index in targets:
                    facility = facility_ids[index]
                    lower, upper = _bounds(semantics_map[facility])
                    if _is_binary(facility, semantics_map[facility]):
                        requested[:steps, index] = lower
                    else:
                        requested[:steps, index] = np.clip(
                            anchor[:steps, index] * fraction,
                            lower,
                            upper,
                        )
                requests.append(requested)
    elif family == "temporal_2step_4step":
        for steps in (2, 4):
            requested = anchor.copy()
            requested[:steps, :] = base[:steps, :]
            requests.append(requested)
    elif family == "staggered_release":
        movers = ranked[:3]
        if movers:
            for stride in (2, 4):
                requested = anchor.copy()
                for order, index in enumerate(movers):
                    start = min(order * stride, HORIZON_STEPS - 1)
                    requested[start:, index] = di[start:, index]
                requests.append(requested)
    elif family == "beam_sparse_k2_k4":
        combos = []
        if len(ranked) >= 2:
            combos.extend([ranked[:2], [ranked[0], ranked[-1]]])
        if len(ranked) >= 4:
            combos.append(ranked[:4])
        for chosen in combos:
            requested = anchor.copy()
            for index in chosen:
                facility = facility_ids[index]
                lower, upper = _bounds(semantics_map[facility])
                if _is_binary(facility, semantics_map[facility]):
                    midpoint = (lower + upper) / 2.0
                    current = requested[:, index]
                    requested[:, index] = np.where(
                        current >= midpoint, lower, upper
                    )
                else:
                    requested[:, index] = di[:, index]
            requests.append(requested)
    elif family == "boundary_interpolation":
        if best_safe is not None and best_tfv is not None:
            for weight in (0.25, 0.5, 0.75):
                requests.append(_interp(best_safe, best_tfv, weight))
        for weight in (0.33, 0.66):
            requests.append(_interp(anchor, di, weight))
    else:
        raise ValueError(f"unknown feasibility family: {family}")
    # Deterministic jitter fallback keeps second-pass slots distinct.
    if requests and rng.integers(2) == 1:
        requests = requests[::-1]
    return requests


def materialize_feasibility_candidates(
    state_ctx: dict,
    *,
    facility_semantics: pd.DataFrame,
    seen: set[str],
    per_state_max: int,
    round_tag: str,
) -> list[dict]:
    """Project family requests into unique legal candidates for one state.

    No minimum is enforced: a shortfall is recorded by the caller in the
    search-coverage table, never coerced.  ``seen`` must hold every v1/v2
    requested/projected/actual SHA for the state plus anchor, all-ones,
    DI and NC schedule SHAs.
    """
    facility_ids: list[str] = state_ctx["facility_ids"]
    anchor: np.ndarray = state_ctx["anchor"]
    seed = int(
        hashlib.sha256(
            f"{state_ctx['state_id']}::{round_tag}".encode("utf-8")
        ).hexdigest()[:8],
        16,
    )
    accepted: list[dict] = []
    for round_index in (0, 1):
        for family_index, family in enumerate(FEASIBILITY_FAMILIES):
            if len(accepted) >= int(per_state_max):
                break
            if round_index == 0 and any(
                item["family"] == family for item in accepted
            ):
                continue
            rng = np.random.default_rng(
                seed + family_index * 7919 + round_index * 104729
            )
            for requested in _feasibility_family_requests(
                family, state_ctx, rng
            ):
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
                    lower, upper = _bounds(
                        state_ctx["semantics_map"][facility]
                    )
                    column = projected[:, index]
                    if np.any(column < lower - 1e-8) or np.any(
                        column > upper + 1e-8
                    ):
                        bounds_ok = False
                if not bounds_ok:
                    continue
                seen.add(requested_sha)
                seen.add(projected_sha)
                accepted.append(
                    {
                        "family": family,
                        "requested": requested,
                        "projected": projected,
                        "requested_sha": requested_sha,
                        "projected_sha": projected_sha,
                        "projection": projection,
                        "flags": flags,
                    }
                )
                break
        if len(accepted) >= int(per_state_max):
            break
    return accepted


def _project_item(
    requested: np.ndarray,
    family: str,
    state_ctx: dict,
    facility_semantics: pd.DataFrame,
    seen: set[str],
    *,
    allow_known_actual: bool = False,
) -> dict | None:
    """Single-request variant of the materializer for probe/replay rows."""
    facility_ids: list[str] = state_ctx["facility_ids"]
    anchor: np.ndarray = state_ctx["anchor"]
    requested_sha = _schedule_sha(requested)
    if not allow_known_actual and requested_sha in seen:
        return None
    projected, projection = project_candidate_schedule(
        requested, anchor, facility_ids, facility_semantics, max_k=MAX_K
    )
    projected_sha = projection["projected_schedule_hash"]
    if not allow_known_actual and projected_sha in seen:
        return None
    if np.allclose(projected, anchor):
        return None
    flags = verify_projection_constraints(
        projected, anchor, facility_ids, facility_semantics
    )
    if not all(flags.values()):
        return None
    seen.add(requested_sha)
    seen.add(projected_sha)
    return {
        "family": family,
        "requested": requested,
        "projected": projected,
        "requested_sha": requested_sha,
        "projected_sha": projected_sha,
        "projection": projection,
        "flags": flags,
    }


def positive_control_items(
    state_ctx: dict,
    state_samples: pd.DataFrame,
    *,
    facility_semantics: pd.DataFrame,
    seen: set[str],
) -> list[dict]:
    """The four Round A rows for one positive-control state.

    The replay row deliberately re-runs the best v2 joint candidate (its
    SHA is expected to match v2; replay success feeds the P3 replay gate).
    The other three probes must be actual-unique like any candidate.
    """
    facility_ids: list[str] = state_ctx["facility_ids"]
    anchor: np.ndarray = state_ctx["anchor"]
    di: np.ndarray = state_ctx["di"]
    semantics_map: dict = state_ctx["semantics_map"]
    joint = state_samples[
        _bool(state_samples.get("joint_noninferior"), state_samples.index)
    ]
    if joint.empty:
        raise ValueError(
            f"positive-control state {state_ctx['state_id']} has no v2 "
            "joint sample"
        )
    best = joint.loc[
        pd.to_numeric(joint[_DELTA_TFV], errors="coerce").idxmin()
    ]
    best_schedule = np.asarray(
        json.loads(str(best["projected_schedule_json"])), dtype=float
    )
    items: list[dict] = []
    replay = _project_item(
        best_schedule.copy(),
        "positive_control_replay",
        state_ctx,
        facility_semantics,
        seen,
        allow_known_actual=True,
    )
    if replay is None:
        raise ValueError(
            f"state {state_ctx['state_id']}: best joint candidate no "
            "longer projects legally under the frozen contract"
        )
    replay["expected_replay_of"] = str(best["sample_id"])
    replay["expected_actual_sha"] = str(
        best.get("actual_schedule_sha256", "")
    )
    items.append(replay)
    movers = _ranked_moved_facilities(anchor, best_schedule)
    # Neighborhood positive control: weaken the least-important mover.
    neighborhood = None
    for drop in reversed(movers):
        requested = best_schedule.copy()
        requested[:, drop] = anchor[:, drop]
        neighborhood = _project_item(
            requested,
            "positive_control_neighborhood",
            state_ctx,
            facility_semantics,
            seen,
        )
        if neighborhood is not None:
            break
    if neighborhood is None:
        requested = best_schedule.copy()
        if movers:
            index = movers[0]
            requested[1:, index] = best_schedule[:-1, index]
            requested[0, index] = anchor[0, index]
        neighborhood = _project_item(
            requested,
            "positive_control_neighborhood",
            state_ctx,
            facility_semantics,
            seen,
        )
    if neighborhood is not None:
        items.append(neighborhood)
    # Hard negative: strong full-horizon toggle away from DI on top movers.
    hard = None
    for top_n in (2, 1, 3):
        requested = anchor.copy()
        for index in movers[:top_n] or _ranked_moved_facilities(
            anchor, di
        )[:top_n]:
            facility = facility_ids[index]
            lower, upper = _bounds(semantics_map[facility])
            midpoint = (lower + upper) / 2.0
            requested[:, index] = np.where(
                di[:, index] >= midpoint, lower, upper
            )
        hard = _project_item(
            requested,
            "hard_negative_probe",
            state_ctx,
            facility_semantics,
            seen,
        )
        if hard is not None:
            break
    if hard is not None:
        items.append(hard)
    # Neutral: smallest legal single-facility, single-step deviation.
    neutral = None
    for index in movers or list(range(len(facility_ids))):
        facility = facility_ids[index]
        lower, upper = _bounds(semantics_map[facility])
        requested = anchor.copy()
        if _is_binary(facility, semantics_map[facility]):
            midpoint = (lower + upper) / 2.0
            requested[:1, index] = np.where(
                requested[:1, index] >= midpoint, lower, upper
            )
        else:
            span = upper - lower
            requested[:1, index] = np.clip(
                requested[:1, index] + 0.1 * span, lower, upper
            )
        neutral = _project_item(
            requested,
            "neutral_probe",
            state_ctx,
            facility_semantics,
            seen,
        )
        if neutral is not None:
            break
    if neutral is not None:
        items.append(neutral)
    return items


def _storage_indices(
    facility_ids: list[str], facility_semantics: pd.DataFrame
) -> dict[str, list[int]]:
    roles = {
        str(row["facility_id"]): str(row.get("storage_role", "none"))
        for _, row in facility_semantics.iterrows()
    }
    result = {"inlet": [], "outlet": [], "regulator": []}
    for index, facility in enumerate(facility_ids):
        role = roles.get(facility, "none")
        if role == "storage_inlet":
            result["inlet"].append(index)
        elif role == "storage_outlet":
            result["outlet"].append(index)
        elif role == "downstream_regulator":
            result["regulator"].append(index)
    return result


def build_state_context(
    catalog_row: pd.Series,
    base_row: pd.Series,
    state_samples: pd.DataFrame,
    *,
    reference_root: Path,
    facility_ids: list[str],
    facility_semantics: pd.DataFrame,
    replay_plan: pd.DataFrame | None,
) -> dict:
    """Frozen-input search context for one responsive state."""
    semantics_map = _semantics_by_id(facility_ids, facility_semantics)
    event_id = str(catalog_row["event_id"])
    checkpoint_id = str(catalog_row["checkpoint_id"])
    checkpoint_min = float(base_row["checkpoint_min"])
    anchor = np.asarray(
        json.loads(str(base_row["anchor_schedule_json"])), dtype=float
    )
    schedules: dict[str, np.ndarray] = {}
    for branch, key in (
        ("dynamic_internal_rules", "di"),
        ("no_control", "nc"),
    ):
        detail_path = (
            Path(reference_root)
            / "references"
            / event_id
            / checkpoint_id
            / branch
            / "detail.csv"
        )
        if not detail_path.exists():
            raise ValueError(f"reference detail missing: {detail_path}")
        schedules[key] = di_schedule_from_reference(
            pd.read_csv(detail_path),
            checkpoint_min=checkpoint_min,
            facility_ids=facility_ids,
            semantics_map=semantics_map,
            anchor=anchor,
        )
    pfv_safe = state_samples[
        _bool(state_samples.get("pfv_safe"), state_samples.index)
    ]
    best_pfv_safe = None
    if len(pfv_safe):
        best = pfv_safe.loc[
            pd.to_numeric(pfv_safe[_DELTA_PFV], errors="coerce").idxmin()
        ]
        best_pfv_safe = np.asarray(
            json.loads(str(best["projected_schedule_json"])), dtype=float
        )
    best_tfv_schedule = None
    if len(state_samples):
        best_tfv_row = state_samples.loc[
            pd.to_numeric(
                state_samples[_DELTA_TFV], errors="coerce"
            ).idxmin()
        ]
        best_tfv_schedule = np.asarray(
            json.loads(str(best_tfv_row["projected_schedule_json"])),
            dtype=float,
        )
    tfv_degraded = pfv_safe[
        ~_bool(pfv_safe.get("tfv_noninferior"), pfv_safe.index)
    ]
    tfv_bases = [
        np.asarray(
            json.loads(str(row["projected_schedule_json"])), dtype=float
        )
        for _, row in tfv_degraded.sort_values(_DELTA_TFV).iterrows()
    ][:3]
    try:
        active_ids = [
            str(item)
            for item in json.loads(
                str(catalog_row.get("active_facility_ids_json", "[]"))
            )
            if str(item) in facility_ids
        ]
    except (ValueError, TypeError):
        active_ids = []
    if not active_ids:
        moved = (
            np.abs(schedules["di"] - anchor).max(axis=0) > 1e-9
        )
        active_ids = [
            facility_ids[i] for i in range(len(facility_ids)) if moved[i]
        ]
    storage = _storage_indices(facility_ids, facility_semantics)
    seeds = load_seed_schedules(
        replay_plan if replay_plan is not None else pd.DataFrame(),
        event_id=event_id,
        checkpoint_min=checkpoint_min,
        facility_ids=facility_ids,
        anchor=anchor,
    )
    return {
        "state_id": str(base_row["checkpoint_state_sha256"]),
        "facility_ids": facility_ids,
        "semantics_map": semantics_map,
        "anchor": anchor,
        "di": schedules["di"],
        "nc": schedules["nc"],
        "active_ids": active_ids,
        "best_pfv_safe": best_pfv_safe,
        "best_tfv_schedule": best_tfv_schedule,
        "tfv_degraded_bases": tfv_bases,
        "seed_schedules": seeds,
        "storage_inlet_indices": storage["inlet"],
        "storage_outlet_indices": storage["outlet"],
        "downstream_regulator_indices": storage["regulator"],
    }


def _p3_plan_row(
    base_row: pd.Series,
    item: dict,
    *,
    sample_id: str,
    catalog_row: pd.Series,
    round_tag: str,
    search_role: str,
    schedule_dir: Path | None,
    schedule_dir_relative_to: Path | None,
    anchor: np.ndarray,
) -> dict:
    """v1-schema plan row plus the frozen P3 provenance columns."""
    row = _candidate_plan_row(
        base_row,
        item,
        sample_id=sample_id,
        source_phase=FEASIBILITY_PHASE,
        base_candidate_id=str(base_row["sample_id"]),
        schedule_dir=schedule_dir,
        schedule_dir_relative_to=schedule_dir_relative_to,
        anchor=anchor,
    )
    row["source_contract_version"] = P3_CONTRACT_VERSION
    row["search_round"] = round_tag
    row["search_role"] = search_role
    row["positive_control_state"] = bool(
        catalog_row["positive_control_state"]
    )
    row["joint_missing_state"] = bool(catalog_row["joint_missing_state"])
    row["oracle_revealed_flag_required"] = bool(
        catalog_row["oracle_revealed_flag_required"]
    )
    row["search_result_training_eligible"] = bool(
        catalog_row["search_result_training_eligible"]
    )
    row["expected_replay_of"] = str(item.get("expected_replay_of", ""))
    row["expected_actual_sha"] = str(item.get("expected_actual_sha", ""))
    return row


def plan_pilot_feasibility_map(
    catalog: pd.DataFrame,
    v1_candidate_plan: pd.DataFrame,
    v2_samples: pd.DataFrame,
    replay_plan: pd.DataFrame | None,
    *,
    reference_root: Path,
    facility_ids: list[str],
    facility_semantics: pd.DataFrame,
    schedule_dir: Path | None = None,
    schedule_dir_relative_to: Path | None = None,
    round_b_directives: pd.DataFrame | None = None,
) -> dict:
    """Deterministic Round A (+ optional Round B) feasibility search plan.

    Round A rows are a pure function of the frozen v1/v2 evidence, so
    re-planning with Round B directives appends rows without changing any
    Round A row (``--resume`` then skips completed samples).  A shortfall
    below the per-state budget is recorded in ``search_coverage``, never
    coerced.
    """
    ones_sha = _schedule_sha(
        np.ones((HORIZON_STEPS, len(facility_ids)), dtype=float)
    )
    directives_by_state: dict[tuple[str, str], pd.Series] = {}
    if round_b_directives is not None and len(round_b_directives):
        for _, directive in round_b_directives.iterrows():
            directives_by_state[
                (
                    str(directive["event_id"]),
                    str(directive["checkpoint_id"]),
                )
            ] = directive
    rows: list[dict] = []
    coverage: list[dict] = []
    for _, catalog_row in catalog.iterrows():
        event_id = str(catalog_row["event_id"])
        checkpoint_id = str(catalog_row["checkpoint_id"])
        state_plan = v1_candidate_plan[
            (v1_candidate_plan["event_id"].astype(str) == event_id)
            & (
                v1_candidate_plan["checkpoint_id"].astype(str)
                == checkpoint_id
            )
        ]
        if state_plan.empty:
            raise ValueError(
                f"state not in v1 candidate plan: {event_id}/{checkpoint_id}"
            )
        base_row = state_plan.iloc[0]
        state_samples = v2_samples[
            (v2_samples["event_id"].astype(str) == event_id)
            & (v2_samples["checkpoint_id"].astype(str) == checkpoint_id)
        ]
        ctx = build_state_context(
            catalog_row,
            base_row,
            state_samples,
            reference_root=reference_root,
            facility_ids=facility_ids,
            facility_semantics=facility_semantics,
            replay_plan=replay_plan,
        )
        seen = _state_seen_shas(
            v1_candidate_plan, v2_samples, event_id, checkpoint_id
        )
        seen.update(
            {
                _schedule_sha(ctx["anchor"]),
                ones_sha,
                _schedule_sha(ctx["di"]),
                _schedule_sha(ctx["nc"]),
            }
        )
        state_counter = 0

        def _emit(
            items: list[dict], round_tag: str, default_role: str
        ) -> int:
            nonlocal state_counter
            emitted = 0
            for item in items:
                role = (
                    item["family"]
                    if item["family"] in POSITIVE_CONTROL_ROLES
                    else default_role
                )
                sample_id = (
                    f"pilotfeas__{event_id}__{checkpoint_id}__"
                    f"{round_tag}__{item['family']}__{state_counter}"
                )
                rows.append(
                    _p3_plan_row(
                        base_row,
                        item,
                        sample_id=sample_id,
                        catalog_row=catalog_row,
                        round_tag=round_tag,
                        search_role=role,
                        schedule_dir=schedule_dir,
                        schedule_dir_relative_to=schedule_dir_relative_to,
                        anchor=ctx["anchor"],
                    )
                )
                state_counter += 1
                emitted += 1
            return emitted

        if bool(catalog_row["positive_control_state"]):
            items = positive_control_items(
                ctx,
                state_samples,
                facility_semantics=facility_semantics,
                seen=seen,
            )
            planned = _emit(items, ROUND_A, "positive_control_probe")
            budget = len(POSITIVE_CONTROL_ROLES)
        else:
            items = materialize_feasibility_candidates(
                ctx,
                facility_semantics=facility_semantics,
                seen=seen,
                per_state_max=ROUND_A_MAX_PER_MISSING_STATE,
                round_tag=ROUND_A,
            )
            planned = _emit(items, ROUND_A, "feasibility_candidate")
            budget = ROUND_A_MAX_PER_MISSING_STATE
        coverage.append(
            {
                "event_id": event_id,
                "checkpoint_id": checkpoint_id,
                "state_id": str(catalog_row["state_id"]),
                "search_round": ROUND_A,
                "budget": int(budget),
                "planned": int(planned),
                "shortfall": int(max(0, budget - planned)),
                "families_planned": int(
                    len({item["family"] for item in items})
                ),
                "seed_schedules_available": int(
                    len(ctx["seed_schedules"])
                ),
            }
        )
        directive = directives_by_state.get((event_id, checkpoint_id))
        if directive is not None and bool(
            catalog_row["joint_missing_state"]
        ):
            round_b_budget = int(
                min(
                    int(directive["round_b_budget"]),
                    ROUND_B_MAX_PER_STATE,
                    TOTAL_BUDGET_PER_MISSING_STATE - planned,
                )
            )
            if round_b_budget > 0:
                items_b = materialize_feasibility_candidates(
                    ctx,
                    facility_semantics=facility_semantics,
                    seen=seen,
                    per_state_max=round_b_budget,
                    round_tag=ROUND_B,
                )
                planned_b = _emit(items_b, ROUND_B, "feasibility_candidate")
                coverage.append(
                    {
                        "event_id": event_id,
                        "checkpoint_id": checkpoint_id,
                        "state_id": str(catalog_row["state_id"]),
                        "search_round": ROUND_B,
                        "budget": int(round_b_budget),
                        "planned": int(planned_b),
                        "shortfall": int(
                            max(0, round_b_budget - planned_b)
                        ),
                        "families_planned": int(
                            len({item["family"] for item in items_b})
                        ),
                        "seed_schedules_available": int(
                            len(ctx["seed_schedules"])
                        ),
                    }
                )
    candidate_plan = pd.DataFrame(rows)
    if len(candidate_plan) > MAX_TOTAL_CANDIDATE_RUNS:
        raise ValueError(
            f"feasibility plan exceeds the frozen budget: "
            f"{len(candidate_plan)} > {MAX_TOTAL_CANDIDATE_RUNS}"
        )
    return {
        "candidate_plan": candidate_plan,
        "search_coverage": pd.DataFrame(coverage),
    }


def audit_feasibility_plan(
    candidate_plan: pd.DataFrame,
    branch_plan: pd.DataFrame,
    v1_candidate_plan: pd.DataFrame,
    v2_samples: pd.DataFrame,
    catalog: pd.DataFrame,
) -> dict:
    """Fail-closed mechanical audit of the P3 feasibility candidate plan.

    Replay rows (``expected_replay_of`` non-empty) are the only rows allowed
    to collide with frozen v1/v2 SHAs of their own state; every other row
    must be actual-unique per state against both frozen datasets and within
    the plan.  Uniqueness is per state by construction (the same schedule
    matrix applied at two different checkpoints is two distinct actions),
    matching the ``_state_seen_shas`` semantics of the generators.
    """
    if candidate_plan.empty:
        return {"status": "blocked", "reason": "empty candidate plan"}
    frozen_by_state: dict[tuple[str, str], set[str]] = {}
    for frame in (v1_candidate_plan, v2_samples):
        for _, row in frame.iterrows():
            key = (str(row["event_id"]), str(row["checkpoint_id"]))
            bucket = frozen_by_state.setdefault(key, set())
            for column in (
                "requested_schedule_sha256",
                "projected_schedule_sha256",
                "actual_schedule_sha256",
            ):
                if column in frame:
                    bucket.add(str(row[column]))
    replay_mask = (
        candidate_plan["expected_replay_of"].fillna("").astype(str) != ""
    )
    non_replay = candidate_plan[~replay_mask]
    catalog_states = set(
        map(tuple, catalog[_STATE_KEYS].astype(str).values)
    )
    plan_states = set(
        map(tuple, candidate_plan[_STATE_KEYS].astype(str).values)
    )
    missing_states = set(
        map(
            tuple,
            catalog.loc[
                catalog["joint_missing_state"], _STATE_KEYS
            ].astype(str).values,
        )
    )
    positive_states = catalog_states - missing_states
    per_state_round = candidate_plan.groupby(
        _STATE_KEYS + ["search_round"]
    ).size()
    per_state_total = candidate_plan.groupby(_STATE_KEYS).size()
    round_a_missing_ok = True
    positive_round_a_ok = True
    for (event_id, checkpoint_id, round_tag), count in (
        per_state_round.items()
    ):
        key = (str(event_id), str(checkpoint_id))
        if round_tag == ROUND_A and key in missing_states:
            round_a_missing_ok &= count <= ROUND_A_MAX_PER_MISSING_STATE
        if key in positive_states:
            positive_round_a_ok &= (
                round_tag == ROUND_A
                and count <= len(POSITIVE_CONTROL_ROLES)
            )
    missing_total_ok = all(
        count <= TOTAL_BUDGET_PER_MISSING_STATE
        for state, count in per_state_total.items()
        if (str(state[0]), str(state[1])) in missing_states
    )
    replay_states = set(
        map(
            tuple,
            candidate_plan.loc[replay_mask, _STATE_KEYS]
            .astype(str)
            .values,
        )
    )
    v2_actual = set(
        v2_samples.get(
            "actual_schedule_sha256", pd.Series(dtype=str)
        ).astype(str)
    )
    constraint_columns = (
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
    known_roles = set(POSITIVE_CONTROL_ROLES) | {
        "feasibility_candidate",
        "positive_control_probe",
    }
    checks = {
        "total_within_frozen_budget": len(candidate_plan)
        <= MAX_TOTAL_CANDIDATE_RUNS,
        "families_known": set(
            candidate_plan["family"].astype(str)
        ).issubset(set(FEASIBILITY_FAMILIES) | set(POSITIVE_CONTROL_ROLES)),
        "search_roles_known": set(
            candidate_plan["search_role"].astype(str)
        ).issubset(known_roles),
        "rounds_known": set(
            candidate_plan["search_round"].astype(str)
        ).issubset({ROUND_A, ROUND_B}),
        "states_cover_catalog": plan_states == catalog_states,
        "round_a_missing_within_budget": bool(round_a_missing_ok),
        "positive_round_a_within_budget": bool(positive_round_a_ok),
        "missing_total_within_budget": bool(missing_total_ok),
        "round_b_only_missing_states": set(
            map(
                tuple,
                candidate_plan.loc[
                    candidate_plan["search_round"].astype(str) == ROUND_B,
                    _STATE_KEYS,
                ]
                .astype(str)
                .values,
            )
        ).issubset(missing_states),
        "replay_present_all_positive_states": replay_states
        == positive_states,
        "replay_targets_in_v2": bool(
            candidate_plan.loc[replay_mask, "expected_actual_sha"]
            .astype(str)
            .isin(v2_actual)
            .all()
        ),
        "k_within_limit": bool(
            (pd.to_numeric(candidate_plan["k_actual"]) <= MAX_K).all()
        ),
        "sample_ids_unique": not candidate_plan["sample_id"]
        .duplicated()
        .any(),
        "requested_sha_unique": not candidate_plan.duplicated(
            _STATE_KEYS + ["requested_schedule_sha256"]
        ).any(),
        "projected_sha_unique": not candidate_plan.duplicated(
            _STATE_KEYS + ["projected_schedule_sha256"]
        ).any(),
        "non_replay_no_frozen_sha_overlap": not any(
            str(row["requested_schedule_sha256"])
            in frozen_by_state.get(
                (str(row["event_id"]), str(row["checkpoint_id"])), set()
            )
            or str(row["projected_schedule_sha256"])
            in frozen_by_state.get(
                (str(row["event_id"]), str(row["checkpoint_id"])), set()
            )
            for _, row in non_replay.iterrows()
        ),
        "constraints_all_true": all(
            bool(candidate_plan[column].all())
            for column in constraint_columns
            if column in candidate_plan
        ),
        "contract_version_uniform": bool(
            (
                candidate_plan["source_contract_version"].astype(str)
                == P3_CONTRACT_VERSION
            ).all()
        ),
        "source_phase_uniform": bool(
            (
                candidate_plan["source_phase"].astype(str)
                == FEASIBILITY_PHASE
            ).all()
        ),
        "reference_reused_all": bool(
            candidate_plan["reference_reused"].all()
        ),
        "branch_rows_four_per_sample": len(branch_plan)
        == 4 * len(candidate_plan),
        "identity_matches_v1": _identity_matches_v1(
            candidate_plan, v1_candidate_plan
        ),
        "training_eligibility_matches_catalog": bool(
            candidate_plan.merge(
                catalog[
                    _STATE_KEYS + ["search_result_training_eligible"]
                ],
                on=_STATE_KEYS,
                suffixes=("", "_catalog"),
            )
            .eval(
                "search_result_training_eligible "
                "== search_result_training_eligible_catalog"
            )
            .all()
        ),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    return {
        "status": "pass" if all(checks.values()) else "blocked",
        "checks": checks,
        "candidate_count": int(len(candidate_plan)),
        "state_count": int(per_state_total.index.size),
        "positive_control_rows": int(
            candidate_plan["positive_control_state"].sum()
        ),
        "round_counts": {
            str(key): int(value)
            for key, value in candidate_plan["search_round"]
            .value_counts()
            .items()
        },
        "family_counts": {
            str(key): int(value)
            for key, value in candidate_plan["family"]
            .value_counts()
            .items()
        },
    }
