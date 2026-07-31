"""Pilot Coverage Extension v1 (Pilot Gate v2 recovery, spec sections III-X).

Read-only gap diagnosis over the immutable Pilot400 v1 sample manifest,
joint-targeted extension state selection, five deterministic candidate
families anchored on the dynamic-internal-rules reference, the conditional
flat auxiliary plan, and the extension-scope plan/dataset audits.

Nothing in this module mutates Pilot400 v1 planning, runs, references or
dataset files; all extension outputs live under ``pilot_extension_v1/``.
Candidate generation never reads candidate-branch SWMM outcomes: the only
simulation-derived input is the same-state dynamic-internal-rules reference
schedule, which is one of the four v1 branches by contract.
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
from .pilot_candidates import (
    HORIZON_STEPS,
    MAX_K,
    _bounds,
    _build_role_request,
    _sha256_json,
)


EXTENSION_FAMILIES = (
    "toward_di_sparse",
    "leave_one_out_tfv",
    "di_nontrivial_boundary",
    "hold_then_di",
    "tfv_coordinate_repair",
)

FLAT_AUXILIARY_ROLES = (
    "k1_strong_legal",
    "k2_strong_legal",
    "binary_legal_toggle",
    "continuous_absolute_level",
    "temporal_pulse",
)

JOINT_EXTENSION_PHASE = "joint_extension"
FLAT_AUXILIARY_PHASE = "flat_auxiliary"
GATE_V2_CONTRACT_VERSION = "project6_v4_pilot_gate_v2"

FLAT_EVENT_SUPPORT_REQUIRED = 3
FLAT_AUXILIARY_MAX_CANDIDATES = 30
NOOP_DISTANCE_FLOOR = 1e-3

_DELTA_PFV = "delta_pfv_h120_vs_no_control"
_DELTA_TFV = "delta_tfv_h120_vs_dynamic_internal"
_DELTA_PEAK = "delta_peak_h120_vs_dynamic_internal"

_STATE_KEYS = ["event_id", "checkpoint_id"]


def _bool(series: pd.Series | None, index: pd.Index) -> pd.Series:
    if series is None:
        return pd.Series(False, index=index)
    return series.fillna(False).astype(bool)


def _facility_moved_mask(row: pd.Series) -> np.ndarray:
    """Per-facility bool mask: projected differs from anchor anywhere."""
    try:
        projected = np.asarray(
            json.loads(str(row.get("projected_schedule_json", "[]"))),
            dtype=float,
        )
        anchor = np.asarray(
            json.loads(str(row.get("anchor_schedule_json", "[]"))),
            dtype=float,
        )
    except (ValueError, TypeError):
        return np.zeros(0, dtype=bool)
    if projected.shape != anchor.shape or projected.ndim != 2:
        return np.zeros(0, dtype=bool)
    return np.abs(projected - anchor).max(axis=0) > 1e-9


def build_state_gap_catalog(
    samples: pd.DataFrame, *, facility_ids: list[str]
) -> pd.DataFrame:
    """One row per Pilot v1 state with joint/flat/actuator gap evidence."""
    required = {
        "event_id",
        "checkpoint_id",
        "checkpoint_role",
        "split",
        "actual_schedule_sha256",
        "pfv_safe",
        "tfv_noninferior",
        "peak_noninferior",
        "joint_noninferior",
        "materially_beneficial",
        "neutral",
        "confirmed_flat",
        "locally_responsive",
        _DELTA_PFV,
        _DELTA_TFV,
        _DELTA_PEAK,
    }
    missing = required - set(samples)
    if missing:
        raise ValueError(f"v1 sample manifest missing: {sorted(missing)}")
    rows: list[dict] = []
    width = len(facility_ids)
    for (event_id, checkpoint_id), group in samples.groupby(_STATE_KEYS):
        moved_union = np.zeros(width, dtype=bool)
        for _, sample in group.iterrows():
            mask = _facility_moved_mask(sample)
            if mask.size == width:
                moved_union |= mask
        joint = _bool(group.get("joint_noninferior"), group.index)
        pfv_safe = _bool(group.get("pfv_safe"), group.index)
        tfv_ok = _bool(group.get("tfv_noninferior"), group.index)
        peak_ok = _bool(group.get("peak_noninferior"), group.index)
        if joint.any():
            reason = ""
        elif not pfv_safe.any():
            reason = "no_pfv_safe_candidate"
        elif not tfv_ok.any():
            reason = "tfv_always_degraded"
        elif not peak_ok.any():
            reason = "peak_always_degraded"
        else:
            reason = "labels_never_jointly_noninferior"
        accepted = int(len(group))
        rows.append(
            {
                "event_id": str(event_id),
                "checkpoint_id": str(checkpoint_id),
                "state_id": str(
                    group["checkpoint_state_sha256"].iloc[0]
                    if "checkpoint_state_sha256" in group
                    else ""
                ),
                "split": str(group["split"].iloc[0]),
                "checkpoint_min": float(
                    pd.to_numeric(
                        group.get("checkpoint_min"), errors="coerce"
                    ).iloc[0]
                )
                if "checkpoint_min" in group
                else float("nan"),
                "checkpoint_role": str(group["checkpoint_role"].iloc[0]),
                "responsive": str(group["checkpoint_role"].iloc[0])
                == "responsive",
                "accepted": accepted,
                "informative": int(
                    group["actual_schedule_sha256"].astype(str).nunique()
                ),
                "pfv_safe_count": int(pfv_safe.sum()),
                "tfv_noninferior_count": int(tfv_ok.sum()),
                "peak_noninferior_count": int(peak_ok.sum()),
                "joint_count": int(joint.sum()),
                "materially_beneficial_count": int(
                    _bool(group.get("materially_beneficial"), group.index).sum()
                ),
                "neutral_count": int(
                    _bool(group.get("neutral"), group.index).sum()
                ),
                "flat_count": int(
                    _bool(group.get("confirmed_flat"), group.index).sum()
                ),
                "locally_responsive_count": int(
                    _bool(group.get("locally_responsive"), group.index).sum()
                ),
                "best_delta_pfv": float(
                    pd.to_numeric(group[_DELTA_PFV], errors="coerce").min()
                ),
                "best_delta_tfv": float(
                    pd.to_numeric(group[_DELTA_TFV], errors="coerce").min()
                ),
                "best_delta_peak": float(
                    pd.to_numeric(group[_DELTA_PEAK], errors="coerce").min()
                ),
                "dominant_failure_reason": reason,
                "actual_schedule_coverage": (
                    float(
                        group["actual_schedule_sha256"].astype(str).nunique()
                    )
                    / accepted
                    if accepted
                    else 0.0
                ),
                "active_actuator_coverage": (
                    float(moved_union.sum()) / width if width else 0.0
                ),
                "actuated_facility_count": int(moved_union.sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(_STATE_KEYS).reset_index(drop=True)


def build_actuator_coverage_gap(
    samples: pd.DataFrame, *, facility_ids: list[str]
) -> pd.DataFrame:
    """Long-form per-state facility actuation counts (read-only evidence)."""
    rows: list[dict] = []
    width = len(facility_ids)
    for (event_id, checkpoint_id), group in samples.groupby(_STATE_KEYS):
        counts = np.zeros(width, dtype=int)
        for _, sample in group.iterrows():
            mask = _facility_moved_mask(sample)
            if mask.size == width:
                counts += mask.astype(int)
        for index, facility in enumerate(facility_ids):
            rows.append(
                {
                    "event_id": str(event_id),
                    "checkpoint_id": str(checkpoint_id),
                    "facility_id": facility,
                    "times_actuated": int(counts[index]),
                }
            )
    return pd.DataFrame(rows)


def build_candidate_family_gap(samples: pd.DataFrame) -> pd.DataFrame:
    """Per responsive state x family label yield (joint gap evidence)."""
    responsive = samples[
        samples["checkpoint_role"].astype(str) == "responsive"
    ]
    rows: list[dict] = []
    for (event_id, checkpoint_id, family), group in responsive.groupby(
        [*_STATE_KEYS, "family"]
    ):
        rows.append(
            {
                "event_id": str(event_id),
                "checkpoint_id": str(checkpoint_id),
                "family": str(family),
                "accepted": int(len(group)),
                "pfv_safe_count": int(
                    _bool(group.get("pfv_safe"), group.index).sum()
                ),
                "tfv_noninferior_count": int(
                    _bool(group.get("tfv_noninferior"), group.index).sum()
                ),
                "joint_count": int(
                    _bool(group.get("joint_noninferior"), group.index).sum()
                ),
            }
        )
    return pd.DataFrame(rows)


def flat_event_support(samples: pd.DataFrame) -> pd.DataFrame:
    """Per-event confirmed-flat support over the given manifest."""
    rows: list[dict] = []
    for event_id, group in samples.groupby("event_id"):
        flat = _bool(group.get("confirmed_flat"), group.index)
        low = group[
            group["checkpoint_role"].astype(str) == "low_opportunity"
        ]
        rows.append(
            {
                "event_id": str(event_id),
                "split": str(group["split"].iloc[0]),
                "confirmed_flat_count": int(flat.sum()),
                "low_opportunity_sample_count": int(len(low)),
                "has_flat_support": bool(flat.any()),
            }
        )
    return pd.DataFrame(rows)


def audit_pilot_coverage_gaps(
    samples: pd.DataFrame,
    *,
    facility_ids: list[str],
    flat_event_support_required: int = FLAT_EVENT_SUPPORT_REQUIRED,
    joint_target_state_count: int = 12,
) -> dict:
    """Read-only V1 gap diagnosis: frames plus the gap audit payload."""
    catalog = build_state_gap_catalog(samples, facility_ids=facility_ids)
    responsive = catalog[catalog["responsive"]]
    missing_joint = responsive[responsive["joint_count"] == 0].copy()
    family_gap = build_candidate_family_gap(samples)
    actuator_gap = build_actuator_coverage_gap(
        samples, facility_ids=facility_ids
    )
    flat_support = flat_event_support(samples)
    support_events = int(flat_support["has_flat_support"].sum())
    joint_states = int((responsive["joint_count"] > 0).sum())
    audit = {
        "stage": "AuditPilotCoverageGaps",
        "read_only": True,
        "source_contract_version": GATE_V2_CONTRACT_VERSION,
        "states_total": int(len(catalog)),
        "responsive_states": int(len(responsive)),
        "joint_state_count": joint_states,
        "joint_state_fraction": (
            float(joint_states) / int(len(responsive))
            if len(responsive)
            else 0.0
        ),
        "missing_joint_state_count": int(len(missing_joint)),
        "joint_target_state_count": int(joint_target_state_count),
        "confirmed_flat_total": int(
            _bool(samples.get("confirmed_flat"), samples.index).sum()
        ),
        "confirmed_flat_event_support": support_events,
        "flat_event_support_required": int(flat_event_support_required),
        "must_run_flat_auxiliary": bool(
            support_events < int(flat_event_support_required)
        ),
        "dominant_failure_reasons": {
            str(key): int(value)
            for key, value in missing_joint[
                "dominant_failure_reason"
            ].value_counts().items()
        },
        "status": "pass",
    }
    return {
        "state_gap_catalog": catalog,
        "missing_joint_states": missing_joint,
        "missing_flat_event_support": flat_support,
        "candidate_family_gap": family_gap,
        "actuator_coverage_gap": actuator_gap,
        "gap_audit": audit,
    }


def select_joint_extension_states(
    gap_catalog: pd.DataFrame,
    *,
    min_states: int = 8,
    max_states: int = 12,
    min_events: int = 3,
    tfv_low_max: int = 1,
) -> pd.DataFrame:
    """Spec section IV filter: pilot_train responsive no-joint states only.

    Preference order: states blocked purely by TFV with PFV margin first,
    then round-robin across events so at least ``min_events`` pilot_train
    events participate.  Never selects calibration/validation/challenge.
    """
    eligible = gap_catalog[
        (gap_catalog["split"].astype(str) == "pilot_train")
        & gap_catalog["responsive"]
        & (gap_catalog["joint_count"] == 0)
        & (gap_catalog["pfv_safe_count"] > 0)
        & (gap_catalog["tfv_noninferior_count"] <= int(tfv_low_max))
        & (gap_catalog["locally_responsive_count"] > 0)
    ].copy()
    if eligible.empty:
        raise ValueError("no eligible joint-extension states")
    eligible["tfv_blocked"] = (
        eligible["dominant_failure_reason"] == "tfv_always_degraded"
    )
    eligible = eligible.sort_values(
        ["tfv_blocked", "best_delta_pfv"], ascending=[False, True]
    )
    # Round-robin over events so selection spans phases and events.
    queues = {
        str(event): list(group.index)
        for event, group in eligible.groupby("event_id", sort=False)
    }
    order: list = []
    while any(queues.values()) and len(order) < int(max_states):
        for event in list(queues):
            if queues[event]:
                order.append(queues[event].pop(0))
                if len(order) >= int(max_states):
                    break
    selected = eligible.loc[order].copy()
    if len(selected) < int(min_states):
        raise ValueError(
            f"only {len(selected)} eligible states; need >= {min_states}"
        )
    if selected["event_id"].nunique() < int(min_events):
        raise ValueError(
            "joint extension selection spans fewer than "
            f"{min_events} pilot_train events"
        )
    selected["selection_rank"] = range(len(selected))
    return selected.reset_index(drop=True)


def di_schedule_from_reference(
    detail: pd.DataFrame,
    *,
    checkpoint_min: float,
    facility_ids: list[str],
    semantics_map: dict,
    anchor: np.ndarray,
) -> np.ndarray:
    """12x36 dynamic-internal-rules setting schedule from the DI reference.

    Binary facilities snap to the dominant bound per decision step;
    continuous facilities take the in-step mean clipped to bounds.  Steps
    without samples fall back to the anchor action (fail-safe, no guess).
    """
    elapsed = pd.to_numeric(detail["elapsed_min"], errors="coerce")
    schedule = anchor.copy()
    for index, facility in enumerate(facility_ids):
        column = f"actual_setting:{facility}"
        if column not in detail:
            raise ValueError(f"DI reference missing column: {column}")
        values = pd.to_numeric(detail[column], errors="coerce")
        semantics = semantics_map[facility]
        lower, upper = _bounds(semantics)
        for step in range(HORIZON_STEPS):
            low = float(checkpoint_min) + step * 10.0
            chunk = values[(elapsed > low) & (elapsed <= low + 10.0)]
            if not len(chunk) or not np.isfinite(chunk.mean()):
                continue
            mean = float(chunk.mean())
            if _is_binary(facility, semantics):
                midpoint = (lower + upper) / 2.0
                schedule[step, index] = upper if mean >= midpoint else lower
            else:
                schedule[step, index] = float(np.clip(mean, lower, upper))
    return schedule


def _ranked_moved_facilities(
    base: np.ndarray, target: np.ndarray
) -> list[int]:
    """Facility indices ranked by |base - target| column mass, desc."""
    gap = np.abs(base - target).sum(axis=0)
    order = np.argsort(-gap)
    return [int(i) for i in order if gap[int(i)] > 1e-9]


def _family_requests(
    family: str,
    ctx: dict,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Deterministic request variants for one family (pre-projection)."""
    anchor: np.ndarray = ctx["anchor"]
    di: np.ndarray = ctx["di"]
    facility_ids: list[str] = ctx["facility_ids"]
    semantics_map: dict = ctx["semantics_map"]
    active_ids: list[str] = ctx["active_ids"]
    best_safe: np.ndarray | None = ctx.get("best_pfv_safe")
    tfv_bases: list[np.ndarray] = ctx.get("tfv_degraded_bases", [])
    requests: list[np.ndarray] = []
    if family == "toward_di_sparse":
        if best_safe is None:
            return []
        ranked = _ranked_moved_facilities(best_safe, di)
        for top_n in (1, 2, 4):
            for steps in (2, 4):
                for weight in (1.0, 0.5):
                    chosen = ranked[:top_n]
                    if not chosen:
                        continue
                    requested = best_safe.copy()
                    for index in chosen:
                        requested[:steps, index] = (
                            best_safe[:steps, index]
                            + weight
                            * (di[:steps, index] - best_safe[:steps, index])
                        )
                    requests.append(requested)
    elif family == "leave_one_out_tfv":
        for base in tfv_bases:
            for index in _ranked_moved_facilities(base, anchor)[:6]:
                requested = base.copy()
                requested[:, index] = anchor[:, index]
                requests.append(requested)
    elif family == "di_nontrivial_boundary":
        pool = [
            facility_ids.index(item)
            for item in (active_ids or facility_ids)
            if item in facility_ids
        ]
        if not pool:
            pool = list(range(len(facility_ids)))
        for k in (1, 2):
            for duration in (2, 4):
                order = rng.permutation(len(pool))
                chosen = [pool[int(i)] for i in order[:k]]
                requested = di.copy()
                for index in chosen:
                    facility = facility_ids[index]
                    semantics = semantics_map[facility]
                    lower, upper = _bounds(semantics)
                    if _is_binary(facility, semantics):
                        current = requested[:duration, index]
                        midpoint = (lower + upper) / 2.0
                        toggled = np.where(
                            current >= midpoint, lower, upper
                        )
                        requested[:duration, index] = toggled
                    else:
                        span = upper - lower
                        sign = 1.0 if rng.integers(2) else -1.0
                        requested[:duration, index] = np.clip(
                            requested[:duration, index]
                            + sign * 0.12 * span,
                            lower,
                            upper,
                        )
                requests.append(requested)
    elif family == "hold_then_di":
        for hold_steps in (1, 2):
            requested = di.copy()
            requested[:hold_steps, :] = anchor[:hold_steps, :]
            requests.append(requested)
    elif family == "tfv_coordinate_repair":
        pool = [
            facility_ids.index(item)
            for item in (active_ids or facility_ids)
            if item in facility_ids
        ]
        ranked = [
            index
            for index in _ranked_moved_facilities(anchor, di)
            if index in pool
        ] or pool
        for k in (1, 2, 4):
            for hold in (2, 4):
                chosen = ranked[: min(k, 4)]
                if not chosen:
                    continue
                requested = anchor.copy()
                for index in chosen:
                    facility = facility_ids[index]
                    semantics = semantics_map[facility]
                    lower, upper = _bounds(semantics)
                    if _is_binary(facility, semantics):
                        requested[:hold, index] = anchor[:hold, index]
                        requested[hold:, index] = di[hold:, index]
                    else:
                        midpoint = (lower + upper) / 2.0
                        requested[:hold, index] = (
                            anchor[:hold, index]
                            + 0.5 * (di[:hold, index] - anchor[:hold, index])
                        )
                        requested[hold:, index] = np.clip(
                            (di[hold:, index] + midpoint) / 2.0,
                            lower,
                            upper,
                        )
                requests.append(requested)
    else:
        raise ValueError(f"unknown extension family: {family}")
    return requests


def materialize_extension_candidates(
    state_ctx: dict,
    *,
    facility_semantics: pd.DataFrame,
    seen: set[str],
    per_state_min: int = 4,
    per_state_max: int = 5,
) -> list[dict]:
    """Project family requests into unique legal candidates for one state.

    ``seen`` must already contain the v1 requested/projected/actual SHAs for
    this state plus the anchor, all-ones and DI schedule SHAs, so extension
    candidates can never duplicate the original 400 or any reference.
    """
    facility_ids: list[str] = state_ctx["facility_ids"]
    anchor: np.ndarray = state_ctx["anchor"]
    seed = int(
        hashlib.sha256(
            str(state_ctx["state_id"]).encode("utf-8")
        ).hexdigest()[:8],
        16,
    )
    accepted: list[dict] = []
    # First pass gives every family one slot; second pass fills remaining.
    for round_index in (0, 1):
        for family_index, family in enumerate(EXTENSION_FAMILIES):
            if len(accepted) >= per_state_max:
                break
            if round_index == 0 and any(
                item["family"] == family for item in accepted
            ):
                continue
            rng = np.random.default_rng(
                seed + family_index * 7919 + round_index * 104729
            )
            for requested in _family_requests(family, state_ctx, rng):
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
        if len(accepted) >= per_state_max:
            break
    if len(accepted) < per_state_min:
        raise ValueError(
            f"state {state_ctx['state_id']}: only {len(accepted)} unique "
            f"legal extension candidates; need >= {per_state_min}"
        )
    return accepted


def _state_seen_shas(
    v1_candidate_plan: pd.DataFrame,
    v1_samples: pd.DataFrame,
    event_id: str,
    checkpoint_id: str,
) -> set[str]:
    seen: set[str] = set()
    for frame in (v1_candidate_plan, v1_samples):
        state = frame[
            (frame["event_id"].astype(str) == event_id)
            & (frame["checkpoint_id"].astype(str) == checkpoint_id)
        ]
        for column in (
            "requested_schedule_sha256",
            "projected_schedule_sha256",
            "actual_schedule_sha256",
        ):
            if column in state:
                seen.update(
                    sha
                    for sha in state[column].astype(str)
                    if sha and sha != "nan"
                )
    return seen


def _candidate_plan_row(
    base_row: pd.Series,
    item: dict,
    *,
    sample_id: str,
    source_phase: str,
    base_candidate_id: str,
    schedule_dir: Path | None,
    schedule_dir_relative_to: Path | None,
    anchor: np.ndarray,
) -> dict:
    """One extension candidate plan row, v1-schema compatible."""
    base_kwargs = json.loads(str(base_row["runner_kwargs"]))
    base_kwargs["post_action"] = item["projected"].tolist()
    schedule_path = ""
    if schedule_dir is not None:
        schedule_dir.mkdir(parents=True, exist_ok=True)
        target = schedule_dir / f"{sample_id}.json"
        target.write_text(
            json.dumps(
                {
                    "sample_id": sample_id,
                    "requested_schedule": item["requested"].tolist(),
                    "projected_schedule": item["projected"].tolist(),
                    "anchor_schedule": anchor.tolist(),
                    "requested_schedule_sha256": item["requested_sha"],
                    "projected_schedule_sha256": item["projected_sha"],
                }
            ),
            encoding="utf-8",
        )
        base = schedule_dir_relative_to or schedule_dir
        schedule_path = target.relative_to(base).as_posix()
    k_by_step = item["projection"].get("k_by_step", [])
    max_k_used = int(max([int(k) for k in k_by_step] or [0]))
    return {
        "sample_id": sample_id,
        "case_id": sample_id,
        "event_id": str(base_row["event_id"]),
        "rainfall_sha256": str(base_row["rainfall_sha256"]),
        "checkpoint_id": str(base_row["checkpoint_id"]),
        "checkpoint_role": str(base_row["checkpoint_role"]),
        "checkpoint_min": float(base_row["checkpoint_min"]),
        "checkpoint_state_sha256": str(
            base_row["checkpoint_state_sha256"]
        ),
        "candidate_role": item["family"],
        "candidate_family": item["family"],
        "family": item["family"],
        "priority": 0,
        "source_anchor_role": "checkpoint_anchor",
        "source_anchor_id": base_candidate_id,
        "requested_schedule_json": json.dumps(item["requested"].tolist()),
        "projected_schedule_json": json.dumps(item["projected"].tolist()),
        "anchor_schedule_json": json.dumps(anchor.tolist()),
        "requested_schedule_sha256": item["requested_sha"],
        "projected_schedule_sha256": item["projected_sha"],
        "requested_schedule_path": schedule_path,
        "projected_schedule_path": schedule_path,
        "schedule_format": "json",
        "k_target": max(1, max_k_used),
        "k_actual": max_k_used,
        "k_sequence": json.dumps([int(k) for k in k_by_step]),
        "binary_semantics_ok": bool(
            item["flags"]["binary_semantics_ok"]
        ),
        "vsp_semantics_ok": True,
        "bounds_ok": True,
        "rate_limit_ok": bool(item["flags"]["rate_limit_ok"]),
        "ramp_ok": bool(item["flags"]["rate_limit_ok"]),
        "dwell_ok": bool(item["flags"]["dwell_ok"]),
        "interlock_ok": bool(item["flags"]["interlock_ok"]),
        "no_reversal_ok": bool(
            item["projection"].get("no_reversal_ok", True)
        ),
        "projection_valid": True,
        "materialization_attempt": 0,
        "split": str(base_row["split"]),
        "runner_group_key": sample_id,
        "runner_function": "run_swmm_fixed_action",
        "runner_kwargs": json.dumps(base_kwargs, separators=(",", ":")),
        "network_sha256": str(base_row["network_sha256"]),
        "contract_sha256": str(base_row["contract_sha256"]),
        "config_sha256": str(base_row["config_sha256"]),
        "code_sha256": str(base_row.get("code_sha256", "")),
        "status": "planned",
        "source_phase": source_phase,
        "source_contract_version": GATE_V2_CONTRACT_VERSION,
        "selected_after_pilot_v1": True,
        "reference_reused": True,
        "base_candidate_id": base_candidate_id,
    }


def plan_pilot_coverage_extension(
    selected_states: pd.DataFrame,
    v1_candidate_plan: pd.DataFrame,
    v1_samples: pd.DataFrame,
    *,
    reference_root: Path,
    facility_ids: list[str],
    facility_semantics: pd.DataFrame,
    schedule_dir: Path | None = None,
    schedule_dir_relative_to: Path | None = None,
    per_state_min: int = 4,
    per_state_max: int = 5,
) -> dict:
    """Materialize the joint extension plan (40-60 candidates, 5 families).

    Reuses the v1 checkpoint state verbatim (same runner kwargs, network,
    anchor and split) so the v1 reference cache is hit unchanged; only the
    candidate branch is ever a new SWMM run.
    """
    semantics_map = _semantics_by_id(facility_ids, facility_semantics)
    ones_sha = _schedule_sha(
        np.ones((HORIZON_STEPS, len(facility_ids)), dtype=float)
    )
    rows: list[dict] = []
    coverage_missing: list[dict] = []
    for _, state in selected_states.iterrows():
        event_id = str(state["event_id"])
        checkpoint_id = str(state["checkpoint_id"])
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
        anchor = np.asarray(
            json.loads(str(base_row["anchor_schedule_json"])), dtype=float
        )
        di_detail_path = (
            Path(reference_root)
            / "references"
            / event_id
            / checkpoint_id
            / "dynamic_internal_rules"
            / "detail.csv"
        )
        if not di_detail_path.exists():
            raise ValueError(
                f"DI reference detail missing: {di_detail_path}"
            )
        di = di_schedule_from_reference(
            pd.read_csv(di_detail_path),
            checkpoint_min=float(base_row["checkpoint_min"]),
            facility_ids=facility_ids,
            semantics_map=semantics_map,
            anchor=anchor,
        )
        state_samples = v1_samples[
            (v1_samples["event_id"].astype(str) == event_id)
            & (v1_samples["checkpoint_id"].astype(str) == checkpoint_id)
        ]
        pfv_safe = state_samples[
            _bool(state_samples.get("pfv_safe"), state_samples.index)
        ]
        best_pfv_safe = None
        if len(pfv_safe):
            best = pfv_safe.loc[
                pd.to_numeric(
                    pfv_safe[_DELTA_PFV], errors="coerce"
                ).idxmin()
            ]
            best_pfv_safe = np.asarray(
                json.loads(str(best["projected_schedule_json"])),
                dtype=float,
            )
        tfv_degraded = pfv_safe[
            ~_bool(pfv_safe.get("tfv_noninferior"), pfv_safe.index)
        ]
        tfv_bases = [
            np.asarray(
                json.loads(str(row["projected_schedule_json"])),
                dtype=float,
            )
            for _, row in tfv_degraded.sort_values(_DELTA_TFV).iterrows()
        ][:3]
        try:
            active_ids = [
                str(item)
                for item in json.loads(
                    str(base_row.get("active_facility_ids_json", "[]"))
                )
                if str(item) in facility_ids
            ]
        except (ValueError, TypeError):
            active_ids = []
        if not active_ids:
            moved = np.abs(di - anchor).max(axis=0) > 1e-9
            active_ids = [
                facility_ids[i] for i in range(len(facility_ids)) if moved[i]
            ]
        seen = _state_seen_shas(
            v1_candidate_plan, v1_samples, event_id, checkpoint_id
        )
        seen.update(
            {_schedule_sha(anchor), ones_sha, _schedule_sha(di)}
        )
        ctx = {
            "state_id": str(base_row["checkpoint_state_sha256"]),
            "facility_ids": facility_ids,
            "semantics_map": semantics_map,
            "anchor": anchor,
            "di": di,
            "active_ids": active_ids,
            "best_pfv_safe": best_pfv_safe,
            "tfv_degraded_bases": tfv_bases,
        }
        try:
            items = materialize_extension_candidates(
                ctx,
                facility_semantics=facility_semantics,
                seen=seen,
                per_state_min=per_state_min,
                per_state_max=per_state_max,
            )
        except ValueError as exc:
            coverage_missing.append(
                {
                    "event_id": event_id,
                    "checkpoint_id": checkpoint_id,
                    "reason": str(exc),
                }
            )
            continue
        for number, item in enumerate(items):
            sample_id = (
                f"pilotext__{event_id}__{checkpoint_id}__"
                f"{item['family']}__{number}"
            )
            rows.append(
                _candidate_plan_row(
                    base_row,
                    item,
                    sample_id=sample_id,
                    source_phase=JOINT_EXTENSION_PHASE,
                    base_candidate_id=str(base_row["sample_id"]),
                    schedule_dir=schedule_dir,
                    schedule_dir_relative_to=schedule_dir_relative_to,
                    anchor=anchor,
                )
            )
    candidate_plan = pd.DataFrame(rows)
    return {
        "candidate_plan": candidate_plan,
        "coverage_missing": pd.DataFrame(coverage_missing),
    }


def plan_pilot_flat_auxiliary(
    gap_catalog: pd.DataFrame,
    v1_candidate_plan: pd.DataFrame,
    v1_samples: pd.DataFrame,
    *,
    facility_ids: list[str],
    facility_semantics: pd.DataFrame,
    schedule_dir: Path | None = None,
    schedule_dir_relative_to: Path | None = None,
    min_states: int = 4,
    max_states: int = 6,
    per_state_min: int = 3,
    per_state_max: int = 5,
    max_candidates: int = FLAT_AUXILIARY_MAX_CANDIDATES,
) -> dict:
    """Flat auxiliary plan: strong legal probes on low-opportunity states.

    Only pilot_train low-opportunity states across >= 3 events; candidates
    are strong-but-legal probes and are never forced to come back flat.
    """
    low_states = gap_catalog[
        (gap_catalog["split"].astype(str) == "pilot_train")
        & (~gap_catalog["responsive"])
    ].copy()
    if low_states["event_id"].nunique() < 3:
        raise ValueError(
            "flat auxiliary needs low-opportunity states from >= 3 events"
        )
    low_states = low_states.sort_values("event_id")
    selected = low_states.head(int(max_states))
    if len(selected) < int(min_states):
        raise ValueError(
            f"only {len(selected)} low-opportunity pilot_train states; "
            f"need >= {min_states}"
        )
    semantics_map = _semantics_by_id(facility_ids, facility_semantics)
    ones_sha = _schedule_sha(
        np.ones((HORIZON_STEPS, len(facility_ids)), dtype=float)
    )
    # Global dedup set: the plan audit checks SHA uniqueness across the
    # whole plan and against every v1 SHA, so seed with all v1 SHAs once
    # and accumulate across states (role-based requests can otherwise
    # collide between states).
    seen: set[str] = {ones_sha}
    for frame in (v1_candidate_plan, v1_samples):
        for column in (
            "requested_schedule_sha256",
            "projected_schedule_sha256",
            "actual_schedule_sha256",
        ):
            if column in frame:
                seen.update(
                    sha
                    for sha in frame[column].astype(str)
                    if sha and sha != "nan"
                )
    rows: list[dict] = []
    for _, state in selected.iterrows():
        if len(rows) >= int(max_candidates):
            break
        event_id = str(state["event_id"])
        checkpoint_id = str(state["checkpoint_id"])
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
        anchor = np.asarray(
            json.loads(str(base_row["anchor_schedule_json"])), dtype=float
        )
        anchor_sha = _schedule_sha(anchor)
        seen.add(anchor_sha)
        seed = int(
            hashlib.sha256(
                f"flataux__{event_id}__{checkpoint_id}".encode("utf-8")
            ).hexdigest()[:8],
            16,
        )
        state_rows = 0
        for role_index, role in enumerate(FLAT_AUXILIARY_ROLES):
            if state_rows >= per_state_max or len(rows) >= int(
                max_candidates
            ):
                break
            for attempt in range(8):
                rng = np.random.default_rng(
                    seed + role_index * 7919 + attempt * 104729
                )
                requested = _build_role_request(
                    role, rng, anchor, facility_ids, semantics_map, [], []
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
                seen.add(requested_sha)
                seen.add(projected_sha)
                sample_id = (
                    f"pilotflataux__{event_id}__{checkpoint_id}__"
                    f"{role}__{state_rows}"
                )
                item = {
                    "family": f"flat_auxiliary_{role}",
                    "requested": requested,
                    "projected": projected,
                    "requested_sha": requested_sha,
                    "projected_sha": projected_sha,
                    "projection": projection,
                    "flags": flags,
                }
                rows.append(
                    _candidate_plan_row(
                        base_row,
                        item,
                        sample_id=sample_id,
                        source_phase=FLAT_AUXILIARY_PHASE,
                        base_candidate_id=str(base_row["sample_id"]),
                        schedule_dir=schedule_dir,
                        schedule_dir_relative_to=schedule_dir_relative_to,
                        anchor=anchor,
                    )
                )
                state_rows += 1
                break
        if state_rows < per_state_min:
            raise ValueError(
                f"flat auxiliary state {event_id}/{checkpoint_id}: only "
                f"{state_rows} candidates; need >= {per_state_min}"
            )
    candidate_plan = pd.DataFrame(rows)
    return {"candidate_plan": candidate_plan}


def audit_pilot_extension_plan(
    candidate_plan: pd.DataFrame,
    branch_plan: pd.DataFrame,
    v1_candidate_plan: pd.DataFrame,
    v1_samples: pd.DataFrame,
    *,
    source_phase: str = JOINT_EXTENSION_PHASE,
    min_candidates: int = 32,
    max_candidates: int = 60,
    min_states: int = 8,
    min_events: int = 3,
    min_families: int = 2,
    per_state_min: int = 4,
    per_state_max: int = 5,
) -> dict:
    """Fail-closed audit for an extension/auxiliary candidate + branch plan."""
    if candidate_plan.empty:
        return {"status": "blocked", "reason": "empty candidate plan"}
    v1_shas: set[str] = set()
    for frame in (v1_candidate_plan, v1_samples):
        for column in (
            "requested_schedule_sha256",
            "projected_schedule_sha256",
            "actual_schedule_sha256",
        ):
            if column in frame:
                v1_shas.update(frame[column].astype(str))
    per_state = candidate_plan.groupby(_STATE_KEYS).size()
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
    v1_states = set(
        map(tuple, v1_candidate_plan[_STATE_KEYS].astype(str).values)
    )
    plan_states = set(map(tuple, candidate_plan[_STATE_KEYS].astype(str).values))
    checks = {
        "candidate_count_in_range": (
            int(min_candidates) <= len(candidate_plan) <= int(max_candidates)
        ),
        "state_count_at_least_min": per_state.index.size >= int(min_states),
        "per_state_within_bounds": bool(
            per_state.between(per_state_min, per_state_max).all()
        ),
        "events_at_least_min": (
            candidate_plan["event_id"].nunique() >= int(min_events)
        ),
        "families_at_least_min": (
            candidate_plan["family"].nunique() >= int(min_families)
        ),
        "all_pilot_train": bool(
            (candidate_plan["split"].astype(str) == "pilot_train").all()
        ),
        "source_phase_uniform": bool(
            (candidate_plan["source_phase"].astype(str) == source_phase).all()
        ),
        "all_states_from_v1": plan_states.issubset(v1_states),
        "constraints_all_true": all(
            bool(candidate_plan[column].all())
            for column in constraint_columns
            if column in candidate_plan
        ),
        "k_within_limit": bool(
            (pd.to_numeric(candidate_plan["k_actual"]) <= MAX_K).all()
        ),
        "sample_ids_unique": not candidate_plan["sample_id"]
        .duplicated()
        .any(),
        "requested_sha_unique": not candidate_plan[
            "requested_schedule_sha256"
        ]
        .duplicated()
        .any(),
        "projected_sha_unique": not candidate_plan[
            "projected_schedule_sha256"
        ]
        .duplicated()
        .any(),
        "no_v1_sha_overlap": not (
            set(candidate_plan["requested_schedule_sha256"].astype(str))
            | set(candidate_plan["projected_schedule_sha256"].astype(str))
        )
        & v1_shas,
        "reference_reused_all": bool(
            candidate_plan["reference_reused"].all()
        ),
        "branch_rows_four_per_sample": len(branch_plan)
        == 4 * len(candidate_plan),
        "identity_matches_v1": _identity_matches_v1(
            candidate_plan, v1_candidate_plan
        ),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    return {
        "status": "pass" if all(checks.values()) else "blocked",
        "checks": checks,
        "candidate_count": int(len(candidate_plan)),
        "state_count": int(per_state.index.size),
        "event_count": int(candidate_plan["event_id"].nunique()),
        "family_counts": {
            str(key): int(value)
            for key, value in candidate_plan["family"]
            .value_counts()
            .items()
        },
        "source_phase": source_phase,
    }


def _identity_matches_v1(
    candidate_plan: pd.DataFrame, v1_candidate_plan: pd.DataFrame
) -> bool:
    """Reference reuse requires byte-identical identity fields per state."""
    v1_index = v1_candidate_plan.drop_duplicates(_STATE_KEYS).set_index(
        [
            v1_candidate_plan.drop_duplicates(_STATE_KEYS)[key].astype(str)
            for key in _STATE_KEYS
        ]
    )
    for _, row in candidate_plan.iterrows():
        key = (str(row["event_id"]), str(row["checkpoint_id"]))
        try:
            v1_row = v1_index.loc[key]
        except KeyError:
            return False
        if isinstance(v1_row, pd.DataFrame):
            v1_row = v1_row.iloc[0]
        for column in (
            "network_sha256",
            "contract_sha256",
            "config_sha256",
            "checkpoint_state_sha256",
            "rainfall_sha256",
        ):
            if str(row[column]) != str(v1_row[column]):
                return False
    return True


def audit_pilot_extension_dataset(
    samples: pd.DataFrame,
    accounting: dict,
    *,
    expected_source_phase: str,
    v1_actual_shas: set[str],
    hard_columns: tuple[str, ...],
    min_families: int = 2,
    min_states: int = 2,
) -> dict:
    """Full-scope extension dataset gate (spec section X quality bars)."""
    if samples.empty:
        return {"status": "scientific_fail", "reason": "no accepted samples"}
    tfv = pd.to_numeric(samples.get(_DELTA_TFV), errors="coerce").round(6)
    distance = pd.to_numeric(
        samples.get("actual_action_distance"), errors="coerce"
    ).fillna(0.0)
    extension_shas = samples["actual_schedule_sha256"].astype(str)
    hard_ok = all(
        bool(samples[column].fillna(False).astype(bool).all())
        for column in hard_columns
        if column in samples
    )
    checks = {
        "accounting_closed": bool(accounting.get("accounting_closed", False)),
        "no_missing": int(accounting.get("missing", -1)) == 0,
        "source_phase_uniform": bool(
            (
                samples["source_phase"].astype(str)
                == expected_source_phase
            ).all()
        ),
        "all_pilot_train": bool(
            (samples["split"].astype(str) == "pilot_train").all()
        ),
        "hard_authenticity_100pct": hard_ok,
        "same_state_100pct": bool(
            samples["state_hash_match"].fillna(False).astype(bool).all()
        ),
        "readback_100pct": bool(
            samples["readback_ok"].fillna(False).astype(bool).all()
        ),
        "actual_duplicates_0_within": not samples.duplicated(
            [*_STATE_KEYS, "actual_schedule_sha256"]
        ).any(),
        "actual_duplicates_0_vs_v1": not (
            set(extension_shas) & v1_actual_shas
        ),
        "tfv_delta_not_constant": int(tfv.nunique(dropna=True)) > 1,
        "not_all_near_noop": bool(
            (distance > NOOP_DISTANCE_FLOOR).any()
        ),
        "families_at_least_min": (
            samples["family"].nunique() >= int(min_families)
        ),
        "states_at_least_min": (
            samples.groupby(_STATE_KEYS).ngroups >= int(min_states)
        ),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    return {
        "status": "pass" if all(checks.values()) else "scientific_fail",
        "checks": checks,
        "accepted": int(len(samples)),
        "family_counts": {
            str(key): int(value)
            for key, value in samples["family"].value_counts().items()
        },
        "state_count": int(samples.groupby(_STATE_KEYS).ngroups),
        "event_count": int(samples["event_id"].nunique()),
        "joint_noninferior_count": int(
            _bool(samples.get("joint_noninferior"), samples.index).sum()
        ),
        "accounting": accounting,
    }
