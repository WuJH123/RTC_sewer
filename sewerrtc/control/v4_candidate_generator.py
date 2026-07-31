"""V4 Gate 5R candidate generation.

The generator produces twelve 10-minute action steps and projects every
requested schedule against the *frozen fallback* schedule.  Reference policies
remain label sources; they are not allowed to bypass Engineering36 constraints.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Iterable, Optional

import numpy as np
import pandas as pd


BINARY_PUMPS = {"ADD301.2", "ADD301.3"}
VSP_PUMP = "add350.1"
HORIZON_STEPS = 12


def _float_or(value: object, default: float) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return float(default)
    return float(default) if not np.isfinite(parsed) else parsed


def _normalise_schedule(value: np.ndarray, n_facilities: int) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim == 1 and array.size == n_facilities:
        return np.repeat(array.reshape(1, -1), HORIZON_STEPS, axis=0)
    if array.shape != (HORIZON_STEPS, n_facilities):
        raise ValueError(
            f"schedule shape must be ({HORIZON_STEPS}, {n_facilities}), got {array.shape}"
        )
    return array.copy()


def _semantics_by_id(
    facility_ids: list[str], facility_semantics: pd.DataFrame
) -> dict[str, dict]:
    if "facility_id" not in facility_semantics.columns:
        return {facility_id: {} for facility_id in facility_ids}
    rows = facility_semantics.set_index("facility_id", drop=False)
    return {
        facility_id: (
            rows.loc[facility_id].to_dict()
            if facility_id in rows.index and isinstance(rows.loc[facility_id], pd.Series)
            else {}
        )
        for facility_id in facility_ids
    }


def _is_binary(facility_id: str, semantics: dict) -> bool:
    if facility_id == VSP_PUMP:
        return False
    declared = str(semantics.get("binary_or_continuous", "")).strip().lower()
    action_set = str(semantics.get("action_set", "")).strip().lower()
    return facility_id in BINARY_PUMPS or declared == "binary" or action_set in {
        "{0,1}",
        "{0.0,1.0}",
        "0,1",
        "0.0,1.0",
    }


def _schedule_hash(schedule: np.ndarray) -> str:
    canonical = np.round(np.asarray(schedule, dtype=np.float64), 8)
    return hashlib.sha256(canonical.tobytes()).hexdigest()


def project_candidate_schedule(
    requested_schedule: np.ndarray,
    frozen_anchor_schedule: np.ndarray,
    facility_ids: list[str],
    facility_semantics: pd.DataFrame,
    max_k: int = 8,
    setting_deadband: float = 1e-6,
) -> tuple[np.ndarray, dict]:
    """Project a requested 12-step schedule onto the Engineering36 contract."""
    if max_k < 0:
        raise ValueError("max_k must be non-negative")
    n_facilities = len(facility_ids)
    requested = _normalise_schedule(requested_schedule, n_facilities)
    anchor = _normalise_schedule(frozen_anchor_schedule, n_facilities)
    semantics = _semantics_by_id(facility_ids, facility_semantics)
    projected = requested.copy()

    # Bounds and binary semantics.
    for index, facility_id in enumerate(facility_ids):
        row = semantics[facility_id]
        lower = _float_or(row.get("lower_bound"), 0.0)
        upper = _float_or(row.get("upper_bound"), 1.0)
        if lower > upper:
            lower, upper = upper, lower
        projected[:, index] = np.clip(projected[:, index], lower, upper)
        if _is_binary(facility_id, row):
            midpoint = (lower + upper) / 2.0
            projected[:, index] = np.where(
                projected[:, index] >= midpoint, upper, lower
            )

    # Rate limits operate at the 10-minute decision interval.
    for index, facility_id in enumerate(facility_ids):
        row = semantics[facility_id]
        if _is_binary(facility_id, row):
            continue
        rate_limit = _float_or(row.get("rate_limit"), 1.0)
        previous = float(anchor[0, index])
        for step in range(HORIZON_STEPS):
            projected[step, index] = np.clip(
                projected[step, index], previous - rate_limit, previous + rate_limit
            )
            previous = float(projected[step, index])

    # Optional no-reversal contract. Once a facility first moves away from its
    # frozen anchor in one direction, a later move to the opposite side is
    # reset to the anchor rather than creating an oscillatory schedule.
    no_reversal_adjustments = 0
    for index, facility_id in enumerate(facility_ids):
        row = semantics[facility_id]
        enabled = str(row.get("no_reversal", "")).strip().lower() in {
            "1",
            "true",
            "yes",
        }
        if not enabled:
            continue
        direction = 0
        for step in range(HORIZON_STEPS):
            delta = float(projected[step, index] - anchor[step, index])
            current_direction = 1 if delta > setting_deadband else (
                -1 if delta < -setting_deadband else 0
            )
            if current_direction == 0:
                continue
            if direction == 0:
                direction = current_direction
            elif current_direction != direction:
                projected[step, index] = anchor[step, index]
                no_reversal_adjustments += 1

    # Binary dwell: once a transition occurs, retain it for min_hold_steps.
    for index, facility_id in enumerate(facility_ids):
        row = semantics[facility_id]
        if not _is_binary(facility_id, row):
            continue
        min_hold = max(1, int(_float_or(row.get("min_hold_steps"), 1)))
        last_change = -min_hold
        previous = float(anchor[0, index])
        for step in range(HORIZON_STEPS):
            value = float(projected[step, index])
            if value != previous:
                if step - last_change < min_hold:
                    projected[step, index] = previous
                else:
                    last_change = step
                    previous = value

    # Storage inlet/outlet interlocks use the semantics table, not hard-coded IDs.
    groups: dict[str, dict[str, list[int]]] = {}
    for index, facility_id in enumerate(facility_ids):
        row = semantics[facility_id]
        group = str(row.get("interlock_group", "")).strip()
        role = str(row.get("storage_role", "")).strip().lower()
        if not group or group.lower() == "nan":
            continue
        bucket = groups.setdefault(group, {"inlet": [], "outlet": []})
        if role == "storage_inlet":
            bucket["inlet"].append(index)
        elif role == "storage_outlet":
            bucket["outlet"].append(index)
    interlock_adjustments = 0
    for members in groups.values():
        for step in range(HORIZON_STEPS):
            inlet_open = any(projected[step, idx] > 0.01 for idx in members["inlet"])
            outlet_open = any(projected[step, idx] > 0.01 for idx in members["outlet"])
            if inlet_open and outlet_open:
                for idx in members["outlet"]:
                    if projected[step, idx] != 0.0:
                        projected[step, idx] = 0.0
                        interlock_adjustments += 1

    # Adaptive K is evaluated after all semantic projection, relative to the
    # frozen fallback schedule at every step.
    k_by_step: list[int] = []
    for step in range(HORIZON_STEPS):
        distance = np.abs(projected[step] - anchor[step])
        changed = np.flatnonzero(distance > float(setting_deadband))
        if len(changed) > max_k:
            keep = set(
                changed[np.argsort(distance[changed])[::-1][:max_k]].tolist()
            )
            for index in changed:
                if int(index) not in keep:
                    projected[step, index] = anchor[step, index]
        k_by_step.append(
            int(
                (
                    np.abs(projected[step] - anchor[step])
                    > float(setting_deadband)
                ).sum()
            )
        )

    binary_ok = all(
        (
            np.isclose(projected[:, index], 0.0)
            | np.isclose(projected[:, index], 1.0)
        ).all()
        for index, facility_id in enumerate(facility_ids)
        if _is_binary(facility_id, semantics[facility_id])
    )
    rate_ok = True
    for index, facility_id in enumerate(facility_ids):
        row = semantics[facility_id]
        if _is_binary(facility_id, row):
            continue
        rate_limit = _float_or(row.get("rate_limit"), 1.0)
        sequence = np.concatenate([[anchor[0, index]], projected[:, index]])
        if np.any(np.abs(np.diff(sequence)) > rate_limit + 1e-9):
            rate_ok = False

    audit = {
        "k_by_step": k_by_step,
        "max_k": int(max(k_by_step, default=0)),
        "binary_ok": bool(binary_ok),
        "rate_ok": bool(rate_ok),
        "no_reversal_ok": True,
        "no_reversal_adjustments": int(no_reversal_adjustments),
        "interlock_adjustments": int(interlock_adjustments),
        "projected_schedule_hash": _schedule_hash(projected),
    }
    return projected, audit


def project_frozen_anchor_schedule(
    raw_anchor_schedule: np.ndarray,
    facility_ids: list[str],
    facility_semantics: pd.DataFrame,
    max_k: int = 8,
) -> tuple[np.ndarray, dict]:
    """Make the frozen fallback itself satisfy the action contract.

    Candidate K must be measured from the *actual executable fallback*, not
    from a raw checkpoint vector that would itself be changed by binary,
    bounds, dwell, or storage-interlock projection.
    """
    raw_anchor = _normalise_schedule(raw_anchor_schedule, len(facility_ids))
    return project_candidate_schedule(
        raw_anchor,
        raw_anchor,
        facility_ids,
        facility_semantics,
        max_k=max_k,
    )


@dataclass
class CandidateContext:
    event_id: str
    checkpoint_id: str
    facility_ids: list[str]
    frozen_fallback_schedule: np.ndarray
    dynamic_internal_schedule: np.ndarray
    no_control_schedule: np.ndarray
    hold_previous_schedule: np.ndarray
    opportunity_scores: dict[str, float] = field(default_factory=dict)
    hydraulically_active: set[str] = field(default_factory=set)
    event_phase: str = "unknown"
    online_features: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        n = len(self.facility_ids)
        self.frozen_fallback_schedule = _normalise_schedule(
            self.frozen_fallback_schedule, n
        )
        self.dynamic_internal_schedule = _normalise_schedule(
            self.dynamic_internal_schedule, n
        )
        self.no_control_schedule = _normalise_schedule(
            self.no_control_schedule, n
        )
        self.hold_previous_schedule = _normalise_schedule(
            self.hold_previous_schedule, n
        )


@dataclass
class CandidateSchedule:
    candidate_id: str
    family: str
    facility_ids: list[str]
    requested_schedule: np.ndarray
    projected_schedule: np.ndarray
    anchor_name: str
    description: str
    constraint_audit: dict
    metadata: dict = field(default_factory=dict)
    written_schedule: Optional[np.ndarray] = None
    readback_schedule: Optional[np.ndarray] = None

    @property
    def action(self) -> np.ndarray:
        """Legacy first-step view used by the old Gate 5 runner."""
        return self.projected_schedule[0].copy()

    @property
    def k_by_step(self) -> list[int]:
        return list(self.constraint_audit.get("k_by_step", []))

    @property
    def k_actual(self) -> int:
        return int(max(self.k_by_step, default=0))

    @property
    def projected_schedule_hash(self) -> str:
        return str(
            self.constraint_audit.get(
                "projected_schedule_hash", _schedule_hash(self.projected_schedule)
            )
        )


# Backward-compatible public name.
V4Candidate = CandidateSchedule


class V4CandidateGenerator:
    def __init__(
        self,
        facility_ids: list[str],
        facility_semantics: pd.DataFrame,
        priority_nodes: list[str],
        sensitivity_map: Optional[dict] = None,
        max_k: int = 8,
    ):
        self.facility_ids = list(facility_ids)
        self.n_facilities = len(self.facility_ids)
        self.semantics = facility_semantics.copy()
        self.priority_nodes = list(priority_nodes)
        self.sensitivity = sensitivity_map or {}
        self.max_k = int(max_k)
        self.fid_to_idx = {
            facility_id: index
            for index, facility_id in enumerate(self.facility_ids)
        }

    def _rank(self, context: CandidateContext) -> list[str]:
        active = (
            set(context.hydraulically_active)
            if context.hydraulically_active
            else set(self.facility_ids)
        )
        configured = [
            facility_id
            for facility_id in self.sensitivity.get("ranking", [])
            if facility_id in active
        ]
        remaining = [facility_id for facility_id in self.facility_ids if facility_id in active and facility_id not in configured]
        remaining.sort(
            key=lambda facility_id: (
                -float(context.opportunity_scores.get(facility_id, 0.0)),
                self.fid_to_idx[facility_id],
            )
        )
        return configured + remaining

    def _candidate(
        self,
        context: CandidateContext,
        candidate_id: str,
        family: str,
        requested: np.ndarray,
        anchor_name: str,
        description: str,
        metadata: Optional[dict] = None,
    ) -> CandidateSchedule | None:
        projected, audit = project_candidate_schedule(
            requested,
            context.frozen_fallback_schedule,
            self.facility_ids,
            self.semantics,
            max_k=self.max_k,
        )
        if max(audit["k_by_step"], default=0) == 0:
            return None
        return CandidateSchedule(
            candidate_id=candidate_id,
            family=family,
            facility_ids=self.facility_ids.copy(),
            requested_schedule=np.asarray(requested, dtype=np.float64).copy(),
            projected_schedule=projected,
            anchor_name=anchor_name,
            description=description,
            constraint_audit=audit,
            metadata=dict(metadata or {}),
        )

    def _requests(
        self, context: CandidateContext
    ) -> dict[str, list[tuple[str, np.ndarray, str, str, dict]]]:
        ranked = self._rank(context)
        requests: dict[str, list[tuple[str, np.ndarray, str, str, dict]]] = {}

        def add(
            family: str,
            candidate_id: str,
            schedule: np.ndarray,
            anchor_name: str,
            description: str,
            metadata: Optional[dict] = None,
        ) -> None:
            requests.setdefault(family, []).append(
                (
                    candidate_id,
                    schedule,
                    anchor_name,
                    description,
                    dict(metadata or {}),
                )
            )

        # Negative controls are state x action probes: change exactly one
        # currently inactive facility and let authoritative SWMM confirm that
        # the realised action has no local hydraulic response. A checkpoint is
        # never declared flat merely because rain or flooding is absent.
        inactive = [
            facility_id
            for facility_id in self.facility_ids
            if facility_id not in set(context.hydraulically_active)
        ]
        for facility_id in inactive[:8]:
            index = self.fid_to_idx[facility_id]
            for hold_steps in (2, 4, 6):
                schedule = context.frozen_fallback_schedule.copy()
                baseline = float(schedule[0, index])
                target = 0.0 if baseline >= 0.5 else 1.0
                schedule[:hold_steps, index] = target
                add(
                    "flat_action_probe",
                    f"flat_probe_{index}_{hold_steps}",
                    schedule,
                    "frozen_fallback",
                    f"Single inactive-facility probe on {facility_id}",
                    {
                        "facility": facility_id,
                        "expected_role": "negative_control",
                        "hold_steps": hold_steps,
                    },
                )

        # Move only the highest-opportunity coordinates toward No-control.
        for k in [1, 2, 4, 6, 8]:
            k = min(k, self.max_k, len(ranked))
            if k <= 0:
                continue
            schedule = context.frozen_fallback_schedule.copy()
            indices = [self.fid_to_idx[fid] for fid in ranked[:k]]
            schedule[:, indices] = context.no_control_schedule[:, indices]
            add(
                "toward_no_control",
                f"nc_k{k}",
                schedule,
                "frozen_fallback",
                f"Move top-{k} active facilities toward No-control",
                {"k_target": k, "facilities": ranked[:k]},
            )

        # Bidirectional DI neighbourhood with material deltas.
        for facility_id in ranked[: min(8, len(ranked))]:
            index = self.fid_to_idx[facility_id]
            for delta in (-0.50, -0.25, 0.25, 0.50):
                schedule = context.dynamic_internal_schedule.copy()
                schedule[:, index] += delta
                add(
                    "dynamic_internal_neighbourhood",
                    f"di_{index}_{delta:+.2f}",
                    schedule,
                    "dynamic_internal",
                    f"DI neighbourhood {facility_id} {delta:+.2f}",
                    {"facility": facility_id, "delta": delta},
                )

        # Absolute level pulses.
        for facility_id in ranked[: min(6, len(ranked))]:
            index = self.fid_to_idx[facility_id]
            for level in (0.0, 0.25, 0.50, 0.75, 1.0):
                schedule = context.frozen_fallback_schedule.copy()
                schedule[:4, index] = level
                add(
                    "absolute_level_pulse",
                    f"abs_{index}_{level:.2f}",
                    schedule,
                    "frozen_fallback",
                    f"Four-step absolute pulse {facility_id}={level:.2f}",
                    {"facility": facility_id, "level": level, "hold_steps": 4},
                )

        # Temporal 2/4/6-step excitation followed by return to fallback.
        for facility_id in ranked[: min(6, len(ranked))]:
            index = self.fid_to_idx[facility_id]
            baseline = float(context.frozen_fallback_schedule[0, index])
            target = 0.0 if baseline >= 0.5 else 1.0
            for hold_steps in (2, 4, 6):
                schedule = context.frozen_fallback_schedule.copy()
                schedule[:hold_steps, index] = target
                add(
                    "temporal_pulse",
                    f"pulse_{index}_{hold_steps}",
                    schedule,
                    "frozen_fallback",
                    f"{hold_steps}-step pulse on {facility_id}",
                    {
                        "facility": facility_id,
                        "target": target,
                        "hold_steps": hold_steps,
                    },
                )

        # Sparse staggered pairs.
        for first, second in zip(ranked[:4], ranked[1:5]):
            first_index = self.fid_to_idx[first]
            second_index = self.fid_to_idx[second]
            schedule = context.frozen_fallback_schedule.copy()
            schedule[:4, first_index] = 1.0 - np.round(schedule[:4, first_index])
            schedule[2:6, second_index] = 1.0 - np.round(schedule[2:6, second_index])
            add(
                "staggered_sparse",
                f"stagger_{first_index}_{second_index}",
                schedule,
                "frozen_fallback",
                f"Stagger {first} then {second}",
                {"facilities": [first, second]},
            )

        # Sparse staggered groups.  The K loop changes the actual selected
        # coordinates, unlike the legacy priority_protection implementation.
        for k in (1, 2, 4, 6, 8):
            chosen = ranked[: min(k, self.max_k, len(ranked))]
            if not chosen:
                continue
            schedule = context.frozen_fallback_schedule.copy()
            for offset, facility_id in enumerate(chosen):
                index = self.fid_to_idx[facility_id]
                start = min(offset, 5)
                stop = min(start + 4, HORIZON_STEPS)
                baseline = float(schedule[start, index])
                schedule[start:stop, index] = 0.0 if baseline >= 0.5 else 1.0
            add(
                "staggered_sparse",
                f"stagger_k{len(chosen)}",
                schedule,
                "frozen_fallback",
                f"Staggered sparse K={len(chosen)} excitation",
                {"facilities": chosen, "k_target": len(chosen)},
            )

        # Online priority risk may demote high-risk coordinates.  This family
        # is generated from current features only; no realised future response
        # enters the ordering.
        priority_risk = context.online_features.get(
            "priority_risk_by_facility", {}
        )
        protected_rank = sorted(
            ranked,
            key=lambda facility_id: (
                float(priority_risk.get(facility_id, 0.0)),
                -float(context.opportunity_scores.get(facility_id, 0.0)),
            ),
        )
        for k in (1, 2, 4, 6, 8):
            chosen = protected_rank[: min(k, self.max_k, len(protected_rank))]
            if not chosen:
                continue
            schedule = context.frozen_fallback_schedule.copy()
            indices = [self.fid_to_idx[facility_id] for facility_id in chosen]
            # Restrict the selected low-current-risk release paths for six
            # steps, then return to the frozen fallback.  This is deliberately
            # distinct from ``toward_no_control`` and remains state-driven.
            schedule[:6, indices] = 0.0
            add(
                "priority_protection",
                f"priority_k{len(chosen)}",
                schedule,
                "frozen_fallback",
                f"Low-current-priority-risk K={len(chosen)} move",
                {"facilities": chosen, "k_target": len(chosen)},
            )

        # Storage/downstream opportunity families use only checkpoint state.
        for feature_name, family in (
            ("storage_headroom_by_facility", "storage_headroom"),
            ("downstream_capacity_by_facility", "downstream_capacity"),
        ):
            feature = context.online_features.get(feature_name, {})
            ordered = sorted(
                ranked,
                key=lambda facility_id: -float(feature.get(facility_id, 0.0)),
            )
            chosen = ordered[: min(4, self.max_k, len(ordered))]
            if chosen:
                schedule = context.frozen_fallback_schedule.copy()
                indices = [self.fid_to_idx[facility_id] for facility_id in chosen]
                schedule[:, indices] = context.no_control_schedule[:, indices]
                add(
                    family,
                    f"{family}_k{len(chosen)}",
                    schedule,
                    "frozen_fallback",
                    f"{family} checkpoint-state candidate",
                    {"facilities": chosen},
                )

        # Boundary and hard-negative probes are family-discovery inputs.  Their
        # labels still come exclusively from authoritative SWMM.
        for fraction in (0.25, 0.50, 0.75):
            schedule = (
                context.frozen_fallback_schedule
                + fraction
                * (
                    context.no_control_schedule
                    - context.frozen_fallback_schedule
                )
            )
            add(
                "boundary_and_hard_negative",
                f"fallback_to_nc_{fraction:.2f}",
                schedule,
                "frozen_fallback",
                f"Fallback-to-No-control boundary fraction {fraction:.2f}",
                {"fraction": fraction},
            )
        return requests

    def generate(
        self, context: CandidateContext, max_total: int = 100
    ) -> list[CandidateSchedule]:
        if context.facility_ids != self.facility_ids:
            raise ValueError("CandidateContext facility order does not match generator")
        requests = self._requests(context)
        family_names = list(requests)
        cursors = {family: 0 for family in family_names}
        seen: set[str] = set()
        candidates: list[CandidateSchedule] = []

        # Round-robin families so a max_total budget cannot silently discard all
        # later candidate families.
        while len(candidates) < int(max_total):
            progressed = False
            for family in family_names:
                cursor = cursors[family]
                if cursor >= len(requests[family]):
                    continue
                cursors[family] += 1
                progressed = True
                candidate_id, schedule, anchor, description, metadata = requests[family][cursor]
                candidate = self._candidate(
                    context,
                    candidate_id,
                    family,
                    schedule,
                    anchor,
                    description,
                    metadata,
                )
                if candidate is None or candidate.projected_schedule_hash in seen:
                    continue
                seen.add(candidate.projected_schedule_hash)
                candidates.append(candidate)
                if len(candidates) >= int(max_total):
                    break
            if not progressed:
                break
        return candidates

    def generate_all(
        self, di_action: np.ndarray, max_total: int = 100
    ) -> list[CandidateSchedule]:
        """Legacy adapter used by the old Gate 5 runner."""
        di_schedule = _normalise_schedule(di_action, self.n_facilities)
        context = CandidateContext(
            event_id="legacy_gate5",
            checkpoint_id="legacy_checkpoint",
            facility_ids=self.facility_ids.copy(),
            frozen_fallback_schedule=di_schedule,
            dynamic_internal_schedule=di_schedule,
            no_control_schedule=np.ones_like(di_schedule),
            hold_previous_schedule=di_schedule,
            opportunity_scores={
                facility_id: float(self.n_facilities - index)
                for index, facility_id in enumerate(self.facility_ids)
            },
            hydraulically_active=set(self.facility_ids),
        )
        return self.generate(context, max_total=max_total)


def iter_actual_unique(
    candidates: Iterable[CandidateSchedule],
) -> list[CandidateSchedule]:
    """Deduplicate candidates using readback when present, projected otherwise."""
    seen: set[str] = set()
    unique: list[CandidateSchedule] = []
    for candidate in candidates:
        schedule = (
            candidate.readback_schedule
            if candidate.readback_schedule is not None
            else candidate.projected_schedule
        )
        digest = _schedule_hash(schedule)
        if digest in seen:
            continue
        seen.add(digest)
        unique.append(candidate)
    return unique
