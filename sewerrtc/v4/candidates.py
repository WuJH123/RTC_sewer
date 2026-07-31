from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from sewerrtc.control.v4_candidate_generator import (
    CandidateContext,
    CandidateSchedule,
    V4CandidateGenerator,
    project_candidate_schedule,
)


CANDIDATE_FAMILIES = (
    "dynamic_internal_neighbourhood",
    "toward_no_control",
    "hold_neighbourhood",
    "absolute_level_pulse",
    "temporal_pulse",
    "staggered_sparse",
    "priority_protection",
    "storage_headroom",
    "downstream_capacity",
    "peak_boundary",
    "PFV_hard_negative",
    "TFV_hard_negative",
    "Peak_hard_negative",
    "neutral_anchor",
    "flat_control",
    "uncertainty_or_coverage_gap",
)


def validate_schedule(schedule: np.ndarray, facilities: int = 36) -> None:
    values = np.asarray(schedule, dtype=float)
    if values.shape != (12, int(facilities)):
        raise ValueError(f"schedule must be 12x{facilities}, got {values.shape}")
    if not np.isfinite(values).all():
        raise ValueError("schedule contains non-finite values")


@dataclass
class ScheduleProjector:
    """Thin Final-V4 facade over the existing authoritative projection code."""

    facility_ids: list[str]
    facility_semantics: object
    max_k: int = 8

    def project(
        self, requested_schedule: np.ndarray, frozen_anchor_schedule: np.ndarray
    ) -> tuple[np.ndarray, dict]:
        validate_schedule(requested_schedule, len(self.facility_ids))
        validate_schedule(frozen_anchor_schedule, len(self.facility_ids))
        return project_candidate_schedule(
            requested_schedule,
            frozen_anchor_schedule,
            self.facility_ids,
            self.facility_semantics,
            max_k=self.max_k,
        )


class FinalCandidateGenerator:
    """Unified online-safe candidate facade for the Final V4 pipeline.

    Exact/Oracle results may only enter through a pre-frozen anchor library.
    The library stores action schedules, never realised future hydraulics.
    """

    def __init__(
        self,
        facility_ids,
        facility_semantics,
        priority_nodes,
        *,
        sensitivity_map=None,
        max_k: int = 8,
    ) -> None:
        self.base = V4CandidateGenerator(
            list(facility_ids),
            facility_semantics,
            list(priority_nodes),
            sensitivity_map=sensitivity_map,
            max_k=max_k,
        )
        self.projector = ScheduleProjector(
            list(facility_ids), facility_semantics, max_k=max_k
        )
        self.facility_ids = list(facility_ids)

    def _from_frozen_anchor(
        self, family: str, schedule, context: CandidateContext
    ) -> CandidateSchedule:
        requested = np.asarray(schedule, dtype=float)
        projected, audit = self.projector.project(
            requested, context.frozen_fallback_schedule
        )
        digest = hashlib.sha256(
            np.round(projected, 8).astype(np.float64).tobytes()
        ).hexdigest()
        return CandidateSchedule(
            candidate_id=f"{family}_{digest[:12]}",
            family=family,
            facility_ids=self.facility_ids,
            requested_schedule=requested,
            projected_schedule=projected,
            anchor_name="frozen_anchor_library",
            description=f"Frozen offline-discovered {family} schedule",
            constraint_audit=audit,
            metadata={"offline_anchor": True, "future_hydraulic_features": False},
        )

    def generate(
        self, context: CandidateContext, max_total: int = 100
    ) -> list[CandidateSchedule]:
        candidates = self.base.generate(context, max_total=max_total * 2)
        aliases = {
            "flat_action_probe": "flat_control",
            "boundary_and_hard_negative": "peak_boundary",
        }
        for candidate in candidates:
            candidate.family = aliases.get(candidate.family, candidate.family)

        frozen_library = context.online_features.get(
            "frozen_anchor_library", {}
        )
        for family in (
            "PFV_hard_negative",
            "TFV_hard_negative",
            "Peak_hard_negative",
            "neutral_anchor",
            "uncertainty_or_coverage_gap",
        ):
            for schedule in frozen_library.get(family, []):
                candidates.append(
                    self._from_frozen_anchor(family, schedule, context)
                )

        # Hold-neighbourhood actions are online-derived and do not need exact
        # future evidence.
        active = [
            facility_id
            for facility_id in context.facility_ids
            if facility_id in context.hydraulically_active
        ]
        for facility_id in active[:2]:
            index = context.facility_ids.index(facility_id)
            for delta in (-0.25, 0.25):
                requested = context.hold_previous_schedule.copy()
                requested[:4, index] += delta
                candidate = self._from_frozen_anchor(
                    "hold_neighbourhood", requested, context
                )
                candidates.append(candidate)

        unique: list[CandidateSchedule] = []
        hashes: set[str] = set()
        for candidate in candidates:
            if candidate.projected_schedule_hash in hashes:
                continue
            hashes.add(candidate.projected_schedule_hash)
            unique.append(candidate)
            if len(unique) >= int(max_total):
                break
        return unique


__all__ = [
    "CANDIDATE_FAMILIES",
    "CandidateContext",
    "CandidateSchedule",
    "ScheduleProjector",
    "V4CandidateGenerator",
    "FinalCandidateGenerator",
    "validate_schedule",
]
