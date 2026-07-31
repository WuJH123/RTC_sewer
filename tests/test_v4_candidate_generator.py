from __future__ import annotations

import numpy as np
import pandas as pd

from sewerrtc.control.v4_candidate_generator import (
    CandidateContext,
    V4CandidateGenerator,
)
from sewerrtc.v4.candidates import CANDIDATE_FAMILIES


def _semantics() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "facility_id": "gate_a",
                "binary_or_continuous": "continuous",
                "lower_bound": 0.0,
                "upper_bound": 1.0,
                "rate_limit": 0.5,
                "min_hold_steps": 1,
                "storage_role": "none",
                "interlock_group": "",
            },
            {
                "facility_id": "gate_b",
                "binary_or_continuous": "continuous",
                "lower_bound": 0.0,
                "upper_bound": 1.0,
                "rate_limit": 0.5,
                "min_hold_steps": 1,
                "storage_role": "none",
                "interlock_group": "",
            },
            {
                "facility_id": "add350.1",
                "binary_or_continuous": "continuous",
                "lower_bound": 0.0,
                "upper_bound": 1.0,
                "rate_limit": 0.5,
                "min_hold_steps": 1,
                "storage_role": "none",
                "interlock_group": "",
            },
            {
                "facility_id": "ADD301.2",
                "binary_or_continuous": "binary",
                "lower_bound": 0.0,
                "upper_bound": 1.0,
                "rate_limit": 1.0,
                "min_hold_steps": 2,
                "storage_role": "none",
                "interlock_group": "",
            },
        ]
    )


def _context() -> CandidateContext:
    anchor = np.full((12, 4), 0.5, dtype=float)
    anchor[:, 3] = 0.0
    di = anchor.copy()
    di[:, 0] = 1.0
    no_control = np.ones((12, 4), dtype=float)
    return CandidateContext(
        event_id="event-a",
        checkpoint_id="cp-1",
        facility_ids=["gate_a", "gate_b", "add350.1", "ADD301.2"],
        frozen_fallback_schedule=anchor,
        dynamic_internal_schedule=di,
        no_control_schedule=no_control,
        hold_previous_schedule=anchor,
        opportunity_scores={"gate_a": 0.9, "gate_b": 0.8, "add350.1": 0.7, "ADD301.2": 0.6},
        hydraulically_active={"gate_a", "gate_b", "add350.1", "ADD301.2"},
        event_phase="pre_peak",
    )


def test_generator_builds_twelve_step_multi_anchor_candidates() -> None:
    generator = V4CandidateGenerator(
        facility_ids=_context().facility_ids,
        facility_semantics=_semantics(),
        priority_nodes=[],
        max_k=2,
    )

    candidates = generator.generate(_context(), max_total=40)

    assert candidates
    assert all(candidate.requested_schedule.shape == (12, 4) for candidate in candidates)
    assert all(candidate.projected_schedule.shape == (12, 4) for candidate in candidates)
    assert {candidate.anchor_name for candidate in candidates} >= {
        "frozen_fallback",
        "dynamic_internal",
    }
    families = {candidate.family for candidate in candidates}
    assert "toward_no_control" in families
    assert "dynamic_internal_neighbourhood" in families
    assert "absolute_level_pulse" in families
    assert "temporal_pulse" in families
    assert "staggered_sparse" in families
    assert "priority_protection" in families
    assert "boundary_and_hard_negative" in families


def test_generator_deduplicates_projected_schedules_and_enforces_k() -> None:
    generator = V4CandidateGenerator(
        facility_ids=_context().facility_ids,
        facility_semantics=_semantics(),
        priority_nodes=[],
        max_k=2,
    )

    candidates = generator.generate(_context(), max_total=100)

    hashes = [candidate.projected_schedule_hash for candidate in candidates]
    assert len(hashes) == len(set(hashes))
    assert all(max(candidate.k_by_step) <= 2 for candidate in candidates)


def test_variable_speed_pump_is_not_binarised() -> None:
    generator = V4CandidateGenerator(
        facility_ids=_context().facility_ids,
        facility_semantics=_semantics(),
        priority_nodes=[],
        max_k=2,
    )

    candidates = generator.generate(_context(), max_total=100)
    vsp_index = _context().facility_ids.index("add350.1")
    values = {
        float(value)
        for candidate in candidates
        for value in candidate.projected_schedule[:, vsp_index]
    }

    assert any(value not in {0.0, 1.0} for value in values)


def test_generator_creates_single_inactive_facility_flat_action_probes() -> None:
    context = _context()
    context.hydraulically_active = {"gate_a", "gate_b"}
    generator = V4CandidateGenerator(
        facility_ids=context.facility_ids,
        facility_semantics=_semantics(),
        priority_nodes=[],
        max_k=2,
    )

    candidates = generator.generate(context, max_total=100)
    probes = [
        candidate for candidate in candidates
        if candidate.family == "flat_action_probe"
    ]

    assert probes
    assert all(candidate.k_actual == 1 for candidate in probes)
    assert {
        candidate.metadata["facility"] for candidate in probes
    }.issubset({"add350.1", "ADD301.2"})


def test_final_generator_contract_registers_all_sixteen_candidate_families() -> None:
    assert len(CANDIDATE_FAMILIES) == 16
    assert {
        "peak_boundary",
        "PFV_hard_negative",
        "TFV_hard_negative",
        "Peak_hard_negative",
        "flat_control",
        "uncertainty_or_coverage_gap",
    }.issubset(CANDIDATE_FAMILIES)
