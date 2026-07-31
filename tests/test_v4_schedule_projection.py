from __future__ import annotations

import numpy as np
import pandas as pd

from sewerrtc.control.v4_candidate_generator import (
    project_candidate_schedule,
    project_frozen_anchor_schedule,
)


def test_projection_enforces_binary_rate_and_k_against_frozen_anchor() -> None:
    facility_ids = ["a", "b", "ADD301.2"]
    semantics = pd.DataFrame(
        [
            {"facility_id": "a", "binary_or_continuous": "continuous", "lower_bound": 0, "upper_bound": 1, "rate_limit": 0.25, "min_hold_steps": 1},
            {"facility_id": "b", "binary_or_continuous": "continuous", "lower_bound": 0, "upper_bound": 1, "rate_limit": 1.0, "min_hold_steps": 1},
            {"facility_id": "ADD301.2", "binary_or_continuous": "binary", "lower_bound": 0, "upper_bound": 1, "rate_limit": 1.0, "min_hold_steps": 2},
        ]
    )
    anchor = np.zeros((12, 3), dtype=float)
    requested = np.ones((12, 3), dtype=float)

    projected, audit = project_candidate_schedule(
        requested,
        anchor,
        facility_ids,
        semantics,
        max_k=3,
    )

    assert projected[0, 0] == 0.25
    assert set(np.unique(projected[:, 2])).issubset({0.0, 1.0})
    assert max(audit["k_by_step"]) <= 3
    assert audit["binary_ok"]
    assert audit["rate_ok"]


def test_projection_rejects_wrong_shape() -> None:
    semantics = pd.DataFrame([{"facility_id": "a"}])
    try:
        project_candidate_schedule(
            np.zeros((2, 2)),
            np.zeros((12, 1)),
            ["a"],
            semantics,
            max_k=1,
        )
    except ValueError as exc:
        assert "shape" in str(exc)
    else:
        raise AssertionError("shape mismatch was accepted")


def test_projection_enforces_binary_dwell_and_storage_interlock() -> None:
    facility_ids = ["ADD301.2", "storage_in", "storage_out"]
    semantics = pd.DataFrame(
        [
            {
                "facility_id": "ADD301.2",
                "binary_or_continuous": "binary",
                "min_hold_steps": 3,
            },
            {
                "facility_id": "storage_in",
                "storage_role": "storage_inlet",
                "interlock_group": "tank-a",
            },
            {
                "facility_id": "storage_out",
                "storage_role": "storage_outlet",
                "interlock_group": "tank-a",
            },
        ]
    )
    anchor = np.zeros((12, 3), dtype=float)
    requested = np.ones((12, 3), dtype=float)
    requested[1, 0] = 0.0

    projected, audit = project_candidate_schedule(
        requested, anchor, facility_ids, semantics, max_k=3
    )

    assert projected[:3, 0].tolist() == [1.0, 1.0, 1.0]
    assert np.all(projected[:, 1] == 1.0)
    assert np.all(projected[:, 2] == 0.0)
    assert audit["interlock_adjustments"] == 12


def test_frozen_anchor_is_projected_before_candidate_k_is_counted() -> None:
    facility_ids = ["probe", "storage_in", "storage_out"]
    semantics = pd.DataFrame(
        [
            {"facility_id": "probe"},
            {
                "facility_id": "storage_in",
                "storage_role": "storage_inlet",
                "interlock_group": "tank-a",
            },
            {
                "facility_id": "storage_out",
                "storage_role": "storage_outlet",
                "interlock_group": "tank-a",
            },
        ]
    )
    raw_anchor = np.ones((12, 3), dtype=float)

    anchor, anchor_audit = project_frozen_anchor_schedule(
        raw_anchor, facility_ids, semantics, max_k=8
    )
    requested = anchor.copy()
    requested[:2, 0] = 0.0
    projected, candidate_audit = project_candidate_schedule(
        requested, anchor, facility_ids, semantics, max_k=8
    )

    assert np.all(anchor[:, 2] == 0.0)
    assert anchor_audit["max_k"] == 1
    assert max(candidate_audit["k_by_step"]) == 1
    assert np.all(projected[:, 2] == anchor[:, 2])


def test_projection_prevents_direction_reversal_relative_to_anchor() -> None:
    semantics = pd.DataFrame(
        [
            {
                "facility_id": "a",
                "binary_or_continuous": "continuous",
                "lower_bound": 0.0,
                "upper_bound": 1.0,
                "rate_limit": 1.0,
                "min_hold_steps": 1,
                "no_reversal": True,
            }
        ]
    )
    anchor = np.full((12, 1), 0.5)
    requested = anchor.copy()
    requested[:4, 0] = 0.75
    requested[4:, 0] = 0.25

    projected, audit = project_candidate_schedule(
        requested, anchor, ["a"], semantics, max_k=1
    )

    assert np.all(projected[:4, 0] >= 0.5)
    assert np.all(projected[4:, 0] == 0.5)
    assert audit["no_reversal_ok"]
