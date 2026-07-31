from __future__ import annotations

import numpy as np

from sewerrtc.experiments.informative_effect_supplement import (
    build_boundary_v4_specifications,
    build_boundary_v5_specifications,
    combine_residual_candidates,
    scale_residual_candidate,
)


def _reference() -> np.ndarray:
    reference = np.ones((6, 36), dtype=np.float32)
    reference[:, 31] = 0.0
    reference[:, 34:36] = 0.0
    return reference


def test_scale_residual_candidate_preserves_shape_and_clips_settings() -> None:
    reference = _reference()
    candidate = reference.copy()
    candidate[1:5, 2] = 0.6

    scaled = scale_residual_candidate(reference, candidate, scale=3.0)

    assert scaled.shape == (6, 36)
    assert np.all((scaled >= 0.0) & (scaled <= 1.0))
    np.testing.assert_allclose(scaled[1:5, 2], 0.0)
    np.testing.assert_allclose(scaled[[0, 5], 2], 1.0)


def test_combine_residual_candidates_keeps_independent_temporal_changes() -> None:
    reference = _reference()
    first = reference.copy()
    second = reference.copy()
    first[1:3, 4] = 0.5
    second[3:5, 7] = 0.25

    combined = combine_residual_candidates(reference, [first, second], max_changed_actuators=8)

    np.testing.assert_allclose(combined[1:3, 4], 0.5)
    np.testing.assert_allclose(combined[3:5, 7], 0.25)
    assert np.any(np.abs(combined - reference) > 1.0e-6)


def test_combine_residual_candidates_rejects_excessive_joint_action() -> None:
    reference = _reference()
    candidates = []
    for actuator_index in range(9):
        candidate = reference.copy()
        candidate[1:5, actuator_index] = 0.5
        candidates.append(candidate)

    try:
        combine_residual_candidates(reference, candidates, max_changed_actuators=8)
    except ValueError as exc:
        assert "max_changed_actuators" in str(exc)
    else:
        raise AssertionError("expected the nine-actuator candidate to be rejected")


def test_boundary_v4_specs_are_sparse_temporal_and_include_binary_pumps() -> None:
    specifications = build_boundary_v4_specifications("peak")

    assert len(specifications) == 3
    assert all(len(specification["actuators"]) <= 8 for specification in specifications)
    assert all(specification["horizon_steps"] == 6 for specification in specifications)
    pump_specification = next(item for item in specifications if item["mode"] == "pump_storage_peak_stress")
    assert pump_specification["target_profiles"]["ADD301.2"] == [0.0, 1.0, 1.0, 1.0, 1.0, 0.0]
    assert pump_specification["target_profiles"]["ADD301.3"] == [0.0, 1.0, 1.0, 1.0, 1.0, 0.0]


def test_boundary_v5_specs_are_offline_only_and_add_new_strength_timing_contrasts() -> None:
    specifications = build_boundary_v5_specifications("recession")

    assert len(specifications) == 5
    assert all(len(specification["actuators"]) <= 8 for specification in specifications)
    assert all(specification["intended_evidence_role"] == "offline_safety_rejection_only" for specification in specifications)
    assert all(specification["online_candidate_eligible"] is False for specification in specifications)
    assert {specification["stress_magnitude"] for specification in specifications} == {0.85, 1.0}
    assert any(specification["stress_profile"] == "hold_through_horizon" for specification in specifications)
    pump_specification = next(item for item in specifications if item["mode"].startswith("pump_storage"))
    assert set(pump_specification["target_profiles"]) == {"ADD301.2", "ADD301.3"}
    assert set(pump_specification["target_profiles"]["ADD301.2"]) <= {0.0, 1.0}
