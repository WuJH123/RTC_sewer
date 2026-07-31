from __future__ import annotations

import numpy as np


def test_materialize_candidate_detects_clipped_noop():
    from sewerrtc.experiments.targeted_joint_pairs import materialize_candidate, sequence_diagnostics

    reference = np.ones((6, 2), dtype=np.float32)
    spec = {"actuators": ["A"], "signed_profile": [0.2] * 6}
    candidate = materialize_candidate(reference, action_ids=["A", "B"], specification=spec)
    audit = sequence_diagnostics(candidate, reference, action_ids=["A", "B"])

    assert np.array_equal(candidate, reference)
    assert audit["is_noop"] is True
    assert audit["valid"] is False


def test_materialize_candidate_retains_independent_time_and_actuator_axes():
    from sewerrtc.experiments.targeted_joint_pairs import materialize_candidate, sequence_diagnostics

    reference = np.full((6, 3), 0.5, dtype=np.float32)
    spec = {
        "signed_profiles": {
            "A": [0.0, 0.1, 0.0, 0.0, -0.1, 0.0],
            "B": [0.0, 0.1, 0.1, 0.0, 0.0, 0.0],
        }
    }
    candidate = materialize_candidate(reference, action_ids=["A", "B", "C"], specification=spec)
    audit = sequence_diagnostics(candidate, reference, action_ids=["A", "B", "C"])

    assert candidate.shape == (6, 3)
    assert audit["changed_actuator_count"] == 2
    assert audit["max_simultaneous_changes"] == 2
    assert audit["changed_time_step_count"] == 3


def test_binary_pump_rejects_fractional_target():
    from sewerrtc.experiments.targeted_joint_pairs import materialize_candidate, sequence_diagnostics

    reference = np.zeros((6, 1), dtype=np.float32)
    candidate = materialize_candidate(
        reference,
        action_ids=["P"],
        specification={"actuators": ["P"], "target_profile": [0.5] * 6},
    )
    audit = sequence_diagnostics(
        candidate,
        reference,
        action_ids=["P"],
        binary_pump_ids={"P"},
    )

    assert audit["valid"] is False
    assert audit["reason"].startswith("fractional_binary_pump")


def test_off_grid_override_uses_consecutive_causal_sequence_tokens():
    from sewerrtc.simulation.pyswmm_runner import _causal_override_sequence_step

    steps = [
        _causal_override_sequence_step(
            elapsed_min=elapsed,
            override_start_min=82.5,
            control_step_sec=300,
        )
        for elapsed in (85.0, 90.0, 95.0, 100.0, 105.0, 110.0)
    ]

    assert steps == [0, 1, 2, 3, 4, 5]
