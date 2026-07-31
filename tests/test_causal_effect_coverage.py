from __future__ import annotations

import numpy as np


def test_coverage_counts_independent_events_by_actuator_direction_phase_and_split():
    from sewerrtc.experiments.causal_effect_coverage import summarize_effect_coverage

    reference = np.full((3, 6, 2), 0.5, dtype=np.float32)
    candidate = reference.copy()
    candidate[0, 1:4, 0] -= 0.1
    candidate[1, 1:4, 0] -= 0.2
    candidate[2, 2:5, 1] += 0.1

    coverage = summarize_effect_coverage(
        event_ids=np.asarray(["E1", "E2", "E3"]),
        splits=np.asarray(["train", "train", "validation"]),
        phases=np.asarray(["rising", "rising", "peak"]),
        candidate_action_seq=candidate,
        reference_action_seq=reference,
        action_ids=["A", "B"],
    )

    row = coverage.query("actuator_id == 'A' and direction == 'decrease' and phase == 'rising'").iloc[0]
    assert row["independent_events"] == 2
    assert row["rows"] == 2
    row = coverage.query("actuator_id == 'B' and direction == 'increase' and phase == 'peak'").iloc[0]
    assert row["split"] == "validation"


def test_coverage_gap_plan_is_cell_based_and_does_not_treat_rows_as_events():
    from sewerrtc.experiments.causal_effect_coverage import build_coverage_gaps

    existing = [
        {"actuator_id": "A", "direction": "increase", "phase": "peak", "split": "train", "independent_events": 2},
        {"actuator_id": "A", "direction": "increase", "phase": "peak", "split": "validation", "independent_events": 1},
    ]
    gaps = build_coverage_gaps(
        existing,
        action_ids=["A"],
        phases=["rising", "peak", "recession"],
        min_train_events=3,
        min_validation_events=2,
    )

    peak_train = next(row for row in gaps if row["phase"] == "peak" and row["split"] == "train" and row["direction"] == "increase")
    peak_validation = next(row for row in gaps if row["phase"] == "peak" and row["split"] == "validation" and row["direction"] == "increase")
    assert peak_train["missing_events"] == 1
    assert peak_validation["missing_events"] == 1


def test_phase_profile_retains_direction_and_temporal_change_points():
    from sewerrtc.experiments.causal_effect_coverage import phase_delta_profile

    rising = phase_delta_profile("increase", 0.2, "rising", horizon_steps=6)
    recession = phase_delta_profile("increase", 0.2, "recession", horizon_steps=6)

    assert rising.shape == (6,)
    assert np.max(rising) == 0.2
    assert not np.array_equal(rising, recession)
    assert np.count_nonzero(np.diff(rising)) >= 1


def test_phase_profile_library_covers_magnitude_and_timing_without_noops():
    from sewerrtc.experiments.causal_effect_coverage import build_phase_profile_library

    profiles = build_phase_profile_library(
        "decrease",
        magnitudes=(0.05, 0.10, 0.20),
        phase="peak",
        horizon_steps=6,
    )

    assert {item["magnitude"] for item in profiles} == {0.05, 0.10, 0.20}
    assert {item["variant"] for item in profiles} == {"phase_hold", "delayed_restore"}
    assert len({np.asarray(item["profile"]).tobytes() for item in profiles}) == 6
    assert all(np.max(np.abs(item["profile"])) >= item["magnitude"] - 1.0e-6 for item in profiles)
    assert all(np.any(np.asarray(item["profile"]) < 0.0) for item in profiles)
