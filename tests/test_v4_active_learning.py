import pandas as pd

from sewerrtc.v4.active_learning import select_active_learning_cases


def test_active_learning_selects_coverage_and_uncertainty_without_checkpoint_density() -> None:
    candidates = pd.DataFrame(
        {
            "case_id": [f"x{i}" for i in range(20)],
            "event_id": ["e"] * 20,
            "checkpoint_id": ["c1"] * 10 + ["c2"] * 10,
            "uncertainty": list(range(20)),
            "coverage_gap": [0.0, 1.0] * 10,
            "boundary_distance": [float(i % 5) for i in range(20)],
        }
    )
    selected = select_active_learning_cases(candidates, limit=6, per_checkpoint=3)

    assert len(selected) == 6
    assert selected.groupby(["event_id", "checkpoint_id"]).size().le(3).all()
