from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.calibrate_v42_formal_step2_safety_f2 import _admission_risk
from sewerrtc.control.pfvfirst_mpc_v42 import EngineeringStatus
from sewerrtc.v4 import v42_pfv_tfv_runtime_patch as runtime_patch
from sewerrtc.v4.v42_pfv_tfv_runtime_patch import (
    _global_tfv_sequences,
    _project_dedupe_and_cap,
)


def test_global_tfv_candidates_cover_assets_outside_priority_domains() -> None:
    actuators = pd.DataFrame(
        {
            "actuator_id": ["ordinary_gate", "ADD301.2", "add350.1"],
        }
    )
    base = np.asarray([0.5, 0.0, 1.0], dtype=np.float32)
    rows = _global_tfv_sequences(base, actuators)
    labels = {str(row["label"]) for row in rows}

    assert any("actuator=ordinary_gate|direction=decrease" in x for x in labels)
    assert any("actuator=ordinary_gate|direction=increase" in x for x in labels)
    assert any("global_binary_toggle|actuator=ADD301.2|target=1" in x for x in labels)
    assert any("global_tfv_single|actuator=add350.1|direction=decrease" in x for x in labels)

    for row in rows:
        sequence = np.asarray(row["sequence"], dtype=float)
        # Only the executable H3 prefix may differ from the current setting.
        assert np.allclose(sequence[3:], base[None, :])


def test_candidate_cap_is_applied_after_projection_and_dedup(monkeypatch) -> None:
    base = np.zeros(2, dtype=np.float32)
    actuators = pd.DataFrame({"actuator_id": ["a", "b"]})

    def fake_project(sequence, current, _actuators):
        projected = np.asarray(sequence, dtype=np.float32).copy()
        projected[3:] = current[None, :]
        changed = int(np.any(np.abs(projected[:3] - current[None, :]) > 1e-7, axis=0).sum())
        return (
            projected,
            EngineeringStatus(True, True, True, True, True),
            changed,
            True,
        )

    monkeypatch.setattr(
        runtime_patch.base_runtime,
        "project_candidate_sequence",
        fake_project,
    )
    hold = np.zeros((12, 2), dtype=np.float32)
    late_pulse = hold.copy()
    late_pulse[6:9, 0] = 1.0  # collapses to hold under H3 execution
    effective = hold.copy()
    effective[:3, 1] = 0.1
    raw = [
        {"label": "hold_native", "sequence": hold},
        {"label": "late_pulse", "sequence": late_pulse},
        {
            "label": "global_tfv_single|actuator=b|direction=increase",
            "sequence": effective,
        },
    ]

    selected, stats = _project_dedupe_and_cap(
        raw,
        base=base,
        actuators=actuators,
        requested_cap=2,
    )
    assert stats["raw_candidate_count"] == 3
    assert stats["projected_unique_candidate_count"] == 2
    assert len(selected) == 2
    assert {row[0] for row in selected} == {
        "hold_native",
        "global_tfv_single|actuator=b|direction=increase",
    }


def test_false_safe_risk_is_measured_among_admitted_candidates() -> None:
    frame = pd.DataFrame(
        {
            "split_group_key": ["rain_a", "rain_a", "rain_b", "rain_b"],
        }
    )
    predicted_safe = np.asarray([True, False, True, False])
    actual_safe = np.asarray([False, True, True, True])
    risk = _admission_risk(frame, predicted_safe, actual_safe)

    assert risk["false_safe_rate_marginal"] == 0.25
    assert risk["false_safe_rate_among_admitted"] == 0.5
    assert risk["event_balanced_false_safe_rate_among_admitted"] == 0.5
    assert risk["predicted_safe_count"] == 2
    assert risk["false_safe_count"] == 1
