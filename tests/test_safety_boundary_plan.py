from __future__ import annotations

import numpy as np


def test_reference_load_selection_is_event_grouped_and_split_preserving():
    from sewerrtc.experiments.safety_boundary_plan import select_events_by_reference_load

    event_ids = np.asarray(["T1", "T1", "T2", "V1", "V2"])
    splits = np.asarray(["train", "train", "train", "validation", "validation"])
    reference_risk = np.zeros((5, 6, 3), dtype=np.float32)
    reference_risk[0, :, 0] = 1.0
    reference_risk[1, :, 0] = 3.0
    reference_risk[2, :, 0] = 2.0
    reference_risk[3, :, 0] = 4.0
    reference_risk[4, :, 0] = 1.0

    selected = select_events_by_reference_load(
        event_ids=event_ids,
        splits=splits,
        reference_risk_rate_seq=reference_risk,
        train_count=1,
        validation_count=1,
        dt_sec=300,
    )

    assert selected["train"] == ["T1"]
    assert selected["validation"] == ["V1"]


def test_boundary_case_slots_allocate_40_validation_and_32_training_cases():
    from sewerrtc.experiments.safety_boundary_plan import build_boundary_case_slots

    selected = {
        "train": [f"T{i}" for i in range(8)],
        "validation": [f"V{i}" for i in range(8)],
    }
    slots = build_boundary_case_slots(selected)

    assert len(slots) == 72
    assert sum(slot["split"] == "validation" for slot in slots) == 40
    assert sum(slot["split"] == "train" for slot in slots) == 32
    assert {slot["phase"] for slot in slots} == {"peak", "recession"}


def test_round2_slots_add_two_new_recession_checkpoints_per_selected_event():
    from sewerrtc.experiments.safety_boundary_plan import build_boundary_round2_slots

    selected = {
        "train": [f"T{i}" for i in range(8)],
        "validation": [f"V{i}" for i in range(8)],
    }
    slots = build_boundary_round2_slots(selected, recession_offsets_min=(15.0, 60.0))

    assert len(slots) == 32
    assert sum(slot["split"] == "validation" for slot in slots) == 16
    assert sum(slot["split"] == "train" for slot in slots) == 16
    assert {slot["recession_offset_min"] for slot in slots} == {15.0, 60.0}
    assert {slot["phase"] for slot in slots} == {"recession"}
