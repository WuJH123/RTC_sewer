from __future__ import annotations

import numpy as np

from sewerrtc.control.targeted_candidate_expansion_v42 import (
    BINARY_ACTUATOR_IDS,
    CandidateExpansionConfig,
    generate_targeted_candidate_sequences,
)


def _fixture():
    ids = ["ADD301.2", "ADD301.3", "PUMP_A", "ORIFICE_A", "STORAGE_OUT"]
    current = np.asarray([0.0, 1.0, 0.4, 0.7, 0.5], dtype=np.float32)
    roles = {
        "ADD301.2": "pump",
        "ADD301.3": "pump",
        "PUMP_A": "pump",
        "ORIFICE_A": "orifice",
        "STORAGE_OUT": "storage_outlet",
    }
    return ids, current, roles


def test_candidate_expansion_preserves_h3_tail_and_binary_semantics() -> None:
    ids, current, roles = _fixture()
    candidates = generate_targeted_candidate_sequences(
        current_action=current,
        actuator_ids=ids,
        actuator_roles=roles,
        ranked_actuator_ids=ids,
        config=CandidateExpansionConfig(max_candidates=256),
    )
    assert candidates
    assert candidates[0]["label"] == "hold_native"
    assert any(item["candidate_family"] == "coordinated_pair" for item in candidates)
    assert any(item["candidate_family"] == "coordinated_quad" for item in candidates)

    for item in candidates:
        sequence = np.asarray(item["sequence"], dtype=float)
        assert sequence.shape == (12, len(ids))
        assert np.all(sequence >= 0.0)
        assert np.all(sequence <= 1.0)
        assert np.allclose(sequence[3:], current[None, :])
        for aid in BINARY_ACTUATOR_IDS:
            idx = ids.index(aid)
            assert set(np.unique(sequence[:, idx])).issubset({0.0, 1.0})


def test_candidate_expansion_is_deterministic_and_deduplicated() -> None:
    ids, current, roles = _fixture()
    kwargs = dict(
        current_action=current,
        actuator_ids=ids,
        actuator_roles=roles,
        ranked_actuator_ids=list(reversed(ids)),
        config=CandidateExpansionConfig(max_candidates=120),
    )
    first = generate_targeted_candidate_sequences(**kwargs)
    second = generate_targeted_candidate_sequences(**kwargs)
    first_keys = [np.round(np.asarray(item["sequence"]), 6).tobytes() for item in first]
    second_keys = [np.round(np.asarray(item["sequence"]), 6).tobytes() for item in second]
    assert first_keys == second_keys
    assert len(first_keys) == len(set(first_keys))
    assert len(first) <= 120


def test_positive_control_neighbourhood_is_included() -> None:
    ids, current, roles = _fixture()
    candidates = generate_targeted_candidate_sequences(
        current_action=current,
        actuator_ids=ids,
        actuator_roles=roles,
        successful_action_templates=[
            {
                "actuator_ids": ["PUMP_A", "ORIFICE_A"],
                "deltas": [0.10, -0.05],
                "profile": "constant_h3",
            }
        ],
        config=CandidateExpansionConfig(max_candidates=384),
    )
    neighbours = [
        item for item in candidates
        if item["candidate_family"] == "positive_control_neighbour"
    ]
    assert len(neighbours) == 3
    assert all(item["changed_facilities"] == 2 for item in neighbours)
