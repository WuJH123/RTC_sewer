from __future__ import annotations

import json

import numpy as np
import pandas as pd

from sewerrtc.control.experience_bank_v42 import (
    AuthoritativeExperienceBank,
    ExperienceRetrievalConfig,
    encode_sequence,
    encode_signature,
    state_signature,
)


def test_state_signature_is_causal_and_deterministic() -> None:
    history = np.arange(13 * 5, dtype=np.float32).reshape(13, 5) / 100.0
    rain = np.linspace(0.0, 10.0, 12, dtype=np.float32)
    action = np.linspace(0.0, 1.0, 4, dtype=np.float32)
    a = state_signature(state_history=history, rainfall_forecast=rain, current_action=action)
    b = state_signature(state_history=history.copy(), rainfall_forecast=rain.copy(), current_action=action.copy())
    assert a.ndim == 1
    assert len(a) > 10
    np.testing.assert_allclose(a, b)


def test_experience_bank_retrieves_near_safe_improving_actions() -> None:
    signature_a = np.zeros(25, dtype=np.float32)
    signature_b = np.ones(25, dtype=np.float32) * 10.0
    sequence_good = np.zeros((12, 3), dtype=np.float32)
    sequence_good[:3, 0] = 0.25
    sequence_bad = np.zeros((12, 3), dtype=np.float32)
    sequence_bad[:3, 1] = 0.75
    frame = pd.DataFrame(
        [
            {
                "state_key": "near",
                "candidate_action_sha256": "good",
                "candidate_action_json": encode_sequence(sequence_good),
                "state_signature_json": encode_signature(signature_a),
                "pfv_feasible": True,
                "tfv_reduction_pct": 20.0,
            },
            {
                "state_key": "far",
                "candidate_action_sha256": "bad",
                "candidate_action_json": encode_sequence(sequence_bad),
                "state_signature_json": encode_signature(signature_b),
                "pfv_feasible": True,
                "tfv_reduction_pct": 1.0,
            },
        ]
    )
    bank = AuthoritativeExperienceBank(frame)
    result = bank.retrieve(
        signature=signature_a,
        current_action=np.zeros(3, dtype=np.float32),
        config=ExperienceRetrievalConfig(nearest_states=1, actions_per_state=1, max_warm_starts=1),
    )
    assert len(result) == 1
    assert result[0]["state_key"] == "near"
    np.testing.assert_allclose(result[0]["sequence"], sequence_good)


def test_experience_bank_filters_unsafe_actions() -> None:
    signature = np.zeros(25, dtype=np.float32)
    sequence = np.zeros((12, 2), dtype=np.float32)
    frame = pd.DataFrame(
        [
            {
                "state_key": "s",
                "candidate_action_sha256": "unsafe",
                "candidate_action_json": encode_sequence(sequence),
                "state_signature_json": encode_signature(signature),
                "pfv_feasible": False,
                "tfv_reduction_pct": 99.0,
            }
        ]
    )
    bank = AuthoritativeExperienceBank(frame)
    assert bank.retrieve(signature=signature, current_action=np.zeros(2, dtype=np.float32)) == []
