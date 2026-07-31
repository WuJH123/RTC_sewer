from __future__ import annotations

import numpy as np
import torch


def test_observational_peak_channel_is_running_tfv_peak() -> None:
    from sewerrtc.data.peak_label_semantics import repair_observational_risk_rate_seq

    risk = np.array(
        [[[1.0, 5.0, 5.0], [2.0, 10.0, 10.0], [3.0, 4.0, 4.0]]],
        dtype=np.float32,
    )

    repaired = repair_observational_risk_rate_seq(risk)

    np.testing.assert_allclose(repaired[0, :, 2], [5.0, 10.0, 10.0])
    np.testing.assert_allclose(repaired[0, :, :2], risk[0, :, :2])


def test_paired_peak_channel_preserves_candidate_minus_reference_semantics() -> None:
    from sewerrtc.data.peak_label_semantics import (
        peak_label_semantics_valid,
        repair_paired_risk_rate_sequences,
    )

    reference = np.array(
        [[[0.0, 5.0, 5.0], [0.0, 10.0, 10.0], [0.0, 4.0, 4.0]]],
        dtype=np.float32,
    )
    delta = np.array(
        [[[0.0, 3.0, 3.0], [0.0, -2.0, -2.0], [0.0, 4.0, 4.0]]],
        dtype=np.float32,
    )

    repaired_reference, repaired_delta = repair_paired_risk_rate_sequences(reference, delta)

    np.testing.assert_allclose(repaired_reference[0, :, 2], [5.0, 10.0, 10.0])
    # Candidate TFV rates are [8, 8, 8], so their running peak is [8, 8, 8].
    np.testing.assert_allclose(
        repaired_reference[0, :, 2] + repaired_delta[0, :, 2],
        [8.0, 8.0, 8.0],
    )
    assert repaired_delta[0, -1, 2] == -2.0
    assert peak_label_semantics_valid(repaired_reference + repaired_delta)


def test_horizon_peak_metrics_are_derived_from_tfv_rate_sequence() -> None:
    from sewerrtc.models.observational_action_pretraining import horizon_peak_metrics

    target = np.array(
        [[[0.0, 5.0, 5.0], [0.0, 10.0, 10.0], [0.0, 4.0, 4.0]]],
        dtype=np.float32,
    )
    prediction = np.array(
        [[[0.0, 4.0, 99.0], [0.0, 9.0, 99.0], [0.0, 6.0, 99.0]]],
        dtype=np.float32,
    )

    metrics = horizon_peak_metrics(target, prediction)

    assert metrics["target"].tolist() == [10.0]
    assert metrics["prediction"].tolist() == [9.0]
    assert metrics["MAE"] == 1.0


def test_online_no_control_peak_uses_tfv_rate_not_auxiliary_channel() -> None:
    from sewerrtc.control.no_control_reference_predictor import OnlineNoControlReferencePredictor

    class Stub(torch.nn.Module):
        def forward(self, **kwargs):
            return {
                "reference_risk_rate_seq": torch.tensor(
                    [[[1.0, 5.0, 100.0], [2.0, 10.0, 1.0], [3.0, 4.0, 200.0]]]
                )
            }

    predictor = OnlineNoControlReferencePredictor(Stub())
    result = predictor.predict(
        state=torch.zeros(1, 1),
        reference_action_seq=torch.zeros(1, 3, 1),
        rain_seq=torch.zeros(1, 3, 1),
        actuator_mask=torch.ones(1, 1),
        actuator_features=torch.zeros(1, 1),
        node_static=torch.zeros(1, 1),
        edge_index=torch.zeros(2, 0, dtype=torch.long),
        action_node_map=torch.ones(1, 1),
        priority_indices=torch.zeros(0, dtype=torch.long),
        storage_indices=torch.zeros(0, dtype=torch.long),
    )

    assert result["reference_peak_TFV_rate"].item() == 10.0
