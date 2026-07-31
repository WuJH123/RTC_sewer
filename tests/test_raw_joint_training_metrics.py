from __future__ import annotations

import torch
import numpy as np


def test_aggregate_effect_targets_uses_difference_of_peaks():
    from sewerrtc.models.raw_joint_training import aggregate_effect_targets

    reference = torch.tensor([[[0.0, 5.0, 5.0], [0.0, 10.0, 10.0], [0.0, 4.0, 4.0]]])
    delta = torch.tensor([[[0.0, 3.0, 3.0], [0.0, -2.0, -2.0], [0.0, 4.0, 4.0]]])

    target = aggregate_effect_targets(reference, delta, dt_sec=300.0)

    assert target.shape == (1, 3)
    assert target[0, 1] == 1500.0
    # candidate peak=max(8,8,8)=8; reference peak=max(5,10,4)=10
    assert target[0, 2] == -2.0
    assert delta[:, :, 1].max() == 4.0


def test_direction_metric_keeps_deadband_separate_from_noninferiority_margin():
    from sewerrtc.models.raw_joint_training import direction_accuracy

    pred = torch.tensor([-4.0, 3.0, 0.1])
    target = torch.tensor([-5.0, 2.0, 0.2])
    metric = direction_accuracy(pred, target, tolerance=1.0)

    assert metric["count"] == 2
    assert metric["accuracy"] == 1.0


def test_pfv_noninferiority_uses_absolute_plus_relative_margin():
    from sewerrtc.models.raw_joint_training import noninferiority_metrics

    pred = torch.tensor([80.0, 160.0, 130.0])
    target = torch.tensor([90.0, 170.0, 140.0])
    reference = torch.tensor([0.0, 40_000.0, 30_000.0])
    metric = noninferiority_metrics(
        pred,
        target,
        reference,
        absolute_margin=100.0,
        relative_margin=0.005,
    )

    # Margins are 100, 200, and 150 m3; all target labels are noninferior.
    assert metric["target_noninferior_fraction"] == 1.0
    assert metric["classification_accuracy"] == 1.0


def test_binary_metrics_fail_explicitly_when_one_class_has_no_support():
    from sewerrtc.models.raw_joint_training import binary_classification_metrics

    supported = binary_classification_metrics(
        torch.tensor([0.9, 0.7, 0.4, 0.1]),
        torch.tensor([1, 1, 0, 0]),
    )
    unsupported = binary_classification_metrics(
        torch.tensor([0.9, 0.8]),
        torch.tensor([1, 1]),
    )

    assert supported["balanced_accuracy"] == 1.0
    assert supported["negative_recall"] == 1.0
    assert unsupported["negative_count"] == 0
    assert unsupported["balanced_accuracy"] is None
    assert unsupported["negative_recall"] is None


def test_dynamics_warm_start_only_loads_shared_encoder_parameters():
    from sewerrtc.models.raw_joint_training import load_dynamics_warm_start

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.actuator_input = torch.nn.Linear(2, 2)
            self.reference_risk_head = torch.nn.Linear(2, 1)
            self.effect_risk_head = torch.nn.Linear(2, 1)

    source = Tiny()
    target = Tiny()
    with torch.no_grad():
        source.actuator_input.weight.fill_(3.0)
        source.reference_risk_head.weight.fill_(4.0)
        source.effect_risk_head.weight.fill_(5.0)
        target.effect_risk_head.weight.fill_(9.0)
    loaded = load_dynamics_warm_start(
        target,
        {"model": source.state_dict(), "node_ids": ["N1"], "action_ids": ["A1"]},
        node_ids=["N1"],
        action_ids=["A1"],
    )

    assert "actuator_input.weight" in loaded
    assert "reference_risk_head.weight" in loaded
    assert "effect_risk_head.weight" not in loaded
    assert torch.all(target.actuator_input.weight == 3.0)
    assert torch.all(target.reference_risk_head.weight == 4.0)
    assert torch.all(target.effect_risk_head.weight == 9.0)


def test_configure_effect_training_can_fine_tune_action_identity_without_unfreezing_gat():
    from sewerrtc.models.raw_joint_training import configure_effect_training_parameters

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.actuator_input = torch.nn.Linear(2, 2)
            self.actuator_identity = torch.nn.Embedding(3, 2)
            self.actuator_temporal = torch.nn.GRU(2, 2, batch_first=True)
            self.cross_attention = torch.nn.MultiheadAttention(2, 1, batch_first=True)
            self.gat = torch.nn.Linear(2, 2)
            self.effect_risk_head = torch.nn.Linear(2, 1)
            self.effect_scale_head = torch.nn.Linear(2, 1)
            self.safety_classification_head = torch.nn.Linear(2, 1)

    model = Tiny()
    groups, names = configure_effect_training_parameters(
        model,
        fine_tune_action_encoder=True,
        head_learning_rate=1.0e-3,
        action_learning_rate_scale=0.1,
    )

    assert "actuator_identity.weight" in names
    assert "cross_attention.in_proj_weight" in names
    assert "gat.weight" not in names
    assert model.gat.weight.requires_grad is False
    assert sorted(group["lr"] for group in groups) == [1.0e-4, 1.0e-3]


def test_configure_effect_training_can_adapt_state_action_interaction_at_lower_rate():
    from sewerrtc.models.raw_joint_training import configure_effect_training_parameters

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.node_query = torch.nn.Linear(2, 2)
            self.node_input = torch.nn.Linear(2, 2)
            self.node_norm = torch.nn.LayerNorm(2)
            self.gat = torch.nn.Linear(2, 2)
            self.effect_risk_head = torch.nn.Linear(2, 1)

    model = Tiny()
    groups, names = configure_effect_training_parameters(
        model,
        fine_tune_action_encoder=False,
        fine_tune_state_interaction=True,
        head_learning_rate=1.0e-3,
        action_learning_rate_scale=0.1,
        state_learning_rate_scale=0.02,
    )

    assert "node_query.weight" in names
    assert "node_input.weight" in names
    assert "node_norm.weight" in names
    assert "gat.weight" not in names
    assert model.gat.weight.requires_grad is False
    assert sorted(group["lr"] for group in groups) == [2.0e-5, 1.0e-3]


def test_resolve_event_group_indices_honours_embedded_split() -> None:
    from sewerrtc.models.raw_joint_training import resolve_event_group_indices

    event_ids = ["event_a", "event_a", "event_b", "event_c"]
    split = ["train", "train", "validation", "train"]

    train_idx, validation_idx, validation_events = resolve_event_group_indices(event_ids, split)

    assert train_idx.tolist() == [0, 1, 3]
    assert validation_idx.tolist() == [2]
    assert validation_events == {"event_b"}


def test_effect_sampling_weights_upweight_rare_events_and_unsafe_labels() -> None:
    from sewerrtc.models.raw_joint_training import build_effect_sampling_weights

    event_ids = np.asarray(["common", "common", "common", "rare"])
    aggregate = np.asarray(
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [200.0, -500.0, 2.0]],
        dtype=np.float32,
    )
    classes = np.asarray(
        [[1, 0, 1], [1, 0, 1], [1, 0, 1], [0, 1, 0]],
        dtype=np.float32,
    )

    weights = build_effect_sampling_weights(
        event_ids,
        aggregate,
        classes,
        tolerances=np.asarray([1.0, 100.0, 0.1]),
    )

    assert weights.shape == (4,)
    assert np.isclose(weights.sum(), 1.0)
    assert weights[-1] > weights[0]


def test_effect_sampling_weights_balance_phase_and_actuator_identity() -> None:
    from sewerrtc.models.raw_joint_training import build_effect_sampling_weights

    events = np.asarray(["E1"] * 4 + ["E2"] * 2)
    aggregate = np.asarray([
        [-1.0, -2.0, -0.2],
        [-1.0, -2.0, -0.2],
        [-1.0, -2.0, -0.2],
        [-1.0, -2.0, -0.2],
        [1.0, 2.0, 0.2],
        [1.0, 2.0, 0.2],
    ])
    classes = (aggregate <= 0.0).astype(np.int8)
    phases = np.asarray(["peak", "peak", "peak", "peak", "rising", "recession"])
    signatures = np.asarray(["A", "A", "A", "A", "B", "C"])

    weights = build_effect_sampling_weights(
        events,
        aggregate,
        classes,
        tolerances=np.asarray([0.1, 0.1, 0.01]),
        phases=phases,
        actuator_signatures=signatures,
    )

    assert weights[4] > weights[0]
    assert weights[5] > weights[0]


def test_effect_sampling_weights_prioritise_deployment_rows_without_dropping_replay() -> None:
    from sewerrtc.models.raw_joint_training import build_effect_sampling_weights

    events = np.asarray(["E1", "E2", "E3", "E4"])
    aggregate = np.asarray([
        [-1.0, -2.0, -0.2],
        [1.0, 2.0, 0.2],
        [-1.0, -2.0, -0.2],
        [1.0, 2.0, 0.2],
    ])
    classes = (aggregate <= 0.0).astype(np.int8)
    deployment = np.asarray([True, True, False, False])

    weights = build_effect_sampling_weights(
        events,
        aggregate,
        classes,
        tolerances=np.asarray([0.1, 0.1, 0.01]),
        deployment_mask=deployment,
        replay_weight=0.1,
    )

    assert weights[0] > weights[2]
    assert weights[1] > weights[3]
    assert np.all(weights > 0.0)


def test_effect_sampling_weights_balance_each_safety_channel_without_optional_labels() -> None:
    from sewerrtc.models.raw_joint_training import build_effect_sampling_weights

    events = np.asarray(["E1", "E2", "E3", "E4"])
    aggregate = np.ones((4, 3), dtype=np.float32)
    classes = np.ones((4, 3), dtype=np.int8)
    classes[-1, 0] = 0

    weights = build_effect_sampling_weights(
        events,
        aggregate,
        classes,
        tolerances=np.asarray([0.1, 0.1, 0.1]),
    )

    assert weights[-1] > weights[0]


def test_same_checkpoint_pairwise_ranking_rewards_correct_order() -> None:
    from sewerrtc.models.raw_joint_training import same_checkpoint_pairwise_ranking_loss

    target = torch.tensor([
        [-10.0, -1000.0, -2.0],
        [5.0, 500.0, 1.0],
        [1.0, -200.0, 0.5],
    ])
    groups = np.asarray(["checkpoint_a", "checkpoint_a", "checkpoint_b"])
    scale = torch.tensor([10.0, 1000.0, 2.0])
    tolerance = torch.tensor([1.0, 100.0, 0.1])

    correct = same_checkpoint_pairwise_ranking_loss(
        target.clone(), target, groups, scale=scale, tolerances=tolerance
    )
    reversed_prediction = target.clone()
    reversed_prediction[:2] = reversed_prediction[:2].flip(0)
    wrong = same_checkpoint_pairwise_ranking_loss(
        reversed_prediction, target, groups, scale=scale, tolerances=tolerance
    )

    assert correct.item() < wrong.item()
    assert correct.item() >= 0.0


def test_conformal_sigma_multiplier_expands_undercovered_channel() -> None:
    from sewerrtc.models.raw_joint_training import conformal_sigma_multipliers

    prediction = np.zeros((5, 3), dtype=np.float32)
    target = np.asarray([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]], dtype=np.float32)
    sigma = np.ones((5, 3), dtype=np.float32)

    multiplier = conformal_sigma_multipliers(prediction, target, sigma, coverage=0.9)

    assert multiplier.shape == (3,)
    assert np.all(multiplier >= 1.0)
    assert np.allclose(multiplier, multiplier[0])


def test_select_binary_threshold_uses_calibration_labels() -> None:
    from sewerrtc.models.raw_joint_training import select_binary_threshold

    probability = np.asarray([0.95, 0.70, 0.55, 0.40, 0.20, 0.05])
    target = np.asarray([1, 1, 1, 0, 0, 0])

    result = select_binary_threshold(probability, target, minimum_negative_recall=0.8)

    assert 0.4 < result["threshold"] <= 0.55
    assert result["negative_recall"] >= 0.8


def test_select_binary_threshold_reports_missing_class_without_nan() -> None:
    from sewerrtc.models.raw_joint_training import select_binary_threshold

    result = select_binary_threshold(np.asarray([0.8, 0.9]), np.asarray([1, 1]))

    assert result["threshold"] == 0.5
    assert result["balanced_accuracy"] is None
    assert result["negative_recall"] is None


def test_apply_uncertainty_multipliers_scales_only_aggregate_sigma() -> None:
    from sewerrtc.models.raw_joint_training import apply_uncertainty_multipliers

    outputs = {
        "delta_PFV_sigma": np.asarray([1.0, 2.0], dtype=np.float32),
        "delta_TFV_sigma": np.asarray([3.0, 4.0], dtype=np.float32),
        "delta_peak_sigma": np.asarray([5.0, 6.0], dtype=np.float32),
        "delta_PFV_H": np.asarray([-1.0, -2.0], dtype=np.float32),
    }

    calibrated = apply_uncertainty_multipliers(outputs, [2.0, 3.0, 4.0])

    assert np.allclose(calibrated["delta_PFV_sigma"], [2.0, 4.0])
    assert np.allclose(calibrated["delta_TFV_sigma"], [9.0, 12.0])
    assert np.allclose(calibrated["delta_peak_sigma"], [20.0, 24.0])
    assert np.array_equal(calibrated["delta_PFV_H"], outputs["delta_PFV_H"])
    assert calibrated is not outputs
