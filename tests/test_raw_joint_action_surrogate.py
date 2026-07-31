from __future__ import annotations

import torch


def test_causal_action_scale_preserves_delayed_hydraulic_effect():
    from sewerrtc.models.raw_joint_action_surrogate import causal_action_scale

    residual = torch.zeros(2, 4, 3)
    residual[0, 1, 2] = 0.2
    scale = causal_action_scale(residual)

    assert scale.shape == (2, 4, 1)
    assert torch.allclose(scale[0, :, 0], torch.tensor([0.0, 0.2, 0.2, 0.2]))
    assert torch.count_nonzero(scale[1]) == 0


def _model():
    from sewerrtc.models.raw_joint_action_surrogate import RawJointActionSurrogate

    return RawJointActionSurrogate(
        n_nodes=4,
        n_actions=3,
        node_static_dim=2,
        actuator_feature_dim=4,
        horizon_steps=3,
        hidden_dim=16,
        heads=2,
        architecture_version="priority_aware_v2",
    )


def _inputs():
    batch, horizon, nodes, actions = 2, 3, 4, 3
    state = torch.rand(batch, nodes)
    rain = torch.rand(batch, horizon, 1)
    reference = torch.rand(batch, horizon, actions)
    return {
        "state": state,
        "candidate_action_seq": reference.clone(),
        "reference_action_seq": reference,
        "rain_seq": rain,
        "actuator_mask": torch.ones(batch, actions),
        "actuator_features": torch.rand(actions, 4),
        "node_static": torch.rand(nodes, 2),
        "edge_index": torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]]),
        "action_node_map": torch.tensor([[1., 0., 0., 0.], [0., 1., 0., 0.], [0., 0., 1., 0.]]),
        "priority_indices": torch.tensor([0, 1]),
        "storage_indices": torch.tensor([2]),
    }


def test_raw_joint_surrogate_zero_action_is_exactly_zero_effect():
    model = _model().eval()
    with torch.no_grad():
        output = model(**_inputs())

    assert output["delta_risk_rate_seq"].shape == (2, 3, 3)
    assert output["PFV_rate_seq"].shape == (2, 3)
    assert output["priority_depth_seq"].shape == (2, 3)
    assert torch.allclose(output["delta_risk_rate_seq"], torch.zeros_like(output["delta_risk_rate_seq"]), atol=1e-7)
    assert torch.allclose(output["delta_PFV_H"], torch.zeros(2), atol=1e-7)


def test_legacy_architecture_remains_available_for_old_checkpoints():
    from sewerrtc.models.raw_joint_action_surrogate import RawJointActionSurrogate

    model = RawJointActionSurrogate(
        n_nodes=4,
        n_actions=3,
        node_static_dim=2,
        actuator_feature_dim=4,
        horizon_steps=3,
        hidden_dim=16,
        heads=2,
    ).eval()
    assert model.architecture_version == "legacy_v1"
    assert model.actuator_identity is None


def test_raw_joint_surrogate_masks_disallowed_actuator_changes():
    model = _model().eval()
    kwargs = _inputs()
    kwargs["candidate_action_seq"][:, :, 2] = 0.0
    kwargs["reference_action_seq"][:, :, 2] = 1.0
    kwargs["actuator_mask"][:, 2] = 0.0
    with torch.no_grad():
        output = model(**kwargs)

    assert torch.allclose(output["delta_risk_rate_seq"], torch.zeros_like(output["delta_risk_rate_seq"]), atol=1e-7)


def test_online_no_control_predictor_uses_reference_actions_without_true_future_metrics():
    from sewerrtc.control.no_control_reference_predictor import OnlineNoControlReferencePredictor

    model = _model().eval()
    predictor = OnlineNoControlReferencePredictor(model)
    kwargs = _inputs()
    kwargs.pop("candidate_action_seq")
    with torch.no_grad():
        output = predictor.predict(**kwargs)

    assert output["reference_risk_rate_seq"].shape == (2, 3, 3)
    assert "offline_true_no_control_reference" not in output


def test_online_no_control_action_sequence_does_not_drift_with_proposed_setting():
    from sewerrtc.control.no_control_reference_predictor import constant_default_action_sequence

    passive_default = torch.tensor([1.0, 1.0, 0.0]).numpy()
    proposed_live_setting = torch.tensor([0.8, 1.0, 1.0]).numpy()
    reference = constant_default_action_sequence(passive_default, 6)

    assert reference.shape == (6, 3)
    assert (reference == passive_default[None, :]).all()
    assert not (reference[0] == proposed_live_setting).all()


def test_raw_joint_surrogate_exposes_uncertainty_and_safety_probabilities():
    model = _model().eval()
    kwargs = _inputs()
    kwargs["candidate_action_seq"][:, 0, 0] += 0.1
    with torch.no_grad():
        output = model(**kwargs)

    assert output["delta_risk_log_scale_seq"].shape == (2, 3, 3)
    assert output["delta_PFV_sigma"].shape == (2,)
    assert output["delta_TFV_sigma"].shape == (2,)
    assert output["delta_peak_sigma"].shape == (2,)
    for key in (
        "PFV_noninferiority_probability",
        "TFV_improvement_probability",
        "peak_safe_probability",
    ):
        assert output[key].shape == (2,)
        assert torch.all((output[key] >= 0.0) & (output[key] <= 1.0))


def test_raw_joint_surrogate_preserves_actuator_identity_and_temporal_order():
    torch.manual_seed(7)
    model = _model().eval()
    kwargs = _inputs()
    kwargs["candidate_action_seq"] = kwargs["reference_action_seq"].clone()
    kwargs["candidate_action_seq"][:, 0:1, 0] += 0.2
    with torch.no_grad():
        early_a = model(**kwargs)["delta_risk_rate_seq"]
        identity = _inputs()
        identity["candidate_action_seq"] = identity["reference_action_seq"].clone()
        identity["candidate_action_seq"][:, 0:1, 1] += 0.2
        early_b = model(**identity)["delta_risk_rate_seq"]
        late = _inputs()
        late["candidate_action_seq"] = late["reference_action_seq"].clone()
        late["candidate_action_seq"][:, -1:, 0] += 0.2
        late_a = model(**late)["delta_risk_rate_seq"]

    assert not torch.allclose(early_a, early_b, atol=1e-8)
    assert not torch.allclose(early_a, late_a, atol=1e-8)


def test_causal_active_actuator_mask_tracks_only_changed_facilities_over_time():
    from sewerrtc.models.raw_joint_action_surrogate import causal_active_actuator_mask

    delta = torch.zeros((1, 6, 4))
    delta[:, 1:3, 2] = -0.2
    delta[:, 4, 1] = 1.0

    active = causal_active_actuator_mask(delta)

    assert active.shape == delta.shape
    assert active[0, 0].tolist() == [False, False, False, False]
    assert active[0, 1].tolist() == [False, False, True, False]
    assert active[0, 3].tolist() == [False, False, True, False]
    assert active[0, 4].tolist() == [False, True, True, False]


def test_v3_surrogate_exposes_learned_safety_classifiers_without_breaking_zero_effect():
    from sewerrtc.models.raw_joint_action_surrogate import RawJointActionSurrogate

    model = RawJointActionSurrogate(
        n_nodes=4,
        n_actions=3,
        node_static_dim=2,
        actuator_feature_dim=4,
        horizon_steps=3,
        hidden_dim=16,
        heads=2,
        architecture_version="priority_aware_safety_v3",
    ).eval()
    kwargs = _inputs()
    with torch.no_grad():
        output = model(**kwargs)

    assert output["safety_classification_logits"].shape == (2, 3)
    assert output["PFV_noninferiority_classifier_probability"].shape == (2,)
    assert output["TFV_improvement_classifier_probability"].shape == (2,)
    assert output["peak_safe_classifier_probability"].shape == (2,)
    assert torch.count_nonzero(output["delta_risk_rate_seq"]) == 0


def test_v4_action_residual_path_preserves_v3_warm_start_at_initialization():
    from sewerrtc.models.raw_joint_action_surrogate import RawJointActionSurrogate

    common = dict(
        n_nodes=3,
        n_actions=2,
        node_static_dim=2,
        actuator_feature_dim=4,
        horizon_steps=3,
        hidden_dim=16,
        heads=2,
        dropout=0.0,
    )
    torch.manual_seed(11)
    v3 = RawJointActionSurrogate(**common, architecture_version="priority_aware_safety_v3")
    torch.manual_seed(99)
    v4 = RawJointActionSurrogate(**common, architecture_version="priority_aware_safety_v4")
    incompatible = v4.load_state_dict(v3.state_dict(), strict=False)

    assert set(incompatible.missing_keys) == {
        "effect_action_residual_head.0.weight",
        "effect_action_residual_head.0.bias",
        "effect_action_residual_head.2.weight",
        "effect_action_residual_head.2.bias",
        "safety_action_residual_head.0.weight",
        "safety_action_residual_head.0.bias",
        "safety_action_residual_head.2.weight",
        "safety_action_residual_head.2.bias",
    }
    assert incompatible.unexpected_keys == []
    assert torch.count_nonzero(v4.effect_action_residual_head[-1].weight) == 0
    assert torch.count_nonzero(v4.safety_action_residual_head[-1].weight) == 0


def test_v5_phase_conditioned_heads_preserve_zero_effect_and_receive_phase_identity():
    from sewerrtc.models.raw_joint_action_surrogate import RawJointActionSurrogate

    model = RawJointActionSurrogate(
        n_nodes=4,
        n_actions=3,
        node_static_dim=2,
        actuator_feature_dim=4,
        horizon_steps=3,
        hidden_dim=16,
        heads=2,
        dropout=0.0,
        architecture_version="causal_phase_safety_v5",
    ).eval()
    kwargs = _inputs()
    kwargs["phase_index"] = torch.tensor([1, 2])
    with torch.no_grad():
        zero = model(**kwargs)

    assert model.has_phase_conditioning
    assert zero["phase_index"].tolist() == [1, 2]
    assert torch.count_nonzero(zero["delta_risk_rate_seq"]) == 0
    assert torch.count_nonzero(zero["delta_PFV_H"]) == 0


def test_v5_horizon_effect_head_retains_raw_temporal_order():
    from sewerrtc.models.raw_joint_action_surrogate import RawJointActionSurrogate

    torch.manual_seed(41)
    model = RawJointActionSurrogate(
        n_nodes=4,
        n_actions=3,
        node_static_dim=2,
        actuator_feature_dim=4,
        horizon_steps=3,
        hidden_dim=16,
        heads=2,
        dropout=0.0,
        architecture_version="causal_phase_safety_v5",
    ).eval()
    early = _inputs()
    early["phase_index"] = torch.tensor([1, 1])
    early["candidate_action_seq"] = early["reference_action_seq"].clone()
    early["candidate_action_seq"][:, 0, 0] += 0.2
    late = {key: value.clone() if torch.is_tensor(value) else value for key, value in early.items()}
    late["candidate_action_seq"] = late["reference_action_seq"].clone()
    late["candidate_action_seq"][:, -1, 0] += 0.2

    with torch.no_grad():
        early_effect = model(**early)["delta_TFV_H"]
        late_effect = model(**late)["delta_TFV_H"]

    assert not torch.allclose(early_effect, late_effect, atol=1.0e-8)
