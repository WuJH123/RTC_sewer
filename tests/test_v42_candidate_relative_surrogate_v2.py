import torch

from sewerrtc.models.candidate_relative_differentiable_control_v2 import (
    CandidateRelativeDifferentiableControlSurrogateV2,
)


def test_candidate_relative_v2_shapes_and_action_gradient():
    torch.manual_seed(42)
    model = CandidateRelativeDifferentiableControlSurrogateV2()
    state = torch.zeros(2, 25)
    current = torch.full((2, 36), 0.5)
    no_control = torch.ones(2, 36)
    internal = current.clone()
    action = current[:, None, :].repeat(1, 12, 1).requires_grad_(True)
    action.data[:, 0, 0] = 0.2
    out = model(state, action, current, no_control, internal)
    assert out["mean_g_pfv"].shape == (2,)
    assert out["delta_tfv"].shape == (2,)
    assert out["trajectory_residual"].shape == (2, 12, 4)
    grad = torch.autograd.grad(out["delta_tfv"].sum(), action)[0]
    assert torch.isfinite(grad).all()
    assert float(grad.abs().sum()) > 0.0


def test_raw_baseline_and_relative_variant_have_same_interface():
    model = CandidateRelativeDifferentiableControlSurrogateV2(raw_action_baseline=True)
    inputs = [torch.zeros(1, 25), torch.zeros(1, 12, 36), torch.zeros(1, 36), torch.ones(1, 36), torch.zeros(1, 36)]
    out = model(*inputs)
    assert set(out) == {"mean_g_pfv", "log_scale_g_pfv", "delta_tfv", "trajectory_residual"}
