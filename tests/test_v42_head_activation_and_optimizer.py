"""Head activation and optimizer coverage tests for the four-branch V4.2 model."""
from __future__ import annotations

import torch
from torch import nn

from sewerrtc.v4.models_v42.counterfactual_twin_dynamics import TwinGraphDynamics
from sewerrtc.v4.v42_trainer import TwinWithKPIHeads


def _make_model_and_inputs():
    torch.manual_seed(0)
    B, H, N, A, T = 4, 6, 16, 3, 3
    src = torch.arange(N - 1)
    dst = torch.arange(1, N)
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
    base = TwinGraphDynamics(
        n_nodes=N,
        n_facilities=A,
        n_static_features=3,
        hidden_dim=16,
        gat_heads=2,
        n_gat_layers=1,
        horizon=H,
        history_frames=T,
    )
    model = TwinWithKPIHeads(base, hidden_dim=16)
    with torch.no_grad():
        for layer in model.base.depth_head:
            if isinstance(layer, nn.Linear):
                layer.bias.fill_(1.0)
    inputs = dict(
        state_history=torch.randn(B, T, N),
        rainfall=torch.randn(B, H),
        action_candidate=torch.rand(B, H, A),
        action_reference=torch.rand(B, H, A),
        action_dynamic_internal=torch.rand(B, H, A),
        action_hold_previous=torch.rand(B, H, A),
        edge_index=edge_index,
        node_static=torch.randn(N, 3).abs() + 0.1,
        action_node_map=(torch.rand(A, N) > 0.5).float(),
    )
    return model, inputs


def test_kpi_heads_produce_non_constant_output():
    model, inputs = _make_model_and_inputs()
    model.eval()
    with torch.no_grad():
        out = model(**inputs)
    for key in ("pfv_delta", "tfv_delta", "peak_delta"):
        assert key in out
        assert out[key].std().item() > 1e-4, f"{key} is effectively constant"


def test_kpi_heads_have_gradient():
    model, inputs = _make_model_and_inputs()
    model.train()
    out = model(**inputs)
    total_loss = sum(out[k].pow(2).mean() for k in ("pfv_delta", "tfv_delta", "peak_delta"))
    total_loss.backward()
    for head_name, head_module in (
        ("pfv_hurdle", model.pfv_hurdle),
        ("tfv_head", model.tfv_head),
        ("peak_head", model.peak_head),
    ):
        grad_norm = sum(
            p.grad.norm().item() for p in head_module.parameters() if p.grad is not None
        )
        assert grad_norm > 0.0, f"{head_name} has zero gradient"


def test_kpi_head_final_layer_has_no_relu():
    model, _ = _make_model_and_inputs()
    for head_name, head_module in (
        ("pfv_hurdle", model.pfv_hurdle),
        ("tfv_head", model.tfv_head),
        ("peak_head", model.peak_head),
    ):
        assert isinstance(head_module[-1], nn.Linear), f"{head_name} must end in Linear"


def test_all_kpi_head_architecture_allows_signed_output():
    model, inputs = _make_model_and_inputs()
    key_by_head = {
        "pfv_hurdle": "pfv_delta",
        "tfv_head": "tfv_delta",
        "peak_head": "peak_delta",
    }
    for head_name, head_module in (
        ("pfv_hurdle", model.pfv_hurdle),
        ("tfv_head", model.tfv_head),
        ("peak_head", model.peak_head),
    ):
        last = head_module[-1]
        assert isinstance(last, nn.Linear)
        with torch.no_grad():
            last.bias.fill_(-100.0)
            val_neg = model(**inputs)[key_by_head[head_name]].mean().item()
            last.bias.fill_(100.0)
            val_pos = model(**inputs)[key_by_head[head_name]].mean().item()
            last.bias.zero_()
        assert val_neg < 0.0 < val_pos


def test_optimizer_covers_all_trainable_parameters():
    model, _ = _make_model_and_inputs()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    opt_params = {id(p) for group in optimizer.param_groups for p in group["params"]}
    assert all(id(p) in opt_params for p in trainable)


def test_optimizer_step_changes_parameters():
    model, inputs = _make_model_and_inputs()
    model.train()
    before = {name: p.detach().clone() for name, p in model.named_parameters()}
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-2
    )
    out = model(**inputs)
    loss = (
        out["y_candidate"].pow(2).mean()
        + out["pfv_delta"].pow(2).mean()
        + out["tfv_delta"].pow(2).mean()
        + out["peak_delta"].pow(2).mean()
    )
    loss.backward()
    optimizer.step()
    assert any(not torch.equal(p.data, before[name]) for name, p in model.named_parameters())


def test_no_detach_in_kpi_path():
    model, inputs = _make_model_and_inputs()
    model.train()
    out = model(**inputs)
    for key in ("pfv_delta", "tfv_delta", "peak_delta"):
        assert out[key].requires_grad
        assert out[key].grad_fn is not None
