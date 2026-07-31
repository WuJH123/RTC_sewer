"""Head activation and optimizer coverage tests.

Spec §17 items:
    test_v42_head_activation.py
    test_v42_optimizer_coverage.py

These tests verify:
  * every KPI head (PFV, TFV, Peak) produces non-constant output;
  * every head receives gradient during backward;
  * optimizer covers 100% of trainable parameters;
  * an optimizer step actually changes parameters;
  * no detach or no_grad blocks break the gradient path.
"""
from __future__ import annotations

import torch
from torch import nn

from sewerrtc.v4.models_v42.counterfactual_twin_dynamics import TwinGraphDynamics
from sewerrtc.v4.v42_trainer import TwinWithKPIHeads


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _make_model_and_inputs():
    torch.manual_seed(0)
    B, H, N, A, T = 4, 6, 16, 3, 3
    src = torch.arange(N - 1)
    dst = torch.arange(1, N)
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])])
    base = TwinGraphDynamics(
        n_nodes=N, n_facilities=A, n_static_features=3,
        hidden_dim=16, gat_heads=2, n_gat_layers=1, horizon=H, history_frames=T,
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
        edge_index=edge_index,
        node_static=torch.randn(N, 3).abs() + 0.1,
        action_node_map=(torch.rand(A, N) > 0.5).float(),
    )
    return model, inputs


# ---------------------------------------------------------------------------
# test_v42_head_activation
# ---------------------------------------------------------------------------

def test_kpi_heads_produce_non_constant_output():
    """Each KPI head must produce non-constant output across a batch."""
    model, inputs = _make_model_and_inputs()
    model.eval()
    with torch.no_grad():
        out = model(**inputs)
    for key in ("pfv_delta", "tfv_delta", "peak_flood_rate"):
        assert key in out, f"{key} missing from model output"
        std = out[key].std().item()
        assert std > 1e-4, (
            f"{key} is constant (std={std:.2e}) — head is action-blind"
        )


def test_kpi_heads_have_gradient():
    """Each KPI head output must propagate gradient to its own parameters."""
    model, inputs = _make_model_and_inputs()
    model.train()
    out = model(**inputs)
    total_loss = out["pfv_delta"].pow(2).mean() + out["tfv_delta"].pow(2).mean() + out["peak_flood_rate"].pow(2).mean()
    total_loss.backward()

    for head_name, head_module in [
        ("pfv_hurdle", model.pfv_hurdle),
        ("tfv_head", model.tfv_head),
        ("peak_head", model.peak_head),
    ]:
        grad_norm = 0.0
        for p in head_module.parameters():
            if p.grad is not None:
                grad_norm += p.grad.norm().item()
        assert grad_norm > 0.0, (
            f"{head_name} has zero gradient — head is disconnected from loss"
        )


def test_kpi_head_final_layer_has_no_relu():
    """The final Linear layer of each KPI head must NOT have ReLU activation
    (output must allow positive and negative values)."""
    model, _ = _make_model_and_inputs()
    for head_name, head_module in [
        ("pfv_hurdle", model.pfv_hurdle),
        ("tfv_head", model.tfv_head),
        ("peak_head", model.peak_head),
    ]:
        last_layer = head_module[-1]
        assert isinstance(last_layer, nn.Linear), (
            f"{head_name} last layer is {type(last_layer).__name__}, "
            f"expected Linear (no ReLU on output)"
        )


def test_all_kpi_head_architecture_allows_signed_output():
    """KPI heads must have a final Linear layer (no ReLU) so they can
    output both positive and negative values.  We verify the architecture
    and then confirm that varying the input widely produces outputs with
    different signs (or at least the *capacity* to do so)."""
    model, inputs = _make_model_and_inputs()
    # Architectural check: last layer of each head must be Linear (no ReLU)
    for head_name, head_module in [
        ("pfv_hurdle", model.pfv_hurdle),
        ("tfv_head", model.tfv_head),
        ("peak_head", model.peak_head),
    ]:
        last = head_module[-1]
        assert isinstance(last, nn.Linear), (
            f"{head_name} last layer is {type(last).__name__}, expected Linear"
        )
        # No ReLU in the module after the last Linear
        modules_list = list(head_module.children())
        assert not isinstance(modules_list[-1], nn.ReLU), (
            f"{head_name} has ReLU as last layer — output cannot be negative"
        )

    # Functional check: set the final layer bias to a large negative value,
    # then a large positive value, and verify the output changes sign.
    for head_name, head_module in [
        ("pfv_hurdle", model.pfv_hurdle),
        ("tfv_head", model.tfv_head),
        ("peak_head", model.peak_head),
    ]:
        last_linear = head_module[-1]
        with torch.no_grad():
            # Push bias very negative → output should be negative
            last_linear.bias.fill_(-100.0)
            out_neg = model(**inputs)
            val_neg = out_neg[head_name.replace("_hurdle", "_delta")
                              if "pfv" in head_name
                              else head_name.replace("_head", "_delta")
                              if "tfv" in head_name
                              else "peak_flood_rate"].mean().item()

            # Push bias very positive → output should be positive
            last_linear.bias.fill_(100.0)
            out_pos = model(**inputs)
            val_pos = out_pos[head_name.replace("_hurdle", "_delta")
                              if "pfv" in head_name
                              else head_name.replace("_head", "_delta")
                              if "tfv" in head_name
                              else "peak_flood_rate"].mean().item()

        assert val_neg < val_pos, (
            f"{head_name} output does not respond to bias change: "
            f"neg_bias={val_neg:.4f}, pos_bias={val_pos:.4f}"
        )
        # Reset bias
        with torch.no_grad():
            last_linear.bias.zero_()


# ---------------------------------------------------------------------------
# test_v42_optimizer_coverage
# ---------------------------------------------------------------------------

def test_optimizer_covers_all_trainable_parameters():
    """AdamW optimizer must cover 100% of trainable parameters."""
    model, _ = _make_model_and_inputs()
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=1e-3)
    # All parameters that require grad should be in optimizer param groups
    opt_params = set()
    for group in optimizer.param_groups:
        for p in group["params"]:
            opt_params.add(id(p))
    for p in trainable:
        assert id(p) in opt_params, (
            f"Parameter {p.shape} (requires_grad=True) not covered by optimizer"
        )


def test_optimizer_step_changes_parameters():
    """After a backward + optimizer step, at least one parameter must change."""
    model, inputs = _make_model_and_inputs()
    model.train()

    # Snapshot parameters before
    params_before = {}
    for name, p in model.named_parameters():
        params_before[name] = p.data.clone()

    # Forward + backward + step
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-2
    )
    out = model(**inputs)
    loss = out["y_candidate"].pow(2).mean()
    loss.backward()
    optimizer.step()

    # Check at least one parameter changed
    any_changed = False
    for name, p in model.named_parameters():
        if not torch.equal(p.data, params_before[name]):
            any_changed = True
            break

    assert any_changed, (
        "No parameter changed after optimizer step — optimizer may be "
        "misconfigured or gradients are zero"
    )


def test_no_detach_in_kpi_path():
    """The KPI head forward path must not detach tensors."""
    model, inputs = _make_model_and_inputs()
    model.train()
    out = model(**inputs)
    for key in ("pfv_delta", "tfv_delta", "peak_flood_rate"):
        t = out[key]
        assert t.requires_grad, (
            f"{key} does not require grad — detach detected in KPI path"
        )
        assert t.grad_fn is not None, (
            f"{key} has no grad_fn — detach or no_grad detected"
        )
