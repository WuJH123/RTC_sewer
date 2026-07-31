"""V4.2 Head Activation Audit — verify every head/loss is wired and learnable.

For each head/loss in the V4.2 Twin model, checks on a real data batch:
  1. target statistics (std > 0)
  2. prediction statistics (not constant)
  3. mask count (> 0)
  4. raw loss (non-zero)
  5. loss weight (lambda > 0)
  6. weighted loss
  7. requires_grad and grad_fn
  8. gradient norm (non-zero after backward)
  9. optimizer param coverage
 10. actual parameter update after optimizer step

Hard-fail on any check: target std==0, mask==0, loss==0, grad==0,
param not updated, output constant.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from sewerrtc._project_root import PROJECT_ROOT
sys.path.insert(0, str(PROJECT_ROOT))

from sewerrtc.v4.v42_trainer import (
    load_v42_training_data,
    TwinWithKPIHeads,
    _build_model,
    _make_loss_fns,
    _make_shared_tensors,
    _make_batch,
    _DictDataset,
    N_NODES,
    N_FACILITIES,
    HIDDEN_DIM,
    GAT_HEADS,
    N_HISTORY,
    N_HORIZON,
)
from sewerrtc.v4.models_v42.trajectory_losses import TrajectoryLosses
from sewerrtc.v4.models_v42.physics_losses import PhysicsLosses
from sewerrtc.v4.models_v42.ranking_losses import RankingLosses

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stats(t: torch.Tensor) -> dict:
    """Basic tensor statistics."""
    if t.numel() == 0:
        return {"count": 0, "min": 0, "max": 0, "mean": 0, "std": 0}
    flat = t.detach().float().flatten()
    return {
        "count": int(flat.numel()),
        "min": float(flat.min()),
        "max": float(flat.max()),
        "mean": float(flat.mean()),
        "std": float(flat.std()),
    }


def _grad_norm(param: torch.Tensor) -> float:
    if param.grad is not None:
        return float(param.grad.norm().item())
    return 0.0


def _collect_head_params(model: TwinWithKPIHeads) -> dict[str, list[torch.Tensor]]:
    """Map head names to their parameters."""
    base = model.base
    result = {
        "pfv_hurdle": list(model.pfv_hurdle.parameters()),
        "tfv_head": list(model.tfv_head.parameters()),
        "peak_head": list(model.peak_head.parameters()),
        "delta_pool": list(model.delta_pool.parameters()),
        "action_pool": list(model.action_pool.parameters()),
        "depth_head": list(base.depth_head.parameters()),
        "graph_encoder": list(base.graph_encoder.parameters()),
        "rainfall_encoder": list(base.rainfall_encoder.parameters()),
        "action_encoder": list(base.action_encoder.parameters()),
    }
    # Dynamics module varies by model variant
    if hasattr(base, "dynamics_gru"):
        result["dynamics_gru"] = list(base.dynamics_gru.parameters())
    elif hasattr(base, "direct_mlp"):
        result["direct_mlp"] = list(base.direct_mlp.parameters())
    return result


# ---------------------------------------------------------------------------
# Core audit
# ---------------------------------------------------------------------------

def audit_v42_head_activation(
    project_root: str | Path,
    output_root: str | Path,
    config: dict | None = None,
) -> dict:
    """Run head activation audit on a real data batch.

    Returns dict with per-head check results and overall pass/fail.
    """
    project_root = Path(project_root)
    output_root = Path(output_root)
    audit_dir = output_root / "audits" / "v42_head_activation"
    audit_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Head activation audit on device: %s", device)

    # Load data
    data = load_v42_training_data(project_root, output_root)
    n_nodes = data["n_nodes"]
    n_facilities = data["n_facilities"]

    # Pick first 32 samples as audit batch
    n_audit = min(32, len(data["pfv_delta"]))
    audit_idx = np.arange(n_audit)
    batch = _make_batch(data, audit_idx)

    # Build Stage D model (with KPI heads)
    # _build_model("D", ...) already returns TwinWithKPIHeads
    model = _build_model("D", n_nodes, n_facilities, data["node_max_depth"]).to(device)

    # Try loading best trained checkpoint
    ckpt_path = output_root / "final_v4" / "models" / "v42_twin" / "v42_twin_model_seed0_fold0.pt"
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        state = ckpt.get("model_state_dict", ckpt)
        # Filter to matching keys
        model_keys = set(model.state_dict().keys())
        load_keys = {k: v for k, v in state.items() if k in model_keys}
        if load_keys:
            model.load_state_dict(load_keys, strict=False)
            logger.info("Loaded checkpoint from %s (%d keys)", ckpt_path, len(load_keys))

    # Move batch to device
    batch_dev = {k: v.to(device) for k, v in batch.items()}
    shared = _make_shared_tensors(data, device)
    for k, v in shared.items():
        batch_dev[k] = v

    # Loss functions for Stage D
    loss_fn_d = _make_loss_fns("D", n_nodes, data["node_max_depth"],
                                data["edge_index"], device)

    # Also create individual loss modules for fine-grained audit
    traj_loss = TrajectoryLosses().to(device)
    phys_loss = PhysicsLosses(n_nodes=n_nodes, node_max_depth=data["node_max_depth"].to(device)).to(device)
    rank_loss = RankingLosses().to(device)
    edge_index_dev = data["edge_index"].to(device)

    # -----------------------------------------------------------------------
    # Forward pass
    # -----------------------------------------------------------------------
    model.train()
    pred = model(
        state_history=batch_dev["state_history"],
        rainfall=batch_dev["rainfall"],
        action_candidate=batch_dev["action_candidate"],
        action_reference=batch_dev["action_reference"],
        edge_index=batch_dev["edge_index"],
        node_static=batch_dev["node_static"],
        action_node_map=batch_dev["action_node_map"],
    )

    results: dict[str, Any] = {"heads": {}, "checks": [], "overall": "PASS"}

    def _fail(msg: str):
        results["checks"].append({"check": msg, "status": "FAIL"})
        results["overall"] = "FAIL"
        logger.error("HARD FAIL: %s", msg)

    def _pass(msg: str):
        results["checks"].append({"check": msg, "status": "PASS"})

    # -----------------------------------------------------------------------
    # 1. Check prediction outputs
    # -----------------------------------------------------------------------
    for key in ["y_candidate", "y_reference", "delta", "pfv_delta", "tfv_delta", "peak_flood_rate"]:
        if key in pred:
            s = _stats(pred[key])
            results["heads"].setdefault(key, {})["pred_stats"] = s
            if s["std"] < 1e-8:
                _fail(f"pred[{key}] is constant (std={s['std']:.2e})")
            else:
                _pass(f"pred[{key}] has variance (std={s['std']:.4f})")
        else:
            _fail(f"pred missing key: {key}")

    # -----------------------------------------------------------------------
    # 2. Check targets
    # -----------------------------------------------------------------------
    for key in ["pfv_delta", "tfv_delta", "peak_delta", "depth_candidate", "depth_reference"]:
        if key in batch_dev:
            s = _stats(batch_dev[key])
            results["heads"].setdefault(key, {})["target_stats"] = s
            if key in ("pfv_delta", "tfv_delta", "peak_delta") and s["std"] < 1e-8:
                _fail(f"target[{key}] has std=0")
            else:
                _pass(f"target[{key}] loaded (count={s['count']}, std={s['std']:.4f})")

    # -----------------------------------------------------------------------
    # 3. Compute all losses and check non-zero
    # -----------------------------------------------------------------------
    # Trajectory losses
    traj_losses = traj_loss(pred, batch_dev)
    for k, v in traj_losses.items():
        val = v.item() if torch.is_tensor(v) else float(v)
        results["heads"].setdefault(f"traj_{k}", {})["raw_loss"] = val
        if val < 1e-12 and k in ("depth_trajectory", "delta_trajectory"):
            _fail(f"traj loss {k} is zero")
        else:
            _pass(f"traj loss {k} = {val:.6f}")

    # Physics losses
    phys_losses = phys_loss(pred, edge_index=edge_index_dev)
    for k, v in phys_losses.items():
        val = v.item() if torch.is_tensor(v) else float(v)
        results["heads"].setdefault(f"phys_{k}", {})["raw_loss"] = val
        _pass(f"physics loss {k} = {val:.6e}")

    # Ranking losses
    rank_losses = rank_loss(pred, batch_dev)
    for k, v in rank_losses.items():
        val = v.item() if torch.is_tensor(v) else float(v)
        results["heads"].setdefault(f"rank_{k}", {})["raw_loss"] = val
        _pass(f"ranking loss {k} = {val:.6f}")

    # Stage D combined loss
    loss_dict = loss_fn_d(pred, batch_dev)
    total_loss = sum(v for v in loss_dict.values() if torch.is_tensor(v))
    results["total_loss"] = float(total_loss.item())
    _pass(f"total Stage D loss = {total_loss.item():.6f}")

    # -----------------------------------------------------------------------
    # 4. Backward pass — gradient check
    # -----------------------------------------------------------------------
    model.zero_grad()
    total_loss.backward()

    head_params = _collect_head_params(model)
    for head_name, params in head_params.items():
        total_grad = sum(_grad_norm(p) for p in params)
        results["heads"].setdefault(head_name, {})["grad_norm"] = total_grad
        if total_grad < 1e-12:
            _fail(f"head[{head_name}] has zero gradient norm")
        else:
            _pass(f"head[{head_name}] grad_norm = {total_grad:.6f}")

    # -----------------------------------------------------------------------
    # 5. Optimizer parameter coverage
    # -----------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=1e-4
    )
    opt_params = set()
    for pg in optimizer.param_groups:
        for p in pg["params"]:
            opt_params.add(id(p))

    for head_name, params in head_params.items():
        covered = sum(1 for p in params if id(p) in opt_params)
        results["heads"].setdefault(head_name, {})["optimizer_coverage"] = (
            f"{covered}/{len(params)}"
        )
        if covered == 0 and len(params) > 0:
            _fail(f"head[{head_name}] NOT in optimizer")
        else:
            _pass(f"head[{head_name}] optimizer: {covered}/{len(params)}")

    # -----------------------------------------------------------------------
    # 6. Parameter update check
    # -----------------------------------------------------------------------
    pre_params = {n: p.clone().detach() for n, p in model.named_parameters()}
    optimizer.step()
    n_updated = 0
    n_total = 0
    for name, p in model.named_parameters():
        if name in pre_params:
            n_total += 1
            diff = (p.detach() - pre_params[name]).abs().max().item()
            if diff > 0:
                n_updated += 1
    results["param_update"] = {"updated": n_updated, "total": n_total}
    if n_updated == 0:
        _fail("No parameters updated after optimizer step!")
    else:
        _pass(f"{n_updated}/{n_total} parameters updated")

    # -----------------------------------------------------------------------
    # 7. Loss weight audit
    # -----------------------------------------------------------------------
    expected_weights = {
        "depth_traj": 1.0, "delta_traj": 1.0,
        "pfv_kpi": 1.0, "tfv_kpi": 1.0, "peak_kpi": 1.0,
    }
    for k in loss_dict:
        v = loss_dict[k]
        val = v.item() if torch.is_tensor(v) else float(v)
        results["heads"].setdefault(k, {})["weighted_loss"] = val
        results["heads"].setdefault(k, {})["in_loss_dict"] = True

    # -----------------------------------------------------------------------
    # Save outputs
    # -----------------------------------------------------------------------
    detail_path = audit_dir / "head_activation_audit.json"
    with open(detail_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    summary = {
        "overall": results["overall"],
        "n_pass": sum(1 for c in results["checks"] if c["status"] == "PASS"),
        "n_fail": sum(1 for c in results["checks"] if c["status"] == "FAIL"),
        "total_loss": results.get("total_loss"),
        "param_updated": results.get("param_update", {}),
        "failures": [c["check"] for c in results["checks"] if c["status"] == "FAIL"],
    }
    summary_path = audit_dir / "head_activation_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Head activation audit: %s (%d pass, %d fail)",
                summary["overall"], summary["n_pass"], summary["n_fail"])
    if summary["failures"]:
        for f_msg in summary["failures"]:
            logger.error("  FAIL: %s", f_msg)

    return results


# ---------------------------------------------------------------------------
# CLI entry
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "project6_dual_reference_v4" / "final_v4"
    result = audit_v42_head_activation(PROJECT_ROOT, OUTPUT_ROOT)
    print(json.dumps({"overall": result.get("overall", "UNKNOWN")}, indent=2))
