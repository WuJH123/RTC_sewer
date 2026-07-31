"""V4.2 multi-reference graph-dynamics training pipeline.

Key correctness rules enforced here:

* Candidate, No-control, Dynamic Internal and Hold-Previous are explicit
  branches.  DI/Hold are never silently replaced by No-control.
* the canonical history is 13 frames (60 min at 5-min spacing);
* fold-local KPI normalisation is fitted on training samples only;
* per-sample optional tensors are always indexed with the fold/sample index;
* PFV uses the No-control-relative head, TFV/Peak use DI-relative heads;
* the Peak model output is consistently named ``peak_delta``;
* same-state IDs and normalised safety boundaries are passed to the
  lexicographic ranking loss.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from sewerrtc.v4.v42_priority_contract import (
    PriorityContractError,
    get_pfv_core_node_indices,
)

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

N_NODES = 932
N_EDGES = 1276
N_FACILITIES = 36
N_HISTORY = 13
N_HORIZON = 12
HIDDEN_DIM = 32
GAT_HEADS = 4

STAGE_EPOCHS = {"A": 40, "B": 60, "C": 40, "D": 20}
TRAINING_CONFIG = {
    "seeds": [0, 1, 2, 3, 4],
    "batch_size": 32,
    "learning_rate": 1e-3,
    "lr_stage_d": 1e-4,
    "weight_decay": 1e-4,
    "max_grad_norm": 1.0,
    "early_stop_patience": 10,
    "amp_enabled": True,
    "n_cv_folds": 5,
}

ABLATION_CONFIGS = {
    "full_4stage": {
        "description": "Full 4-stage pipeline (baseline)",
        "use_physics": True,
        "use_twin": True,
        "simplified_dyn": False,
        "loss_override": {},
    },
    "no_physics": {
        "description": "Remove physics/constraint losses",
        "use_physics": False,
        "use_twin": True,
        "simplified_dyn": False,
        "loss_override": {},
    },
    "no_twin": {
        "description": "Single-tower ablation",
        "use_physics": True,
        "use_twin": False,
        "simplified_dyn": False,
        "loss_override": {},
    },
    "simplified_dynamics": {
        "description": "Direct MLP dynamics ablation",
        "use_physics": True,
        "use_twin": True,
        "simplified_dyn": True,
        "loss_override": {},
    },
}


def _parse_json_array(s: str) -> np.ndarray:
    return np.asarray(json.loads(s), dtype=np.float32)


def load_graph_topology(project_root: Path) -> dict:
    from .v42_trajectory_builder import _load_graph_topology

    return _load_graph_topology(Path(project_root))


def _require_columns(df: pd.DataFrame, columns: list[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(
            "V4.2 full-supervision dataset is missing required columns: "
            f"{missing}. Rebuild the trajectory dataset; do not substitute another reference."
        )


def _stack_json_column(df: pd.DataFrame, name: str) -> np.ndarray:
    try:
        return np.stack([_parse_json_array(s) for s in df[name]])
    except Exception as exc:
        raise ValueError(f"Failed to parse trajectory column {name!r}: {exc}") from exc


def _validate_shape(name: str, arr: np.ndarray, expected_tail: tuple[int, ...]) -> None:
    if arr.ndim != len(expected_tail) + 1 or tuple(arr.shape[1:]) != expected_tail:
        raise ValueError(
            f"{name} must have shape [N,{','.join(map(str, expected_tail))}], "
            f"got {tuple(arr.shape)}"
        )
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN/Inf")


def load_v42_training_data(project_root: Path, output_root: Path) -> dict[str, Any]:
    """Load the canonical four-branch V4.2 trajectory dataset fail-closed."""
    project_root = Path(project_root)
    output_root = Path(output_root)
    trajectory_dir = output_root / "v42" / "trajectory_dataset"
    parquet_path = trajectory_dir / "trajectory_manifest_v42.parquet"
    csv_path = trajectory_dir / "trajectory_manifest_v42.csv"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
    else:
        raise FileNotFoundError(f"No trajectory manifest found in {trajectory_dir}")

    graph = load_graph_topology(project_root)
    n_nodes = int(graph["n_nodes"])
    n_facilities = int(graph["n_facilities"])
    if n_facilities != N_FACILITIES:
        raise ValueError(f"Engineering36 contract expected 36 facilities, got {n_facilities}")

    required = [
        "history_depth",
        "rainfall_forecast",
        "candidate_action_seq",
        "ref_no_control_action_seq",
        "ref_dynamic_internal_action_seq",
        "ref_hold_previous_action_seq",
        "trajectory_depth_candidate",
        "trajectory_depth_no_control",
        "trajectory_depth_dynamic_internal",
        "trajectory_depth_hold_previous",
        "pfv_delta",
        "tfv_delta",
        "peak_delta",
        "event_id",
    ]
    _require_columns(df, required)
    N = len(df)
    if N == 0:
        raise ValueError("Trajectory dataset is empty")

    state_history = _stack_json_column(df, "history_depth")
    rainfall = _stack_json_column(df, "rainfall_forecast")
    action_candidate = _stack_json_column(df, "candidate_action_seq")
    action_reference = _stack_json_column(df, "ref_no_control_action_seq")
    action_di = _stack_json_column(df, "ref_dynamic_internal_action_seq")
    action_hold = _stack_json_column(df, "ref_hold_previous_action_seq")
    depth_candidate = _stack_json_column(df, "trajectory_depth_candidate")
    depth_reference = _stack_json_column(df, "trajectory_depth_no_control")
    depth_di = _stack_json_column(df, "trajectory_depth_dynamic_internal")
    depth_hold = _stack_json_column(df, "trajectory_depth_hold_previous")

    _validate_shape("state_history", state_history, (N_HISTORY, n_nodes))
    _validate_shape("rainfall", rainfall, (N_HORIZON,))
    for name, arr in (
        ("action_candidate", action_candidate),
        ("action_reference", action_reference),
        ("action_dynamic_internal", action_di),
        ("action_hold_previous", action_hold),
    ):
        _validate_shape(name, arr, (N_HORIZON, N_FACILITIES))
    for name, arr in (
        ("depth_candidate", depth_candidate),
        ("depth_reference", depth_reference),
        ("depth_dynamic_internal", depth_di),
        ("depth_hold_previous", depth_hold),
    ):
        _validate_shape(name, arr, (N_HORIZON, n_nodes))

    pfv_delta = pd.to_numeric(df["pfv_delta"], errors="raise").to_numpy(np.float32)
    tfv_delta = pd.to_numeric(df["tfv_delta"], errors="raise").to_numpy(np.float32)
    peak_delta = pd.to_numeric(df["peak_delta"], errors="raise").to_numpy(np.float32)

    event_ids = df["event_id"].astype(str).to_numpy()
    # Prefer the rainfall fingerprint/sha for CV isolation.  Fall back to event
    # ID only for legacy synthetic fixtures, and record both in the returned data.
    if "rainfall_sha256" in df.columns:
        cv_group_ids = df["rainfall_sha256"].astype(str).to_numpy()
    elif "base_rainfall_fingerprint" in df.columns:
        cv_group_ids = df["base_rainfall_fingerprint"].astype(str).to_numpy()
    else:
        cv_group_ids = event_ids.copy()
        logger.warning("No rainfall fingerprint column; using event_id for grouped CV")
    unique_groups = sorted(set(cv_group_ids))
    group_to_idx = {g: i for i, g in enumerate(unique_groups)}
    event_indices = np.asarray([group_to_idx[g] for g in cv_group_ids], dtype=np.int64)

    if "state_key" in df.columns:
        state_keys = df["state_key"].astype(str).to_numpy()
    elif "checkpoint_id" in df.columns:
        state_keys = (
            df["event_id"].astype(str) + "::" + df["checkpoint_id"].astype(str)
        ).to_numpy()
    else:
        raise KeyError("Dataset needs state_key or checkpoint_id for same-state ranking")
    unique_states = sorted(set(state_keys))
    state_to_idx = {s: i for i, s in enumerate(unique_states)}
    state_group_index = np.asarray([state_to_idx[s] for s in state_keys], dtype=np.int64)

    edge_index = graph["edge_index"].astype(np.int64)
    node_static = graph["node_static"].astype(np.float32)
    action_node_map = graph["action_node_map"].astype(np.float32)
    node_max_depth = node_static[:, 1].copy()
    node_max_depth[node_max_depth < 0.1] = 5.0
    try:
        priority_node_indices = get_pfv_core_node_indices(list(graph["node_ids"]))
    except Exception as exc:
        raise PriorityContractError(f"Failed to resolve PFV core 8 indices: {exc}") from exc

    def _label(name: str, fallback: np.ndarray) -> np.ndarray:
        if name in df.columns:
            return pd.to_numeric(df[name], errors="raise").to_numpy(np.float32)
        return fallback.astype(np.float32)

    pfv_safe = _label("pfv_safe_label", pfv_delta <= 0.0)
    tfv_improved = _label("tfv_improved_label", tfv_delta <= 0.0)
    peak_noninferior = _label("peak_noninferior_label", peak_delta <= 0.0)

    return {
        "state_history": torch.from_numpy(state_history),
        "rainfall": torch.from_numpy(rainfall),
        "action_candidate": torch.from_numpy(action_candidate),
        "action_reference": torch.from_numpy(action_reference),
        "action_dynamic_internal": torch.from_numpy(action_di),
        "action_hold_previous": torch.from_numpy(action_hold),
        "depth_candidate": torch.from_numpy(depth_candidate),
        "depth_reference": torch.from_numpy(depth_reference),
        "depth_dynamic_internal": torch.from_numpy(depth_di),
        "depth_hold_previous": torch.from_numpy(depth_hold),
        "pfv_delta": torch.from_numpy(pfv_delta),
        "tfv_delta": torch.from_numpy(tfv_delta),
        "peak_delta": torch.from_numpy(peak_delta),
        "pfv_safe_label": torch.from_numpy(pfv_safe),
        "tfv_improved_label": torch.from_numpy(tfv_improved),
        "peak_noninferior_label": torch.from_numpy(peak_noninferior),
        "state_group_index": torch.from_numpy(state_group_index),
        "event_ids": event_ids,
        "cv_group_ids": cv_group_ids,
        "event_indices": torch.from_numpy(event_indices),
        "unique_events": unique_groups,
        "unique_states": unique_states,
        "edge_index": torch.from_numpy(edge_index),
        "node_static": torch.from_numpy(node_static),
        "action_node_map": torch.from_numpy(action_node_map),
        "node_max_depth": torch.from_numpy(node_max_depth),
        "priority_node_indices": torch.tensor(priority_node_indices, dtype=torch.long),
        "n_nodes": n_nodes,
        "n_facilities": n_facilities,
    }


def make_event_grouped_folds(
    event_indices: torch.Tensor,
    unique_events: list[str],
    n_folds: int = 5,
    seed: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Group-held-out folds; ``event_indices`` may represent rainfall fingerprints."""
    rng = np.random.RandomState(seed)
    events = np.arange(len(unique_events))
    rng.shuffle(events)
    fold_events = np.array_split(events, n_folds)
    folds = []
    sample_groups = event_indices.cpu().numpy()
    for fold_idx in range(n_folds):
        val_event_set = set(fold_events[fold_idx].tolist())
        sample_val = np.asarray([g in val_event_set for g in sample_groups], dtype=bool)
        folds.append((~sample_val, sample_val))
    return folds


class SimplifiedDynamicsModel(nn.Module):
    """Direct MLP ablation that still honours all four reference branches."""

    def __init__(
        self,
        n_nodes: int,
        n_facilities: int,
        n_static_features: int = 7,
        hidden_dim: int = 32,
        horizon: int = 12,
        history_frames: int = 13,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.n_nodes = int(n_nodes)
        self.n_facilities = int(n_facilities)
        self.hidden_dim = int(hidden_dim)
        self.horizon = int(horizon)
        self.history_frames = int(history_frames)
        from .models_v42.graph_state_encoder import GraphStateEncoder
        from .models_v42.rainfall_encoder import RainfallEncoder
        from .models_v42.actuator_action_encoder import ActuatorActionEncoder

        self.graph_encoder = GraphStateEncoder(
            n_nodes=n_nodes,
            n_static_features=n_static_features,
            hidden_dim=hidden_dim,
            gat_heads=4,
            n_gat_layers=1,
            dropout=dropout,
        )
        self.rainfall_encoder = RainfallEncoder(horizon=horizon, hidden_dim=hidden_dim)
        self.action_encoder = ActuatorActionEncoder(
            n_facilities=n_facilities, hidden_dim=hidden_dim, horizon=horizon
        )
        feat_dim = self.graph_encoder.output_dim + hidden_dim + hidden_dim
        self.direct_mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
        )
        self.depth_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def _branch(
        self,
        h_state_4d: torch.Tensor,
        rain_4d: torch.Tensor,
        action: torch.Tensor,
        action_node_map: torch.Tensor,
    ) -> torch.Tensor:
        act = self.action_encoder(action, action_node_map)
        feat = torch.cat([h_state_4d, rain_4d, act], dim=-1)
        return self.depth_head(self.direct_mlp(feat)).squeeze(-1)

    def forward(
        self,
        state_history,
        rainfall,
        action_candidate,
        action_reference,
        edge_index,
        node_static,
        action_node_map,
        action_dynamic_internal=None,
        action_hold_previous=None,
    ):
        if action_dynamic_internal is None or action_hold_previous is None:
            raise ValueError("Simplified V4.2 model requires DI and Hold actions")
        B = state_history.shape[0]
        N = self.n_nodes
        h_state = self.graph_encoder(state_history, edge_index, node_static)
        h_state = h_state.reshape(B, N, -1)
        h_state_4d = h_state.unsqueeze(1).expand(-1, self.horizon, -1, -1)
        rain = self.rainfall_encoder(rainfall)
        rain_4d = rain.unsqueeze(2).expand(-1, -1, N, -1)
        y_c = self._branch(h_state_4d, rain_4d, action_candidate, action_node_map)
        y_nc = self._branch(h_state_4d, rain_4d, action_reference, action_node_map)
        y_di = self._branch(h_state_4d, rain_4d, action_dynamic_internal, action_node_map)
        y_hold = self._branch(h_state_4d, rain_4d, action_hold_previous, action_node_map)
        return {
            "y_candidate": y_c,
            "y_reference": y_nc,
            "y_dynamic_internal": y_di,
            "y_hold_previous": y_hold,
            "delta": y_c - y_nc,
            "delta_di": y_c - y_di,
        }


class TwinWithKPIHeads(nn.Module):
    """KPI heads with explicit NC-relative PFV and DI-relative TFV/Peak features."""

    def __init__(self, base_model: nn.Module, hidden_dim: int = 32):
        super().__init__()
        self.base = base_model
        self.n_nodes = base_model.n_nodes
        self.horizon = base_model.horizon
        self.hidden_dim = int(hidden_dim)
        self.delta_pool = nn.Sequential(nn.Linear(self.n_nodes, hidden_dim), nn.ReLU())
        self.action_pool = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        self.delta_pool_di = nn.Sequential(nn.Linear(self.n_nodes, hidden_dim), nn.ReLU())
        self.action_pool_di = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
        kpi_in = 2 * hidden_dim
        self.pfv_hurdle = nn.Sequential(
            nn.Linear(kpi_in, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
        self.tfv_head = nn.Sequential(
            nn.Linear(kpi_in, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
        self.peak_head = nn.Sequential(
            nn.Linear(kpi_in, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def _compute_action_diff_features(self, action_a, action_b, action_node_map):
        h_a = self.base.action_encoder(action_a, action_node_map)
        h_b = self.base.action_encoder(action_b, action_node_map)
        diff = h_a - h_b
        actuated = action_node_map.abs().sum(dim=0) > 0
        if not bool(actuated.any()):
            raise ValueError("action_node_map contains no actuated graph nodes")
        return diff[:, :, actuated, :].sum(dim=(1, 2))

    def forward(self, **kwargs):
        if kwargs.get("action_dynamic_internal") is None:
            raise ValueError("TwinWithKPIHeads requires action_dynamic_internal")
        if kwargs.get("action_hold_previous") is None:
            raise ValueError("TwinWithKPIHeads requires action_hold_previous")
        out = self.base(**kwargs)
        if "delta_di" not in out or "y_hold_previous" not in out:
            raise RuntimeError("Base model did not produce required DI/Hold branches")

        action_c = kwargs["action_candidate"]
        action_nc = kwargs["action_reference"]
        action_di = kwargs["action_dynamic_internal"]
        action_node_map = kwargs["action_node_map"]

        delta_nc_global = self.delta_pool(out["delta"].mean(dim=1))
        act_nc_global = self.action_pool(
            self._compute_action_diff_features(action_c, action_nc, action_node_map)
        )
        out["pfv_delta"] = self.pfv_hurdle(
            torch.cat([delta_nc_global, act_nc_global], dim=-1)
        ).squeeze(-1)

        delta_di_global = self.delta_pool_di(out["delta_di"].mean(dim=1))
        act_di_global = self.action_pool_di(
            self._compute_action_diff_features(action_c, action_di, action_node_map)
        )
        di_feat = torch.cat([delta_di_global, act_di_global], dim=-1)
        out["tfv_delta"] = self.tfv_head(di_feat).squeeze(-1)
        out["peak_delta"] = self.peak_head(di_feat).squeeze(-1)
        return out


def compute_kpi_normalization_stats(
    data: dict,
    train_idx: np.ndarray | None = None,
) -> dict[str, tuple[float, float]]:
    """Compute KPI z-score statistics; callers doing CV must pass train_idx."""
    stats: dict[str, tuple[float, float]] = {}
    for key in ("pfv_delta", "tfv_delta", "peak_delta"):
        vals = data[key].detach().cpu().numpy()
        if train_idx is not None:
            vals = vals[np.asarray(train_idx, dtype=np.int64)]
        mean = float(np.mean(vals))
        std = float(np.std(vals))
        if std < 1e-8:
            std = 1.0
        stats[key] = (mean, std)
    return stats


def _apply_kpi_normalization(batch, kpi_stats):
    for key, (mean, std) in kpi_stats.items():
        if key in batch:
            batch[key] = (batch[key] - mean) / std
    return batch


def _denormalize_kpi_predictions(preds, kpi_stats):
    out = {}
    for key, arr in preds.items():
        if key in kpi_stats:
            mean, std = kpi_stats[key]
            out[key] = arr * std + mean
        else:
            out[key] = arr
    return out


def compute_r2(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=float)
    target = np.asarray(target, dtype=float)
    ss_res = float(np.sum((target - pred) ** 2))
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    if ss_tot <= 1e-12:
        return float("nan")
    return float(1.0 - ss_res / ss_tot)


def compute_metrics(preds, targets):
    metrics = {}
    for key in ("pfv_delta", "tfv_delta", "peak_delta"):
        if key not in preds or key not in targets:
            continue
        p = np.asarray(preds[key])
        t = np.asarray(targets[key])
        metrics[f"{key}_r2"] = compute_r2(p, t)
        metrics[f"{key}_mae"] = float(np.mean(np.abs(p - t)))
        metrics[f"{key}_sign_acc"] = float(np.mean(np.sign(p) == np.sign(t)))
    return metrics


def _freeze_graph_encoder(model):
    base = model.base if hasattr(model, "base") else model
    for p in base.graph_encoder.parameters():
        p.requires_grad = False


def _unfreeze_all(model):
    for p in model.parameters():
        p.requires_grad = True


def _count_trainable(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _forward_model(model: nn.Module, batch: dict[str, torch.Tensor]):
    """Single source of truth for the four-branch forward contract."""
    required = (
        "state_history",
        "rainfall",
        "action_candidate",
        "action_reference",
        "action_dynamic_internal",
        "action_hold_previous",
        "edge_index",
        "node_static",
        "action_node_map",
    )
    missing = [k for k in required if k not in batch]
    if missing:
        raise KeyError(f"Four-branch forward missing tensors: {missing}")
    return model(
        state_history=batch["state_history"],
        rainfall=batch["rainfall"],
        action_candidate=batch["action_candidate"],
        action_reference=batch["action_reference"],
        action_dynamic_internal=batch["action_dynamic_internal"],
        action_hold_previous=batch["action_hold_previous"],
        edge_index=batch["edge_index"],
        node_static=batch["node_static"],
        action_node_map=batch["action_node_map"],
    )


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    scaler,
    device,
    loss_fn,
    amp=True,
    max_grad_norm=1.0,
    shared_tensors=None,
):
    model.train()
    epoch_losses: dict[str, float] = {}
    n_batches = 0
    t0 = time.time()
    for batch in dataloader:
        b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        if shared_tensors:
            b.update({k: v.to(device) for k, v in shared_tensors.items()})
        with autocast(enabled=amp):
            pred = _forward_model(model, b)
            loss_dict = loss_fn(pred, b)
            loss = sum(v for v in loss_dict.values() if torch.is_tensor(v))
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        for k, v in loss_dict.items():
            epoch_losses[k] = epoch_losses.get(k, 0.0) + (
                v.detach().item() if torch.is_tensor(v) else float(v)
            )
        n_batches += 1
    avg = {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}
    avg["epoch_time"] = time.time() - t0
    return avg


@torch.no_grad()
def validate(
    model,
    dataloader,
    device,
    loss_fn,
    shared_tensors=None,
    kpi_stats=None,
):
    model.eval()
    all_preds = {"pfv_delta": [], "tfv_delta": [], "peak_delta": []}
    all_targets = {"pfv_delta": [], "tfv_delta": [], "peak_delta": []}
    epoch_losses: dict[str, float] = {}
    n_batches = 0
    for batch in dataloader:
        b = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        if shared_tensors:
            b.update({k: v.to(device) for k, v in shared_tensors.items()})
        pred = _forward_model(model, b)
        for key in all_preds:
            if key in pred:
                all_preds[key].append(pred[key].cpu().numpy())
                all_targets[key].append(b[key].cpu().numpy())
        loss_dict = loss_fn(pred, b)
        for k, v in loss_dict.items():
            epoch_losses[k] = epoch_losses.get(k, 0.0) + (
                v.detach().item() if torch.is_tensor(v) else float(v)
            )
        n_batches += 1
    avg = {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}
    preds_np = {k: np.concatenate(v) for k, v in all_preds.items() if v}
    targets_np = {k: np.concatenate(v) for k, v in all_targets.items() if v}
    if kpi_stats:
        preds_np = _denormalize_kpi_predictions(preds_np, kpi_stats)
        targets_np = _denormalize_kpi_predictions(targets_np, kpi_stats)
    avg.update(compute_metrics(preds_np, targets_np))
    return avg


def _make_loss_fns(
    stage,
    n_nodes,
    node_max_depth,
    edge_index,
    device,
    ablation_mode="full_4stage",
    kpi_stats=None,
):
    from .models_v42.trajectory_losses import TrajectoryLosses
    from .models_v42.physics_losses import PhysicsLosses
    from .models_v42.ranking_losses import RankingLosses

    abl = ABLATION_CONFIGS.get(ablation_mode, ABLATION_CONFIGS["full_4stage"])
    norm_std = {k: v[1] for k, v in kpi_stats.items()} if kpi_stats else None
    traj_loss = TrajectoryLosses(norm_std=norm_std).to(device)
    phys_loss = (
        PhysicsLosses(n_nodes=n_nodes, node_max_depth=node_max_depth).to(device)
        if abl["use_physics"]
        else None
    )
    rank_loss = RankingLosses().to(device)

    def add_valid_physics(result, pred, weight):
        if phys_loss is None:
            return
        physics = phys_loss(pred)
        # Only physically defensible/currently supported terms enter training.
        for key in ("non_negative", "capacity_bounds", "storage_continuity"):
            result[f"phys_{key}"] = physics[key] * weight

    def loss_a(pred, target):
        losses = traj_loss(pred, target)
        return {"depth_traj": losses["depth_trajectory"]}

    def loss_b(pred, target):
        losses = traj_loss(pred, target)
        result = {
            "depth_traj": losses["depth_trajectory"],
            "delta_traj": losses["delta_trajectory"],
        }
        add_valid_physics(result, pred, 0.1)
        return result

    def loss_c(pred, target):
        return loss_b(pred, target)

    def loss_d(pred, target):
        losses = traj_loss(pred, target)
        result = {
            "depth_traj": losses["depth_trajectory"],
            "delta_traj": losses["delta_trajectory"],
            "pfv_kpi": losses["pfv_kpi"],
            "tfv_kpi": losses["tfv_kpi"],
            "peak_kpi": losses["peak_kpi"],
        }
        add_valid_physics(result, pred, 0.05)
        ranking = rank_loss(pred, target)
        for key, value in ranking.items():
            if key == "valid_pair_count":
                continue
            result[f"rank_{key}"] = value * 0.2
        return result

    return {"A": loss_a, "B": loss_b, "C": loss_c, "D": loss_d}[stage]


def _build_model(
    stage,
    n_nodes,
    n_facilities,
    node_max_depth,
    hidden_dim=HIDDEN_DIM,
    ablation_mode="full_4stage",
):
    from .models_v42.counterfactual_twin_dynamics import TwinGraphDynamics

    abl = ABLATION_CONFIGS.get(ablation_mode, ABLATION_CONFIGS["full_4stage"])
    if abl["simplified_dyn"]:
        base = SimplifiedDynamicsModel(
            n_nodes=n_nodes,
            n_facilities=n_facilities,
            n_static_features=7,
            hidden_dim=hidden_dim,
            horizon=N_HORIZON,
            history_frames=N_HISTORY,
        )
    else:
        base = TwinGraphDynamics(
            n_nodes=n_nodes,
            n_facilities=n_facilities,
            n_static_features=7,
            hidden_dim=hidden_dim,
            gat_heads=GAT_HEADS,
            n_gat_layers=2,
            horizon=N_HORIZON,
            history_frames=N_HISTORY,
        )
    return TwinWithKPIHeads(base, hidden_dim=hidden_dim) if stage == "D" else base


def _make_shared_tensors(data, device):
    shared = {
        "edge_index": data["edge_index"].to(device),
        "node_static": data["node_static"].to(device),
        "action_node_map": data["action_node_map"].to(device),
    }
    if "priority_node_indices" in data:
        shared["priority_node_indices"] = data["priority_node_indices"].to(device)
    return shared


def _make_batch(data, indices, kpi_stats=None):
    idx = torch.as_tensor(np.asarray(indices, dtype=np.int64), dtype=torch.long)
    sample_keys = (
        "state_history",
        "rainfall",
        "action_candidate",
        "action_reference",
        "action_dynamic_internal",
        "action_hold_previous",
        "depth_candidate",
        "depth_reference",
        "depth_dynamic_internal",
        "depth_hold_previous",
        "pfv_delta",
        "tfv_delta",
        "peak_delta",
        "pfv_safe_label",
        "tfv_improved_label",
        "peak_noninferior_label",
        "state_group_index",
    )
    batch = {}
    for key in sample_keys:
        if key not in data:
            continue
        value = data[key]
        if not torch.is_tensor(value):
            raise TypeError(f"Sample field {key} must be a tensor")
        # Critical: every per-sample field uses the same selected indices.
        batch[key] = value[idx].clone()

    if kpi_stats is not None:
        _apply_kpi_normalization(batch, kpi_stats)
        pfv_mean, pfv_std = kpi_stats["pfv_delta"]
        peak_mean, peak_std = kpi_stats["peak_delta"]
        n = len(idx)
        batch["pfv_boundary_norm"] = torch.full(
            (n,), (0.0 - pfv_mean) / pfv_std, dtype=torch.float32
        )
        batch["peak_boundary_norm"] = torch.full(
            (n,), (0.0 - peak_mean) / peak_std, dtype=torch.float32
        )
    return batch


class _DictDataset(torch.utils.data.Dataset):
    def __init__(self, data_dict):
        self.data = data_dict
        lengths = {k: int(v.shape[0]) for k, v in data_dict.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"Batch fields are misaligned: {lengths}")
        self.n = next(iter(lengths.values()))

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


def train_v42_twin(
    project_root: Path,
    output_root: Path,
    config: dict | None = None,
    ablation_mode: str = "full_4stage",
) -> dict[str, Any]:
    project_root = Path(project_root)
    output_root = Path(output_root)
    abl = ABLATION_CONFIGS.get(ablation_mode, ABLATION_CONFIGS["full_4stage"])
    model_dir = output_root / "models" / (
        "v42_twin" if ablation_mode == "full_4stage" else f"v42_twin_ablation_{ablation_mode}"
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_v42_training_data(project_root, output_root)
    cfg = {**TRAINING_CONFIG, **(config or {})}
    seeds = cfg["seeds"]
    n_folds = cfg["n_cv_folds"]
    batch_size = cfg["batch_size"]
    amp_enabled = cfg["amp_enabled"] and device.type == "cuda"
    all_seed_results = []
    fold_norm_stats: dict[str, dict] = {}

    for seed in seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        folds = make_event_grouped_folds(
            data["event_indices"], data["unique_events"], n_folds, seed
        )
        fold_results = []
        for fold_idx, (train_mask, val_mask) in enumerate(folds):
            train_idx = np.where(train_mask)[0]
            val_idx = np.where(val_mask)[0]
            # P0 fix: fit target scaling on this fold's training partition only.
            kpi_stats = compute_kpi_normalization_stats(data, train_idx=train_idx)
            fold_norm_stats[f"seed{seed}_fold{fold_idx}"] = {
                k: {"mean": m, "std": s} for k, (m, s) in kpi_stats.items()
            }
            train_data = _make_batch(data, train_idx, kpi_stats)
            val_data = _make_batch(data, val_idx, kpi_stats)
            train_loader = DataLoader(
                _DictDataset(train_data), batch_size=batch_size, shuffle=True
            )
            val_loader = DataLoader(
                _DictDataset(val_data), batch_size=batch_size, shuffle=False
            )
            shared = _make_shared_tensors(data, device)
            history = []
            model = None

            for stage_name in ("A", "B", "C", "D"):
                if stage_name == "D" and model is not None:
                    model = TwinWithKPIHeads(model, hidden_dim=HIDDEN_DIM).to(device)
                elif model is None:
                    model = _build_model(
                        stage_name,
                        data["n_nodes"],
                        data["n_facilities"],
                        data["node_max_depth"],
                        ablation_mode=ablation_mode,
                    ).to(device)
                if stage_name in ("A", "B"):
                    _freeze_graph_encoder(model)
                else:
                    _unfreeze_all(model)

                lr = cfg["lr_stage_d"] if stage_name == "D" else cfg["learning_rate"]
                optimizer = torch.optim.AdamW(
                    [p for p in model.parameters() if p.requires_grad],
                    lr=lr,
                    weight_decay=cfg["weight_decay"],
                )
                scaler = GradScaler(enabled=amp_enabled)
                loss_fn = _make_loss_fns(
                    stage_name,
                    data["n_nodes"],
                    data["node_max_depth"],
                    data["edge_index"],
                    device,
                    ablation_mode=ablation_mode,
                    kpi_stats=kpi_stats,
                )
                best_val = float("inf")
                best_state = None
                patience = 0
                for epoch in range(STAGE_EPOCHS[stage_name]):
                    train_metrics = train_one_epoch(
                        model,
                        train_loader,
                        optimizer,
                        scaler,
                        device,
                        loss_fn,
                        amp=amp_enabled,
                        max_grad_norm=cfg["max_grad_norm"],
                        shared_tensors=shared,
                    )
                    val_metrics = validate(
                        model,
                        val_loader,
                        device,
                        loss_fn,
                        shared_tensors=shared,
                        kpi_stats=kpi_stats,
                    )
                    train_loss = sum(
                        v for k, v in train_metrics.items() if k != "epoch_time"
                    )
                    val_loss = sum(
                        v
                        for k, v in val_metrics.items()
                        if not k.endswith(("_r2", "_mae", "_sign_acc"))
                        and k != "epoch_time"
                        and np.isfinite(v)
                    )
                    rec = {
                        "stage": stage_name,
                        "epoch": epoch,
                        "seed": seed,
                        "fold": fold_idx,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        **{f"train_{k}": v for k, v in train_metrics.items()},
                        **{f"val_{k}": v for k, v in val_metrics.items()},
                    }
                    history.append(rec)
                    if val_loss < best_val:
                        best_val = val_loss
                        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                        patience = 0
                    else:
                        patience += 1
                    if patience >= cfg["early_stop_patience"]:
                        break
                if best_state is not None:
                    model.load_state_dict(best_state)

            final_metrics = validate(
                model,
                val_loader,
                device,
                loss_fn,
                shared_tensors=shared,
                kpi_stats=kpi_stats,
            )
            checkpoint_path = model_dir / f"v42_twin_model_seed{seed}_fold{fold_idx}.pt"
            torch.save(
                {
                    "seed": seed,
                    "fold": fold_idx,
                    "ablation_mode": ablation_mode,
                    "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                    "model_config": {
                        "n_nodes": data["n_nodes"],
                        "n_facilities": data["n_facilities"],
                        "hidden_dim": HIDDEN_DIM,
                        "gat_heads": GAT_HEADS,
                        "n_history": N_HISTORY,
                        "n_horizon": N_HORIZON,
                    },
                    "kpi_normalization_stats": fold_norm_stats[
                        f"seed{seed}_fold{fold_idx}"
                    ],
                    "final_metrics": final_metrics,
                },
                checkpoint_path,
            )
            with open(
                model_dir / f"training_history_seed{seed}_fold{fold_idx}.json", "w"
            ) as f:
                json.dump(history, f, indent=1, default=str)
            fold_results.append(
                {
                    "fold": fold_idx,
                    "n_train": len(train_idx),
                    "n_val": len(val_idx),
                    "final_metrics": {k: float(v) for k, v in final_metrics.items()},
                    "training_history": _summarize_training_history(history),
                    "kpi_normalization_stats": fold_norm_stats[
                        f"seed{seed}_fold{fold_idx}"
                    ],
                }
            )
        if len(fold_results) != n_folds:
            raise RuntimeError(f"Seed {seed}: expected {n_folds} folds, got {len(fold_results)}")
        if any(x["seed"] == seed for x in all_seed_results):
            raise RuntimeError(f"Duplicate seed {seed} would overwrite training evidence")
        all_seed_results.append(
            {"seed": seed, "folds": fold_results, "aggregate": _aggregate_fold_metrics(fold_results)}
        )

    combined = {
        "ablation_mode": ablation_mode,
        "ablation_description": abl["description"],
        "seeds": seeds,
        "n_folds": n_folds,
        "n_samples": len(data["pfv_delta"]),
        "n_events": len(data["unique_events"]),
        "per_seed": all_seed_results,
        "overall_aggregate": _aggregate_seed_results(all_seed_results),
        "normalization_policy": "train_fold_only",
    }
    with open(model_dir / "training_history.json", "w") as f:
        json.dump(combined, f, indent=2, default=str)
    with open(model_dir / "cv_metrics.json", "w") as f:
        json.dump(
            {
                "seeds": seeds,
                "n_folds": n_folds,
                "per_seed_folds": [
                    {"seed": s["seed"], "folds": s["folds"]} for s in all_seed_results
                ],
                "aggregate": combined["overall_aggregate"],
            },
            f,
            indent=2,
            default=str,
        )
    with open(model_dir / "kpi_normalization_stats.json", "w") as f:
        json.dump(
            {"policy": "train_fold_only", "per_seed_fold": fold_norm_stats},
            f,
            indent=2,
        )
    return combined


def _summarize_training_history(history):
    if not history:
        return {"n_epochs": 0, "stages": {}}
    stages: dict[str, list[dict]] = {}
    for rec in history:
        stages.setdefault(rec.get("stage", "?"), []).append(rec)
    summary = {"n_epochs": len(history), "stages": {}}
    for name, recs in sorted(stages.items()):
        first, last = recs[0], recs[-1]
        summary["stages"][name] = {
            "n_epochs": len(recs),
            "first_epoch": {"train_loss": first.get("train_loss"), "val_loss": first.get("val_loss")},
            "last_epoch": {"train_loss": last.get("train_loss"), "val_loss": last.get("val_loss")},
            "best_val_loss": min(r.get("val_loss", float("inf")) for r in recs),
        }
        for key in (
            "val_pfv_delta_r2",
            "val_tfv_delta_r2",
            "val_peak_delta_r2",
            "val_pfv_delta_sign_acc",
            "val_tfv_delta_sign_acc",
        ):
            vals = [r[key] for r in recs if key in r and r[key] is not None]
            if vals:
                summary["stages"][name][key] = {
                    "first": vals[0],
                    "last": vals[-1],
                    "best": max(vals),
                }
    return summary


def _aggregate_fold_metrics(fold_results):
    keys = set()
    for f in fold_results:
        keys.update(f["final_metrics"].keys())
    out = {}
    for key in sorted(keys):
        vals = [f["final_metrics"][key] for f in fold_results if key in f["final_metrics"]]
        vals = [v for v in vals if np.isfinite(v)]
        if vals:
            out[f"{key}_mean"] = float(np.mean(vals))
            out[f"{key}_std"] = float(np.std(vals))
    return out


def _aggregate_seed_results(seed_results):
    keys = set()
    for s in seed_results:
        keys.update(s["aggregate"].keys())
    out = {}
    for key in sorted(keys):
        vals = [s["aggregate"][key] for s in seed_results if key in s["aggregate"]]
        vals = [v for v in vals if np.isfinite(v)]
        if vals:
            out[key] = float(np.mean(vals))
    return out
