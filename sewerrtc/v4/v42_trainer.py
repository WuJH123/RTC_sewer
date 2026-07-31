"""V4.2 TwinGraphDynamics training pipeline.

4-stage curriculum with event-grouped CV across 5 seeds:
  A: One-step dynamics (40 ep) — depth MSE only, freeze graph encoder
  B: Multi-step rollout (60 ep) — depth + physics losses, freeze graph encoder
  C: Counterfactual twin (40 ep) — full trajectory + physics, train all
  D: Decision-focused (20 ep) — KPI heads + ranking, lower LR
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any

from sewerrtc.v4.v42_priority_contract import PFV_CORE_8_IDS, get_pfv_core_node_indices, PriorityContractError

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_NODES = 932
N_EDGES = 1276
N_FACILITIES = 36
N_HISTORY = 13  # 13 frames × 5 min = 60 min look-back
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

# ---------------------------------------------------------------------------
# Ablation experiment configurations
# ---------------------------------------------------------------------------
# Each variant modifies the training pipeline to test the contribution of
# a specific component. "full_4stage" is the baseline (no changes).
#
# Keys:
#   use_physics   : include physics losses in stages B/C/D
#   use_twin      : train both candidate & reference (False = candidate only)
#   simplified_dyn: replace GRU dynamics with direct MLP, reduce GAT layers
#   loss_override : optional dict of stage -> loss weight overrides
# ---------------------------------------------------------------------------

ABLATION_CONFIGS = {
    "full_4stage": {
        "description": "Full 4-stage pipeline (baseline)",
        "use_physics": True,
        "use_twin": True,
        "simplified_dyn": False,
        "loss_override": {},
    },
    "no_physics": {
        "description": "Remove all physics losses (Stages B/C/D)",
        "use_physics": False,
        "use_twin": True,
        "simplified_dyn": False,
        "loss_override": {},
    },
    "no_twin": {
        "description": "Single-tower: train candidate only, no reference",
        "use_physics": True,
        "use_twin": False,
        "simplified_dyn": False,
        "loss_override": {},
    },
    "simplified_dynamics": {
        "description": "Replace GRU with direct MLP, reduce GAT to 1 layer",
        "use_physics": True,
        "use_twin": True,
        "simplified_dyn": True,
        "loss_override": {},
    },
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _parse_json_array(s: str) -> np.ndarray:
    """Parse a JSON-encoded array string into numpy array."""
    return np.array(json.loads(s), dtype=np.float32)


def load_graph_topology(project_root: Path) -> dict:
    """Load edge_index, node_static, action_node_map from INP."""
    from .v42_trajectory_builder import _load_graph_topology
    return _load_graph_topology(Path(project_root))


def load_v42_training_data(
    project_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Load trajectory dataset and prepare PyTorch tensors."""
    project_root = Path(project_root)
    output_root = Path(output_root)
    trajectory_dir = output_root / "v42" / "trajectory_dataset"

    # Read manifest
    parquet_path = trajectory_dir / "trajectory_manifest_v42.parquet"
    if not parquet_path.exists():
        csv_path = trajectory_dir / "trajectory_manifest_v42.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
        else:
            raise FileNotFoundError(f"No trajectory manifest found in {trajectory_dir}")
    else:
        df = pd.read_parquet(parquet_path)

    N = len(df)
    logger.info("Loading %d samples from trajectory dataset", N)

    # Parse JSON array columns
    logger.info("Parsing history_depth...")
    state_history = np.stack([_parse_json_array(s) for s in df["history_depth"]])  # [N,7,932]

    logger.info("Parsing rainfall_forecast...")
    rainfall = np.stack([_parse_json_array(s) for s in df["rainfall_forecast"]])  # [N,12]

    logger.info("Parsing action sequences...")
    action_candidate = np.stack([_parse_json_array(s) for s in df["candidate_action_seq"]])  # [N,12,36]
    action_reference = np.stack([_parse_json_array(s) for s in df["ref_no_control_action_seq"]])  # [N,12,36]

    # Dynamic Internal action sequence (for TFV/Peak DI-relative targets)
    if "ref_dynamic_internal_action_seq" in df.columns:
        logger.info("Parsing DI action sequences...")
        action_di = np.stack([_parse_json_array(s) for s in df["ref_dynamic_internal_action_seq"]])  # [N,12,36]
    else:
        logger.warning("ref_dynamic_internal_action_seq not found; falling back to NC")
        action_di = action_reference.copy()

    # Hold Previous action sequence
    if "ref_hold_previous_action_seq" in df.columns:
        logger.info("Parsing Hold Previous action sequences...")
        action_hold = np.stack([_parse_json_array(s) for s in df["ref_hold_previous_action_seq"]])  # [N,12,36]
    else:
        logger.warning("ref_hold_previous_action_seq not found; falling back to NC")
        action_hold = action_reference.copy()

    logger.info("Parsing trajectory depths (4 branches)...")
    depth_candidate = np.stack([_parse_json_array(s) for s in df["trajectory_depth_candidate"]])  # [N,12,932]
    depth_reference = np.stack([_parse_json_array(s) for s in df["trajectory_depth_no_control"]])  # [N,12,932]

    # DI trajectory
    if "trajectory_depth_dynamic_internal" in df.columns:
        depth_di = np.stack([_parse_json_array(s) for s in df["trajectory_depth_dynamic_internal"]])  # [N,12,932]
    else:
        logger.warning("trajectory_depth_dynamic_internal not found; falling back to NC")
        depth_di = depth_reference.copy()

    # Hold Previous trajectory
    if "trajectory_depth_hold_previous" in df.columns:
        depth_hold = np.stack([_parse_json_array(s) for s in df["trajectory_depth_hold_previous"]])  # [N,12,932]
    else:
        logger.warning("trajectory_depth_hold_previous not found; falling back to NC")
        depth_hold = depth_reference.copy()

    # KPI labels
    pfv_delta = df["pfv_delta"].values.astype(np.float32)
    tfv_delta = df["tfv_delta"].values.astype(np.float32)
    peak_delta = df["peak_delta"].values.astype(np.float32)

    # Event IDs for grouped CV
    event_ids = df["event_id"].values.astype(str)
    unique_events = sorted(set(event_ids))
    event_to_idx = {e: i for i, e in enumerate(unique_events)}
    event_indices = np.array([event_to_idx[e] for e in event_ids], dtype=np.int64)

    # Load graph topology
    logger.info("Loading graph topology...")
    graph = load_graph_topology(project_root)
    edge_index = graph["edge_index"].astype(np.int64)  # [2, E]
    node_static = graph["node_static"].astype(np.float32)  # [N, F]
    action_node_map = graph["action_node_map"].astype(np.float32)  # [36, N]
    n_nodes = graph["n_nodes"]

    # Node max depth for physics losses
    node_max_depth = node_static[:, 1].copy()  # max_depth column
    node_max_depth[node_max_depth < 0.1] = 5.0  # default

    # Priority node indices (for PFV computation) — fail-closed via contract
    node_ids = graph["node_ids"]
    try:
        priority_node_indices = get_pfv_core_node_indices(list(node_ids))
    except Exception as exc:
        raise PriorityContractError(
            f"Failed to resolve PFV core 8 indices: {exc}"
        ) from exc
    priority_indices_arr = np.array(priority_node_indices, dtype=np.int64)

    # Safety labels (if available)
    pfv_safe = df["pfv_safe_label"].values.astype(np.float32) if "pfv_safe_label" in df.columns else np.zeros(N, dtype=np.float32)
    tfv_improved = df["tfv_improved_label"].values.astype(np.float32) if "tfv_improved_label" in df.columns else np.zeros(N, dtype=np.float32)
    peak_noninferior = df["peak_noninferior_label"].values.astype(np.float32) if "peak_noninferior_label" in df.columns else np.zeros(N, dtype=np.float32)

    logger.info("Data loaded: %d samples, %d nodes, %d events",
                N, n_nodes, len(unique_events))

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
        "event_ids": event_ids,
        "event_indices": torch.from_numpy(event_indices),
        "unique_events": unique_events,
        "edge_index": torch.from_numpy(edge_index),
        "node_static": torch.from_numpy(node_static),
        "action_node_map": torch.from_numpy(action_node_map),
        "node_max_depth": torch.from_numpy(node_max_depth),
        "priority_node_indices": torch.from_numpy(priority_indices_arr),
        "n_nodes": n_nodes,
        "n_facilities": graph["n_facilities"],
    }


# ---------------------------------------------------------------------------
# Event-grouped CV fold splitting
# ---------------------------------------------------------------------------

def make_event_grouped_folds(
    event_indices: torch.Tensor,
    unique_events: list[str],
    n_folds: int = 5,
    seed: int = 0,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split events into n_folds groups, return (train_idx, val_idx) sample masks."""
    rng = np.random.RandomState(seed)
    events = np.arange(len(unique_events))
    rng.shuffle(events)
    fold_events = np.array_split(events, n_folds)

    folds = []
    for fold_idx in range(n_folds):
        val_event_set = set(fold_events[fold_idx].tolist())
        val_mask = np.array([e in val_event_set for e in range(len(unique_events))])
        sample_val = val_mask[event_indices.numpy()]
        sample_train = ~sample_val
        folds.append((sample_train, sample_val))
    return folds


# ---------------------------------------------------------------------------
# KPI head augmented model
# ---------------------------------------------------------------------------

class SimplifiedDynamicsModel(nn.Module):
    """Ablation variant: replaces GRU dynamics with direct MLP.

    Instead of autoregressive GRU rollout, directly maps encoded features
    to 12-step depth trajectory via a fully-connected head.
    Uses 1-layer GAT instead of 2.
    """

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

        # 1-layer GAT (reduced from 2)
        self.graph_encoder = GraphStateEncoder(
            n_nodes=n_nodes,
            n_static_features=n_static_features,
            hidden_dim=hidden_dim,
            gat_heads=4,
            n_gat_layers=1,
            dropout=dropout,
        )
        state_embed_dim = self.graph_encoder.output_dim

        self.rainfall_encoder = RainfallEncoder(
            horizon=horizon, hidden_dim=hidden_dim,
        )
        self.action_encoder = ActuatorActionEncoder(
            n_facilities=n_facilities, hidden_dim=hidden_dim, horizon=horizon,
        )

        # Direct MLP instead of GRU (no autoregressive rollout)
        feat_dim = state_embed_dim + hidden_dim + hidden_dim
        self.direct_mlp = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        # Per-node depth head (single value per node per step)
        self.depth_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

        self._edge_cache = {}

    def forward(self, state_history, rainfall, action_candidate,
                action_reference, edge_index, node_static, action_node_map):
        from .models_v42.graph_state_encoder import batch_edge_index
        B = state_history.shape[0]
        N = self.n_nodes
        device = state_history.device

        # Encode graph state → [B*N, feat]
        h_state = self.graph_encoder(state_history, edge_index, node_static)
        # Reshape to [B, N, feat]
        state_dim = h_state.shape[-1]
        h_state_3d = h_state.reshape(B, N, state_dim)

        # Encode rainfall → [B, H, hidden]
        rain_emb = self.rainfall_encoder(rainfall)

        # Encode actions → [B, H, N, hidden]
        act_cand = self.action_encoder(action_candidate, action_node_map)
        act_ref = self.action_encoder(action_reference, action_node_map)

        # Broadcast state & rainfall to [B, H, N, feat]
        h_state_4d = h_state_3d.unsqueeze(1).expand(-1, self.horizon, -1, -1)
        rain_4d = rain_emb.unsqueeze(2).expand(-1, -1, N, -1)

        # Candidate path
        feat_cand = torch.cat([h_state_4d, rain_4d, act_cand], dim=-1)
        mlp_cand = self.direct_mlp(feat_cand)  # [B,H,N,hidden*2]
        y_candidate = self.depth_head(mlp_cand).squeeze(-1)  # [B,H,N]

        # Reference path
        feat_ref = torch.cat([h_state_4d, rain_4d, act_ref], dim=-1)
        mlp_ref = self.direct_mlp(feat_ref)
        y_reference = self.depth_head(mlp_ref).squeeze(-1)

        delta = y_candidate - y_reference

        return {
            "y_candidate": y_candidate,
            "y_reference": y_reference,
            "delta": delta,
        }


class TwinWithKPIHeads(nn.Module):
    """TwinGraphDynamics + KPI prediction heads for Stage D.

    The KPI heads (PFV/TFV/Peak) must predict the *control effect* — i.e. the
    delta between Candidate and Reference — not the absolute trajectory.

    Control objective contract (V4.2):
      - PFV target = Candidate − No-control       → PFV head uses NC delta features
      - TFV target = Candidate − Dynamic Internal  → TFV head uses DI delta features
      - Peak target = Candidate − Dynamic Internal → Peak head uses DI delta features

    Architecture:
      * Base model (TwinGraphDynamics) runs dual-tower Candidate vs NC,
        producing delta_nc = y_candidate - y_reference.
      * For PFV: delta_pool processes delta_nc; action_pool processes
        (action_candidate - action_nc).
      * For TFV/Peak: delta_pool_di processes the *candidate trajectory*
        (since we lack a DI trajectory rollout from the base model);
        action_pool_di processes (action_candidate - action_di).
      * This ensures each head sees features aligned with its target reference.
    """

    def __init__(self, base_model: nn.Module, hidden_dim: int = 32):
        super().__init__()
        self.base = base_model
        self.n_nodes = base_model.n_nodes
        self.horizon = base_model.horizon
        self.hidden_dim = int(hidden_dim)

        # --- NC-relative pools (for PFV) ---
        self.delta_pool = nn.Sequential(
            nn.Linear(self.n_nodes, hidden_dim), nn.ReLU()
        )
        self.action_pool = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )

        # --- DI-relative pools (for TFV/Peak) ---
        # Uses candidate trajectory as proxy (base model predicts hydraulic
        # response to candidate actions) combined with DI action difference.
        self.delta_pool_di = nn.Sequential(
            nn.Linear(self.n_nodes, hidden_dim), nn.ReLU()
        )
        self.action_pool_di = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )

        # KPI head input dim = delta_pool + action_pool (each hidden_dim)
        kpi_in_dim = 2 * hidden_dim
        # PFV hurdle (NC-relative)
        self.pfv_hurdle = nn.Sequential(
            nn.Linear(kpi_in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
        # TFV regression (DI-relative)
        self.tfv_head = nn.Sequential(
            nn.Linear(kpi_in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )
        # Peak regression (DI-relative)
        self.peak_head = nn.Sequential(
            nn.Linear(kpi_in_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 1)
        )

    def _compute_action_diff_features(
        self, action_a: torch.Tensor, action_b: torch.Tensor,
        action_node_map: torch.Tensor,
    ) -> torch.Tensor:
        """Compute sum of action embedding differences over actuated nodes.

        Returns [B, hidden_dim] — the globally pooled action difference.
        """
        h_act_a = self.base.action_encoder(action_a, action_node_map)  # [B, H, N, D]
        h_act_b = self.base.action_encoder(action_b, action_node_map)  # [B, H, N, D]
        act_diff = h_act_a - h_act_b  # [B, H, N, D]
        actuated = action_node_map.sum(dim=0) > 0  # [N]
        if actuated.any():
            act_diff_sum = act_diff[:, :, actuated, :].sum(dim=(1, 2))  # [B, D]
        else:
            act_diff_sum = act_diff.sum(dim=(1, 2, 3))  # [B, D] fallback
        return act_diff_sum

    def forward(self, **kwargs) -> dict[str, torch.Tensor]:
        out = self.base(**kwargs)
        y_cand = out["y_candidate"]  # [B, H, N]
        y_ref = out["y_reference"]  # [B, H, N]
        delta_nc = out["delta"]  # [B, H, N] = y_cand - y_ref (NC)

        action_c = kwargs["action_candidate"]  # [B, H, A]
        action_r = kwargs["action_reference"]  # [B, H, A] (NC)
        action_node_map = kwargs["action_node_map"]  # [A, N]

        # ================================================================
        # PFV features (NC-relative): delta_nc + action_c minus action_nc
        # ================================================================
        delta_feat_nc = delta_nc.mean(dim=1)  # [B, N]
        delta_global_nc = self.delta_pool(delta_feat_nc)  # [B, hidden]

        act_diff_nc_sum = self._compute_action_diff_features(
            action_c, action_r, action_node_map
        )
        act_global_nc = self.action_pool(act_diff_nc_sum)  # [B, hidden]

        kpi_feat_nc = torch.cat([delta_global_nc, act_global_nc], dim=-1)
        out["pfv_delta"] = self.pfv_hurdle(kpi_feat_nc).squeeze(-1)  # [B]

        # ================================================================
        # TFV/Peak features (DI-relative): use actual DI delta from model
        # ================================================================
        # If the base model produced a DI rollout, use the true delta_di;
        # otherwise fall back to candidate trajectory as proxy.
        if "delta_di" in out:
            delta_di = out["delta_di"]  # [B, H, N] = y_cand - y_di
            delta_feat_di = delta_di.mean(dim=1)  # [B, N]
        else:
            # Fallback: use candidate trajectory as proxy
            delta_feat_di = y_cand.mean(dim=1)  # [B, N]
        delta_global_di = self.delta_pool_di(delta_feat_di)  # [B, hidden]

        if "action_dynamic_internal" in kwargs:
            action_di = kwargs["action_dynamic_internal"]  # [B, H, A]
        else:
            # Fallback: if DI actions not provided, use NC (degraded)
            action_di = action_r

        act_diff_di_sum = self._compute_action_diff_features(
            action_c, action_di, action_node_map
        )
        act_global_di = self.action_pool_di(act_diff_di_sum)  # [B, hidden]

        kpi_feat_di = torch.cat([delta_global_di, act_global_di], dim=-1)
        out["tfv_delta"] = self.tfv_head(kpi_feat_di).squeeze(-1)  # [B]
        out["peak_flood_rate"] = self.peak_head(kpi_feat_di).squeeze(-1)  # [B]

        # Derived sequence rates for physics losses (kept for backward
        # compatibility with PhysicsLosses.kpi_trajectory_consistency which
        # uses these as trajectory-space proxies).
        out["tfv_rate_seq"] = y_cand.mean(dim=2)  # [B, H]
        out["pfv_rate_seq"] = y_cand.sum(dim=2) / max(self.n_nodes, 1)  # [B, H]

        return out


# ---------------------------------------------------------------------------
# KPI target normalization (z-score)
# ---------------------------------------------------------------------------

def compute_kpi_normalization_stats(
    data: dict,
    train_idx: np.ndarray | None = None,
) -> dict[str, tuple[float, float]]:
    """Compute mean/std for KPI targets from training data.

    Args:
        data: Full dataset dict (with raw KPI tensors).
        train_idx: Optional train sample indices. If None, uses all data.

    Returns:
        kpi_stats: {"pfv_delta": (mean, std), "tfv_delta": ..., "peak_delta": ...}
    """
    kpi_stats = {}
    for key in ["pfv_delta", "tfv_delta", "peak_delta"]:
        vals = data[key].numpy()
        if train_idx is not None:
            vals = vals[train_idx]
        mean_val = float(np.mean(vals))
        std_val = float(np.std(vals))
        if std_val < 1e-8:
            std_val = 1.0  # avoid division by zero
        kpi_stats[key] = (mean_val, std_val)
        logger.info("KPI norm stats [%s]: mean=%.4f, std=%.4f", key, mean_val, std_val)
    return kpi_stats


def _apply_kpi_normalization(
    batch: dict[str, torch.Tensor],
    kpi_stats: dict[str, tuple[float, float]],
) -> dict[str, torch.Tensor]:
    """Z-score normalize KPI targets in batch (in-place)."""
    for key, (mean_val, std_val) in kpi_stats.items():
        if key in batch:
            batch[key] = (batch[key] - mean_val) / std_val
    return batch


def _denormalize_kpi_predictions(
    preds: dict[str, np.ndarray],
    kpi_stats: dict[str, tuple[float, float]],
) -> dict[str, np.ndarray]:
    """Reverse z-score normalization for model predictions."""
    denorm = {}
    for key, arr in preds.items():
        if key in kpi_stats:
            mean_val, std_val = kpi_stats[key]
            denorm[key] = arr * std_val + mean_val
        else:
            denorm[key] = arr
    return denorm


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_r2(pred: np.ndarray, target: np.ndarray) -> float:
    ss_res = np.sum((target - pred) ** 2)
    ss_tot = np.sum((target - target.mean()) ** 2)
    return float(1 - ss_res / max(ss_tot, 1e-12))


def compute_metrics(
    preds: dict[str, np.ndarray],
    targets: dict[str, np.ndarray],
) -> dict[str, float]:
    """Compute R², MAE, sign accuracy for KPI deltas."""
    metrics = {}
    for key in ["pfv_delta", "tfv_delta", "peak_delta"]:
        if key in preds and key in targets:
            p, t = preds[key], targets[key]
            metrics[f"{key}_r2"] = compute_r2(p, t)
            metrics[f"{key}_mae"] = float(np.mean(np.abs(p - t)))
            metrics[f"{key}_sign_acc"] = float(np.mean(np.sign(p) == np.sign(t)))
    return metrics


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _freeze_graph_encoder(model: nn.Module) -> None:
    """Freeze graph encoder parameters."""
    base = model.base if hasattr(model, "base") else model
    for p in base.graph_encoder.parameters():
        p.requires_grad = False


def _unfreeze_all(model: nn.Module) -> None:
    """Unfreeze all parameters."""
    for p in model.parameters():
        p.requires_grad = True


def _count_trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    loss_fn: callable,
    amp: bool = True,
    max_grad_norm: float = 1.0,
    shared_tensors: dict[str, torch.Tensor] | None = None,
) -> dict[str, float]:
    """Train one epoch, return averaged loss dict."""
    model.train()
    epoch_losses: dict[str, float] = {}
    n_batches = 0
    t0 = time.time()

    for batch in dataloader:
        batch_on_device = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}

        # Add shared graph tensors
        if shared_tensors:
            for k, v in shared_tensors.items():
                batch_on_device[k] = v.to(device)

        with autocast(enabled=amp):
            pred = model(
                state_history=batch_on_device["state_history"],
                rainfall=batch_on_device["rainfall"],
                action_candidate=batch_on_device["action_candidate"],
                action_reference=batch_on_device["action_reference"],
                edge_index=batch_on_device["edge_index"],
                node_static=batch_on_device["node_static"],
                action_node_map=batch_on_device["action_node_map"],
            )
            loss_dict = loss_fn(pred, batch_on_device)
            loss = sum(v for v in loss_dict.values() if torch.is_tensor(v))

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()

        for k, v in loss_dict.items():
            val = v.detach().item() if torch.is_tensor(v) else float(v)
            epoch_losses[k] = epoch_losses.get(k, 0.0) + val
        n_batches += 1

    avg_losses = {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}
    avg_losses["epoch_time"] = time.time() - t0
    return avg_losses


@torch.no_grad()
def validate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn: callable,
    shared_tensors: dict[str, torch.Tensor] | None = None,
    kpi_stats: dict[str, tuple[float, float]] | None = None,
) -> dict[str, float]:
    """Validate and compute metrics.

    Args:
        kpi_stats: If provided, denormalize predictions/targets before
                   computing R², MAE, sign_acc (metrics are in raw space).
    """
    model.eval()
    all_preds: dict[str, list] = {"pfv_delta": [], "tfv_delta": [], "peak_delta": []}
    all_targets: dict[str, list] = {"pfv_delta": [], "tfv_delta": [], "peak_delta": []}
    epoch_losses: dict[str, float] = {}
    n_batches = 0

    for batch in dataloader:
        batch_on_device = {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}
        if shared_tensors:
            for k, v in shared_tensors.items():
                batch_on_device[k] = v.to(device)

        pred = model(
            state_history=batch_on_device["state_history"],
            rainfall=batch_on_device["rainfall"],
            action_candidate=batch_on_device["action_candidate"],
            action_reference=batch_on_device["action_reference"],
            edge_index=batch_on_device["edge_index"],
            node_static=batch_on_device["node_static"],
            action_node_map=batch_on_device["action_node_map"],
        )

        # Collect KPI predictions
        for key in all_preds:
            if key in pred:
                all_preds[key].append(pred[key].cpu().numpy())
                all_targets[key].append(batch_on_device.get(key, torch.zeros_like(pred[key])).cpu().numpy())

        loss_dict = loss_fn(pred, batch_on_device)
        for k, v in loss_dict.items():
            val = v.detach().item() if torch.is_tensor(v) else float(v)
            epoch_losses[k] = epoch_losses.get(k, 0.0) + val
        n_batches += 1

    avg_losses = {k: v / max(n_batches, 1) for k, v in epoch_losses.items()}

    # Compute R² metrics — denormalize if normalization was applied
    preds_np = {k: np.concatenate(v) for k, v in all_preds.items() if v}
    targets_np = {k: np.concatenate(v) for k, v in all_targets.items() if v}
    if kpi_stats is not None:
        preds_np = _denormalize_kpi_predictions(preds_np, kpi_stats)
        targets_np = _denormalize_kpi_predictions(targets_np, kpi_stats)
    metrics = compute_metrics(preds_np, targets_np)
    avg_losses.update(metrics)

    return avg_losses


def _make_loss_fns(
    stage: str,
    n_nodes: int,
    node_max_depth: torch.Tensor,
    edge_index: torch.Tensor,
    device: torch.device,
    ablation_mode: str = "full_4stage",
    kpi_stats: dict[str, tuple[float, float]] | None = None,
):
    """Create loss functions for each training stage, with ablation support.

    Args:
        ablation_mode: Controls which loss components are active.
            - full_4stage: all losses as designed
            - no_physics: remove physics losses from B/C/D
            - no_twin: only candidate trajectory loss (no reference/delta)
            - simplified_dynamics: same losses as full_4stage
        kpi_stats: KPI normalization stats. If provided, TrajectoryLosses
            dead zones are scaled by 1/std so they remain meaningful in
            normalized space.
    """
    from .models_v42.trajectory_losses import TrajectoryLosses
    from .models_v42.physics_losses import PhysicsLosses
    from .models_v42.ranking_losses import RankingLosses

    abl = ABLATION_CONFIGS.get(ablation_mode, ABLATION_CONFIGS["full_4stage"])
    use_physics = abl["use_physics"]
    use_twin = abl["use_twin"]

    # Pass norm_std so dead zones are scaled to normalized target space
    norm_std = {k: v[1] for k, v in kpi_stats.items()} if kpi_stats else None
    traj_loss = TrajectoryLosses(norm_std=norm_std).to(device)
    phys_loss = PhysicsLosses(n_nodes=n_nodes, node_max_depth=node_max_depth).to(device) if use_physics else None
    rank_loss = RankingLosses().to(device)
    edge_index_dev = edge_index.to(device)

    def loss_a(pred, target):
        """Stage A: depth trajectory MSE."""
        if use_twin:
            return {"depth_mse": nn.functional.mse_loss(
                pred["y_candidate"], target["depth_candidate"]
            )}
        else:
            # no_twin: only candidate
            return {"depth_mse_cand": nn.functional.mse_loss(
                pred["y_candidate"], target["depth_candidate"]
            )}

    def loss_b(pred, target):
        """Stage B: depth + delta + physics (trajectory-based only)."""
        losses = traj_loss(pred, target)
        result = {}
        if use_twin:
            result["depth_traj"] = losses["depth_trajectory"]
            result["delta_traj"] = losses["delta_trajectory"]
        else:
            result["depth_traj"] = nn.functional.mse_loss(
                pred["y_candidate"], target["depth_candidate"]
            )
        if use_physics and phys_loss is not None:
            physics = phys_loss(pred, edge_index=edge_index_dev)
            # Include all trajectory-based physics losses (exclude KPI-dependent ones)
            for k in ("non_negative", "mass_balance", "storage_continuity",
                       "capacity_bounds", "flooding_consistency", "shared_init_state"):
                if k in physics:
                    result[f"phys_{k}"] = physics[k] * 0.1
        return result

    def loss_c(pred, target):
        """Stage C: full trajectory + physics (trajectory-based only)."""
        losses = traj_loss(pred, target)
        combined = {}
        if use_twin:
            combined["depth_traj"] = losses["depth_trajectory"]
            combined["delta_traj"] = losses["delta_trajectory"]
        else:
            combined["depth_traj"] = nn.functional.mse_loss(
                pred["y_candidate"], target["depth_candidate"]
            )
        if use_physics and phys_loss is not None:
            physics = phys_loss(pred, edge_index=edge_index_dev)
            # Include trajectory-based physics losses only
            # (kpi_trajectory_consistency and peak_consistency need KPI heads → stage D)
            for k in ("non_negative", "mass_balance", "storage_continuity",
                       "capacity_bounds", "flooding_consistency", "shared_init_state"):
                if k in physics:
                    combined[f"phys_{k}"] = physics[k] * 0.1
        return combined

    def loss_d(pred, target):
        """Stage D: trajectory + physics + KPI + ranking."""
        losses = traj_loss(pred, target)
        combined = {}
        if use_twin:
            combined["depth_traj"] = losses["depth_trajectory"]
            combined["delta_traj"] = losses["delta_trajectory"]
            combined["pfv_kpi"] = losses["pfv_kpi"]
            combined["tfv_kpi"] = losses["tfv_kpi"]
            combined["peak_kpi"] = losses["peak_kpi"]
        else:
            combined["depth_traj"] = nn.functional.mse_loss(
                pred["y_candidate"], target["depth_candidate"]
            )
            # KPI losses still apply if predictions exist
            if "pfv_kpi" in losses:
                combined["pfv_kpi"] = losses["pfv_kpi"]
            if "tfv_kpi" in losses:
                combined["tfv_kpi"] = losses["tfv_kpi"]
            if "peak_kpi" in losses:
                combined["peak_kpi"] = losses["peak_kpi"]
        if use_physics and phys_loss is not None:
            physics = phys_loss(pred, edge_index=edge_index_dev)
            for k, v in physics.items():
                combined[f"phys_{k}"] = v * 0.05
        ranking = rank_loss(pred, target)
        for k, v in ranking.items():
            combined[f"rank_{k}"] = v * 0.2
        return combined

    return {"A": loss_a, "B": loss_b, "C": loss_c, "D": loss_d}[stage]


def _build_model(
    stage: str,
    n_nodes: int,
    n_facilities: int,
    node_max_depth: torch.Tensor,
    hidden_dim: int = HIDDEN_DIM,
    ablation_mode: str = "full_4stage",
) -> nn.Module:
    """Build model for given stage, with ablation variant support.

    Args:
        stage: One of A, B, C, D.
        ablation_mode: One of full_4stage, no_physics, no_twin, simplified_dynamics.
    """
    from .models_v42.counterfactual_twin_dynamics import TwinGraphDynamics

    abl = ABLATION_CONFIGS.get(ablation_mode, ABLATION_CONFIGS["full_4stage"])

    if abl["simplified_dyn"]:
        # Use simplified MLP dynamics instead of GRU
        base = SimplifiedDynamicsModel(
            n_nodes=n_nodes,
            n_facilities=n_facilities,
            n_static_features=7,
            hidden_dim=hidden_dim,
            horizon=N_HORIZON,
            history_frames=N_HISTORY,
            dropout=0.05,
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
            dropout=0.05,
        )

    if stage == "D":
        return TwinWithKPIHeads(base, hidden_dim=hidden_dim)
    return base


def _make_shared_tensors(data: dict, device: torch.device) -> dict[str, torch.Tensor]:
    shared = {
        "edge_index": data["edge_index"].to(device),
        "node_static": data["node_static"].to(device),
        "action_node_map": data["action_node_map"].to(device),
    }
    if "priority_node_indices" in data:
        shared["priority_node_indices"] = data["priority_node_indices"].to(device)
    return shared


def _make_batch(
    data: dict,
    indices: np.ndarray,
    kpi_stats: dict[str, tuple[float, float]] | None = None,
) -> dict[str, torch.Tensor]:
    idx = torch.from_numpy(indices.astype(np.int64))
    batch = {
        "state_history": data["state_history"][idx],
        "rainfall": data["rainfall"][idx],
        "action_candidate": data["action_candidate"][idx],
        "action_reference": data["action_reference"][idx],
        "depth_candidate": data["depth_candidate"][idx],
        "depth_reference": data["depth_reference"][idx],
        "pfv_delta": data["pfv_delta"][idx].clone(),
        "tfv_delta": data["tfv_delta"][idx].clone(),
        "peak_delta": data["peak_delta"][idx].clone(),
    }
    # Optional 4-branch fields
    for key in (
        "action_dynamic_internal", "action_hold_previous",
        "depth_dynamic_internal", "depth_hold_previous",
        "pfv_safe_label", "tfv_improved_label", "peak_noninferior_label",
    ):
        if key in data:
            batch[key] = data[key][idx] if data[key].shape[0] == len(idx) else data[key]
    # Apply KPI z-score normalization if stats provided
    if kpi_stats is not None:
        _apply_kpi_normalization(batch, kpi_stats)
    return batch


class _DictDataset(torch.utils.data.Dataset):
    def __init__(self, data_dict: dict[str, torch.Tensor]):
        self.data = data_dict
        self.n = next(iter(data_dict.values())).shape[0]

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}


# ---------------------------------------------------------------------------
# Main training entry point
# ---------------------------------------------------------------------------

def train_v42_twin(
    project_root: Path,
    output_root: Path,
    config: dict | None = None,
    ablation_mode: str = "full_4stage",
) -> dict[str, Any]:
    """Run full 4-stage training curriculum with event-grouped CV.

    Args:
        project_root: Project root directory.
        output_root: Output root directory.
        config: Optional config overrides.
        ablation_mode: One of full_4stage, no_physics, no_twin, simplified_dynamics.

    Returns dict with per-seed, per-fold metrics and paths.
    """
    project_root = Path(project_root)
    output_root = Path(output_root)

    # Resolve ablation config
    abl = ABLATION_CONFIGS.get(ablation_mode, ABLATION_CONFIGS["full_4stage"])
    logger.info("Ablation mode: %s — %s", ablation_mode, abl["description"])

    # Output directory: base for full_4stage, variant-specific for ablations
    if ablation_mode == "full_4stage":
        model_dir = output_root / "models" / "v42_twin"
    else:
        model_dir = output_root / "models" / f"v42_twin_ablation_{ablation_mode}"
    model_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training device: %s", device)

    # Load data
    data = load_v42_training_data(project_root, output_root)
    n_nodes = data["n_nodes"]
    n_facilities = data["n_facilities"]
    unique_events = data["unique_events"]
    event_indices = data["event_indices"]

    # Compute KPI normalization stats (from all data for consistency)
    kpi_stats = compute_kpi_normalization_stats(data)
    logger.info("KPI normalization stats: %s",
                {k: f"mean={m:.4f}, std={s:.4f}" for k, (m, s) in kpi_stats.items()})

    cfg = {**TRAINING_CONFIG, **(config or {})}
    seeds = cfg["seeds"]
    batch_size = cfg["batch_size"]
    n_folds = cfg["n_cv_folds"]
    amp_enabled = cfg["amp_enabled"] and device.type == "cuda"

    all_seed_results: list[dict] = []
    all_cv_metrics: list[dict] = []

    for seed_idx, seed in enumerate(seeds):
        logger.info("=" * 60)
        logger.info("SEED %d/%d: seed=%d", seed_idx + 1, len(seeds), seed)
        logger.info("=" * 60)

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Event-grouped CV folds
        folds = make_event_grouped_folds(event_indices, unique_events, n_folds, seed)

        fold_results: list[dict] = []

        for fold_idx in range(n_folds):
            logger.info("--- Fold %d/%d ---", fold_idx + 1, n_folds)
            train_mask, val_mask = folds[fold_idx]
            train_idx = np.where(train_mask)[0]
            val_idx = np.where(val_mask)[0]

            logger.info("Train: %d samples, Val: %d samples", len(train_idx), len(val_idx))

            # Build datasets with KPI normalization
            train_data = _make_batch(data, train_idx, kpi_stats=kpi_stats)
            val_data = _make_batch(data, val_idx, kpi_stats=kpi_stats)
            train_ds = _DictDataset(train_data)
            val_ds = _DictDataset(val_data)
            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

            # 4-stage curriculum
            history: list[dict] = []
            model = None

            for stage_name in ["A", "B", "C", "D"]:
                n_epochs = STAGE_EPOCHS[stage_name]
                logger.info("Stage %s: %d epochs", stage_name, n_epochs)

                # Build/load model
                if stage_name == "D" and model is not None:
                    # Wrap existing model with KPI heads
                    model = TwinWithKPIHeads(model, hidden_dim=HIDDEN_DIM).to(device)
                elif model is None:
                    model = _build_model(stage_name, n_nodes, n_facilities, data["node_max_depth"],
                                         ablation_mode=ablation_mode)
                    model = model.to(device)

                # Freeze/unfreeze
                if stage_name in ("A", "B"):
                    _freeze_graph_encoder(model)
                elif stage_name == "C":
                    _unfreeze_all(model)
                # Stage D: all trainable

                # Optimizer
                lr = cfg["lr_stage_d"] if stage_name == "D" else cfg["learning_rate"]
                trainable_params = [p for p in model.parameters() if p.requires_grad]
                optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=cfg["weight_decay"])
                scaler = GradScaler(enabled=amp_enabled)

                # Loss function
                loss_fn = _make_loss_fns(stage_name, n_nodes, data["node_max_depth"],
                                         data["edge_index"], device,
                                         ablation_mode=ablation_mode,
                                         kpi_stats=kpi_stats)
                shared = _make_shared_tensors(data, device)

                stage_best_val = float("inf")
                stage_best_state = None
                stage_patience = 0

                for epoch in range(n_epochs):
                    train_metrics = train_one_epoch(
                        model, train_loader, optimizer, scaler, device,
                        loss_fn, amp=amp_enabled, max_grad_norm=cfg["max_grad_norm"],
                        shared_tensors=shared,
                    )
                    val_metrics = validate(
                        model, val_loader, device, loss_fn, shared_tensors=shared,
                        kpi_stats=kpi_stats,
                    )

                    epoch_record = {
                        "stage": stage_name,
                        "epoch": epoch,
                        "seed": seed,
                        "fold": fold_idx,
                        "train_loss": sum(v for k, v in train_metrics.items() if k not in ("epoch_time", "lr")),
                        "val_loss": sum(v for k, v in val_metrics.items() if not k.endswith("_r2") and not k.endswith("_mae") and not k.endswith("_sign_acc") and k != "epoch_time"),
                        **{f"train_{k}": v for k, v in train_metrics.items()},
                        **{f"val_{k}": v for k, v in val_metrics.items()},
                    }
                    history.append(epoch_record)

                    val_loss = epoch_record["val_loss"]
                    if val_loss < stage_best_val:
                        stage_best_val = val_loss
                        stage_patience = 0
                        stage_best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                    else:
                        stage_patience += 1

                    if (epoch + 1) % 10 == 0 or epoch == 0:
                        logger.info(
                            "  Stage %s Ep %d: train=%.4f val=%.4f pfv_r2=%.4f tfv_r2=%.4f peak_r2=%.4f",
                            stage_name, epoch, epoch_record["train_loss"],
                            epoch_record["val_loss"],
                            val_metrics.get("pfv_delta_r2", 0),
                            val_metrics.get("tfv_delta_r2", 0),
                            val_metrics.get("peak_delta_r2", 0),
                        )

                    if stage_patience >= cfg["early_stop_patience"]:
                        logger.info("  Early stop at epoch %d", epoch)
                        break

                # Restore best model for this stage
                if stage_best_state is not None:
                    model.load_state_dict(stage_best_state)

            # Final validation metrics for this fold
            final_metrics = validate(model, val_loader, device, loss_fn,
                                     shared_tensors=shared, kpi_stats=kpi_stats)

            # Save fold model
            fold_model_path = model_dir / f"v42_twin_model_seed{seed}_fold{fold_idx}.pt"
            torch.save({
                "seed": seed,
                "fold": fold_idx,
                "ablation_mode": ablation_mode,
                "model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                "model_config": {
                    "n_nodes": n_nodes,
                    "n_facilities": n_facilities,
                    "hidden_dim": HIDDEN_DIM,
                    "gat_heads": GAT_HEADS,
                    "n_history": N_HISTORY,
                    "n_horizon": N_HORIZON,
                    "ablation_mode": ablation_mode,
                    "ablation_description": abl["description"],
                },
                "final_metrics": final_metrics,
            }, fold_model_path)

            # Save training history
            fold_history_path = model_dir / f"training_history_seed{seed}_fold{fold_idx}.json"
            with open(fold_history_path, "w") as f:
                json.dump(history, f, indent=1, default=str)

            # Build training history summary: first/last epoch per stage
            history_summary = _summarize_training_history(history)

            fold_results.append({
                "fold": fold_idx,
                "n_train": len(train_idx),
                "n_val": len(val_idx),
                "final_metrics": {k: float(v) for k, v in final_metrics.items()},
                "training_history": history_summary,
            })

            logger.info("Fold %d final: %s", fold_idx,
                        {k: f"{v:.4f}" for k, v in final_metrics.items() if k.endswith("_r2")})

        # Aggregate across folds for this seed
        assert len(fold_results) == n_folds, (
            f"Seed {seed}: expected {n_folds} folds, got {len(fold_results)}. "
            "Silent fold loss detected!"
        )
        seed_result = {
            "seed": seed,
            "folds": fold_results,
            "aggregate": _aggregate_fold_metrics(fold_results),
        }
        # Anti-overwrite guard: verify no duplicate seed in results
        existing_seeds = [r["seed"] for r in all_seed_results]
        assert seed not in existing_seeds, (
            f"DUPLICATE SEED {seed} detected! Existing seeds: {existing_seeds}. "
            "This would silently overwrite previous results."
        )
        all_seed_results.append(seed_result)
        logger.info("Seed %d aggregate: %s", seed,
                    {k: f"{v:.4f}" for k, v in seed_result["aggregate"].items() if k.endswith("_r2_mean")})

    # Save aggregate results
    combined = {
        "ablation_mode": ablation_mode,
        "ablation_description": abl["description"],
        "seeds": seeds,
        "n_folds": n_folds,
        "n_samples": len(data["pfv_delta"]),
        "n_events": len(unique_events),
        "per_seed": all_seed_results,
        "overall_aggregate": _aggregate_seed_results(all_seed_results),
    }

    # Write outputs
    combined_path = model_dir / "training_history.json"
    with open(combined_path, "w") as f:
        json.dump(combined, f, indent=2, default=str)

    cv_path = model_dir / "cv_metrics.json"
    cv_data = {
        "seeds": seeds,
        "n_folds": n_folds,
        "per_seed_folds": [
            {"seed": s["seed"], "folds": s["folds"]}
            for s in all_seed_results
        ],
        "aggregate": combined["overall_aggregate"],
    }
    with open(cv_path, "w") as f:
        json.dump(cv_data, f, indent=2, default=str)

    # Save best model (across all folds of seed 0 as representative)
    best_model_path = model_dir / "v42_twin_model_seed0.pt"
    if all_seed_results and all_seed_results[0]["folds"]:
        best_fold = min(all_seed_results[0]["folds"],
                       key=lambda f: f["final_metrics"].get("val_loss", float("inf")))
        fold_model = model_dir / f"v42_twin_model_seed0_fold{best_fold['fold']}.pt"
        if fold_model.exists():
            import shutil
            shutil.copy2(fold_model, best_model_path)

    # Save KPI normalization stats
    norm_stats_path = model_dir / "kpi_normalization_stats.json"
    norm_stats_serializable = {
        k: {"mean": m, "std": s} for k, (m, s) in kpi_stats.items()
    }
    with open(norm_stats_path, "w") as f:
        json.dump(norm_stats_serializable, f, indent=2)
    logger.info("KPI normalization stats saved to: %s", norm_stats_path)

    logger.info("=" * 60)
    logger.info("Training complete!")
    logger.info("Overall: %s", {k: f"{v:.4f}" for k, v in combined["overall_aggregate"].items() if "r2" in k})
    logger.info("Outputs: %s", model_dir)

    return combined


def _summarize_training_history(history: list[dict]) -> dict:
    """Summarize training history: first/last epoch per stage + key metrics."""
    if not history:
        return {"n_epochs": 0, "stages": {}}
    stages: dict[str, list[dict]] = {}
    for rec in history:
        s = rec.get("stage", "?")
        stages.setdefault(s, []).append(rec)
    summary: dict = {"n_epochs": len(history), "stages": {}}
    for s_name in sorted(stages):
        recs = stages[s_name]
        first, last = recs[0], recs[-1]
        summary["stages"][s_name] = {
            "n_epochs": len(recs),
            "first_epoch": {
                "train_loss": first.get("train_loss"),
                "val_loss": first.get("val_loss"),
            },
            "last_epoch": {
                "train_loss": last.get("train_loss"),
                "val_loss": last.get("val_loss"),
            },
            "best_val_loss": min(r.get("val_loss", float("inf")) for r in recs),
        }
        # Include R² if available (Stage D)
        for key in ("val_pfv_delta_r2", "val_tfv_delta_r2", "val_peak_delta_r2",
                     "val_pfv_delta_sign_acc", "val_tfv_delta_sign_acc"):
            vals = [r[key] for r in recs if key in r and r[key] is not None]
            if vals:
                summary["stages"][s_name][key] = {
                    "first": vals[0], "last": vals[-1], "best": max(vals),
                }
    return summary


def _aggregate_fold_metrics(fold_results: list[dict]) -> dict[str, float]:
    """Average metrics across folds."""
    all_keys = set()
    for f in fold_results:
        all_keys.update(f["final_metrics"].keys())
    agg = {}
    for key in sorted(all_keys):
        vals = [f["final_metrics"][key] for f in fold_results if key in f["final_metrics"]]
        if vals:
            agg[f"{key}_mean"] = float(np.mean(vals))
            agg[f"{key}_std"] = float(np.std(vals))
    return agg


def _aggregate_seed_results(seed_results: list[dict]) -> dict[str, float]:
    """Average across seeds and folds."""
    all_keys = set()
    for s in seed_results:
        all_keys.update(s["aggregate"].keys())
    agg = {}
    for key in sorted(all_keys):
        vals = [s["aggregate"][key] for s in seed_results if key in s["aggregate"]]
        if vals:
            agg[key] = float(np.mean(vals))
    return agg
