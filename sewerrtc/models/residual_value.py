from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch import nn


class ResidualActionValueNet(nn.Module):
    """Small action-value model for internal-rule residual control.

    It predicts event-horizon KPI deltas for a small residual action around the
    native SWMM/engineering rule action:
      (state summary, u_native, u_candidate-u_native) -> delta_PFV, delta_TFV, delta_peak.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 128, output_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> dict:
        y = self.net(x)
        return {
            "delta": y[:, :3],
            "logits": y[:, 3:],
        }


def _group_masks(actuators: pd.DataFrame) -> dict[str, np.ndarray]:
    typ = actuators.get("link_type", pd.Series("", index=actuators.index)).fillna("").astype(str).to_numpy()
    role = actuators.get("storage_control_type", pd.Series("", index=actuators.index)).fillna("").astype(str).to_numpy()
    near_storage = actuators.get("near_storage", pd.Series(False, index=actuators.index)).fillna(False).astype(bool).to_numpy()
    return {
        "pump": typ == "pump",
        "storage_inlet": role == "storage_inlet",
        "storage_outlet": role == "storage_outlet",
        "storage_other": near_storage & (typ != "pump") & (role != "storage_inlet") & (role != "storage_outlet"),
        "other": ~(typ == "pump") & ~near_storage,
    }


def build_residual_feature_dict(
    actuators: pd.DataFrame,
    nominal: np.ndarray,
    candidate: np.ndarray,
    phase: str,
    rainfall_mm_h: float,
    priority_depth_max: float,
    elapsed_min: float,
) -> dict[str, float]:
    nominal = np.asarray(nominal, dtype=np.float32).reshape(-1)
    candidate = np.asarray(candidate, dtype=np.float32).reshape(-1)
    n = min(len(nominal), len(candidate), len(actuators))
    nominal = nominal[:n]
    candidate = candidate[:n]
    delta = candidate - nominal
    out: dict[str, float] = {
        "feat_elapsed_min": float(elapsed_min),
        "feat_rainfall_mm_h": float(rainfall_mm_h),
        "feat_priority_depth_max": float(priority_depth_max),
        "feat_phase_pre_peak": 1.0 if phase == "pre_peak" else 0.0,
        "feat_phase_peak": 1.0 if phase == "peak" else 0.0,
        "feat_phase_recession": 1.0 if phase == "recession" else 0.0,
        "feat_delta_mean": float(delta.mean()) if n else 0.0,
        "feat_delta_abs_mean": float(np.abs(delta).mean()) if n else 0.0,
        "feat_delta_abs_max": float(np.abs(delta).max()) if n else 0.0,
        "feat_delta_pos_count": float((delta > 1e-6).sum()),
        "feat_delta_neg_count": float((delta < -1e-6).sum()),
    }
    masks = _group_masks(actuators.iloc[:n].reset_index(drop=True))
    for name, mask in masks.items():
        d = delta[mask]
        out[f"feat_{name}_delta_mean"] = float(d.mean()) if d.size else 0.0
        out[f"feat_{name}_delta_abs_mean"] = float(np.abs(d).mean()) if d.size else 0.0
        out[f"feat_{name}_delta_abs_max"] = float(np.abs(d).max()) if d.size else 0.0
        out[f"feat_{name}_changed_count"] = float((np.abs(d) > 1e-6).sum()) if d.size else 0.0
    return out


class ResidualValuePredictor:
    def __init__(self, ckpt_path: str | Path, device: str = "cpu"):
        self.path = Path(ckpt_path)
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        ckpt = torch.load(self.path, map_location=self.device, weights_only=False)
        self.feature_cols = [str(x) for x in ckpt["feature_cols"]]
        self.x_mean = torch.tensor(ckpt["x_mean"], dtype=torch.float32, device=self.device)
        self.x_std = torch.tensor(ckpt["x_std"], dtype=torch.float32, device=self.device)
        self.y_mean = torch.tensor(ckpt["y_mean"], dtype=torch.float32, device=self.device)
        self.y_std = torch.tensor(ckpt["y_std"], dtype=torch.float32, device=self.device)
        self.safe_threshold = float(
            ckpt.get("safe_threshold", ckpt.get("metrics", {}).get("safe_threshold", 0.5))
        )
        self.model = ResidualActionValueNet(
            len(self.feature_cols),
            int(ckpt.get("hidden_dim", 128)),
            int(ckpt.get("output_dim", 6)),
        ).to(self.device)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()

    def predict_one(self, features: dict[str, float]) -> dict[str, float]:
        x = np.asarray([float(features.get(c, 0.0)) for c in self.feature_cols], dtype=np.float32)
        xt = torch.tensor(x[None, :], dtype=torch.float32, device=self.device)
        xt = (xt - self.x_mean[None, :]) / self.x_std[None, :]
        with torch.no_grad():
            out = self.model(xt)
            delta = out["delta"] * self.y_std[None, :] + self.y_mean[None, :]
            probs = torch.sigmoid(out["logits"])
        d = delta.detach().cpu().numpy()[0]
        p = probs.detach().cpu().numpy()[0]
        return {
            "delta_pfv": float(d[0]),
            "delta_tfv": float(d[1]),
            "delta_peak": float(d[2]),
            "pfv_improve_prob": float(p[0]),
            "safe_prob": float(p[1]),
            "safe_threshold": float(self.safe_threshold),
            "pfv_nonzero_prob": float(p[2]),
            "tfv_nonworse_prob": float(p[3]) if len(p) > 3 else 1.0,
            "peak_nonworse_prob": float(p[4]) if len(p) > 4 else 1.0,
        }


def feature_columns_from_frame(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith("feat_")]
