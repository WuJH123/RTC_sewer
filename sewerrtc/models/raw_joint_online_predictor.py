from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .raw_joint_action_surrogate import RawJointActionSurrogate, encode_phase_indices
from .raw_joint_training import apply_uncertainty_multipliers


class RawJointOnlinePredictor:
    """Batched deployment wrapper with strict canonical-order validation."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        *,
        canonical_action_ids: list[str],
        device: str = "cpu",
        batch_size: int = 256,
    ) -> None:
        self.path = Path(checkpoint_path)
        if not self.path.exists():
            raise FileNotFoundError(f"Missing raw joint checkpoint: {self.path}")
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        checkpoint = torch.load(self.path, map_location=self.device, weights_only=False)
        action_ids = [str(item) for item in checkpoint.get("action_ids", [])]
        if action_ids != list(canonical_action_ids):
            raise ValueError(
                "Raw joint checkpoint action order is incompatible with online control: "
                f"checkpoint={len(action_ids)}, online={len(canonical_action_ids)}"
            )
        if str(checkpoint.get("label_semantics", "")) != "same_state_candidate_minus_no_control":
            raise ValueError("Raw joint checkpoint lacks verified same-state No-control effect semantics")
        self.action_ids = action_ids
        self.batch_size = max(1, int(batch_size))
        self.node_static = torch.as_tensor(checkpoint["node_static"], dtype=torch.float32, device=self.device)
        self.edge_index = torch.as_tensor(checkpoint["edge_index"], dtype=torch.long, device=self.device)
        self.action_node_map = torch.as_tensor(checkpoint["action_node_map"], dtype=torch.float32, device=self.device)
        self.actuator_features = torch.as_tensor(checkpoint["actuator_features"], dtype=torch.float32, device=self.device)
        self.priority_indices = torch.as_tensor(checkpoint["priority_indices"], dtype=torch.long, device=self.device)
        self.storage_indices = torch.as_tensor(checkpoint["storage_indices"], dtype=torch.long, device=self.device)
        self.model = RawJointActionSurrogate(
            n_nodes=int(self.node_static.shape[0]),
            n_actions=len(action_ids),
            node_static_dim=int(self.node_static.shape[1]),
            actuator_feature_dim=int(self.actuator_features.shape[1]),
            horizon_steps=int(checkpoint["horizon_steps"]),
            hidden_dim=int(checkpoint["hidden_dim"]),
            heads=int(checkpoint.get("heads", 4)),
            architecture_version=str(checkpoint.get("architecture_version", "legacy_v1")),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["model"], strict=True)
        self.model.eval()
        calibration = dict(checkpoint.get("uncertainty_calibration", {}) or {})
        self.uncertainty_multipliers = np.asarray(
            calibration.get("aggregate_sigma_multipliers", [1.0, 1.0, 1.0]),
            dtype=np.float32,
        )
        if self.uncertainty_multipliers.shape != (3,):
            raise ValueError("checkpoint uncertainty calibration must contain three aggregate multipliers")
        self.provenance = dict(checkpoint.get("provenance", {}))
        thresholds = dict(checkpoint.get("classification_thresholds", {}) or {})
        self.classification_thresholds = {
            "PFV_noninferiority": float(thresholds.get("PFV_noninferiority", 0.5)),
            "TFV_improvement": float(thresholds.get("TFV_improvement", 0.5)),
            "peak_safe": float(thresholds.get("peak_safe", 0.5)),
        }

    def predict_many(self, **kwargs: Any) -> dict[str, np.ndarray]:
        candidate = np.asarray(kwargs["candidate_action_seq"], dtype=np.float32)
        reference = np.asarray(kwargs["reference_action_seq"], dtype=np.float32)
        if candidate.ndim != 3 or candidate.shape != reference.shape:
            raise ValueError("candidate/reference actions must be aligned [B,H,A] tensors")
        batch, horizon, actions = candidate.shape
        if actions != len(self.action_ids):
            raise ValueError("online candidate action axis does not match checkpoint canonical order")
        state = np.asarray(kwargs["state"], dtype=np.float32)
        if state.ndim == 1:
            state = np.repeat(state[None, :], batch, axis=0)
        rain = np.asarray(kwargs["rain_seq"], dtype=np.float32)
        if rain.ndim == 1:
            rain = rain[:, None]
        if rain.ndim == 2:
            rain = np.repeat(rain[None, :, :], batch, axis=0)
        mask = np.asarray(kwargs.get("actuator_mask", np.ones((batch, actions))), dtype=np.float32)
        context = dict(kwargs.get("context", {}) or {})
        phase = str(context.get("phase", "unknown"))
        phase_index = encode_phase_indices([phase] * batch).to(self.device)
        collected: dict[str, list[np.ndarray]] = {}
        with torch.no_grad():
            for start in range(0, batch, self.batch_size):
                end = min(batch, start + self.batch_size)
                output = self.model(
                    state=torch.as_tensor(state[start:end], device=self.device),
                    candidate_action_seq=torch.as_tensor(candidate[start:end], device=self.device),
                    reference_action_seq=torch.as_tensor(reference[start:end], device=self.device),
                    rain_seq=torch.as_tensor(rain[start:end], device=self.device),
                    actuator_mask=torch.as_tensor(mask[start:end], device=self.device),
                    actuator_features=self.actuator_features,
                    node_static=self.node_static,
                    edge_index=self.edge_index,
                    action_node_map=self.action_node_map,
                    priority_indices=self.priority_indices,
                    storage_indices=self.storage_indices,
                    phase_index=phase_index[start:end],
                )
                for key, value in output.items():
                    collected.setdefault(key, []).append(value.detach().cpu().numpy())
        merged = {key: np.concatenate(parts, axis=0) for key, parts in collected.items()}
        merged = apply_uncertainty_multipliers(merged, self.uncertainty_multipliers)
        merged["reference_PFV_H"] = merged["reference_risk_rate_seq"][:, :, 0].sum(axis=1) * 300.0
        merged["reference_TFV_H"] = merged["reference_risk_rate_seq"][:, :, 1].sum(axis=1) * 300.0
        merged["reference_peak"] = merged["reference_risk_rate_seq"][:, :, 1].max(axis=1)
        for name, threshold in self.classification_thresholds.items():
            merged[f"{name}_classifier_threshold"] = np.full(batch, threshold, dtype=np.float32)
        return merged
