from __future__ import annotations

import numpy as np
import pandas as pd


def generate_influence_candidates(
    native_action: np.ndarray,
    priority_to_actuators: pd.DataFrame,
    actuators: pd.DataFrame,
    max_delta: float = 0.08,
) -> list[dict]:
    native = np.asarray(native_action, dtype=float).reshape(-1)
    id_to_idx = {str(a): i for i, a in enumerate(actuators["actuator_id"].astype(str).tolist())}
    rows = []
    for _, r in priority_to_actuators.iterrows():
        aid = str(r.get("actuator_id", ""))
        if aid not in id_to_idx:
            continue
        role = str(r.get("asset_role", "")).lower()
        for direction, label in [(-1.0, "retain_or_throttle"), (1.0, "release_or_boost")]:
            if "pump" in role and direction > 0:
                template = "pump_boost_if_downstream_capacity_available"
            elif "pump" in role:
                template = "pump_throttle_if_downstream_peak_high"
            elif direction < 0:
                template = "storage_retain_if_storage_available"
            else:
                template = "storage_release_if_recession_and_downstream_safe"
            action = native.copy()
            action[id_to_idx[aid]] = np.clip(action[id_to_idx[aid]] + direction * float(max_delta), 0.0, 1.0)
            rows.append(
                {
                    "label": f"{template}|priority={r.get('priority_node')}|actuator={aid}|d={direction * float(max_delta):.3f}",
                    "action": action,
                    "target_priority_nodes": str(r.get("priority_node", "")),
                    "influence_path_length": int(r.get("influence_path_length", 999)),
                    "bottleneck_score": 1.0 / max(1, int(r.get("influence_path_length", 999))),
                    "physical_rationale": str(r.get("physical_rationale", "")),
                }
            )
    return rows
