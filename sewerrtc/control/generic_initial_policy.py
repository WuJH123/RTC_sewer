from __future__ import annotations

import numpy as np
import pandas as pd


class AutoRBCPolicy:
    def __init__(self, high_depth: float = 0.8, low_depth: float = 0.3):
        self.high_depth = float(high_depth)
        self.low_depth = float(low_depth)

    def action(self, actuators: pd.DataFrame, priority_depth_max: float, current_action: np.ndarray | None = None) -> np.ndarray:
        base = np.ones(len(actuators), dtype=float) if current_action is None else np.asarray(current_action, dtype=float).copy()
        if priority_depth_max >= self.high_depth:
            return np.clip(base - 0.08, 0.0, 1.0)
        if priority_depth_max <= self.low_depth:
            return np.clip(base + 0.04, 0.0, 1.0)
        return base


class StorageEqualizationPolicy:
    def action(self, actuators: pd.DataFrame, storage_fill_ratio: float, current_action: np.ndarray | None = None) -> np.ndarray:
        base = np.ones(len(actuators), dtype=float) if current_action is None else np.asarray(current_action, dtype=float).copy()
        roles = actuators.get("asset_role", actuators.get("link_type", pd.Series([""] * len(actuators)))).astype(str).str.lower()
        storage_mask = roles.str.contains("storage|orifice|weir", regex=True, na=False).to_numpy()
        if storage_fill_ratio > 0.8:
            base[storage_mask] = np.clip(base[storage_mask] + 0.05, 0.0, 1.0)
        elif storage_fill_ratio < 0.4:
            base[storage_mask] = np.clip(base[storage_mask] - 0.05, 0.0, 1.0)
        return base


class SafeHeuristicPolicy:
    def action(self, actuators: pd.DataFrame, downstream_peak_risk: float, current_action: np.ndarray | None = None) -> np.ndarray:
        base = np.ones(len(actuators), dtype=float) if current_action is None else np.asarray(current_action, dtype=float).copy()
        roles = actuators.get("asset_role", actuators.get("link_type", pd.Series([""] * len(actuators)))).astype(str).str.lower()
        pump_mask = roles.str.contains("pump", regex=True, na=False).to_numpy()
        if downstream_peak_risk > 0.7:
            base[pump_mask] = np.clip(base[pump_mask] - 0.08, 0.0, 1.0)
        return base
