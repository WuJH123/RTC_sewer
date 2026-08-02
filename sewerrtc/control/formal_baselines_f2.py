"""Formal deterministic baseline policies for Project6 V4.2.

These functions generate desired Engineering36 actions from decision-time state
only. EFD follows the standard normalized filling-degree idea: compare local
controlled-zone filling degree with the system mean and preferentially release
more-filled zones. Auto-RBC is a fixed, reproducible depth-threshold rule. Both
return desired settings only; the shared authoritative engineering projector
must enforce binary/bounds/rate/ramp/dwell/interlock/K constraints before SWMM.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class BaselineAction:
    strategy: str
    desired_action: np.ndarray
    local_filling_degree: np.ndarray
    system_filling_degree: float


def controlled_filling_degree(
    *,
    node_depth: np.ndarray,
    node_full_depth: np.ndarray,
    action_node_map: np.ndarray,
) -> np.ndarray:
    depth = np.asarray(node_depth, dtype=float).reshape(-1)
    full = np.asarray(node_full_depth, dtype=float).reshape(-1)
    amap = np.asarray(action_node_map, dtype=float)
    if depth.shape != full.shape or amap.ndim != 2 or amap.shape[1] != depth.size:
        raise ValueError("filling-degree graph dimensions mismatch")
    safe = np.where(full > 1e-6, full, np.nan)
    node_fill = np.divide(depth, safe, out=np.zeros_like(depth), where=np.isfinite(safe))
    node_fill = np.clip(np.nan_to_num(node_fill, nan=0.0, posinf=1.5, neginf=0.0), 0.0, 1.5)
    result = np.zeros(amap.shape[0], dtype=float)
    for i in range(amap.shape[0]):
        indices = np.flatnonzero(np.abs(amap[i]) > 0)
        if indices.size:
            # Conservative zone signal: a local bottleneck must not be hidden by
            # averaging many dry nodes.
            result[i] = float(np.max(node_fill[indices]))
    return np.clip(result, 0.0, 1.5)


def _binary(action: np.ndarray, binary_indices: Iterable[int]) -> np.ndarray:
    out = np.asarray(action, dtype=float).copy()
    for index in binary_indices:
        i = int(index)
        if 0 <= i < out.size:
            out[i] = 1.0 if out[i] >= 0.5 else 0.0
    return out


def equal_filling_degree_action(
    *,
    node_depth: np.ndarray,
    node_full_depth: np.ndarray,
    action_node_map: np.ndarray,
    anchor_action: np.ndarray,
    binary_indices: Iterable[int] = (),
    gain: float = 0.8,
    deadband: float = 0.02,
) -> BaselineAction:
    """Return a deterministic EFD desired action before engineering projection."""
    anchor = np.clip(np.asarray(anchor_action, dtype=float).reshape(-1), 0.0, 1.0)
    fill = controlled_filling_degree(node_depth=node_depth, node_full_depth=node_full_depth, action_node_map=action_node_map)
    if fill.size != anchor.size:
        raise ValueError("EFD facility dimension mismatch")
    mean_fill = float(np.mean(fill)) if fill.size else 0.0
    delta = fill - mean_fill
    desired = anchor.copy()
    active = np.abs(delta) > float(deadband)
    desired[active] = np.clip(anchor[active] + float(gain) * delta[active], 0.0, 1.0)
    desired = _binary(desired, binary_indices)
    return BaselineAction("efd", desired.astype(np.float32), fill.astype(np.float32), mean_fill)


def auto_rbc_action(
    *,
    node_depth: np.ndarray,
    node_full_depth: np.ndarray,
    action_node_map: np.ndarray,
    anchor_action: np.ndarray,
    binary_indices: Iterable[int] = (),
    low: float = 0.30,
    high: float = 0.70,
    mid_gain: float = 0.5,
) -> BaselineAction:
    """Fixed depth/filling rule with no event-specific post-hoc tuning."""
    if not 0.0 <= low < high:
        raise ValueError("Auto-RBC requires 0 <= low < high")
    anchor = np.clip(np.asarray(anchor_action, dtype=float).reshape(-1), 0.0, 1.0)
    fill = controlled_filling_degree(node_depth=node_depth, node_full_depth=node_full_depth, action_node_map=action_node_map)
    if fill.size != anchor.size:
        raise ValueError("Auto-RBC facility dimension mismatch")
    desired = anchor.copy()
    desired[fill <= low] = 0.0
    desired[fill >= high] = 1.0
    middle = (fill > low) & (fill < high)
    desired[middle] = np.clip(anchor[middle] + mid_gain * (fill[middle] - 0.5), 0.0, 1.0)
    desired = _binary(desired, binary_indices)
    return BaselineAction("auto_rbc", desired.astype(np.float32), fill.astype(np.float32), float(np.mean(fill)) if fill.size else 0.0)


def all_close_action(n_facilities: int) -> BaselineAction:
    if int(n_facilities) <= 0:
        raise ValueError("n_facilities must be positive")
    return BaselineAction(
        "all_close",
        np.zeros(int(n_facilities), dtype=np.float32),
        np.zeros(int(n_facilities), dtype=np.float32),
        0.0,
    )
