"""Raw INP node metadata for V4.2 priority-depth and target contracts.

Graph node features are standardized before entering the neural networks. They
must therefore never be used as physical depth thresholds or as the authority
for storage/outfall identity. This module re-reads the frozen INP and aligns raw
physical metadata to the graph node order.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sewerrtc.v4.v42_trajectory_builder import _load_graph_topology, _parse_inp_topology


@dataclass(frozen=True)
class NodePhysicalContract:
    node_ids: tuple[str, ...]
    max_depth_m: np.ndarray
    storage_indices: tuple[int, ...]
    outfall_indices: tuple[int, ...]


def load_node_physical_contract(project_root: str | Path) -> NodePhysicalContract:
    root = Path(project_root)
    graph = _load_graph_topology(root)
    graph_ids = list(map(str, graph["node_ids"]))
    nodes, _ = _parse_inp_topology(root / "data" / "wuhan_v8_storage_retrofit.inp")
    rows = {str(row["node_id"]).casefold(): row for _, row in nodes.iterrows()}
    max_depth: list[float] = []
    storage: list[int] = []
    outfall: list[int] = []
    for index, node_id in enumerate(graph_ids):
        key = node_id.casefold()
        if key not in rows:
            raise KeyError(f"graph node {node_id!r} missing from frozen INP")
        row = rows[key]
        depth = float(row.get("max_depth", 0.0))
        if not np.isfinite(depth):
            raise ValueError(f"node {node_id!r} has non-finite max_depth")
        max_depth.append(depth)
        kind = str(row.get("node_type", "")).casefold()
        if kind == "storage":
            storage.append(index)
        elif kind == "outfall":
            outfall.append(index)
    return NodePhysicalContract(
        node_ids=tuple(graph_ids),
        max_depth_m=np.asarray(max_depth, dtype=np.float64),
        storage_indices=tuple(storage),
        outfall_indices=tuple(outfall),
    )


def priority_depth_limits_m(
    project_root: str | Path,
    priority_indices: list[int] | tuple[int, ...],
    *,
    max_depth_fraction: float = 0.95,
    minimum_freeboard_m: float = 0.05,
) -> np.ndarray:
    physical = load_node_physical_contract(project_root)
    idx = np.asarray(priority_indices, dtype=int)
    selected = physical.max_depth_m[idx]
    if not np.isfinite(selected).all() or np.any(selected <= 0.0):
        raise ValueError("priority-node raw INP max_depth must be finite and positive")
    return np.maximum(
        0.0,
        np.minimum(
            float(max_depth_fraction) * selected,
            selected - float(minimum_freeboard_m),
        ),
    )
