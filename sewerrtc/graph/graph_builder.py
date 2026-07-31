from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd


def build_node_link_graph(nodes: pd.DataFrame, links: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    node_ids = list(nodes["node_id"])
    node_index = {n: i for i, n in enumerate(node_ids)}
    valid = links[links["from_node"].isin(node_index) & links["to_node"].isin(node_index)].copy()
    edge_index = np.array([[node_index[a], node_index[b]] for a, b in zip(valid["from_node"], valid["to_node"])], dtype=np.int64)
    if edge_index.size == 0:
        edge_index = np.zeros((0, 2), dtype=np.int64)
    deg_in = np.zeros(len(nodes), dtype=np.float32)
    deg_out = np.zeros(len(nodes), dtype=np.float32)
    for a, b in edge_index:
        deg_out[a] += 1
        deg_in[b] += 1
    nodes = nodes.copy()
    nodes["degree_in"] = deg_in
    nodes["degree_out"] = deg_out
    nodes["is_storage"] = (nodes["node_type"] == "storage").astype(int)
    nodes["is_outfall"] = (nodes["node_type"] == "outfall").astype(int)
    links = links.copy()
    links["from_index"] = links["from_node"].map(node_index)
    links["to_index"] = links["to_node"].map(node_index)
    return nodes, links, edge_index.T


def khop_nodes(links: pd.DataFrame, sources: Sequence[str], k: int = 3, direction: str = "upstream") -> set[str]:
    adj: Dict[str, list[str]] = defaultdict(list)
    for _, row in links.iterrows():
        if direction == "upstream":
            adj[str(row["to_node"])].append(str(row["from_node"]))
        else:
            adj[str(row["from_node"])].append(str(row["to_node"]))
    seen = set(sources)
    q = deque((s, 0) for s in sources)
    while q:
        n, d = q.popleft()
        if d >= k:
            continue
        for nb in adj.get(n, []):
            if nb not in seen:
                seen.add(nb)
                q.append((nb, d + 1))
    return seen


def make_actuator_table(nodes: pd.DataFrame, links: pd.DataFrame, max_actuators: int = 0) -> pd.DataFrame:
    acts = links[links["link_type"].isin(["pump", "orifice", "weir", "outlet"])].copy()
    if acts.empty:
        return acts
    node_meta = nodes.set_index("node_id")
    storage_nodes = set(nodes.loc[nodes["node_type"] == "storage", "node_id"])
    acts["near_storage"] = acts["from_node"].isin(storage_nodes) | acts["to_node"].isin(storage_nodes)
    acts["storage_control_type"] = np.where(
        acts["from_node"].isin(storage_nodes),
        "storage_outlet",
        np.where(acts["to_node"].isin(storage_nodes), "storage_inlet", "not_storage"),
    )
    acts["from_node_type"] = acts["from_node"].map(node_meta["node_type"]).fillna("")
    acts["to_node_type"] = acts["to_node"].map(node_meta["node_type"]).fillna("")
    acts["from_max_depth"] = pd.to_numeric(acts["from_node"].map(node_meta.get("max_depth", pd.Series(dtype=float))), errors="coerce")
    acts["to_max_depth"] = pd.to_numeric(acts["to_node"].map(node_meta.get("max_depth", pd.Series(dtype=float))), errors="coerce")
    acts["from_invert"] = pd.to_numeric(acts["from_node"].map(node_meta.get("invert", pd.Series(dtype=float))), errors="coerce")
    acts["to_invert"] = pd.to_numeric(acts["to_node"].map(node_meta.get("invert", pd.Series(dtype=float))), errors="coerce")
    acts["storage_node_id"] = np.where(
        acts["from_node"].isin(storage_nodes),
        acts["from_node"],
        np.where(acts["to_node"].isin(storage_nodes), acts["to_node"], ""),
    )
    acts["storage_node_max_depth"] = np.where(
        acts["from_node"].isin(storage_nodes),
        acts["from_max_depth"],
        np.where(acts["to_node"].isin(storage_nodes), acts["to_max_depth"], np.nan),
    )
    # Wuhan-specific EFD benchmark metadata. Equal-filling-degree control must
    # use the local connected node of this INP, not Astlingen's named tanks.
    acts["efd_reference_node"] = np.where(
        acts["storage_control_type"].eq("storage_inlet"),
        acts["to_node"],
        acts["from_node"],
    )
    acts["efd_reference_max_depth"] = np.where(
        acts["storage_control_type"].eq("storage_inlet"),
        acts["to_max_depth"],
        acts["from_max_depth"],
    )
    acts["efd_is_storage_control"] = acts["storage_control_type"].isin(["storage_inlet", "storage_outlet"])
    if "has_internal_rule" not in acts:
        acts["has_internal_rule"] = False
    # Water Research version: do not let numerous pumps crowd out detention
    # controls. All storage inlet/outlet actuators and native-rule actuators are
    # forced into the controllable set before filling the remaining budget.
    acts["force_include"] = acts["near_storage"] | acts["has_internal_rule"].fillna(False)
    storage_rank = np.where(acts["near_storage"], 0, 1)
    rule_rank = np.where(acts["has_internal_rule"].fillna(False), 0, 1)
    type_rank = acts["link_type"].map({"orifice": 0, "weir": 1, "outlet": 2, "pump": 3}).fillna(9)
    acts["_rank"] = storage_rank * 100 + rule_rank * 10 + type_rank
    acts = acts.sort_values(["_rank", "link_id"]).drop(columns=["_rank"])
    if max_actuators and len(acts) > max_actuators:
        forced = acts[acts["force_include"]].copy()
        rest = acts[~acts["force_include"]].copy()
        slots = max(0, max_actuators - len(forced))
        acts = pd.concat([forced, rest.head(slots)], ignore_index=True)
    acts["actuator_index"] = range(len(acts))
    return acts.rename(columns={"link_id": "actuator_id"})


def node_feature_matrix(nodes: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    cols = ["invert", "max_depth", "ponded_area", "degree_in", "degree_out", "is_storage", "is_outfall"]
    for c in cols:
        if c not in nodes:
            nodes[c] = 0.0
    x = nodes[cols].fillna(0).to_numpy(np.float32)
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    x = (x - mean) / std
    return x.astype(np.float32), cols
