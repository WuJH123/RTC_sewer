from __future__ import annotations

import numpy as np
import pandas as pd

from sewerrtc.io.project_paths import cfg_path


def _numeric_column(frame: pd.DataFrame, name: str) -> np.ndarray:
    if name not in frame:
        return np.zeros(len(frame), dtype=np.float32)
    return pd.to_numeric(frame[name], errors="coerce").fillna(0.0).to_numpy(np.float32)


def _robust_scale(values: np.ndarray) -> np.ndarray:
    values = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    nonzero = np.abs(values[np.abs(values) > 0.0])
    scale = float(np.quantile(nonzero, 0.90)) if nonzero.size else 1.0
    return np.clip(values / max(scale, 1.0e-6), -5.0, 5.0).astype(np.float32)


def build_actuator_feature_matrix(actuators: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Build typed hydraulic actuator features without relying on identity alone."""
    link_type = actuators["link_type"].astype(str).str.lower()
    storage_type = actuators.get(
        "storage_control_type", pd.Series("not_storage", index=actuators.index)
    ).astype(str)
    bool_names = [
        "is_pump",
        "is_orifice",
        "is_weir",
        "near_storage",
        "is_storage_inlet",
        "is_storage_outlet",
        "is_existing_rtc",
        "is_physically_controllable",
        "has_internal_rule",
    ]
    bool_features = np.stack(
        [
            link_type.eq("pump"),
            link_type.eq("orifice"),
            link_type.eq("weir"),
            actuators.get("near_storage", pd.Series(False, index=actuators.index)).astype(bool),
            storage_type.eq("storage_inlet"),
            storage_type.eq("storage_outlet"),
            actuators.get("is_existing_rtc", pd.Series(False, index=actuators.index)).astype(bool),
            actuators.get("is_physically_controllable", pd.Series(False, index=actuators.index)).astype(bool),
            actuators.get("has_internal_rule", pd.Series(False, index=actuators.index)).astype(bool),
        ],
        axis=1,
    ).astype(np.float32)

    geom1 = np.log1p(np.maximum(_numeric_column(actuators, "geom1"), 0.0))
    geom2 = np.log1p(np.maximum(_numeric_column(actuators, "geom2"), 0.0))
    from_depth = _numeric_column(actuators, "from_max_depth")
    to_depth = _numeric_column(actuators, "to_max_depth")
    head_lift = _numeric_column(actuators, "to_invert") - _numeric_column(actuators, "from_invert")
    depth_drop = from_depth - to_depth
    area_proxy = np.maximum(_numeric_column(actuators, "geom1"), 0.0) * np.maximum(
        _numeric_column(actuators, "geom2"), 0.0
    )
    numeric_names = [
        "geom1_log_scaled",
        "geom2_log_scaled",
        "area_proxy_log_scaled",
        "from_max_depth_scaled",
        "to_max_depth_scaled",
        "depth_drop_scaled",
        "invert_head_lift_scaled",
    ]
    numeric_features = np.stack(
        [
            _robust_scale(geom1),
            _robust_scale(geom2),
            _robust_scale(np.log1p(area_proxy)),
            _robust_scale(from_depth),
            _robust_scale(to_depth),
            _robust_scale(depth_drop),
            _robust_scale(head_lift),
        ],
        axis=1,
    ).astype(np.float32)
    return np.concatenate([bool_features, numeric_features], axis=1), bool_names + numeric_names


def diffuse_action_node_map(
    base_map: np.ndarray,
    edge_index: np.ndarray,
    *,
    hops: int,
    decay: float,
) -> np.ndarray:
    """Expand actuator endpoint influence over a normalized graph neighbourhood."""
    base = np.asarray(base_map, dtype=np.float32)
    edges = np.asarray(edge_index, dtype=np.int64)
    if base.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("base_map must be [A,N] and edge_index must be [2,E]")
    if int(hops) <= 0:
        return base.copy()
    if not 0.0 < float(decay) <= 1.0:
        raise ValueError("decay must be in (0,1]")
    node_count = base.shape[1]
    adjacency = np.zeros((node_count, node_count), dtype=np.float32)
    adjacency[edges[0], edges[1]] = 1.0
    adjacency[np.arange(node_count), np.arange(node_count)] = 1.0
    adjacency /= np.maximum(adjacency.sum(axis=1, keepdims=True), 1.0)
    current = base.copy()
    expanded = base.copy()
    for hop in range(1, int(hops) + 1):
        current = current @ adjacency
        expanded += (float(decay) ** hop) * current
    expanded /= np.maximum(expanded.sum(axis=1, keepdims=True), 1.0e-12)
    return expanded.astype(np.float32)


def build_raw_joint_assets(
    cfg: dict,
    node_ids: list[str],
    action_ids: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    nodes = pd.read_csv(cfg_path(cfg, "outputs.audit") / "node_table.csv").set_index("node_id")
    links = pd.read_csv(cfg_path(cfg, "outputs.audit") / "link_table.csv").set_index("link_id")
    actuators = pd.read_csv(cfg_path(cfg, "outputs.audit") / "actuator_table.csv").set_index("actuator_id").loc[action_ids]
    static_cols = ["invert", "max_depth", "ponded_area", "degree_in", "degree_out", "is_storage", "is_outfall"]
    node_static = np.asarray(
        [nodes.loc[node, static_cols].to_numpy(float) if node in nodes.index else np.zeros(len(static_cols)) for node in node_ids],
        dtype=np.float32,
    )
    node_static = (node_static - node_static.mean(0, keepdims=True)) / np.maximum(node_static.std(0, keepdims=True), 1.0e-6)
    node_index = {node: i for i, node in enumerate(node_ids)}
    edges: list[tuple[int, int]] = []
    for row in links.itertuples():
        source, target = str(row.from_node), str(row.to_node)
        if source in node_index and target in node_index:
            edges.extend([(node_index[source], node_index[target]), (node_index[target], node_index[source])])
    edge_index = np.asarray(edges or [(i, i) for i in range(len(node_ids))], dtype=np.int64).T
    action_node_map = np.zeros((len(action_ids), len(node_ids)), dtype=np.float32)
    for action_index, action_id in enumerate(action_ids):
        row = links.loc[action_id] if action_id in links.index else None
        if row is not None:
            for node, weight in ((str(row.from_node), 1.0), (str(row.to_node), 0.6)):
                if node in node_index:
                    action_node_map[action_index, node_index[node]] = weight
        if action_node_map[action_index].sum() <= 0:
            action_node_map[action_index] = 1.0 / len(node_ids)
        else:
            action_node_map[action_index] /= action_node_map[action_index].sum()
    temporal = ((cfg.get("controller", {}) or {}).get("temporal_joint", {}) or {})
    influence_hops = int(temporal.get("action_influence_hops", 0))
    influence_decay = float(temporal.get("action_influence_decay", 0.6))
    action_node_map = diffuse_action_node_map(
        action_node_map,
        edge_index,
        hops=influence_hops,
        decay=influence_decay,
    )
    actuator_features, _ = build_actuator_feature_matrix(actuators)
    priority = [
        value.strip()
        for value in (cfg_path(cfg, "outputs.design") / "priority_nodes.txt").read_text().splitlines()
        if value.strip()
    ]
    storage = nodes[nodes["node_type"].astype(str).eq("storage")].index.astype(str).tolist()
    return (
        node_static,
        edge_index,
        action_node_map,
        actuator_features,
        np.asarray([node_index[node] for node in priority if node in node_index], dtype=np.int64),
        np.asarray([node_index[node] for node in storage if node in node_index], dtype=np.int64),
    )
