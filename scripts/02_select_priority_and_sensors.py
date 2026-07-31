from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from sewerrtc.graph.priority_zone import select_priority_nodes
from sewerrtc.graph.sensor_selection import select_sensors
from sewerrtc.io.priority_config import configured_priority_sentinel_nodes
from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config


def _read_priority_nodes(path: Path) -> list[str]:
    return [
        x.strip()
        for x in path.read_text(encoding="utf-8-sig", errors="ignore").splitlines()
        if x.strip() and not x.strip().startswith("#")
    ]


def _explicit_priority_table(cfg: dict, nodes: pd.DataFrame) -> pd.DataFrame | None:
    priority_file_raw = cfg.get("network", {}).get("priority_nodes_file", "")
    if not priority_file_raw:
        return None
    priority_file = cfg_path(cfg, "network.priority_nodes_file")
    if not priority_file.exists():
        raise FileNotFoundError(f"Configured priority_nodes_file does not exist: {priority_file}")

    priority_ids = _read_priority_nodes(priority_file)
    node_table = nodes.copy()
    node_table["node_id"] = node_table["node_id"].astype(str)
    node_lookup = node_table.set_index("node_id", drop=False)
    missing = [n for n in priority_ids if n not in node_lookup.index]
    if missing:
        raise ValueError(f"Configured priority nodes are absent from audited SWMM nodes: {missing}")

    rows = []
    metadata = pd.DataFrame()
    metadata_raw = cfg.get("network", {}).get("priority_nodes_metadata", "")
    if metadata_raw:
        metadata_path = cfg_path(cfg, "network.priority_nodes_metadata")
        if metadata_path.exists():
            metadata = pd.read_csv(metadata_path)
            if "node_id" in metadata:
                metadata["node_id"] = metadata["node_id"].astype(str)
                metadata = metadata.set_index("node_id", drop=False)
    for rank, node_id in enumerate(priority_ids, start=1):
        src = node_lookup.loc[node_id]
        meta = metadata.loc[node_id].to_dict() if not metadata.empty and node_id in metadata.index else {}
        rows.append(
            {
                "priority_rank": rank,
                "node_id": node_id,
                "node_type": src.get("node_type", ""),
                "invert": src.get("invert", float("nan")),
                "max_depth": src.get("max_depth", float("nan")),
                "priority_score": float(len(priority_ids) - rank + 1) / max(len(priority_ids), 1),
                "priority_source": cfg.get("network", {}).get("priority_definition", "explicit_file"),
                "role": meta.get("role", "explicit_priority_node"),
            }
        )
    return pd.DataFrame(rows)


def _explicit_sentinel_table(cfg: dict, nodes: pd.DataFrame) -> pd.DataFrame:
    sentinel_ids = configured_priority_sentinel_nodes(cfg)
    node_table = nodes.copy()
    node_table["node_id"] = node_table["node_id"].astype(str)
    node_lookup = node_table.set_index("node_id", drop=False)
    missing = [n for n in sentinel_ids if n not in node_lookup.index]
    if missing:
        raise ValueError(f"Configured priority sentinel nodes are absent from audited SWMM nodes: {missing}")

    metadata = pd.DataFrame()
    metadata_raw = cfg.get("network", {}).get("priority_sentinel_nodes_metadata", "")
    if metadata_raw:
        metadata_path = cfg_path(cfg, "network.priority_sentinel_nodes_metadata")
        if metadata_path.exists():
            metadata = pd.read_csv(metadata_path)
            if "node_id" in metadata:
                metadata["node_id"] = metadata["node_id"].astype(str)
                metadata = metadata.set_index("node_id", drop=False)

    rows = []
    for rank, node_id in enumerate(sentinel_ids, start=1):
        src = node_lookup.loc[node_id]
        meta = metadata.loc[node_id].to_dict() if not metadata.empty and node_id in metadata.index else {}
        rows.append(
            {
                "sentinel_rank": rank,
                "node_id": node_id,
                "node_type": src.get("node_type", ""),
                "invert": src.get("invert", float("nan")),
                "max_depth": src.get("max_depth", float("nan")),
                "sentinel_source": cfg.get("network", {}).get("priority_definition", "explicit_file"),
                "role": meta.get("role", "depth_surcharge_sentinel"),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wuhan.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    audit = cfg_path(cfg, "outputs.audit")
    out = ensure_dir(cfg_path(cfg, "outputs.design"))
    nodes = pd.read_csv(audit / "node_table.csv")
    links = pd.read_csv(audit / "link_table.csv")
    priority = _explicit_priority_table(cfg, nodes)
    if priority is None:
        priority = select_priority_nodes(nodes, links, int(cfg["experiment"]["priority_top_k"]))
        priority["priority_source"] = "structural_top_k_fallback"
        priority.insert(0, "priority_rank", range(1, len(priority) + 1))
    sentinel = _explicit_sentinel_table(cfg, nodes)
    priority.to_csv(out / "priority_nodes.csv", index=False)
    (out / "priority_nodes.txt").write_text("\n".join(priority["node_id"].astype(str)) + "\n", encoding="utf-8")
    if not sentinel.empty:
        sentinel.to_csv(out / "priority_sentinel_nodes.csv", index=False)
        (out / "priority_sentinel_nodes.txt").write_text(
            "\n".join(sentinel["node_id"].astype(str)) + "\n", encoding="utf-8"
        )
    else:
        (out / "priority_sentinel_nodes.txt").write_text("", encoding="utf-8")
    domain_nodes = []
    for node_id in priority["node_id"].astype(str).tolist() + sentinel.get("node_id", pd.Series(dtype=str)).astype(str).tolist():
        if node_id not in domain_nodes:
            domain_nodes.append(node_id)
    (out / "priority_domain_nodes.txt").write_text("\n".join(domain_nodes) + "\n", encoding="utf-8")
    sensor_cfg = cfg.get("sensor_selection", {}) or {}
    sensor_priority_nodes = priority["node_id"].astype(str).tolist()
    if bool(sensor_cfg.get("include_sentinel_nodes", False)):
        for node_id in sentinel.get("node_id", pd.Series(dtype=str)).astype(str).tolist():
            if node_id not in sensor_priority_nodes:
                sensor_priority_nodes.append(node_id)
    sensors = select_sensors(
        nodes,
        sensor_priority_nodes,
        float(cfg["experiment"]["sensor_ratio"]),
        int(cfg["experiment"]["random_seed"]),
        include_priority_nodes=bool(sensor_cfg.get("include_priority_nodes", True)),
        priority_sensor_fraction=float(sensor_cfg.get("priority_sensor_fraction", 0.25)),
    )
    sensors.to_csv(out / "sensor_nodes.csv", index=False)
    (out / "sensor_nodes.txt").write_text("\n".join(sensors["node_id"].astype(str)) + "\n", encoding="utf-8")
    sensor_ids = set(sensors["node_id"].astype(str))
    report = {
        "priority_nodes": len(priority),
        "priority_sentinel_nodes": len(sentinel),
        "priority_domain_nodes": len(domain_nodes),
        "sensors": len(sensors),
        "priority_sensor_overlap": len(sensor_ids.intersection(priority["node_id"].astype(str))),
        "sentinel_sensor_overlap": len(sensor_ids.intersection(sentinel.get("node_id", pd.Series(dtype=str)).astype(str))),
        "include_priority_nodes": bool(sensor_cfg.get("include_priority_nodes", True)),
        "passed": True,
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
