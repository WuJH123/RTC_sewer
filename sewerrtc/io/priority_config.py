from __future__ import annotations

from pathlib import Path
from typing import Any

from .project_paths import cfg_path


def read_node_list(path: str | Path, *, allow_missing: bool = False) -> list[str]:
    p = Path(path)
    if not p.exists():
        if allow_missing:
            return []
        raise FileNotFoundError(f"Missing node list: {p}")
    return [
        x.strip()
        for x in p.read_text(encoding="utf-8-sig", errors="ignore").splitlines()
        if x.strip() and not x.strip().startswith("#")
    ]


def _optional_cfg_path(cfg: dict[str, Any], dotted: str) -> Path | None:
    obj: Any = cfg
    for key in dotted.split("."):
        if not isinstance(obj, dict) or key not in obj or obj[key] in ("", None):
            return None
        obj = obj[key]
    return cfg_path(cfg, dotted)


def configured_priority_nodes(cfg: dict[str, Any]) -> list[str]:
    path = _optional_cfg_path(cfg, "network.priority_nodes_file")
    if path is None:
        path = cfg_path(cfg, "outputs.design") / "priority_nodes.txt"
    return read_node_list(path)


def configured_priority_sentinel_nodes(cfg: dict[str, Any]) -> list[str]:
    path = _optional_cfg_path(cfg, "network.priority_sentinel_nodes_file")
    if path is None:
        path = cfg_path(cfg, "outputs.design") / "priority_sentinel_nodes.txt"
    return read_node_list(path, allow_missing=True)


def combined_priority_depth_nodes(cfg: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for node_id in configured_priority_nodes(cfg) + configured_priority_sentinel_nodes(cfg):
        if node_id not in out:
            out.append(node_id)
    return out


def priority_config_summary(cfg: dict[str, Any]) -> dict[str, object]:
    core = configured_priority_nodes(cfg)
    sentinels = configured_priority_sentinel_nodes(cfg)
    return {
        "priority_definition": cfg.get("network", {}).get("priority_definition", ""),
        "priority_node_role": "pfv_core",
        "priority_nodes": core,
        "priority_node_count": len(core),
        "sentinel_node_role": "depth_surcharge_monitoring",
        "sentinel_nodes": sentinels,
        "sentinel_node_count": len(sentinels),
        "depth_safety_nodes": combined_priority_depth_nodes(cfg),
    }
