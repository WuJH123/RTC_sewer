from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml


def _find_project_root(start: Path) -> Path | None:
    """Walk up from *start* looking for a directory containing .git."""
    cur = start.resolve()
    for parent in [cur, *cur.parents]:
        if (parent / ".git").exists():
            return parent
    return None


def load_config(config_path: str | Path) -> Dict[str, Any]:
    config_path = Path(config_path).resolve()
    with config_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if "_inherits" in cfg:
        parent_path = Path(cfg["_inherits"])
        if not parent_path.is_absolute():
            # Try .git-based project root detection first
            detected = _find_project_root(config_path)
            if detected is not None:
                parent_path = detected / parent_path
            else:
                parent_path = config_path.parent.parent / parent_path
        parent = load_config(parent_path)
        cfg = _deep_merge(parent, {k: v for k, v in cfg.items() if k != "_inherits"})
    root_raw = cfg.get("project_root", ".")
    root = Path(root_raw)
    if not root.is_absolute():
        # Try .git-based detection first, fall back to config-relative
        detected = _find_project_root(config_path)
        if detected is not None:
            root = detected
        else:
            root = config_path.parent.parent / root
    cfg["project_root"] = str(root.resolve())
    cfg["_config_path"] = str(config_path)
    return cfg


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def project_path(cfg: Dict[str, Any], *parts: str | Path) -> Path:
    p = Path(cfg["project_root"])
    for part in parts:
        part = Path(part)
        p = part if part.is_absolute() else p / part
    return p


def cfg_path(cfg: Dict[str, Any], dotted: str) -> Path:
    obj: Any = cfg
    for key in dotted.split("."):
        obj = obj[key]
    p = Path(obj)
    if not p.is_absolute():
        p = Path(cfg["project_root"]) / p
    return p


def optional_cfg_path(cfg: Dict[str, Any], dotted: str) -> Path | None:
    obj: Any = cfg
    for key in dotted.split("."):
        if not isinstance(obj, dict) or key not in obj:
            return None
        obj = obj[key]
    if obj in (None, ""):
        return None
    p = Path(obj)
    if not p.is_absolute():
        p = Path(cfg["project_root"]) / p
    return p


def resolve_gat_model_path(cfg: Dict[str, Any]) -> Path:
    """Resolve the online GAT checkpoint without tying it to outputs.models."""
    for dotted in (
        "controller.gat_model_path",
        "models.gat_model_path",
        "gat.model_path",
    ):
        path = optional_cfg_path(cfg, dotted)
        if path is not None:
            return path
    return cfg_path(cfg, "outputs.models") / "gat_sr0p10.pt"


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, data: Dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
