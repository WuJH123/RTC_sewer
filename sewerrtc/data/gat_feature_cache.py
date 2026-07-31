from __future__ import annotations

from pathlib import Path


def gat_feature_cache_path(cache_dir: str | Path, detail_path: str | Path) -> Path:
    detail = Path(detail_path)
    return Path(cache_dir) / f"{detail.stem}__gat_features.npz"
