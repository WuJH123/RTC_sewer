from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from .contracts import sha256_file


def partition_events(
    events: pd.DataFrame,
    counts: Mapping[str, int],
    *,
    seed: int = 20260727,
) -> pd.DataFrame:
    required = {"event_id", "rainfall_sha256", "eligible", "revealed"}
    missing = required - set(events)
    if missing:
        raise ValueError(f"inventory missing columns: {sorted(missing)}")
    source = events[
        events["eligible"].astype(bool) & ~events["revealed"].astype(bool)
    ].copy()
    if source["rainfall_sha256"].duplicated().any():
        raise ValueError("duplicate rainfall_sha256 in eligible inventory")
    total = sum(int(value) for value in counts.values())
    if len(source) < total:
        raise ValueError(f"need {total} eligible events, found {len(source)}")
    order = np.random.default_rng(seed).permutation(len(source))[:total]
    selected = source.iloc[order].reset_index(drop=True)
    splits: list[str] = []
    for split, count in counts.items():
        splits.extend([str(split)] * int(count))
    selected["split"] = splits
    return selected


def build_inventory_from_catalog(
    catalog: pd.DataFrame,
    *,
    project_root: str,
    revealed_event_ids: set[str] | None = None,
) -> pd.DataFrame:
    if "event_id" not in catalog:
        raise ValueError("event catalog is missing event_id")
    rainfall_column = next(
        (
            name
            for name in ("rainfall_path", "rain_file", "rainfall_file")
            if name in catalog
        ),
        None,
    )
    if rainfall_column is None:
        raise ValueError("event catalog is missing rainfall path")
    root = Path(project_root)
    result = catalog.copy()
    hashes = []
    exists = []
    for raw in result[rainfall_column].astype(str):
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = root / candidate
        exists.append(candidate.exists())
        hashes.append(sha256_file(candidate) if candidate.exists() else "")
    result["rainfall_sha256"] = hashes
    result["rainfall_exists"] = exists
    revealed = revealed_event_ids or set()
    result["revealed"] = result["event_id"].astype(str).isin(revealed)
    result["eligible"] = (
        result["rainfall_exists"]
        & result["rainfall_sha256"].astype(str).str.len().eq(64)
        & ~result["revealed"]
    )
    return result
