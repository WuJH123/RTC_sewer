"""Build event/rainfall isolation groups for reused historical trajectories.

The same rainfall may exist in several historical directories or DWF/no-DWF
domains.  Row-random splitting is therefore prohibited.  This module derives a
base rainfall fingerprint directly from the recorded rainfall time series and
uses it as the cross-version grouping key.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


def _read_table(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def _rainfall_identity(detail_path: str | Path, fallback_event: str) -> tuple[str, str]:
    path = Path(detail_path)
    if not path.exists():
        return fallback_event, ""
    header = pd.read_csv(path, nrows=0)
    cols = set(str(x) for x in header.columns)
    usecols = [x for x in ("elapsed_min", "rainfall_mm_h", "event_id") if x in cols]
    if "elapsed_min" not in usecols or "rainfall_mm_h" not in usecols:
        return fallback_event, ""
    df = pd.read_csv(path, usecols=usecols)
    elapsed = pd.to_numeric(df["elapsed_min"], errors="coerce").to_numpy(float)
    rain = pd.to_numeric(df["rainfall_mm_h"], errors="coerce").to_numpy(float)
    finite = np.isfinite(elapsed) & np.isfinite(rain)
    elapsed = elapsed[finite]
    rain = rain[finite]
    order = np.argsort(elapsed)
    payload = np.column_stack([elapsed[order], rain[order]]).astype(np.float64)
    fingerprint = hashlib.sha256(payload.tobytes(order="C")).hexdigest() if payload.size else ""
    event_id = fallback_event
    if "event_id" in df.columns:
        values = [str(x).strip() for x in df["event_id"].dropna().unique().tolist() if str(x).strip()]
        if len(values) == 1:
            event_id = values[0]
    return event_id, fingerprint


def build_reuse_split_groups(
    *,
    reusable_physical_manifest: str | Path,
    output_path: str | Path,
) -> pd.DataFrame:
    source = Path(reusable_physical_manifest)
    frame = _read_table(source)
    if frame.empty:
        raise ValueError("reusable physical manifest is empty")
    rows = []
    cache: dict[str, tuple[str, str]] = {}
    for row in frame.itertuples(index=False):
        detail = str(getattr(row, "detail_path"))
        fallback_event = str(getattr(row, "event_id", ""))
        if detail not in cache:
            cache[detail] = _rainfall_identity(detail, fallback_event)
        event_id, fingerprint = cache[detail]
        rows.append(
            {
                "physical_identity_sha256": str(getattr(row, "physical_identity_sha256")),
                "event_id_effective": event_id,
                "base_rainfall_fingerprint": fingerprint,
                "domain_id": str(getattr(row, "domain_id", "")),
                "source_role": str(getattr(row, "source_role", "development")),
                "split_group_key": fingerprint or event_id,
                "reserved_evaluation": str(getattr(row, "source_role", "")) == "reserved_evaluation",
            }
        )
    result = pd.DataFrame(rows)
    # A fingerprint is deliberately shared across DWF/no-DWF variants.  It must
    # remain within a single future train/evaluation partition.
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix.lower() == ".parquet":
        result.to_parquet(target, index=False)
    else:
        result.to_csv(target, index=False)
    return result
