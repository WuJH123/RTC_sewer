from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Any


def _float_or_none(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def read_rainfall_series(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    with p.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {
            "row_count": 0,
            "timestamp_values": [],
            "intensity_values": [],
            "interval_min": "",
            "total_depth_mm": "",
            "peak_intensity_mm_h": "",
            "peak_time_min": "",
            "status": "empty",
        }
    fieldnames = list(rows[0].keys())
    time_col = next((c for c in ["elapsed_min", "time_min", "minute", "t_min"] if c in fieldnames), fieldnames[0])
    intensity_col = next((c for c in ["intensity_mm_h", "rainfall_mm_h", "intensity", "value"] if c in fieldnames), fieldnames[1] if len(fieldnames) > 1 else fieldnames[0])
    times = [_float_or_none(str(row.get(time_col, ""))) for row in rows]
    intensities = [_float_or_none(str(row.get(intensity_col, ""))) for row in rows]
    valid_pairs = [(t, i) for t, i in zip(times, intensities) if t is not None and i is not None]
    if not valid_pairs:
        return {
            "row_count": len(rows),
            "timestamp_values": [],
            "intensity_values": [],
            "interval_min": "",
            "total_depth_mm": "",
            "peak_intensity_mm_h": "",
            "peak_time_min": "",
            "status": "no_numeric_series",
        }
    clean_times = [t for t, _ in valid_pairs]
    clean_intensities = [i for _, i in valid_pairs]
    diffs = [b - a for a, b in zip(clean_times, clean_times[1:]) if b > a]
    interval = min(diffs) if diffs else 0.0
    total_depth = sum(i * interval / 60.0 for i in clean_intensities) if interval else 0.0
    peak = max(clean_intensities)
    peak_index = clean_intensities.index(peak)
    return {
        "row_count": len(clean_intensities),
        "timestamp_values": clean_times,
        "intensity_values": clean_intensities,
        "interval_min": interval,
        "total_depth_mm": total_depth,
        "peak_intensity_mm_h": peak,
        "peak_time_min": clean_times[peak_index],
        "status": "parsed",
    }


def normalized_series_hash(values: list[float]) -> str:
    payload = "\n".join(f"{v:.8f}" for v in values)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def timestamp_hash(values: list[float]) -> str:
    payload = "\n".join(f"{v:.8f}" for v in values)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
