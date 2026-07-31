from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


def make_hyetograph(duration_min: int, depth_mm: float, pattern: str, dt_min: int = 5) -> pd.DataFrame:
    n = max(2, int(duration_min / dt_min) + 1)
    t = np.arange(n) * dt_min
    x = np.linspace(0, 1, n)
    if pattern == "chicago_early":
        w = np.exp(-((x - 0.30) / 0.16) ** 2) + 0.12
    elif pattern == "chicago_late":
        w = np.exp(-((x - 0.70) / 0.16) ** 2) + 0.12
    elif pattern == "block":
        w = np.where((x >= 0.25) & (x <= 0.75), 1.0, 0.18)
    elif pattern == "double_peak":
        w = np.exp(-((x - 0.35) / 0.12) ** 2) + 0.85 * np.exp(-((x - 0.72) / 0.10) ** 2) + 0.08
    else:
        w = np.exp(-((x - 0.50) / 0.14) ** 2) + 0.10
    intensity = w / (w.sum() * dt_min / 60.0) * depth_mm
    intensity[-1] = 0.0
    return pd.DataFrame({"elapsed_min": t, "intensity_mm_h": intensity})


def build_rainfall_library(
    rain_ids: list[str],
    durations: list[int],
    depths: Dict[str, float],
    patterns: list[str],
    out_dir: str | Path,
    recession_min: int = 180,
) -> pd.DataFrame:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for rid in rain_ids:
        for dur in durations:
            for pattern in patterns:
                depth = float(depths[rid])
                event_id = f"{rid}_D{dur}_{pattern}"
                hy = make_hyetograph(dur, depth, pattern)
                hy["event_id"] = event_id
                hy.to_csv(out / f"{event_id}.csv", index=False)
                rows.append(
                    {
                        "event_id": event_id,
                        "rain_id": rid,
                        "duration_min": dur,
                        "pattern": pattern,
                        "total_depth_mm": depth,
                        "peak_intensity_mm_h": float(hy["intensity_mm_h"].max()),
                        "recession_min": recession_min,
                        "simulation_duration_min": dur + recession_min,
                        "rainfall_csv": str(out / f"{event_id}.csv"),
                    }
                )
    df = pd.DataFrame(rows)
    df.to_csv(out / "rainfall_event_table.csv", index=False)
    return df

