"""Gate 2.5-real Stage 4: Select two checkpoints at different hydraulic phases.

Analyzes the native internal run1 detail to find:
  Checkpoint A: pre-peak, with native-rule changes 30-60 min after
  Checkpoint B: recession, with native-rule changes 30-60 min after

Outputs (in outputs/project6_dual_reference_v4/recovery_validation/gate2p5_real/):
  - checkpoint_catalog.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "outputs" / "project6_dual_reference_v4" / "recovery_validation" / "gate2p5_real"
RUN1_DETAIL = OUT_DIR / "native_internal_run1_detail.csv"


def _find_change_times(df: pd.DataFrame) -> list[dict]:
    """Find timestamps where action changes occur."""
    action_cols = [c for c in df.columns if c.startswith("a:")]
    changes = []
    for i in range(1, len(df)):
        changed_facilities = []
        for c in action_cols:
            v1 = float(df.iloc[i - 1][c])
            v2 = float(df.iloc[i][c])
            if abs(v2 - v1) > 1e-6:
                aid = c.split(":", 1)[1]
                changed_facilities.append(aid)
        if changed_facilities:
            changes.append({
                "elapsed_min": float(df.iloc[i]["elapsed_min"]),
                "phase": str(df.iloc[i]["phase"]),
                "rainfall_mm_h": float(df.iloc[i]["rainfall_mm_h"]),
                "changed_facilities": changed_facilities,
                "change_count": len(changed_facilities),
            })
    return changes


def _select_checkpoint(changes: list[dict], target_phase: str, duration_min: int,
                       label: str, avoid_range: tuple[float, float] | None = None) -> dict | None:
    """Select a checkpoint where changes happen 30-60 min after it."""
    # Find change times in the target phase
    phase_changes = [c for c in changes if c["phase"] == target_phase]
    if not phase_changes:
        return None

    # Try different checkpoint candidates
    # We want: checkpoint + 30 to checkpoint + 60 contains at least 2 changes
    best_cp = None
    best_score = 0

    for change in phase_changes:
        # Checkpoint should be before the change, with 30-60 min window containing changes
        cp_min = change["elapsed_min"] - 60  # earliest checkpoint
        if cp_min < 10:
            cp_min = 10
        cp_max = change["elapsed_min"] - 30  # latest checkpoint
        if cp_max < cp_min:
            continue

        # Try checkpoints at 5-min intervals
        for cp in np.arange(cp_min, cp_max + 1, 5):
            if avoid_range and avoid_range[0] <= cp <= avoid_range[1]:
                continue
            # Count changes in [cp+30, cp+60]
            window_changes = [c for c in changes if cp + 30 <= c["elapsed_min"] <= cp + 60]
            total_facilities = sum(c["change_count"] for c in window_changes)
            if total_facilities > best_score:
                best_score = total_facilities
                best_cp = {
                    "checkpoint_label": label,
                    "checkpoint_elapsed_min": float(cp),
                    "target_phase": target_phase,
                    "window_changes_30_60": len(window_changes),
                    "window_facilities_30_60": total_facilities,
                    "rainfall_at_checkpoint": 0.0,
                    "planning_only_future_information": True,
                }

    return best_cp


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(RUN1_DETAIL)
    duration_min = 300  # 5h event
    sim_end = float(df["elapsed_min"].max())

    changes = _find_change_times(df)
    print(f"[Stage 4] Found {len(changes)} action-change timestamps")
    for c in changes:
        print(f"  t={c['elapsed_min']:.0f}min phase={c['phase']} rain={c['rainfall_mm_h']:.2f} "
              f"facilities={len(c['changed_facilities'])}: {c['changed_facilities'][:3]}")

    # Get rainfall at each candidate checkpoint
    def _rain_at(df, t):
        row = df.iloc[(df["elapsed_min"] - t).abs().argsort()[:1]]
        return float(row["rainfall_mm_h"].iloc[0])

    # Checkpoint A: peak phase, changes 30-60 min after
    cp_a = _select_checkpoint(changes, "peak", duration_min, "A_pre_peak")
    if cp_a is None:
        # Fallback: use any phase with enough changes
        cp_a = _select_checkpoint(changes, "recession", duration_min, "A_fallback")

    if cp_a:
        cp_a["rainfall_at_checkpoint"] = _rain_at(df, cp_a["checkpoint_elapsed_min"])
        # Compute accumulated rainfall
        rain_before = df[df["elapsed_min"] <= cp_a["checkpoint_elapsed_min"]]
        cp_a["accumulated_rainfall_mm"] = float(
            (rain_before["rainfall_mm_h"] * (5.0 / 60.0)).sum()
        )

    # Checkpoint B: recession phase, different from A
    avoid = None
    if cp_a:
        avoid = (cp_a["checkpoint_elapsed_min"] - 15, cp_a["checkpoint_elapsed_min"] + 15)
    cp_b = _select_checkpoint(changes, "recession", duration_min, "B_recession", avoid_range=avoid)
    if cp_b is None and cp_a:
        # Fallback: pick a different peak time
        cp_b = _select_checkpoint(changes, "peak", duration_min, "B_late_peak", avoid_range=avoid)

    if cp_b:
        cp_b["rainfall_at_checkpoint"] = _rain_at(df, cp_b["checkpoint_elapsed_min"])
        rain_before = df[df["elapsed_min"] <= cp_b["checkpoint_elapsed_min"]]
        cp_b["accumulated_rainfall_mm"] = float(
            (rain_before["rainfall_mm_h"] * (5.0 / 60.0)).sum()
        )

    # Build catalog
    catalog_rows = []
    for cp in [cp_a, cp_b]:
        if cp is None:
            continue
        # Get storage occupancy at checkpoint
        t = cp["checkpoint_elapsed_min"]
        row = df.iloc[(df["elapsed_min"] - t).abs().argsort()[:1]]
        storage_cols = [c for c in df.columns if c.startswith("storage_volume:")]
        total_storage = 0.0
        for sc in storage_cols:
            val = float(row[sc].iloc[0]) if not pd.isna(row[sc].iloc[0]) else 0.0
            total_storage += max(0.0, val)
        cp["storage_occupancy_m3"] = total_storage
        cp["phase_at_checkpoint"] = str(row["phase"].iloc[0])
        cp["sim_end_min"] = sim_end
        cp["duration_min"] = duration_min
        catalog_rows.append(cp)

    catalog_df = pd.DataFrame(catalog_rows)
    catalog_path = OUT_DIR / "checkpoint_catalog.csv"
    catalog_df.to_csv(catalog_path, index=False)
    print(f"\n[Stage 4] Wrote {catalog_path}")

    # Summary
    print(f"\n=== Stage 4 Summary ===")
    for cp in catalog_rows:
        print(f"  Checkpoint {cp['checkpoint_label']}: t={cp['checkpoint_elapsed_min']:.0f}min, "
              f"phase={cp['phase_at_checkpoint']}, rain={cp['rainfall_at_checkpoint']:.2f}mm/h, "
              f"accum_rain={cp['accumulated_rainfall_mm']:.1f}mm, "
              f"storage={cp['storage_occupancy_m3']:.1f}m3, "
              f"window_changes={cp['window_changes_30_60']}")

    if len(catalog_rows) < 2:
        print("WARNING: Could not select 2 distinct checkpoints!")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
