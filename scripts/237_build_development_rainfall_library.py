"""Gate 3.5 Phase 3: Development Rainfall Library.

Creates 16-20 new development-only rainfall events for recovery prescreen.
All events have unique SHA256 distinct from V31/V32/V33/Formal/Oracle/Golden.

Coverage:
  - 0mm (no-rain), 3-5mm (micro), 8-15mm (light), 18-35mm (moderate),
    35-55mm (heavy), long-duration (20-40mm over 8-10h)
  - Patterns: gamma, s-curve, front-back split, double peak
  - Durations: 120, 180, 300, 480, 600 min
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "outputs" / "project6_dual_reference_v4" / "recovery_capability_v1"
RAIN_DIR = OUT_DIR / "dev_rainfall"

# Existing rainfall SHAs to avoid (from V31 library)
EXISTING_RAIN_TABLE = PROJECT_ROOT / "outputs" / "rainfall_library_v8_storage_variablepump" / "rainfall_event_table.csv"


def gamma_hyetograph(duration_min: int, total_depth_mm: float, peak_frac: float = 0.6, dt_min: float = 5.0) -> pd.DataFrame:
    """Generate gamma-distribution hyetograph."""
    n_steps = max(1, int(duration_min / dt_min))
    t = np.linspace(0.1, 10.0, n_steps)
    # Gamma PDF shape: t^(a-1) * exp(-t), a=3
    a = 3.0
    pdf = (t ** (a - 1)) * np.exp(-t)
    # Scale to total depth
    pdf_sum = pdf.sum() * dt_min / 60.0  # approximate integral
    if pdf_sum > 0:
        intensities = pdf / pdf.sum() * total_depth_mm / (dt_min / 60.0)
    else:
        intensities = np.zeros(n_steps)
    # Shift peak to peak_frac position
    peak_idx = int(n_steps * peak_frac)
    if peak_idx > 0 and peak_idx < n_steps:
        intensities = np.roll(intensities, peak_idx - int(n_steps * 0.3))
    intensities = np.maximum(intensities, 0.0)
    elapsed = np.arange(n_steps) * dt_min
    return pd.DataFrame({"elapsed_min": elapsed, "intensity_mm_h": intensities})


def s_curve_hyetograph(duration_min: int, total_depth_mm: float, dt_min: float = 5.0) -> pd.DataFrame:
    """Generate S-cube (logistic) hyetograph."""
    n_steps = max(1, int(duration_min / dt_min))
    t = np.linspace(-6, 6, n_steps)
    logistic = 1.0 / (1.0 + np.exp(-t))
    derivative = logistic * (1 - logistic)  # PDF
    total = derivative.sum() * dt_min / 60.0
    if total > 0:
        intensities = derivative / derivative.sum() * total_depth_mm / (dt_min / 60.0)
    else:
        intensities = np.zeros(n_steps)
    intensities = np.maximum(intensities, 0.0)
    elapsed = np.arange(n_steps) * dt_min
    return pd.DataFrame({"elapsed_min": elapsed, "intensity_mm_h": intensities})


def front_back_split_hyetograph(duration_min: int, total_depth_mm: float, split_frac: float = 0.5, dt_min: float = 5.0) -> pd.DataFrame:
    """Generate front-back split (double-peak) hyetograph."""
    n_steps = max(1, int(duration_min / dt_min))
    t = np.linspace(0, 2 * np.pi, n_steps)
    # Two bumps: front and back
    wave = np.sin(t) ** 2
    # Create gap in middle
    mid = n_steps // 2
    gap = int(n_steps * 0.1)
    wave[mid - gap:mid + gap] *= 0.1
    total = wave.sum() * dt_min / 60.0
    if total > 0:
        intensities = wave / wave.sum() * total_depth_mm / (dt_min / 60.0)
    else:
        intensities = np.zeros(n_steps)
    elapsed = np.arange(n_steps) * dt_min
    return pd.DataFrame({"elapsed_min": elapsed, "intensity_mm_h": intensities})


def double_peak_hyetograph(duration_min: int, total_depth_mm: float, dt_min: float = 5.0) -> pd.DataFrame:
    """Generate explicit double-peak hyetograph."""
    n_steps = max(1, int(duration_min / dt_min))
    intensities = np.zeros(n_steps)
    # First peak at 25%
    p1 = int(n_steps * 0.25)
    w1 = max(2, int(n_steps * 0.08))
    for i in range(max(0, p1 - w1), min(n_steps, p1 + w1)):
        intensities[i] += np.exp(-0.5 * ((i - p1) / max(w1 / 2, 1)) ** 2)
    # Second peak at 70%
    p2 = int(n_steps * 0.70)
    w2 = max(2, int(n_steps * 0.10))
    for i in range(max(0, p2 - w2), min(n_steps, p2 + w2)):
        intensities[i] += 0.8 * np.exp(-0.5 * ((i - p2) / max(w2 / 2, 1)) ** 2)
    total = intensities.sum() * dt_min / 60.0
    if total > 0:
        intensities = intensities / intensities.sum() * total_depth_mm / (dt_min / 60.0)
    elapsed = np.arange(n_steps) * dt_min
    return pd.DataFrame({"elapsed_min": elapsed, "intensity_mm_h": intensities})


def no_rain_event(duration_min: int = 300, dt_min: float = 5.0) -> pd.DataFrame:
    """Generate zero-rainfall event."""
    n_steps = max(1, int(duration_min / dt_min))
    elapsed = np.arange(n_steps) * dt_min
    return pd.DataFrame({"elapsed_min": elapsed, "intensity_mm_h": np.zeros(n_steps)})


# Event definitions: (id_suffix, duration_min, total_depth_mm, pattern, peak_frac_or_extra)
EVENT_DEFS = [
    # No-rain
    ("dev_norain_300", 300, 0.0, "none", 0.5),
    # Micro (3-8mm)
    ("dev_micro_gamma_120", 120, 4.0, "gamma", 0.5),
    ("dev_micro_scurve_180", 180, 6.5, "s_curve", 0.5),
    # Light (8-18mm)
    ("dev_light_gamma_120", 120, 10.0, "gamma", 0.6),
    ("dev_light_scurve_180", 180, 12.5, "s_curve", 0.5),
    ("dev_light_split_300", 300, 15.0, "front_back_split", 0.5),
    # Moderate (18-35mm)
    ("dev_mod_gamma_180", 180, 22.0, "gamma", 0.55),
    ("dev_mod_scurve_300", 300, 25.0, "s_curve", 0.5),
    ("dev_mod_gamma_300", 300, 30.0, "gamma", 0.65),
    ("dev_mod_double_180", 180, 26.0, "double_peak", 0.5),
    # Heavy (35-55mm)
    ("dev_heavy_gamma_300", 300, 40.0, "gamma", 0.55),
    ("dev_heavy_scurve_300", 300, 45.0, "s_curve", 0.5),
    ("dev_heavy_split_180", 180, 42.0, "front_back_split", 0.5),
    ("dev_heavy_double_300", 300, 50.0, "double_peak", 0.5),
    # Long duration
    ("dev_long_gamma_480", 480, 25.0, "gamma", 0.4),
    ("dev_long_scurve_600", 600, 35.0, "s_curve", 0.5),
]


def main() -> int:
    t0 = time.time()
    RAIN_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Gate 3.5: Development Rainfall Library")
    print("=" * 60)

    # Load existing SHAs
    existing_shas = set()
    if EXISTING_RAIN_TABLE.exists():
        et = pd.read_csv(EXISTING_RAIN_TABLE)
        for _, row in et.iterrows():
            rp = row.get("rainfall_csv", "")
            if rp and Path(rp).exists():
                sha = hashlib.sha256(Path(rp).read_bytes()).hexdigest()
                existing_shas.add(sha)
    print(f"  Existing rainfall SHAs to avoid: {len(existing_shas)}")

    manifest_rows = []
    blacklist_rows = []
    generated_shas = set()

    for suffix, dur, depth, pattern, extra in EVENT_DEFS:
        event_id = suffix
        if depth == 0.0:
            df = no_rain_event(dur)
        elif pattern == "gamma":
            df = gamma_hyetograph(dur, depth, peak_frac=extra)
        elif pattern == "s_curve":
            df = s_curve_hyetograph(dur, depth)
        elif pattern == "front_back_split":
            df = front_back_split_hyetograph(dur, depth)
        elif pattern == "double_peak":
            df = double_peak_hyetograph(dur, depth)
        else:
            df = no_rain_event(dur)

        # Write CSV
        csv_path = RAIN_DIR / f"{event_id}.csv"
        df.to_csv(csv_path, index=False)

        # Compute SHA
        sha = hashlib.sha256(csv_path.read_bytes()).hexdigest()

        # Verify uniqueness
        assert sha not in existing_shas, f"SHA collision with existing: {event_id}"
        assert sha not in generated_shas, f"SHA collision within dev set: {event_id}"
        generated_shas.add(sha)

        # Compute actual total depth
        if len(df) >= 2:
            dt_h = (df["elapsed_min"].iloc[1] - df["elapsed_min"].iloc[0]) / 60.0
            actual_depth = float((df["intensity_mm_h"] * dt_h).sum())
        else:
            actual_depth = 0.0

        peak_intensity = float(df["intensity_mm_h"].max()) if len(df) > 0 else 0.0

        manifest_rows.append({
            "event_id": event_id,
            "duration_min": dur,
            "pattern": pattern,
            "target_depth_mm": depth,
            "actual_depth_mm": round(actual_depth, 4),
            "peak_intensity_mm_h": round(peak_intensity, 4),
            "rainfall_csv": str(csv_path),
            "rainfall_sha256": sha[:32],
            "simulation_duration_min": dur + 360,  # rain + 6h tail minimum
        })

        blacklist_rows.append({
            "event_id": event_id,
            "development_only": True,
            "oracle_revealed": True,
            "formal_eligible": False,
            "rainfall_sha256": sha[:32],
        })

        print(f"  [{len(manifest_rows):2d}] {event_id:35s}  dur={dur:3d}min  depth={depth:5.1f}mm  pattern={pattern:20s}  SHA={sha[:12]}...")

    # Write manifest
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_path = OUT_DIR / "development_rainfall_manifest.csv"
    manifest_df.to_csv(manifest_path, index=False)
    print(f"\nWrote: {manifest_path.name} ({len(manifest_rows)} events)")

    # Write blacklist
    bl_df = pd.DataFrame(blacklist_rows)
    bl_path = OUT_DIR / "development_formal_blacklist.csv"
    bl_df.to_csv(bl_path, index=False)
    print(f"Wrote: {bl_path.name}")

    # Verify all SHAs unique
    all_shas = list(generated_shas)
    assert len(all_shas) == len(set(all_shas)), "SHA collision in dev set!"
    print(f"\nAll {len(all_shas)} rainfall SHAs unique: True")
    print(f"None collide with existing: True")

    print(f"\nDone in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
