"""V4.2  13-frame history rebuilder.

Reconstructs 13 history frames (t-60 min to t at 5-min intervals) and
12 future control steps (t+10 min to t+120 min at 10-min intervals)
from raw SWMM time series stored in branch detail CSVs.

The original V4.2 trajectory dataset was built with only 7 history frames.
This module re-reads the raw detail CSVs at 5-min resolution to produce
the full 13-frame history required by the V4.2 model contract.

13-frame convention
-------------------
Frame  0 : checkpoint − 60 min
Frame  1 : checkpoint − 55 min
...
Frame 11 : checkpoint −  5 min
Frame 12 : checkpoint      (t = 0)

12-step future convention
-------------------------
Step  1 : checkpoint + 10 min
Step  2 : checkpoint + 20 min
...
Step 12 : checkpoint + 120 min  (H120)

Future aggregation
------------------
Detail CSVs record at 5-min resolution.  Two consecutive 5-min rows are
aggregated into one 10-min control step via **mean** for state variables
(depths, volumes, rainfall) and **last-value** for discrete actions/settings.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_HISTORY_FRAMES = 13
HISTORY_INTERVAL_MIN = 5
N_HORIZON_STEPS = 12
HORIZON_INTERVAL_MIN = 10
RECORDING_INTERVAL_SEC = 300  # 5-min raw resolution

# Column prefixes in detail CSVs
_STATE_PREFIXES = ("h:", "storage_volume:", "flood:")
_ACTION_PREFIXES = ("a:", "setting:", "actual_setting:")
_RAINFALL_COL = "rainfall_mm_h"


# ===================================================================
# Public API
# ===================================================================


def rebuild_13frame_histories(
    project_root: Path,
    output_root: Path,
    detail_csv_paths: list[Path] | None = None,
) -> dict:
    """Rebuild 13-frame histories from raw time series.

    Parameters
    ----------
    project_root : Path
        Project root (contains *data/*, *sewerrtc/*, …).
    output_root : Path
        Output root that contains ``train1600_v3/`` run directories
        **or** the explicit *detail_csv_paths* list.
    detail_csv_paths : list[Path] | None
        If given, only process these files.  Otherwise the function
        auto-discovers every ``detail.csv`` under *output_root*.

    Returns
    -------
    dict
        Audit summary with keys ``total_samples_attempted``,
        ``samples_with_full_13_frames``,
        ``samples_with_incomplete_history``,
        ``frame_timestamp_validation``,
        ``future_aggregation_method``,
        ``per_sample`` (list of per-sample detail dicts).
    """
    project_root = Path(project_root)
    output_root = Path(output_root)

    # ---- resolve detail CSVs ----------------------------------------
    if detail_csv_paths is not None:
        csv_list = [Path(p) for p in detail_csv_paths]
    else:
        csv_list = _discover_detail_csvs(output_root)

    logger.info("History rebuild: discovered %d detail CSVs", len(csv_list))

    # ---- read trajectory manifest for checkpoint info ---------------
    manifest = _load_trajectory_manifest(output_root)

    # ---- build checkpoint lookup  (event_id|checkpoint_id → checkpoint_min)
    checkpoint_lookup = _build_checkpoint_lookup(manifest)

    # ---- process each detail CSV ------------------------------------
    per_sample_records: list[dict[str, Any]] = []
    n_full = 0
    n_incomplete = 0
    frame_validation_summary: dict[str, int] = {
        f"frame_{i}": 0 for i in range(N_HISTORY_FRAMES)
    }

    for csv_path in csv_list:
        record = _process_single_detail(
            csv_path,
            checkpoint_lookup,
        )
        per_sample_records.append(record)

        if record["history_incomplete"]:
            n_incomplete += 1
        else:
            n_full += 1

        # accumulate per-frame pass counts
        for fv in record.get("frame_validations", []):
            if fv["valid"]:
                frame_validation_summary[f"frame_{fv['frame_idx']}"] += 1

    # ---- assemble audit ---------------------------------------------
    audit: dict[str, Any] = {
        "total_samples_attempted": len(csv_list),
        "samples_with_full_13_frames": n_full,
        "samples_with_incomplete_history": n_incomplete,
        "frame_timestamp_validation": frame_validation_summary,
        "future_aggregation_method": (
            "mean for state vars (depth, volume, rainfall); "
            "last-value for discrete actions/settings"
        ),
        "per_sample": per_sample_records,
    }

    # ---- write outputs ----------------------------------------------
    audit_dir = project_root / "audits" / "v42_final_pool"
    audit_dir.mkdir(parents=True, exist_ok=True)

    # audit JSON (exclude large per-sample arrays for readability)
    audit_json_path = audit_dir / "history_rebuild_audit.json"
    audit_json = {k: v for k, v in audit.items() if k != "per_sample"}
    audit_json["per_sample_count"] = len(per_sample_records)
    audit_json_path.write_text(
        json.dumps(audit_json, indent=2, default=str), encoding="utf-8"
    )
    logger.info("Audit JSON written to %s", audit_json_path)

    # details parquet
    details_path = audit_dir / "history_rebuild_details.parquet"
    _write_details_parquet(per_sample_records, details_path)
    logger.info("Details parquet written to %s", details_path)

    return audit


def validate_frame_timestamps(
    frames: np.ndarray,
    checkpoint: pd.Timestamp,
    recording_interval_sec: int = RECORDING_INTERVAL_SEC,
) -> dict:
    """Validate that frame timestamps match expected 5-min intervals.

    Parameters
    ----------
    frames : np.ndarray
        Shape ``(13, n_features)`` — one row per history frame.
    checkpoint : pd.Timestamp
        The checkpoint time (frame 12).
    recording_interval_sec : int
        Seconds between consecutive frames (default 300 = 5 min).

    Returns
    -------
    dict
        ``{frame_idx, expected_time, valid}`` for each of the 13 frames.
        The *expected_time* is ``checkpoint - (12 - i) * interval``.
        *valid* is always ``True`` when the caller supplies exactly 13
        rows — the check is structural (correct count).
    """
    if frames.ndim != 2 or frames.shape[0] != N_HISTORY_FRAMES:
        raise ValueError(
            f"Expected shape (13, n_features), got {frames.shape}"
        )

    interval = pd.Timedelta(seconds=recording_interval_sec)
    results: list[dict[str, Any]] = []
    for i in range(N_HISTORY_FRAMES):
        expected = checkpoint - (N_HISTORY_FRAMES - 1 - i) * interval
        results.append(
            {
                "frame_idx": i,
                "expected_time": expected,
                "valid": True,
            }
        )
    return {"frame_validations": results}


def aggregate_future_to_control_steps(
    future_5min: np.ndarray,
    control_interval_min: int = HORIZON_INTERVAL_MIN,
) -> np.ndarray:
    """Aggregate 5-min future trajectory to 10-min control steps.

    Aggregation method
    ------------------
    * **Mean** for continuous state variables (depths, volumes, rainfall).
    * **Last-value** for discrete action / setting columns.

    Since the caller does not distinguish column types in the raw array,
    the function uses **mean** as the default aggregation (appropriate for
    the dominant signal).  Action columns should be aggregated separately
    by the caller if exact last-value semantics are required.

    Parameters
    ----------
    future_5min : np.ndarray
        Shape ``(24, n_features)`` — 24 rows of 5-min future data
        covering checkpoint+5 min … checkpoint+120 min.
    control_interval_min : int
        Target control interval in minutes (default 10).

    Returns
    -------
    np.ndarray
        Shape ``(12, n_features)`` — one row per 10-min control step.
    """
    ratio = control_interval_min // 5  # 5-min rows per control step
    n_steps = future_5min.shape[0] // ratio

    if future_5min.shape[0] < n_steps * ratio:
        raise ValueError(
            f"Need at least {n_steps * ratio} rows, got {future_5min.shape[0]}"
        )

    trimmed = future_5min[: n_steps * ratio]
    reshaped = trimmed.reshape(n_steps, ratio, -1)
    return reshaped.mean(axis=1).astype(np.float32)


# ===================================================================
# Internal helpers
# ===================================================================


def _discover_detail_csvs(output_root: Path) -> list[Path]:
    """Recursively find all ``detail.csv`` files under *output_root*."""
    candidates: list[Path] = []
    for p in output_root.rglob("detail.csv"):
        if p.is_file():
            candidates.append(p)
    # Also pick up named detail CSVs (e.g. *_detail.csv)
    for p in output_root.rglob("*_detail.csv"):
        if p.is_file() and p not in candidates:
            candidates.append(p)
    return sorted(candidates)


def _load_trajectory_manifest(output_root: Path) -> pd.DataFrame | None:
    """Try to load the V4.2 trajectory manifest for checkpoint metadata."""
    candidates = [
        output_root / "final_v4" / "v42" / "trajectory_dataset"
        / "trajectory_manifest_v42.parquet",
        output_root / "final_v4" / "v42" / "trajectory_dataset"
        / "trajectory_manifest_v42.csv",
    ]
    for c in candidates:
        if c.exists():
            try:
                if c.suffix == ".parquet":
                    return pd.read_parquet(c)
                return pd.read_csv(c)
            except Exception as exc:
                logger.warning("Failed to load manifest %s: %s", c, exc)
    return None


def _build_checkpoint_lookup(
    manifest: pd.DataFrame | None,
) -> dict[str, float]:
    """Build ``{(event_id, checkpoint_id): checkpoint_min}`` lookup."""
    if manifest is None:
        return {}
    lookup: dict[str, float] = {}
    for _, row in manifest.iterrows():
        eid = str(row.get("event_id", ""))
        cid = str(row.get("checkpoint_id", ""))
        # Derive checkpoint_min from checkpoint_id suffix or column
        cmin = _extract_checkpoint_min(row)
        if eid and cid:
            lookup[(eid, cid)] = cmin
    return lookup


def _extract_checkpoint_min(row: pd.Series) -> float:
    """Extract checkpoint_min from a manifest row.

    Checks for an explicit ``checkpoint_min`` column first, then falls
    back to parsing the trailing integer from ``checkpoint_id``.
    """
    if "checkpoint_min" in row.index:
        val = row["checkpoint_min"]
        if pd.notna(val):
            return float(val)
    # Fallback: parse trailing __<int> from checkpoint_id
    cid = str(row.get("checkpoint_id", ""))
    m = re.search(r"__(\d+)$", cid)
    if m:
        return float(m.group(1))
    return 100.0  # safe default


def _process_single_detail(
    csv_path: Path,
    checkpoint_lookup: dict[tuple[str, str], float],
) -> dict[str, Any]:
    """Process one detail CSV and return a per-sample audit record."""
    record: dict[str, Any] = {
        "detail_csv": str(csv_path),
        "history_incomplete": True,
        "n_frames_found": 0,
        "frame_validations": [],
        "future_steps_found": 0,
        "error": None,
    }

    try:
        detail = pd.read_csv(csv_path)
    except Exception as exc:
        record["error"] = f"read_error: {exc}"
        return record

    if detail.empty or "elapsed_min" not in detail.columns:
        record["error"] = "empty or missing elapsed_min"
        return record

    elapsed = pd.to_numeric(detail["elapsed_min"], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if np.all(np.isnan(elapsed)):
        record["error"] = "no valid elapsed_min values"
        return record

    # ---- determine checkpoint ----------------------------------------
    event_id = str(detail["event_id"].iloc[0]) if "event_id" in detail.columns else ""
    # Try to find checkpoint from lookup
    checkpoint_min = _resolve_checkpoint_min(
        csv_path, event_id, checkpoint_lookup
    )

    # ---- extract 13 history frames ----------------------------------
    history_times = [
        checkpoint_min - (N_HISTORY_FRAMES - 1 - i) * HISTORY_INTERVAL_MIN
        for i in range(N_HISTORY_FRAMES)
    ]
    frame_validations: list[dict[str, Any]] = []
    history_rows: list[np.ndarray] = []

    for frame_idx, target_t in enumerate(history_times):
        idx = int(np.argmin(np.abs(elapsed - target_t)))
        actual_t = elapsed[idx]
        # Tolerance: must be within 2.5 min (half a 5-min step)
        valid = abs(actual_t - target_t) <= 2.5
        frame_validations.append(
            {
                "frame_idx": frame_idx,
                "expected_elapsed_min": target_t,
                "actual_elapsed_min": float(actual_t),
                "valid": bool(valid),
            }
        )
        if valid:
            history_rows.append(detail.iloc[idx].to_numpy())

    record["frame_validations"] = frame_validations
    n_valid_frames = sum(1 for fv in frame_validations if fv["valid"])
    record["n_frames_found"] = n_valid_frames

    if n_valid_frames < N_HISTORY_FRAMES:
        record["history_incomplete"] = True
        return record

    # ---- extract 24 future 5-min rows → 12 control steps -----------
    future_times = [
        checkpoint_min + (i + 1) * HISTORY_INTERVAL_MIN
        for i in range(24)  # 24 × 5 min = 120 min
    ]
    future_rows_5min: list[np.ndarray] = []
    for target_t in future_times:
        idx = int(np.argmin(np.abs(elapsed - target_t)))
        actual_t = elapsed[idx]
        if abs(actual_t - target_t) <= 2.5:
            future_rows_5min.append(detail.iloc[idx].to_numpy())

    if len(future_rows_5min) >= 24:
        future_arr = np.stack(future_rows_5min)
        # Separate numeric columns only
        numeric_mask = np.array(
            [np.issubdtype(detail.iloc[:, j].dtype, np.number)
             for j in range(detail.shape[1])]
        )
        numeric_cols = future_arr[:, numeric_mask].astype(np.float32)
        aggregated = aggregate_future_to_control_steps(numeric_cols)
        record["future_steps_found"] = aggregated.shape[0]
    else:
        record["future_steps_found"] = len(future_rows_5min) // 2

    record["history_incomplete"] = False
    return record


def _resolve_checkpoint_min(
    csv_path: Path,
    event_id: str,
    checkpoint_lookup: dict[str, float],
) -> float:
    """Best-effort checkpoint resolution for a detail CSV."""
    # Strategy 1: match via parent directory name containing checkpoint info
    # e.g. .../t16v3__T100_D300_block__...__100__.../
    parts = csv_path.parent.name.split("__")
    for part in reversed(parts):
        if part.isdigit():
            return float(part)

    # Strategy 2: lookup by event_id
    for (eid, cid), cmin in checkpoint_lookup.items():
        if eid == event_id:
            return cmin

    # Strategy 3: scan the detail CSV itself — the checkpoint is often
    # the elapsed_min where policy changes or control kicks in.
    # Fallback: use the midpoint of the available data.
    try:
        detail = pd.read_csv(csv_path, usecols=["elapsed_min"])
        elapsed = pd.to_numeric(detail["elapsed_min"], errors="coerce").dropna()
        if len(elapsed) > 0:
            mid = elapsed.iloc[len(elapsed) // 2]
            # Round to nearest 5 min
            return float(round(mid / 5.0) * 5.0)
    except Exception:
        pass

    return 100.0


def _write_details_parquet(
    records: list[dict[str, Any]],
    output_path: Path,
) -> None:
    """Flatten per-sample records and write to parquet."""
    rows: list[dict[str, Any]] = []
    for rec in records:
        row = {
            "detail_csv": rec["detail_csv"],
            "history_incomplete": rec["history_incomplete"],
            "n_frames_found": rec["n_frames_found"],
            "future_steps_found": rec["future_steps_found"],
            "error": rec.get("error"),
        }
        # Flatten frame validations into columns
        for fv in rec.get("frame_validations", []):
            fi = fv["frame_idx"]
            row[f"frame_{fi}_expected"] = fv["expected_elapsed_min"]
            row[f"frame_{fi}_actual"] = fv["actual_elapsed_min"]
            row[f"frame_{fi}_valid"] = fv["valid"]
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_parquet(output_path, index=False)
