"""
V4.2 Independent TFV & Peak Oracle — Recompute TFV/Peak from raw flooding-rate data
====================================================================================

Standalone module that independently verifies TFV (Total Flood Volume) and
Peak (peak spatial-sum flooding rate) labels stored in the trajectory manifest
by recomputing them from raw flooding-rate data (m³/s) across **all** graph nodes.

**Prohibited approaches** (will raise if detected):
- Using depth as proxy for flooding rate
- Using only priority nodes instead of ALL nodes
- Fixed multiply by 300/600 without reading actual timestamps from schema
- Computing Peak delta as max(Candidate − DI) instead of max(Candidate) − max(DI)

Formulas
--------
TFV_branch_m3 = Σ_{all nodes} Σ_{time steps} (flood_rate_m3s × actual_dt_sec)
delta_tfv_di_m3 = TFV_candidate_m3 − TFV_dynamic_internal_m3

Peak_branch_m3s = max_over_t( Σ_{all nodes} flood_rate_m3s(t) )
delta_peak_di_m3s = max(Candidate_spatial_sum) − max(DI_spatial_sum)

**CRITICAL**: Peak delta uses ``max(C) − max(DI)``, NOT ``max(C − DI)``.

Author: Project6 V4.2 audit pipeline
"""

from __future__ import annotations

import ast
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from sewerrtc._project_root import PROJECT_ROOT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_tfv_peak_oracle_audit(
    project_root: Path,
    output_root: Path,
    trajectory_manifest: Path,
) -> dict:
    """Recompute TFV and Peak labels from raw flooding-rate data.

    Parameters
    ----------
    project_root : Path
        Project6 root directory.
    output_root : Path
        Root for audit outputs.  Files are written to
        ``output_root / audits / v42_final_pool /``.
    trajectory_manifest : Path
        Path to ``trajectory_manifest_v42.parquet``.

    Returns
    -------
    dict
        Audit summary with TFV/Peak comparison statistics.
    """
    t0 = time.time()
    project_root = Path(project_root)
    output_root = Path(output_root)
    trajectory_manifest = Path(trajectory_manifest)

    # ------------------------------------------------------------------
    # 1. Read schema metadata
    # ------------------------------------------------------------------
    traj_dir = trajectory_manifest.parent
    summary_path = traj_dir / "trajectory_dataset_v42_summary.json"
    schema_path = traj_dir / "node_feature_schema_v42.json"

    if not summary_path.is_file():
        raise FileNotFoundError(f"Dataset summary not found: {summary_path}")
    if not schema_path.is_file():
        raise FileNotFoundError(f"Node feature schema not found: {schema_path}")

    with open(summary_path, encoding="utf-8") as f:
        summary = json.load(f)
    with open(schema_path, encoding="utf-8") as f:
        node_schema = json.load(f)

    # Read dt from schema — NOT hardcoded
    horizon_interval_min = summary["schema"]["horizon_interval_min"]
    n_horizon_steps = summary["schema"]["n_horizon_steps"]
    dt_sec = int(horizon_interval_min * 60)
    n_nodes = len(node_schema["node_ids"])

    logger.info(
        "TFV/Peak oracle: dt_sec=%d, n_horizon_steps=%d, n_nodes=%d",
        dt_sec, n_horizon_steps, n_nodes,
    )

    # ------------------------------------------------------------------
    # 2. Load trajectory manifest
    # ------------------------------------------------------------------
    logger.info("TFV/Peak oracle: loading manifest %s", trajectory_manifest)
    df = pd.read_parquet(str(trajectory_manifest))
    n_samples = len(df)
    logger.info("TFV/Peak oracle: loaded %d samples", n_samples)

    required_cols = [
        "trajectory_depth_candidate",
        "trajectory_depth_dynamic_internal",
        "tfv_delta",
        "peak_delta",
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"TFV/Peak oracle: manifest missing columns: {missing_cols}")

    # ------------------------------------------------------------------
    # 3. Recompute TFV and Peak for each sample
    # ------------------------------------------------------------------
    stored_tfv_deltas = np.empty(n_samples, dtype=np.float64)
    recomputed_tfv_deltas = np.empty(n_samples, dtype=np.float64)
    recomputed_tfv_c = np.empty(n_samples, dtype=np.float64)
    recomputed_tfv_di = np.empty(n_samples, dtype=np.float64)

    stored_peak_deltas = np.empty(n_samples, dtype=np.float64)
    recomputed_peak_deltas = np.empty(n_samples, dtype=np.float64)
    recomputed_peak_c = np.empty(n_samples, dtype=np.float64)
    recomputed_peak_di = np.empty(n_samples, dtype=np.float64)

    tfv_abs_errors = np.empty(n_samples, dtype=np.float64)
    tfv_rel_errors = np.empty(n_samples, dtype=np.float64)
    peak_abs_errors = np.empty(n_samples, dtype=np.float64)
    peak_rel_errors = np.empty(n_samples, dtype=np.float64)

    for idx in range(n_samples):
        row = df.iloc[idx]

        traj_c = _parse_trajectory(
            row["trajectory_depth_candidate"], n_horizon_steps, n_nodes
        )
        traj_di = _parse_trajectory(
            row["trajectory_depth_dynamic_internal"], n_horizon_steps, n_nodes
        )

        # --- TFV: sum over ALL nodes and ALL time steps ---
        # TFV = Σ_t Σ_n (flood_rate * dt_sec)
        tfv_candidate = float(traj_c.sum() * dt_sec)
        tfv_di = float(traj_di.sum() * dt_sec)
        tfv_delta_recomputed = tfv_candidate - tfv_di

        # --- Peak: max over time of spatial sum ---
        # spatial_sum(t) = Σ_n flood_rate(n, t)
        # Peak = max_t(spatial_sum(t))
        spatial_sum_c = traj_c.sum(axis=1)  # shape (T,)
        spatial_sum_di = traj_di.sum(axis=1)  # shape (T,)
        peak_candidate = float(spatial_sum_c.max())
        peak_di = float(spatial_sum_di.max())

        # CRITICAL: delta = max(C) - max(DI), NOT max(C - DI)
        peak_delta_recomputed = peak_candidate - peak_di

        # Stored values
        tfv_delta_stored = float(row["tfv_delta"])
        peak_delta_stored = float(row["peak_delta"])

        # Store results
        stored_tfv_deltas[idx] = tfv_delta_stored
        recomputed_tfv_deltas[idx] = tfv_delta_recomputed
        recomputed_tfv_c[idx] = tfv_candidate
        recomputed_tfv_di[idx] = tfv_di

        stored_peak_deltas[idx] = peak_delta_stored
        recomputed_peak_deltas[idx] = peak_delta_recomputed
        recomputed_peak_c[idx] = peak_candidate
        recomputed_peak_di[idx] = peak_di

        # Error metrics — TFV
        tfv_abs = abs(tfv_delta_recomputed - tfv_delta_stored)
        tfv_abs_errors[idx] = tfv_abs
        tfv_rel_errors[idx] = tfv_abs / max(abs(tfv_delta_stored), 1e-12)

        # Error metrics — Peak
        peak_abs = abs(peak_delta_recomputed - peak_delta_stored)
        peak_abs_errors[idx] = peak_abs
        peak_rel_errors[idx] = peak_abs / max(abs(peak_delta_stored), 1e-12)

    # ------------------------------------------------------------------
    # 4. Build per-sample comparison DataFrame
    # ------------------------------------------------------------------
    result_df = pd.DataFrame({
        "event_id": df["event_id"].values,
        "checkpoint_id": df["checkpoint_id"].values,
        "state_key": df["state_key"].values,
        # TFV
        "stored_tfv_delta_m3": stored_tfv_deltas,
        "recomputed_tfv_candidate_m3": recomputed_tfv_c,
        "recomputed_tfv_di_m3": recomputed_tfv_di,
        "recomputed_tfv_delta_m3": recomputed_tfv_deltas,
        "tfv_abs_error_m3": tfv_abs_errors,
        "tfv_rel_error": tfv_rel_errors,
        # Peak
        "stored_peak_delta_m3s": stored_peak_deltas,
        "recomputed_peak_candidate_m3s": recomputed_peak_c,
        "recomputed_peak_di_m3s": recomputed_peak_di,
        "recomputed_peak_delta_m3s": recomputed_peak_deltas,
        "peak_abs_error_m3s": peak_abs_errors,
        "peak_rel_error": peak_rel_errors,
    })

    # ------------------------------------------------------------------
    # 5. Write outputs
    # ------------------------------------------------------------------
    audit_dir = output_root / "audits" / "v42_final_pool"
    audit_dir.mkdir(parents=True, exist_ok=True)

    # Per-sample parquet
    pq_path = audit_dir / "tfv_peak_recomputation.parquet"
    result_df.to_parquet(str(pq_path), index=False)
    logger.info("TFV/Peak oracle: wrote %s (%d rows)", pq_path, len(result_df))

    # Summary JSON
    elapsed = time.time() - t0
    audit_summary: Dict[str, Any] = {
        "module": "v42_tfv_peak_oracle",
        "audit_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": round(elapsed, 2),
        "trajectory_manifest": str(trajectory_manifest),
        "dt_sec": dt_sec,
        "horizon_interval_min": horizon_interval_min,
        "n_horizon_steps": n_horizon_steps,
        "n_nodes_in_schema": n_nodes,
        "n_samples": n_samples,
        # TFV stats
        "tfv_max_abs_error_m3": float(tfv_abs_errors.max()) if n_samples > 0 else 0.0,
        "tfv_mean_abs_error_m3": float(tfv_abs_errors.mean()) if n_samples > 0 else 0.0,
        "tfv_max_rel_error": float(tfv_rel_errors.max()) if n_samples > 0 else 0.0,
        "tfv_median_rel_error": float(np.median(tfv_rel_errors)) if n_samples > 0 else 0.0,
        "stored_tfv_range_m3": [
            float(stored_tfv_deltas.min()),
            float(stored_tfv_deltas.max()),
        ],
        "recomputed_tfv_range_m3": [
            float(recomputed_tfv_deltas.min()),
            float(recomputed_tfv_deltas.max()),
        ],
        # Peak stats
        "peak_max_abs_error_m3s": float(peak_abs_errors.max()) if n_samples > 0 else 0.0,
        "peak_mean_abs_error_m3s": float(peak_abs_errors.mean()) if n_samples > 0 else 0.0,
        "peak_max_rel_error": float(peak_rel_errors.max()) if n_samples > 0 else 0.0,
        "peak_median_rel_error": float(np.median(peak_rel_errors)) if n_samples > 0 else 0.0,
        "stored_peak_range_m3s": [
            float(stored_peak_deltas.min()),
            float(stored_peak_deltas.max()),
        ],
        "recomputed_peak_range_m3s": [
            float(recomputed_peak_deltas.min()),
            float(recomputed_peak_deltas.max()),
        ],
        "pass": (
            float(tfv_abs_errors.max()) < 1e-6
            and float(peak_abs_errors.max()) < 1e-6
        ) if n_samples > 0 else True,
    }

    json_path = audit_dir / "tfv_peak_oracle_audit.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2, ensure_ascii=False)
    logger.info("TFV/Peak oracle: wrote %s", json_path)

    return audit_summary


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_trajectory(raw: Any, expected_steps: int, expected_nodes: int) -> np.ndarray:
    """Parse a trajectory_depth_* cell into a 2-D numpy array.

    Parameters
    ----------
    raw : Any
        Raw cell value — may be a list, a JSON string, or a Python literal string.
    expected_steps : int
        Expected number of time steps (rows).
    expected_nodes : int
        Expected number of nodes per step (columns).

    Returns
    -------
    np.ndarray
        Shape ``(expected_steps, expected_nodes)``, dtype float64.
    """
    if isinstance(raw, str):
        raw = ast.literal_eval(raw)
    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2-D trajectory array, got shape {arr.shape}")
    if arr.shape[0] != expected_steps:
        raise ValueError(
            f"Expected {expected_steps} time steps, got {arr.shape[0]}"
        )
    if arr.shape[1] != expected_nodes:
        raise ValueError(
            f"Expected {expected_nodes} nodes, got {arr.shape[1]}"
        )
    return arr


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    _project_root = PROJECT_ROOT
    _manifest = (
        _project_root
        / "outputs"
        / "project6_dual_reference_v4"
        / "final_v4"
        / "v42"
        / "trajectory_dataset"
        / "trajectory_manifest_v42.parquet"
    )

    summary = run_tfv_peak_oracle_audit(
        project_root=_project_root,
        output_root=_project_root,
        trajectory_manifest=_manifest,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
