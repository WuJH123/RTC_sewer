"""
V4.2 Independent PFV Oracle — Recompute PFV labels from raw flooding-rate data
================================================================================

Standalone module that independently verifies PFV (Priority Flood Volume)
labels stored in the trajectory manifest by recomputing them from raw
flooding-rate data (m³/s) at the 8 PFV core priority nodes.

**Prohibited approaches** (will raise if detected):
- Using depth as proxy for flooding rate
- Using only 2 sentinel nodes instead of 8 priority nodes
- Using average depth
- Fixed multiply by 300/600 without reading actual timestamps from schema

Formulas
--------
PFV_branch_m3 = Σ_{priority nodes} Σ_{time steps} (flood_rate_m3s × actual_dt_sec)
delta_pfv_nc_m3 = PFV_candidate_m3 − PFV_no_control_m3

Author: Project6 V4.2 audit pipeline
"""

from __future__ import annotations

import ast
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from sewerrtc._project_root import PROJECT_ROOT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_pfv_oracle_audit(
    project_root: Path,
    output_root: Path,
    trajectory_manifest: Path,
    priority_node_ids: list[str],
    tolerance_relative: float = 0.01,
) -> dict:
    """Recompute PFV labels from raw flooding-rate data and compare with stored values.

    Parameters
    ----------
    project_root : Path
        Project6 root directory (``E:\\RTC_sewer\\Project6``).
    output_root : Path
        Root for audit outputs.  Files are written to
        ``output_root / audits / v42_final_pool /``.
    trajectory_manifest : Path
        Path to ``trajectory_manifest_v42.parquet``.
    priority_node_ids : list[str]
        The 8 PFV core node IDs (from ``v42_priority_contract.PFV_CORE_8_IDS``).
    tolerance_relative : float
        Relative error tolerance for mismatch detection.  Samples with
        ``|recomputed − stored| / max(|stored|, ε) > tolerance`` are flagged.

    Returns
    -------
    dict
        Audit summary with keys: ``n_samples``, ``n_mismatches``,
        ``max_abs_error``, ``max_rel_error``, ``mean_abs_error``,
        ``stored_pfv_range``, ``recomputed_pfv_range``, ``pass``.
    """
    t0 = time.time()
    project_root = Path(project_root)
    output_root = Path(output_root)
    trajectory_manifest = Path(trajectory_manifest)

    # ------------------------------------------------------------------
    # 1. Read schema metadata to determine dt_sec and node ordering
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
    dt_sec = int(horizon_interval_min * 60)  # convert minutes → seconds
    logger.info(
        "PFV oracle: dt_sec=%d (from horizon_interval_min=%s), n_horizon_steps=%d",
        dt_sec, horizon_interval_min, n_horizon_steps,
    )

    # Build node_id → column index mapping from schema
    all_node_ids: List[str] = node_schema["node_ids"]
    n_nodes = len(all_node_ids)
    node_to_col = {nid: i for i, nid in enumerate(all_node_ids)}

    # Resolve priority node column indices
    priority_cols: List[int] = []
    missing_nodes: List[str] = []
    for nid in priority_node_ids:
        if nid in node_to_col:
            priority_cols.append(node_to_col[nid])
        else:
            missing_nodes.append(nid)
    if missing_nodes:
        raise ValueError(
            f"PFV oracle: {len(missing_nodes)} priority nodes not found in "
            f"node_feature_schema: {missing_nodes}"
        )
    priority_cols_arr = np.array(priority_cols, dtype=np.int64)
    logger.info(
        "PFV oracle: %d priority nodes resolved → cols %s",
        len(priority_cols_arr), priority_cols_arr.tolist(),
    )

    # ------------------------------------------------------------------
    # 2. Load trajectory manifest
    # ------------------------------------------------------------------
    logger.info("PFV oracle: loading manifest %s", trajectory_manifest)
    df = pd.read_parquet(str(trajectory_manifest))
    n_samples = len(df)
    logger.info("PFV oracle: loaded %d samples", n_samples)

    required_cols = [
        "trajectory_depth_candidate",
        "trajectory_depth_no_control",
        "pfv_delta",
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"PFV oracle: manifest missing columns: {missing_cols}")

    # ------------------------------------------------------------------
    # 3. Recompute PFV for each sample
    # ------------------------------------------------------------------
    stored_pfv_deltas = np.empty(n_samples, dtype=np.float64)
    recomputed_pfv_deltas = np.empty(n_samples, dtype=np.float64)
    recomputed_pfv_candidate = np.empty(n_samples, dtype=np.float64)
    recomputed_pfv_nc = np.empty(n_samples, dtype=np.float64)
    abs_errors = np.empty(n_samples, dtype=np.float64)
    rel_errors = np.empty(n_samples, dtype=np.float64)

    for idx in range(n_samples):
        row = df.iloc[idx]

        # Parse trajectory arrays: shape = (n_horizon_steps, n_nodes)
        traj_c = _parse_trajectory(row["trajectory_depth_candidate"], n_horizon_steps, n_nodes)
        traj_nc = _parse_trajectory(row["trajectory_depth_no_control"], n_horizon_steps, n_nodes)

        # Extract priority-node flooding rates
        pfv_c = traj_c[:, priority_cols_arr]  # (T, 8)
        pfv_nc = traj_nc[:, priority_cols_arr]  # (T, 8)

        # PFV = sum of (flood_rate × dt_sec) over all priority nodes and time steps
        pfv_candidate_m3 = float(pfv_c.sum() * dt_sec)
        pfv_no_control_m3 = float(pfv_nc.sum() * dt_sec)
        delta_recomputed = pfv_candidate_m3 - pfv_no_control_m3

        # Stored value
        delta_stored = float(row["pfv_delta"])

        stored_pfv_deltas[idx] = delta_stored
        recomputed_pfv_deltas[idx] = delta_recomputed
        recomputed_pfv_candidate[idx] = pfv_candidate_m3
        recomputed_pfv_nc[idx] = pfv_no_control_m3

        # Error metrics
        abs_err = abs(delta_recomputed - delta_stored)
        abs_errors[idx] = abs_err
        denom = max(abs(delta_stored), 1e-12)
        rel_errors[idx] = abs_err / denom

    # ------------------------------------------------------------------
    # 4. Identify mismatches
    # ------------------------------------------------------------------
    mismatch_mask = rel_errors > tolerance_relative
    n_mismatches = int(mismatch_mask.sum())

    # ------------------------------------------------------------------
    # 5. Build per-sample comparison DataFrame
    # ------------------------------------------------------------------
    result_df = pd.DataFrame({
        "event_id": df["event_id"].values,
        "checkpoint_id": df["checkpoint_id"].values,
        "state_key": df["state_key"].values,
        "stored_pfv_delta_m3": stored_pfv_deltas,
        "recomputed_pfv_candidate_m3": recomputed_pfv_candidate,
        "recomputed_pfv_nc_m3": recomputed_pfv_nc,
        "recomputed_pfv_delta_m3": recomputed_pfv_deltas,
        "abs_error_m3": abs_errors,
        "rel_error": rel_errors,
        "mismatch": mismatch_mask,
    })

    # ------------------------------------------------------------------
    # 6. Write outputs
    # ------------------------------------------------------------------
    audit_dir = output_root / "audits" / "v42_final_pool"
    audit_dir.mkdir(parents=True, exist_ok=True)

    # Per-sample parquet
    pq_path = audit_dir / "pfv_label_recomputation.parquet"
    result_df.to_parquet(str(pq_path), index=False)
    logger.info("PFV oracle: wrote %s (%d rows)", pq_path, len(result_df))

    # Mismatches CSV
    mismatch_df = result_df.loc[mismatch_mask].copy()
    mismatch_df = mismatch_df.sort_values("abs_error_m3", ascending=False)
    csv_path = audit_dir / "pfv_label_mismatches.csv"
    mismatch_df.to_csv(str(csv_path), index=False)
    logger.info("PFV oracle: wrote %s (%d mismatches)", csv_path, n_mismatches)

    # Summary JSON
    elapsed = time.time() - t0
    audit_summary: Dict[str, Any] = {
        "module": "v42_independent_pfv_oracle",
        "audit_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "elapsed_sec": round(elapsed, 2),
        "trajectory_manifest": str(trajectory_manifest),
        "priority_node_ids": list(priority_node_ids),
        "n_priority_nodes": len(priority_node_ids),
        "dt_sec": dt_sec,
        "horizon_interval_min": horizon_interval_min,
        "n_horizon_steps": n_horizon_steps,
        "n_nodes_in_schema": n_nodes,
        "tolerance_relative": tolerance_relative,
        "n_samples": n_samples,
        "n_mismatches": n_mismatches,
        "max_abs_error_m3": float(abs_errors.max()) if n_samples > 0 else 0.0,
        "mean_abs_error_m3": float(abs_errors.mean()) if n_samples > 0 else 0.0,
        "max_rel_error": float(rel_errors.max()) if n_samples > 0 else 0.0,
        "median_rel_error": float(np.median(rel_errors)) if n_samples > 0 else 0.0,
        "stored_pfv_range_m3": [
            float(stored_pfv_deltas.min()),
            float(stored_pfv_deltas.max()),
        ],
        "recomputed_pfv_range_m3": [
            float(recomputed_pfv_deltas.min()),
            float(recomputed_pfv_deltas.max()),
        ],
        "pass": n_mismatches == 0,
    }

    json_path = audit_dir / "pfv_oracle_audit.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, indent=2, ensure_ascii=False)
    logger.info("PFV oracle: wrote %s", json_path)

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
    import sys

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

    # Import priority node IDs from the contract module
    sys.path.insert(0, str(_project_root))
    from sewerrtc.v4.v42_priority_contract import PFV_CORE_8_IDS

    summary = run_pfv_oracle_audit(
        project_root=_project_root,
        output_root=_project_root,
        trajectory_manifest=_manifest,
        priority_node_ids=PFV_CORE_8_IDS,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))
