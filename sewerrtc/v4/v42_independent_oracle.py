"""Independent KPI oracle for the canonical V4.2 trajectory pool.

This module intentionally does **not** call the production dataset KPI helper.
It recomputes PFV/TFV/Peak from stored flood-rate trajectories and timestamps so
that a bug in dataset construction cannot silently validate itself.

Usage (on the local Project6 machine after rebuilding the trajectory dataset)::

    python -m sewerrtc.v4.v42_independent_oracle \
      --project-root E:/RTC_sewer/Project6 \
      --manifest outputs/project6_dual_reference_v4/final_v4/v42/trajectory_dataset/trajectory_manifest_v42.parquet \
      --expected-count 1200
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sewerrtc.v4.v42_priority_contract import get_pfv_core_node_indices


def _parse_array(value: Any) -> np.ndarray:
    if isinstance(value, str):
        value = json.loads(value)
    arr = np.asarray(value, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ValueError("Array contains NaN/Inf")
    return arr


def _dt_seconds(row: pd.Series, n_steps: int) -> np.ndarray:
    times = _parse_array(row["future_elapsed_min"])
    if times.ndim != 1 or len(times) != n_steps:
        raise ValueError(
            f"future_elapsed_min must contain {n_steps} timestamps, got {times.shape}"
        )
    checkpoint = float(row["checkpoint_min"])
    dt = np.diff(np.concatenate([[checkpoint], times])) * 60.0
    if np.any(dt <= 0):
        raise ValueError("Future timestamps are not strictly increasing")
    return dt


def _branch_kpis(
    flood_rate: np.ndarray,
    dt_seconds: np.ndarray,
    priority_indices: list[int],
) -> tuple[float, float, float]:
    if flood_rate.ndim != 2:
        raise ValueError("flood trajectory must have shape [H,N]")
    if flood_rate.shape[0] != len(dt_seconds):
        raise ValueError("flood trajectory and timestamp lengths differ")
    if any(i < 0 or i >= flood_rate.shape[1] for i in priority_indices):
        raise ValueError("Priority index outside flood trajectory")
    total_rate = flood_rate.sum(axis=1)
    priority_rate = flood_rate[:, priority_indices].sum(axis=1)
    pfv = float(np.sum(priority_rate * dt_seconds))
    tfv = float(np.sum(total_rate * dt_seconds))
    peak = float(np.max(total_rate))
    return pfv, tfv, peak


def recompute_row(row: pd.Series, priority_indices: list[int]) -> dict[str, float]:
    required = (
        "trajectory_flood_candidate",
        "trajectory_flood_no_control",
        "trajectory_flood_dynamic_internal",
        "future_elapsed_min",
        "checkpoint_min",
    )
    missing = [k for k in required if k not in row.index or pd.isna(row[k])]
    if missing:
        raise KeyError(f"Row is missing independent-oracle inputs: {missing}")

    cand = _parse_array(row["trajectory_flood_candidate"])
    nc = _parse_array(row["trajectory_flood_no_control"])
    di = _parse_array(row["trajectory_flood_dynamic_internal"])
    if cand.shape != nc.shape or cand.shape != di.shape:
        raise ValueError(
            f"Candidate/NC/DI flood shapes differ: {cand.shape}, {nc.shape}, {di.shape}"
        )
    dt = _dt_seconds(row, cand.shape[0])
    pfv_c, tfv_c, peak_c = _branch_kpis(cand, dt, priority_indices)
    pfv_nc, _, _ = _branch_kpis(nc, dt, priority_indices)
    _, tfv_di, peak_di = _branch_kpis(di, dt, priority_indices)
    return {
        "recomputed_pfv_candidate_m3": pfv_c,
        "recomputed_pfv_no_control_m3": pfv_nc,
        "recomputed_tfv_candidate_m3": tfv_c,
        "recomputed_tfv_dynamic_internal_m3": tfv_di,
        "recomputed_peak_candidate_m3s": peak_c,
        "recomputed_peak_dynamic_internal_m3s": peak_di,
        "recomputed_pfv_delta_m3": pfv_c - pfv_nc,
        "recomputed_tfv_delta_m3": tfv_c - tfv_di,
        # Correct definition: max(C) - max(DI), not max(C-DI).
        "recomputed_peak_delta_m3s": peak_c - peak_di,
    }


def audit_manifest(
    project_root: Path,
    manifest_path: Path,
    *,
    expected_count: int | None = None,
    atol_pfv_m3: float = 1e-3,
    atol_tfv_m3: float = 1e-3,
    atol_peak_m3s: float = 1e-6,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    project_root = Path(project_root)
    manifest_path = Path(manifest_path)
    if manifest_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(manifest_path)
    else:
        df = pd.read_csv(manifest_path)
    if expected_count is not None and len(df) != expected_count:
        raise ValueError(f"Expected {expected_count} rows, found {len(df)}")

    # Resolve PFV_CORE8 against the exact graph node order used by the builder.
    from sewerrtc.v4.v42_trajectory_builder import _load_graph_topology

    graph = _load_graph_topology(project_root)
    priority_indices = get_pfv_core_node_indices(list(graph["node_ids"]))

    rows: list[dict[str, Any]] = []
    failures = 0
    for idx, row in df.iterrows():
        rec: dict[str, Any] = {
            "row_index": int(idx),
            "event_id": str(row.get("event_id", "")),
            "checkpoint_id": str(row.get("checkpoint_id", "")),
            "case_id": str(row.get("case_id", "")),
        }
        try:
            rec.update(recompute_row(row, priority_indices))
            stored_pfv = float(row["pfv_delta"])
            stored_tfv = float(row["tfv_delta"])
            stored_peak = float(row["peak_delta"])
            rec.update(
                {
                    "stored_pfv_delta_m3": stored_pfv,
                    "stored_tfv_delta_m3": stored_tfv,
                    "stored_peak_delta_m3s": stored_peak,
                    "pfv_abs_error_m3": abs(stored_pfv - rec["recomputed_pfv_delta_m3"]),
                    "tfv_abs_error_m3": abs(stored_tfv - rec["recomputed_tfv_delta_m3"]),
                    "peak_abs_error_m3s": abs(stored_peak - rec["recomputed_peak_delta_m3s"]),
                }
            )
            rec["pfv_pass"] = rec["pfv_abs_error_m3"] <= atol_pfv_m3
            rec["tfv_pass"] = rec["tfv_abs_error_m3"] <= atol_tfv_m3
            rec["peak_pass"] = rec["peak_abs_error_m3s"] <= atol_peak_m3s
            rec["row_pass"] = bool(rec["pfv_pass"] and rec["tfv_pass"] and rec["peak_pass"])
        except Exception as exc:
            rec["row_pass"] = False
            rec["error"] = f"{type(exc).__name__}: {exc}"
        if not rec["row_pass"]:
            failures += 1
        rows.append(rec)

    audit = pd.DataFrame(rows)
    summary = {
        "manifest": str(manifest_path),
        "row_count": int(len(df)),
        "priority_node_count": len(priority_indices),
        "pass_count": int(len(df) - failures),
        "fail_count": int(failures),
        "all_pass": failures == 0,
        "tolerances": {
            "pfv_m3": atol_pfv_m3,
            "tfv_m3": atol_tfv_m3,
            "peak_m3s": atol_peak_m3s,
        },
    }
    return audit, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    audit, summary = audit_manifest(
        args.project_root,
        args.manifest,
        expected_count=args.expected_count,
    )
    output_dir = args.output_dir or args.manifest.parent / "independent_oracle"
    output_dir.mkdir(parents=True, exist_ok=True)
    audit.to_csv(output_dir / "independent_oracle_rows.csv", index=False)
    (output_dir / "independent_oracle_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["all_pass"] else 5


if __name__ == "__main__":
    raise SystemExit(main())
