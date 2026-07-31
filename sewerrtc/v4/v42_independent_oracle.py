"""Independent PFV/TFV/Peak oracle for the canonical V4.2 pool.

Two audit modes are supported:

``stored``
    Recompute from the canonical manifest's stored flood-rate trajectories.
``raw``
    Resolve ``case_id`` back to local Train1600 run directories, read the
    original Candidate/No-control/Dynamic-Internal ``detail.csv`` files, select
    every raw post-checkpoint row through H120, align ``flood:<node_id>`` by
    physical node ID, integrate using the actual timestamps, and compare those
    independent values with the stored training labels.

The raw mode is the authoritative admission check.  It intentionally does not
call the production trajectory builder's KPI helper, so a builder/resampling
bug cannot validate itself.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sewerrtc.v4.v42_priority_contract import PFV_CORE_8_IDS

HORIZON_MIN = 120.0


def _parse_array(value: Any) -> np.ndarray:
    if isinstance(value, str):
        value = json.loads(value)
    arr = np.asarray(value, dtype=np.float64)
    if not np.isfinite(arr).all():
        raise ValueError("Array contains NaN/Inf")
    return arr


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
    return (
        float(np.sum(priority_rate * dt_seconds)),
        float(np.sum(total_rate * dt_seconds)),
        float(np.max(total_rate)),
    )


def _stored_dt_seconds(row: pd.Series, n_steps: int) -> np.ndarray:
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


def recompute_row(row: pd.Series, priority_indices: list[int]) -> dict[str, float]:
    """Recompute from canonical manifest trajectories (secondary audit mode)."""
    required = (
        "trajectory_flood_candidate",
        "trajectory_flood_no_control",
        "trajectory_flood_dynamic_internal",
        "future_elapsed_min",
        "checkpoint_min",
    )
    missing = [k for k in required if k not in row.index or pd.isna(row[k])]
    if missing:
        raise KeyError(f"Row is missing stored-oracle inputs: {missing}")
    cand = _parse_array(row["trajectory_flood_candidate"])
    nc = _parse_array(row["trajectory_flood_no_control"])
    di = _parse_array(row["trajectory_flood_dynamic_internal"])
    if cand.shape != nc.shape or cand.shape != di.shape:
        raise ValueError(
            f"Candidate/NC/DI flood shapes differ: {cand.shape}, {nc.shape}, {di.shape}"
        )
    dt = _stored_dt_seconds(row, cand.shape[0])
    pfv_c, tfv_c, peak_c = _branch_kpis(cand, dt, priority_indices)
    pfv_nc, _, _ = _branch_kpis(nc, dt, priority_indices)
    _, tfv_di, peak_di = _branch_kpis(di, dt, priority_indices)
    return _format_kpis(pfv_c, pfv_nc, tfv_c, tfv_di, peak_c, peak_di)


def _format_kpis(pfv_c, pfv_nc, tfv_c, tfv_di, peak_c, peak_di) -> dict[str, float]:
    return {
        "recomputed_pfv_candidate_m3": float(pfv_c),
        "recomputed_pfv_no_control_m3": float(pfv_nc),
        "recomputed_tfv_candidate_m3": float(tfv_c),
        "recomputed_tfv_dynamic_internal_m3": float(tfv_di),
        "recomputed_peak_candidate_m3s": float(peak_c),
        "recomputed_peak_dynamic_internal_m3s": float(peak_di),
        "recomputed_pfv_delta_m3": float(pfv_c - pfv_nc),
        "recomputed_tfv_delta_m3": float(tfv_c - tfv_di),
        # Formal definition: max(C) - max(DI), never max(C-DI).
        "recomputed_peak_delta_m3s": float(peak_c - peak_di),
    }


def _load_graph_node_ids(project_root: Path) -> list[str]:
    # Only topology/order is reused; KPI integration remains independent.
    from sewerrtc.v4.v42_trajectory_builder import _load_graph_topology

    return list(_load_graph_topology(project_root)["node_ids"])


def _priority_indices(node_ids: list[str]) -> list[int]:
    lookup = {str(x): i for i, x in enumerate(node_ids)}
    missing = [x for x in PFV_CORE_8_IDS if x not in lookup]
    if missing:
        raise ValueError(f"PFV_CORE8 nodes missing from graph order: {missing}")
    return [lookup[x] for x in PFV_CORE_8_IDS]


def _raw_flood_matrix(detail: pd.DataFrame, node_ids: list[str]) -> np.ndarray:
    """Independently align raw ``flood:<node_id>`` columns to graph node IDs."""
    raw: dict[str, str] = {}
    for column in detail.columns:
        text = str(column)
        if not text.startswith("flood:"):
            continue
        node = text[len("flood:"):]
        key = node.casefold()
        if key in raw:
            raise ValueError(f"Duplicate flood-rate column for node {node!r}")
        raw[key] = text
    missing = [node for node in node_ids if node.casefold() not in raw]
    if missing:
        raise ValueError(f"Raw detail missing {len(missing)} flood-rate nodes: {missing[:10]}")
    columns = [raw[node.casefold()] for node in node_ids]
    frame = detail[columns].apply(pd.to_numeric, errors="coerce")
    if frame.isna().any().any():
        raise ValueError("Raw flood-rate columns contain NaN/non-numeric values")
    return frame.to_numpy(dtype=np.float64)


def _raw_branch_kpis(
    detail_path: Path,
    checkpoint_min: float,
    node_ids: list[str],
    priority_indices: list[int],
) -> tuple[float, float, float, int]:
    detail = pd.read_csv(detail_path)
    if "elapsed_min" not in detail.columns:
        raise KeyError(f"{detail_path} is missing elapsed_min")
    elapsed = pd.to_numeric(detail["elapsed_min"], errors="coerce").to_numpy(float)
    if not np.isfinite(elapsed).all():
        raise ValueError(f"{detail_path}: elapsed_min contains NaN/Inf")
    mask = (elapsed > checkpoint_min + 1e-9) & (
        elapsed <= checkpoint_min + HORIZON_MIN + 1e-9
    )
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        raise ValueError(f"{detail_path}: no raw rows in H120")
    times = elapsed[indices]
    order = np.argsort(times)
    indices = indices[order]
    times = times[order]
    if np.any(np.diff(times) <= 0):
        raise ValueError(f"{detail_path}: duplicate/non-monotonic H120 timestamps")
    if abs(times[-1] - (checkpoint_min + HORIZON_MIN)) > 1e-6:
        raise ValueError(
            f"{detail_path}: H120 ends at {times[-1]}, expected {checkpoint_min + HORIZON_MIN}"
        )
    dt = np.diff(np.concatenate([[checkpoint_min], times])) * 60.0
    flood = _raw_flood_matrix(detail, node_ids)[indices]
    pfv, tfv, peak = _branch_kpis(flood, dt, priority_indices)
    return pfv, tfv, peak, int(len(indices))


def _scan_case_dirs(output_root: Path) -> dict[str, Path]:
    """Index local run directories using completion.json case IDs."""
    output_root = Path(output_root)
    case_map: dict[str, Path] = {}
    roots = [output_root / "train1600_v3", output_root / "train1600"]
    for root in roots:
        if not root.exists():
            continue
        for completion in root.glob("round*/runs/*/completion.json"):
            try:
                payload = json.loads(completion.read_text(encoding="utf-8"))
                case_id = str(payload.get("case_id", ""))
                if case_id:
                    case_map[case_id] = completion.parent
            except Exception:
                continue
    return case_map


def _branch_detail_paths(run_dir: Path) -> dict[str, Path]:
    completion = json.loads((run_dir / "completion.json").read_text(encoding="utf-8"))
    branches = completion.get("branches", {})
    result: dict[str, Path] = {}
    for role in ("candidate", "no_control", "dynamic_internal_rules"):
        text = str(branches.get(role, {}).get("detail_path", ""))
        if not text:
            raise KeyError(f"{run_dir}: branch {role} has no detail_path")
        path = Path(text)
        if not path.exists():
            # Some archived outputs store absolute Windows paths.  If the file
            # has moved with its run directory, try the basename locally.
            local = run_dir / path.name
            if local.exists():
                path = local
            else:
                raise FileNotFoundError(f"{role} detail not found: {text}")
        result[role] = path
    return result


def recompute_row_from_raw(
    row: pd.Series,
    case_map: dict[str, Path],
    node_ids: list[str],
    priority_indices: list[int],
) -> dict[str, float]:
    case_id = str(row.get("case_id", ""))
    if not case_id or case_id not in case_map:
        raise KeyError(f"Raw run not found for case_id={case_id!r}")
    checkpoint = float(row["checkpoint_min"])
    paths = _branch_detail_paths(case_map[case_id])
    pfv_c, tfv_c, peak_c, n_c = _raw_branch_kpis(
        paths["candidate"], checkpoint, node_ids, priority_indices
    )
    pfv_nc, _, _, n_nc = _raw_branch_kpis(
        paths["no_control"], checkpoint, node_ids, priority_indices
    )
    _, tfv_di, peak_di, n_di = _raw_branch_kpis(
        paths["dynamic_internal_rules"], checkpoint, node_ids, priority_indices
    )
    result = _format_kpis(pfv_c, pfv_nc, tfv_c, tfv_di, peak_c, peak_di)
    result.update(
        {
            "raw_candidate_rows": n_c,
            "raw_no_control_rows": n_nc,
            "raw_dynamic_internal_rows": n_di,
        }
    )
    return result


def _stored_labels(row: pd.Series) -> tuple[float, float, float]:
    return float(row["pfv_delta"]), float(row["tfv_delta"]), float(row["peak_delta"])


def audit_manifest(
    project_root: Path,
    manifest_path: Path,
    *,
    expected_count: int | None = None,
    output_root: Path | None = None,
    raw: bool = False,
    atol_pfv_m3: float = 1e-3,
    atol_tfv_m3: float = 1e-3,
    atol_peak_m3s: float = 1e-6,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    project_root = Path(project_root)
    manifest_path = Path(manifest_path)
    df = pd.read_parquet(manifest_path) if manifest_path.suffix.lower() == ".parquet" else pd.read_csv(manifest_path)
    if expected_count is not None and len(df) != expected_count:
        raise ValueError(f"Expected {expected_count} rows, found {len(df)}")

    node_ids = _load_graph_node_ids(project_root)
    priority_indices = _priority_indices(node_ids)
    case_map = _scan_case_dirs(Path(output_root)) if raw and output_root is not None else {}
    if raw and not case_map:
        raise FileNotFoundError(
            "Raw Independent Oracle requested but no local Train1600 run directories were found"
        )

    rows: list[dict[str, Any]] = []
    failures = 0
    for idx, row in df.iterrows():
        rec: dict[str, Any] = {
            "row_index": int(idx),
            "event_id": str(row.get("event_id", "")),
            "checkpoint_id": str(row.get("checkpoint_id", "")),
            "case_id": str(row.get("case_id", "")),
            "audit_mode": "raw" if raw else "stored",
        }
        try:
            if raw:
                rec.update(recompute_row_from_raw(row, case_map, node_ids, priority_indices))
            else:
                rec.update(recompute_row(row, priority_indices))
            stored_pfv, stored_tfv, stored_peak = _stored_labels(row)
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
        failures += int(not rec["row_pass"])
        rows.append(rec)

    audit = pd.DataFrame(rows)
    summary = {
        "manifest": str(manifest_path),
        "audit_mode": "raw" if raw else "stored",
        "row_count": int(len(df)),
        "priority_node_count": len(priority_indices),
        "pass_count": int(len(df) - failures),
        "fail_count": int(failures),
        "all_pass": failures == 0,
        "expected_count": expected_count,
        "raw_run_case_count": int(len(case_map)),
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
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--raw", action="store_true", help="Audit from original 5-min SWMM detail files")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    if args.raw and args.output_root is None:
        parser.error("--raw requires --output-root")

    audit, summary = audit_manifest(
        args.project_root,
        args.manifest,
        expected_count=args.expected_count,
        output_root=args.output_root,
        raw=args.raw,
    )
    output_dir = args.output_dir or args.manifest.parent / "independent_oracle"
    output_dir.mkdir(parents=True, exist_ok=True)
    suffix = "raw" if args.raw else "stored"
    audit.to_csv(output_dir / f"independent_oracle_rows_{suffix}.csv", index=False)
    (output_dir / f"independent_oracle_summary_{suffix}.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))
    return 0 if summary["all_pass"] else 5


if __name__ == "__main__":
    raise SystemExit(main())
