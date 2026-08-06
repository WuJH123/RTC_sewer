"""Read-only consistency gate for existing authoritative candidate details."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sewerrtc.control.authoritative_control_metrics_v42 import (
    causal_prefix_matches,
    detail_horizon_metrics,
    pfv_budget_metric,
    trajectory_metrics,
)
from sewerrtc.v4.v42_priority_contract import get_pfv_core_node_indices
from sewerrtc.v4.v42_trajectory_builder import _load_graph_topology


def _array(value: object) -> np.ndarray:
    if isinstance(value, str):
        value = json.loads(value)
    return np.asarray(value, dtype=float)


def _read_metric_detail(path: Path) -> pd.DataFrame:
    header = pd.read_csv(path, nrows=0).columns.tolist()
    usecols = [
        column for column in header
        if column == "elapsed_min" or column == "rainfall_mm_h" or str(column).startswith("flood:") or str(column).startswith("a:")
    ]
    return pd.read_csv(path, usecols=usecols)


def _compare(stored: dict[str, float], actual: dict[str, float], tolerance: float) -> dict[str, float | bool]:
    errors = {key: abs(float(stored[key]) - float(actual[key])) for key in ("PFV", "TFV", "peak_TFV_rate")}
    return {
        **{f"{key}_abs_error": float(value) for key, value in errors.items()},
        "metrics_match": bool(max(errors.values()) <= float(tolerance)),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--plan-csv", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--states", type=int, default=5)
    parser.add_argument("--candidates-per-state", type=int, default=3)
    parser.add_argument("--tolerance", type=float, default=1.0e-3)
    args = parser.parse_args()

    plan = pd.read_csv(args.plan_csv).head(int(args.states))
    columns = [
        "state_key", "event_id", "rainfall_sha256", "checkpoint_min",
        "candidate_action_sha256", "trajectory_flood_candidate",
        "trajectory_flood_no_control", "source_detail_path_candidate",
        "source_detail_path_no_control", "action_candidate_readback",
    ]
    manifest = pd.read_parquet(args.manifest, columns=columns)
    graph = _load_graph_topology(args.project_root)
    node_ids = [str(value) for value in graph["node_ids"]]
    priority_indices = get_pfv_core_node_indices(node_ids)
    priority_nodes = [node_ids[int(index)] for index in priority_indices]
    detail_cache: dict[str, pd.DataFrame] = {}
    records: list[dict[str, object]] = []

    for state in plan.itertuples(index=False):
        group = manifest[manifest["state_key"].astype(str).eq(str(state.state_key))].sort_values("candidate_action_sha256", kind="stable").head(int(args.candidates_per_state))
        if len(group) < int(args.candidates_per_state):
            raise RuntimeError(f"state {state.state_key} has fewer than {args.candidates_per_state} candidates")
        for _, row in group.iterrows():
            reference_path = Path(str(row["source_detail_path_no_control"]))
            if str(reference_path) not in detail_cache:
                detail_cache[str(reference_path)] = _read_metric_detail(reference_path)
            reference_detail = detail_cache[str(reference_path)]
            candidate_path = Path(str(row["source_detail_path_candidate"]))
            if str(candidate_path) not in detail_cache:
                detail_cache[str(candidate_path)] = _read_metric_detail(candidate_path)
            candidate_detail = detail_cache[str(candidate_path)]
            stored_candidate = trajectory_metrics(_array(row["trajectory_flood_candidate"]), priority_indices)
            stored_no_control = trajectory_metrics(_array(row["trajectory_flood_no_control"]), priority_indices)
            actual_candidate = detail_horizon_metrics(
                candidate_detail, priority_nodes, checkpoint_min=float(row["checkpoint_min"]), steps=12
            )
            actual_no_control = detail_horizon_metrics(
                reference_detail, priority_nodes, checkpoint_min=float(row["checkpoint_min"]), steps=12
            )
            prefix = causal_prefix_matches(
                candidate_detail, reference_detail, checkpoint_min=float(row["checkpoint_min"])
            )
            stored_budget = pfv_budget_metric(stored_candidate["PFV"], stored_no_control["PFV"], relative_margin=0.05)
            actual_budget = pfv_budget_metric(actual_candidate["PFV"], actual_no_control["PFV"], relative_margin=0.05)
            candidate_compare = _compare(stored_candidate, actual_candidate, args.tolerance)
            no_control_compare = _compare(stored_no_control, actual_no_control, args.tolerance)
            records.append({
                "state_key": str(row["state_key"]),
                "candidate_action_sha256": str(row["candidate_action_sha256"]),
                "candidate_detail": str(candidate_path),
                "stored_candidate": stored_candidate,
                "actual_candidate": actual_candidate,
                "stored_no_control": stored_no_control,
                "actual_no_control": actual_no_control,
                "stored_budget_metric_m3": float(stored_budget),
                "actual_budget_metric_m3": float(actual_budget),
                "budget_metric_abs_error_m3": abs(float(stored_budget) - float(actual_budget)),
                "candidate_compare": candidate_compare,
                "no_control_compare": no_control_compare,
                "prefix_audit": prefix,
                "admission_match": bool((stored_budget <= 100.0 + 1.0e-9) == (actual_budget <= 100.0 + 1.0e-9)),
            })

    metric_ok = all(bool(item["candidate_compare"]["metrics_match"]) and bool(item["no_control_compare"]["metrics_match"]) for item in records)
    budget_ok = all(float(item["budget_metric_abs_error_m3"]) <= float(args.tolerance) for item in records)
    admission_ok = all(bool(item["admission_match"]) for item in records)
    prefix_ok = all(bool(item["prefix_audit"]["prefix_match"]) for item in records)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = args.output_dir / "DIRECT_SWMM_METRIC_CONSISTENCY_ROWS.jsonl"
    rows_path.write_text("\n".join(json.dumps(item, ensure_ascii=False, allow_nan=False) for item in records) + "\n", encoding="utf-8")
    audit = {
        "audit_id": "V42_DIRECT_SWMM_METRIC_CONSISTENCY_V1",
        "development_only": True,
        "new_swmm_started": False,
        "states_tested": int(plan["state_key"].nunique()),
        "candidate_rows_tested": int(len(records)),
        "metric_tolerance": float(args.tolerance),
        "metric_consistency_pass": metric_ok,
        "pfv_budget_metric_consistency_pass": budget_ok,
        "pfv_admission_consistency_pass": admission_ok,
        "same_prefix_consistency_pass": prefix_ok,
        "status": "pass" if metric_ok and budget_ok and admission_ok and prefix_ok else "fail",
        "rows_path": str(rows_path),
    }
    (args.output_dir / "DIRECT_SWMM_METRIC_CONSISTENCY_AUDIT.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(audit, indent=2, ensure_ascii=False), flush=True)
    return 0 if audit["status"] == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
