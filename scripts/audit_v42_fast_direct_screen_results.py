"""Summarize authoritative FAST direct-screen trajectories with shared metrics."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_v42_fast_direct_screen import _detail_metrics
from sewerrtc.v4.v42_formal_runtime import load_actuators
from sewerrtc.v4.v42_priority_contract import get_pfv_core_node_indices
from sewerrtc.v4.v42_trajectory_builder import _load_graph_topology


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _detail_cache(path: str, cache: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if path not in cache:
        cache[path] = pd.read_csv(path)
    return cache[path]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--screen-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--plan-csv", type=Path, required=True)
    args = parser.parse_args()

    ledger_paths = sorted(set(args.screen_dir.glob("DIRECT_SCREEN_*_LEDGER.jsonl")) | set(args.screen_dir.rglob("DIRECT_SCREEN_*_LEDGER.jsonl")))
    if not ledger_paths:
        raise RuntimeError(f"no screen ledgers under {args.screen_dir}")
    rows = [row for path in ledger_paths for row in _jsonl(path)]
    # A resumed queue may append a second ledger record for the same action;
    # the last record is authoritative and avoids double-counting evaluations.
    deduped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("state_key", "")), str(row.get("candidate_action_sha256", "")))
        deduped[key] = row
    rows = list(deduped.values())
    manifest = pd.read_parquet(args.manifest, columns=[
        "state_key", "candidate_action_sha256", "checkpoint_min",
        "source_detail_path_candidate", "source_detail_path_no_control",
        "source_detail_path_dynamic_internal",
    ])
    internal_paths = (
        manifest.dropna(subset=["state_key", "source_detail_path_dynamic_internal"])
        .assign(state_key=lambda x: x.state_key.astype(str))
        .drop_duplicates("state_key")
        .set_index("state_key")["source_detail_path_dynamic_internal"]
        .to_dict()
    )
    plan = pd.read_csv(args.plan_csv).assign(state_key=lambda x: x.state_key.astype(str))
    plan_by_state = plan.set_index("state_key").to_dict("index")
    internal_paths = {key: value for key, value in internal_paths.items() if key in plan_by_state}
    graph = _load_graph_topology(args.project_root)
    node_ids = [str(x) for x in graph["node_ids"]]
    priority_nodes = [node_ids[int(i)] for i in get_pfv_core_node_indices(node_ids)]
    cache: dict[str, pd.DataFrame] = {}
    metric_rows: list[dict[str, Any]] = []
    internal_metrics: dict[str, dict[str, float]] = {}
    for state_key, internal_path in internal_paths.items():
        internal_metrics[state_key] = _detail_metrics(
            Path(internal_path), Path(internal_path), float(plan_by_state[state_key]["checkpoint_min"]), priority_nodes
        )
    # Re-evaluate only the frozen 8-state Round2 rows from existing details.
    # This repairs comparison lineage without launching SWMM or rescanning all
    # historical data.
    round2_best: dict[str, dict[str, Any]] = {}
    for state_key, group in manifest.assign(state_key=lambda x: x.state_key.astype(str)).groupby("state_key", sort=False):
        if state_key not in plan_by_state:
            continue
        seen_actions: set[str] = set()
        candidates: list[dict[str, Any]] = []
        reference_path = str(group["source_detail_path_no_control"].dropna().iloc[0])
        checkpoint = float(plan_by_state[state_key]["checkpoint_min"])
        for _, source_row in group.iterrows():
            action_sha = str(source_row.get("candidate_action_sha256", ""))
            detail_path = str(source_row.get("source_detail_path_candidate", ""))
            if not action_sha or action_sha in seen_actions or not Path(detail_path).exists():
                continue
            seen_actions.add(action_sha)
            metrics = _detail_metrics(Path(detail_path), Path(reference_path), checkpoint, priority_nodes)
            tfv_internal = float(internal_metrics[state_key]["tfv_candidate_m3"])
            reduction = 100.0 * (tfv_internal - metrics["tfv_candidate_m3"]) / tfv_internal if tfv_internal else float("nan")
            candidates.append({"action_sha256": action_sha, "tfv_reduction_pct": reduction, **metrics})
        safe = [item for item in candidates if item["pfv_feasible"]]
        best = min(safe, key=lambda item: (item["tfv_candidate_m3"], item["action_sha256"])) if safe else None
        round2_best[state_key] = {
            "count": len(candidates), "safe_count": len(safe), "best": best,
        }
    for row in rows:
        status = str(row.get("status", ""))
        if status not in {"pass", "reused"}:
            continue
        state_key = str(row["state_key"])
        detail_path = str(row.get("candidate_detail") or row.get("detail_path") or "")
        reference_path = str(row.get("source_detail_path_no_control") or "")
        if not detail_path or not reference_path or not Path(detail_path).exists() or not Path(reference_path).exists():
            continue
        checkpoint = float(row["checkpoint_min"])
        metrics = _detail_metrics(Path(detail_path), Path(reference_path), checkpoint, priority_nodes)
        internal = internal_metrics[state_key]
        tfv_internal = float(internal["tfv_candidate_m3"])
        reduction = 100.0 * (tfv_internal - metrics["tfv_candidate_m3"]) / tfv_internal if tfv_internal else float("nan")
        metric_rows.append({
            "stage": "screen",
            "state_key": state_key, "event_id": row.get("event_id"), "load_regime": plan_by_state[state_key]["load_regime"],
            "candidate_action_sha256": row.get("candidate_action_sha256"), "detail_path": detail_path,
            "reused": bool(row.get("reused", False)), "pfv_candidate_m3": metrics["pfv_candidate_m3"],
            "pfv_no_control_m3": metrics["pfv_no_control_m3"], "pfv_budget_metric_m3": metrics["pfv_budget_metric_m3"],
            "pfv_feasible": metrics["pfv_feasible"], "tfv_candidate_m3": metrics["tfv_candidate_m3"],
            "tfv_internal_m3": tfv_internal, "tfv_reduction_pct": reduction,
            "peak_candidate_rate": metrics["peak_candidate_rate"],
            "round2_reduction_pct": plan_by_state[state_key].get("oracle_tfv_reduction_pct"),
        })
    detail_rows = pd.DataFrame(metric_rows)
    if detail_rows.empty:
        raise RuntimeError("no valid screen result rows")
    detail_rows.to_csv(args.screen_dir / "FAST_DIRECT_SCREEN_ROWS.csv", index=False)
    states = []
    for state_key, group in detail_rows.groupby("state_key", sort=False):
        safe = group[group.pfv_feasible]
        best = safe.sort_values(["tfv_candidate_m3", "candidate_action_sha256"], kind="stable").head(1)
        plan_row = plan_by_state[state_key]
        stage_a = best.iloc[0] if not best.empty else group.iloc[0]
        round2 = float(plan_row["oracle_tfv_reduction_pct"]) if pd.notna(plan_row["oracle_tfv_reduction_pct"]) else float("nan")
        recomputed = round2_best.get(state_key, {})
        recomputed_best = recomputed.get("best")
        round2_shared = float(recomputed_best["tfv_reduction_pct"]) if recomputed_best else float("nan")
        direct = float(stage_a["tfv_reduction_pct"]) if not best.empty else float("nan")
        states.append({
            "state_key": state_key, "event_id": plan_row["event_id"], "load_regime": plan_row["load_regime"],
            "round2_reduction_pct": round2, "round2_shared_metric_reduction_pct": round2_shared,
            "round2_shared_metric_safe_count": int(recomputed.get("safe_count", 0)),
            "round2_shared_metric_candidate_count": int(recomputed.get("count", 0)),
            "stage_a_reduction_pct": direct,
            "gain_over_round2_pp": direct - round2_shared if np.isfinite(direct) and np.isfinite(round2_shared) else float("nan"),
            "pfv_feasible": bool(not best.empty), "evaluations": int(len(group)),
            "safe_evaluations": int(len(safe)), "best_action_sha256": stage_a["candidate_action_sha256"],
            "best_tfv_m3": float(stage_a["tfv_candidate_m3"]) if not best.empty else float("nan"),
            "pfv_budget_metric_m3": float(stage_a["pfv_budget_metric_m3"]) if not best.empty else float("nan"),
            "search_status": "REFINE" if (not np.isfinite(round2_shared) or (np.isfinite(direct) and direct - round2_shared >= 2.0)) else "FAST_SATURATED",
        })
    state_rows = pd.DataFrame(states)
    state_rows.to_csv(args.screen_dir / "FAST_DIRECT_SCREEN_STATES.csv", index=False)
    summary = {
        "screen": "FAST_DIRECT_SWMM_CONTROL_POTENTIAL_SCREEN",
        "development_only": True, "online_deployable": False,
        "states": int(len(state_rows)), "result_rows": int(len(detail_rows)),
        "safe_states": int(state_rows.pfv_feasible.sum()),
        "new_detail_rows": int((~detail_rows.reused).sum()), "reused_rows": int(detail_rows.reused.sum()),
        "overall_median_reduction_pct": float(state_rows.stage_a_reduction_pct.median()),
        "regime_medians": state_rows.groupby("load_regime").stage_a_reduction_pct.median().to_dict(),
        "saturated_states": state_rows.loc[state_rows.search_status.eq("FAST_SATURATED"), "state_key"].tolist(),
        "refine_states": state_rows.loc[state_rows.search_status.eq("REFINE"), "state_key"].tolist(),
    }
    (args.screen_dir / "FAST_DIRECT_SCREEN.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    (args.screen_dir / "FAST_DIRECT_SCREEN.md").write_text(
        "# FAST TRUE-STATE SWMM CONTROL-POTENTIAL SCREEN\n\n"
        + "```json\n" + json.dumps(summary, indent=2, ensure_ascii=False) + "\n```\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
