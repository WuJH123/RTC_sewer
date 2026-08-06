"""Bounded authoritative SWMM fast screen; development-only and resumable."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import qmc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_v42_targeted_candidate_expansion import _run_one
from sewerrtc.control.authoritative_control_metrics_v42 import (
    action_sha256,
    detail_horizon_metrics,
    rolling_pfv_budget_metric,
    trajectory_metrics,
)
from sewerrtc.v4.v42_formal_runtime import load_actuators
from sewerrtc.v4.v42_priority_contract import get_pfv_core_node_indices
from sewerrtc.v4.v42_trajectory_builder import _load_graph_topology


_MANIFEST_COLUMNS = [
    "state_key", "event_id", "rainfall_sha256", "checkpoint_min",
    "candidate_action_sha256", "action_candidate_readback",
    "pfv_delta", "tfv_delta",
    "source_detail_path_candidate", "source_detail_path_no_control",
    "source_detail_path_hold_previous", "action_hold_previous_readback",
]


def _array(value: Any) -> np.ndarray:
    return np.asarray(json.loads(value) if isinstance(value, str) else value, dtype=np.float32)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _detail_metrics(path: Path, reference: Path, checkpoint: float, priority_nodes: list[str]) -> dict[str, Any]:
    usecols = lambda c: c == "elapsed_min" or c == "rainfall_mm_h" or str(c).startswith("flood:") or str(c).startswith("a:")
    candidate = pd.read_csv(path, usecols=usecols)
    no_control = pd.read_csv(reference, usecols=usecols)
    future = detail_horizon_metrics(candidate, priority_nodes, checkpoint_min=checkpoint, steps=12)
    reference_future = detail_horizon_metrics(no_control, priority_nodes, checkpoint_min=checkpoint, steps=12)
    budget = rolling_pfv_budget_metric(candidate, no_control, priority_nodes=priority_nodes, checkpoint_min=checkpoint, relative_margin=0.05, steps=12)
    return {
        "pfv_candidate_m3": future["PFV"],
        "pfv_no_control_m3": reference_future["PFV"],
        "tfv_candidate_m3": future["TFV"],
        "peak_candidate_rate": future["peak_TFV_rate"],
        "pfv_budget_metric_m3": budget,
        "pfv_feasible": bool(budget <= 100.0 + 1.0e-9),
    }


def _build_sequence(current: np.ndarray, continuous: list[str], ids: list[str], values: np.ndarray, binary_ids: list[str], binary_mode: str = "hold") -> np.ndarray:
    sequence = np.tile(current, (12, 1)).astype(np.float32)
    for name, value in zip(continuous, values):
        sequence[:3, ids.index(name)] = float(np.clip(value, 0.0, 1.0))
    if binary_mode != "hold":
        for name in binary_ids:
            sequence[:3, ids.index(name)] = 1.0 if binary_mode == "111" else 0.0
    return sequence


def _authoritative_seed_rows(group: pd.DataFrame, source: pd.Series, checkpoint: float, priority_nodes: list[str]) -> pd.DataFrame:
    """Rank existing actions with the same metric used by the screen.

    ponytail: bounded selected-state scan; avoid trusting stale manifest labels,
    and do not generalize this expensive read to the full 7,908-row population.
    """
    rows: list[dict[str, Any]] = []
    reference = Path(str(source.source_detail_path_no_control))
    for _, row in group.drop_duplicates("candidate_action_sha256", keep="first").iterrows():
        raw = row.get("action_candidate_readback")
        path = Path(str(row.get("source_detail_path_candidate", "")))
        if pd.isna(raw) or not path.exists() or not reference.exists():
            continue
        try:
            metrics = _detail_metrics(path, reference, checkpoint, priority_nodes)
        except Exception:
            continue
        item = row.to_dict()
        item["_seed_safe"] = bool(metrics["pfv_feasible"])
        item["_seed_tfv"] = float(metrics["tfv_candidate_m3"])
        item["_seed_budget"] = float(metrics["pfv_budget_metric_m3"])
        rows.append(item)
    if not rows:
        return group
    return pd.DataFrame(rows).sort_values(
        ["_seed_safe", "_seed_tfv", "_seed_budget"],
        ascending=[False, True, True],
        kind="stable",
    )


def _state_jobs(state: pd.Series, group: pd.DataFrame, importance: pd.DataFrame, ids: list[str], actuators: pd.DataFrame, priority_nodes: list[str], output_root: Path, existing_hashes: set[str], count: int = 48) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    source = group.iloc[0]
    current = _array(source["action_hold_previous_readback"])[0]
    continuous = importance[(importance.state_key == str(state.state_key)) & (~importance.binary)].sort_values(["frequency", "best_tfv_gain_pct", "facility_id"], ascending=[False, False, True], na_position="last", kind="stable").head(6)["facility_id"].astype(str).tolist()
    binary = importance[(importance.state_key == str(state.state_key)) & importance.binary & (importance.safe_improving_change_count >= 2)].sort_values(["frequency", "best_tfv_gain_pct", "facility_id"], ascending=[False, False, True], kind="stable")["facility_id"].astype(str).tolist()
    seeds: list[np.ndarray] = [np.tile(current, (12, 1)).astype(np.float32)]
    seed_rows = _authoritative_seed_rows(group, source, float(state.checkpoint_min), priority_nodes)
    canonical_rows: dict[str, pd.Series] = {}
    for _, existing_row in group.iterrows():
        raw = existing_row.get("action_candidate_readback")
        if pd.isna(raw):
            continue
        try:
            canonical_rows[action_sha256(_array(raw))] = existing_row
        except Exception:
            continue
    seen_seed_hashes: set[str] = set()
    for _, seed_row in seed_rows.iterrows():
        raw = seed_row.get("action_candidate_readback")
        if pd.isna(raw):
            continue
        candidate = _array(raw)
        if candidate.shape == (12, len(ids)):
            seed_sha = action_sha256(candidate)
            if seed_sha in seen_seed_hashes:
                continue
            seen_seed_hashes.add(seed_sha)
            seeds.append(candidate)
        if len(seeds) >= 12:
            break
    while len(seeds) < 12:
        seeds.append(np.tile(current, (12, 1)).astype(np.float32))
    lows = np.maximum(0.0, current[[ids.index(x) for x in continuous]] - 0.25) if continuous else np.zeros(1)
    highs = np.minimum(1.0, current[[ids.index(x) for x in continuous]] + 0.25) if continuous else np.ones(1)
    sampler = qmc.Sobol(d=max(1, len(continuous)), scramble=False, seed=42)
    sobol = sampler.random_base2(m=6)[: max(0, count - len(seeds))]
    candidates = seeds + [_build_sequence(current, continuous, ids, lows + row[: len(continuous)] * (highs - lows), binary_ids=binary) for row in sobol]
    jobs: list[dict[str, Any]] = []
    reused: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ordinal, sequence in enumerate(candidates[:count]):
        candidate_sha = action_sha256(sequence)
        if candidate_sha in seen:
            continue
        seen.add(candidate_sha)
        base = {
            "state_key": str(state.state_key), "event_id": str(state.event_id), "rainfall_sha256": str(state.rainfall_sha256),
            "checkpoint_min": float(state.checkpoint_min), "candidate_action_sha256": candidate_sha, "candidate_label": f"fast_screen_{ordinal}",
            "candidate_family": "authoritative_seed" if ordinal < len(seeds) else "sobol_stage_a", "candidate_round": 4,
            "sequence": sequence.tolist(), "current_action": current.tolist(), "actuator_ids": ids, "priority_nodes": priority_nodes,
            "actuators": actuators, "source_detail_path_no_control": str(source.source_detail_path_no_control),
            "source_detail_path_hold_previous": str(source.source_detail_path_hold_previous),
            "output_dir": str(output_root / str(state.state_key) / candidate_sha), "resume": True,
        }
        if candidate_sha in existing_hashes:
            row = group[group.candidate_action_sha256.astype(str).eq(candidate_sha)].head(1)
            if row.empty and candidate_sha in canonical_rows:
                row = pd.DataFrame([canonical_rows[candidate_sha]])
            reused.append({**base, "status": "reused", "detail_path": str(row.iloc[0].source_detail_path_candidate) if not row.empty else ""})
        elif ordinal == 0:
            reused.append({**base, "status": "reused", "detail_path": str(source.source_detail_path_hold_previous), "candidate_family": "hold_reference_reuse"})
        else:
            jobs.append(base)
    return jobs, reused


def _write_progress(path: Path, *, planned: int, completed: int, reused: int, failed: int, running: int, started: float, best: dict[str, Any]) -> None:
    elapsed = max(1.0, time.time() - started)
    done = completed + reused
    path.write_text(json.dumps({"planned": planned, "completed": completed, "reused": reused, "failed": failed, "running": running, "throughput_per_min": done / elapsed * 60.0, "best_by_state": best, "updated_at_epoch": time.time()}, indent=2, ensure_ascii=False), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--plan-csv", type=Path, required=True)
    parser.add_argument("--plan-lock", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--importance", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("smoke", "stage_a"), default="smoke")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers != 16:
        raise RuntimeError("fast screen requires exactly 16 workers")
    lock = _read_json(args.plan_lock)
    plan = pd.read_csv(args.plan_csv)
    if hashlib.sha256(args.plan_csv.read_bytes()).hexdigest() != lock.get("plan_sha256"):
        raise RuntimeError("fast plan SHA mismatch")
    if args.stage == "smoke":
        plan = plan.head(1)
    # ponytail: read only the columns needed to construct/reuse jobs; the full
    # manifest contains many wide nested trajectory columns and is not needed
    # for a bounded screening queue.
    manifest = pd.read_parquet(args.manifest, columns=_MANIFEST_COLUMNS)
    importance = pd.read_csv(args.importance)
    actuators = load_actuators(args.project_root)
    ids = actuators["actuator_id"].astype(str).tolist()
    graph = _load_graph_topology(args.project_root)
    node_ids = [str(x) for x in graph["node_ids"]]
    priority_nodes = [node_ids[int(i)] for i in get_pfv_core_node_indices(node_ids)]
    existing_by_state: dict[str, set[str]] = {}
    for key, group in manifest.groupby(manifest.state_key.astype(str), sort=False):
        hashes = set(group.candidate_action_sha256.dropna().astype(str))
        for raw in group.action_candidate_readback.dropna():
            try:
                hashes.add(action_sha256(_array(raw)))
            except Exception:
                continue
        existing_by_state[str(key)] = hashes
    output_root = args.output_dir / "evaluations"
    output_root.mkdir(parents=True, exist_ok=True)
    jobs: list[dict[str, Any]] = []
    reused: list[dict[str, Any]] = []
    for _, state in plan.iterrows():
        group = manifest[manifest.state_key.astype(str).eq(str(state.state_key))]
        state_jobs, state_reused = _state_jobs(state, group, importance, ids, actuators, priority_nodes, output_root, existing_by_state.get(str(state.state_key), set()), 48)
        jobs.extend(state_jobs)
        reused.extend(state_reused)
    if args.stage == "smoke":
        jobs = jobs[:16]
        reused = reused[: max(0, 16 - len(jobs))]
    if len(jobs) + len(reused) > (16 if args.stage == "smoke" else 384):
        raise RuntimeError("fast screen budget exceeded")
    runtime = args.project_root / "outputs" / "project6_dual_reference_v4" / "final_v4" / "v42_paper" / "formal_f2" / "_codex" / "runtime" / "fast_direct_screen"
    runtime.mkdir(parents=True, exist_ok=True)
    (runtime / "pid.json").write_text(json.dumps({"pid": __import__("os").getpid(), "stage": args.stage, "workers": args.workers}, indent=2), encoding="utf-8")
    ledger_path = args.output_dir / f"DIRECT_SCREEN_{args.stage.upper()}_LEDGER.jsonl"
    progress_path = runtime / "progress.json"
    started = time.time()
    results: list[dict[str, Any]] = []
    best: dict[str, Any] = {}
    failed = 0
    with ledger_path.open("a", encoding="utf-8") as ledger:
        for item in reused:
            item["detail_path"] = item.pop("detail_path", "")
            item.pop("actuators", None)
            item["reused"] = True
            results.append(item)
            ledger.write(json.dumps(item, ensure_ascii=False) + "\n")
        _write_progress(progress_path, planned=len(jobs) + len(reused), completed=0, reused=len(reused), failed=0, running=0, started=started, best=best)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(_run_one, job): job for job in jobs}
            for future in as_completed(futures):
                job = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"status": "fail", "state_key": job["state_key"], "candidate_action_sha256": job["candidate_action_sha256"], "error": repr(exc)}
                if result.get("status") == "fail":
                    failed += 1
                result["reused"] = bool(result.get("status") == "reused")
                results.append(result)
                ledger.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                ledger.flush()
                state_key = str(result.get("state_key"))
                if result.get("status") == "pass":
                    best[state_key] = {"candidate_action_sha256": result.get("candidate_action_sha256"), "status": "completed"}
                _write_progress(progress_path, planned=len(jobs) + len(reused), completed=len(results) - len(reused), reused=len(reused), failed=failed, running=max(0, len(futures) - len(results) + len(reused)), started=started, best=best)
    summary = {"stage": args.stage, "development_only": True, "online_deployable": False, "workers": args.workers, "planned": len(jobs) + len(reused), "new_jobs": len(jobs), "reused": sum(bool(item.get("reused")) for item in results), "failed": failed, "new_swmm_started": any(item.get("status") == "pass" and not item.get("reused") for item in results), "ledger": str(ledger_path), "status": "pass" if failed == 0 else "fail"}
    (args.output_dir / f"DIRECT_SCREEN_{args.stage.upper()}_SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0 if failed == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
