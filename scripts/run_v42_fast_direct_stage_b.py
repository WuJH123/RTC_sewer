"""Bounded local refinement for the frozen FAST direct-screen states."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_v42_fast_direct_screen import _array
from scripts.run_v42_targeted_candidate_expansion import _run_one
from sewerrtc.control.authoritative_control_metrics_v42 import action_sha256
from sewerrtc.v4.v42_formal_runtime import load_actuators
from sewerrtc.v4.v42_priority_contract import get_pfv_core_node_indices
from sewerrtc.v4.v42_trajectory_builder import _load_graph_topology


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _sequence(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def _perturb(base: np.ndarray, index: int, delta: float, pattern: str = "constant", second: tuple[int, float] | None = None) -> np.ndarray:
    out = base.copy()
    if pattern == "early_high":
        offsets = np.asarray([delta, delta / 2.0, 0.0], dtype=np.float32)
    elif pattern == "early_low":
        offsets = np.asarray([-delta, -delta / 2.0, 0.0], dtype=np.float32)
    else:
        offsets = np.full(3, delta, dtype=np.float32)
    out[:3, index] = np.clip(out[:3, index] + offsets, 0.0, 1.0)
    if second is not None:
        second_index, second_delta = second
        out[:3, second_index] = np.clip(out[:3, second_index] + second_delta, 0.0, 1.0)
    return out


def build_stage_b_sequences(base_actions: list[np.ndarray], facility_indices: list[int]) -> list[np.ndarray]:
    """Build at most 32 local candidates; H4-H12 stay at each base action."""
    if not base_actions or not facility_indices:
        return []
    candidates: list[np.ndarray] = []
    for base in base_actions[:2]:
        for index in facility_indices[:4]:
            for delta in (-0.10, -0.05, 0.05, 0.10):
                candidates.append(_perturb(base, index, delta))
        for index in facility_indices[:2]:
            candidates.append(_perturb(base, index, 0.10, "early_high"))
            candidates.append(_perturb(base, index, 0.10, "early_low"))
        if len(facility_indices) >= 2:
            first, second = facility_indices[:2]
            for first_delta, second_delta in ((0.05, 0.05), (0.05, -0.05), (-0.05, 0.05), (-0.05, -0.05)):
                candidates.append(_perturb(base, first, first_delta, second=(second, second_delta)))
    unique: dict[str, np.ndarray] = {}
    for candidate in candidates:
        unique.setdefault(action_sha256(candidate), candidate)
    return list(unique.values())[:32]


def _canonical_manifest_cache(group: pd.DataFrame) -> dict[str, str]:
    cache: dict[str, str] = {}
    for _, row in group.iterrows():
        raw = row.get("action_candidate_readback")
        path = str(row.get("source_detail_path_candidate", ""))
        if pd.isna(raw) or not path or not Path(path).exists():
            continue
        try:
            cache[action_sha256(_array(raw))] = path
        except Exception:
            continue
    return cache


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--screen-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--plan-csv", type=Path, required=True)
    parser.add_argument("--importance", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers != 16:
        raise RuntimeError("fast screen requires exactly 16 workers")

    stage_a = args.screen_root / "stage_a"
    stage_b = args.screen_root / "stage_b"
    stage_b.mkdir(parents=True, exist_ok=True)
    plan = pd.read_csv(args.plan_csv).assign(state_key=lambda x: x.state_key.astype(str))
    states = pd.read_csv(stage_a / "FAST_DIRECT_SCREEN_STATES.csv").assign(state_key=lambda x: x.state_key.astype(str))
    priority = {"MODERATE_LOAD": 0, "LOW_LOAD": 1, "NEAR_CAPACITY": 2, "SEVERE_OVERLOAD": 3}
    states = states[states.search_status.eq("REFINE")].copy()
    states["_priority"] = states.load_regime.map(priority).fillna(9)
    states = states.sort_values(["_priority", "gain_over_round2_pp"], ascending=[True, False], kind="stable").head(4)
    if states.empty:
        summary = {"stage": "stage_b", "development_only": True, "online_deployable": False, "selected_states": [], "planned": 0, "new_jobs": 0, "reused": 0, "failed": 0, "status": "not_needed"}
        (stage_b / "DIRECT_SCREEN_STAGE_B_SUMMARY.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2), flush=True)
        return 0

    manifest = pd.read_parquet(args.manifest, columns=[
        "state_key", "event_id", "rainfall_sha256", "checkpoint_min", "action_candidate_readback",
        "source_detail_path_candidate", "source_detail_path_no_control", "source_detail_path_hold_previous",
        "action_hold_previous_readback",
    ]).assign(state_key=lambda x: x.state_key.astype(str))
    importance = pd.read_csv(args.importance)
    actuators = load_actuators(args.project_root)
    ids = actuators.actuator_id.astype(str).tolist()
    graph = _load_graph_topology(args.project_root)
    node_ids = [str(x) for x in graph["node_ids"]]
    priority_nodes = [node_ids[int(i)] for i in get_pfv_core_node_indices(node_ids)]

    ledger_rows = _jsonl(stage_a / "DIRECT_SCREEN_STAGE_A_LEDGER.jsonl")
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in ledger_rows:
        latest[(str(row.get("state_key", "")), str(row.get("candidate_action_sha256", "")))] = row
    jobs: list[dict[str, Any]] = []
    reused: list[dict[str, Any]] = []
    selected_states: list[str] = []
    for _, state in states.iterrows():
        state_key = str(state.state_key)
        selected_states.append(state_key)
        group = manifest[manifest.state_key.eq(state_key)]
        if group.empty:
            raise RuntimeError(f"missing manifest state {state_key}")
        source = group.iloc[0]
        action_paths = _canonical_manifest_cache(group)
        metric_rows = pd.read_csv(stage_a / "FAST_DIRECT_SCREEN_ROWS.csv")
        metric_rows = metric_rows[(metric_rows.state_key.astype(str) == state_key) & (metric_rows.pfv_feasible.astype(str).str.lower() == "true")]
        metric_rows = metric_rows.sort_values(["tfv_candidate_m3", "candidate_action_sha256"], kind="stable")
        base_actions: list[np.ndarray] = []
        for _, metric in metric_rows.iterrows():
            row = latest.get((state_key, str(metric.candidate_action_sha256)))
            if not row:
                continue
            raw = row.get("candidate_action") or row.get("sequence")
            if raw is None:
                continue
            candidate = _sequence(raw)
            if candidate.shape == (12, len(ids)):
                base_actions.append(candidate)
            if len(base_actions) >= 2:
                break
        if not base_actions:
            for row in latest.values():
                if str(row.get("state_key")) != state_key:
                    continue
                raw = row.get("candidate_action") or row.get("sequence")
                if raw is not None and _sequence(raw).shape == (12, len(ids)):
                    base_actions.append(_sequence(raw))
                if len(base_actions) >= 2:
                    break
        if not base_actions:
            raise RuntimeError(f"no Stage A action seed for {state_key}")
        selected = importance[(importance.state_key.astype(str) == state_key) & (~importance.binary)]
        selected = selected.sort_values(["frequency", "best_tfv_gain_pct", "facility_id"], ascending=[False, False, True], kind="stable").head(4)
        facility_indices = [ids.index(str(x)) for x in selected.facility_id if str(x) in ids]
        candidates = build_stage_b_sequences(base_actions, facility_indices)
        output_root = stage_b / "evaluations"
        for ordinal, sequence in enumerate(candidates):
            candidate_sha = action_sha256(sequence)
            record = latest.get((state_key, candidate_sha))
            base = {
                "state_key": state_key, "event_id": str(source.event_id), "rainfall_sha256": str(source.rainfall_sha256),
                "checkpoint_min": float(state.checkpoint_min) if "checkpoint_min" in state else float(source.checkpoint_min),
                "candidate_action_sha256": candidate_sha, "candidate_label": f"fast_stage_b_{ordinal}",
                "candidate_family": "stage_b_local_refinement", "candidate_round": 5,
                "sequence": sequence.tolist(), "current_action": _array(source["action_hold_previous_readback"])[0].tolist(),
                "actuator_ids": ids, "priority_nodes": priority_nodes, "actuators": actuators,
                "source_detail_path_no_control": str(source.source_detail_path_no_control),
                "source_detail_path_hold_previous": str(source.source_detail_path_hold_previous),
                "output_dir": str(output_root / state_key / candidate_sha), "resume": bool(args.resume),
            }
            detail = action_paths.get(candidate_sha)
            if detail:
                reused.append({**base, "status": "reused", "detail_path": detail})
            elif record and record.get("status") in {"pass", "reused"}:
                reused.append({**base, "status": "reused", "detail_path": str(record.get("candidate_detail") or record.get("detail_path") or "")})
            elif base["output_dir"] and args.resume and (Path(base["output_dir"]) / "detail.csv").exists() and (Path(base["output_dir"]) / "result.json").exists():
                jobs.append(base)
            else:
                jobs.append(base)
    if len(jobs) + len(reused) > 128:
        raise RuntimeError("Stage B budget exceeded")

    ledger_path = stage_b / "DIRECT_SCREEN_STAGE_B_LEDGER.jsonl"
    with ledger_path.open("a", encoding="utf-8") as ledger:
        for item in reused:
            clean = {key: value for key, value in item.items() if key not in {"actuators"}}
            clean["reused"] = True
            ledger.write(json.dumps(clean, ensure_ascii=False) + "\n")
        failed = 0
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
                ledger.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + "\n")
                ledger.flush()
    summary = {
        "stage": "stage_b", "development_only": True, "online_deployable": False,
        "workers": args.workers, "selected_states": selected_states, "planned": len(jobs) + len(reused),
        "new_jobs": len(jobs), "reused": len(reused), "failed": failed,
        "ledger": str(ledger_path), "status": "pass" if failed == 0 else "fail",
    }
    (stage_b / "DIRECT_SCREEN_STAGE_B_SUMMARY.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)
    return 0 if failed == 0 else 3


if __name__ == "__main__":
    raise SystemExit(main())
