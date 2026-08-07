"""Development-only replay of expanded candidates missing CONTROL_CORE targets."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_v42_targeted_candidate_expansion import (
    _build_expanded_manifest,
    _ranked_actuators,
    _run_one,
)
from sewerrtc.v4.v42_formal_runtime import load_actuators
from sewerrtc.v4.v42_node_safety import load_node_physical_contract
from sewerrtc.v4.v42_trajectory_builder import _load_graph_topology


def _array(value: object) -> np.ndarray:
    if isinstance(value, str):
        value = json.loads(value)
    return np.asarray(value, dtype=np.float32)


def _write_progress(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _build_jobs(frame: pd.DataFrame, args: argparse.Namespace) -> tuple[list[dict], pd.DataFrame]:
    audit = json.loads(args.state_audit.read_text(encoding="utf-8"))
    state_keys = {str(value) for value in audit["state_keys"]}
    if args.state_key:
        state_keys &= {str(value) for value in args.state_key}
    frame = frame[frame["state_key"].astype(str).isin(state_keys)].copy()
    if frame["state_key"].astype(str).nunique() != len(state_keys):
        raise RuntimeError("repair state audit and combined manifest do not match")

    actuators = load_actuators(args.project_root)
    ids = actuators["actuator_id"].astype(str).tolist()
    _, priority_nodes = _ranked_actuators(args.project_root, actuators)
    graph = _load_graph_topology(args.project_root)
    node_ids = [str(value) for value in graph["node_ids"]]
    physical = load_node_physical_contract(args.project_root)
    storage_node_ids = [node_ids[index] for index in physical.storage_indices]
    outfall_node_ids = [node_ids[index] for index in physical.outfall_indices]

    jobs: list[dict] = []
    for _, row in frame.iterrows():
        if bool(row.get("trajectory_storage_volume_candidate_available", False)) and bool(
            row.get("trajectory_facility_flow_candidate_available", False)
        ):
            continue
        sequence = _array(row["action_candidate_readback"])
        hold = _array(row["action_hold_previous_readback"])
        if sequence.shape != (12, len(ids)) or hold.shape[1:] != (len(ids),):
            raise RuntimeError(f"invalid action shape for {row['state_key']}")
        candidate_sha = str(row["candidate_action_sha256"])
        output = args.output_root / str(row["state_key"]) / candidate_sha
        candidate_round = row.get("candidate_expansion_round", 2)
        if pd.isna(candidate_round):
            candidate_round = 2
        jobs.append(
            {
                "state_key": str(row["state_key"]),
                "event_id": str(row["event_id"]),
                "rainfall_sha256": str(row["rainfall_sha256"]),
                "checkpoint_min": float(row["checkpoint_min"]),
                "candidate_label": f"repair:{row.get('candidate_expansion_family', '')}",
                "candidate_round": int(candidate_round),
                "candidate_family": str(row.get("candidate_expansion_family", "repair")),
                "candidate_action_sha256": candidate_sha,
                "sequence": sequence.tolist(),
                "current_action": hold[0].tolist(),
                "actuator_ids": ids,
                "priority_nodes": priority_nodes,
                "storage_node_ids": storage_node_ids,
                "outfall_node_ids": outfall_node_ids,
                "actuators": actuators,
                "source_detail_path_no_control": str(row["source_detail_path_no_control"]),
                "source_detail_path_hold_previous": str(row["source_detail_path_hold_previous"]),
                "output_dir": str(output),
                "resume": bool(args.resume),
            }
        )
    if args.max_jobs:
        jobs = jobs[: int(args.max_jobs)]
    return jobs, frame


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--state-audit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--max-jobs", type=int, default=0)
    parser.add_argument("--state-key", action="append")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 1 <= int(args.workers) <= 16:
        raise ValueError("workers must be in [1, 16]")

    columns = [
        "event_id", "rainfall_sha256", "checkpoint_min", "state_key",
        "candidate_action_sha256", "action_candidate_readback",
        "action_hold_previous_readback", "source_detail_path_no_control",
        "source_detail_path_hold_previous", "candidate_expansion_family",
        "candidate_expansion_round", "trajectory_storage_volume_candidate_available",
        "trajectory_facility_flow_candidate_available",
    ]
    frame = pd.read_parquet(args.manifest, columns=columns)
    jobs, selected = _build_jobs(frame, args)
    selected_states = sorted(selected["state_key"].astype(str).unique())
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress_path = args.output_root / "REPAIR_PROGRESS.json"
    results: list[dict] = []
    started = time.time()
    total = len(jobs)
    print(json.dumps({"planned": total, "workers": int(args.workers), "reused_targets": int(len(selected) - total)}), flush=True)

    def record(result: dict, index: int) -> None:
        results.append(result)
        elapsed = max(time.time() - started, 1.0e-6)
        completed = len(results)
        _write_progress(
            progress_path,
            {
                "planned": total,
                "completed": completed,
                "failed": sum(item.get("status") == "fail" for item in results),
                "workers": int(args.workers),
                "throughput_per_min": completed / elapsed * 60.0,
                "last_state": result.get("state_key"),
                "last_candidate_action_sha256": result.get("candidate_action_sha256"),
            },
        )
        print(json.dumps({"completed": index, "total": total, "status": result.get("status"), "state_key": result.get("state_key")}), flush=True)

    if int(args.workers) == 1:
        for index, job in enumerate(jobs, 1):
            try:
                result = _run_one(job)
            except Exception as exc:
                result = {"status": "fail", "state_key": job["state_key"], "candidate_action_sha256": job["candidate_action_sha256"], "error": repr(exc)}
            record(result, index)
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
            futures = {pool.submit(_run_one, job): job for job in jobs}
            for index, future in enumerate(as_completed(futures), 1):
                job = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"status": "fail", "state_key": job["state_key"], "candidate_action_sha256": job["candidate_action_sha256"], "error": repr(exc)}
                record(result, index)

    result_frame = pd.DataFrame(results)
    result_frame.to_csv(args.output_root / "TARGET_REPAIR_FUNNEL.csv", index=False)
    if results:
        selected_full = pd.read_parquet(
            args.manifest,
            filters=[("state_key", "in", selected_states)],
        )
        repaired = _build_expanded_manifest(results, selected_full, args.project_root)
        merged = selected_full.set_index("candidate_action_sha256", drop=False)
        merged.update(repaired.set_index("candidate_action_sha256", drop=False))
        merged.reset_index(drop=True).to_parquet(args.output_root / "FAST8_REPAIRED_CONTROL_CORE_MANIFEST.parquet", index=False)
    audit_out = {
        "audit_id": "V42_STEP2_TARGET_REPAIR_AUDIT_V1",
        "development_only": True,
        "formal_mainline_authorized": False,
        "input_rows": int(len(selected)),
        "planned_repair_rows": total,
        "completed_rows": int(len(results)),
        "failed_rows": int(sum(item.get("status") == "fail" for item in results)),
        "workers": int(args.workers),
        "no_reference_runs": True,
        "output_manifest": str(args.output_root / "FAST8_REPAIRED_CONTROL_CORE_MANIFEST.parquet"),
        "output_funnel": str(args.output_root / "TARGET_REPAIR_FUNNEL.csv"),
    }
    (args.output_root / "FAST8_TARGET_REPAIR_AUDIT.json").write_text(json.dumps(audit_out, indent=2, ensure_ascii=False), encoding="utf-8")
    return 2 if audit_out["failed_rows"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
