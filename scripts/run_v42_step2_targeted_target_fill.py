"""Run the bounded development-only Step-2 target fill using existing actions."""
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

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_v42_targeted_candidate_expansion import _run_one
from sewerrtc.v4.v42_formal_runtime import load_actuators
from sewerrtc.v4.v42_node_safety import load_node_physical_contract
from sewerrtc.v4.v42_priority_contract import get_pfv_core_node_indices
from sewerrtc.v4.v42_trajectory_builder import _load_graph_topology


def array(value: object) -> np.ndarray:
    return np.asarray(json.loads(str(value)), dtype=np.float32)


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    for _ in range(20):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            time.sleep(0.1)
    # Windows readers can briefly hold the destination open.  The final
    # progress write is recoverable; do not turn completed SWMM results into
    # a failed campaign because a telemetry file was locked.
    path.write_text(temporary.read_text(encoding="utf-8"), encoding="utf-8")
    temporary.unlink(missing_ok=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=ROOT)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--output-root", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    if not 1 <= int(args.workers) <= 16:
        raise ValueError("workers must be in [1, 16]")
    plan = pd.read_csv(args.plan)
    if len(plan) > 128:
        raise RuntimeError("target fill exceeds hard maximum 128")
    actuators = load_actuators(args.project_root)
    ids = actuators["actuator_id"].astype(str).tolist()
    graph = _load_graph_topology(args.project_root)
    node_ids = [str(value) for value in graph["node_ids"]]
    physical = load_node_physical_contract(args.project_root)
    storage_ids = [node_ids[index] for index in physical.storage_indices]
    outfall_ids = [node_ids[index] for index in physical.outfall_indices]
    priority_ids = [node_ids[index] for index in get_pfv_core_node_indices(node_ids)]
    jobs = []
    for row in plan.to_dict("records"):
        sequence = array(row["action_candidate_readback"])
        hold = array(row["action_hold_previous_readback"])
        if sequence.shape != (12, len(ids)) or hold.shape != (12, len(ids)):
            raise RuntimeError(f"invalid action shape for {row['state_key']}")
        action_sha = str(row["canonical_action_sha256"])
        jobs.append({
            "state_key": str(row["state_key"]), "event_id": str(row["event_id"]),
            "rainfall_sha256": str(row["rainfall_sha256"]), "checkpoint_min": float(row["checkpoint_min"]),
            "candidate_label": "targeted_step2_target_fill", "candidate_round": 99,
            "candidate_family": "target_fill", "candidate_action_sha256": action_sha,
            "sequence": sequence.tolist(), "current_action": hold[0].tolist(), "actuator_ids": ids,
            "priority_nodes": priority_ids, "storage_node_ids": storage_ids, "outfall_node_ids": outfall_ids,
            "actuators": actuators, "source_detail_path_no_control": str(row["source_detail_path_no_control"]),
            "source_detail_path_hold_previous": str(row["source_detail_path_hold_previous"]),
            "output_dir": str(args.output_root / str(row["state_key"]) / action_sha), "resume": bool(args.resume),
        })
    args.output_root.mkdir(parents=True, exist_ok=True)
    progress = args.output_root / "TARGET_FILL_PROGRESS.json"
    results = []
    started = time.time()
    print(json.dumps({"planned": len(jobs), "workers": int(args.workers), "new_reference_runs": False}), flush=True)
    with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
        futures = {pool.submit(_run_one, job): job for job in jobs}
        for completed, future in enumerate(as_completed(futures), 1):
            job = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"status": "fail", "state_key": job["state_key"], "candidate_action_sha256": job["candidate_action_sha256"], "error": repr(exc)}
            results.append(result)
            elapsed = max(time.time() - started, 1.0e-6)
            write_json(progress, {
                "planned": len(jobs), "completed": completed,
                "passed": sum(x.get("status") in {"pass", "reused"} for x in results),
                "failed": sum(x.get("status") == "fail" for x in results),
                "reused": sum(x.get("status") == "reused" for x in results),
                "workers": int(args.workers), "throughput_per_min": completed / elapsed * 60.0,
                "last_state": result.get("state_key"), "last_action_sha256": result.get("candidate_action_sha256"),
            })
            print(json.dumps({"completed": completed, "total": len(jobs), "status": result.get("status"), "state_key": result.get("state_key")}), flush=True)
    result_frame = pd.DataFrame(results)
    result_frame.to_csv(args.output_root / "TARGETED_STEP2_TARGET_FILL_FUNNEL.csv", index=False)
    audit = {
        "audit_id": "V42_STEP2_TARGETED_TARGET_FILL_AUDIT_V1", "development_only": True,
        "formal_mainline_authorized": False, "plan": str(args.plan.resolve()),
        "plan_sha256": hashlib.sha256(args.plan.read_bytes()).hexdigest(), "planned_rows": len(jobs),
        "completed_rows": len(results), "passed_rows": int(result_frame.get("status", pd.Series(dtype=str)).isin(["pass", "reused"]).sum()),
        "failed_rows": int((result_frame.get("status", pd.Series(dtype=str)) == "fail").sum()),
        "reused_rows": int((result_frame.get("status", pd.Series(dtype=str)) == "reused").sum()),
        "workers": int(args.workers), "new_reference_runs": False, "reference_reuse": True,
        "output_funnel": str((args.output_root / "TARGETED_STEP2_TARGET_FILL_FUNNEL.csv").resolve()),
    }
    write_json(args.output_root / "TARGETED_STEP2_TARGET_FILL_AUDIT.json", audit)
    return 2 if audit["failed_rows"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
