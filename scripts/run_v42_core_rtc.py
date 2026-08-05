"""Run the Project6 V4.2 core RTC loop without legacy engineering hard gates.

This runner is intentionally narrow. It exercises the paper's central online
chain on authoritative SWMM events:

sparse sensors -> Temporal GAT -> H120 surrogate -> PFV-UCB admission ->
minimum-TFV candidate -> target_setting write -> SWMM readback -> replan.

The only hydraulic hard constraint is
UCB(PFV_candidate - 1.05 * PFV_no_control) <= 100 m3.
K/rate/ramp/dwell/interlock are not used to reject candidates.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import pandas as pd

from sewerrtc.v4.v42_simple_rtc_contract import (
    CONTRACT_ID,
    apply_simple_rtc_contract,
)


DEFAULT_STRATEGIES = ("Proposed", "No-control", "Internal", "Hold")


def _load_core_calibration_events(project_root: Path) -> list[FormalEventInput]:
    """Resolve Fresh Calibration12's three branch rows into 12 SWMM events."""
    from sewerrtc.v4.v42_formal_runtime import FormalEventInput

    manifest = (
        project_root
        / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2"
        / "pfv_only_v2/FRESH_PFV_ONLY_CALIBRATION_CASE_MANIFEST.csv"
    )
    if not manifest.exists():
        raise FileNotFoundError(manifest)
    frame = pd.read_csv(manifest, low_memory=False)
    required = {
        "event_id",
        "rainfall_sha256",
        "inp_path",
        "rain_duration_min",
        "simulation_duration_min",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"Fresh Calibration manifest missing columns: {missing}")
    events: list[FormalEventInput] = []
    for (event_id, rainfall_sha), group in frame.groupby(
        ["event_id", "rainfall_sha256"], sort=True
    ):
        if group["inp_path"].astype(str).nunique() != 1:
            raise RuntimeError(
                f"Fresh Calibration event has multiple INP inputs: {event_id}"
            )
        row = group.iloc[0]
        inp = Path(str(row["inp_path"]))
        if not inp.is_absolute():
            inp = project_root / inp
        inp = inp.resolve()
        if not inp.exists():
            raise FileNotFoundError(inp)
        rain_duration = int(row["rain_duration_min"])
        simulation_duration = int(row["simulation_duration_min"])
        if simulation_duration < max(rain_duration, 240):
            raise RuntimeError(
                f"Fresh Calibration event has insufficient simulation duration: {event_id}"
            )
        events.append(
            FormalEventInput(
                role="calibration",
                event_id=str(event_id),
                rainfall_sha256=str(rainfall_sha),
                inp_path=inp,
                rain_duration_min=rain_duration,
                simulation_duration_min=simulation_duration,
            )
        )
    if len(events) != 12:
        raise RuntimeError(
            f"Fresh Calibration12 must resolve to 12 unique events, got {len(events)}"
        )
    return sorted(events, key=lambda event: (event.rainfall_sha256, event.event_id))


def _event_map(results: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for result in results:
        event = str(result.get("event_id", ""))
        strategy = str(result.get("strategy", ""))
        if event and strategy:
            out.setdefault(event, {})[strategy] = result
    return out


def _finite_or_none(value: Any) -> float | None:
    try:
        number = float(value)
    except Exception:
        return None
    return number if np.isfinite(number) else None


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_event = _event_map(results)
    rows: list[dict[str, Any]] = []
    pfv_pass_count = 0
    proposed_event_count = 0
    total_decisions = 0
    nonfallback_decisions = 0
    total_fallback_weighted = 0.0
    write_pass = True
    causal_pass = True

    for event_id, strategies in sorted(by_event.items()):
        proposed = strategies.get("Proposed")
        no_control = strategies.get("No-control")
        internal = strategies.get("Internal")
        if proposed is None:
            continue
        proposed_event_count += 1
        decision_count = int(proposed.get("decision_count", 0))
        fallback_rate = float(proposed.get("fallback_rate", 1.0))
        total_decisions += decision_count
        nonfallback_decisions += max(
            0, int(round(decision_count * (1.0 - min(max(fallback_rate, 0.0), 1.0))))
        )
        total_fallback_weighted += fallback_rate
        event_write_pass = bool(proposed.get("target_write_all_decisions_verified", False))
        event_causal_pass = bool(
            proposed.get("state_source") == "gat_sparse_reconstruction"
            and proposed.get("online_future_hydraulic_truth_used") is False
            and proposed.get("realized_future_rainfall_used_online") is False
            and proposed.get("internal_shadow_future_state_used_online") is False
        )
        write_pass = write_pass and event_write_pass
        causal_pass = causal_pass and event_causal_pass

        pk = proposed.get("kpis", {}) or {}
        nk = (no_control or {}).get("kpis", {}) or {}
        ik = (internal or {}).get("kpis", {}) or {}
        pfv_proposed = _finite_or_none(pk.get("PFV"))
        pfv_no_control = _finite_or_none(nk.get("PFV"))
        tfv_proposed = _finite_or_none(pk.get("TFV"))
        tfv_internal = _finite_or_none(ik.get("TFV"))
        budget = (
            100.0 + 1.05 * max(0.0, pfv_no_control)
            if pfv_no_control is not None
            else None
        )
        pfv_pass = bool(
            pfv_proposed is not None
            and budget is not None
            and pfv_proposed <= budget + 1e-6
        )
        pfv_pass_count += int(pfv_pass)
        delta_tfv = (
            tfv_proposed - tfv_internal
            if tfv_proposed is not None and tfv_internal is not None
            else None
        )
        rows.append(
            {
                "event_id": event_id,
                "PFV_proposed_m3": pfv_proposed,
                "PFV_no_control_m3": pfv_no_control,
                "PFV_budget_m3": budget,
                "PFV_constraint_pass": pfv_pass,
                "TFV_proposed_m3": tfv_proposed,
                "TFV_internal_m3": tfv_internal,
                "delta_TFV_vs_internal_m3": delta_tfv,
                "fallback_rate": fallback_rate,
                "decision_count": decision_count,
                "active_nonfallback_decisions": max(
                    0,
                    int(
                        round(
                            decision_count
                            * (1.0 - min(max(fallback_rate, 0.0), 1.0))
                        )
                    ),
                ),
                "target_write_all_decisions_verified": event_write_pass,
                "causal_sparse_GAT_online_pass": event_causal_pass,
            }
        )

    delta_values = [
        float(row["delta_TFV_vs_internal_m3"])
        for row in rows
        if row["delta_TFV_vs_internal_m3"] is not None
    ]
    mean_delta_tfv = float(np.mean(delta_values)) if delta_values else None
    mean_fallback = (
        total_fallback_weighted / proposed_event_count if proposed_event_count else 1.0
    )
    status = "pass" if (
        proposed_event_count > 0
        and total_decisions > 0
        and nonfallback_decisions > 0
        and write_pass
        and causal_pass
        and pfv_pass_count == proposed_event_count
    ) else "fail"
    return {
        "status": status,
        "contract_id": CONTRACT_ID,
        "online_chain": "sparse_sensors->GAT->H120_surrogate->PFV_UCB->min_TFV->SWMM_write/readback->replan",
        "hydraulic_hard_constraint": "UCB(PFV_candidate - 1.05 * PFV_no_control) <= 100 m3",
        "objective": "minimize_TFV_subject_to_PFV_budget",
        "engineering_gate_policy": "minimal_physical_validity_only",
        "K_rate_ramp_dwell_interlock_role": "diagnostic_only",
        "event_count": proposed_event_count,
        "PFV_event_pass_count": pfv_pass_count,
        "decision_count": total_decisions,
        "active_nonfallback_decision_count": nonfallback_decisions,
        "mean_fallback_rate": mean_fallback,
        "target_write_readback_pass": write_pass,
        "causal_sparse_GAT_online_pass": causal_pass,
        "mean_delta_TFV_vs_internal_m3": mean_delta_tfv,
        "per_event": rows,
    }


def _core_worker(task: dict[str, Any]) -> dict[str, Any]:
    """Run one isolated event/strategy; the parent alone writes the ledger."""
    from sewerrtc.v4.v42_formal_runtime import FormalEventInput

    event = FormalEventInput(
        role=str(task["role"]),
        event_id=str(task["event_id"]),
        rainfall_sha256=str(task["rainfall_sha256"]),
        inp_path=Path(str(task["inp_path"])),
        rain_duration_min=int(task["rain_duration_min"]),
        simulation_duration_min=int(task["simulation_duration_min"]),
    )
    project_root = Path(str(task["project_root"]))
    strategy = str(task["strategy"])
    output_dir = Path(str(task["output_dir"]))
    try:
        if strategy == "Proposed":
            apply_simple_rtc_contract()
            import scripts.run_v42_formal_production_f2 as production

            result = production.run_proposed_event(
                event,
                project_root=project_root,
                output_dir=output_dir,
                state_source="gat_sparse_reconstruction",
                device=str(task["device"]),
                max_candidate_sequences=int(task["max_candidate_sequences"]),
                internal_shadow_detail_path=task.get("internal_shadow_detail_path"),
            )
        else:
            from sewerrtc.v4.v42_formal_runtime_safe import run_baseline_event

            result = run_baseline_event(
                event,
                strategy=strategy,
                project_root=project_root,
                output_dir=output_dir,
            )
        return {"status": "pass", "result": result}
    except Exception as exc:
        return {
            "status": "fail",
            "error": f"{type(exc).__name__}: {exc}",
            "detail_path": str(output_dir / "detail.csv"),
        }


def _run_core_parallel(
    *,
    project_root: Path,
    role: str,
    strategies: list[str],
    device: str,
    max_candidate_sequences: int,
    workers: int,
    event_limit: int | None = None,
    pipeline_proposed: bool = True,
) -> list[dict[str, Any]]:
    import scripts.run_v42_formal_production_f2 as production

    orchestrator = production.orchestrator
    events = (
        _load_core_calibration_events(project_root)
        if role == "calibration"
        else orchestrator.load_formal_event_inputs(project_root, role=role)
    )
    if event_limit is not None:
        events = events[: max(1, int(event_limit))]
        print(
            f"[CORE] development event subset={len(events)} of full={12 if role == 'calibration' else 'role inventory'}",
            flush=True,
        )
    lock = orchestrator.policy_lock_payload(project_root)
    policy_sha = orchestrator._policy_sha(project_root)
    pending: list[dict[str, Any]] = []
    results: list[dict[str, Any]] = []
    for event in events:
        for strategy in strategies:
            state_source = (
                "gat_sparse_reconstruction"
                if strategy == "Proposed"
                else "swmm_native_or_rule_baseline"
            )
            model_sha = (
                f"{lock['gat_model_sha256']}:{lock['model_sha256']}"
                if strategy == "Proposed"
                else "none"
            )
            task = {
                "project_root": str(project_root),
                "role": role,
                "event_id": event.event_id,
                "rainfall_sha256": event.rainfall_sha256,
                "inp_path": str(event.inp_path),
                "rain_duration_min": event.rain_duration_min,
                "simulation_duration_min": event.simulation_duration_min,
                "strategy": strategy,
                "state_source": state_source,
                "device": device,
                "max_candidate_sequences": max_candidate_sequences,
                "model_sha256": model_sha,
                "policy_sha256": (
                    policy_sha
                    if strategy == "Proposed"
                    else orchestrator.sha256_json({"strategy": strategy, "formal": True})
                ),
                "output_dir": str(
                    orchestrator.FORMAL_ROOT
                    / "paper_execution"
                    / role
                    / event.event_id
                    / strategy
                    / state_source
                ),
            }
            if strategy == "Internal":
                task["detail_path"] = str(Path(str(task["output_dir"])) / "detail.csv")
            if orchestrator._ledger_reusable(
                event=event,
                role=role,
                strategy=strategy,
                state_source=state_source,
                model_sha256=model_sha,
                policy_sha256=str(task["policy_sha256"]),
            ):
                print(
                    f"[CORE] REUSE role={role} event={event.event_id} strategy={strategy}",
                    flush=True,
                )
                results.append(
                    orchestrator._read_json(Path(str(task["output_dir"])) / "run_result.json")
                )
            else:
                pending.append(task)

    if not pending:
        return results
    baseline_tasks = [x for x in pending if x["strategy"] != "Proposed"]
    # Internal is the only baseline needed as the Proposed causal shadow.
    # Queue all Internal tasks first so the single GPU worker can start events
    # as soon as possible; No-control/Hold continue filling the CPU pool.
    baseline_tasks.sort(
        key=lambda task: (task["strategy"] != "Internal", str(task["event_id"]))
    )
    proposed_tasks = [x for x in pending if x["strategy"] == "Proposed"]
    failures: list[str] = []
    internal_shadow_paths: dict[str, str] = {}

    def consume(future: Any, task: dict[str, Any]) -> None:
        payload = future.result()
        status = str(payload.get("status", "fail"))
        result = payload.get("result") if status == "pass" else None
        detail = Path(str(payload.get("detail_path", task["output_dir"] + "/detail.csv")))
        error = "" if status == "pass" else str(payload.get("error", "worker failed"))
        row = {
            "role": role,
            "event_id": task["event_id"],
            "rainfall_sha256": task["rainfall_sha256"],
            "strategy": task["strategy"],
            "state_source": task["state_source"],
            "status": status,
            "input_sha256": "",
            "model_sha256": task["model_sha256"],
            "policy_sha256": task["policy_sha256"],
            "detail_path": str(detail),
            "detail_sha256": orchestrator.sha256_file(detail) if detail.exists() else "",
            "runtime_sec": float(result.get("runtime_sec", 0.0)) if result else 0.0,
            "authority": "authoritative_swmm",
            "error": error,
        }
        row["input_sha256"] = orchestrator.sha256_file(Path(str(task["inp_path"])))
        orchestrator.append_csv(orchestrator.LEDGER, row)
        if status != "pass" or not isinstance(result, dict):
            failures.append(f"{task['event_id']} {task['strategy']}: {error or 'missing result'}")
            return
        if task["strategy"] == "Internal":
            internal_shadow_paths[task["event_id"]] = str(detail)
        if task["strategy"] in {"No-control", "All-close"}:
            orchestrator._validate_explicit_baseline(project_root, result, task["strategy"])
        results.append(result)
        print(
            f"[CORE] PASS role={role} event={task['event_id']} strategy={task['strategy']}",
            flush=True,
        )

    # The 16-job SWMM benchmark completed with zero failures and was 7.9%
    # faster than 8 workers on this machine. Keep the user-selectable cap at
    # 16. The GPU Proposed worker is pipelined behind each event's Internal
    # result, so CPU SWMM work and GPU inference overlap without duplicating
    # the native Internal shadow.
    max_baseline_workers = max(1, min(int(workers), 16, len(baseline_tasks) or 1))
    def expected_internal_shadow(task: dict[str, Any]) -> str:
        return str(
            Path(task["output_dir"]).parent.parent
            / "Internal"
            / "swmm_native_or_rule_baseline"
            / "detail.csv"
        )

    proposed_by_event = {str(task["event_id"]): task for task in proposed_tasks}
    proposed_submitted: set[str] = set()
    proposed_futures: dict[Any, dict[str, Any]] = {}
    baseline_started = time.perf_counter()
    proposed_started: float | None = None

    # Keep one GPU process.  It persists across events, so the model bundle
    # remains cached in that process while CPU SWMM workers continue.
    with ProcessPoolExecutor(max_workers=1) as proposed_pool:
        def submit_proposed(task: dict[str, Any]) -> None:
            nonlocal proposed_started
            event_id = str(task["event_id"])
            if event_id in proposed_submitted:
                return
            shadow = internal_shadow_paths.get(event_id) or expected_internal_shadow(task)
            if Path(shadow).exists():
                task["internal_shadow_detail_path"] = shadow
            proposed_futures[proposed_pool.submit(_core_worker, task)] = task
            proposed_submitted.add(event_id)
            if proposed_started is None:
                proposed_started = time.perf_counter()
                print(
                    f"[CORE] Proposed GPU pipeline started workers=1 event={event_id}",
                    flush=True,
                )

        # If Internal is already reusable, or was not requested, start Proposed
        # immediately. Pending Internal events are submitted as soon as their
        # authoritative detail becomes available below.
        pending_internal_events = {
            str(task["event_id"])
            for task in baseline_tasks
            if task["strategy"] == "Internal"
        }
        if pipeline_proposed:
            for task in proposed_tasks:
                event_id = str(task["event_id"])
                if event_id not in pending_internal_events:
                    submit_proposed(task)

        with ProcessPoolExecutor(max_workers=max_baseline_workers) as baseline_pool:
            print(
                f"[CORE] CPU baseline pipeline started workers={max_baseline_workers} tasks={len(baseline_tasks)}",
                flush=True,
            )
            future_map = {
                baseline_pool.submit(_core_worker, task): task
                for task in baseline_tasks
            }
            for future in as_completed(future_map):
                task = future_map[future]
                consume(future, task)
                if pipeline_proposed and task["strategy"] == "Internal":
                    proposed = proposed_by_event.get(str(task["event_id"]))
                    if proposed is not None:
                        submit_proposed(proposed)

        print(
            f"[CORE] CPU baseline pipeline done elapsed_sec={time.perf_counter() - baseline_started:.1f}",
            flush=True,
        )

        # With no Internal task (or after an Internal failure), allow the
        # Proposed worker to fall back to its own causal Internal shadow. This
        # preserves the existing fail-closed error handling while avoiding a
        # deadlock in partial/resume runs.
        for task in proposed_tasks:
            submit_proposed(task)

        for future in as_completed(proposed_futures):
            consume(future, proposed_futures[future])
        if proposed_started is not None:
            print(
                f"[CORE] Proposed GPU pipeline done elapsed_sec={time.perf_counter() - proposed_started:.1f}",
                flush=True,
            )
    if failures:
        raise RuntimeError("core RTC worker failure: " + failures[0])
    return results


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
    )
    ap.add_argument(
        "--role",
        choices=("calibration", "challenge", "locked_validation", "formal_blind"),
        default="calibration",
    )
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--max-candidate-sequences", type=int, default=64)
    ap.add_argument(
        "--workers",
        type=int,
        default=16,
        help="CPU baseline workers (measured safe cap=16); Proposed remains one GPU worker",
    )
    ap.add_argument(
        "--strategies",
        default=",".join(DEFAULT_STRATEGIES),
        help="comma-separated strategies; core default is Proposed,No-control,Internal,Hold",
    )
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument(
        "--run-root",
        type=Path,
        default=None,
        help="isolated Core output root; preserves earlier runs when supplied",
    )
    ap.add_argument(
        "--event-limit",
        type=int,
        default=None,
        help="development-only prefix of resolved events for a performance benchmark",
    )
    ap.add_argument(
        "--pipeline-proposed",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="overlap one persistent GPU Proposed worker with CPU SWMM baselines",
    )
    args = ap.parse_args()

    root = args.project_root.resolve()
    apply_simple_rtc_contract()
    import scripts.run_v42_formal_production_f2 as production

    orchestrator = production.orchestrator
    strategies = [
        item.strip() for item in str(args.strategies).split(",") if item.strip()
    ]
    if "Proposed" not in strategies:
        raise ValueError("core RTC run must include Proposed")

    legacy_loader = orchestrator.load_formal_event_inputs

    def _load_core_events(project_root: Path, *, role: str):
        if role == "calibration":
            return _load_core_calibration_events(project_root)
        return legacy_loader(project_root, role=role)

    # The Fresh Calibration case manifest has one row per candidate branch;
    # core RTC needs one authoritative SWMM input per event.
    orchestrator.load_formal_event_inputs = _load_core_events
    # Set runner roots exactly as the Formal orchestrator expects, but do not
    # invoke the legacy Stage18 engineering gate or candidate-lineage blocker.
    orchestrator.OUTPUT_ROOT = root / "outputs/project6_dual_reference_v4/final_v4"
    orchestrator.PAPER_ROOT = orchestrator.OUTPUT_ROOT / "v42_paper"
    orchestrator.FORMAL_ROOT = (
        args.run_root.resolve()
        if args.run_root is not None
        else orchestrator.PAPER_ROOT / "core_rtc"
    )
    orchestrator.LEDGER = (
        orchestrator.FORMAL_ROOT / "FORMAL_EXECUTION_LEDGER.csv"
    )

    results = _run_core_parallel(
        project_root=root,
        role=args.role,
        strategies=strategies,
        device=args.device,
        max_candidate_sequences=int(args.max_candidate_sequences),
        workers=max(1, min(int(args.workers), 16)),
        event_limit=args.event_limit,
        pipeline_proposed=bool(args.pipeline_proposed),
    )
    evidence = _summarize(results)
    evidence["role"] = args.role
    evidence["strategies"] = strategies
    output = args.output or (
        orchestrator.PAPER_ROOT
        / "core_rtc"
        / f"{args.role}_CORE_RTC_EVIDENCE.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(
        json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False),
        flush=True,
    )
    return 0 if evidence["status"] == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
