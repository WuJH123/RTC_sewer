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
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sewerrtc.v4.v42_simple_rtc_contract import (
    CONTRACT_ID,
    apply_simple_rtc_contract,
)

# Apply the simplified semantics before importing the production entrypoint.
apply_simple_rtc_contract()
import scripts.run_v42_formal_production_f2 as production  # noqa: E402
from sewerrtc.v4.v42_formal_runtime import FormalEventInput  # noqa: E402

# Re-apply after production import in case a legacy module rebound a global.
apply_simple_rtc_contract()

orchestrator = production.orchestrator


DEFAULT_STRATEGIES = ("Proposed", "No-control", "Internal", "Hold")


def _load_core_calibration_events(project_root: Path) -> list[FormalEventInput]:
    """Resolve Fresh Calibration12's three branch rows into 12 SWMM events."""
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


def _fresh_pfv_calibration_path(project_root: Path) -> Path:
    path = (
        project_root
        / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2"
        / "pfv_only_v2/FRESH_PFV_ONLY_SAFETY_CALIBRATION.json"
    )
    if not path.exists():
        raise FileNotFoundError(path)
    return path


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
        "--strategies",
        default=",".join(DEFAULT_STRATEGIES),
        help="comma-separated strategies; core default is Proposed,No-control,Internal,Hold",
    )
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args()

    root = args.project_root.resolve()
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
    legacy_proposed = orchestrator.run_proposed_event

    def _run_core_proposed(*args, **kwargs):
        kwargs.setdefault("step2_calibration_path", _fresh_pfv_calibration_path(root))
        return legacy_proposed(*args, **kwargs)

    orchestrator.run_proposed_event = _run_core_proposed

    # Set runner roots exactly as the Formal orchestrator expects, but do not
    # invoke the legacy Stage18 engineering gate or candidate-lineage blocker.
    orchestrator.OUTPUT_ROOT = root / "outputs/project6_dual_reference_v4/final_v4"
    orchestrator.PAPER_ROOT = orchestrator.OUTPUT_ROOT / "v42_paper"
    orchestrator.FORMAL_ROOT = orchestrator.PAPER_ROOT / "core_rtc"
    orchestrator.LEDGER = (
        orchestrator.FORMAL_ROOT / "FORMAL_EXECUTION_LEDGER.csv"
    )

    results = orchestrator._run_role(
        project_root=root,
        role=args.role,
        strategies=strategies,
        state_source="gat_sparse_reconstruction",
        device=args.device,
        max_candidate_sequences=int(args.max_candidate_sequences),
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
