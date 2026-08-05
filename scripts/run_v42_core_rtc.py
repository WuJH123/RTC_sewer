"""Run the Project6 V4.2 core RTC loop without legacy engineering hard gates.

This runner is intentionally narrow.  It exercises the paper's central online
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

from sewerrtc.v4.v42_simple_rtc_contract import (
    CONTRACT_ID,
    apply_simple_rtc_contract,
)

# Apply the simplified semantics before importing the production entrypoint.
apply_simple_rtc_contract()
import scripts.run_v42_formal_production_f2 as production  # noqa: E402

# Re-apply after production import in case a legacy module rebound a global.
apply_simple_rtc_contract()

orchestrator = production.orchestrator


DEFAULT_STRATEGIES = ("Proposed", "No-control", "Internal", "Hold")


def _event_map(results: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for result in results:
        event = str(result.get("event_id", ""))
        strategy = str(result.get("strategy", ""))
        if event and strategy:
            out.setdefault(event, {})[strategy] = result
    return out


def _summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_event = _event_map(results)
    rows: list[dict[str, Any]] = []
    pfv_pass_count = 0
    proposed_event_count = 0
    total_decisions = 0
    total_fallback_weighted = 0.0
    write_pass = True

    for event_id, strategies in sorted(by_event.items()):
        proposed = strategies.get("Proposed")
        no_control = strategies.get("No-control")
        internal = strategies.get("Internal")
        if proposed is None:
            continue
        proposed_event_count += 1
        total_decisions += int(proposed.get("decision_count", 0))
        total_fallback_weighted += float(proposed.get("fallback_rate", 1.0))
        write_pass = write_pass and bool(
            proposed.get("target_write_all_decisions_verified", False)
        )

        pk = proposed.get("kpis", {}) or {}
        nk = (no_control or {}).get("kpis", {}) or {}
        ik = (internal or {}).get("kpis", {}) or {}
        pfv_proposed = float(pk.get("PFV", np.nan))
        pfv_no_control = float(nk.get("PFV", np.nan))
        tfv_proposed = float(pk.get("TFV", np.nan))
        tfv_internal = float(ik.get("TFV", np.nan))
        budget = 100.0 + 1.05 * max(0.0, pfv_no_control) if np.isfinite(pfv_no_control) else np.nan
        pfv_pass = bool(
            np.isfinite(pfv_proposed)
            and np.isfinite(budget)
            and pfv_proposed <= budget + 1e-6
        )
        pfv_pass_count += int(pfv_pass)
        rows.append(
            {
                "event_id": event_id,
                "PFV_proposed_m3": pfv_proposed,
                "PFV_no_control_m3": pfv_no_control,
                "PFV_budget_m3": budget,
                "PFV_constraint_pass": pfv_pass,
                "TFV_proposed_m3": tfv_proposed,
                "TFV_internal_m3": tfv_internal,
                "delta_TFV_vs_internal_m3": (
                    tfv_proposed - tfv_internal
                    if np.isfinite(tfv_proposed) and np.isfinite(tfv_internal)
                    else np.nan
                ),
                "fallback_rate": float(proposed.get("fallback_rate", np.nan)),
                "decision_count": int(proposed.get("decision_count", 0)),
                "target_write_all_decisions_verified": bool(
                    proposed.get("target_write_all_decisions_verified", False)
                ),
            }
        )

    mean_delta_tfv = float(
        np.nanmean([row["delta_TFV_vs_internal_m3"] for row in rows])
    ) if rows else np.nan
    mean_fallback = (
        total_fallback_weighted / proposed_event_count if proposed_event_count else 1.0
    )
    status = "pass" if (
        proposed_event_count > 0
        and total_decisions > 0
        and write_pass
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
        "mean_fallback_rate": mean_fallback,
        "target_write_readback_pass": write_pass,
        "mean_delta_TFV_vs_internal_m3": mean_delta_tfv,
        "per_event": rows,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    ap.add_argument("--role", choices=("calibration", "challenge", "locked_validation", "formal_blind"), default="calibration")
    ap.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    ap.add_argument("--max-candidate-sequences", type=int, default=64)
    ap.add_argument(
        "--strategies",
        default=",".join(DEFAULT_STRATEGIES),
        help="comma-separated strategies; core default is Proposed,No-control,Internal,Hold",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=None,
    )
    args = ap.parse_args()

    root = args.project_root.resolve()
    strategies = [item.strip() for item in str(args.strategies).split(",") if item.strip()]
    if "Proposed" not in strategies:
        raise ValueError("core RTC run must include Proposed")

    # Set runner roots exactly as the Formal orchestrator expects, but do not
    # invoke the legacy Stage18 engineering gate or candidate-lineage blocker.
    orchestrator.OUTPUT_ROOT = root / "outputs/project6_dual_reference_v4/final_v4"
    orchestrator.FORMAL_ROOT = orchestrator.OUTPUT_ROOT / "v42_paper/formal_f2"
    orchestrator.PAPER_ROOT = orchestrator.OUTPUT_ROOT / "v42_paper"
    orchestrator.LEDGER = orchestrator.FORMAL_ROOT / "paper_execution/FORMAL_EXECUTION_LEDGER.csv"

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
        orchestrator.PAPER_ROOT / "core_rtc" / f"{args.role}_CORE_RTC_EVIDENCE.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(json.dumps(evidence, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    return 0 if evidence["status"] == "pass" else 3


if __name__ == "__main__":
    raise SystemExit(main())
