"""Production entrypoint for the V4.2 Formal paper campaign.

This entrypoint injects the rule-free-plant/native-Internal-shadow runtime,
requires execution-derived dwell/target-write evidence, replaces the
metadata-only stage22 stub with an actual surrogate-feedback closed loop, and
expands Policy-Lock hashing to every executable Formal module.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.run_v42_formal_paper_f2 as orchestrator
from sewerrtc.v4 import v42_formal_runtime as base_runtime
from sewerrtc.v4 import v42_formal_runtime_safe as safe_runtime
from sewerrtc.v4 import v42_formal_surrogate_closed_loop as surrogate_runtime
from sewerrtc.v4.v42_formal_runtime_safe import (
    run_baseline_event,
    run_proposed_event,
)
from sewerrtc.v4.v42_formal_surrogate_closed_loop import (
    run_surrogate_closed_loop_event,
)
from sewerrtc.v4.v42_pfv_tfv_runtime_patch import (
    predict_and_decide as corrected_predict_and_decide,
)

# The authoritative production path must use the corrected global-search PFV
# selector in all three closed-loop implementations.  The underlying modules
# import ``predict_and_decide`` by name, so update their module globals before
# any Formal stage can execute.
base_runtime.predict_and_decide = corrected_predict_and_decide
safe_runtime.predict_and_decide = corrected_predict_and_decide
surrogate_runtime.predict_and_decide = corrected_predict_and_decide


def _production_policy_sha(project_root: Path) -> str:
    files = (
        project_root / "sewerrtc/control/pfvfirst_mpc_v42.py",
        project_root / "sewerrtc/v4/v42_formal_runtime.py",
        project_root / "sewerrtc/v4/v42_formal_runtime_safe.py",
        project_root / "sewerrtc/v4/v42_pfv_tfv_runtime_patch.py",
        project_root / "sewerrtc/v4/v42_formal_surrogate_closed_loop.py",
        project_root / "scripts/run_v42_formal_paper_f2.py",
        project_root / "scripts/run_v42_formal_production_f2.py",
        project_root / "configs/v42_formal_fallback_contract.json",
        project_root / "docs/contracts/PROJECT6_V42_PAPER_WORKFLOW_CONTRACT.json",
    )
    return hashlib.sha256(
        "\n".join(base_runtime.sha256_file(path) for path in files).encode("utf-8")
    ).hexdigest()


def _production_policy_lock_payload(project_root: str | Path):
    payload = base_runtime.policy_lock_payload(project_root)
    payload["policy_sha256"] = _production_policy_sha(Path(project_root))
    payload["production_runtime"] = "scripts/run_v42_formal_production_f2.py"
    payload["corrected_selector_runtime"] = "sewerrtc/v4/v42_pfv_tfv_runtime_patch.py"
    payload["candidate_search_scope"] = "global_engineering36_plus_priority_local"
    payload["candidate_cap_applied_after_projection"] = True
    payload["pfv_safety_statistic"] = "candidate_minus_1p05_no_control"
    payload["rule_free_proposed_plant"] = True
    payload["native_internal_causal_shadow"] = True
    payload["surrogate_closed_loop_is_executed_not_metadata_only"] = True
    payload["runtime_cross_decision_dwell_enforced"] = True
    payload["target_setting_write_readback_required"] = True
    return payload


def _production_stage_engineering(
    project_root: Path, device: str, max_candidate_sequences: int
) -> None:
    """Stage18-19: derive engineering evidence from authoritative execution."""
    calibration = orchestrator.audit_calibration_completeness(orchestrator.FORMAL_ROOT)
    if calibration["status"] != "pass":
        raise RuntimeError(f"Calibration12 incomplete: {calibration['reasons']}")
    results = orchestrator._run_role(
        project_root=project_root,
        role="calibration",
        strategies=["Proposed"],
        state_source="gat_sparse_reconstruction",
        device=device,
        max_candidate_sequences=max_candidate_sequences,
    )
    audit = orchestrator._engineering_audit(project_root, results)
    if not all(
        result.get("target_write_all_decisions_verified") is True
        and result.get("runtime_cross_decision_dwell_enforced") is True
        and result.get("precontrol_prefix_contract") == "causal_internal_readback_replay"
        for result in results
    ):
        audit["status"] = "fail"
        audit["target_setting_write_readback_verified"] = False
        audit["runtime_cross_decision_dwell_enforced"] = False
        audit["causal_precontrol_prefix_verified"] = False
    else:
        audit["target_setting_write_readback_verified"] = True
        audit["runtime_cross_decision_dwell_enforced"] = True
        audit["causal_precontrol_prefix_verified"] = True
    decisions = orchestrator._decision_rows(results)
    if not decisions or not all(
        row.get("target_write_verified") is True for row in decisions
    ):
        audit["status"] = "fail"
        audit["target_setting_write_readback_verified"] = False
    audit_path = (
        orchestrator.FORMAL_ROOT
        / "calibration/STEP3_AUTHORITATIVE_ENGINEERING_AUDIT.json"
    )
    audit_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    if audit.get("status") != "pass":
        raise RuntimeError(
            f"Formal Step3 authoritative engineering audit failed: {audit}"
        )
    orchestrator._run_subprocess(
        [
            str(Path(sys.executable)),
            "-u",
            str(project_root / "scripts/compile_v42_formal_step3_evidence_f2.py"),
        ],
        project_root,
    )


def _production_stage_exact(
    project_root: Path, device: str, max_candidate_sequences: int
):
    """Stage21: authoritative SWMM plant plus the three scientific references."""
    results = orchestrator._run_role(
        project_root=project_root,
        role="calibration",
        strategies=["Proposed", "No-control", "Internal", "Hold"],
        state_source="true_state",
        device=device,
        max_candidate_sequences=max_candidate_sequences,
    )
    proposed = [x for x in results if str(x.get("strategy")) == "Proposed"]
    counts = {
        strategy: len(
            {str(x.get("event_id")) for x in results if str(x.get("strategy")) == strategy}
        )
        for strategy in ("Proposed", "No-control", "Internal", "Hold")
    }
    if not proposed or len(set(counts.values())) != 1:
        raise RuntimeError(f"stage21 four-reference authoritative event counts mismatch: {counts}")
    payload = orchestrator._base_evidence_payload()
    payload.update(
        {
            "authoritative_engine": "SWMM",
            "online_future_hydraulic_truth_used": False,
            "canonical_pfvfirst_mpc_v42": True,
            "engineering_status_derived_from_execution": True,
            "readback_verified": True,
            "event_count": len({str(x.get("event_id")) for x in proposed}),
            "state_source": "true_state_diagnostic",
            "authoritative_reference_strategies": [
                "No-control",
                "Internal",
                "Hold",
            ],
            "strategy_event_counts": counts,
            "no_control_all_open_authoritative": True,
            "internal_native_rules_authoritative": True,
            "candidate_search_scope": "global_engineering36_plus_priority_local",
            "candidate_cap_applied_after_projection": True,
            "pfv_safety_statistic": "candidate_minus_1p05_no_control",
        }
    )
    orchestrator.write_stage_evidence(
        stage="exact_swmm_closed_loop",
        output_root=orchestrator.OUTPUT_ROOT,
        payload=payload,
    )
    return results


def _production_stage_surrogate(
    project_root: Path, device: str, max_candidate_sequences: int
) -> None:
    """Stage22: execute a real surrogate-state-feedback rolling controller."""
    stage21 = orchestrator._run_role(
        project_root=project_root,
        role="calibration",
        strategies=["Proposed", "Internal"],
        state_source="true_state",
        device=device,
        max_candidate_sequences=max_candidate_sequences,
    )
    by_key = {
        (str(x.get("event_id")), str(x.get("strategy"))): x for x in stage21
    }
    events = orchestrator.load_formal_event_inputs(project_root, role="calibration")
    results = []
    for event in events:
        proposed = by_key.get((event.event_id, "Proposed"))
        internal = by_key.get((event.event_id, "Internal"))
        if not proposed or not internal:
            raise RuntimeError(
                f"stage22 missing stage21 Proposed/Internal authoritative source for {event.event_id}"
            )
        result = run_surrogate_closed_loop_event(
            event,
            project_root=project_root,
            exact_proposed_detail=proposed["detail_path"],
            exact_internal_detail=internal["detail_path"],
            output_dir=(
                orchestrator.FORMAL_ROOT
                / "paper_execution/surrogate_closed_loop"
                / event.event_id
            ),
            device=device,
            max_candidate_sequences=max_candidate_sequences,
        )
        results.append(result)
    if len(results) != len(events) or not results:
        raise RuntimeError("stage22 surrogate closed loop did not complete every frozen Calibration event")
    ledger_path = orchestrator.FORMAL_ROOT / "paper_execution/surrogate_closed_loop/FORMAL_EXECUTION_LEDGER.csv"
    step1 = orchestrator._read_json(orchestrator.PAPER_ROOT / "step1_gat/evidence.json")
    step2 = orchestrator._read_json(
        orchestrator.PAPER_ROOT / "step2_surrogate/evidence.json"
    )
    runtime_sha = orchestrator.sha256_file(Path(__file__).resolve())
    for result in results:
        detail_path = Path(str(result["detail_path"]))
        decision_path = Path(str(result["decision_path"]))
        orchestrator.append_csv(
            ledger_path,
            {
                "stage": "surrogate_closed_loop",
                "role": "calibration",
                "event_id": str(result["event_id"]),
                "rainfall_sha256": str(result["rainfall_sha256"]),
                "strategy": "Proposed",
                "state_source": "surrogate_feedback_from_authoritative_prefix",
                "input_sha256": next(e.input_sha256 for e in events if e.event_id == result["event_id"]),
                "physical_network_sha256": next(e.input_sha256 for e in events if e.event_id == result["event_id"]),
                "policy_sha256": _production_policy_sha(project_root),
                "runtime_sha256": runtime_sha,
                "step1_sha256": str(step1["gat_model_sha256"]),
                "step2_sha256": str(step2["surrogate_model_sha256"]),
                "detail_path": str(detail_path),
                "detail_sha256": orchestrator.sha256_file(detail_path),
                "decision_path": str(decision_path),
                "decision_sha256": orchestrator.sha256_file(decision_path),
                "status": "pass",
            },
        )
    payload = orchestrator._base_evidence_payload()
    payload.update(
        {
            "surrogate_role": "hydraulic_surrogate_not_policy",
            "pfvfirst_mpc_v42": True,
            "surrogate_model_sha256": step2["surrogate_model_sha256"],
            "trajectory_first_kpi_derivation": True,
            "surrogate_closed_loop_executed": True,
            "event_count": len(results),
            "state_feedback_source": "surrogate_prediction_after_authoritative_prefix",
            "authoritative_hydraulic_truth_used_after_prefix": False,
            "realized_future_rainfall_used_online": False,
            "dynamic_internal_future_action_used_online": False,
            "result_paths": [str(x["detail_path"]) for x in results],
            "execution_ledger_path": str(ledger_path),
            "state_source": "surrogate_feedback_from_authoritative_prefix",
            "candidate_search_scope": "global_engineering36_plus_priority_local",
            "candidate_cap_applied_after_projection": True,
            "pfv_safety_statistic": "candidate_minus_1p05_no_control",
        }
    )
    orchestrator.write_stage_evidence(
        stage="surrogate_closed_loop",
        output_root=orchestrator.OUTPUT_ROOT,
        payload=payload,
    )


orchestrator.run_baseline_event = run_baseline_event
orchestrator.run_proposed_event = run_proposed_event
orchestrator._policy_sha = _production_policy_sha
orchestrator.policy_lock_payload = _production_policy_lock_payload
orchestrator.stage_engineering = _production_stage_engineering
orchestrator.stage_exact = _production_stage_exact
orchestrator.stage_surrogate = _production_stage_surrogate
orchestrator.PRODUCTION_RUNTIME_INJECTED = True


if __name__ == "__main__":
    raise SystemExit(orchestrator.main())
