"""Production entrypoint for the V4.2 Formal paper campaign.

This entrypoint injects the rule-free-plant/native-Internal-shadow runtime,
replaces the metadata-only stage22 stub with an actual surrogate-feedback closed
loop, and expands Policy-Lock hashing to every executable Formal module.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.run_v42_formal_paper_f2 as orchestrator
from sewerrtc.v4 import v42_formal_runtime as base_runtime
from sewerrtc.v4.v42_formal_runtime_safe import (
    run_baseline_event,
    run_proposed_event,
)
from sewerrtc.v4.v42_formal_surrogate_closed_loop import (
    run_surrogate_closed_loop_event,
)


def _production_policy_sha(project_root: Path) -> str:
    files = (
        project_root / "sewerrtc/control/pfvfirst_mpc_v42.py",
        project_root / "sewerrtc/v4/v42_formal_runtime.py",
        project_root / "sewerrtc/v4/v42_formal_runtime_safe.py",
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
    payload["rule_free_proposed_plant"] = True
    payload["native_internal_causal_shadow"] = True
    payload["surrogate_closed_loop_is_executed_not_metadata_only"] = True
    return payload


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
    step2 = orchestrator._read_json(
        orchestrator.PAPER_ROOT / "step2_surrogate/evidence.json"
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
        }
    )
    orchestrator.write_stage_evidence(
        stage="surrogate_closed_loop",
        output_root=orchestrator.OUTPUT_ROOT,
        payload=payload,
    )


# Replace low-level execution and scientific stage implementations. Split,
# one-shot, evidence-order and held-out lineage logic remains in the Formal
# orchestrator.
orchestrator.run_baseline_event = run_baseline_event
orchestrator.run_proposed_event = run_proposed_event
orchestrator._policy_sha = _production_policy_sha
orchestrator.policy_lock_payload = _production_policy_lock_payload
orchestrator.stage_exact = _production_stage_exact
orchestrator.stage_surrogate = _production_stage_surrogate


if __name__ == "__main__":
    raise SystemExit(orchestrator.main())
