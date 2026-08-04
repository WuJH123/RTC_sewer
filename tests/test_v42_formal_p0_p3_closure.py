from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pandas as pd

from sewerrtc.v4.v42_formal_strict import (
    audit_calibration_completeness,
    audit_closed_loop_execution_strict,
    audit_step2_evidence_strict,
    audit_step3_evidence_strict,
)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_calibration_strict_requires_exact_frozen_12(tmp_path: Path):
    formal = tmp_path / "formal_f2"
    groups = [f"rain-{i:02d}" for i in range(12)]
    ledger = pd.DataFrame(
        {
            "formal_f2_role": ["calibration"] * 12,
            "rainfall_group_key": groups,
        }
    )
    (formal / "prepare").mkdir(parents=True)
    ledger.to_csv(formal / "prepare/FORMAL_F2_EVENT_LEDGER.csv", index=False)
    for name in (
        "STEP1_UNCERTAINTY_OOD_CALIBRATION.json",
        "PFV_ONLY_SAFETY_CALIBRATION.json",
    ):
        _write_json(
            formal / "calibration" / name,
            {
                "status": "pass",
                "calibration_rainfall_groups": groups,
                "calibration_rainfall_group_count": 12,
            },
        )
    assert audit_calibration_completeness(formal)["status"] == "pass"

    _write_json(
        formal / "calibration/PFV_ONLY_SAFETY_CALIBRATION.json",
        {
            "status": "pass",
            "calibration_rainfall_groups": groups[:8],
            "calibration_rainfall_group_count": 8,
        },
    )
    result = audit_calibration_completeness(formal)
    assert result["status"] == "fail"
    assert any("does_not_equal_frozen_plan" in reason for reason in result["reasons"])


def test_control_core_strict_does_not_require_outfall(tmp_path: Path):
    paper = tmp_path / "v42_paper"
    base = {
        "step2_target_contract": "CONTROL_CORE",
        "control_core_target_coverage_complete": True,
        "storage_supervised": True,
        "facility_flow_supervised": True,
        "outfall_supervised": False,
        "outfall_claim_authorized": False,
        "no_control_all_open_verified": True,
        "trajectory_first_kpi_derivation": True,
        "peak_is_hard_safety_constraint": False,
        "control_objective_contract": "PROJECT6_V42_PFV_ONLY_TFV_MIN_MPC_V2",
        "pfv_budget_applied": True,
        "objective": "minimize_TFV_subject_to_PFV_budget",
        "priority_depth_hard_gate": False,
        "global_peak_objective_term": False,
        "independent_OOD_gate": False,
        "independent_uncertainty_gate": False,
    }
    _write_json(paper / "step2_surrogate/evidence.json", base)
    assert audit_step2_evidence_strict(paper)["status"] == "pass"

    full = dict(base)
    full["step2_target_contract"] = "FULL_HYDRAULIC"
    _write_json(paper / "step2_surrogate/evidence.json", full)
    result = audit_step2_evidence_strict(paper)
    assert result["status"] == "fail"
    assert "full_hydraulic_outfall_supervision_not_proven" in result["reasons"]


def test_step3_strict_requires_pfv_only_objective_and_peak_reporting(tmp_path: Path):
    paper = tmp_path / "v42_paper"
    payload = {
        "selector": "decide_pfvfirst_mpc",
        "control_objective_contract": "PROJECT6_V42_PFV_ONLY_TFV_MIN_MPC_V2",
        "pfv_reference": "no_control",
        "pfv_absolute_allowance_m3": 100.0,
        "pfv_relative_allowance_fraction": 0.05,
        "pfv_budget_applied": True,
        "objective": "minimize_TFV_subject_to_PFV_budget",
        "priority_depth_hard_gate": False,
        "tfv_reference": "dynamic_internal",
        "tfv_is_primary_performance_objective": True,
        "tfv_is_hard_safety_constraint": False,
        "global_peak_objective_term": False,
        "peak_is_hard_safety_constraint": False,
        "peak_penalty_weight": 0.0,
        "action_penalty_weight": 0.0,
        "terminal_penalty_weight": 0.0,
        "uncertainty_penalty_weight": 0.0,
        "independent_OOD_gate": False,
        "independent_uncertainty_gate": False,
        "facility_count": 36,
        "max_changed_facilities": 8,
        "horizon_steps": 12,
        "engineering_status_derived_from_execution": True,
        "changed_facilities_derived_from_executed_action": True,
        "readback_verified": True,
        "uncertainty_and_ood_linked_to_calibrated_models": True,
    }
    _write_json(paper / "step3_mpc/evidence.json", payload)
    assert audit_step3_evidence_strict(paper)["status"] == "pass"
    payload["peak_is_hard_safety_constraint"] = True
    _write_json(paper / "step3_mpc/evidence.json", payload)
    result = audit_step3_evidence_strict(paper)
    assert result["status"] == "fail"
    assert "peak_must_not_be_hard_gate" in result["reasons"]


def test_closed_loop_strict_rejects_metadata_only_surrogate_stub(tmp_path: Path):
    paper = tmp_path / "v42_paper"
    _write_json(
        paper / "exact_closed_loop/evidence.json",
        {
            "status": "pass",
            "authoritative_engine": "SWMM",
            "canonical_pfvfirst_mpc_v42": True,
            "authoritative_reference_strategies": ["No-control", "Internal", "Hold"],
            "event_count": 12,
            "strategy_event_counts": {
                "Proposed": 12,
                "No-control": 12,
                "Internal": 12,
                "Hold": 12,
            },
            "no_control_all_open_authoritative": True,
            "internal_native_rules_authoritative": True,
        },
    )
    _write_json(
        paper / "surrogate_closed_loop/evidence.json",
        {
            "status": "pass",
            "surrogate_role": "hydraulic_surrogate_not_policy",
            "pfvfirst_mpc_v42": True,
            "event_count": 12,
            "surrogate_closed_loop_executed": False,
            "authoritative_hydraulic_truth_used_after_prefix": False,
            "realized_future_rainfall_used_online": False,
            "dynamic_internal_future_action_used_online": False,
        },
    )
    _write_json(
        paper / "gat_integrated_closed_loop/evidence.json",
        {
            "status": "pass",
            "state_source": "gat_sparse_reconstruction",
            "event_count": 12,
            "authoritative_swmm_outcome": True,
            "authoritative_swmm_history_used_as_online_input": False,
            "current_frame_repetition_used": False,
            "gat_uncertainty_used": True,
            "ood_gate_used": False,
            "ood_diagnostic_used": True,
        },
    )
    result = audit_closed_loop_execution_strict(paper)
    assert result["status"] == "fail"
    assert "stage22_metadata_only_stub_forbidden" in result["reasons"]

    surrogate = json.loads(
        (paper / "surrogate_closed_loop/evidence.json").read_text(encoding="utf-8")
    )
    surrogate["surrogate_closed_loop_executed"] = True
    ledger_dir = paper / "surrogate_closed_loop"
    rows = []
    for i in range(12):
        detail = ledger_dir / f"detail_{i}.csv"
        decision = ledger_dir / f"decision_{i}.jsonl"
        detail.write_text("elapsed_min\n120\n", encoding="utf-8")
        decision.write_text("{}\n", encoding="utf-8")
        sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
        rows.append(
            {
                "stage": "surrogate_closed_loop",
                "state_source": "surrogate_feedback_from_authoritative_prefix",
                "detail_path": str(detail),
                "detail_sha256": sha(detail),
                "decision_path": str(decision),
                "decision_sha256": sha(decision),
            }
        )
    ledger_path = ledger_dir / "FORMAL_EXECUTION_LEDGER.csv"
    pd.DataFrame(rows).to_csv(ledger_path, index=False)
    surrogate["execution_ledger_path"] = str(ledger_path)
    _write_json(paper / "surrogate_closed_loop/evidence.json", surrogate)
    assert audit_closed_loop_execution_strict(paper)["status"] == "pass"


def test_production_entrypoint_never_imports_qualification_controller():
    path = Path(__file__).resolve().parents[1] / "scripts/run_v42_formal_production_f2.py"
    text = path.read_text(encoding="utf-8")
    assert "run_v42_qualification" not in text
    assert "qualification_micro" not in text
    assert "v42_formal_runtime_safe" in text
    assert "v42_formal_surrogate_closed_loop" in text
    assert "orchestrator.stage_surrogate = _production_stage_surrogate" in text


def test_formal_orchestrator_contains_one_shot_locked_and_final_guards():
    path = Path(__file__).resolve().parents[1] / "scripts/run_v42_formal_paper_f2.py"
    text = path.read_text(encoding="utf-8")
    assert "Locked Validation is one-shot" in text
    assert "Final held-out test already has evidence" in text
    assert "FORMAL_STRATEGIES" in text


def test_formal_production_scripts_compile():
    root = Path(__file__).resolve().parents[1]
    paths = [
        root / "scripts/run_v42_formal_paper_f2.py",
        root / "scripts/run_v42_formal_production_f2.py",
        root / "scripts/run_v42_formal_calibration12_f2.py",
        root / "scripts/compile_v42_formal_training_evidence_strict_f2.py",
        root / "scripts/audit_v42_formal_strict_f2.py",
    ]
    for path in paths:
        compile(path.read_text(encoding="utf-8"), str(path), "exec")


def test_strict_evidence_entrypoint_mentions_calibration12_gate():
    root = Path(__file__).resolve().parents[1]
    text = (root / "scripts/compile_v42_formal_training_evidence_strict_f2.py").read_text(
        encoding="utf-8"
    )
    assert "audit_calibration_completeness" in text
    calibration = (root / "scripts/run_v42_formal_calibration12_f2.py").read_text(
        encoding="utf-8"
    )
    assert "FORMAL_F2_CALIBRATION12_GATE.json" in calibration
