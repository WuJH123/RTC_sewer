from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.compile_v42_formal_training_evidence_f2 import (
    _assert_full_step2_target_supervision,
)
from sewerrtc.v4.paper_workflow_v42 import (
    CAUSAL_HISTORY_CONTRACT,
    PAPER_STAGE_ORDER,
    REQUIRED_FORMAL_BLIND_STRATEGIES,
    audit_paper_workflow,
    write_stage_evidence,
)


def _lock_fields() -> dict:
    return {
        "policy_sha256": "policy",
        "model_sha256": "model",
        "gat_model_sha256": "gat",
        "fallback_contract_sha256": "fallback",
    }


def _payload(stage: str) -> dict:
    base = {
        "status": "pass",
        "development_evidence_substituted": False,
        "legacy_locked_evidence_substituted": False,
    }
    if stage == "true_state_offline_validation":
        base.update(
            state_source="true_state",
            four_reference_surrogate=True,
            trajectory_first_kpi_derivation=True,
            training_admission_authorized=True,
            raw_independent_oracle_all_pass=True,
            surrogate_model_sha256="model",
        )
    elif stage == "exact_swmm_closed_loop":
        base.update(
            authoritative_engine="SWMM",
            online_future_hydraulic_truth_used=False,
            canonical_pfvfirst_mpc_v42=True,
            engineering_status_derived_from_execution=True,
            readback_verified=True,
        )
    elif stage == "surrogate_closed_loop":
        base.update(
            surrogate_role="hydraulic_surrogate_not_policy",
            pfvfirst_mpc_v42=True,
            surrogate_model_sha256="model",
        )
    elif stage == "gat_integrated_closed_loop":
        base.update(
            state_source="gat_sparse_reconstruction",
            reconstructor_contract="formal_temporal_v42",
            reconstructed_history_contract=CAUSAL_HISTORY_CONTRACT,
            reconstructed_history_ready_before_mpc=True,
            authoritative_swmm_history_used_as_online_input=False,
            current_frame_repetition_used=False,
            gat_uncertainty_used=True,
            ood_gate_used=True,
            uncertainty_calibrated=True,
            ood_calibrated=True,
            gat_model_sha256="gat",
            surrogate_model_sha256="model",
        )
    elif stage == "policy_lock":
        base.update(**_lock_fields(), post_lock_parameter_updates_allowed=False)
    elif stage == "challenge":
        base.update(
            **_lock_fields(),
            policy_locked_before_reveal=True,
            used_for_retraining=False,
            rainfall_sha256s=["challenge-rain"],
        )
    elif stage == "locked_validation":
        base.update(
            **_lock_fields(),
            event_count=16,
            policy_locked_before_reveal=True,
            new_rainfall_sha_only=True,
            post_reveal_exclusion_used=False,
            used_for_retraining=False,
            rainfall_sha256s=[f"locked-{i}" for i in range(16)],
            revealed_rainfall_sha256s=["train-a", "challenge-rain"],
            revealed_rainfall_overlap_count=0,
        )
    elif stage == "formal_blind":
        base.update(
            **_lock_fields(),
            event_count=24,
            policy_locked_before_reveal=True,
            new_rainfall_sha_only=True,
            post_reveal_exclusion_used=False,
            used_for_retraining=False,
            rainfall_sha256s=[f"blind-{i}" for i in range(24)],
            revealed_rainfall_sha256s=["train-a", "challenge-rain", "locked-0"],
            revealed_rainfall_overlap_count=0,
            strategy_authority={name: "authoritative_swmm" for name in REQUIRED_FORMAL_BLIND_STRATEGIES},
            strategy_event_counts={name: 24 for name in REQUIRED_FORMAL_BLIND_STRATEGIES},
        )
    return base


def _write_through(tmp_path: Path, last_stage: str) -> None:
    for stage in PAPER_STAGE_ORDER:
        write_stage_evidence(stage=stage, output_root=tmp_path, payload=_payload(stage))
        if stage == last_stage:
            return


def test_locked_validation_is_mandatory_between_challenge_and_formal_blind() -> None:
    assert PAPER_STAGE_ORDER[-3:] == ("challenge", "locked_validation", "formal_blind")


def test_locked_validation_is_one_shot_and_policy_locked(tmp_path: Path) -> None:
    _write_through(tmp_path, "challenge")
    bad = _payload("locked_validation")
    bad["policy_locked_before_reveal"] = False
    write_stage_evidence(stage="locked_validation", output_root=tmp_path, payload=bad)
    audit = audit_paper_workflow(tmp_path)
    assert audit.complete is False
    assert audit.next_stage == "locked_validation"
    assert "locked_revealed_before_policy_lock" in audit.stage_audits[-1].reasons


def test_formal_blind_requires_all_authoritative_swmm_strategies(tmp_path: Path) -> None:
    _write_through(tmp_path, "locked_validation")
    bad = _payload("formal_blind")
    bad["strategy_authority"]["EFD"] = "surrogate_proxy"
    write_stage_evidence(stage="formal_blind", output_root=tmp_path, payload=bad)
    audit = audit_paper_workflow(tmp_path)
    assert audit.complete is False
    assert audit.next_stage == "formal_blind"
    assert "formal_strategy_not_authoritative_swmm:EFD" in audit.stage_audits[-1].reasons


def test_formal_blind_requires_every_strategy_on_every_event(tmp_path: Path) -> None:
    _write_through(tmp_path, "locked_validation")
    bad = _payload("formal_blind")
    bad["strategy_event_counts"]["Auto-RBC"] = 23
    write_stage_evidence(stage="formal_blind", output_root=tmp_path, payload=bad)
    audit = audit_paper_workflow(tmp_path)
    assert audit.complete is False
    assert "formal_strategy_event_count_mismatch:Auto-RBC" in audit.stage_audits[-1].reasons


def test_formal_workflow_passes_with_locked_and_complete_authoritative_blind(tmp_path: Path) -> None:
    _write_through(tmp_path, "formal_blind")
    audit = audit_paper_workflow(tmp_path)
    assert audit.complete is True
    assert audit.passed_through == "formal_blind"


def test_formal_training_evidence_rejects_missing_full_hydraulic_supervision() -> None:
    reports = [
        {
            "seed": 17,
            "storage_supervised": True,
            "facility_flow_supervised": True,
            "outfall_supervised": False,
        },
        {
            "seed": 42,
            "storage_supervised": True,
            "facility_flow_supervised": True,
            "outfall_supervised": True,
        },
        {
            "seed": 73,
            "storage_supervised": True,
            "facility_flow_supervised": True,
            "outfall_supervised": True,
        },
    ]
    with pytest.raises(RuntimeError, match="outfall_supervised"):
        _assert_full_step2_target_supervision(reports)


def test_formal_training_evidence_accepts_full_hydraulic_supervision() -> None:
    reports = [
        {
            "seed": seed,
            "storage_supervised": True,
            "facility_flow_supervised": True,
            "outfall_supervised": True,
        }
        for seed in (17, 42, 73)
    ]
    _assert_full_step2_target_supervision(reports)
