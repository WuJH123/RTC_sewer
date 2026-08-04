from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from sewerrtc.v4.paper_workflow_v42 import (
    CAUSAL_HISTORY_CONTRACT,
    PAPER_STAGE_ORDER,
    REQUIRED_FORMAL_BLIND_STRATEGIES,
    audit_paper_workflow,
    write_stage_evidence,
)
from sewerrtc.v4.v42_mainline_workflow import audit_v42_mainline
from sewerrtc.v4.v42_r0_strict import _action_sha_semantic_compatible


def test_optimized_action_hash_preserves_elapsed_semantics(tmp_path: Path):
    path = tmp_path / "detail.csv"
    facilities = ["A", "B"]
    frame = pd.DataFrame(
        {
            "elapsed_min": [0.0, 5.0, 10.0],
            "setting:A": [0.0, 1.0, 1.0],
            "setting:B": [1.0, 1.0, 0.0],
        }
    )
    frame.to_csv(path, index=False)
    disk = _action_sha_semantic_compatible(path, facilities)
    memory = _action_sha_semantic_compatible(path, facilities, _df=frame)
    assert disk == memory

    shifted = frame.copy()
    shifted["elapsed_min"] += 1.0
    assert _action_sha_semantic_compatible(path, facilities, _df=shifted) != disk


def _payload(stage: str) -> dict:
    p = {
        "status": "pass",
        "development_evidence_substituted": False,
        "legacy_locked_evidence_substituted": False,
    }
    if stage == "true_state_offline_validation":
        p.update(
            state_source="true_state",
            four_reference_surrogate=True,
            trajectory_first_kpi_derivation=True,
            training_admission_authorized=True,
            raw_independent_oracle_all_pass=True,
            surrogate_model_sha256="surrogate",
        )
    elif stage == "exact_swmm_closed_loop":
        p.update(
            authoritative_engine="SWMM",
            online_future_hydraulic_truth_used=False,
            canonical_pfvfirst_mpc_v42=True,
            engineering_status_derived_from_execution=True,
            readback_verified=True,
        )
    elif stage == "surrogate_closed_loop":
        p.update(
            surrogate_role="hydraulic_surrogate_not_policy",
            pfvfirst_mpc_v42=True,
            surrogate_model_sha256="surrogate",
        )
    elif stage == "gat_integrated_closed_loop":
        p.update(
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
            surrogate_model_sha256="surrogate",
        )
    elif stage == "policy_lock":
        p.update(
            policy_sha256="policy",
            model_sha256="surrogate",
            fallback_contract_sha256="fallback",
            gat_model_sha256="gat",
            post_lock_parameter_updates_allowed=False,
            control_objective_contract="PROJECT6_V42_PFV_BUDGETED_TFV_MPC_V1",
        )
    elif stage == "challenge":
        p.update(
            event_count=12,
            policy_locked_before_evaluation=True,
            current_generation_holdout_only=True,
            used_for_retraining=False,
            rainfall_sha256s=[f"challenge-rain-{i}" for i in range(12)],
            training_rainfall_sha256s=["train-a", "train-b"],
            training_rainfall_overlap_count=0,
            policy_sha256="policy",
            model_sha256="surrogate",
            gat_model_sha256="gat",
            fallback_contract_sha256="fallback",
        )
    elif stage == "locked_validation":
        p.update(
            event_count=16,
            policy_locked_before_evaluation=True,
            current_generation_holdout_only=True,
            new_rainfall_sha_only=True,
            post_reveal_exclusion_used=False,
            used_for_retraining=False,
            rainfall_sha256s=[f"locked-{i:02d}" for i in range(16)],
            revealed_rainfall_sha256s=["train-a", "train-b", "challenge-rain"],
            revealed_rainfall_overlap_count=0,
            training_rainfall_sha256s=["train-a", "train-b"],
            training_rainfall_overlap_count=0,
            policy_sha256="policy",
            model_sha256="surrogate",
            gat_model_sha256="gat",
            fallback_contract_sha256="fallback",
        )
    elif stage == "formal_blind":
        p.update(
            event_count=24,
            policy_locked_before_evaluation=True,
            current_generation_holdout_only=True,
            new_rainfall_sha_only=True,
            post_reveal_exclusion_used=False,
            used_for_retraining=False,
            rainfall_sha256s=[f"rain-{i:02d}" for i in range(24)],
            revealed_rainfall_sha256s=["train-a", "train-b", "challenge-rain", "locked-00"],
            revealed_rainfall_overlap_count=0,
            training_rainfall_sha256s=["train-a", "train-b"],
            training_rainfall_overlap_count=0,
            policy_sha256="policy",
            model_sha256="surrogate",
            gat_model_sha256="gat",
            fallback_contract_sha256="fallback",
            strategy_authority={name: "authoritative_swmm" for name in REQUIRED_FORMAL_BLIND_STRATEGIES},
            strategy_event_counts={name: 24 for name in REQUIRED_FORMAL_BLIND_STRATEGIES},
        )
    return p


def test_policy_lock_lineage_is_enforced_for_challenge_and_formal(tmp_path: Path):
    for stage in PAPER_STAGE_ORDER:
        write_stage_evidence(stage=stage, output_root=tmp_path, payload=_payload(stage))
    assert audit_paper_workflow(tmp_path).complete

    challenge = tmp_path / "v42_paper/challenge/evidence.json"
    p = json.loads(challenge.read_text(encoding="utf-8"))
    p["policy_sha256"] = "different"
    challenge.write_text(json.dumps(p), encoding="utf-8")
    audit = audit_paper_workflow(tmp_path)
    assert not audit.complete
    assert audit.next_stage == "challenge"
    assert "policy_sha256_does_not_match_policy_lock" in audit.stage_audits[-1].reasons


def test_changed_gat_cannot_pass_challenge_lineage(tmp_path: Path):
    for stage in PAPER_STAGE_ORDER:
        write_stage_evidence(stage=stage, output_root=tmp_path, payload=_payload(stage))
    challenge = tmp_path / "v42_paper/challenge/evidence.json"
    p = json.loads(challenge.read_text(encoding="utf-8"))
    p["gat_model_sha256"] = "different-gat"
    challenge.write_text(json.dumps(p), encoding="utf-8")
    audit = audit_paper_workflow(tmp_path)
    assert not audit.complete
    assert audit.next_stage == "challenge"
    assert "gat_model_sha256_does_not_match_policy_lock" in audit.stage_audits[-1].reasons


def test_formal_blind_requires_explicit_unique_rainfall_sha_list(tmp_path: Path):
    for stage in PAPER_STAGE_ORDER[:-1]:
        write_stage_evidence(stage=stage, output_root=tmp_path, payload=_payload(stage))
    bad = _payload("formal_blind")
    bad["rainfall_sha256s"] = ["same"] * 24
    write_stage_evidence(stage="formal_blind", output_root=tmp_path, payload=bad)
    audit = audit_paper_workflow(tmp_path)
    assert not audit.complete
    assert "formal_test_rainfall_sha_not_unique" in audit.stage_audits[-1].reasons


def test_end_to_end_mainline_stops_before_missing_r0(tmp_path: Path):
    audit = audit_v42_mainline(tmp_path)
    assert not audit.complete
    assert audit.next_stage == "phase_r0"
    assert audit.stages[0].stage == "phase_r0"
