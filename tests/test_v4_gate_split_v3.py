"""Gate split V3: DGA passes independently, MSG deferred, P3 verdict kept."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from sewerrtc.v4.train1600_v3 import (
    MODEL_SAFETY_DEFERRED_METRICS,
    build_p3_freeze_payload,
    evaluate_data_generation_authorization_v3,
    model_safety_gate_v3_status,
)
from train_v3_helpers import make_p3_evidence


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONTRACTS = PROJECT_ROOT / "docs" / "contracts"


def test_dga_passes_independently_and_preserves_underpowered_verdict() -> None:
    map_audit, gate_verdict, dataset_audit = make_p3_evidence()

    dga = evaluate_data_generation_authorization_v3(
        map_audit, gate_verdict, dataset_audit
    )

    assert dga["status"] == "pass"
    assert dga["scientific_pass"] is True
    assert dga["train1600_planning_authorized"] is True
    assert len(dga["conditions"]) == 12
    assert all(dga["conditions"].values())
    # The P3 model-safety verdict is quoted verbatim, never upgraded.
    assert dga["p3_verdict_preserved"] == "underpowered_validation"
    assert dga["p3_verdict_never_overwritten"] is True
    assert dga["no_joint_found_under_budget_kept_verbatim"] is True
    assert dga["values"]["robust_feasible_states"] == 9
    assert dga["values"]["fallback_only_states"] == 19
    assert dga["values"]["boundary_states"] == 4
    assert dga["values"]["candidate_generator_recall"] == 1.0
    json.dumps(dga, allow_nan=False)


def test_dga_blocks_when_any_frozen_condition_fails() -> None:
    map_audit, gate_verdict, dataset_audit = make_p3_evidence()
    map_audit["recall_report"]["candidate_generator_state_recall"] = 0.5

    dga = evaluate_data_generation_authorization_v3(
        map_audit, gate_verdict, dataset_audit
    )

    assert dga["status"] == "blocked"
    assert dga["scientific_pass"] is False
    assert dga["train1600_planning_authorized"] is False
    assert not dga["conditions"]["candidate_generator_recall_at_least_0p80"]


def test_model_safety_gate_is_deferred_and_never_gates_data_generation() -> None:
    msg = model_safety_gate_v3_status()

    assert msg["status"] == "deferred"
    assert msg["reason"] == "requires_train1600_and_powered_locked_validation"
    for metric in ("mcc", "auprc", "balanced_accuracy", "decision_regret"):
        assert metric in msg["deferred_metrics"]
    assert set(msg["controls"]) == {
        "policy_lock",
        "surrogate_closed_loop",
        "challenge",
        "formal_blind",
    }
    assert msg["does_not_control"] == ["train1600_data_generation"]
    assert msg["insufficient_positives_never_interpreted_as_pass"] is True
    assert "held_out_feasible_states_at_least_5" in (
        MODEL_SAFETY_DEFERRED_METRICS
    )


def test_p3_freeze_payload_is_immutable_and_requires_underpowered() -> None:
    map_audit, gate_verdict, dataset_audit = make_p3_evidence()

    payload = build_p3_freeze_payload(
        verdict=gate_verdict,
        map_audit=map_audit,
        dataset_audit=dataset_audit,
        file_manifest={"a.csv": "sha_a"},
        reference_cache_sha256="refsha",
        code_sha256="codesha",
    )
    assert payload["verdict"] == "underpowered_validation"
    assert payload["immutable"] is True
    assert payload["data_generation_evidence_pass"] is True
    assert payload["model_safety_evaluation_underpowered"] is True
    assert payload["robust_feasible_states"] == 9
    assert payload["fallback_only_states"] == 19
    assert payload["boundary_states"] == 4
    assert payload["candidate_generator_recall"] == 1.0
    assert payload["unresolved"] == 0
    assert payload["dataset_total_samples"] == 1132

    # A rewritten (upgraded) verdict must never be freezable.
    with pytest.raises(ValueError, match="underpowered_validation"):
        build_p3_freeze_payload(
            verdict={"status": "pass"},
            map_audit=map_audit,
            dataset_audit=dataset_audit,
            file_manifest={},
            reference_cache_sha256="",
            code_sha256="",
        )


def test_contract_files_encode_the_gate_split() -> None:
    dga = json.loads(
        (CONTRACTS / "PROJECT6_V4_DATA_GENERATION_AUTHORIZATION_V3.json")
        .read_text(encoding="utf-8")
    )
    msg = json.loads(
        (CONTRACTS / "PROJECT6_V4_MODEL_SAFETY_GATE_V3.json").read_text(
            encoding="utf-8"
        )
    )

    expected = dga["expected_outcome_under_current_frozen_p3_evidence"]
    assert expected["scientific_pass"] is True
    assert expected["train1600_planning_authorized"] is True
    assert expected["p3_verdict_preserved"] == "underpowered_validation"
    assert len(dga["pass_conditions"]) == 12
    assert (
        dga["immutable_upstream"]["p3_verdict_status"]
        == "underpowered_validation"
    )
    # Frozen thresholds must stay untouched by the gate split.
    frozen = dga["immutable_upstream"]["frozen_thresholds_untouched"]
    assert frozen["dead_zone"] == {
        "pfv_m3": 1.0,
        "tfv_m3": 1.0,
        "peak_m3s": 0.001,
    }

    assert msg["status"] == "deferred"
    assert msg["reason"] == "requires_train1600_and_powered_locked_validation"
    assert "Train1600 V3 data generation" in msg["scope"]["does_not_control"]
    assert "Policy Lock" in msg["scope"]["controls"]


def test_pipeline_light_import_chain_never_loads_torch() -> None:
    code = (
        "import sys\n"
        "import sewerrtc.v4.pipeline\n"
        "import sewerrtc.v4.pipeline_train_v3\n"
        "import sewerrtc.v4.train1600_v3\n"
        "sys.exit(1 if 'torch' in sys.modules else 0)\n"
    )
    probe = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        timeout=300,
        cwd=str(PROJECT_ROOT),
    )
    assert probe.returncode == 0, probe.stderr.decode(errors="replace")
