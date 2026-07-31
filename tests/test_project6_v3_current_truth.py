from __future__ import annotations

import json
from pathlib import Path

from sewerrtc.status.current_truth import ALLOWED_STAGE_STATUSES, sha256_file


ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "docs" / "contracts" / "PROJECT6_V3_CURRENT_TRUTH_CONTRACT.json"
CURRENT_TRUTH = ROOT / "sewerrtc" / "status" / "current_truth.py"
COMPLETION_SCRIPT = ROOT / "scripts" / "165_evaluate_prompt3a_completion.py"
STATE_CLONE_SCRIPT = ROOT / "scripts" / "138_prepare_state_clone_test.py"
DRYRUN_SCRIPT = ROOT / "scripts" / "164_dryrun_round0_v3.py"
RUNNER = ROOT / "scripts" / "project6_runs" / "RUN_PROJECT6_PFVFIRST_DUALFALLBACK_10MIN_V3.ps1"


def test_current_truth_status_enum_is_frozen() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["allowed_stage_statuses"] == ALLOWED_STAGE_STATUSES
    assert "completed" not in contract["allowed_stage_statuses"]
    assert "completed_structural_dryrun" not in contract["allowed_stage_statuses"]


def test_schema_only_hotstart_and_hydraulic_not_run_cannot_make_runtime_pass() -> None:
    source = CURRENT_TRUTH.read_text(encoding="utf-8")
    assert '"state_clone_equivalence_pass": state_clone_pass' in source
    assert 'dryrun_report.get("same_state_hotstart_execution_status") == "pass"' in source
    assert '"hydraulic_candidate_dryrun_pass": hydraulic_dryrun_pass' in source
    assert '"tail_recovery_verified_or_censored": recovery_contract_complete' in source


def test_baseline_state_count_mismatch_blocks_runtime_gate() -> None:
    source = CURRENT_TRUTH.read_text(encoding="utf-8")
    assert "all_trajectories_in_state = processable_count > 0 and state_input_count == processable_count" in source
    assert "state input rows" in source


def test_feature_contract_dimension_mismatch_blocks_runtime_gate() -> None:
    source = CURRENT_TRUTH.read_text(encoding="utf-8")
    assert "node_contract_field_count_match" in source
    assert "facility_contract_field_count_match" in source
    assert '"actual_features_match_schema": state_schema_match' in source


def test_nonexistent_checkpoint_file_has_no_valid_hash() -> None:
    assert sha256_file(ROOT / "outputs" / "missing_checkpoint_should_not_hash.bin") is None


def test_marker_audit_detects_missing_stale_and_forbidden_markers() -> None:
    source = CURRENT_TRUTH.read_text(encoding="utf-8")
    assert "missing_output_count" in source
    assert "missing_output_hash_count" in source
    assert "config_hash_stale" in source
    assert "forbidden_marker_reason" in source
    assert "hotstart_equivalence_not_run" in source
    assert "hydraulic_dryrun_not_run" in source


def test_state_clone_and_structural_dryrun_do_not_allow_completion_markers() -> None:
    state_clone = STATE_CLONE_SCRIPT.read_text(encoding="utf-8")
    dryrun = DRYRUN_SCRIPT.read_text(encoding="utf-8")
    assert '"hotstart_equivalence_status": "not_run"' in state_clone
    assert '"completion_marker_allowed": False' in state_clone
    assert "return 3" in state_clone
    assert '"hydraulic_branch_execution_status": "not_run"' in dryrun
    assert '"completion_marker_allowed": False' in dryrun
    assert "return 3" in dryrun


def test_prompt3a_engineering_and_runtime_gates_are_separate() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    completion = COMPLETION_SCRIPT.read_text(encoding="utf-8")
    assert "EvaluatePrompt3AEngineeringGate" in runner
    assert "EvaluatePrompt3ARuntimeGate" in runner
    assert "write_prompt3a_completion" in completion
    assert 'gate["status"] == "pass"' in completion
    assert "runtime_blocking_reasons" in completion


def test_blocked_stage_status_writes_null_completion_marker() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    assert 'Write-StageStatus -Stage $s -Status "blocked" -ExitCode 3' in runner
    assert "-CompletionMarker $null" in runner


def test_new_completion_markers_include_truth_contract_fields() -> None:
    runner = RUNNER.read_text(encoding="utf-8")
    for token in [
        "stage_implementation_version",
        "config_scope_hash",
        "input_hashes",
        "output_paths",
        "output_hashes",
        "runtime_actually_executed",
        "scientific_gate_evaluated",
    ]:
        assert token in runner
