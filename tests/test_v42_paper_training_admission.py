from __future__ import annotations

import json
from pathlib import Path

from sewerrtc.v4.v42_paper_training_admission import audit_training_admission


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _oracle(*, raw: bool = True, all_pass: bool = True, count: int = 137) -> dict:
    return {
        "audit_mode": "raw" if raw else "stored",
        "row_count": count,
        "pass_count": count if all_pass else count - 1,
        "fail_count": 0 if all_pass else 1,
        "all_pass": all_pass,
        "expected_count": count,
    }


def _targets(*, complete: bool = True, count: int = 548) -> dict:
    return {
        "contract": "PROJECT6_V42_PAPER_WORKFLOW_V1",
        "detail_count": count,
        "formal_complete_count": count if complete else count - 1,
        "formal_complete": complete,
        "required_target_groups": [
            "node_depth",
            "node_flooding_rate",
            "storage_volume",
            "managed_facility_flow",
            "outfall_flow",
        ],
    }


def test_formal_training_requires_raw_not_stored_oracle(tmp_path: Path):
    oracle = tmp_path / "oracle.json"
    targets = tmp_path / "targets.json"
    _write(oracle, _oracle(raw=False))
    _write(targets, _targets())
    admission = audit_training_admission(
        independent_oracle_summary=oracle,
        hydraulic_target_audit=targets,
    )
    assert not admission.authorized
    assert "stored_oracle_cannot_authorize_formal_training" in admission.reasons


def test_formal_training_blocks_partial_hydraulic_target_pool(tmp_path: Path):
    oracle = tmp_path / "oracle.json"
    targets = tmp_path / "targets.json"
    _write(oracle, _oracle())
    _write(targets, _targets(complete=False))
    admission = audit_training_admission(
        independent_oracle_summary=oracle,
        hydraulic_target_audit=targets,
    )
    assert not admission.authorized
    assert "hydraulic_target_coverage_incomplete" in admission.reasons


def test_formal_training_has_no_fixed_1200_quota(tmp_path: Path):
    oracle = tmp_path / "oracle.json"
    targets = tmp_path / "targets.json"
    _write(oracle, _oracle(count=137))
    # Target audit may count four raw branch details per case; it is an
    # independent coverage population and must be complete, not equal 1200.
    _write(targets, _targets(count=548))
    admission = audit_training_admission(
        independent_oracle_summary=oracle,
        hydraulic_target_audit=targets,
    )
    assert admission.authorized
    assert admission.reasons == ()
    assert admission.expected_sample_count is None
    assert admission.admitted_sample_count == 137


def test_explicit_frozen_experiment_count_is_still_enforced(tmp_path: Path):
    oracle = tmp_path / "oracle.json"
    targets = tmp_path / "targets.json"
    _write(oracle, _oracle(count=137))
    _write(targets, _targets())
    admission = audit_training_admission(
        independent_oracle_summary=oracle,
        hydraulic_target_audit=targets,
        expected_sample_count=1200,
    )
    assert not admission.authorized
    assert any(x.startswith("oracle_row_count_mismatch") for x in admission.reasons)
