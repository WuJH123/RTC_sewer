"""R0_STATE_READY must pass when dynamics_pretrain > 0 even if counterfactual_flood = 0.

This test verifies the three-level gate redesign:
- R0_STATE_READY (Step 1 entry) does NOT require counterfactual_flood > 0
- R0_COUNTERFACTUAL_READY (Step 2 entry) DOES require counterfactual_flood > 0
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from sewerrtc.v4.v42_mainline_workflow import (
    _audit_r0_counterfactual_gate,
    audit_phase_r0,
)


@pytest.fixture()
def r0_state_ready_no_counterfactual(tmp_path: Path) -> Path:
    """Build a minimal R0 evidence tree with counterfactual_flood=0."""
    root = tmp_path / "v42_paper" / "data_reuse"
    root.mkdir(parents=True)

    # data_reuse_audit.json
    (root / "data_reuse_audit.json").write_text(json.dumps({
        "full_finite_check": True,
        "missing_targets_are_imputed": False,
        "strict_semantics_wrapper": True,
        "discovery_cache_current": True,
        "pfvfirst_continuation_role_policy": "auxiliary_until_proven_by_provenance",
    }), encoding="utf-8")

    # reusable_pool_summary.json — dynamics_pretrain > 0, counterfactual_flood = 0
    (root / "reusable_pool_summary.json").write_text(json.dumps({
        "strict_scientific_admission": True,
        "counterfactual_requires_all_four_roles_finite": True,
        "formal_counterfactual_requires_target_no_dwf": True,
        "source_domain_formal_admission_forbidden": True,
        "task_counts": {
            "dynamics_pretrain_physical_runs": 11888,
            "storage_supervision_cases": 5509,
            "counterfactual_flood_cases": 0,
            "formal_target_domain_cases": 85,
        },
    }), encoding="utf-8")

    # case_alignment_audit.csv — at least one aligned case
    align = pd.DataFrame({
        "case_uid": ["c1", "c2"],
        "same_state_numeric_pass": [True, True],
        "same_forcing_pass": [True, True],
    })
    align.to_csv(root / "case_alignment_audit.csv", index=False)

    # split_group_manifest.parquet — at least 2 groups, no reserved
    split = pd.DataFrame({
        "physical_identity_sha256": ["a", "b", "c"],
        "split_group_key": ["g1", "g1", "g2"],
        "reserved_evaluation": [False, False, False],
    })
    split.to_parquet(root / "split_group_manifest.parquet", index=False)

    return tmp_path


@pytest.fixture()
def r0_counterfactual_blocked(tmp_path: Path) -> Path:
    """Same as above but counterfactual_flood_cases = 0."""
    root = tmp_path / "v42_paper" / "data_reuse"
    root.mkdir(parents=True)
    (root / "data_reuse_audit.json").write_text(json.dumps({
        "full_finite_check": True,
        "missing_targets_are_imputed": False,
        "strict_semantics_wrapper": True,
        "discovery_cache_current": True,
        "pfvfirst_continuation_role_policy": "auxiliary_until_proven_by_provenance",
    }), encoding="utf-8")
    (root / "reusable_pool_summary.json").write_text(json.dumps({
        "strict_scientific_admission": True,
        "counterfactual_requires_all_four_roles_finite": True,
        "formal_counterfactual_requires_target_no_dwf": True,
        "source_domain_formal_admission_forbidden": True,
        "task_counts": {
            "dynamics_pretrain_physical_runs": 11888,
            "counterfactual_flood_cases": 0,
            "formal_target_domain_cases": 85,
        },
    }), encoding="utf-8")
    return tmp_path


def test_r0_state_ready_without_counterfactual(r0_state_ready_no_counterfactual: Path) -> None:
    """dynamics_pretrain > 0 and counterfactual=0 → R0_STATE_READY must PASS."""
    audit = audit_phase_r0(r0_state_ready_no_counterfactual)
    assert audit.passed is True, f"R0_STATE_READY should pass but got reasons: {audit.reasons}"
    assert audit.stage == "phase_r0"


def test_step2_blocked_when_counterfactual_zero(r0_counterfactual_blocked: Path) -> None:
    """counterfactual_flood=0 → Step 2 counterfactual gate must FAIL."""
    gate = _audit_r0_counterfactual_gate(r0_counterfactual_blocked)
    assert gate.passed is False
    assert any("step2_no_aligned_counterfactual_cases" in r for r in gate.reasons)
