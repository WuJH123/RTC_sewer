"""Outfall missing must NOT disable counterfactual_flood computation.

The counterfactual gate depends on four-reference alignment and
target_no_dwf, not on outfall flow labels.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from sewerrtc.v4.v42_mainline_workflow import audit_phase_r0, _audit_r0_counterfactual_gate


def _make_r0_root(tmp_path: Path, *, counterfactual_cases: int, has_outfall: bool = False) -> Path:
    root = tmp_path / "v42_paper" / "data_reuse"
    root.mkdir(parents=True)
    (root / "data_reuse_audit.json").write_text(json.dumps({
        "full_finite_check": True,
        "missing_targets_are_imputed": False,
        "strict_semantics_wrapper": True,
        "discovery_cache_current": True,
        "pfvfirst_continuation_role_policy": "auxiliary_until_proven_by_provenance",
    }), encoding="utf-8")
    task_counts = {
        "dynamics_pretrain_physical_runs": 1000,
        "counterfactual_flood_cases": counterfactual_cases,
        "formal_target_domain_cases": 85,
    }
    if has_outfall:
        task_counts["outfall_supervision_cases"] = 100
    (root / "reusable_pool_summary.json").write_text(json.dumps({
        "strict_scientific_admission": True,
        "counterfactual_requires_all_four_roles_finite": True,
        "formal_counterfactual_requires_target_no_dwf": True,
        "source_domain_formal_admission_forbidden": True,
        "task_counts": task_counts,
    }), encoding="utf-8")
    align = pd.DataFrame({
        "case_uid": ["c1"],
        "same_state_numeric_pass": [True],
        "same_forcing_pass": [True],
    })
    align.to_csv(root / "case_alignment_audit.csv", index=False)
    split = pd.DataFrame({
        "physical_identity_sha256": ["a", "b"],
        "split_group_key": ["g1", "g2"],
        "reserved_evaluation": [False, False],
    })
    split.to_parquet(root / "split_group_manifest.parquet", index=False)
    return tmp_path


def test_outfall_missing_does_not_disable_counterfactual_flood(tmp_path: Path) -> None:
    """No outfall labels + counterfactual>0 → counterfactual gate must PASS."""
    root = _make_r0_root(tmp_path, counterfactual_cases=100, has_outfall=False)
    gate = _audit_r0_counterfactual_gate(root)
    assert gate.passed is True, f"Expected pass but got: {gate.reasons}"


def test_outfall_missing_blocks_full_target_only(tmp_path: Path) -> None:
    """Outfall missing should not affect R0_STATE_READY or counterfactual gate."""
    root = _make_r0_root(tmp_path, counterfactual_cases=100, has_outfall=False)
    r0 = audit_phase_r0(root)
    assert r0.passed is True, f"R0_STATE_READY should pass: {r0.reasons}"
    gate = _audit_r0_counterfactual_gate(root)
    assert gate.passed is True, f"Counterfactual gate should pass: {gate.reasons}"
