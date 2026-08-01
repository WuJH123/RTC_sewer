"""Step 1 can run from partial-aux R0 without counterfactual data.

Verifies that Step 1 window manifest can be built even when the R0 pool
contains PFV-first aux-only cases (no full four-reference counterfactual).
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from sewerrtc.v4.v42_mainline_workflow import audit_phase_r0


def test_step1_can_run_from_partial_aux_r0(tmp_path: Path) -> None:
    """R0 with dynamics_pretrain > 0 and counterfactual=0 → Step 1 allowed."""
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
            "dynamics_pretrain_physical_runs": 5000,
            "counterfactual_flood_cases": 0,
            "formal_target_domain_cases": 85,
        },
    }), encoding="utf-8")
    align = pd.DataFrame({
        "case_uid": ["c1", "c2"],
        "same_state_numeric_pass": [True, True],
        "same_forcing_pass": [True, True],
    })
    align.to_csv(root / "case_alignment_audit.csv", index=False)
    split = pd.DataFrame({
        "physical_identity_sha256": ["a", "b", "c"],
        "split_group_key": ["g1", "g1", "g2"],
        "reserved_evaluation": [False, False, False],
    })
    split.to_parquet(root / "split_group_manifest.parquet", index=False)

    audit = audit_phase_r0(tmp_path)
    assert audit.passed is True, f"R0_STATE_READY should pass: {audit.reasons}"
