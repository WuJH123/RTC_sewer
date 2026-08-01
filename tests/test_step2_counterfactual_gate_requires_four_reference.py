"""Step 2 counterfactual gate requires four-reference alignment.

Verifies that the counterfactual gate blocks Step 2 when
counterfactual_flood_cases = 0 regardless of other data availability.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from sewerrtc.v4.v42_mainline_workflow import _audit_r0_counterfactual_gate


def test_step2_counterfactual_gate_requires_four_reference(tmp_path: Path) -> None:
    root = tmp_path / "v42_paper" / "data_reuse"
    root.mkdir(parents=True)
    (root / "reusable_pool_summary.json").write_text(json.dumps({
        "task_counts": {
            "counterfactual_flood_cases": 0,
            "dynamics_pretrain_physical_runs": 5000,
        },
    }), encoding="utf-8")
    gate = _audit_r0_counterfactual_gate(tmp_path)
    assert gate.passed is False
    assert any("step2" in r for r in gate.reasons)


def test_step2_counterfactual_gate_passes_with_cases(tmp_path: Path) -> None:
    root = tmp_path / "v42_paper" / "data_reuse"
    root.mkdir(parents=True)
    (root / "reusable_pool_summary.json").write_text(json.dumps({
        "task_counts": {
            "counterfactual_flood_cases": 376,
            "dynamics_pretrain_physical_runs": 5000,
        },
    }), encoding="utf-8")
    gate = _audit_r0_counterfactual_gate(tmp_path)
    assert gate.passed is True
