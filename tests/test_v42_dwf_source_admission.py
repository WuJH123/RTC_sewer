"""Tests for V4.2 DWF source admission (v42_dwf_audit).

Verifies:
- DWF audit classifies samples
- All 4 categories exist in enum
- Classification is deterministic
"""

from __future__ import annotations

import pytest

from sewerrtc.v4.v42_dwf_audit import (
    _classify_dwf,
    DWF_FULL_SUPERVISION,
    DWF_DYNAMICS_PRETRAIN,
    DWF_ACTUATOR_EFFECT,
    DWF_INCOMPLETE_REJECT,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestDWFCategoriesExist:
    """All 4 categories exist in enum."""

    def test_full_supervision_exists(self):
        assert DWF_FULL_SUPERVISION == "SOURCE_DWF_FULL_SUPERVISION"

    def test_dynamics_pretrain_exists(self):
        assert DWF_DYNAMICS_PRETRAIN == "SOURCE_DWF_DYNAMICS_PRETRAIN"

    def test_actuator_effect_exists(self):
        assert DWF_ACTUATOR_EFFECT == "SOURCE_DWF_ACTUATOR_EFFECT"

    def test_incomplete_reject_exists(self):
        assert DWF_INCOMPLETE_REJECT == "DWF_INCOMPLETE_REJECT"


class TestDWFClassification:
    """DWF audit classifies samples."""

    def test_no_dwf_inflow_rejected(self):
        result = _classify_dwf(
            dwf_node_inflow_present=False,
            h120_complete=True,
            h120_eligible=True,
            four_branches=True,
            model_dwf_present=True,
            actual_actions_present=True,
            hydraulic_complete=True,
            labels_present=True,
            k_actual=3,
            candidate_role="primary",
        )
        assert result == DWF_INCOMPLETE_REJECT

    def test_full_supervision_conditions(self):
        result = _classify_dwf(
            dwf_node_inflow_present=True,
            h120_complete=True,
            h120_eligible=True,
            four_branches=True,
            model_dwf_present=True,
            actual_actions_present=True,
            hydraulic_complete=True,
            labels_present=True,
            k_actual=3,
            candidate_role="primary",
        )
        assert result == DWF_FULL_SUPERVISION

    def test_no_actions_dynamics_only(self):
        result = _classify_dwf(
            dwf_node_inflow_present=True,
            h120_complete=True,
            h120_eligible=True,
            four_branches=True,
            model_dwf_present=True,
            actual_actions_present=False,
            hydraulic_complete=True,
            labels_present=True,
            k_actual=0,
            candidate_role="primary",
        )
        assert result == DWF_DYNAMICS_PRETRAIN

    def test_no_labels_actuator_effect(self):
        result = _classify_dwf(
            dwf_node_inflow_present=True,
            h120_complete=True,
            h120_eligible=True,
            four_branches=True,
            model_dwf_present=True,
            actual_actions_present=True,
            hydraulic_complete=True,
            labels_present=False,
            k_actual=3,
            candidate_role="primary",
        )
        assert result == DWF_ACTUATOR_EFFECT


class TestDWFClassificationDeterministic:
    """Classification is deterministic."""

    def test_same_input_same_output(self):
        kwargs = dict(
            dwf_node_inflow_present=True,
            h120_complete=True,
            h120_eligible=True,
            four_branches=True,
            model_dwf_present=True,
            actual_actions_present=True,
            hydraulic_complete=True,
            labels_present=True,
            k_actual=3,
            candidate_role="primary",
        )
        r1 = _classify_dwf(**kwargs)
        r2 = _classify_dwf(**kwargs)
        assert r1 == r2

    def test_multiple_calls_consistent(self):
        kwargs = dict(
            dwf_node_inflow_present=False,
            h120_complete=False,
            h120_eligible=False,
            four_branches=False,
            model_dwf_present=False,
            actual_actions_present=False,
            hydraulic_complete=False,
            labels_present=False,
            k_actual=0,
            candidate_role="",
        )
        results = {_classify_dwf(**kwargs) for _ in range(10)}
        assert len(results) == 1  # always the same
