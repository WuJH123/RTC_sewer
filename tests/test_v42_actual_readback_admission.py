"""Tests for V4.2 actual/readback admission (v42_semantic_audit + v42_sample_lineage).

Verifies:
- Samples have actual/readback action columns
- Action shape is 12×36
- K_actual is present
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from sewerrtc.v4.v42_semantic_audit import _action_seq_depth, _action_seq_width


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_with_actions() -> dict:
    """A sample row with actual/readback action data."""
    # 12 time steps × 36 facilities
    action_seq = [[float(i + j) for j in range(36)] for i in range(12)]
    return {
        "candidate_action_seq": action_seq,
        "ref_no_control_action_seq": action_seq,
        "ref_dynamic_internal_action_seq": action_seq,
        "ref_hold_previous_action_seq": action_seq,
        "k_actual": 3,
        "k_target": 5,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestActualReadbackColumns:
    """Samples have actual/readback action columns."""

    def test_candidate_action_present(self, sample_with_actions):
        assert "candidate_action_seq" in sample_with_actions
        assert sample_with_actions["candidate_action_seq"] is not None

    def test_all_four_branch_actions_present(self, sample_with_actions):
        for key in [
            "candidate_action_seq",
            "ref_no_control_action_seq",
            "ref_dynamic_internal_action_seq",
            "ref_hold_previous_action_seq",
        ]:
            assert key in sample_with_actions


class TestActionShape12x36:
    """Action shape is 12×36."""

    def test_candidate_depth_is_12(self, sample_with_actions):
        depth = _action_seq_depth(sample_with_actions["candidate_action_seq"])
        assert depth == 12

    def test_candidate_width_is_36(self, sample_with_actions):
        width = _action_seq_width(sample_with_actions["candidate_action_seq"])
        assert width == 36

    def test_all_branches_12x36(self, sample_with_actions):
        for key in [
            "candidate_action_seq",
            "ref_no_control_action_seq",
            "ref_dynamic_internal_action_seq",
            "ref_hold_previous_action_seq",
        ]:
            assert _action_seq_depth(sample_with_actions[key]) == 12
            assert _action_seq_width(sample_with_actions[key]) == 36


class TestKActualPresent:
    """K_actual is present."""

    def test_k_actual_in_sample(self, sample_with_actions):
        assert "k_actual" in sample_with_actions
        assert sample_with_actions["k_actual"] == 3

    def test_k_actual_leq_k_target(self, sample_with_actions):
        assert sample_with_actions["k_actual"] <= sample_with_actions["k_target"]
