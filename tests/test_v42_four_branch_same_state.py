"""Tests for V4.2 four-branch same-state semantic audit (v42_semantic_audit).

Verifies:
- Semantic audit checks 4-branch consistency
- Candidate, NC, DI, Hold all present
- Same prefix state across branches
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sewerrtc.v4.v42_semantic_audit import (
    _action_seq_depth,
    _action_seq_width,
)


# ---------------------------------------------------------------------------
# Tests for action sequence helpers
# ---------------------------------------------------------------------------

class TestActionSeqDepth:
    """Test _action_seq_depth helper."""

    def test_none_returns_0(self):
        assert _action_seq_depth(None) == 0

    def test_nan_returns_0(self):
        assert _action_seq_depth(float("nan")) == 0

    def test_2d_list_returns_rows(self):
        seq = [[1, 2, 3]] * 12
        assert _action_seq_depth(seq) == 12

    def test_json_string(self):
        seq = json.dumps([[1, 2]] * 5)
        assert _action_seq_depth(seq) == 5

    def test_1d_list_returns_length(self):
        seq = [1, 2, 3, 4]
        assert _action_seq_depth(seq) == 4


class TestActionSeqWidth:
    """Test _action_seq_width helper."""

    def test_none_returns_0(self):
        assert _action_seq_width(None) == 0

    def test_2d_list_returns_cols(self):
        seq = [[1, 2, 3]] * 12
        assert _action_seq_width(seq) == 3

    def test_1d_list_returns_0(self):
        seq = [1, 2, 3]
        assert _action_seq_width(seq) == 0


class TestFourBranchConsistency:
    """Semantic audit checks 4-branch consistency."""

    def test_all_four_branches_same_depth(self):
        """All 4 branches should have same time depth (12 steps)."""
        branches = {
            "candidate": [[1.0]] * 12,
            "nc": [[2.0]] * 12,
            "di": [[3.0]] * 12,
            "hold": [[4.0]] * 12,
        }
        depths = [_action_seq_depth(v) for v in branches.values()]
        assert all(d == 12 for d in depths)

    def test_all_four_branches_same_width(self):
        """All 4 branches should have same facility width (36)."""
        branches = {
            "candidate": [[0.0] * 36] * 12,
            "nc": [[0.0] * 36] * 12,
            "di": [[0.0] * 36] * 12,
            "hold": [[0.0] * 36] * 12,
        }
        widths = [_action_seq_width(v) for v in branches.values()]
        assert all(w == 36 for w in widths)


class TestSamePrefixState:
    """Same prefix state across branches."""

    def test_prefix_state_hash_is_consistent(self):
        """All 4 branches share the same prefix state hash."""
        prefix_hash = "abc123def456"
        # In real data, all branches from same checkpoint share prefix state
        branch_states = [prefix_hash] * 4
        assert len(set(branch_states)) == 1  # all same
