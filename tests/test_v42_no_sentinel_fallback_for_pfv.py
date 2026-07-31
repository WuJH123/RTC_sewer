"""V4.2 Priority Contract — Sentinel nodes are never used for PFV computation."""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from sewerrtc.v4.v42_priority_contract import (
    get_sentinel_node_indices,
    get_pfv_core_node_indices,
    PriorityContractError,
)

V4_DIR = Path(__file__).resolve().parent.parent / "sewerrtc" / "v4"

FIXED_FILES = [
    "v42_trainer.py",
    "v42_trajectory_builder.py",
    "v42_full_verification.py",
    "v42_pool_audit.py",
]


class TestNoSentinelStringInFixedFiles:
    @pytest.mark.parametrize("filename", FIXED_FILES)
    def test_does_not_contain_sentinel_file_path(self, filename):
        content = (V4_DIR / filename).read_text(encoding="utf-8")
        assert "project6_v3_sentinel_nodes.txt" not in content, (
            f"{filename} still contains raw sentinel file path string"
        )


class TestImportsFromPriorityContract:
    @pytest.mark.parametrize("filename", FIXED_FILES)
    def test_imports_from_v42_priority_contract(self, filename):
        content = (V4_DIR / filename).read_text(encoding="utf-8")
        assert "v42_priority_contract" in content, (
            f"{filename} does not import from v42_priority_contract"
        )


class TestSentinelFunctionExistsSeparately:
    def test_get_sentinel_node_indices_is_callable(self):
        assert callable(get_sentinel_node_indices)

    def test_get_sentinel_node_indices_returns_indices(self):
        from sewerrtc.v4.v42_priority_contract import DEPTH_SENTINEL_2_IDS

        graph = ["A", "B"] + list(DEPTH_SENTINEL_2_IDS) + ["C"]
        indices = get_sentinel_node_indices(graph)
        assert indices == [2, 3]

    def test_sentinel_raises_when_missing(self):
        with pytest.raises(PriorityContractError):
            get_sentinel_node_indices(["A", "B", "C"])


class TestSentinelNotCalledInPFVPath:
    def test_pfv_indices_differ_from_sentinel_indices(self):
        """PFV core and sentinel must resolve to different index sets."""
        from sewerrtc.v4.v42_priority_contract import (
            PFV_CORE_8_IDS,
            DEPTH_SENTINEL_2_IDS,
        )

        graph = list(PFV_CORE_8_IDS) + list(DEPTH_SENTINEL_2_IDS)
        pfv_idx = set(get_pfv_core_node_indices(graph))
        sentinel_idx = set(get_sentinel_node_indices(graph))
        assert pfv_idx & sentinel_idx == set()
