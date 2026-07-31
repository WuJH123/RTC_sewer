"""V4.2 Priority Contract — Fail-closed: missing nodes raise PriorityContractError."""
from __future__ import annotations

import pytest

from sewerrtc.v4.v42_priority_contract import (
    PFV_CORE_8_IDS,
    PriorityContractError,
    get_pfv_core_node_indices,
)


class TestMissingNodeRaises:
    def test_raises_when_one_node_missing(self):
        # Provide only 7 of the 8 core nodes
        partial = PFV_CORE_8_IDS[:7]
        with pytest.raises(PriorityContractError):
            get_pfv_core_node_indices(partial)

    def test_raises_when_multiple_nodes_missing(self):
        # Provide only 4 of the 8 core nodes
        partial = PFV_CORE_8_IDS[:4]
        with pytest.raises(PriorityContractError):
            get_pfv_core_node_indices(partial)

    def test_raises_when_graph_empty(self):
        with pytest.raises(PriorityContractError):
            get_pfv_core_node_indices([])


class TestNoSilentFallback:
    def test_does_not_accept_subset_silently(self):
        """Returning fewer than 8 indices without raising is forbidden."""
        partial = PFV_CORE_8_IDS[:6]
        with pytest.raises(PriorityContractError):
            get_pfv_core_node_indices(partial)

    def test_sentinel_nodes_do_not_substitute_for_core(self):
        """Sentinel node IDs must not satisfy PFV core lookups."""
        from sewerrtc.v4.v42_priority_contract import DEPTH_SENTINEL_2_IDS

        # Build a graph containing sentinel nodes but only 7 core nodes
        graph = list(DEPTH_SENTINEL_2_IDS) + PFV_CORE_8_IDS[:7]
        with pytest.raises(PriorityContractError):
            get_pfv_core_node_indices(graph)
