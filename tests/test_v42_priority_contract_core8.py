"""V4.2 Priority Contract — PFV core 8 node contract verification."""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from sewerrtc.v4.v42_priority_contract import (
    CONTRACT_ID,
    PFV_CORE_8_IDS,
    get_pfv_core_node_indices,
    sha256_file,
    _PROJECT_ROOT,
)


EXPECTED_IDS = [
    "MSLBZW001", "HS1316314", "YS2530050", "HS2529198",
    "MH0200773", "HS1330349", "HS2529139", "HS2529052",
]

NODE_FILE_REL = "data/project5_design/priority_pfv_core_nodes.txt"
EXPECTED_SHA = (
    "915908de0c3205ee143d4187710ef6680cc49d1839161995d165144df83313a6"
)


class TestAll8IDsPresent:
    def test_all_8_ids_are_correct(self):
        assert PFV_CORE_8_IDS == EXPECTED_IDS


class TestGetPFVCoreNodeIndices:
    def test_returns_correct_indices_for_known_graph(self):
        # Build a graph where we know the positions
        graph = ["DUMMY_A"] + EXPECTED_IDS + ["DUMMY_B"]
        indices = get_pfv_core_node_indices(graph)
        assert indices == list(range(1, 9))  # positions 1..8

    def test_returns_correct_indices_scattered(self):
        # Interleave with other nodes
        graph = EXPECTED_IDS[:4] + ["X", "Y"] + EXPECTED_IDS[4:]
        indices = get_pfv_core_node_indices(graph)
        expected_indices = [0, 1, 2, 3, 6, 7, 8, 9]
        assert indices == expected_indices


class TestContractID:
    def test_contract_id_is_correct(self):
        assert CONTRACT_ID == "PFV_CORE8_V1"


class TestSHA256OfNodeFile:
    def test_sha256_matches_expected(self):
        full_path = _PROJECT_ROOT / NODE_FILE_REL
        actual = sha256_file(full_path)
        assert actual == EXPECTED_SHA


class TestContractJSONFileExists:
    def test_node_file_exists_on_disk(self):
        full_path = _PROJECT_ROOT / NODE_FILE_REL
        assert full_path.is_file(), f"Node file missing: {full_path}"
