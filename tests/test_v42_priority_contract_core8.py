"""V4.2 PFV Core8 scientific-contract verification."""
from __future__ import annotations

from sewerrtc.v4.v42_priority_contract import (
    CONTRACT_ID,
    PFV_CORE_8_IDS,
    audit_contract,
    get_pfv_core_node_indices,
    sha256_file,
    _PROJECT_ROOT,
)


EXPECTED_IDS = [
    "MSLBZW001",
    "HS1316314",
    "YS2530050",
    "HS2529198",
    "MH0200773",
    "HS1330349",
    "HS2529139",
    "HS2529052",
]

NODE_FILE_REL = "data/project5_design/priority_pfv_core_nodes.txt"


def test_all_8_ids_and_order_are_frozen():
    assert PFV_CORE_8_IDS == EXPECTED_IDS
    assert len(PFV_CORE_8_IDS) == 8
    assert len(set(PFV_CORE_8_IDS)) == 8


def test_returns_correct_indices_for_known_graph():
    graph = ["DUMMY_A"] + EXPECTED_IDS + ["DUMMY_B"]
    assert get_pfv_core_node_indices(graph) == list(range(1, 9))


def test_returns_correct_indices_scattered():
    graph = EXPECTED_IDS[:4] + ["X", "Y"] + EXPECTED_IDS[4:]
    assert get_pfv_core_node_indices(graph) == [0, 1, 2, 3, 6, 7, 8, 9]


def test_contract_id_is_correct():
    assert CONTRACT_ID == "PFV_CORE8_V1"


def test_source_file_hash_is_lineage_not_admission():
    full_path = _PROJECT_ROOT / NODE_FILE_REL
    assert full_path.is_file()
    digest = sha256_file(full_path)
    assert len(digest) == 64
    assert audit_contract()["hash_is_admission_criterion"] is False


def test_semantic_contract_passes():
    result = audit_contract()
    assert result["status"] == "PASS", result
    assert result["pfv_core_ids"] == EXPECTED_IDS
