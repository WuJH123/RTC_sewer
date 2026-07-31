"""
V4.2 Priority Node Contract — Single Source of Truth
=====================================================

Freezes the 8 PFV core priority nodes, 2 depth sentinel nodes,
and 11 sensitivity zone nodes for the Project6 V4.2 pipeline.

All V4.2 code MUST import node lists from this module.
Never hardcode node IDs or read raw text files in downstream code.

Fail-closed: if node files are missing or SHA doesn't match the
recorded contract, a PriorityContractError is raised at import time.

Contract ID : PFV_CORE8_V1
Freeze date : 2026-07-31
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Dict, List

# ---------------------------------------------------------------------------
# Project root — this file lives at sewerrtc/v4/v42_priority_contract.py
# so project root is three parents up.
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# ---------------------------------------------------------------------------
# Expected SHA-256 digests (computed from the canonical data files).
# If a file changes on disk the contract is broken → import fails.
# ---------------------------------------------------------------------------
_EXPECTED_SHA: Dict[str, str] = {
    "data/project5_design/priority_pfv_core_nodes.txt": (
        "915908de0c3205ee143d4187710ef6680cc49d1839161995d165144df83313a6"
    ),
    "data/project6_v3_sentinel_nodes.txt": (
        "06816141dc99bc3c9ea79d6f261a197a8705faedd9e428e9169a599ce088cd1f"
    ),
    "data/project2_design/priority_zone_nodes.csv": (
        "17ad0a51656ace056a9edcbb723a62616fc9b341a84afc87af102ae006a7f2b4"
    ),
}

CONTRACT_ID = "PFV_CORE8_V1"


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------
class PriorityContractError(Exception):
    """Raised when the priority node contract cannot be verified.

    Possible causes:
    * A node data file is missing from disk.
    * The SHA-256 of a node file does not match the frozen contract.
    * The 8 PFV core nodes and 2 sentinel nodes overlap.
    * A requested node is not found in the graph node list.
    """


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def sha256_file(path: str | Path) -> str:
    """Compute the SHA-256 hex digest of *path*."""
    h = hashlib.sha256()
    p = Path(path)
    if not p.is_file():
        raise PriorityContractError(f"File not found: {p}")
    h.update(p.read_bytes())
    return h.hexdigest()


def _load_txt_ids(rel_path: str) -> List[str]:
    """Load non-empty stripped lines from a .txt file under project root."""
    full = _PROJECT_ROOT / rel_path
    if not full.is_file():
        raise PriorityContractError(
            f"Priority contract file missing: {rel_path}"
        )
    actual_sha = sha256_file(full)
    expected = _EXPECTED_SHA.get(rel_path)
    if expected is not None and actual_sha != expected:
        raise PriorityContractError(
            f"SHA-256 mismatch for {rel_path}\n"
            f"  expected: {expected}\n"
            f"  actual  : {actual_sha}"
        )
    ids = [
        line.strip()
        for line in full.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return ids


def _load_csv_ids(rel_path: str, column: str = "node_id") -> List[str]:
    """Load *column* values from a CSV file under project root."""
    full = _PROJECT_ROOT / rel_path
    if not full.is_file():
        raise PriorityContractError(
            f"Priority contract file missing: {rel_path}"
        )
    actual_sha = sha256_file(full)
    expected = _EXPECTED_SHA.get(rel_path)
    if expected is not None and actual_sha != expected:
        raise PriorityContractError(
            f"SHA-256 mismatch for {rel_path}\n"
            f"  expected: {expected}\n"
            f"  actual  : {actual_sha}"
        )
    ids: List[str] = []
    with open(full, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            val = row.get(column, "").strip()
            if val:
                ids.append(val)
    return ids


# ---------------------------------------------------------------------------
# Canonical node lists — loaded once at import time (fail-closed).
# ---------------------------------------------------------------------------

# 8 PFV core priority nodes
# Source: data/project5_design/priority_pfv_core_nodes.txt
PFV_CORE_8_IDS: List[str] = _load_txt_ids(
    "data/project5_design/priority_pfv_core_nodes.txt"
)

# 2 depth sentinel nodes (monitoring features only)
# Source: data/project6_v3_sentinel_nodes.txt
DEPTH_SENTINEL_2_IDS: List[str] = _load_txt_ids(
    "data/project6_v3_sentinel_nodes.txt"
)

# 11 sensitivity zone nodes (secondary analysis only)
# Source: data/project2_design/priority_zone_nodes.csv
SENSITIVITY_ZONE_11_IDS: List[str] = _load_csv_ids(
    "data/project2_design/priority_zone_nodes.csv"
)

# ---------------------------------------------------------------------------
# Import-time integrity checks
# ---------------------------------------------------------------------------
if len(PFV_CORE_8_IDS) != 8:
    raise PriorityContractError(
        f"Expected 8 PFV core nodes, got {len(PFV_CORE_8_IDS)}: "
        f"{PFV_CORE_8_IDS}"
    )

if len(DEPTH_SENTINEL_2_IDS) != 2:
    raise PriorityContractError(
        f"Expected 2 sentinel nodes, got {len(DEPTH_SENTINEL_2_IDS)}: "
        f"{DEPTH_SENTINEL_2_IDS}"
    )

if len(SENSITIVITY_ZONE_11_IDS) != 11:
    raise PriorityContractError(
        f"Expected 11 sensitivity zone nodes, got "
        f"{len(SENSITIVITY_ZONE_11_IDS)}: {SENSITIVITY_ZONE_11_IDS}"
    )

_overlap_pfv_sentinel = set(PFV_CORE_8_IDS) & set(DEPTH_SENTINEL_2_IDS)
if _overlap_pfv_sentinel:
    raise PriorityContractError(
        f"PFV core and sentinel nodes must not overlap: {_overlap_pfv_sentinel}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def get_pfv_core_node_indices(graph_node_ids: List[str]) -> List[int]:
    """Return integer indices of the 8 PFV core nodes within *graph_node_ids*.

    Parameters
    ----------
    graph_node_ids : list[str]
        Ordered list of node IDs as they appear in the graph / tensor cache.

    Returns
    -------
    list[int]
        Indices into *graph_node_ids* corresponding to each of the 8 PFV
        core nodes, in the canonical order defined by
        ``PFV_CORE_8_IDS``.

    Raises
    ------
    PriorityContractError
        If any of the 8 PFV core nodes is not found in *graph_node_ids*.
    """
    id_to_idx: Dict[str, int] = {nid: i for i, nid in enumerate(graph_node_ids)}
    indices: List[int] = []
    missing: List[str] = []
    for nid in PFV_CORE_8_IDS:
        if nid in id_to_idx:
            indices.append(id_to_idx[nid])
        else:
            missing.append(nid)
    if missing:
        raise PriorityContractError(
            f"PFV core nodes not found in graph: {missing}"
        )
    return indices


def get_sentinel_node_indices(graph_node_ids: List[str]) -> List[int]:
    """Return integer indices of the 2 sentinel nodes within *graph_node_ids*.

    Parameters
    ----------
    graph_node_ids : list[str]
        Ordered list of node IDs as they appear in the graph / tensor cache.

    Returns
    -------
    list[int]
        Indices into *graph_node_ids* corresponding to each of the 2
        sentinel nodes, in the canonical order defined by
        ``DEPTH_SENTINEL_2_IDS``.

    Raises
    ------
    PriorityContractError
        If any sentinel node is not found in *graph_node_ids*.
    """
    id_to_idx: Dict[str, int] = {nid: i for i, nid in enumerate(graph_node_ids)}
    indices: List[int] = []
    missing: List[str] = []
    for nid in DEPTH_SENTINEL_2_IDS:
        if nid in id_to_idx:
            indices.append(id_to_idx[nid])
        else:
            missing.append(nid)
    if missing:
        raise PriorityContractError(
            f"Sentinel nodes not found in graph: {missing}"
        )
    return indices


def audit_contract() -> dict:
    """Verify the full priority contract and return an audit report.

    Checks performed
    ----------------
    1. All three data files exist on disk.
    2. SHA-256 of each file matches the frozen contract value.
    3. Node counts are correct (8 + 2 + 11).
    4. No overlap between PFV core and sentinel nodes.

    Returns
    -------
    dict
        ``{"status": "PASS", ...}`` on success, or
        ``{"status": "CONTRACT_CONFLICT", "errors": [...]}`` on failure.
    """
    errors: List[str] = []

    # --- file existence & SHA -------------------------------------------
    for rel, expected in _EXPECTED_SHA.items():
        full = _PROJECT_ROOT / rel
        if not full.is_file():
            errors.append(f"Missing file: {rel}")
            continue
        actual = sha256_file(full)
        if actual != expected:
            errors.append(
                f"SHA mismatch {rel}: expected {expected}, got {actual}"
            )

    # --- counts ---------------------------------------------------------
    if len(PFV_CORE_8_IDS) != 8:
        errors.append(f"PFV core count: {len(PFV_CORE_8_IDS)} (expected 8)")
    if len(DEPTH_SENTINEL_2_IDS) != 2:
        errors.append(
            f"Sentinel count: {len(DEPTH_SENTINEL_2_IDS)} (expected 2)"
        )
    if len(SENSITIVITY_ZONE_11_IDS) != 11:
        errors.append(
            f"Sensitivity zone count: {len(SENSITIVITY_ZONE_11_IDS)} "
            f"(expected 11)"
        )

    # --- overlap --------------------------------------------------------
    overlap = set(PFV_CORE_8_IDS) & set(DEPTH_SENTINEL_2_IDS)
    if overlap:
        errors.append(f"PFV/sentinel overlap: {overlap}")

    if errors:
        return {"status": "CONTRACT_CONFLICT", "errors": errors}
    return {
        "status": "PASS",
        "contract_id": CONTRACT_ID,
        "pfv_core_count": len(PFV_CORE_8_IDS),
        "sentinel_count": len(DEPTH_SENTINEL_2_IDS),
        "sensitivity_zone_count": len(SENSITIVITY_ZONE_11_IDS),
    }


# ---------------------------------------------------------------------------
# Convenience: print summary when executed directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print(f"Contract ID      : {CONTRACT_ID}")
    print(f"PFV core nodes   : {len(PFV_CORE_8_IDS)}  {PFV_CORE_8_IDS}")
    print(f"Sentinel nodes   : {len(DEPTH_SENTINEL_2_IDS)}  {DEPTH_SENTINEL_2_IDS}")
    print(f"Sensitivity zone : {len(SENSITIVITY_ZONE_11_IDS)}  {SENSITIVITY_ZONE_11_IDS}")
    print(f"Overlap (core∩sentinel): {set(PFV_CORE_8_IDS) & set(DEPTH_SENTINEL_2_IDS)}")
    print(f"Audit result     : {audit_contract()}")
