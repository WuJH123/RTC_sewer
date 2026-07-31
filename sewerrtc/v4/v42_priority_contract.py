"""Project6 V4.2 priority/sentinel semantic contract.

The scientific contract is defined by the *node identities and roles*, not by a
particular byte-for-byte representation of the source files.  File SHA-256
values are still reported for lineage, but a harmless BOM/newline/metadata
change must not make the whole training package unimportable.

Authoritative roles
-------------------
* PFV_CORE8_V1: eight nodes used to compute the formal Priority Flooding Volume.
* SENTINEL2_V1: two monitoring/input-feature nodes; never substituted for PFV.
* PRIORITY_ZONE11_SENSITIVITY_V1: secondary sensitivity-analysis zone only.

Any semantic ID mismatch is fail-closed.  In particular, downstream code must
never fall back from PFV_CORE8 to Sentinel2.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Dict, List

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

CONTRACT_ID = "PFV_CORE8_V1"
SENTINEL_CONTRACT_ID = "SENTINEL2_V1"
SENSITIVITY_CONTRACT_ID = "PRIORITY_ZONE11_SENSITIVITY_V1"

# Frozen scientific identities.  These are deliberately independent of source
# file formatting/hash and reflect the research contract agreed for Project6.
_EXPECTED_PFV_CORE8 = (
    "MSLBZW001",
    "HS1316314",
    "YS2530050",
    "HS2529198",
    "MH0200773",
    "HS1330349",
    "HS2529139",
    "HS2529052",
)
_EXPECTED_SENTINEL2 = (
    "MH0200770",
    "HS1355904",
)
_EXPECTED_SENSITIVITY11 = (
    "YS2530050",
    "HS1316314",
    "HS2529139",
    "HS1330349",
    "MH0200770",
    "HS2529198",
    "HS1355904",
    "MH0249284",
    "HS2529052",
    "MSLBZW001",
    "MH0200773",
)

_SOURCE_FILES = {
    "pfv_core8": "data/project5_design/priority_pfv_core_nodes.txt",
    "sentinel2": "data/project6_v3_sentinel_nodes.txt",
    "sensitivity11": "data/project2_design/priority_zone_nodes.csv",
}


class PriorityContractError(Exception):
    """Raised when the semantic priority-node contract cannot be verified."""


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    p = Path(path)
    if not p.is_file():
        raise PriorityContractError(f"File not found: {p}")
    h.update(p.read_bytes())
    return h.hexdigest()


def _load_txt_ids(rel_path: str) -> List[str]:
    full = _PROJECT_ROOT / rel_path
    if not full.is_file():
        raise PriorityContractError(f"Priority contract file missing: {rel_path}")
    return [
        line.strip()
        for line in full.read_text(encoding="utf-8-sig").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _load_csv_ids(rel_path: str, column: str = "node_id") -> List[str]:
    full = _PROJECT_ROOT / rel_path
    if not full.is_file():
        raise PriorityContractError(f"Priority contract file missing: {rel_path}")
    ids: List[str] = []
    with open(full, encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or column not in reader.fieldnames:
            raise PriorityContractError(
                f"Priority CSV {rel_path} is missing required column {column!r}"
            )
        for row in reader:
            val = str(row.get(column, "")).strip()
            if val:
                ids.append(val)
    return ids


def _require_exact_semantic_ids(
    *,
    role: str,
    observed: List[str],
    expected: tuple[str, ...],
) -> List[str]:
    if observed != list(expected):
        missing = [x for x in expected if x not in observed]
        extra = [x for x in observed if x not in expected]
        order_only = not missing and not extra and set(observed) == set(expected)
        raise PriorityContractError(
            f"{role} semantic node contract mismatch. "
            f"expected={list(expected)}, observed={observed}, "
            f"missing={missing}, extra={extra}, order_only={order_only}"
        )
    if len(set(observed)) != len(observed):
        raise PriorityContractError(f"{role} contains duplicate node IDs")
    return observed


# Load source files and validate their *contents* against the frozen scientific
# identities.  Byte hashes are lineage evidence only and never the scientific
# admission criterion.
PFV_CORE_8_IDS: List[str] = _require_exact_semantic_ids(
    role="PFV_CORE8_V1",
    observed=_load_txt_ids(_SOURCE_FILES["pfv_core8"]),
    expected=_EXPECTED_PFV_CORE8,
)
DEPTH_SENTINEL_2_IDS: List[str] = _require_exact_semantic_ids(
    role="SENTINEL2_V1",
    observed=_load_txt_ids(_SOURCE_FILES["sentinel2"]),
    expected=_EXPECTED_SENTINEL2,
)
SENSITIVITY_ZONE_11_IDS: List[str] = _require_exact_semantic_ids(
    role="PRIORITY_ZONE11_SENSITIVITY_V1",
    observed=_load_csv_ids(_SOURCE_FILES["sensitivity11"]),
    expected=_EXPECTED_SENSITIVITY11,
)

_overlap_pfv_sentinel = set(PFV_CORE_8_IDS) & set(DEPTH_SENTINEL_2_IDS)
if _overlap_pfv_sentinel:
    raise PriorityContractError(
        f"PFV core and sentinel nodes must not overlap: {_overlap_pfv_sentinel}"
    )


def _indices_exact(graph_node_ids: List[str], expected_ids: List[str], role: str) -> List[int]:
    if len(set(graph_node_ids)) != len(graph_node_ids):
        raise PriorityContractError("Graph node IDs are not unique")
    id_to_idx: Dict[str, int] = {nid: i for i, nid in enumerate(graph_node_ids)}
    missing = [nid for nid in expected_ids if nid not in id_to_idx]
    if missing:
        raise PriorityContractError(f"{role} nodes not found in graph: {missing}")
    return [id_to_idx[nid] for nid in expected_ids]


def get_pfv_core_node_indices(graph_node_ids: List[str]) -> List[int]:
    """Indices of the eight formal PFV nodes in canonical PFV order."""
    return _indices_exact(graph_node_ids, PFV_CORE_8_IDS, CONTRACT_ID)


def get_sentinel_node_indices(graph_node_ids: List[str]) -> List[int]:
    """Indices of the two monitoring sentinels; never a PFV fallback."""
    return _indices_exact(graph_node_ids, DEPTH_SENTINEL_2_IDS, SENTINEL_CONTRACT_ID)


def audit_contract() -> dict:
    """Return semantic contract and byte-lineage evidence.

    A file SHA change alone is not a contract failure; exact node IDs/order and
    their scientific roles are the admission criteria.
    """
    errors: List[str] = []
    source_lineage: dict[str, dict[str, str]] = {}
    for role, rel in _SOURCE_FILES.items():
        full = _PROJECT_ROOT / rel
        if not full.is_file():
            errors.append(f"Missing file: {rel}")
            continue
        source_lineage[role] = {"path": rel, "sha256": sha256_file(full)}

    if PFV_CORE_8_IDS != list(_EXPECTED_PFV_CORE8):
        errors.append("PFV_CORE8 semantic IDs differ from frozen contract")
    if DEPTH_SENTINEL_2_IDS != list(_EXPECTED_SENTINEL2):
        errors.append("SENTINEL2 semantic IDs differ from frozen contract")
    if SENSITIVITY_ZONE_11_IDS != list(_EXPECTED_SENSITIVITY11):
        errors.append("Sensitivity11 semantic IDs differ from frozen contract")
    overlap = set(PFV_CORE_8_IDS) & set(DEPTH_SENTINEL_2_IDS)
    if overlap:
        errors.append(f"PFV/sentinel overlap: {sorted(overlap)}")

    result = {
        "status": "CONTRACT_CONFLICT" if errors else "PASS",
        "contract_id": CONTRACT_ID,
        "sentinel_contract_id": SENTINEL_CONTRACT_ID,
        "sensitivity_contract_id": SENSITIVITY_CONTRACT_ID,
        "pfv_core_count": len(PFV_CORE_8_IDS),
        "sentinel_count": len(DEPTH_SENTINEL_2_IDS),
        "sensitivity_zone_count": len(SENSITIVITY_ZONE_11_IDS),
        "pfv_core_ids": PFV_CORE_8_IDS,
        "sentinel_ids": DEPTH_SENTINEL_2_IDS,
        "sensitivity_ids": SENSITIVITY_ZONE_11_IDS,
        "source_lineage": source_lineage,
        "hash_is_admission_criterion": False,
        "errors": errors,
    }
    return result


if __name__ == "__main__":
    print(f"Contract ID      : {CONTRACT_ID}")
    print(f"PFV core nodes   : {len(PFV_CORE_8_IDS)}  {PFV_CORE_8_IDS}")
    print(f"Sentinel nodes   : {len(DEPTH_SENTINEL_2_IDS)}  {DEPTH_SENTINEL_2_IDS}")
    print(f"Sensitivity zone : {len(SENSITIVITY_ZONE_11_IDS)}  {SENSITIVITY_ZONE_11_IDS}")
    print(f"Audit result     : {audit_contract()}")
