from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pandas as pd

from sewerrtc.simulation.pyswmm_runner import physical_network_sha256


NETWORK_VARIANT = "rainfall_only_no_dwf"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def active_dwf_flow_rows(path: str | Path) -> int:
    """Count non-comment FLOW rows in the INP [DWF] section."""
    count = 0
    in_dwf = False
    for raw in Path(path).read_text(
        encoding="utf-8", errors="replace"
    ).splitlines():
        stripped = raw.strip()
        upper = stripped.upper()
        if upper.startswith("[") and upper.endswith("]"):
            in_dwf = upper == "[DWF]"
            continue
        if not in_dwf or not stripped or stripped.startswith(";"):
            continue
        tokens = stripped.split()
        if len(tokens) >= 2 and tokens[1].upper() == "FLOW":
            count += 1
    return count


def audit_network(
    path: str | Path,
    *,
    expected_sha256: str | None = None,
    expected_physical_sha256: str | None = None,
) -> dict[str, Any]:
    network = Path(path)
    if not network.exists():
        return {
            "status": "blocked",
            "reason": "network_missing",
            "network_path": str(network),
            "active_dwf_flow_rows": None,
        }
    full_sha = sha256_file(network)
    physical_sha = physical_network_sha256(network)
    dwf_rows = active_dwf_flow_rows(network)
    checks = {
        "network_exists": True,
        "no_active_dwf_flow": dwf_rows == 0,
        "network_sha256_matches": (
            expected_sha256 is None or full_sha == str(expected_sha256)
        ),
        "physical_network_sha256_matches": (
            expected_physical_sha256 is None
            or physical_sha == str(expected_physical_sha256)
        ),
    }
    return {
        "status": "pass" if all(checks.values()) else "blocked",
        "checks": checks,
        "network_variant": NETWORK_VARIANT,
        "network_path": str(network.resolve()),
        "network_sha256": full_sha,
        "physical_network_sha256": physical_sha,
        "active_dwf_flow_rows": dwf_rows,
        "runtime_dwf_mutation_permitted": False,
    }


def audit_final_contract(contract: dict[str, Any], project_root: str | Path) -> dict:
    root = Path(project_root)
    network = root / str(contract.get("network_path", ""))
    order_path = root / str(contract.get("canonical_facility_order", ""))
    semantics_path = root / str(contract.get("facility_semantics", ""))
    network_audit = audit_network(
        network,
        expected_sha256=contract.get("network_sha256"),
        expected_physical_sha256=contract.get("physical_network_sha256"),
    )
    order_ids: list[str] = []
    if order_path.exists():
        order_ids = [
            line.strip()
            for line in order_path.read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
            if line.strip() and not line.lstrip().startswith(("#", ";"))
        ]
    semantics = (
        pd.read_csv(semantics_path)
        if semantics_path.exists()
        else pd.DataFrame()
    )
    semantics_by_id = (
        semantics.set_index("facility_id")
        if "facility_id" in semantics
        else pd.DataFrame()
    )
    binary_ids = set(contract.get("binary_facilities", []))
    binary_semantics_ok = bool(binary_ids) and all(
        facility in semantics_by_id.index
        and str(
            semantics_by_id.loc[facility, "binary_or_continuous"]
        ).lower()
        == "binary"
        for facility in binary_ids
    )
    variable_speed_ids = set(contract.get("variable_speed_facilities", []))
    variable_speed_semantics_ok = bool(variable_speed_ids) and all(
        facility in semantics_by_id.index
        and str(
            semantics_by_id.loc[facility, "pump_control_mode"]
        ).lower()
        == "variable_speed"
        and str(
            semantics_by_id.loc[facility, "binary_or_continuous"]
        ).lower()
        == "continuous"
        for facility in variable_speed_ids
    )
    checks = {
        "network": network_audit["status"] == "pass",
        "network_variant": contract.get("network_variant") == NETWORK_VARIANT,
        "engineering36": (
            contract.get("facility_count") == 36
            and len(order_ids) == 36
            and len(set(order_ids)) == 36
        ),
        "canonical_order_exists": order_path.exists(),
        "canonical_order_sha256": (
            order_path.exists()
            and sha256_file(order_path)
            == contract.get("canonical_facility_order_sha256")
        ),
        "facility_semantics_exists": semantics_path.exists(),
        "facility_semantics_sha256": (
            semantics_path.exists()
            and sha256_file(semantics_path)
            == contract.get("facility_semantics_sha256")
        ),
        "facility_semantics_cover_order": (
            not semantics.empty
            and set(order_ids)
            == set(semantics["facility_id"].astype(str))
        ),
        "binary_semantics": binary_semantics_ok,
        "variable_speed_semantics": variable_speed_semantics_ok,
        "record_step_300": contract.get("state_record_step_sec") == 300,
        "control_interval_600": contract.get("control_interval_sec") == 600,
        "history_60_7": (
            contract.get("history_min") == 60
            and contract.get("history_frames") == 7
        ),
        "h120_12": (
            contract.get("horizon_min") == 120
            and contract.get("horizon_steps") == 12
        ),
        "k_at_most_8": contract.get("max_active_changes") == 8,
        "no_hotstart": contract.get("use_hotstart") is False,
        "authoritative_swmm": contract.get("authoritative_swmm") is True,
        "candidate_minus_reference": (
            contract.get("delta_direction") == "candidate_minus_reference"
        ),
        "full_fail_closed": (
            contract.get("full_label_policy")
            == "NaN_unless_full_event_eligible_true"
        ),
    }
    return {
        "status": "pass" if all(checks.values()) else "blocked",
        "checks": checks,
        "network": network_audit,
        "canonical_facility_order": {
            "path": str(order_path.resolve()) if order_path.exists() else str(order_path),
            "count": len(order_ids),
            "sha256": sha256_file(order_path) if order_path.exists() else None,
        },
        "facility_semantics": {
            "path": (
                str(semantics_path.resolve())
                if semantics_path.exists()
                else str(semantics_path)
            ),
            "count": int(len(semantics)),
            "sha256": (
                sha256_file(semantics_path)
                if semantics_path.exists()
                else None
            ),
        },
    }
