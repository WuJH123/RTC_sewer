"""V4.2 event usage ledger — build, audit, and write event provenance tracking.

Consolidates event sources from V4.1 ledger, event inventory, and Train1600
split assignments into a unified 25-column ledger with 7 audit checks.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LEDGER_DIR = "audits/v42_event_usage"

LEDGER_COLUMNS: list[str] = [
    "event_id",
    "rainfall_sha256",
    "rainfall_family",
    "return_period",
    "duration",
    "total_depth",
    "peak_intensity",
    "peak_time",
    "phase_coverage",
    "source_inventory",
    "contract_compatible",
    "used_in_pilot",
    "used_in_p3",
    "used_in_train1600_train",
    "used_in_v40_calibration",
    "used_in_v40_locked",
    "used_in_v41_calibration",
    "used_in_v41_locked",
    "used_in_challenge",
    "used_in_formal",
    "current_role",
    "eligible_for_v42_development",
    "eligible_for_v42_fresh_evaluation",
    "exclusion_reason",
]


# ---------------------------------------------------------------------------
# Event-ID parsing helpers
# ---------------------------------------------------------------------------

def _parse_return_period(event_id: str) -> int | None:
    """Extract return period from event_id, e.g. 'T100' -> 100."""
    parts = event_id.split("_")
    if parts and parts[0].startswith("T"):
        try:
            return int(parts[0][1:])
        except ValueError:
            pass
    return None


def _parse_duration(event_id: str) -> int | None:
    """Extract duration in minutes from event_id, e.g. 'D240' -> 240."""
    parts = event_id.split("_")
    for p in parts:
        if p.startswith("D") and len(p) > 1:
            try:
                return int(p[1:])
            except ValueError:
                pass
    return None


def _parse_rainfall_family(event_id: str) -> str:
    """Extract rainfall family from event_id.

    Convention: everything after the duration token, e.g.
    'T100_D240_chicago_late' -> 'chicago_late'
    'T100_D240_block'        -> 'block'
    'T100_D240_double_peak'  -> 'double_peak'
    """
    parts = event_id.split("_")
    # Skip T<return_period> and D<duration>
    idx = 0
    if parts and parts[0].startswith("T"):
        idx = 1
    if idx < len(parts) and parts[idx].startswith("D"):
        idx += 1
    return "_".join(parts[idx:]) if idx < len(parts) else "unknown"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_v41_ledger(inventory_dir: Path) -> pd.DataFrame:
    """Load V4.1 event_usage_ledger.csv."""
    path = inventory_dir / "event_usage_ledger.csv"
    if not path.exists():
        log.warning("V4.1 ledger not found: %s", path)
        return pd.DataFrame()
    df = pd.read_csv(path)
    log.info("Loaded V4.1 ledger: %d events from %s", len(df), path)
    return df


def _load_event_inventory(inventory_dir: Path) -> pd.DataFrame:
    """Load V4.1 event_inventory.csv."""
    path = inventory_dir / "event_inventory.csv"
    if not path.exists():
        log.warning("Event inventory not found: %s", path)
        return pd.DataFrame()
    df = pd.read_csv(path)
    log.info("Loaded event inventory: %d events from %s", len(df), path)
    return df


def _load_train1600_split(output_root: Path) -> dict[str, list[str]]:
    """Load Train1600 v3 event selection JSON."""
    path = output_root / "train1600_v3" / "planning" / "train_event_selection_v3.json"
    if not path.exists():
        log.warning("Train1600 split not found: %s", path)
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    selection = data.get("selection", {})
    log.info(
        "Loaded Train1600 split: train=%d, cal=%d, locked=%d, reserve=%d",
        len(selection.get("train", [])),
        len(selection.get("calibration", [])),
        len(selection.get("locked_validation", [])),
        len(selection.get("reserve", [])),
    )
    return selection


# ---------------------------------------------------------------------------
# Role determination
# ---------------------------------------------------------------------------

def _determine_current_role(
    used_in_train1600_train: bool,
    used_in_v40_calibration: bool,
    used_in_v40_locked: bool,
    used_in_v41_calibration: bool,
    used_in_v41_locked: bool,
    used_in_pilot: bool,
    used_in_challenge: bool,
    used_in_formal: bool,
    v41_assigned_split: str,
) -> str:
    """Determine the current_role for an event.

    Priority order:
    1. sealed — Challenge/Formal events
    2. consumed_development — any prior calibration/locked usage
    3. train1600_train — in the train split
    4. pilot — used in pilot
    5. available / fresh_candidate — never used
    """
    if used_in_challenge or used_in_formal:
        return "sealed"
    if (used_in_v40_calibration or used_in_v40_locked
            or used_in_v41_calibration or used_in_v41_locked):
        return "consumed_development"
    if used_in_train1600_train:
        return "train1600_train"
    if used_in_pilot:
        return "pilot"
    # Check V4.1 assigned_split for reserve / calibration / locked_validation
    if v41_assigned_split in ("calibration", "locked_validation"):
        return "consumed_development"
    if v41_assigned_split == "reserve":
        return "reserve"
    return "available"


def _determine_eligibility(
    used_in_train1600_train: bool,
    used_in_v40_calibration: bool,
    used_in_v40_locked: bool,
    used_in_v41_calibration: bool,
    used_in_v41_locked: bool,
    used_in_challenge: bool,
    used_in_formal: bool,
    contract_compatible: bool,
    current_role: str,
) -> tuple[bool, bool, str]:
    """Return (eligible_for_v42_development, eligible_for_v42_fresh_evaluation, exclusion_reason)."""
    reasons: list[str] = []

    # Development eligibility: any prior development usage
    dev_eligible = (
        used_in_train1600_train
        or used_in_v40_calibration or used_in_v40_locked
        or used_in_v41_calibration or used_in_v41_locked
    )

    # Fresh evaluation eligibility: never used anywhere + contract compatible
    any_used = (
        used_in_train1600_train
        or used_in_v40_calibration or used_in_v40_locked
        or used_in_v41_calibration or used_in_v41_locked
        or used_in_challenge or used_in_formal
    )
    fresh_eligible = (not any_used) and contract_compatible

    # Build exclusion reason
    if not dev_eligible and not fresh_eligible:
        if used_in_challenge or used_in_formal:
            reasons.append("sealed_challenge_formal")
        if not contract_compatible:
            reasons.append("contract_incompatible")
        if not reasons:
            reasons.append("no_prior_development_usage")
    if not fresh_eligible and not dev_eligible:
        if any_used and not (used_in_challenge or used_in_formal):
            reasons.append("consumed_by_development")

    exclusion_reason = ";".join(reasons) if reasons else ""
    return dev_eligible, fresh_eligible, exclusion_reason


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------

def build_v42_event_usage_ledger(
    project_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Build the V4.2 event usage ledger from all source files.

    Parameters
    ----------
    project_root : Path
        Project root (contains data/ directory).
    output_root : Path
        Output root (contains final_v4/inventory/ and train1600_v3/).

    Returns
    -------
    dict with keys:
        - ledger_df: pd.DataFrame with 25 columns
        - source_counts: dict of source event counts
    """
    inventory_dir = output_root / "inventory"

    # --- Load sources ---
    v41_ledger = _load_v41_ledger(inventory_dir)
    event_inv = _load_event_inventory(inventory_dir)
    train_split = _load_train1600_split(output_root)

    train_set = set(train_split.get("train", []))
    cal_set = set(train_split.get("calibration", []))
    locked_set = set(train_split.get("locked_validation", []))
    # reserve_set = set(train_split.get("reserve", []))

    # --- Build unified event set ---
    # Collect all event_ids from both sources
    ledger_ids: dict[str, dict] = {}
    if not v41_ledger.empty:
        for _, row in v41_ledger.iterrows():
            eid = row["event_id"]
            ledger_ids[eid] = {
                "rainfall_sha256": row.get("rainfall_sha256", ""),
                "v41_event_tier": row.get("event_tier", ""),
                "v41_opportunity_scanned": row.get("opportunity_scanned", False),
                "v41_used_gate2": row.get("used_gate2", False),
                "v41_used_gate3": row.get("used_gate3", False),
                "v41_used_gate4": row.get("used_gate4", False),
                "v41_used_gate5r": row.get("used_gate5r", False),
                "v41_used_peak_boundary": row.get("used_peak_boundary", False),
                "v41_used_pilot": row.get("used_pilot", False),
                "v41_used_train": row.get("used_train", False),
                "v41_used_calibration": row.get("used_calibration", False),
                "v41_used_locked_validation": row.get("used_locked_validation", False),
                "v41_used_challenge": row.get("used_challenge", False),
                "v41_used_formal": row.get("used_formal", False),
                "v41_assigned_split": row.get("assigned_split", ""),
            }

    inv_ids: dict[str, dict] = {}
    if not event_inv.empty:
        for _, row in event_inv.iterrows():
            eid = row["event_id"]
            inv_ids[eid] = {
                "rainfall_sha256": row.get("rainfall_sha256", row.get("rainfall_file_sha256", "")),
                "total_depth": row.get("total_depth", None),
                "peak_intensity": row.get("peak_intensity", None),
                "peak_time": row.get("peak_time", None),
                "provenance_status": row.get("provenance_status", ""),
                "revealed": row.get("revealed", False),
            }

    # Merge all event_ids
    all_event_ids = sorted(set(list(ledger_ids.keys()) + list(inv_ids.keys())))
    log.info("Total unique events: %d (ledger=%d, inventory=%d)",
             len(all_event_ids), len(ledger_ids), len(inv_ids))

    # --- Build rows ---
    rows: list[dict[str, Any]] = []
    for eid in all_event_ids:
        li = ledger_ids.get(eid, {})
        ii = inv_ids.get(eid, {})

        rainfall_sha = li.get("rainfall_sha256") or ii.get("rainfall_sha256", "")
        rainfall_family = _parse_rainfall_family(eid)
        return_period = _parse_return_period(eid)
        duration = _parse_duration(eid)

        # Phase coverage: which gates/phases this event was scanned in
        phases: list[str] = []
        if li.get("v41_opportunity_scanned"):
            phases.append("opportunity_scanned")
        if li.get("v41_used_gate2"):
            phases.append("gate2")
        if li.get("v41_used_gate3"):
            phases.append("gate3")
        if li.get("v41_used_gate4"):
            phases.append("gate4")
        if li.get("v41_used_gate5r"):
            phases.append("gate5r")
        if li.get("v41_used_peak_boundary"):
            phases.append("peak_boundary")
        phase_coverage = "|".join(phases) if phases else ""

        # Source inventory flag
        source_inventory = "ledger+inventory" if eid in ledger_ids and eid in inv_ids else (
            "ledger" if eid in ledger_ids else "inventory"
        )

        # Contract compatibility
        provenance = ii.get("provenance_status", "")
        contract_compatible = provenance in ("development_fit", "inventory", "") or eid not in inv_ids

        # Usage flags — map from V4.1 columns
        used_in_pilot = bool(li.get("v41_used_pilot", False))
        used_in_p3 = bool(
            li.get("v41_used_gate2", False)
            or li.get("v41_used_gate3", False)
            or li.get("v41_used_gate4", False)
            or li.get("v41_used_gate5r", False)
        )
        used_in_train1600_train = eid in train_set
        # V4.0/V4.1 calibration/locked: use V4.1 ledger columns
        # Since V4.1 has single used_calibration / used_locked_validation,
        # we treat them as V4.1 era (V4.0 had separate but we don't have
        # separate columns — map based on assigned_split era)
        v41_cal = bool(li.get("v41_used_calibration", False))
        v41_locked = bool(li.get("v41_used_locked_validation", False))
        v41_split = li.get("v41_assigned_split", "")

        # V4.0 vs V4.1 distinction: V4.0 events had no assigned_split or
        # different run_uuid.  In the V4.1 ledger, calibration/locked events
        # from Train1600 v3 have assignment_run_uuid.  Events without run_uuid
        # but with used_calibration/used_locked are from earlier V4.0 era.
        # For simplicity, map V4.1 ledger used_calibration -> used_in_v41_calibration
        # and used_locked_validation -> used_in_v41_locked, since the V4.1
        # ledger IS the V4.1 tracking.
        used_in_v40_calibration = False  # No separate V4.0 data source
        used_in_v40_locked = False
        used_in_v41_calibration = v41_cal
        used_in_v41_locked = v41_locked

        used_in_challenge = bool(li.get("v41_used_challenge", False))
        used_in_formal = bool(li.get("v41_used_formal", False))

        # Current role
        current_role = _determine_current_role(
            used_in_train1600_train=used_in_train1600_train,
            used_in_v40_calibration=used_in_v40_calibration,
            used_in_v40_locked=used_in_v40_locked,
            used_in_v41_calibration=used_in_v41_calibration,
            used_in_v41_locked=used_in_v41_locked,
            used_in_pilot=used_in_pilot,
            used_in_challenge=used_in_challenge,
            used_in_formal=used_in_formal,
            v41_assigned_split=v41_split,
        )

        # Eligibility
        dev_eligible, fresh_eligible, exclusion_reason = _determine_eligibility(
            used_in_train1600_train=used_in_train1600_train,
            used_in_v40_calibration=used_in_v40_calibration,
            used_in_v40_locked=used_in_v40_locked,
            used_in_v41_calibration=used_in_v41_calibration,
            used_in_v41_locked=used_in_v41_locked,
            used_in_challenge=used_in_challenge,
            used_in_formal=used_in_formal,
            contract_compatible=contract_compatible,
            current_role=current_role,
        )

        rows.append({
            "event_id": eid,
            "rainfall_sha256": rainfall_sha,
            "rainfall_family": rainfall_family,
            "return_period": return_period,
            "duration": duration,
            "total_depth": ii.get("total_depth"),
            "peak_intensity": ii.get("peak_intensity"),
            "peak_time": ii.get("peak_time"),
            "phase_coverage": phase_coverage,
            "source_inventory": source_inventory,
            "contract_compatible": contract_compatible,
            "used_in_pilot": used_in_pilot,
            "used_in_p3": used_in_p3,
            "used_in_train1600_train": used_in_train1600_train,
            "used_in_v40_calibration": used_in_v40_calibration,
            "used_in_v40_locked": used_in_v40_locked,
            "used_in_v41_calibration": used_in_v41_calibration,
            "used_in_v41_locked": used_in_v41_locked,
            "used_in_challenge": used_in_challenge,
            "used_in_formal": used_in_formal,
            "current_role": current_role,
            "eligible_for_v42_development": dev_eligible,
            "eligible_for_v42_fresh_evaluation": fresh_eligible,
            "exclusion_reason": exclusion_reason,
        })

    ledger_df = pd.DataFrame(rows, columns=LEDGER_COLUMNS)

    source_counts = {
        "v41_ledger_events": len(ledger_ids),
        "event_inventory_events": len(inv_ids),
        "merged_total": len(all_event_ids),
        "train1600_train": len(train_set),
        "train1600_calibration": len(cal_set),
        "train1600_locked": len(locked_set),
    }

    log.info("Built V4.2 ledger: %d rows, columns=%s", len(ledger_df), list(ledger_df.columns))
    return {"ledger_df": ledger_df, "source_counts": source_counts}


# ---------------------------------------------------------------------------
# Audit: 7 checks
# ---------------------------------------------------------------------------

def audit_v42_event_usage_ledger(ledger_df: pd.DataFrame) -> dict[str, Any]:
    """Run 7 audit checks on the V4.2 event usage ledger.

    Returns dict with per-check results and overall pass/blocked status.
    """
    checks: dict[str, dict[str, Any]] = {}

    # 1. event_id uniqueness
    dup_ids = ledger_df[ledger_df["event_id"].duplicated(keep=False)]
    checks["no_duplicate_event_ids"] = {
        "pass": len(dup_ids) == 0,
        "duplicates": dup_ids["event_id"].unique().tolist() if len(dup_ids) > 0 else [],
    }

    # 2. rainfall_sha256 no cross-role duplication
    sha_role_conflicts: list[dict] = []
    sha_groups = ledger_df.groupby("rainfall_sha256")
    for sha, grp in sha_groups:
        if sha == "" or len(grp) < 2:
            continue
        roles = set(grp["current_role"].unique())
        # Fresh evaluation events must not share SHA with any development event
        if "fresh_candidate" in roles or "available" in roles:
            dev_roles = roles - {"fresh_candidate", "available"}
            if dev_roles:
                sha_role_conflicts.append({
                    "rainfall_sha256": sha,
                    "roles": sorted(roles),
                    "event_ids": grp["event_id"].tolist(),
                })
    checks["no_cross_role_sha_conflicts"] = {
        "pass": len(sha_role_conflicts) == 0,
        "conflicts": sha_role_conflicts,
    }

    # 3. Same event not in multiple splits
    multi_split_events: list[str] = []
    split_cols = [
        "used_in_train1600_train",
        "used_in_v41_calibration",
        "used_in_v41_locked",
    ]
    for _, row in ledger_df.iterrows():
        n_splits = sum(1 for c in split_cols if row.get(c, False))
        if n_splits > 1:
            multi_split_events.append(row["event_id"])
    checks["no_multi_split_events"] = {
        "pass": len(multi_split_events) == 0,
        "events": multi_split_events,
    }

    # 4. V4.0/V4.1 locked not marked as fresh
    locked_as_fresh: list[str] = []
    for _, row in ledger_df.iterrows():
        if (row.get("used_in_v41_locked", False) or row.get("used_in_v40_locked", False)):
            if row.get("eligible_for_v42_fresh_evaluation", False):
                locked_as_fresh.append(row["event_id"])
    checks["locked_not_fresh"] = {
        "pass": len(locked_as_fresh) == 0,
        "events": locked_as_fresh,
    }

    # 5. Challenge/Formal events fully excluded
    challenge_formal_not_excluded: list[str] = []
    for _, row in ledger_df.iterrows():
        if row.get("used_in_challenge", False) or row.get("used_in_formal", False):
            if row.get("eligible_for_v42_development", False) or row.get("eligible_for_v42_fresh_evaluation", False):
                challenge_formal_not_excluded.append(row["event_id"])
    checks["challenge_formal_excluded"] = {
        "pass": len(challenge_formal_not_excluded) == 0,
        "events": challenge_formal_not_excluded,
    }

    # 6. Contract-incompatible data not mixed in
    contract_violations: list[str] = []
    for _, row in ledger_df.iterrows():
        if not row.get("contract_compatible", True):
            if row.get("eligible_for_v42_development", False) or row.get("eligible_for_v42_fresh_evaluation", False):
                contract_violations.append(row["event_id"])
    checks["contract_compatible_only"] = {
        "pass": len(contract_violations) == 0,
        "events": contract_violations,
    }

    # 7. Report unpolluted event totals
    total_events = len(ledger_df)
    fresh_count = int(ledger_df["eligible_for_v42_fresh_evaluation"].sum())
    dev_count = int(ledger_df["eligible_for_v42_development"].sum())
    sealed_count = int((ledger_df["current_role"] == "sealed").sum())
    checks["unpolluted_event_totals"] = {
        "pass": True,
        "total_events": total_events,
        "fresh_evaluation": fresh_count,
        "consumed_development": dev_count,
        "sealed": sealed_count,
    }

    # Overall status
    all_pass = all(c["pass"] for c in checks.values())
    return {
        "status": "pass" if all_pass else "blocked",
        "exit_code": 0 if all_pass else 2,
        "checks": checks,
        "summary": {
            "total_events": total_events,
            "n_checks": len(checks),
            "n_passed": sum(1 for c in checks.values() if c["pass"]),
            "n_failed": sum(1 for c in checks.values() if not c["pass"]),
        },
    }


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------

def write_v42_event_ledger_outputs(
    output_root: Path,
    ledger_df: pd.DataFrame,
    audit_result: dict[str, Any],
) -> dict[str, Any]:
    """Write 6 output files to audits/v42_event_usage/.

    Returns status dict with exit_code and audit summary.
    """
    out_dir = output_root / LEDGER_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Full ledger CSV
    ledger_path = out_dir / "event_usage_ledger_v42.csv"
    ledger_df.to_csv(ledger_path, index=False)
    log.info("Wrote ledger: %s (%d rows)", ledger_path, len(ledger_df))

    # 2. Role conflicts (events appearing in multiple roles)
    conflict_rows: list[dict] = []
    for _, row in ledger_df.iterrows():
        n_roles = 0
        if row.get("used_in_train1600_train"):
            n_roles += 1
        if row.get("used_in_v41_calibration") or row.get("used_in_v40_calibration"):
            n_roles += 1
        if row.get("used_in_v41_locked") or row.get("used_in_v40_locked"):
            n_roles += 1
        if row.get("used_in_challenge"):
            n_roles += 1
        if row.get("used_in_formal"):
            n_roles += 1
        if n_roles > 1:
            conflict_rows.append({
                "event_id": row["event_id"],
                "current_role": row["current_role"],
                "n_roles_assigned": n_roles,
            })
    conflicts_df = pd.DataFrame(conflict_rows)
    conflicts_path = out_dir / "event_role_conflicts.csv"
    conflicts_df.to_csv(conflicts_path, index=False)
    log.info("Wrote role conflicts: %s (%d rows)", conflicts_path, len(conflicts_df))

    # 3. Rainfall SHA conflicts
    sha_conflicts = audit_result.get("checks", {}).get(
        "no_cross_role_sha_conflicts", {}
    ).get("conflicts", [])
    sha_conflicts_df = pd.DataFrame(sha_conflicts)
    sha_path = out_dir / "rainfall_sha_conflicts.csv"
    sha_conflicts_df.to_csv(sha_path, index=False)
    log.info("Wrote SHA conflicts: %s (%d rows)", sha_path, len(sha_conflicts_df))

    # 4. Fresh evaluation inventory
    fresh_mask = ledger_df["eligible_for_v42_fresh_evaluation"] == True  # noqa: E712
    fresh_df = ledger_df[fresh_mask].copy()
    fresh_path = out_dir / "fresh_evaluation_inventory.csv"
    fresh_df.to_csv(fresh_path, index=False)
    log.info("Wrote fresh evaluation inventory: %s (%d rows)", fresh_path, len(fresh_df))

    # 5. Consumed development inventory
    dev_mask = ledger_df["eligible_for_v42_development"] == True  # noqa: E712
    dev_df = ledger_df[dev_mask].copy()
    dev_path = out_dir / "consumed_development_inventory.csv"
    dev_df.to_csv(dev_path, index=False)
    log.info("Wrote consumed development inventory: %s (%d rows)", dev_path, len(dev_df))

    # 6. Audit JSON
    audit_path = out_dir / "event_usage_audit.json"
    audit_path.write_text(
        json.dumps(audit_result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    log.info("Wrote audit JSON: %s", audit_path)

    status = audit_result.get("status", "blocked")
    exit_code = audit_result.get("exit_code", 2)
    return {
        "status": status,
        "exit_code": exit_code,
        "audit": audit_result.get("summary", {}),
    }
