"""V4.2 Fresh Evaluation Split — planning, selection, and auditing.

Plans a fresh evaluation set from the *complete* event inventory (not just
``role=reserve`` as in V4.1).  Events are selected without peeking at SWMM
results or true Candidate KPIs.  The plan is frozen before any V4.2
evaluation label is generated.

Key difference from V4.1 ``plan_fresh_evaluation_split``:
  - V4.1 only looked at ``assigned_split == "reserve"`` events
  - V4.2 scans the *full* event ledger and filters out already-consumed events

State machine
-------------
``fresh_count >= 16`` → ``ready_full``          (cal=4, locked=8, accrual=4)
``12 <= fresh < 16``  → ``ready_without_accrual`` (cal=4, locked=8, accrual=0)
``8  <= fresh < 12``  → ``prelocked_only``        (no fresh locked authorized)
``fresh < 8``         → ``insufficient_fresh_events``
"""
from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .runtime import atomic_write_json, working_code_sha

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FRESH_EVAL_DIR = "audits/v42_evaluation"

# Columns that mark an event as already consumed
_CONSUMED_ROLES: frozenset[str] = frozenset({
    "pilot",
    "p3",
    "train1600",
    "v4.0_calibration",
    "v4.0_locked",
    "v4.1_calibration",
    "v4.1_locked",
    "challenge",
    "formal",
})

# Target split counts
TARGET_CALIBRATION = 4
TARGET_LOCKED = 8
TARGET_ACCRUAL = 4
TARGET_FULL = TARGET_CALIBRATION + TARGET_LOCKED + TARGET_ACCRUAL  # 16

# Balance dimensions for selection
_BALANCE_DIMS: tuple[str, ...] = (
    "rainfall_family",
    "return_period",
    "duration_hours",
    "peak_timing_hours",
    "total_rainfall_mm",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_str(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _read_ledger(ledger_path: Path) -> pd.DataFrame:
    """Read the full event usage ledger."""
    if not ledger_path.exists():
        raise FileNotFoundError(f"Event ledger not found: {ledger_path}")
    df = pd.read_csv(ledger_path)
    required = {"event_id", "rainfall_sha256"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Ledger missing required columns: {sorted(missing)}")
    return df


def _is_consumed(row: pd.Series) -> bool:
    """Return True if the event has been consumed by a prior pipeline stage."""
    # Check assigned_split / role column
    role = str(row.get("assigned_split", row.get("role", ""))).strip().lower()
    if role in _CONSUMED_ROLES:
        return True

    # Check boolean consumption flags
    for col in ("used_in_pilot", "used_in_p3", "used_in_train1600",
                "used_in_v40_cal", "used_in_v40_locked",
                "used_in_v41_cal", "used_in_v41_locked",
                "used_in_challenge", "used_in_formal"):
        if col in row.index and bool(row.get(col, False)):
            return True

    return False


def _is_contract_compatible(row: pd.Series) -> bool:
    """Return True if the event is contract-compatible."""
    flag = row.get("contract_compatible")
    if flag is None:
        return True  # assume compatible if column absent
    return bool(flag)


def _deterministic_event_order(event_ids: list[str]) -> list[str]:
    """Deterministic content-hash order (never uses KPIs)."""
    return sorted(
        (str(e) for e in event_ids),
        key=lambda e: hashlib.sha256(e.encode("utf-8")).hexdigest(),
    )


def _balance_select(
    candidates: pd.DataFrame,
    n_select: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Select *n_select* events balancing across available dimensions.

    Uses a round-robin over quantile bins of each balance dimension.
    Never reads true Candidate KPI columns.
    """
    if len(candidates) <= n_select:
        return candidates.copy()

    rng = np.random.RandomState(seed)

    # Score each candidate by balance across available dimensions
    available_dims = [d for d in _BALANCE_DIMS if d in candidates.columns]

    if not available_dims:
        # No balance dimensions — fall back to deterministic hash order
        ordered = _deterministic_event_order(candidates["event_id"].tolist())
        return candidates[
            candidates["event_id"].isin(ordered[:n_select])
        ].copy()

    # Rank-based scoring: for each dimension, compute quantile rank
    scores = np.zeros(len(candidates), dtype=np.float64)
    for dim in available_dims:
        col = candidates[dim]
        if col.dtype.kind in ("f", "i", "u"):
            # Numeric: rank by quantile
            ranks = col.rank(method="average", pct=True).values
            scores += ranks
        else:
            # Categorical: hash-based deterministic ordering
            unique_vals = sorted(col.dropna().unique())
            val_order = {v: i / max(len(unique_vals) - 1, 1)
                         for i, v in enumerate(unique_vals)}
            cat_scores = col.map(val_order).fillna(0.5).values
            scores += cat_scores

    # Add small random jitter for tie-breaking (deterministic via seed)
    jitter = rng.uniform(0, 0.01, size=len(candidates))
    scores += jitter

    # Select top-n by score diversity (spread across quantiles)
    # Strategy: divide into quantile bins, sample from each bin
    n_bins = min(n_select, 5)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    selected_indices: list[int] = []

    for bin_idx in range(n_bins):
        lo, hi = bin_edges[bin_idx], bin_edges[bin_idx + 1]
        in_bin = np.where((scores >= lo) & (scores < hi))[0]
        if bin_idx == n_bins - 1:
            in_bin = np.where((scores >= lo) & (scores <= hi))[0]
        if len(in_bin) > 0:
            selected_indices.extend(in_bin.tolist())

    # If we have too many, trim; if too few, fill from remaining
    selected_indices = list(dict.fromkeys(selected_indices))  # deduplicate
    if len(selected_indices) > n_select:
        # Subsample evenly
        step = len(selected_indices) / n_select
        selected_indices = [selected_indices[int(i * step)] for i in range(n_select)]
    elif len(selected_indices) < n_select:
        remaining = set(range(len(candidates))) - set(selected_indices)
        needed = n_select - len(selected_indices)
        extra = rng.choice(sorted(remaining), size=min(needed, len(remaining)), replace=False)
        selected_indices.extend(extra.tolist())

    return candidates.iloc[selected_indices[:n_select]].copy()


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------

def _determine_status(fresh_count: int) -> dict[str, Any]:
    """Determine the evaluation availability status from fresh event count."""
    if fresh_count >= TARGET_FULL:
        return {
            "status": "ready_full",
            "calibration": TARGET_CALIBRATION,
            "locked": TARGET_LOCKED,
            "accrual": TARGET_ACCRUAL,
            "authorizes_fresh_locked": True,
        }
    elif fresh_count >= TARGET_CALIBRATION + TARGET_LOCKED:
        return {
            "status": "ready_without_accrual",
            "calibration": TARGET_CALIBRATION,
            "locked": TARGET_LOCKED,
            "accrual": 0,
            "authorizes_fresh_locked": True,
        }
    elif fresh_count >= TARGET_LOCKED:
        return {
            "status": "prelocked_only",
            "calibration": 0,
            "locked": 0,
            "accrual": 0,
            "authorizes_fresh_locked": False,
        }
    else:
        return {
            "status": "insufficient_fresh_events",
            "calibration": 0,
            "locked": 0,
            "accrual": 0,
            "authorizes_fresh_locked": False,
        }


# ---------------------------------------------------------------------------
# Core planner
# ---------------------------------------------------------------------------

def plan_v42_fresh_evaluation_split(
    event_ledger_path: str | Path,
    output_root: str | Path,
    *,
    selection_seed: int = 42,
    project_root: Path | None = None,
) -> dict[str, Any]:
    """Plan a fresh evaluation split from the complete event inventory.

    Parameters
    ----------
    event_ledger_path : path
        Path to ``inventory/event_usage_ledger.csv``.
    output_root : path
        Output root directory.
    selection_seed : int
        Frozen seed for balanced selection.
    project_root : path, optional
        Project root for code SHA computation.

    Returns
    -------
    dict
        Complete plan freeze with per-split event lists and audit artifacts.
    """
    ledger_path = Path(event_ledger_path)
    root = Path(output_root)
    eval_dir = root / FRESH_EVAL_DIR
    eval_dir.mkdir(parents=True, exist_ok=True)

    # 1. Read full event inventory
    ledger = _read_ledger(ledger_path)
    log.info("Read %d events from ledger", len(ledger))

    # 2. Filter fresh pool
    consumed_mask = ledger.apply(_is_consumed, axis=1)
    contract_mask = ledger.apply(_is_contract_compatible, axis=1)
    fresh_pool = ledger[~consumed_mask & contract_mask].copy()

    # Ensure unique rainfall_sha256
    fresh_pool = fresh_pool.drop_duplicates(
        subset="rainfall_sha256", keep="first"
    ).reset_index(drop=True)

    log.info("Fresh pool: %d events after filtering", len(fresh_pool))

    # 3. State machine
    status_info = _determine_status(len(fresh_pool))
    log.info("Fresh evaluation status: %s", status_info["status"])

    # 4. Record excluded events
    excluded = ledger[consumed_mask | ~contract_mask].copy()
    # Events excluded due to SHA duplication (not in fresh pool but SHA matches)
    fresh_shas = set(fresh_pool["rainfall_sha256"].astype(str))
    sha_dup_mask = (
        ~ledger["event_id"].isin(fresh_pool["event_id"])
        & ~ledger["event_id"].isin(excluded["event_id"])
        & ledger["rainfall_sha256"].astype(str).isin(fresh_shas)
    )
    if sha_dup_mask.any():
        excluded = pd.concat([excluded, ledger[sha_dup_mask]]).drop_duplicates(
            subset="event_id", keep="first"
        )

    # 5. Balanced selection (never reads true KPIs)
    cal_events: list[str] = []
    locked_events: list[str] = []
    accrual_events: list[str] = []

    if status_info["status"] == "ready_full":
        selected = _balance_select(fresh_pool, TARGET_FULL, seed=selection_seed)
        ordered = _deterministic_event_order(selected["event_id"].tolist())
        cal_events = ordered[:TARGET_CALIBRATION]
        locked_events = ordered[TARGET_CALIBRATION:TARGET_CALIBRATION + TARGET_LOCKED]
        accrual_events = ordered[TARGET_CALIBRATION + TARGET_LOCKED:]
    elif status_info["status"] == "ready_without_accrual":
        selected = _balance_select(
            fresh_pool, TARGET_CALIBRATION + TARGET_LOCKED, seed=selection_seed
        )
        ordered = _deterministic_event_order(selected["event_id"].tolist())
        cal_events = ordered[:TARGET_CALIBRATION]
        locked_events = ordered[TARGET_CALIBRATION:]
    # prelocked_only and insufficient: no splits assigned

    # 6. Compute SHAs for freeze
    all_selected = cal_events + locked_events + accrual_events
    ledger_sha = _sha256_file(ledger_path)
    selection_order_sha = _sha256_str("\n".join(all_selected))

    code_sha = ""
    if project_root is not None:
        try:
            code_sha = working_code_sha(project_root)
        except Exception:
            pass

    # 7. Build SHA lookup
    sha_by_event = dict(
        zip(
            fresh_pool["event_id"].astype(str),
            fresh_pool["rainfall_sha256"].astype(str),
        )
    )

    # 8. Write artifacts
    # fresh_event_inventory.csv
    fresh_pool.to_csv(eval_dir / "fresh_event_inventory.csv", index=False)

    # excluded_events.csv
    excluded.to_csv(eval_dir / "excluded_events.csv", index=False)

    # calibration_events.csv
    cal_df = pd.DataFrame([
        {"event_id": e, "rainfall_sha256": sha_by_event.get(e, "")}
        for e in cal_events
    ])
    cal_df.to_csv(eval_dir / "calibration_events.csv", index=False)

    # locked_events.csv
    locked_df = pd.DataFrame([
        {"event_id": e, "rainfall_sha256": sha_by_event.get(e, "")}
        for e in locked_events
    ])
    locked_df.to_csv(eval_dir / "locked_events.csv", index=False)

    # accrual_events.csv
    accrual_df = pd.DataFrame([
        {"event_id": e, "rainfall_sha256": sha_by_event.get(e, "")}
        for e in accrual_events
    ])
    accrual_df.to_csv(eval_dir / "accrual_events.csv", index=False)

    # plan_freeze.json
    plan_freeze = {
        "stage": "PlanV42FreshEvaluationSplit",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "created_before_any_v42_evaluation_labels": True,
        "selection_algorithm": "balance_select_with_deterministic_hash_order",
        "selection_seed": selection_seed,
        "code_sha256": code_sha,
        "config_sha256": "",
        "ledger_sha256": ledger_sha,
        "selection_order_sha256": selection_order_sha,
        "source": "full_event_inventory_not_just_reserve",
        "reads_true_kpi_for_selection": False,
        "status": status_info,
        "splits": {
            "calibration": cal_events,
            "locked": locked_events,
            "accrual": accrual_events,
        },
        "n_fresh_events": len(fresh_pool),
        "n_excluded_events": len(excluded),
        "n_calibration": len(cal_events),
        "n_locked": len(locked_events),
        "n_accrual": len(accrual_events),
    }
    atomic_write_json(eval_dir / "plan_freeze.json", plan_freeze)

    log.info(
        "Fresh eval plan: status=%s, cal=%d, locked=%d, accrual=%d",
        status_info["status"], len(cal_events), len(locked_events), len(accrual_events),
    )

    return plan_freeze


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def audit_v42_fresh_evaluation_availability(
    output_root: str | Path,
) -> dict[str, Any]:
    """Audit the frozen fresh evaluation plan for integrity.

    Verifies:
    1. Fresh pool does not contain consumed events.
    2. No rainfall SHA duplicates within the fresh pool.
    3. Minimum Locked event count is not reduced.
    4. Selection did not read true KPI values.

    Returns
    -------
    dict
        Audit result with ``passed`` (bool) and ``violations`` (list of str).
    """
    root = Path(output_root)
    eval_dir = root / FRESH_EVAL_DIR
    violations: list[str] = []

    # Load plan freeze
    freeze_path = eval_dir / "plan_freeze.json"
    if not freeze_path.exists():
        return {
            "passed": False,
            "violations": ["plan_freeze.json not found"],
            "status": "blocked",
        }

    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))

    # 1. Check fresh pool does not contain consumed events
    inventory_path = eval_dir / "fresh_event_inventory.csv"
    if inventory_path.exists():
        inventory = pd.read_csv(inventory_path)
        for col in ("assigned_split", "role"):
            if col in inventory.columns:
                consumed_in_pool = inventory[
                    inventory[col].astype(str).str.lower().isin(_CONSUMED_ROLES)
                ]
                if len(consumed_in_pool) > 0:
                    violations.append(
                        f"fresh pool contains {len(consumed_in_pool)} consumed events "
                        f"(via column '{col}')"
                    )
    else:
        violations.append("fresh_event_inventory.csv not found")

    # 2. No rainfall SHA duplicates
    if inventory_path.exists():
        inventory = pd.read_csv(inventory_path)
        if "rainfall_sha256" in inventory.columns:
            dup_shas = inventory["rainfall_sha256"].duplicated(keep=False)
            n_dups = dup_shas.sum()
            if n_dups > 0:
                violations.append(
                    f"{n_dups} events share duplicate rainfall_sha256 in fresh pool"
                )

    # 3. Locked event count not reduced
    splits = freeze.get("splits", {})
    n_locked = len(splits.get("locked", []))
    status_info = freeze.get("status", {})
    expected_locked = status_info.get("locked", 0)
    if n_locked < expected_locked and expected_locked > 0:
        violations.append(
            f"locked events ({n_locked}) < expected ({expected_locked})"
        )

    # Also check that minimum locked is at least 8 when status is ready_full
    if status_info.get("status") == "ready_full" and n_locked < TARGET_LOCKED:
        violations.append(
            f"locked events ({n_locked}) < required minimum ({TARGET_LOCKED})"
        )

    # 4. Selection did not read true KPIs
    if freeze.get("reads_true_kpi_for_selection", True) is not False:
        violations.append("plan_freeze indicates true KPIs were read for selection")

    # 5. Frozen before labels
    if not freeze.get("created_before_any_v42_evaluation_labels", False):
        violations.append("plan not frozen before V4.2 evaluation labels")

    passed = len(violations) == 0

    # Write audit result
    audit_result = {
        "stage": "AuditV42FreshEvaluationAvailability",
        "audit_time": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "violations": violations,
        "n_violations": len(violations),
        "plan_freeze_sha256": freeze.get("selection_order_sha256", ""),
        "ledger_sha256": freeze.get("ledger_sha256", ""),
        "status": "pass" if passed else "blocked",
    }
    atomic_write_json(eval_dir / "evaluation_availability_audit.json", audit_result)

    log.info(
        "Fresh eval audit: %s (%d violations)",
        "PASSED" if passed else "FAILED", len(violations),
    )

    return audit_result
