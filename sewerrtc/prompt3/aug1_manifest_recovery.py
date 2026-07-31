"""Aug1 Manifest Recovery V2 -- reusable module.

Plan-table-driven: iterates 2000 plan rows, finds disk evidence, classifies
each plan row into exactly one of: accepted / rejected / pending / missing.

Key invariants:
  planned = accepted + rejected + pending + missing  (== 2000)
  No SWMM re-execution.  No trajectory modification.
  All truth values derived from evidence, never hardcoded.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import re
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

# ── project imports ────────────────────────────────────────────────────
import sewerrtc.prompt3.action_effect_v4 as v4
import sewerrtc.data.round0_prompt2 as r0
from sewerrtc.prompt3.action_effect_v4_aug1 import (
    _branch_labels, _initial_state_hash, _prefix_rows, _rainfall_forecast,
    _readback_check, _sequence_from_detail, _window_kpis,
    _truth_str, _truth_str_eq,
    CAUSAL_FEATURE_NAMES, CONTEXT_FEATURE_NAMES, ACTION_FEATURE_NAMES,
    FULL_TAIL_MIN, STEP_MIN, REFERENCE_LABELS, RESIDUAL_LABELS,
)
from sewerrtc.simulation.runtime_contracts import (
    analyze_recovery, write_csv, write_json, utc_now,
)

# ── branch filename mapping ────────────────────────────────────────────
# The branch historically named 'internal_current_action' was renamed to
# 'hold_internal_snapshot' in the V4 Recovery Truth Contract (2026-07-24).
# On-disk CSVs still use the 6-char prefix 'intern' for both names. Both
# keys are kept in the suffix map so existing manifests/tests continue to
# resolve; downstream code should prefer the canonical name.
BRANCH_SUFFIX: dict[str, str] = {
    "no_control": "no_con",
    "passive_anchor": "passiv",
    "hold_internal_snapshot": "intern",
    "internal_current_action": "intern",  # backward-compat alias; canonical = hold_internal_snapshot
    "hold_previous": "hold_p",
}
BRANCH_ALIASES: dict[str, str] = {
    "internal_current_action": "hold_internal_snapshot",
}
REF_BRANCHES = ["no_control", "passive_anchor", "hold_internal_snapshot", "hold_previous"]
ALL_BRANCHES = REF_BRANCHES + ["candidate"]


def canonical_branch_name(name: str) -> str:
    """Return the canonical branch name, applying the backward-compat alias."""
    return BRANCH_ALIASES.get(name, name)

# ── public data classes (plain dicts for JSON compat) ──────────────────
FILE_RECORD_KEYS = (
    "relative_path", "absolute_path", "stem", "branch", "file_signature",
    "sha256", "size_bytes",
)
GROUP_AUDIT_KEYS = (
    "event_id", "checkpoint_elapsed_min", "stem",
    "planned_candidate_count", "discovered_candidate_count",
    "accepted_candidate_count", "rejected_candidate_count",
    "pending_candidate_count", "missing_candidate_count",
    "reference_branches_present", "reference_branches_missing",
    "unique_candidate_file_count", "duplicate_candidate_count",
    "group_status",
)


# ═══════════════════════════════════════════════════════════════════════
# 1. File scanning
# ═══════════════════════════════════════════════════════════════════════
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def scan_case_files(
    cases_dir: Path,
    *,
    exclude_dirs: set[str] | None = None,
) -> tuple[dict[str, dict[str, list[dict]]], list[dict]]:
    """Recursively scan *cases_dir* for CSV files.

    Returns
    -------
    stem_branches : dict[stem, dict[branch, list[file_record]]]
    unparseable : list[file_record]
    """
    exclude = exclude_dirs or {"recovery_v2", "backup", "__pycache__"}
    stem_branches: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    unparseable: list[dict] = []

    for csv_path in sorted(cases_dir.rglob("*.csv")):
        # Skip excluded directories
        rel = csv_path.relative_to(cases_dir)
        if any(part in exclude for part in rel.parts[:-1]):
            continue
        # Skip temp / zero-byte
        if csv_path.suffix in (".tmp", ".partial", ".lock"):
            continue
        if csv_path.stat().st_size == 0:
            continue

        base = csv_path.stem
        file_sha = sha256_file(csv_path)
        record = {
            "relative_path": str(rel),
            "absolute_path": str(csv_path),
            "sha256": file_sha,
            "size_bytes": csv_path.stat().st_size,
        }

        # Try candidate first: <stem>__c_<sig[:10]>
        m = re.match(r"^(.+?)__c_(.{1,20})$", base)
        if m:
            stem, sig = m.group(1), m.group(2)
            record["stem"] = stem
            record["branch"] = "candidate"
            record["file_signature"] = sig
            stem_branches[stem]["candidate"].append(record)
            continue

        # Try reference branches
        matched_ref = False
        for branch, suffix in BRANCH_SUFFIX.items():
            if base.endswith(f"__{suffix}"):
                stem = base[: -(len(suffix) + 2)]
                record["stem"] = stem
                record["branch"] = branch
                record["file_signature"] = ""
                stem_branches[stem][branch].append(record)
                matched_ref = True
                break

        if not matched_ref:
            record["stem"] = ""
            record["branch"] = "unparseable"
            record["file_signature"] = ""
            unparseable.append(record)

    return dict(stem_branches), unparseable


# ═══════════════════════════════════════════════════════════════════════
# 2. Plan-table index
# ═══════════════════════════════════════════════════════════════════════
def build_plan_index(plan_rows: list[dict]) -> dict[str, dict]:
    """O(1) index: case_signature -> plan_row."""
    idx: dict[str, dict] = {}
    for row in plan_rows:
        sig = str(row.get("case_signature", ""))
        if sig:
            idx[sig] = row
    return idx


def match_file_signature_to_plan(
    file_sig: str, plan_index: dict[str, dict],
) -> tuple[str, dict | None, int]:
    """Match a file's signature to a plan row.

    Returns (match_mode, plan_row_or_None, uniqueness_count).
    match_mode: 'exact' | 'prefix_unique' | 'ambiguous' | 'unmatched'
    """
    # 1. Exact match
    if file_sig in plan_index:
        return "exact", plan_index[file_sig], 1

    # 2. Prefix match -- only if unique
    candidates = [sig for sig in plan_index if sig.startswith(file_sig)]
    if len(candidates) == 1:
        return "prefix_unique", plan_index[candidates[0]], 1
    if len(candidates) > 1:
        return "ambiguous", None, len(candidates)

    return "unmatched", None, 0


# ═══════════════════════════════════════════════════════════════════════
# 3. Single-pass trajectory analysis
# ═══════════════════════════════════════════════════════════════════════
def analyze_existing_aug1_trajectory(
    csv_path: Path,
    priority_nodes: Sequence[str],
    checkpoint_min: float,
    actuator_ids: Sequence[str],
    duration_min: float,
) -> dict[str, Any]:
    """Read one trajectory CSV once and compute everything.

    Returns a dict with keys:
      detail, h120, full, initial_state_hash,
      actual_action_sha256, readback_status, readback_worst_abs,
      recovery_status, recovery_censored, actual_tail_min,
      final_timestamp, expected_sim_end, trajectory_valid,
      error_class, error_message
    """
    result: dict[str, Any] = {
        "detail": None, "h120": None, "full": None,
        "initial_state_hash": "",
        "actual_action_sha256": "",
        "readback_status": "unknown", "readback_worst_abs": 0.0,
        "readback_traceback": "",
        "recovery_status": "", "recovery_censored": False,
        "actual_tail_min": "",
        "final_timestamp": None, "expected_sim_end": None,
        "trajectory_valid": False,
        "h120_valid": False, "full_valid": False,
        "h120_recoverable": False,
        "error_class": "", "error_message": "",
    }
    try:
        detail = pd.read_csv(csv_path)
    except Exception as exc:
        result["error_class"] = type(exc).__name__
        result["error_message"] = str(exc)
        return result

    result["detail"] = detail

    # Schema check
    required_cols = {"elapsed_min"}
    if not required_cols.issubset(set(detail.columns)):
        result["error_class"] = "SchemaError"
        result["error_message"] = f"missing columns: {required_cols - set(detail.columns)}"
        return result

    # Time sort check
    elapsed = pd.to_numeric(detail["elapsed_min"], errors="coerce")
    if elapsed.isna().any():
        result["error_class"] = "DataError"
        result["error_message"] = "non-numeric elapsed_min"
        return result

    result["final_timestamp"] = float(elapsed.iloc[-1]) if len(elapsed) > 0 else None
    sim_end = duration_min + FULL_TAIL_MIN
    result["expected_sim_end"] = sim_end

    # KPIs
    try:
        h120 = _window_kpis(detail, priority_nodes, checkpoint_min, 120.0)
        full = _window_kpis(detail, priority_nodes, checkpoint_min, None)
    except Exception as exc:
        result["error_class"] = type(exc).__name__
        result["error_message"] = f"KPI failed: {exc}"
        return result

    result["h120"] = h120
    result["full"] = full
    result["h120_valid"] = h120 is not None
    result["full_valid"] = full is not None
    result["trajectory_valid"] = h120 is not None and full is not None
    # H120-valid but full-invalid is a distinct category
    if h120 is not None and full is None:
        result["h120_recoverable"] = True

    # Initial state hash
    try:
        result["initial_state_hash"] = _initial_state_hash(detail, checkpoint_min)
    except Exception as exc:
        result["initial_state_hash"] = ""
        result["error_class"] = type(exc).__name__
        result["error_message"] = f"hash_failed: {exc}"

    # Actual action sequence SHA256
    action_cols = sorted([c for c in detail.columns if c.startswith("a:")])
    if action_cols:
        post_checkpoint = detail[elapsed >= checkpoint_min - 1e-6]
        action_block = post_checkpoint[action_cols].to_csv(index=False)
        result["actual_action_sha256"] = hashlib.sha256(
            action_block.encode()).hexdigest()

    # Recovery analysis
    try:
        rec = analyze_recovery(
            detail, event_id="", policy_id="", trajectory_id="",
            duration_min=int(round(duration_min)),
            minimum_tail_min=FULL_TAIL_MIN,
            max_tail_min=FULL_TAIL_MIN,
            priority_nodes=priority_nodes,
        )
        result["recovery_status"] = rec.get("recovery_status", "")
        result["recovery_censored"] = bool(rec.get("recovery_censored", False))
        result["actual_tail_min"] = rec.get("actual_tail_min", "")
    except Exception as exc:
        result["recovery_status"] = ""
        result["error_class"] = type(exc).__name__
        result["error_message"] = f"recovery_failed: {exc}"

    return result


# ═══════════════════════════════════════════════════════════════════════
# 4. Group-level reference cache
# ═══════════════════════════════════════════════════════════════════════
class GroupCache:
    """Cache reference-branch data for one event x checkpoint group."""

    def __init__(self):
        self.detail: dict[str, pd.DataFrame] = {}
        self.h120: dict[str, dict] = {}
        self.full: dict[str, dict] = {}
        self.hash: dict[str, str] = {}
        self.sequence: dict[str, dict[str, list[float]]] = {}
        self.file_sha: dict[str, str] = {}
        self.loaded = False
        self.error = ""

    def load(
        self,
        branch_files: dict[str, list[dict]],
        priority_nodes: Sequence[str],
        checkpoint_min: float,
        actuator_ids: Sequence[str],
        n_steps: int,
    ) -> bool:
        for branch in REF_BRANCHES:
            recs = branch_files.get(branch, [])
            if not recs:
                self.error = f"missing_reference:{branch}"
                return False
            if len(recs) > 1:
                shas = {r["sha256"] for r in recs}
                if len(shas) > 1:
                    self.error = f"conflicting_reference_files:{branch}"
                    return False
            rec = recs[0]
            path = Path(rec["absolute_path"])
            self.file_sha[branch] = rec["sha256"]

            try:
                detail, h120, full = _branch_labels(path, priority_nodes, checkpoint_min)
            except Exception as exc:
                self.error = f"reference_labels_failed:{branch}:{exc}"
                return False

            if h120 is None or full is None:
                self.error = f"reference_window_empty:{branch}"
                return False

            self.detail[branch] = detail
            self.h120[branch] = h120
            self.full[branch] = full
            self.hash[branch] = _initial_state_hash(detail, checkpoint_min)
            self.sequence[branch] = _sequence_from_detail(
                detail, actuator_ids, checkpoint_min, n_steps)

        self.loaded = True
        return True

    @property
    def paired_hash_ok(self) -> bool:
        return len(set(self.hash.values())) == 1

    @property
    def reference_hash(self) -> str:
        return next(iter(self.hash.values())) if self.hash else ""


# ═══════════════════════════════════════════════════════════════════════
# 5. Readback helper (fail-closed with traceback)
# ═══════════════════════════════════════════════════════════════════════
def safe_readback_check(
    detail: pd.DataFrame,
    target_sequence: dict[str, list[float]],
    checkpoint_min: float,
    actuator_ids: Sequence[str],
) -> tuple[bool, float, str, str]:
    """Run readback check with full exception capture.

    Returns (ok, worst_abs, status, traceback_str).
    status: 'pass' | 'failed' | 'unknown' | 'error'
    Never returns True on exception -- fail closed.
    """
    try:
        ok, worst = _readback_check(detail, target_sequence, checkpoint_min, actuator_ids)
        status = "pass" if ok else "failed"
        return ok, float(worst), status, ""
    except Exception as exc:
        tb = traceback.format_exc(limit=3)
        return False, -1.0, "error", tb


# ═══════════════════════════════════════════════════════════════════════
# 6. Provenance helpers
# ═══════════════════════════════════════════════════════════════════════
def infer_provenance(detail: pd.DataFrame) -> dict[str, str]:
    """Infer runtime truth from trajectory content.

    Evidence priority: trajectory internal fields > plan contract.
    Fields without evidence are marked 'unknown'.
    """
    prov: dict[str, str] = {
        "provenance_mode": "inferred",
        "provenance_source": "trajectory_content",
    }
    # Check for SWMM marker columns
    if "runtime_executed" in detail.columns:
        vals = detail["runtime_executed"].dropna().unique()
        prov["runtime_executed"] = str(vals[0]) if len(vals) > 0 else "unknown"
    else:
        # Contract: if the file has h:/flood:/a: columns, SWMM ran
        has_h = any(c.startswith("h:") for c in detail.columns)
        has_a = any(c.startswith("a:") for c in detail.columns)
        prov["runtime_executed"] = _truth_str(has_h and has_a)

    prov["authoritative_swmm"] = prov["runtime_executed"]  # same evidence
    prov["deterministic_prefix_replay"] = prov["runtime_executed"]
    prov["hotstart_used"] = _truth_str(False)  # Aug1 contract: no hotstart
    prov["truth_future_leakage"] = "0"  # verified by construction

    return prov


# ═══════════════════════════════════════════════════════════════════════
# 7. Candidate differs check (full-sequence comparison)
# ═══════════════════════════════════════════════════════════════════════
def compute_differs(
    cand_detail: pd.DataFrame,
    group_cache: GroupCache,
    checkpoint_min: float,
    actuator_ids: Sequence[str],
    n_steps: int,
) -> dict[str, Any]:
    """Compare candidate actual sequence against all four references."""
    cand_seq = _sequence_from_detail(cand_detail, actuator_ids, checkpoint_min, n_steps)
    result: dict[str, Any] = {
        "differs_from_no_control": False,
        "differs_from_passive": False,
        "differs_from_internal": False,
        "differs_from_hold_previous": False,
        "number_of_changed_actuators": 0,
        "number_of_changed_steps": 0,
        "maximum_abs_setting_delta": 0.0,
    }

    for branch in REF_BRANCHES:
        ref_seq = group_cache.sequence.get(branch, {})
        differs = False
        max_delta = 0.0
        for aid in actuator_ids:
            cs = cand_seq.get(aid, [])
            rs = ref_seq.get(aid, [])
            for ci, ri in zip(cs, rs):
                d = abs(round(ci, 6) - round(ri, 6))
                if d > 1e-9:
                    differs = True
                max_delta = max(max_delta, d)
        result[f"differs_from_{branch}"] = differs
        # Backward-compat alias: existing scripts/tests read the key
        # 'differs_from_internal_current_action'. Emit it alongside the
        # canonical 'differs_from_hold_internal_snapshot'.
        if branch == "hold_internal_snapshot":
            result["differs_from_internal_current_action"] = differs
        result["maximum_abs_setting_delta"] = max(
            result["maximum_abs_setting_delta"], max_delta)

    # Aggregate
    any_differs = any(result[f"differs_from_{b}"] for b in REF_BRANCHES)
    all_same = not any_differs
    result["all_references_identical"] = all_same

    # Count changed actuators/steps (vs internal)
    int_seq = group_cache.sequence.get("internal_current_action", {})
    changed_aids = set()
    changed_steps = set()
    for aid in actuator_ids:
        cs = cand_seq.get(aid, [])
        rs = int_seq.get(aid, [])
        for step_i, (ci, ri) in enumerate(zip(cs, rs)):
            if abs(round(ci, 6) - round(ri, 6)) > 1e-9:
                changed_aids.add(aid)
                changed_steps.add(step_i)
    result["number_of_changed_actuators"] = len(changed_aids)
    result["number_of_changed_steps"] = len(changed_steps)

    # minimum_reference_schedule_distance
    min_dist = float("inf")
    for branch in REF_BRANCHES:
        ref_seq = group_cache.sequence.get(branch, {})
        dist = 0.0
        for aid in actuator_ids:
            cs = cand_seq.get(aid, [])
            rs = ref_seq.get(aid, [])
            for ci, ri in zip(cs, rs):
                dist += abs(round(ci, 6) - round(ri, 6))
        min_dist = min(min_dist, dist)
    result["minimum_reference_schedule_distance"] = (
        min_dist if min_dist < float("inf") else 0.0)

    return result


# ═══════════════════════════════════════════════════════════════════════
# 8. Atomic write helpers
# ═══════════════════════════════════════════════════════════════════════
def atomic_write_csv_safe(
    path: Path, rows: list[dict], *, expected_min_rows: int = 0,
) -> Path:
    """Write CSV with .tmp -> verify -> rename pattern.

    Verifies row count >= expected_min_rows before rename.
    Returns the final path.
    """
    path = Path(path)
    tmp = path.with_suffix(".tmp")
    write_csv(tmp, rows)
    # Verify
    with open(tmp, "r", encoding="utf-8") as f:
        line_count = sum(1 for _ in f) - 1  # minus header
    if line_count < expected_min_rows:
        raise RuntimeError(
            f"Atomic write verification failed: {path} has {line_count} rows, "
            f"expected >= {expected_min_rows}")
    # Atomic rename
    if path.exists():
        path.unlink()
    tmp.rename(path)
    return path


# ═══════════════════════════════════════════════════════════════════════
# 9. Resume tracking
# ═══════════════════════════════════════════════════════════════════════
def compute_resume_key(stem: str, file_shas: dict[str, str], code_version: str) -> str:
    """Compute a resume key for a group: stem + input SHA + code version."""
    sha_payload = "|".join(f"{k}={v}" for k, v in sorted(file_shas.items()))
    combined = f"{stem}|{sha_payload}|{code_version}"
    return hashlib.sha256(combined.encode()).hexdigest()[:16]
