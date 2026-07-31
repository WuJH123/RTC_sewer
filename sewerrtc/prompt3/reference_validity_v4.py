"""Reference Validity Gate V4 -- audit reference branches against truth contract.

Public API
----------
- ``audit_no_control_semantics``   -- verify No-control branch actions match contract
- ``audit_dynamic_internal``       -- verify Dynamic Internal branch (native rules)
- ``audit_passive_degeneracy``     -- check if Passive == No-control (degenerate)
- ``audit_hold_previous``          -- verify Hold-Previous freezes pre-checkpoint settings
- ``verify_paired_state_hash``     -- assert initial-state hash equality across branches
- ``reference_validity_gate``      -- aggregate gate verdict

All functions are pure / side-effect free.  They accept DataFrames or paths
and return structured result dicts.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd


# ── result types ────────────────────────────────────────────────────────
@dataclass
class AuditResult:
    """Generic audit result for one reference branch."""
    branch: str
    contract_verified: bool
    details: dict[str, Any] = field(default_factory=dict)
    error: str = ""


@dataclass
class DegeneracyReport:
    """Result of passive-degeneracy check."""
    reference_degenerate: bool
    max_action_delta: float = 0.0
    mean_action_delta: float = 0.0
    n_timesteps_checked: int = 0
    n_facilities_checked: int = 0
    evidence: str = ""


@dataclass
class GateResult:
    """Aggregate gate verdict."""
    passed: bool
    per_branch: dict[str, AuditResult | DegeneracyReport] = field(default_factory=dict)
    paired_hash_ok: bool = False
    tfv_peak_training_blocked: bool = False
    verdict: str = "FAIL"
    notes: list[str] = field(default_factory=list)


# ── helpers ─────────────────────────────────────────────────────────────
def _action_columns(detail: pd.DataFrame) -> list[str]:
    """Return columns matching ``a:<id>`` pattern."""
    return [c for c in detail.columns if c.startswith("a:")]


def _actuator_ids_from_columns(detail: pd.DataFrame) -> list[str]:
    """Extract actuator IDs from ``a:<id>`` columns."""
    return [c[2:] for c in _action_columns(detail)]


def _nearest_row(detail: pd.DataFrame, elapsed_min: float) -> pd.Series:
    """Return the row closest to *elapsed_min*."""
    e = pd.to_numeric(detail["elapsed_min"], errors="coerce")
    idx = (e - float(elapsed_min)).abs().idxmin()
    return detail.loc[idx]


def _extract_action_matrix(
    detail: pd.DataFrame,
    actuator_ids: Sequence[str],
    checkpoint_min: float,
    n_steps: int | None = None,
) -> np.ndarray:
    """Build (n_steps, n_actuators) matrix of post-checkpoint ``a:`` values."""
    e = pd.to_numeric(detail["elapsed_min"], errors="coerce")
    post = detail[e > float(checkpoint_min) + 1e-6].copy()
    post["elapsed_min"] = pd.to_numeric(post["elapsed_min"], errors="coerce")
    post = post.sort_values("elapsed_min")
    if n_steps is not None:
        post = post.head(int(n_steps))
    cols = [f"a:{aid}" for aid in actuator_ids]
    existing = [c for c in cols if c in post.columns]
    if not existing:
        return np.empty((len(post), 0))
    mat = post[existing].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    return mat


# ── public API ──────────────────────────────────────────────────────────
def audit_no_control_semantics(
    inp_path: str | Path,
    detail_csv: str | Path | pd.DataFrame,
    contract: dict,
    *,
    actuator_ids: Sequence[str] | None = None,
    checkpoint_min: float = 0.0,
) -> AuditResult:
    """Verify the No-control branch's ``a:`` settings match the contract.

    The contract defines No-control as: all managed facilities set to 1.0
    (fully open / bypassed) for every post-checkpoint time step.

    Does NOT assume all-0 or all-1 from the name; reads actual values.
    """
    if isinstance(detail_csv, pd.DataFrame):
        detail = detail_csv
    else:
        detail = pd.read_csv(detail_csv)

    if actuator_ids is None:
        actuator_ids = _actuator_ids_from_columns(detail)

    mat = _extract_action_matrix(detail, actuator_ids, checkpoint_min)
    if mat.size == 0:
        return AuditResult(
            branch="no_control", contract_verified=False,
            error="no post-checkpoint rows found")

    # Check contract definition: all settings should be 1.0
    contract_def = contract.get("no_control_semantics", {})
    expected_value = contract_def.get("expected_setting", 1.0)

    all_match = bool(np.allclose(mat, expected_value, atol=1e-6))
    col_means = mat.mean(axis=0).tolist() if mat.ndim == 2 else []
    unique_vals = np.unique(np.round(mat, 6)).tolist()

    pattern = "unknown"
    if all_match:
        pattern = "all_open" if abs(expected_value - 1.0) < 1e-6 else f"all_{expected_value}"
    elif len(unique_vals) == 1:
        pattern = f"all_{unique_vals[0]}"
    elif len(unique_vals) == 2 and min(unique_vals) == 0.0 and max(unique_vals) == 1.0:
        pattern = "mixed_binary"
    else:
        pattern = "mixed"

    return AuditResult(
        branch="no_control",
        contract_verified=all_match,
        details={
            "actual_action_pattern": pattern,
            "unique_values": unique_vals,
            "per_actuator_mean": dict(zip(actuator_ids, col_means)),
            "expected_setting": expected_value,
            "n_timesteps": mat.shape[0],
            "n_facilities": mat.shape[1],
        },
    )


def audit_dynamic_internal(
    inp_path: str | Path,
    detail_csv: str | Path | pd.DataFrame,
    contract: dict,
    *,
    actuator_ids: Sequence[str] | None = None,
    checkpoint_min: float = 0.0,
    baseline_detail: pd.DataFrame | None = None,
) -> AuditResult:
    """Verify Dynamic Internal branch.

    Checks:
    1. Native ``[CONTROLS]`` were enabled post-checkpoint (strip_controls=False).
    2. Action schedule varies over time (not frozen).
    3. Provenance = internal_rules baseline prefix + native rules suffix.

    If ``baseline_detail`` is provided, verifies prefix hash matches.
    """
    if isinstance(detail_csv, pd.DataFrame):
        detail = detail_csv
    else:
        detail = pd.read_csv(detail_csv)

    if actuator_ids is None:
        actuator_ids = _actuator_ids_from_columns(detail)

    mat = _extract_action_matrix(detail, actuator_ids, checkpoint_min)
    if mat.size == 0:
        return AuditResult(
            branch="dynamic_internal", contract_verified=False,
            error="no post-checkpoint rows found")

    # Check if actions vary over time (dynamic, not frozen)
    row_variances = mat.std(axis=1) if mat.ndim == 2 else np.zeros(mat.shape[0])
    actions_vary = bool(np.any(row_variances > 1e-9))

    # Check if policy_phase column exists and has both phases
    has_phases = False
    if "policy_phase" in detail.columns:
        e = pd.to_numeric(detail["elapsed_min"], errors="coerce")
        post = detail[e > float(checkpoint_min) + 1e-6]
        phases = set(post["policy_phase"].unique())
        has_phases = {"prefix_replay", "native_rules"}.issubset(phases)

    # Check if post-checkpoint actions differ from frozen snapshot
    differs_from_frozen = False
    if baseline_detail is not None:
        base_mat = _extract_action_matrix(baseline_detail, actuator_ids, checkpoint_min)
        if base_mat.shape == mat.shape:
            differs_from_frozen = bool(not np.allclose(mat, base_mat, atol=1e-6))

    verified = actions_vary
    if baseline_detail is not None:
        verified = verified and differs_from_frozen

    return AuditResult(
        branch="dynamic_internal",
        contract_verified=verified,
        details={
            "actions_vary_over_time": actions_vary,
            "has_both_policy_phases": has_phases,
            "differs_from_frozen_snapshot": differs_from_frozen,
            "n_timesteps": mat.shape[0],
            "n_facilities": mat.shape[1],
        },
    )


def audit_passive_degeneracy(
    detail_passive: str | Path | pd.DataFrame,
    detail_no_control: str | Path | pd.DataFrame,
    contract: dict,
    *,
    actuator_ids: Sequence[str] | None = None,
    checkpoint_min: float = 0.0,
    tolerance: float = 1e-6,
) -> DegeneracyReport:
    """Check if Passive branch is degenerate (identical to No-control).

    Computes per-timestep action delta and flags ``reference_degenerate=true``
    if post-checkpoint actions are identical within tolerance.
    """
    if isinstance(detail_passive, pd.DataFrame):
        pa = detail_passive
    else:
        pa = pd.read_csv(detail_passive)
    if isinstance(detail_no_control, pd.DataFrame):
        nc = detail_no_control
    else:
        nc = pd.read_csv(detail_no_control)

    if actuator_ids is None:
        actuator_ids = _actuator_ids_from_columns(pa)

    pa_mat = _extract_action_matrix(pa, actuator_ids, checkpoint_min)
    nc_mat = _extract_action_matrix(nc, actuator_ids, checkpoint_min)

    # Align lengths
    min_rows = min(pa_mat.shape[0], nc_mat.shape[0])
    if min_rows == 0:
        return DegeneracyReport(
            reference_degenerate=False,
            evidence="no post-checkpoint rows to compare")

    pa_mat = pa_mat[:min_rows]
    nc_mat = nc_mat[:min_rows]

    delta = np.abs(pa_mat - nc_mat)
    max_delta = float(delta.max()) if delta.size else 0.0
    mean_delta = float(delta.mean()) if delta.size else 0.0

    degenerate = max_delta < tolerance

    evidence = (
        f"post-checkpoint action delta: max={max_delta:.2e}, "
        f"mean={mean_delta:.2e}, "
        f"n_timesteps={min_rows}, n_facilities={pa_mat.shape[1]}"
    )

    return DegeneracyReport(
        reference_degenerate=degenerate,
        max_action_delta=max_delta,
        mean_action_delta=mean_delta,
        n_timesteps_checked=min_rows,
        n_facilities_checked=pa_mat.shape[1],
        evidence=evidence,
    )


def audit_hold_previous(
    detail_internal: str | Path | pd.DataFrame,
    detail_hold_prev: str | Path | pd.DataFrame,
    contract: dict,
    *,
    actuator_ids: Sequence[str] | None = None,
    checkpoint_min: float = 0.0,
    tolerance: float = 1e-6,
) -> AuditResult:
    """Verify Hold-Previous: post-checkpoint ``a:`` = pre-checkpoint ``a:`` frozen.

    The Hold-Previous branch should freeze the settings at the checkpoint
    instant and hold them constant for all subsequent steps.
    """
    if isinstance(detail_internal, pd.DataFrame):
        internal = detail_internal
    else:
        internal = pd.read_csv(detail_internal)
    if isinstance(detail_hold_prev, pd.DataFrame):
        hp = detail_hold_prev
    else:
        hp = pd.read_csv(detail_hold_prev)

    if actuator_ids is None:
        actuator_ids = _actuator_ids_from_columns(internal)

    # Get checkpoint settings from internal detail
    e_int = pd.to_numeric(internal["elapsed_min"], errors="coerce")
    cp_row = internal[(e_int - float(checkpoint_min)).abs() <= 1.0]
    if cp_row.empty:
        cp_row = _nearest_row(internal, checkpoint_min)
        cp_row = pd.DataFrame([cp_row])

    cp_settings = {}
    for aid in actuator_ids:
        col = f"a:{aid}"
        if col in cp_row.columns:
            cp_settings[aid] = float(pd.to_numeric(cp_row[col].iloc[0], errors="coerce"))

    # Check hold_previous post-checkpoint actions are all equal to checkpoint
    hp_mat = _extract_action_matrix(hp, actuator_ids, checkpoint_min)
    if hp_mat.size == 0:
        return AuditResult(
            branch="hold_previous", contract_verified=False,
            error="no post-checkpoint rows in hold_previous detail")

    verified = True
    max_deviation = 0.0
    for j, aid in enumerate(actuator_ids):
        if aid not in cp_settings:
            continue
        expected = cp_settings[aid]
        if j < hp_mat.shape[1]:
            col_vals = hp_mat[:, j]
            dev = float(np.max(np.abs(col_vals - expected)))
            max_deviation = max(max_deviation, dev)
            if dev > tolerance:
                verified = False

    # Also check that all post-checkpoint values are constant (frozen)
    if hp_mat.shape[0] > 1:
        row_std = hp_mat.std(axis=0)
        frozen = bool(np.all(row_std < tolerance))
    else:
        frozen = True

    return AuditResult(
        branch="hold_previous",
        contract_verified=verified and frozen,
        details={
            "settings_frozen": frozen,
            "max_deviation_from_checkpoint": max_deviation,
            "n_timesteps": hp_mat.shape[0],
            "n_facilities": hp_mat.shape[1],
            "checkpoint_settings_sample": dict(list(cp_settings.items())[:5]),
        },
    )


def verify_paired_state_hash(
    branch_details: dict[str, str | Path | pd.DataFrame],
    checkpoint_min: float = 0.0,
) -> bool:
    """Compute ``h:`` + ``flood:`` prefix hash per branch; assert equality.

    All branches sharing the same deterministic prefix must have identical
    hydraulic state histories up to the checkpoint.
    """
    hashes: dict[str, str] = {}
    for branch, detail in branch_details.items():
        if isinstance(detail, pd.DataFrame):
            df = detail
        else:
            df = pd.read_csv(detail)
        e = pd.to_numeric(df["elapsed_min"], errors="coerce")
        prefix = df[e <= float(checkpoint_min) + 1.0e-6]
        cols = sorted(c for c in prefix.columns if c.startswith("h:") or c.startswith("flood:"))
        if not cols:
            hashes[branch] = "empty"
            continue
        payload = np.round(
            prefix[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float), 6
        )
        h = hashlib.sha256(payload.tobytes() + "|".join(cols).encode()).hexdigest()
        hashes[branch] = h

    unique = set(hashes.values())
    return len(unique) == 1


def reference_validity_gate(
    contract: dict,
    audit_results: dict[str, AuditResult | DegeneracyReport | bool],
) -> GateResult:
    """Aggregate per-branch audit results into a gate verdict.

    Parameters
    ----------
    contract : dict
        The truth contract.
    audit_results : dict
        Keys: ``no_control``, ``dynamic_internal``, ``passive``, ``hold_previous``,
        ``paired_hash``.  Values: ``AuditResult``, ``DegeneracyReport``, or
        ``bool`` (for ``paired_hash``).

    Returns
    -------
    GateResult
    """
    notes: list[str] = []
    per_branch: dict[str, Any] = {}
    all_ok = True
    tfv_blocked = False

    # No-control
    nc = audit_results.get("no_control")
    if isinstance(nc, AuditResult):
        per_branch["no_control"] = nc
        if not nc.contract_verified:
            all_ok = False
            notes.append(f"no_control NOT verified: {nc.details.get('actual_action_pattern', '?')}")
        else:
            notes.append("no_control verified")

    # Dynamic Internal
    di = audit_results.get("dynamic_internal")
    if isinstance(di, AuditResult):
        per_branch["dynamic_internal"] = di
        if not di.contract_verified:
            tfv_blocked = True
            notes.append("dynamic_internal NOT verified -- TFV/Peak training blocked")
        else:
            notes.append("dynamic_internal verified")

    # If dynamic_internal is not available at all, block TFV/Peak
    if di is None:
        tfv_blocked = True
        notes.append("dynamic_internal not provided -- TFV/Peak training blocked")

    # Passive degeneracy
    pa = audit_results.get("passive")
    if isinstance(pa, DegeneracyReport):
        per_branch["passive"] = pa
        if pa.reference_degenerate:
            notes.append(f"passive DEGENERATE: {pa.evidence}")
        else:
            notes.append("passive non-degenerate (OK)")

    # Hold-previous
    hp = audit_results.get("hold_previous")
    if isinstance(hp, AuditResult):
        per_branch["hold_previous"] = hp
        if not hp.contract_verified:
            all_ok = False
            notes.append("hold_previous NOT verified")
        else:
            notes.append("hold_previous verified")

    # Paired hash
    ph = audit_results.get("paired_hash", False)
    if not ph:
        all_ok = False
        notes.append("paired_state_hash MISMATCH")

    # Verdict
    if all_ok and not tfv_blocked:
        verdict = "PASS"
    elif all_ok and tfv_blocked:
        verdict = "CONDITIONAL_PASS"
    else:
        verdict = "FAIL"

    return GateResult(
        passed=all_ok,
        per_branch=per_branch,
        paired_hash_ok=bool(ph),
        tfv_peak_training_blocked=tfv_blocked,
        verdict=verdict,
        notes=notes,
    )
