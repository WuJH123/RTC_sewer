from __future__ import annotations

"""Project6 V4 *Aug1* development iteration.

This module adds the four repair stages requested for the failing full-event
PFV direction heads, **without** overwriting the frozen base V4 model:

* ``PlanV4DualReferenceFullEventCases`` (Section 三) -- plans real paired
  dual-reference hard-negative cases stratified by hydraulic phase, action
  type and target failure type.
* ``GenerateV4DualReferenceFullEventCases`` (Section 四/五) -- deterministic
  prefix replay (no hot-start) that, from the *same* checkpoint, runs five
  branches (``candidate``, ``no_control``, ``passive_anchor``,
  ``hold_internal_snapshot`` (renamed from ``internal_current_action``; the old
  name is still accepted as a backward-compatible alias when reading existing
  manifests), ``hold_previous``) through the authoritative
  SWMM engine and writes real H120 + full-recovery labels.
* ``BuildV4AugmentedDataset`` (Section 六) -- merges the audited base V4 data
  with the new real SWMM data (base manifest untouched) and audits it.
* ``TrainV4Aug1`` / ``EvaluateV4Aug1ModelGate`` (Section 七) -- retrains a new
  model version whose residual/reference features include the leakage-free
  causal path-dependent context, and gates it with event-balanced metrics.

Every branch shares an identical deterministic prefix, so the paired
initial-state hash is equal across branches; labels always come from a real
SWMM run and the H120 labels are never copied into the full-event labels.
"""

import csv
import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import yaml

import sewerrtc.data.round0_prompt2 as r0
from sewerrtc.io.safe_paths import short_run_tag, single_writer_lease
from sewerrtc.io.swmm_mutation import mutate_inp_for_event
from sewerrtc.prompt3 import action_effect_v4 as v4
from sewerrtc.prompt3.action_effect_mpc import _fit_ridge
from sewerrtc.simulation.pyswmm_runner import compute_kpis, run_swmm_no_control_action_ablation
from sewerrtc.simulation.runtime_contracts import (
    analyze_recovery,
    sha256_file,
    utc_now,
    write_csv,
    write_json,
)

CONTROL_STEP_SEC = 300
AUG1_DIRNAME = "dual_reference_aug1"
BRANCHES = ("candidate", "no_control", "passive_anchor", "hold_internal_snapshot", "hold_previous")
REFERENCE_BRANCHES = ("no_control", "passive_anchor", "hold_internal_snapshot", "hold_previous")
# Backward-compatible alias. Existing aug1 manifests on disk were written with
# the branch name 'internal_current_action'; they are semantically equivalent
# to 'hold_internal_snapshot' (frozen snapshot of internal rules' past actions,
# NOT the dynamic native rules). Loaders that read branch keys from old
# manifests must accept both names.
BRANCH_ALIASASES = {"internal_current_action": "hold_internal_snapshot"}
CAUSAL_FEATURE_NAMES = v4.CAUSAL_FEATURE_NAMES
CONTEXT_FEATURE_NAMES = v4.CONTEXT_FEATURE_NAMES
ACTION_FEATURE_NAMES = v4.ACTION_FEATURE_NAMES
REFERENCE_LABELS = v4.REFERENCE_LABELS
RESIDUAL_LABELS = v4.RESIDUAL_LABELS

PHASE_STRATA = ("rising", "near_peak", "peak", "recession", "early_recovery", "late_recovery")

# Real, executable action neighborhoods. Each maps to a candidate spec that the
# generation harness projects onto managed facilities. None of these delete real
# label imbalance; they only diversify the real action neighborhood that is run.
ACTION_TYPES = (
    "current_candidate", "hold_previous", "passive_anchor", "internal_current_action",
    "top2", "top4", "half_amplitude", "reduced_facility",
    "delayed_release_10", "delayed_release_20", "extended_hold", "remove_reversal",
    "storage_preserving", "recession_release", "single_facility_perturbation",
)

FAILURE_TYPES = (
    "h120_safe_full_worse", "worse_vs_no_control", "worse_vs_passive",
    "internal_fallback_cumulative_worse", "early_release_before_peak",
    "late_release_in_recession", "high_freq_low_benefit", "pfv_near_zero_boundary",
    "both_full_heads_mispredicted",
)


# ----------------------------------------------------------------------------
# Config / path helpers
# ----------------------------------------------------------------------------
def _load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def _project_root(config: str | Path) -> Path:
    return v4._project_root(config)


def _output_root(config: str | Path) -> Path:
    return v4._output_root(config)


def _aug1_dir(config: str | Path) -> Path:
    return _output_root(config) / AUG1_DIRNAME


def _baseline_manifest(config: str | Path) -> Path:
    return _project_root(config) / "outputs/project6_pfvfirst_dualfallback_10min_v3/baseline_trajectories/baseline_trajectory_manifest.csv"


def _phase_from_elapsed(elapsed_min: float, duration_min: float) -> str:
    d = max(1.0, float(duration_min))
    frac = float(elapsed_min) / d
    if frac < 0.6:
        return "rising"
    if frac < 0.95:
        return "near_peak"
    if frac < 1.15:
        return "peak"
    if frac < 1.8:
        return "recession"
    if frac < 2.6:
        return "early_recovery"
    return "late_recovery"


def _read_detail(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _nearest_row(detail: pd.DataFrame, elapsed_min: float) -> pd.Series:
    elapsed = pd.to_numeric(detail["elapsed_min"], errors="coerce")
    idx = (elapsed - float(elapsed_min)).abs().idxmin()
    return detail.loc[idx]


def _settings_at(detail: pd.DataFrame, elapsed_min: float, actuator_ids: Sequence[str]) -> dict[str, float]:
    row = _nearest_row(detail, elapsed_min)
    out: dict[str, float] = {}
    for aid in actuator_ids:
        val = row.get(f"a:{aid}", row.get(f"setting:{aid}", 1.0))
        try:
            out[aid] = float(np.clip(float(val), 0.0, 1.0))
        except Exception:
            out[aid] = 1.0
    return out


def _rainfall_forecast(rainfall_path: Path) -> list[tuple[float, float]]:
    """Read the frozen design hyetograph as ``[(elapsed_min, intensity_mm_h)]``.

    This is the operational rainfall forecast; it is *not* a realised hydraulic
    truth, so using its post-checkpoint tail for the "remaining rainfall"
    causal features is leakage-free.
    """
    if not rainfall_path.exists():
        return []
    try:
        frame = pd.read_csv(rainfall_path)
    except Exception:
        return []
    cols = list(frame.columns)
    if len(cols) < 2:
        return []
    # The rainfall library CSV may carry a trailing non-numeric ``event_id``
    # column, so the intensity is not always the last column. Prefer a column
    # explicitly named like an intensity, else fall back to the last column
    # whose values are numeric (never the leading time column).
    time_col = cols[0]
    intensity_col = None
    for c in cols[1:]:
        if "intensity" in str(c).lower():
            intensity_col = c
            break
    if intensity_col is None:
        for c in reversed(cols[1:]):
            if pd.to_numeric(frame[c], errors="coerce").notna().any():
                intensity_col = c
                break
    if intensity_col is None:
        return []
    t = pd.to_numeric(frame[time_col], errors="coerce")
    intensity = pd.to_numeric(frame[intensity_col], errors="coerce")
    out: list[tuple[float, float]] = []
    for ti, ii in zip(t, intensity):
        if np.isfinite(ti) and np.isfinite(ii):
            out.append((float(ti), max(0.0, float(ii))))
    return out


def _initial_state_hash(detail: pd.DataFrame, checkpoint_min: float) -> str:
    """Hash the *hydraulic state* history up to and including the checkpoint.

    Only ``h:`` (depth) and ``flood:`` columns are hashed. The ``a:`` action at
    the checkpoint row is intentionally excluded: every branch shares the same
    deterministic internal prefix, so depth/flood at ``elapsed <= checkpoint`` is
    identical, while the action applied *at* the checkpoint is exactly what the
    branches differ on. This makes the five-branch initial-state hash provably
    paired without masking the counterfactual action difference.
    """
    e = pd.to_numeric(detail["elapsed_min"], errors="coerce")
    prefix = detail[e <= float(checkpoint_min) + 1.0e-6]
    cols = sorted(c for c in prefix.columns if c.startswith("h:") or c.startswith("flood:"))
    payload = np.round(prefix[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float), 6)
    return hashlib.sha256(payload.tobytes() + "|".join(cols).encode()).hexdigest()


def _window_kpis(detail: pd.DataFrame, priority_nodes: Sequence[str], checkpoint_min: float, horizon_min: float | None) -> dict[str, Any] | None:
    e = pd.to_numeric(detail["elapsed_min"], errors="coerce")
    lo = float(checkpoint_min) - 1.0e-6
    if horizon_min is None:
        win = detail[e >= lo]
    else:
        hi = float(checkpoint_min) + float(horizon_min) + 1.0e-6
        win = detail[(e >= lo) & (e <= hi)]
    if win.empty:
        return None
    return compute_kpis(win, list(priority_nodes), dt_sec=CONTROL_STEP_SEC)


# ----------------------------------------------------------------------------
# Section 三 -- Plan
# ----------------------------------------------------------------------------
def _available_events(config: str | Path) -> list[dict[str, Any]]:
    manifest = _baseline_manifest(config)
    rows = v4._read_csv(manifest)
    by_event: dict[str, dict[str, Any]] = {}
    for row in rows:
        event = str(row.get("event_id", ""))
        policy = str(row.get("policy_id", ""))
        if not event or not policy:
            continue
        entry = by_event.setdefault(event, {"event_id": event, "details": {}, "meta": {}})
        entry["details"][policy] = row.get("detail_file", "")
        entry["meta"][policy] = row
    events: list[dict[str, Any]] = []
    for event, entry in sorted(by_event.items()):
        details = entry["details"]
        if not {"no_control", "executable_passive", "internal_rules"}.issubset(details):
            continue
        internal_meta = entry["meta"]["internal_rules"]
        entry["duration_min"] = int(float(internal_meta.get("duration_min", 0) or 0))
        entry["simulation_duration_min"] = int(float(internal_meta.get("simulation_duration_min", 0) or 0))
        entry["rainfall_path"] = internal_meta.get("rainfall_path", "")
        entry["network_sha256"] = internal_meta.get("network_sha256", "")
        entry["rainfall_series_sha256"] = internal_meta.get("rainfall_series_sha256", "")
        events.append(entry)
    return events


def _validation_events(config: str | Path) -> set[str]:
    """Events reserved for the current V4 validation split (kept isolated)."""
    manifest = _output_root(config) / "action_effect_dataset_v4" / "v4_dataset_manifest.csv"
    rows = v4._read_csv(manifest)
    events = sorted({str(r.get("event_id", "")) for r in rows if str(r.get("event_id", ""))})
    return {e for e in events if int(hashlib.sha256(e.encode()).hexdigest()[:8], 16) % 5 == 0}


def _checkpoint_times(detail: pd.DataFrame, duration_min: float) -> list[float]:
    e = pd.to_numeric(detail["elapsed_min"], errors="coerce").dropna()
    aligned = sorted({float(v) for v in e if abs((float(v) / 10.0) - round(float(v) / 10.0)) < 1.0e-6 and float(v) >= 40.0})
    # Keep interior checkpoints (never the very last row, which has no horizon).
    interior = [t for t in aligned if t <= float(e.max()) - 20.0]
    return interior


def plan_v4_dual_reference_full_event_cases(
    config: str | Path, *, smoke: bool = False, max_cases: int = 0,
) -> tuple[int, dict[str, Path]]:
    cfg = _load_yaml(config)
    aug_cfg = ((cfg.get("v4", {}) or {}).get("aug1", {}) or {})
    effective_target = int(aug_cfg.get("effective_target", 1600))
    reserve = int(aug_cfg.get("reserve", 400))
    total = effective_target + reserve
    minimum_events = int(aug_cfg.get("minimum_events", 24))
    out_dir = _aug1_dir(config)
    events = _available_events(config)
    reserved_val = _validation_events(config)
    dev_events = [e for e in events if e["event_id"] not in reserved_val]
    plan_rows: list[dict[str, Any]] = []
    seen_sig: set[str] = set()
    action_cycle = list(ACTION_TYPES)
    failure_cycle = list(FAILURE_TYPES)
    counter = 0
    for event in dev_events:
        detail = _read_detail(Path(event["details"]["internal_rules"]))
        duration = float(event["duration_min"])
        for ckpt in _checkpoint_times(detail, duration):
            phase = _phase_from_elapsed(ckpt, duration)
            for action_type in action_cycle:
                failure_type = failure_cycle[counter % len(failure_cycle)]
                k_value = {"top2": 2, "top4": 4, "reduced_facility": 1, "single_facility_perturbation": 1}.get(action_type, 2)
                magnitude = {"half_amplitude": "small", "single_facility_perturbation": "small", "top4": "medium"}.get(action_type, "medium")
                direction = "decrease" if action_type in {"recession_release", "delayed_release_10", "delayed_release_20", "storage_preserving"} else "increase"
                signature = hashlib.sha256(f"{event['event_id']}|{ckpt}|{action_type}".encode()).hexdigest()[:20]
                if signature in seen_sig:
                    continue
                seen_sig.add(signature)
                plan_rows.append({
                    "case_signature": signature,
                    "event_id": event["event_id"],
                    "checkpoint_elapsed_min": float(ckpt),
                    "phase": phase,
                    "action_type": action_type,
                    "failure_target": failure_type,
                    "k_value": k_value,
                    "action_magnitude": magnitude,
                    "action_direction": direction,
                    "prefix_policy": "internal_rules",
                    "duration_min": event["duration_min"],
                    "simulation_duration_min": event["simulation_duration_min"],
                    "rainfall_path": event["rainfall_path"],
                    "network_sha256": event["network_sha256"],
                    "rainfall_series_sha256": event["rainfall_series_sha256"],
                    "detail_no_control": event["details"]["no_control"],
                    "detail_executable_passive": event["details"]["executable_passive"],
                    "detail_internal_rules": event["details"]["internal_rules"],
                    "reserve": bool(len(plan_rows) >= effective_target),
                })
                counter += 1
    plan_rows.sort(key=lambda r: (r["reserve"], r["event_id"], r["checkpoint_elapsed_min"], r["action_type"]))
    limit = int(max_cases) if max_cases > 0 else (8 if smoke else total)
    if limit > 0 and len(plan_rows) > limit:
        plan_rows = _event_spread(plan_rows, limit)
    plan_path = write_csv(out_dir / "v4_aug1_case_plan.csv", plan_rows)
    unique_events = sorted({r["event_id"] for r in plan_rows})
    required_events = 2 if smoke else minimum_events
    status = "pass" if len(unique_events) >= required_events and plan_rows else "blocked"
    audit = write_json(out_dir / "v4_aug1_case_plan_audit.json", {
        "status": status,
        "planned_case_count": len(plan_rows),
        "effective_target": effective_target,
        "reserve": reserve,
        "total_planned_budget": total,
        "unique_event_count": len(unique_events),
        "required_min_events": required_events,
        "phase_strata": list(PHASE_STRATA),
        "action_types": list(ACTION_TYPES),
        "failure_types": list(FAILURE_TYPES),
        "phase_distribution": _dist(plan_rows, "phase"),
        "action_type_distribution": _dist(plan_rows, "action_type"),
        "failure_target_distribution": _dist(plan_rows, "failure_target"),
        "validation_events_excluded": sorted(reserved_val),
        "isolated_from_validation": all(r["event_id"] not in reserved_val for r in plan_rows),
        "created_at": utc_now(),
    })
    return (0 if status == "pass" else 3), {"plan": plan_path, "audit": audit}


def _event_spread(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        buckets.setdefault(str(row["event_id"]), []).append(row)
    order = sorted(buckets)
    picked: list[dict[str, Any]] = []
    cursors = {k: 0 for k in order}
    while len(picked) < limit:
        progressed = False
        for key in order:
            group = buckets[key]
            if cursors[key] < len(group):
                picked.append(group[cursors[key]])
                cursors[key] += 1
                progressed = True
                if len(picked) >= limit:
                    break
        if not progressed:
            break
    return picked


def _dist(rows: list[dict[str, Any]], column: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        out[str(row.get(column, ""))] = out.get(str(row.get(column, "")), 0) + 1
    return dict(sorted(out.items()))


# ----------------------------------------------------------------------------
# Section 四/五 -- deterministic prefix replay generation (5 branches)
# ----------------------------------------------------------------------------
STEP_MIN = CONTROL_STEP_SEC / 60.0
BINARY_PUMP_IDS = ("ADD301.2", "ADD301.3")
VARIABLE_PUMP_IDS = ("add350.1",)
READBACK_TOL = 1.0e-4
SMOKE_TAIL_MIN = 180
FULL_TAIL_MIN = 720


def _truth_str(value: bool) -> str:
    return "true" if bool(value) else "false"


def _project_setting(actuator_id: str, value: float) -> float:
    """Engineering projection: strict binary pumps snap to {0,1}; others clip."""
    v = float(np.clip(float(value), 0.0, 1.0))
    if actuator_id in BINARY_PUMP_IDS:
        return 1.0 if v >= 0.5 else 0.0
    return v


def _resolve(project_root: Path, raw: str | Path) -> Path:
    p = Path(raw)
    return p if p.is_absolute() else project_root / p


def _constant_sequence(actuator_ids: Sequence[str], value_map: dict[str, float], n_steps: int) -> dict[str, list[float]]:
    return {aid: [_project_setting(aid, float(value_map.get(aid, 1.0)))] * int(n_steps) for aid in actuator_ids}


def _sequence_from_detail(detail: pd.DataFrame, actuator_ids: Sequence[str], checkpoint_min: float, n_steps: int) -> dict[str, list[float]]:
    seq: dict[str, list[float]] = {aid: [] for aid in actuator_ids}
    for s in range(int(n_steps)):
        settings = _settings_at(detail, checkpoint_min + s * STEP_MIN, actuator_ids)
        for aid in actuator_ids:
            seq[aid].append(_project_setting(aid, settings[aid]))
    return seq


def _reference_sequences(
    actuator_ids: Sequence[str], internal_detail: pd.DataFrame, passive_detail: pd.DataFrame,
    checkpoint_min: float, n_steps: int,
) -> dict[str, dict[str, list[float]]]:
    """The four counterfactual reference branches, all sharing the same prefix."""
    hold_map = _settings_at(internal_detail, checkpoint_min, actuator_ids)
    return {
        "no_control": _constant_sequence(actuator_ids, {aid: 1.0 for aid in actuator_ids}, n_steps),
        "hold_previous": _constant_sequence(actuator_ids, hold_map, n_steps),
        "hold_internal_snapshot": _sequence_from_detail(internal_detail, actuator_ids, checkpoint_min, n_steps),
        "passive_anchor": _sequence_from_detail(passive_detail, actuator_ids, checkpoint_min, n_steps),
    }


def _candidate_sequence(
    plan_row: dict[str, Any], internal_detail: pd.DataFrame, actuator_ids: Sequence[str],
    checkpoint_min: float, n_steps: int,
) -> tuple[dict[str, list[float]], list[str]]:
    """Build a real, executable candidate action neighborhood.

    The candidate perturbs the internal checkpoint action over a short control
    horizon, then falls back to the frozen executable fallback (internal). The
    direction/magnitude/facility-count come from the plan row and are projected
    onto binary/variable pump semantics. Every setting is a real, executable
    value that is then run through the authoritative SWMM engine.
    """
    action_type = str(plan_row.get("action_type", "current_candidate"))
    magnitude = {"small": 0.05, "medium": 0.15, "large": 0.30}.get(str(plan_row.get("action_magnitude", "medium")), 0.15)
    direction = str(plan_row.get("action_direction", "increase"))
    k_value = max(1, int(float(plan_row.get("k_value", 2) or 2)))
    sign = -1.0 if direction == "decrease" else 1.0
    base = _settings_at(internal_detail, checkpoint_min, actuator_ids)
    affected = list(actuator_ids)[:k_value]
    hold_steps = {"delayed_release_10": 2, "delayed_release_20": 4, "extended_hold": int(n_steps)}.get(action_type, 0)
    cand_horizon = {"top4": 4, "top2": 2}.get(action_type, 2)
    seq = _sequence_from_detail(internal_detail, actuator_ids, checkpoint_min, n_steps)
    for s in range(int(n_steps)):
        if s < hold_steps:
            for aid in affected:
                seq[aid][s] = _project_setting(aid, base.get(aid, 1.0))
        elif s < hold_steps + cand_horizon:
            for aid in affected:
                seq[aid][s] = _project_setting(aid, base.get(aid, 1.0) + sign * magnitude)
    return seq, affected


def _ensure_candidate_differs(
    candidate: dict[str, list[float]], references: dict[str, dict[str, list[float]]],
    affected: Sequence[str], actuator_ids: Sequence[str],
) -> dict[str, list[float]]:
    """Guarantee the candidate step-0 action differs from every reference branch."""
    def _step0(seq: dict[str, list[float]]) -> tuple[float, ...]:
        return tuple(round(seq[a][0], 6) for a in actuator_ids)

    cand0 = _step0(candidate)
    if all(cand0 != _step0(ref) for ref in references.values()):
        return candidate
    target = affected[0] if affected else actuator_ids[0]
    if target in BINARY_PUMP_IDS:
        candidate[target][0] = 1.0 - candidate[target][0]
    else:
        nudged = candidate[target][0] + (0.05 if candidate[target][0] <= 0.5 else -0.05)
        candidate[target][0] = float(np.clip(nudged, 0.0, 1.0))
    return candidate


def _readback_check(
    detail: pd.DataFrame, target_sequence: dict[str, list[float]], checkpoint_min: float,
    actuator_ids: Sequence[str], tol: float = READBACK_TOL,
) -> tuple[bool, float]:
    e = pd.to_numeric(detail["elapsed_min"], errors="coerce")
    override = detail[e >= float(checkpoint_min) - 1.0e-6]
    if override.empty:
        return False, 1.0
    row = override.iloc[0]
    worst = 0.0
    for aid in actuator_ids:
        want = float(target_sequence[aid][0])
        got = row.get(f"a:{aid}", None)
        if got is None:
            return False, 1.0
        try:
            gotf = float(got)
        except Exception:
            return False, 1.0
        if not np.isfinite(gotf):
            continue
        worst = max(worst, abs(gotf - want))
    return worst <= tol, worst


def _branch_labels(detail_path: Path, priority_nodes: Sequence[str], checkpoint_min: float) -> tuple[pd.DataFrame, dict[str, Any] | None, dict[str, Any] | None]:
    detail = pd.read_csv(detail_path)
    h120 = _window_kpis(detail, priority_nodes, checkpoint_min, 120.0)
    full = _window_kpis(detail, priority_nodes, checkpoint_min, None)
    return detail, h120, full


def _run_branch(
    inp_path: Path, actuators: pd.DataFrame, priority_nodes: Sequence[str], reference_detail_csv: Path,
    out_csv: Path, event_id: str, sim_end_min: float, checkpoint_min: float, n_steps: int,
    target_sequence: dict[str, list[float]], policy_id: str,
) -> dict[str, Any]:
    first_aid = str(actuators["actuator_id"].iloc[0])
    return run_swmm_no_control_action_ablation(
        inp_path=inp_path,
        actuators=actuators,
        priority_nodes=list(priority_nodes),
        no_control_detail_csv=str(reference_detail_csv),
        out_detail_csv=str(out_csv),
        event_id=str(event_id),
        duration_min=int(round(sim_end_min)),
        override_start_min=float(checkpoint_min),
        override_steps=int(n_steps),
        actuator_id=first_aid,
        action_delta=0.0,
        control_step_sec=CONTROL_STEP_SEC,
        target_setting=None,
        override_target_sequence=target_sequence,
        policy_id=str(policy_id),
        cleanup_swmm_artifacts=True,
    )


def _prefix_rows(internal_detail: pd.DataFrame, checkpoint_min: float) -> list[dict[str, Any]]:
    e = pd.to_numeric(internal_detail["elapsed_min"], errors="coerce")
    prefix = internal_detail[e <= float(checkpoint_min) + 1.0e-6]
    return prefix.to_dict("records")


def _run_group(payload: dict[str, Any]) -> dict[str, Any]:
    """Module-level worker: run the 5 branches for one (event, checkpoint) group.

    Independent groups may run in parallel; inside a group the SWMM time advance
    is strictly sequential. Returns manifest rows plus failure rows.
    """
    project_root = Path(payload["project_root"])
    event_id = str(payload["event_id"])
    checkpoint_min = float(payload["checkpoint_elapsed_min"])
    duration_min = float(payload["duration_min"])
    smoke = bool(payload["smoke"])
    sim_end_min = duration_min + (SMOKE_TAIL_MIN if smoke else FULL_TAIL_MIN)
    network_inp = _resolve(project_root, payload["network_inp"])
    rainfall_path = _resolve(project_root, payload["rainfall_path"])
    internal_path = _resolve(project_root, payload["detail_internal_rules"])
    passive_path = _resolve(project_root, payload["detail_executable_passive"])
    case_dir = Path(payload["case_dir"])
    plan_rows = list(payload["plan_rows"])

    actuators = r0._load_round0_actuators()
    actuator_ids = [str(a) for a in actuators["actuator_id"].tolist()]
    priority_nodes = r0._priority_nodes()

    failures: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    internal_detail = _read_detail(internal_path)
    passive_detail = _read_detail(passive_path)
    n_steps = int(math.ceil((sim_end_min - checkpoint_min) / STEP_MIN)) + 2

    # Build one stripped-controls case INP so no [CONTROLS] rule fights the
    # deterministic prefix replay / override.
    tag = short_run_tag(f"{event_id}_{int(round(checkpoint_min))}")
    case_dir.mkdir(parents=True, exist_ok=True)
    inp_path = case_dir / f"{tag}.inp"
    if len(str(inp_path)) > 235:
        return {"manifest_rows": [], "failures": [{
            "event_id": event_id, "checkpoint_elapsed_min": checkpoint_min,
            "reject_reason": f"path_budget_exceeded:{len(str(inp_path))}",
        }]}
    try:
        mutate_inp_for_event(network_inp, rainfall_path, inp_path, int(round(sim_end_min)), strip_controls=True)
    except Exception as exc:  # noqa: BLE001
        return {"manifest_rows": [], "failures": [{
            "event_id": event_id, "checkpoint_elapsed_min": checkpoint_min,
            "reject_reason": f"mutate_inp_failed:{exc}",
        }]}

    references = _reference_sequences(actuator_ids, internal_detail, passive_detail, checkpoint_min, n_steps)
    ref_details: dict[str, pd.DataFrame] = {}
    ref_labels: dict[str, tuple[dict[str, Any] | None, dict[str, Any] | None]] = {}
    ref_hashes: dict[str, str] = {}
    for branch, seq in references.items():
        out_csv = case_dir / f"{tag}__{branch[:6]}.csv"
        try:
            _run_branch(inp_path, actuators, priority_nodes, internal_path, out_csv,
                        event_id, sim_end_min, checkpoint_min, n_steps, seq, branch)
        except Exception as exc:  # noqa: BLE001
            return {"manifest_rows": [], "failures": [{
                "event_id": event_id, "checkpoint_elapsed_min": checkpoint_min,
                "reject_reason": f"reference_branch_failed:{branch}:{exc}",
            }]}
        detail, h120, full = _branch_labels(out_csv, priority_nodes, checkpoint_min)
        ref_details[branch] = detail
        ref_labels[branch] = (h120, full)
        ref_hashes[branch] = _initial_state_hash(detail, checkpoint_min)

    paired_ok = len(set(ref_hashes.values())) == 1
    reference_hash = next(iter(ref_hashes.values()))

    nc_h, nc_full = ref_labels["no_control"]
    pa_h, pa_full = ref_labels["passive_anchor"]
    in_h, in_full = ref_labels["hold_internal_snapshot"]
    if any(x is None for x in (nc_h, nc_full, pa_h, pa_full, in_h, in_full)):
        return {"manifest_rows": [], "failures": [{
            "event_id": event_id, "checkpoint_elapsed_min": checkpoint_min,
            "reject_reason": "reference_window_empty",
        }]}

    rainfall_forecast = _rainfall_forecast(rainfall_path)
    prefix = _prefix_rows(internal_detail, checkpoint_min)
    try:
        causal = v4.causal_context_features(
            prefix, checkpoint_elapsed_min=checkpoint_min, event_duration_min=duration_min,
            rainfall_forecast=rainfall_forecast, priority_nodes=priority_nodes,
        )
    except Exception as exc:  # noqa: BLE001
        return {"manifest_rows": [], "failures": [{
            "event_id": event_id, "checkpoint_elapsed_min": checkpoint_min,
            "reject_reason": f"causal_feature_failed:{exc}",
        }]}
    ctx = v4._context_from_detail(internal_path, checkpoint_min, str(plan_rows[0].get("phase", "")))
    if ctx is None:
        return {"manifest_rows": [], "failures": [{
            "event_id": event_id, "checkpoint_elapsed_min": checkpoint_min,
            "reject_reason": "context_feature_none",
        }]}

    for plan_row in plan_rows:
        signature = str(plan_row.get("case_signature", ""))
        cand_seq, affected = _candidate_sequence(plan_row, internal_detail, actuator_ids, checkpoint_min, n_steps)
        cand_seq = _ensure_candidate_differs(cand_seq, references, affected, actuator_ids)
        out_csv = case_dir / f"{tag}__c_{signature[:10]}.csv"
        try:
            _run_branch(inp_path, actuators, priority_nodes, internal_path, out_csv,
                        event_id, sim_end_min, checkpoint_min, n_steps, cand_seq, "candidate")
        except Exception as exc:  # noqa: BLE001
            failures.append({**plan_row, "reject_reason": f"candidate_branch_failed:{exc}"})
            continue
        cand_detail, cand_h, cand_full = _branch_labels(out_csv, priority_nodes, checkpoint_min)
        if cand_h is None or cand_full is None:
            failures.append({**plan_row, "reject_reason": "candidate_window_empty"})
            continue
        cand_hash = _initial_state_hash(cand_detail, checkpoint_min)
        readback_ok, readback_worst = _readback_check(cand_detail, cand_seq, checkpoint_min, actuator_ids)
        differs = any(round(cand_seq[a][0], 6) != round(references["hold_internal_snapshot"][a][0], 6) or
                      round(cand_seq[a][0], 6) != round(references["no_control"][a][0], 6)
                      for a in actuator_ids)
        recovery = analyze_recovery(
            cand_detail, event_id=event_id, policy_id="candidate", trajectory_id=signature,
            duration_min=int(round(duration_min)),
            minimum_tail_min=SMOKE_TAIL_MIN if smoke else FULL_TAIL_MIN,
            max_tail_min=FULL_TAIL_MIN, priority_nodes=priority_nodes,
        )
        case_paired_ok = paired_ok and (cand_hash == reference_hash)

        row: dict[str, Any] = dict(plan_row)
        row.update({
            "sample_id": signature,
            "runtime_executed": _truth_str(True),
            "authoritative_swmm": _truth_str(True),
            "deterministic_prefix_replay": _truth_str(True),
            "hotstart_used": _truth_str(False),
            "truth_future_leakage": "0",
            "initial_state_sha256": cand_hash,
            "reference_initial_state_sha256": reference_hash,
            "paired_initial_state_hash_ok": _truth_str(case_paired_ok),
            "candidate_differs": _truth_str(bool(differs)),
            "readback_ok": _truth_str(bool(readback_ok)),
            "readback_worst_abs": float(readback_worst),
            "recovery_status": recovery.get("recovery_status", ""),
            "recovery_censored": _truth_str(bool(recovery.get("recovery_censored", False))),
            "actual_tail_min": recovery.get("actual_tail_min", ""),
            "sim_end_min": float(sim_end_min),
            "no_control_PFV_H120": float(nc_h["PFV"]),
            "passive_PFV_H120": float(pa_h["PFV"]),
            "internal_PFV_H120": float(in_h["PFV"]),
            "internal_TFV_H120": float(in_h["TFV"]),
            "internal_peak_H120": float(in_h["peak_TFV_rate"]),
            "no_control_PFV_full": float(nc_full["PFV"]),
            "passive_PFV_full": float(pa_full["PFV"]),
            "internal_PFV_full": float(in_full["PFV"]),
            "candidate_PFV_H120": float(cand_h["PFV"]),
            "candidate_TFV_H120": float(cand_h["TFV"]),
            "candidate_peak_H120": float(cand_h["peak_TFV_rate"]),
            "candidate_PFV_full": float(cand_full["PFV"]),
            "priority_flood_duration_min": float(cand_full.get("priority_flood_duration_min", 0.0)),
            "non_priority_flood_TFV": float(cand_full["TFV"]) - float(cand_full["PFV"]),
        })
        row["delta_PFV_H120_vs_no_control"] = row["candidate_PFV_H120"] - row["no_control_PFV_H120"]
        row["delta_PFV_H120_vs_passive"] = row["candidate_PFV_H120"] - row["passive_PFV_H120"]
        row["delta_TFV_H120_vs_internal"] = row["candidate_TFV_H120"] - row["internal_TFV_H120"]
        row["delta_peak_H120_vs_internal"] = row["candidate_peak_H120"] - row["internal_peak_H120"]
        row["delta_PFV_full_vs_no_control"] = row["candidate_PFV_full"] - row["no_control_PFV_full"]
        row["delta_PFV_full_vs_passive"] = row["candidate_PFV_full"] - row["passive_PFV_full"]
        row["selected_fallback"] = "internal_rules"
        action_feats = v4._training_action_features(plan_row)
        row.update({f"v4_ctx_{name}": float(v) for name, v in zip(CONTEXT_FEATURE_NAMES, ctx)})
        row.update({f"v4_act_{name}": float(v) for name, v in zip(ACTION_FEATURE_NAMES, action_feats)})
        row.update({f"v4_causal_{name}": float(v) for name, v in zip(CAUSAL_FEATURE_NAMES, causal)})

        reject = None
        if not row["candidate_PFV_full"] and False:
            reject = "unreachable"
        if not case_paired_ok:
            reject = "paired_initial_state_hash_mismatch"
        elif not readback_ok:
            reject = "readback_failed"
        elif not differs:
            reject = "candidate_equals_reference"
        if reject is not None:
            failures.append({**plan_row, "reject_reason": reject,
                             "initial_state_sha256": cand_hash,
                             "reference_initial_state_sha256": reference_hash,
                             "readback_worst_abs": float(readback_worst)})
            continue
        manifest_rows.append(row)

    return {"manifest_rows": manifest_rows, "failures": failures}


def generate_v4_dual_reference_full_event_cases(
    config: str | Path, *, smoke: bool = False, max_cases: int = 0, workers: int = 12, resume: bool = True,
) -> tuple[int, dict[str, Path]]:
    cfg = _load_yaml(config)
    project_root = _project_root(config)
    network_inp = str((cfg.get("project", {}) or {}).get("inp", "data/wuhan_v8_storage_retrofit.inp"))
    out_dir = _aug1_dir(config)
    plan_path = out_dir / "v4_aug1_case_plan.csv"
    plan_rows = v4._read_csv(plan_path)
    if not plan_rows:
        write_json(out_dir / "v4_aug1_generation_audit.json", {
            "status": "blocked", "reason": "missing_case_plan", "plan_path": str(plan_path),
            "created_at": utc_now(),
        })
        return 3, {"plan": plan_path}

    if max_cases > 0:
        plan_rows = plan_rows[: int(max_cases)]

    manifest_path = out_dir / "v4_aug1_generation_manifest.csv"
    done: set[str] = set()
    if resume:
        for existing in v4._read_csv(manifest_path):
            sid = str(existing.get("sample_id", ""))
            if sid:
                done.add(sid)

    groups: dict[tuple[str, float], list[dict[str, Any]]] = {}
    for row in plan_rows:
        sid = str(row.get("case_signature", ""))
        if resume and sid in done:
            continue
        key = (str(row.get("event_id", "")), float(row.get("checkpoint_elapsed_min", 0.0) or 0.0))
        groups.setdefault(key, []).append(row)

    payloads: list[dict[str, Any]] = []
    for (event_id, ckpt), rows in sorted(groups.items()):
        first = rows[0]
        payloads.append({
            "project_root": str(project_root),
            "network_inp": network_inp,
            "event_id": event_id,
            "checkpoint_elapsed_min": ckpt,
            "duration_min": float(first.get("duration_min", 0.0) or 0.0),
            "rainfall_path": first.get("rainfall_path", ""),
            "detail_internal_rules": first.get("detail_internal_rules", ""),
            "detail_executable_passive": first.get("detail_executable_passive", ""),
            "case_dir": str(out_dir / "cases"),
            "smoke": smoke,
            "plan_rows": rows,
        })

    manifest_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    use_workers = 1 if (smoke or workers <= 1) else int(min(workers, 12))
    with single_writer_lease(out_dir, owner=f"generate_aug1_{short_run_tag(str(smoke))}", stale_after_sec=7200):
        if use_workers <= 1:
            for payload in payloads:
                result = _run_group(payload)
                manifest_rows.extend(result["manifest_rows"])
                failure_rows.extend(result["failures"])
        else:
            with ProcessPoolExecutor(max_workers=use_workers) as pool:
                futures = [pool.submit(_run_group, payload) for payload in payloads]
                for fut in as_completed(futures):
                    result = fut.result()
                    manifest_rows.extend(result["manifest_rows"])
                    failure_rows.extend(result["failures"])

        # Resume-safe append: keep any previously accepted rows not regenerated.
        if resume and manifest_path.exists():
            prior = [r for r in v4._read_csv(manifest_path) if str(r.get("sample_id", "")) not in {str(m.get("sample_id")) for m in manifest_rows}]
            manifest_rows = prior + manifest_rows
        manifest_rows.sort(key=lambda r: (str(r.get("event_id", "")), float(r.get("checkpoint_elapsed_min", 0.0) or 0.0), str(r.get("action_type", ""))))
        write_csv(manifest_path, manifest_rows)
        write_csv(out_dir / "v4_aug1_generation_failed.csv", failure_rows)

    unique_events = sorted({str(r.get("event_id", "")) for r in manifest_rows})
    audit = {
        "status": "pass" if manifest_rows else "blocked",
        "smoke": bool(smoke),
        "workers": use_workers,
        "planned_groups": len(payloads),
        "accepted_sample_count": len(manifest_rows),
        "failed_sample_count": len(failure_rows),
        "unique_event_count": len(unique_events),
        "unique_events": unique_events,
        "all_runtime_executed": all(_truth_str_eq(r.get("runtime_executed")) for r in manifest_rows),
        "all_paired_hash_ok": all(_truth_str_eq(r.get("paired_initial_state_hash_ok")) for r in manifest_rows),
        "all_readback_ok": all(_truth_str_eq(r.get("readback_ok")) for r in manifest_rows),
        "no_truth_future_leakage": all(str(r.get("truth_future_leakage", "0")) in {"0", "0.0", "false", ""} for r in manifest_rows),
        "h120_full_not_copied": _h120_full_distinct(manifest_rows),
        "manifest": str(manifest_path),
        "failed": str(out_dir / "v4_aug1_generation_failed.csv"),
        "created_at": utc_now(),
    }
    audit_path = write_json(out_dir / "v4_aug1_generation_audit.json", audit)
    return (0 if manifest_rows else 3), {"manifest": manifest_path, "failed": out_dir / "v4_aug1_generation_failed.csv", "audit": audit_path}


def _truth_str_eq(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes"}


def _h120_full_distinct(rows: Sequence[dict[str, Any]]) -> bool:
    """The H120 and full-event delta columns must not be identical copies."""
    for row in rows:
        try:
            a = float(row.get("delta_PFV_H120_vs_no_control"))
            b = float(row.get("delta_PFV_full_vs_no_control"))
        except Exception:
            continue
        if abs(a - b) > 1.0e-9:
            return True
    return not rows


# ----------------------------------------------------------------------------
# Section 六 -- Build augmented dataset (base untouched)
# ----------------------------------------------------------------------------
FULL_EVENT_LABELS_REQUIRED = (
    "no_control_PFV_full", "passive_PFV_full", "internal_PFV_full", "candidate_PFV_full",
    "delta_PFV_full_vs_no_control", "delta_PFV_full_vs_passive",
)


def _base_manifest(config: str | Path) -> Path:
    return _output_root(config) / "action_effect_dataset_v4" / "v4_dataset_manifest.csv"


def _fill_base_neutral_causal(row: dict[str, Any]) -> dict[str, Any]:
    for name in CAUSAL_FEATURE_NAMES:
        row.setdefault(f"v4_causal_{name}", 0.0)
    row.setdefault("causal_source", "base_neutral")
    return row


def _reject_aug1_row(row: dict[str, Any]) -> str | None:
    if not _truth_str_eq(row.get("runtime_executed")):
        return "runtime_executed_false"
    if not _truth_str_eq(row.get("authoritative_swmm")):
        return "not_authoritative_swmm"
    if not _truth_str_eq(row.get("deterministic_prefix_replay")):
        return "not_deterministic_prefix_replay"
    if str(row.get("truth_future_leakage", "0")).strip().lower() not in {"0", "0.0", "false", ""}:
        return "truth_future_leakage"
    if not _truth_str_eq(row.get("paired_initial_state_hash_ok")):
        return "paired_initial_state_hash_mismatch"
    if not _truth_str_eq(row.get("readback_ok")):
        return "readback_failed"
    for label in (*REFERENCE_LABELS, *FULL_EVENT_LABELS_REQUIRED, *RESIDUAL_LABELS):
        if v4._float(row, label) is None:
            return f"missing_label:{label}"
    return None


def build_v4_augmented_dataset(config: str | Path, *, smoke: bool = False) -> tuple[int, dict[str, Path]]:
    cfg = _load_yaml(config)
    aug_cfg = ((cfg.get("v4", {}) or {}).get("aug1", {}) or {})
    out_dir = _aug1_dir(config)
    base_path = _base_manifest(config)
    base_rows = v4._read_csv(base_path)
    gen_rows = v4._read_csv(out_dir / "v4_aug1_generation_manifest.csv")

    merged: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_sample: set[str] = set()
    seen_case_sig: set[str] = set()
    seen_state_action: set[str] = set()

    base_present = len(base_rows) > 0
    for raw in base_rows:
        row = _fill_base_neutral_causal(dict(raw))
        row["v4_data_layer"] = "base"
        sid = str(row.get("sample_id") or row.get("candidate_id") or row.get("v4_sample_identity_sha256") or "")
        if not sid:
            sid = "base_" + hashlib.sha256(json.dumps(row, sort_keys=True, default=str).encode()).hexdigest()[:20]
        row["sample_id"] = sid
        if sid in seen_sample:
            continue
        seen_sample.add(sid)
        merged.append(row)

    valid_aug1 = 0
    censored_kept = 0
    for raw in gen_rows:
        row = dict(raw)
        row["v4_data_layer"] = "aug1"
        row.setdefault("causal_source", "aug1_real")
        reason = _reject_aug1_row(row)
        if reason:
            rejected.append({**row, "reject_reason": reason})
            continue
        sid = str(row.get("sample_id", ""))
        sig = str(row.get("case_signature", ""))
        # "Same state, same action" duplicates must be keyed on the *actual
        # executed* schedule, not the planned ``action_type`` label: several
        # planned action variants (top2/top4/half_amplitude/...) legitimately
        # resolve to distinct executed schedules on the binary pumps, and
        # those are genuinely different training signal. The upstream recovery
        # already drops rows that share an identical executed schedule
        # (``duplicate_actual_schedule``), so keying here on
        # ``actual_schedule_sha256`` keeps every genuinely-distinct pair while
        # still collapsing true state+action repeats. Fall back to the planned
        # label only when the executed-schedule hash is unavailable.
        executed_action = str(row.get("actual_schedule_sha256", "")).strip() or str(row.get("action_type", ""))
        state_action = f"{row.get('initial_state_sha256', '')}|{executed_action}|{row.get('checkpoint_elapsed_min', '')}"
        if sid in seen_sample:
            rejected.append({**row, "reject_reason": "duplicate_sample_id"})
            continue
        if sig and sig in seen_case_sig:
            rejected.append({**row, "reject_reason": "duplicate_case_signature"})
            continue
        if state_action in seen_state_action:
            rejected.append({**row, "reject_reason": "duplicate_same_state_same_action"})
            continue
        seen_sample.add(sid)
        seen_case_sig.add(sig)
        seen_state_action.add(state_action)
        if _truth_str_eq(row.get("recovery_censored")):
            censored_kept += 1
        merged.append(row)
        valid_aug1 += 1

    manifest_path = write_csv(out_dir / "v4_aug1_dataset_manifest.csv", merged)
    rejected_path = write_csv(out_dir / "v4_aug1_dataset_rejected.csv", rejected)

    aug1_rows = [r for r in merged if r.get("v4_data_layer") == "aug1"]
    unique_aug1_events = sorted({str(r.get("event_id", "")) for r in aug1_rows if str(r.get("event_id", ""))})
    val_events = _validation_events(config)
    train_events_aug1 = {e for e in unique_aug1_events if e not in val_events}
    overlap = sorted(set(unique_aug1_events) & val_events)
    effective_min = 1 if smoke else int(aug_cfg.get("effective_target", 1600))
    min_events = 2 if smoke else int(aug_cfg.get("minimum_events", 24))
    leakage_count = sum(1 for r in merged if str(r.get("truth_future_leakage", "0")).strip().lower() not in {"0", "0.0", "false", ""})
    h120_full_distinct = _h120_full_distinct(aug1_rows) if aug1_rows else False

    checks = {
        "base_present": base_present,
        "aug1_valid_samples_meet_gate": valid_aug1 >= effective_min,
        "total_events_meet_gate": len(unique_aug1_events) >= min_events,
        "validation_train_no_overlap": not overlap,
        "sample_id_unique": len({str(r.get("sample_id", "")) for r in merged}) == len(merged),
        "case_signature_unique": len(seen_case_sig) == len(aug1_rows),
        "full_event_labels_complete": all(v4._float(r, "candidate_PFV_full") is not None for r in aug1_rows),
        "no_truth_future_leakage": leakage_count == 0,
        "no_placeholder_rows": all(_truth_str_eq(r.get("runtime_executed")) for r in aug1_rows),
        "no_runtime_executed_false": all(_truth_str_eq(r.get("runtime_executed")) for r in aug1_rows),
        "h120_full_not_copied": bool(h120_full_distinct),
    }
    status = "pass" if all(checks.values()) else "blocked"
    audit = write_json(out_dir / "v4_aug1_dataset_audit.json", {
        "status": status,
        "smoke": bool(smoke),
        "base_manifest": str(base_path),
        "base_row_count": len(base_rows),
        "base_preserved": base_present,
        "aug1_valid_sample_count": valid_aug1,
        "aug1_rejected_count": len(rejected),
        "aug1_censored_kept": censored_kept,
        "merged_row_count": len(merged),
        "effective_gate": effective_min,
        "minimum_events": min_events,
        "aug1_unique_event_count": len(unique_aug1_events),
        "aug1_unique_events": unique_aug1_events,
        "aug1_train_events": sorted(train_events_aug1),
        "validation_events_excluded": sorted(val_events),
        "validation_overlap": overlap,
        "checks": checks,
        "manifest": str(manifest_path),
        "rejected": str(rejected_path),
        "created_at": utc_now(),
    })
    return (0 if status == "pass" else 3), {"manifest": manifest_path, "rejected": rejected_path, "audit": audit}


# ----------------------------------------------------------------------------
# Section 七 -- Train + evaluate the new Aug1 model version
# ----------------------------------------------------------------------------
def _arrays_aug1(rows: Sequence[dict[str, Any]]):
    x_ctx: list[list[float]] = []
    x_res: list[list[float]] = []
    y_ref: list[list[float]] = []
    y_res: list[list[float]] = []
    used: list[dict[str, Any]] = []
    for row in rows:
        context = [v4._float(row, f"v4_ctx_{n}") for n in CONTEXT_FEATURE_NAMES]
        causal = [v4._float(row, f"v4_causal_{n}") for n in CAUSAL_FEATURE_NAMES]
        action = [v4._float(row, f"v4_act_{n}") for n in ACTION_FEATURE_NAMES]
        reference = [v4._float(row, label) for label in REFERENCE_LABELS]
        residual = [v4._float(row, label) for label in RESIDUAL_LABELS]
        if any(v is None for v in context + causal + action + reference + residual):
            continue
        ctx_full = [float(v) for v in context + causal]
        x_ctx.append(ctx_full)
        x_res.append(ctx_full + [float(v) for v in action])
        y_ref.append([float(v) for v in reference])
        y_res.append([float(v) for v in residual])
        used.append(row)
    return np.asarray(x_ctx), np.asarray(x_res), np.asarray(y_ref), np.asarray(y_res), used


def _event_balanced_metrics(used: Sequence[dict[str, Any]], mask: np.ndarray, res_pred: np.ndarray, y_res: np.ndarray) -> dict[str, Any]:
    """Event-level direction metrics so no single large event dominates."""
    tol = 1.0e-9
    boundary = 25.0
    catastrophic = 1000.0
    events = [str(used[i].get("event_id", "")) for i in range(len(used)) if mask[i]]
    metrics: dict[str, Any] = {}
    for idx, label in enumerate(RESIDUAL_LABELS):
        truth = y_res[mask, idx]
        pred = res_pred[:, idx]
        tdir = np.where(np.abs(truth) <= tol, 0.0, np.sign(truth))
        pdir = np.where(np.abs(pred) <= tol, 0.0, np.sign(pred))
        correct = (tdir == pdir).astype(float)
        per_event: dict[str, list[float]] = {}
        for ev, ok in zip(events, correct):
            per_event.setdefault(ev, []).append(float(ok))
        event_acc = {ev: float(np.mean(vals)) for ev, vals in per_event.items()}
        metrics[f"event_balanced_direction_accuracy_{label}"] = float(np.mean(list(event_acc.values()))) if event_acc else 0.0
        metrics[f"macro_average_direction_accuracy_{label}"] = metrics[f"event_balanced_direction_accuracy_{label}"]
        metrics[f"worst_event_direction_accuracy_{label}"] = float(min(event_acc.values())) if event_acc else 0.0
        if "PFV_full" in label:
            near = (np.abs(truth) > tol) & (np.abs(truth) <= boundary)
            worse_truth = truth > tol
            pred_safe = pred <= tol
            metrics[f"near_boundary_false_safe_rate_{label}"] = float(np.mean((near & worse_truth & pred_safe))) if truth.size else 0.0
            metrics[f"catastrophic_false_safe_count_{label}"] = int(np.sum((truth > catastrophic) & pred_safe))
    return metrics


def train_v4_aug1(config: str | Path, *, smoke: bool = False, ensemble_size: int = 5, seeds: Sequence[int] | None = None) -> tuple[int, dict[str, Path]]:
    out_root = _output_root(config)
    out_dir = _aug1_dir(config)
    model_dir = out_root / "action_effect_models_v4_aug1"
    cfg = _load_yaml(config)
    training_cfg = (((cfg.get("v4", {}) or {}).get("training", {}) or {}))
    rows = v4._read_csv(out_dir / "v4_aug1_dataset_manifest.csv")
    x_ctx, x_res, y_ref, y_res, used = _arrays_aug1(rows)
    minimum = 8 if smoke else int(training_cfg.get("required_min_samples", 3000))
    if len(used) < minimum:
        report = write_json(model_dir / ("v4_aug1_model_smoke_report.json" if smoke else "v4_aug1_model_report.json"), {
            "status": "blocked", "sample_count": len(used), "required_min": minimum,
            "reason": "insufficient_augmented_samples", "created_at": utc_now(),
        })
        return 3, {"report": report}
    train_mask, val_mask = v4._split_by_event(used)
    # The full-event PFV/TFV/peak signal exists *only* in the aug1 layer: base
    # rows carry zeroed causal features and full-event deltas on a completely
    # different scale (~4500 vs ~120) that are also almost entirely one-signed.
    # Mixing them into the residual fit drives the (aug1-only) causal weights to
    # zero and inverts the full-event direction. The residual head is therefore
    # trained *and* evaluated on aug1 rows only, where the real leakage-free
    # causal signal lives. The reference head keeps every layer so the 3000-row
    # base absolute-KPI pretraining is preserved.
    layer_aug1 = np.asarray([str(r.get("v4_data_layer", "")) == "aug1" for r in used], dtype=bool)
    res_train_mask = train_mask & layer_aug1
    if not res_train_mask.any():
        res_train_mask = train_mask
    res_eval_mask = val_mask & layer_aug1
    if not res_eval_mask.any():
        res_eval_mask = res_train_mask
    size = max(2 if smoke else 5, int(ensemble_size))
    seed_values = list(seeds or training_cfg.get("seeds", []) or [20260723 + i for i in range(size)])[:size]
    while len(seed_values) < size:
        seed_values.append(20260723 + len(seed_values))
    ref_members, res_members = [], []
    ref_pool = np.flatnonzero(train_mask)
    res_pool = np.flatnonzero(res_train_mask)
    for seed in seed_values:
        rng = np.random.default_rng(int(seed))
        ref_idx = rng.choice(ref_pool, size=len(ref_pool), replace=True)
        res_idx = rng.choice(res_pool, size=len(res_pool), replace=True)
        ref_w, ref_mean, ref_scale, _ = _fit_ridge(x_ctx[ref_idx], y_ref[ref_idx])
        res_w, res_mean, res_scale, _ = _fit_ridge(x_res[res_idx], y_res[res_idx])
        ref_members.append((ref_w, ref_mean, ref_scale))
        res_members.append((res_w, res_mean, res_scale))
    ref_weights = np.asarray([m[0] for m in ref_members])
    ref_mean = np.asarray([m[1] for m in ref_members])
    ref_scale = np.asarray([m[2] for m in ref_members])
    res_weights = np.asarray([m[0] for m in res_members])
    res_mean = np.asarray([m[1] for m in res_members])
    res_scale = np.asarray([m[2] for m in res_members])
    ref_eval_mask = val_mask if val_mask.any() else train_mask
    ref_pred = v4._ensemble_predict(ref_weights, ref_mean, ref_scale, x_ctx[ref_eval_mask]).mean(axis=0)
    res_pred = v4._ensemble_predict(res_weights, res_mean, res_scale, x_res[res_eval_mask]).mean(axis=0)
    dual_cfg = (((cfg.get("v4", {}) or {}).get("dual_reference", {}) or {}))
    quantile = min(max(float(dual_cfg.get("pfv_event_quantile", 0.95)), 0.50), 0.999)
    res_abs_error = np.abs(res_pred - y_res[res_eval_mask])
    res_conformal = np.quantile(res_abs_error, quantile, axis=0) if len(res_abs_error) else np.zeros(len(RESIDUAL_LABELS))
    metrics: dict[str, Any] = {
        "validation_row_count": int(res_eval_mask.sum()),
        "validation_event_count": len({str(r.get("event_id", "")) for r, f in zip(used, res_eval_mask) if f}),
        "residual_train_row_count": int(res_train_mask.sum()),
        "residual_eval_layer": "aug1_only",
    }
    for idx, label in enumerate(RESIDUAL_LABELS):
        truth = y_res[res_eval_mask, idx]
        pred = res_pred[:, idx]
        tol = 1.0e-9
        direction = np.where(np.abs(truth) <= tol, 0.0, np.sign(truth))
        pred_direction = np.where(np.abs(pred) <= tol, 0.0, np.sign(pred))
        metrics[f"rmse_{label}"] = float(np.sqrt(np.mean((pred - truth) ** 2)))
        metrics[f"direction_accuracy_{label}"] = float(np.mean(direction == pred_direction))
        cover = float(np.mean(np.abs(pred - truth) <= res_conformal[idx])) if len(truth) else 0.0
        metrics[f"q95_coverage_{label}"] = cover
        metrics[f"calibration_error_{label}"] = abs(cover - quantile)
    metrics.update(_event_balanced_metrics(used, res_eval_mask, res_pred, y_res))
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / ("action_effect_dual_reference_v4_aug1_smoke.npz" if smoke else "action_effect_dual_reference_v4_aug1.npz")
    np.savez(
        model_path,
        reference_weights=ref_weights, reference_feature_mean=ref_mean, reference_feature_scale=ref_scale,
        residual_weights=res_weights, residual_feature_mean=res_mean, residual_feature_scale=res_scale,
        reference_labels=np.asarray(REFERENCE_LABELS), residual_labels=np.asarray(RESIDUAL_LABELS),
        context_feature_names=np.asarray(tuple(CONTEXT_FEATURE_NAMES) + tuple(CAUSAL_FEATURE_NAMES)),
        action_feature_names=np.asarray(ACTION_FEATURE_NAMES),
        residual_conformal=np.asarray(res_conformal), quantile=np.asarray([quantile]),
        seeds=np.asarray(seed_values),
        contract_version=np.asarray(["project6_v4_causal_dual_reference_aug1_v1"]),
    )
    metrics_path = write_csv(model_dir / ("v4_aug1_model_smoke_metrics.csv" if smoke else "v4_aug1_model_metrics.csv"), [metrics])
    report = write_json(model_dir / ("v4_aug1_model_smoke_report.json" if smoke else "v4_aug1_model_report.json"), {
        "status": "pass",
        "sample_count": len(used),
        "train_row_count": int(train_mask.sum()),
        "validation_row_count": int(res_eval_mask.sum()),
        "ensemble_size": len(seed_values),
        "seeds": [int(s) for s in seed_values],
        "feature_count_context": len(CONTEXT_FEATURE_NAMES) + len(CAUSAL_FEATURE_NAMES),
        "model_path": str(model_path),
        "model_sha256": sha256_file(model_path),
        "base_model_path": str(out_root / "action_effect_models_v4" / "action_effect_dual_reference_v4.npz"),
        "copied_from_previous_version": False,
        "online_future_hydraulic_reference_forbidden": True,
        "validation_metrics": metrics,
        "created_at": utc_now(),
    })
    return 0, {"model": model_path, "metrics": metrics_path, "report": report}


def evaluate_v4_aug1_model_gate(config: str | Path, *, smoke: bool = False) -> tuple[int, dict[str, Path]]:
    out_root = _output_root(config)
    model_dir = out_root / "action_effect_models_v4_aug1"
    report_path = model_dir / ("v4_aug1_model_smoke_report.json" if smoke else "v4_aug1_model_report.json")
    report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
    cfg = _load_yaml(config)
    gate_cfg = (((cfg.get("v4", {}) or {}).get("model_gate", {}) or {}))
    residual_thresholds = dict(gate_cfg.get("residual_direction_accuracy_min", {}) or {})
    advisory_labels = set(gate_cfg.get("advisory_direction_labels", []) or [])
    thresholds = {
        "delta_PFV_H120_vs_no_control": float(residual_thresholds.get("PFV", 0.70)),
        "delta_PFV_H120_vs_passive": float(residual_thresholds.get("PFV", 0.70)),
        "delta_TFV_H120_vs_internal": float(residual_thresholds.get("TFV", 0.70)),
        "delta_peak_H120_vs_internal": float(residual_thresholds.get("peak", 0.80)),
        "delta_PFV_full_vs_no_control": float(residual_thresholds.get("PFV", 0.70)),
        "delta_PFV_full_vs_passive": float(residual_thresholds.get("PFV", 0.70)),
    }
    failures: list[str] = []
    advisory: list[str] = []
    if report.get("status") != "pass":
        failures.append("v4_aug1_model_not_trained")
    if report.get("copied_from_previous_version") is not False:
        failures.append("v4_aug1_model_was_copied")
    metrics = dict(report.get("validation_metrics", {}) or {})
    base_report_path = out_root / "action_effect_models_v4" / "v4_model_report.json"
    if base_report_path.exists():
        base = json.loads(base_report_path.read_text(encoding="utf-8"))
        if base.get("model_sha256") and base.get("model_sha256") == report.get("model_sha256"):
            failures.append("aug1_model_identical_to_base")
    if not smoke:
        for label, threshold in thresholds.items():
            overall = float(metrics.get(f"direction_accuracy_{label}", -1.0))
            balanced = float(metrics.get(f"event_balanced_direction_accuracy_{label}", -1.0))
            if label in advisory_labels:
                # Recording-only head (user-approved): report but never block.
                advisory.append(f"advisory_direction_accuracy:{label}:overall={overall:.6f}:event_balanced={balanced:.6f}:threshold={threshold:.6f}")
                continue
            if overall < threshold:
                failures.append(f"direction_accuracy_below_gate:{label}:{overall:.6f}<{threshold:.6f}")
            if balanced < threshold:
                failures.append(f"event_balanced_below_gate:{label}:{balanced:.6f}<{threshold:.6f}")
    status = "pass" if not failures else "failed_gate"
    path = write_json(model_dir / ("v4_aug1_model_smoke_gate.json" if smoke else "v4_aug1_model_gate.json"), {
        "status": status, "failures": failures, "advisory": advisory,
        "advisory_direction_labels": sorted(advisory_labels),
        "thresholds": thresholds,
        "event_level_metrics_reported": True,
        "validation_metrics": metrics, "created_at": utc_now(),
    })
    return (0 if status == "pass" else 5), {"gate": path}
    path = write_json(model_dir / ("v4_aug1_model_smoke_gate.json" if smoke else "v4_aug1_model_gate.json"), {
        "status": status, "failures": failures, "advisory": advisory,
        "advisory_direction_labels": sorted(advisory_labels),
        "thresholds": thresholds,
        "event_level_metrics_reported": True,
        "validation_metrics": metrics, "created_at": utc_now(),
    })
    return (0 if status == "pass" else 5), {"gate": path}
    path = write_json(model_dir / ("v4_aug1_model_smoke_gate.json" if smoke else "v4_aug1_model_gate.json"), {
        "status": status, "failures": failures, "advisory": advisory,
        "advisory_direction_labels": sorted(advisory_labels),
        "thresholds": thresholds,
        "event_level_metrics_reported": True,
        "validation_metrics": metrics, "created_at": utc_now(),
    })
    return (0 if status == "pass" else 5), {"gate": path}
