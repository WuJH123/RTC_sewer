"""Development-only 64+ rainfall end-to-end pilot helpers for Project6 V4.2.

This module deliberately does not weaken the formal paper mainline.  It builds a
compute-bounded *development* population that is large enough to test whether the
Step1 -> Step2 -> PFV-first controller chain has engineering/scientific potential
before multi-seed formal training is authorised.

Key rules
---------
* default 96 rainfall fingerprints; hard minimum 64;
* prefer historical Train1600 evidence when it can satisfy the diversity gate;
* otherwise fill only from the already strict finite/aligned source-domain pool;
* keep >=3 candidate cases for every selected state so policy replay is a real
  choice rather than a one-candidate no-op;
* never promote source/unknown data into formal evidence;
* causal rainfall forecast helpers never consume realised future rainfall;
* baseline action helpers use H12/H3 semantics and K<=8 for EFD/Auto-RBC.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from .v42_fast_feasibility import (
    _hash_key,
    _read,
    _role_rows_core,
    build_fast_step1_aux_allowlist,
    build_fast_step2_core_dataset,
)
from .v42_reusable_pool_strict import _bool as _strict_bool


FAST_E2E_CONTRACT_ID = "PROJECT6_V42_FAST_E2E_64PLUS_V1"
MIN_RAINFALL_GROUPS = 64
DEFAULT_TARGET_RAINFALL_GROUPS = 96
DEFAULT_CANDIDATES_PER_STATE = 3
DEFAULT_MAX_CHANGED_FACILITIES = 8
PREFERRED_SOURCE_TOKENS = ("train1600",)


@dataclass(frozen=True)
class FastE2ESelectionResult:
    filtered_case_manifest: Path
    selection_audit: Path
    selected_cases: int
    selected_states: int
    selected_rainfall_groups: int
    preferred_rainfall_groups: int
    used_nonpreferred_fill: bool


def _write_table(frame: pd.DataFrame, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".parquet":
        frame.to_parquet(p, index=False)
    else:
        frame.to_csv(p, index=False)
    return p


def _state_key(case: Any) -> str:
    event = str(getattr(case, "event_id", ""))
    checkpoint = getattr(case, "checkpoint_min", None)
    try:
        cp = f"{float(checkpoint):.6f}"
    except (TypeError, ValueError):
        cp = str(checkpoint)
    return f"{event}::{cp}"


def _contains_any(text: str, tokens: Sequence[str]) -> bool:
    lowered = str(text).casefold()
    return any(str(token).casefold() in lowered for token in tokens if str(token).strip())


def _source_text(case: Any, roles: dict[str, Any]) -> str:
    values = [
        str(getattr(case, "source_experiment", "")),
        str(getattr(case, "case_id", "")),
        str(getattr(case, "event_id", "")),
    ]
    for row in roles.values():
        values.extend(
            [
                str(getattr(row, "source_experiment", "")),
                str(getattr(row, "detail_path", "")),
            ]
        )
    return "\n".join(values)


def build_fast_step1_aux_allowlist_64plus(
    *,
    manifest_path: str | Path,
    output_path: str | Path,
    target_groups: int = DEFAULT_TARGET_RAINFALL_GROUPS,
    min_groups: int = MIN_RAINFALL_GROUPS,
    seed: int = 42,
) -> dict[str, Any]:
    """Build an auxiliary Step1 allow-list and fail if rainfall diversity is too small."""
    if int(target_groups) < int(min_groups):
        raise ValueError("target_groups cannot be smaller than min_groups")
    payload = build_fast_step1_aux_allowlist(
        manifest_path=manifest_path,
        output_path=output_path,
        max_groups=int(target_groups),
        seed=int(seed),
    )
    selected = int(payload.get("selected_aux_groups", 0))
    if selected < int(min_groups):
        raise RuntimeError(
            f"fast Step1 requires at least {min_groups} independent rainfall groups; got {selected}"
        )
    payload = dict(payload)
    payload.update(
        {
            "fast_e2e_contract_id": FAST_E2E_CONTRACT_ID,
            "minimum_required_groups": int(min_groups),
            "diversity_gate_pass": True,
        }
    )
    Path(output_path).write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return payload


def select_fast_step2_cases_64plus(
    *,
    physical_manifest: str | Path,
    case_manifest: str | Path,
    split_manifest: str | Path,
    output_case_manifest: str | Path,
    audit_output: str | Path,
    target_groups: int = DEFAULT_TARGET_RAINFALL_GROUPS,
    min_groups: int = MIN_RAINFALL_GROUPS,
    candidates_per_state: int = DEFAULT_CANDIDATES_PER_STATE,
    preferred_source_tokens: Sequence[str] = PREFERRED_SOURCE_TOKENS,
    seed: int = 42,
) -> FastE2ESelectionResult:
    """Select diverse strict control-core cases for a meaningful policy replay.

    The selected manifest remains development-only.  It does not change any
    domain/provenance column; it merely chooses a compute-bounded subset from
    ``eligible_source_domain_counterfactual_aux``.
    """
    if int(target_groups) < int(min_groups):
        raise ValueError("target_groups cannot be smaller than min_groups")
    if int(candidates_per_state) < 2:
        raise ValueError("policy replay requires at least two candidates per state")

    physical = _read(physical_manifest)
    cases = _read(case_manifest)
    split = _read(split_manifest)
    if physical.empty or cases.empty or split.empty:
        raise ValueError("strict R0 manifests cannot be empty")
    if "eligible_source_domain_counterfactual_aux" not in cases.columns:
        raise KeyError("case manifest missing eligible_source_domain_counterfactual_aux")

    admitted = cases[_strict_bool(cases, "eligible_source_domain_counterfactual_aux")].copy()
    if "source_role" in admitted.columns:
        admitted = admitted[admitted["source_role"].astype(str) != "reserved_evaluation"].copy()
    if admitted.empty:
        raise ValueError("no strict source-domain control-core cases available")

    physical_by_id = {
        str(r.physical_identity_sha256): r for r in physical.itertuples(index=False)
    }
    split_by_id = {
        str(r.physical_identity_sha256): str(r.split_group_key)
        for r in split.itertuples(index=False)
    }

    entries: list[dict[str, Any]] = []
    blocked: list[dict[str, str]] = []
    for idx, case in zip(admitted.index.tolist(), admitted.itertuples(index=False)):
        case_uid = str(getattr(case, "case_uid", ""))
        try:
            roles = _role_rows_core(case, physical_by_id)
            group_keys = {
                split_by_id.get(str(getattr(r, "physical_identity_sha256", "")), "")
                for r in roles.values()
            }
            group_keys.discard("")
            if len(group_keys) != 1:
                raise ValueError("four branches do not resolve to one rainfall split group")
            group = next(iter(group_keys))
            text = _source_text(case, roles)
            entries.append(
                {
                    "case_uid": case_uid,
                    "row_index": int(idx),
                    "rainfall_group": group,
                    "state_key": _state_key(case),
                    "preferred_source": _contains_any(text, preferred_source_tokens),
                    "source_text": text,
                }
            )
        except Exception as exc:
            blocked.append({"case_uid": case_uid, "error": f"{type(exc).__name__}: {exc}"})

    if not entries:
        raise RuntimeError("no strict cases survived fast E2E metadata resolution")

    entry_frame = pd.DataFrame(entries)
    eligible_states: list[dict[str, Any]] = []
    for (group, state), state_rows in entry_frame.groupby(["rainfall_group", "state_key"], sort=True):
        if len(state_rows) < int(candidates_per_state):
            continue
        preferred = bool(state_rows["preferred_source"].all())
        eligible_states.append(
            {
                "rainfall_group": str(group),
                "state_key": str(state),
                "preferred_source": preferred,
                "case_uids": sorted(
                    state_rows["case_uid"].astype(str).tolist(),
                    key=lambda uid: (_hash_key(uid, seed), uid),
                ),
            }
        )
    if not eligible_states:
        raise RuntimeError(
            f"no states have >= {candidates_per_state} candidate cases for policy replay"
        )

    by_group: dict[str, list[dict[str, Any]]] = {}
    for item in eligible_states:
        by_group.setdefault(item["rainfall_group"], []).append(item)

    chosen_state_by_group: dict[str, dict[str, Any]] = {}
    for group, states in by_group.items():
        states.sort(
            key=lambda item: (
                not bool(item["preferred_source"]),
                _hash_key(str(item["state_key"]), seed),
                str(item["state_key"]),
            )
        )
        chosen_state_by_group[group] = states[0]

    ranked_groups = sorted(
        chosen_state_by_group,
        key=lambda g: (
            not bool(chosen_state_by_group[g]["preferred_source"]),
            _hash_key(g, seed),
            g,
        ),
    )
    available_groups = len(ranked_groups)
    if available_groups < int(min_groups):
        raise RuntimeError(
            f"fast Step2/Step3 requires at least {min_groups} rainfall groups with "
            f">={candidates_per_state} candidates/state; got {available_groups}"
        )
    selected_groups = ranked_groups[: min(int(target_groups), available_groups)]

    selected_case_uids: list[str] = []
    preferred_groups = 0
    selected_state_keys: list[str] = []
    for group in selected_groups:
        item = chosen_state_by_group[group]
        selected_state_keys.append(str(item["state_key"]))
        if bool(item["preferred_source"]):
            preferred_groups += 1
        selected_case_uids.extend(item["case_uids"][: int(candidates_per_state)])

    index_by_uid = {str(r.case_uid): idx for idx, r in admitted.iterrows()}
    selected_indices = [index_by_uid[uid] for uid in selected_case_uids]
    selected_cases = admitted.loc[selected_indices].copy()
    selected_cases["fast_e2e_selected"] = True
    selected_cases["fast_e2e_contract_id"] = FAST_E2E_CONTRACT_ID
    selected_cases["development_only"] = True
    selected_cases["formal_mainline_authorized"] = False
    _write_table(selected_cases, output_case_manifest)

    used_fill = preferred_groups < len(selected_groups)
    audit = {
        "contract_id": FAST_E2E_CONTRACT_ID,
        "development_only": True,
        "formal_mainline_authorized": False,
        "source_provenance_promoted": False,
        "admission_source": "eligible_source_domain_counterfactual_aux",
        "preferred_source_tokens": [str(x) for x in preferred_source_tokens],
        "target_rainfall_groups": int(target_groups),
        "minimum_rainfall_groups": int(min_groups),
        "candidates_per_state": int(candidates_per_state),
        "strict_admitted_cases": int(len(admitted)),
        "metadata_resolved_cases": int(len(entry_frame)),
        "eligible_state_count": int(len(eligible_states)),
        "available_rainfall_groups_with_candidate_choice": int(available_groups),
        "selected_rainfall_groups": int(len(selected_groups)),
        "selected_states": int(len(selected_state_keys)),
        "selected_cases": int(len(selected_cases)),
        "preferred_rainfall_groups": int(preferred_groups),
        "used_nonpreferred_fill": bool(used_fill),
        "selected_group_sha256": hashlib.sha256("\n".join(selected_groups).encode("utf-8")).hexdigest(),
        "selected_groups": selected_groups,
        "selected_state_keys": selected_state_keys,
        "blocked_examples": blocked[:50],
    }
    audit_path = Path(audit_output)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")
    return FastE2ESelectionResult(
        filtered_case_manifest=Path(output_case_manifest),
        selection_audit=audit_path,
        selected_cases=int(len(selected_cases)),
        selected_states=int(len(selected_state_keys)),
        selected_rainfall_groups=int(len(selected_groups)),
        preferred_rainfall_groups=int(preferred_groups),
        used_nonpreferred_fill=bool(used_fill),
    )


def build_fast_step2_dataset_64plus(
    *,
    project_root: str | Path,
    physical_manifest: str | Path,
    case_manifest: str | Path,
    split_manifest: str | Path,
    working_dir: str | Path,
    target_groups: int = DEFAULT_TARGET_RAINFALL_GROUPS,
    min_groups: int = MIN_RAINFALL_GROUPS,
    candidates_per_state: int = DEFAULT_CANDIDATES_PER_STATE,
    preferred_source_tokens: Sequence[str] = PREFERRED_SOURCE_TOKENS,
    seed: int = 42,
):
    """Select then materialise a 64+ rainfall Step2 dataset with replay choice."""
    work = Path(working_dir)
    work.mkdir(parents=True, exist_ok=True)
    selected_cases = work / "step2_fast_e2e_selected_cases.parquet"
    selection_audit = work / "step2_fast_e2e_selection_audit.json"
    selection = select_fast_step2_cases_64plus(
        physical_manifest=physical_manifest,
        case_manifest=case_manifest,
        split_manifest=split_manifest,
        output_case_manifest=selected_cases,
        audit_output=selection_audit,
        target_groups=target_groups,
        min_groups=min_groups,
        candidates_per_state=candidates_per_state,
        preferred_source_tokens=preferred_source_tokens,
        seed=seed,
    )
    output_manifest = work / "step2_fast_e2e_core_manifest.parquet"
    output_audit = work / "step2_fast_e2e_core_audit.json"
    materialized = build_fast_step2_core_dataset(
        project_root=project_root,
        physical_manifest=physical_manifest,
        case_manifest=selected_cases,
        split_manifest=split_manifest,
        output_manifest=output_manifest,
        audit_output=output_audit,
        max_cases=selection.selected_cases,
        seed=seed,
    )
    if int(materialized.rainfall_groups) < int(min_groups):
        raise RuntimeError(
            f"materialized fast Step2 dataset lost rainfall diversity: "
            f"{materialized.rainfall_groups} < {min_groups}"
        )
    frame = _read(output_manifest)
    state_counts = frame.groupby("state_key").size() if not frame.empty else pd.Series(dtype=int)
    if state_counts.empty or int(state_counts.min()) < int(candidates_per_state):
        raise RuntimeError("materialized fast Step2 dataset lost candidate multiplicity per state")
    return selection, materialized


def make_causal_rainfall_forecast(
    observed_history_mm_h: Sequence[float],
    *,
    horizon_steps: int = 12,
    hold_steps: int = 3,
    decay_steps: int = 3,
) -> np.ndarray:
    """Leak-free persistence/decay rainfall forecast from observations available at t."""
    history = np.asarray(list(observed_history_mm_h), dtype=float).reshape(-1)
    if history.size == 0 or not np.isfinite(history).all():
        raise ValueError("observed rainfall history must be finite and non-empty")
    if horizon_steps <= 0 or hold_steps < 0 or decay_steps < 0:
        raise ValueError("invalid rainfall forecast horizon")
    current = max(0.0, float(history[-1]))
    out = np.zeros(int(horizon_steps), dtype=np.float32)
    h = min(int(hold_steps), len(out))
    out[:h] = current
    remaining = len(out) - h
    d = min(int(decay_steps), remaining)
    if d > 0:
        fractions = np.linspace(1.0, 0.0, num=d + 1, dtype=np.float32)[1:]
        out[h : h + d] = current * fractions
    return out


def _local_fill(
    current_depth: np.ndarray,
    max_depth: np.ndarray,
    action_node_map: np.ndarray,
) -> np.ndarray:
    depth = np.asarray(current_depth, dtype=float).reshape(-1)
    cap = np.asarray(max_depth, dtype=float).reshape(-1)
    incidence = np.asarray(action_node_map, dtype=float)
    if incidence.ndim != 2 or incidence.shape[1] != depth.size or cap.size != depth.size:
        raise ValueError("baseline graph/depth dimensions mismatch")
    safe_cap = np.where(cap > 1.0e-6, cap, np.nan)
    fill = np.divide(depth, safe_cap, out=np.zeros_like(depth), where=np.isfinite(safe_cap))
    fill = np.clip(np.nan_to_num(fill, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.5)
    scores = np.zeros(incidence.shape[0], dtype=float)
    for i in range(incidence.shape[0]):
        nodes = np.flatnonzero(np.abs(incidence[i]) > 0)
        scores[i] = 0.0 if nodes.size == 0 else float(np.max(fill[nodes]))
    return np.clip(scores, 0.0, 1.0)


def _apply_binary(values: np.ndarray, binary_indices: Iterable[int]) -> np.ndarray:
    out = np.asarray(values, dtype=float).copy()
    for idx in binary_indices:
        i = int(idx)
        if 0 <= i < out.size:
            out[i] = 1.0 if out[i] >= 0.5 else 0.0
    return out


def _limit_k(desired: np.ndarray, anchor: np.ndarray, max_changed: int) -> np.ndarray:
    desired = np.asarray(desired, dtype=float).reshape(-1)
    anchor = np.asarray(anchor, dtype=float).reshape(-1)
    if desired.shape != anchor.shape:
        raise ValueError("desired/anchor shape mismatch")
    if max_changed < 0:
        raise ValueError("max_changed must be non-negative")
    diff = np.abs(desired - anchor)
    changed = np.flatnonzero(diff > 1.0e-9)
    if changed.size <= int(max_changed):
        return desired
    keep = changed[np.argsort(diff[changed], kind="mergesort")[-int(max_changed) :]]
    out = anchor.copy()
    out[keep] = desired[keep]
    return out


def control_horizon_sequence(
    desired_action: np.ndarray,
    anchor_action: np.ndarray,
    *,
    horizon_steps: int = 12,
    control_horizon_steps: int = 3,
) -> np.ndarray:
    desired = np.asarray(desired_action, dtype=np.float32).reshape(-1)
    anchor = np.asarray(anchor_action, dtype=np.float32).reshape(-1)
    if desired.shape != anchor.shape:
        raise ValueError("desired/anchor shape mismatch")
    if not (0 < int(control_horizon_steps) <= int(horizon_steps)):
        raise ValueError("invalid H12/H3 baseline horizon")
    seq = np.broadcast_to(anchor[None, :], (int(horizon_steps), anchor.size)).copy()
    seq[: int(control_horizon_steps)] = desired[None, :]
    return seq.astype(np.float32)


def build_development_baseline_actions(
    *,
    current_depth: np.ndarray,
    max_depth: np.ndarray,
    action_node_map: np.ndarray,
    anchor_action: np.ndarray,
    binary_indices: Iterable[int] = (),
    max_changed_facilities: int = DEFAULT_MAX_CHANGED_FACILITIES,
    efd_gain: float = 0.8,
    rbc_low: float = 0.30,
    rbc_high: float = 0.70,
) -> dict[str, np.ndarray]:
    """Generate H12/H3 development EFD, Auto-RBC and all-close schedules.

    EFD/Auto-RBC are lightweight screening baselines.  They use the same K<=8
    perturbation budget as Proposed; ``all_close`` is intentionally a negative
    control and is not K-limited.
    """
    anchor = np.clip(np.asarray(anchor_action, dtype=float).reshape(-1), 0.0, 1.0)
    fill = _local_fill(current_depth, max_depth, action_node_map)

    target = float(np.median(fill)) if fill.size else 0.0
    efd = np.clip(anchor + float(efd_gain) * (fill - target), 0.0, 1.0)
    efd = _apply_binary(efd, binary_indices)
    efd = _limit_k(efd, anchor, int(max_changed_facilities))

    rbc = anchor.copy()
    rbc[fill >= float(rbc_high)] = 1.0
    rbc[fill <= float(rbc_low)] = 0.0
    rbc = _apply_binary(rbc, binary_indices)
    rbc = _limit_k(rbc, anchor, int(max_changed_facilities))

    all_close = np.zeros_like(anchor)
    return {
        "efd": control_horizon_sequence(efd, anchor),
        "auto_rbc": control_horizon_sequence(rbc, anchor),
        "all_close": np.broadcast_to(all_close[None, :], (12, anchor.size)).copy().astype(np.float32),
    }


def nearest_recorded_action_proxy(
    target_sequence: np.ndarray,
    candidate_sequences: Sequence[np.ndarray],
    *,
    control_horizon_steps: int = 3,
) -> tuple[int, float]:
    """Return nearest historically simulated candidate over the controllable H3 prefix."""
    target = np.asarray(target_sequence, dtype=float)
    if target.ndim != 2:
        raise ValueError("target_sequence must be [H,A]")
    if not candidate_sequences:
        raise ValueError("candidate_sequences cannot be empty")
    best_idx = -1
    best = float("inf")
    h = min(int(control_horizon_steps), target.shape[0])
    for i, seq in enumerate(candidate_sequences):
        arr = np.asarray(seq, dtype=float)
        if arr.shape != target.shape:
            raise ValueError("candidate sequence shape mismatch")
        dist = float(np.mean(np.abs(arr[:h] - target[:h])))
        if dist < best:
            best = dist
            best_idx = i
    return best_idx, best
