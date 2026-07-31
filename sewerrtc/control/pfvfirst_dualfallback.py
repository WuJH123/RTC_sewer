"""PFV-first dual-fallback controller interfaces for Project6 V3.

The classes here define the contract-level behavior. They are deliberately
side-effect free; PySWMM execution and model inference are connected by scripts
and runners after the user executes the runbook commands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Mapping, Sequence
import hashlib
import json

import numpy as np


@dataclass(frozen=True)
class ActionTrace:
    native_setting: np.ndarray
    anchor_setting: np.ndarray
    requested_residual: np.ndarray
    projected_setting: np.ndarray
    target_setting: np.ndarray
    actual_current_setting: np.ndarray
    actual_executed_setting: np.ndarray
    clipping: np.ndarray
    rate_limit: np.ndarray
    dwell: np.ndarray
    interlock: np.ndarray
    override_ttl: np.ndarray
    release: np.ndarray

    def validate_shape(self, actuator_count: int) -> None:
        for field_name, value in self.__dict__.items():
            arr = np.asarray(value)
            if arr.shape[-1] != actuator_count:
                raise ValueError(f"{field_name} last dimension must be {actuator_count}, got {arr.shape}")


@dataclass(frozen=True)
class FallbackPrediction:
    fallback_id: str
    action_seq: np.ndarray
    pfv_ucb: float
    tfv_ucb: float
    peak_ucb: float
    sentinel_storage_risk: float
    transition_action_cost: float
    legal: bool
    reason: str = ""


@dataclass(frozen=True)
class CandidatePrediction:
    candidate_id: str
    action_seq: np.ndarray
    pfv_lcb_improvement_vs_internal: float
    pfv_lcb_improvement_vs_fallback: float
    tfv_delta_ucb_vs_fallback: float
    peak_delta_ucb_vs_fallback: float
    ood_passed: bool
    backup_reachable_after_action: bool
    support_reason: str = ""


@dataclass(frozen=True)
class DualFallbackDecision:
    selected_fallback_id: str
    selected_candidate_id: str
    execute_action: np.ndarray
    fallback_action: np.ndarray
    rejection_reasons: tuple[str, ...] = field(default_factory=tuple)
    metadata: Mapping[str, object] = field(default_factory=dict)


def stable_file_hash(path: str | Path, *, algorithm: str = "sha256") -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    h = hashlib.new(algorithm)
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def select_safe_fallback(fallbacks: Sequence[FallbackPrediction]) -> FallbackPrediction:
    legal = [f for f in fallbacks if f.legal]
    if not legal:
        raise ValueError("no legal fallback candidate")
    return sorted(
        legal,
        key=lambda f: (
            float(f.tfv_ucb),
            float(f.peak_ucb),
            float(f.sentinel_storage_risk),
            float(f.pfv_ucb),
            float(f.transition_action_cost),
            str(f.fallback_id),
        ),
    )[0]


def filter_safe_candidates(
    candidates: Sequence[CandidatePrediction],
    *,
    minimum_internal_improvement: float,
) -> tuple[list[CandidatePrediction], list[str]]:
    accepted: list[CandidatePrediction] = []
    rejected: list[str] = []
    for c in candidates:
        reasons: list[str] = []
        if c.pfv_lcb_improvement_vs_internal < minimum_internal_improvement:
            reasons.append("pfv_internal_improvement_below_min")
        if c.pfv_lcb_improvement_vs_fallback <= 0.0:
            reasons.append("pfv_not_better_than_selected_fallback")
        if c.tfv_delta_ucb_vs_fallback > 0.0:
            reasons.append("tfv_worse_than_selected_fallback")
        if c.peak_delta_ucb_vs_fallback > 0.0:
            reasons.append("peak_worse_than_selected_fallback")
        if not c.ood_passed:
            reasons.append("ood_rejected")
        if not c.backup_reachable_after_action:
            reasons.append("backup_unreachable_after_action")
        if reasons:
            rejected.append(f"{c.candidate_id}:{'|'.join(reasons)}")
        else:
            accepted.append(c)
    return accepted, rejected


def choose_candidate(
    candidates: Sequence[CandidatePrediction],
    *,
    minimum_internal_improvement: float,
) -> tuple[CandidatePrediction | None, tuple[str, ...]]:
    accepted, rejected = filter_safe_candidates(
        candidates,
        minimum_internal_improvement=minimum_internal_improvement,
    )
    if not accepted:
        return None, tuple(rejected)
    # Among safe candidates, prefer stronger PFV improvement, then lower TFV and peak risk.
    best = sorted(
        accepted,
        key=lambda c: (
            -float(c.pfv_lcb_improvement_vs_internal),
            float(c.tfv_delta_ucb_vs_fallback),
            float(c.peak_delta_ucb_vs_fallback),
            str(c.candidate_id),
        ),
    )[0]
    return best, tuple(rejected)


def decide_dualfallback(
    *,
    fallbacks: Sequence[FallbackPrediction],
    candidates: Sequence[CandidatePrediction],
    minimum_internal_improvement: float,
) -> DualFallbackDecision:
    fallback = select_safe_fallback(fallbacks)
    candidate, rejected = choose_candidate(
        candidates,
        minimum_internal_improvement=minimum_internal_improvement,
    )
    if candidate is None:
        return DualFallbackDecision(
            selected_fallback_id=fallback.fallback_id,
            selected_candidate_id=fallback.fallback_id,
            execute_action=np.asarray(fallback.action_seq)[0].copy(),
            fallback_action=np.asarray(fallback.action_seq)[0].copy(),
            rejection_reasons=rejected,
            metadata={"mode": "fallback_only"},
        )
    return DualFallbackDecision(
        selected_fallback_id=fallback.fallback_id,
        selected_candidate_id=candidate.candidate_id,
        execute_action=np.asarray(candidate.action_seq)[0].copy(),
        fallback_action=np.asarray(fallback.action_seq)[0].copy(),
        rejection_reasons=rejected,
        metadata={"mode": "learned_candidate"},
    )


def manifest_record(paths: Iterable[str | Path]) -> dict[str, object]:
    records = []
    for path in paths:
        p = Path(path)
        records.append(
            {
                "path": str(p),
                "exists": p.exists(),
                "sha256": stable_file_hash(p) if p.exists() and p.is_file() else None,
            }
        )
    return {"files": records}


def write_manifest(path: str | Path, payload: Mapping[str, object]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
