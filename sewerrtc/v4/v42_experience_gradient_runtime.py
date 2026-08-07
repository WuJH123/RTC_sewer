"""Opt-in V4.2 runtime for the new experience-guided differentiable MPC flow.

Flow:
    sparse sensing -> causal GAT state -> differentiable Step2 action effects
    -> authoritative experience warm starts -> continuous gradient refinement
    -> existing Engineering36 projection -> rolling PFV-UCB hard admission
    -> minimum TFV -> first 10-min execution by the plant runtime.

This adapter intentionally reuses the corrected PFV/TFV selector.  Gradient
search only proposes actions; it cannot bypass PFV admission or write/readback.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from sewerrtc.control.differentiable_hybrid_search_v42 import (
    DifferentiableSearchConfig,
    generate_differentiable_candidates,
)
from sewerrtc.control.experience_bank_v42 import (
    AuthoritativeExperienceBank,
    ExperienceRetrievalConfig,
    state_signature,
)
from sewerrtc.v4 import v42_pfv_tfv_runtime_patch as legacy


EXPERIENCE_GRADIENT_RUNTIME_CONTRACT = "PROJECT6_V42_EXPERIENCE_GRADIENT_PFV_TFV_MPC_V1"
DEFAULT_FINAL_CANDIDATE_BUDGET = 160
DEFAULT_BANK_RELATIVE_PATH = Path(
    "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/diagnostics/"
    "experience_gradient/AUTHORITATIVE_EXPERIENCE_BANK.parquet"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_bank_path(bundle: Any, path: str | Path | None) -> Path | None:
    if path is not None:
        source = Path(path)
        if not source.is_absolute():
            source = Path(bundle.project_root) / source
        return source
    default = Path(bundle.project_root) / DEFAULT_BANK_RELATIVE_PATH
    return default if default.exists() else None


def _load_bank(path: Path | None) -> AuthoritativeExperienceBank | None:
    if path is None:
        return None
    if not path.exists():
        raise FileNotFoundError(path)
    return AuthoritativeExperienceBank.load(path)


def predict_and_decide(
    *args: Any,
    experience_bank_path: str | Path | None = None,
    gradient_search_config: DifferentiableSearchConfig | None = None,
    retrieval_config: ExperienceRetrievalConfig | None = None,
    **kwargs: Any,
):
    """Run hybrid search, then delegate safety/selection to the frozen selector."""
    required = {
        "bundle",
        "actuators",
        "state_history",
        "historical_actions",
        "rainfall_forecast",
        "current_action",
        "internal_current_action",
    }
    missing = [name for name in required if name not in kwargs]
    if args:
        raise TypeError("experience-gradient runtime requires keyword arguments")
    if missing:
        raise TypeError(f"missing runtime arguments: {missing}")

    bundle = kwargs["bundle"]
    actuators = kwargs["actuators"]
    state_history = np.asarray(kwargs["state_history"], dtype=np.float32)
    historical_actions = np.asarray(kwargs["historical_actions"], dtype=np.float32)
    rainfall_forecast = np.asarray(kwargs["rainfall_forecast"], dtype=np.float32)
    current_action = np.asarray(kwargs["current_action"], dtype=np.float32)
    internal_current_action = np.asarray(kwargs["internal_current_action"], dtype=np.float32)
    rolling_state = kwargs.get("rolling_pfv_budget_state")
    actuator_ids = actuators["actuator_id"].astype(str).tolist()

    resolved_bank_path = _resolve_bank_path(bundle, experience_bank_path)
    bank = _load_bank(resolved_bank_path)
    warm_records: list[dict[str, Any]] = []
    if bank is not None:
        signature = state_signature(
            state_history=state_history,
            rainfall_forecast=rainfall_forecast,
            current_action=current_action,
        )
        warm_records = bank.retrieve(
            signature=signature,
            current_action=current_action,
            config=retrieval_config or ExperienceRetrievalConfig(),
        )
    warm_starts = [np.asarray(item["sequence"], dtype=np.float32) for item in warm_records]

    gradient_candidates, gradient_info = generate_differentiable_candidates(
        bundle=bundle,
        actuator_ids=actuator_ids,
        state_history=state_history,
        historical_actions=historical_actions,
        rainfall_forecast=rainfall_forecast,
        current_action=current_action,
        internal_current_action=internal_current_action,
        warm_starts=warm_starts,
        rolling_pfv_budget_state=rolling_state,
        config=gradient_search_config or DifferentiableSearchConfig(),
    )

    requested = int(kwargs.pop("max_candidate_sequences", DEFAULT_FINAL_CANDIDATE_BUDGET))
    action, info = legacy.predict_and_decide(
        **kwargs,
        max_candidate_sequences=max(requested, DEFAULT_FINAL_CANDIDATE_BUDGET),
        extra_candidate_sequences=gradient_candidates,
    )

    info = dict(info)
    info.update(
        {
            "runtime_contract": EXPERIENCE_GRADIENT_RUNTIME_CONTRACT,
            "candidate_search": "coverage_plus_experience_plus_differentiable_refinement",
            "experience_bank_used": bank is not None,
            "experience_bank_path": str(resolved_bank_path) if resolved_bank_path else None,
            "experience_bank_sha256": _sha256(resolved_bank_path) if resolved_bank_path else None,
            "experience_warm_start_count": len(warm_records),
            "experience_warm_start_state_keys": [str(x["state_key"]) for x in warm_records],
            "gradient_search": gradient_info,
            "final_candidate_budget_requested": requested,
            "final_candidate_budget_effective_minimum": DEFAULT_FINAL_CANDIDATE_BUDGET,
            "gradient_search_is_safety_authority": False,
            "pfv_admission_authority": "existing_calibrated_rolling_PFV_UCB",
            "objective_inside_admitted_set": "minimum_TFV",
            "execution_contract": "first_10min_then_replan",
        }
    )
    return action, info
