"""Runtime adapter for the V4.2 targeted candidate expansion.

This module is intentionally opt-in.  Importing the existing production entry
point still uses the frozen selector.  Development runners may inject
``predict_and_decide`` from this module after the authoritative candidate-space
plan has been frozen.

The adapter expands the candidate population and raises the default scoring
budget to 384.  It does *not* claim to fix the rolling event-level PFV budget;
that requires the plant runtime to update ``RollingPfvBudgetState`` with
realised candidate and No-control interval PFV increments.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from sewerrtc.control.targeted_candidate_expansion_v42 import (
    CandidateExpansionConfig,
    generate_targeted_candidate_sequences,
)
from sewerrtc.v4 import v42_pfv_tfv_runtime_patch as legacy


DEFAULT_EXPANDED_CANDIDATE_BUDGET = 384


def _roles_from_actuators(actuators: pd.DataFrame) -> dict[str, str]:
    ids = actuators["actuator_id"].astype(str).tolist()
    for column in ("asset_role", "storage_control_type", "link_type"):
        if column in actuators.columns:
            values = actuators[column].fillna("").astype(str).tolist()
            return dict(zip(ids, values))
    return {aid: "" for aid in ids}


def _ranked_ids(actuators: pd.DataFrame) -> list[str]:
    frame = actuators.copy()
    if "candidate_priority" in frame.columns:
        frame = frame.sort_values(
            ["candidate_priority", "actuator_id"], kind="stable"
        )
    elif "influence_path_length" in frame.columns:
        frame = frame.sort_values(
            ["influence_path_length", "actuator_id"], kind="stable"
        )
    else:
        frame = frame.sort_values("actuator_id", kind="stable")
    return frame["actuator_id"].astype(str).tolist()


def expanded_global_tfv_sequences(
    base: np.ndarray,
    actuators: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Return global singles plus pair/quad/H3 profile candidates."""
    return generate_targeted_candidate_sequences(
        current_action=np.asarray(base, dtype=np.float32),
        actuator_ids=actuators["actuator_id"].astype(str).tolist(),
        actuator_roles=_roles_from_actuators(actuators),
        ranked_actuator_ids=_ranked_ids(actuators),
        horizon_steps=legacy.base_runtime.HORIZON_STEPS,
        controllable_prefix_steps=legacy.base_runtime.CONTROLLABLE_PREFIX_STEPS,
        config=CandidateExpansionConfig(
            max_candidates=DEFAULT_EXPANDED_CANDIDATE_BUDGET
        ),
    )


def predict_and_decide(*args: Any, **kwargs: Any):
    """Run the existing selector with the expanded deterministic population."""
    requested = int(
        kwargs.pop("max_candidate_sequences", DEFAULT_EXPANDED_CANDIDATE_BUDGET)
    )
    # The existing function resolves this name from its own module globals.
    # Patch only for the duration of the call so other execution lines cannot
    # silently inherit the development search population.
    previous = legacy._global_tfv_sequences
    legacy._global_tfv_sequences = expanded_global_tfv_sequences
    try:
        action, info = legacy.predict_and_decide(
            *args,
            max_candidate_sequences=max(
                requested, DEFAULT_EXPANDED_CANDIDATE_BUDGET
            ),
            **kwargs,
        )
    finally:
        legacy._global_tfv_sequences = previous
    info = dict(info)
    info.update(
        {
            "candidate_expansion_contract": "V42_TARGETED_CANDIDATE_EXPANSION_V1",
            "expanded_candidate_budget": DEFAULT_EXPANDED_CANDIDATE_BUDGET,
            "candidate_profiles": [
                "constant_h3",
                "early_pulse",
                "ramp_h3",
                "release_h3",
            ],
            "candidate_magnitudes": [0.05, 0.10, 0.20],
            "rolling_event_pfv_budget_wired": False,
            "formal_execution_authorized": False,
            "authorization_reason": (
                "targeted authoritative candidate-oracle development only; "
                "wire realised-prefix rolling PFV accounting before closed-loop Formal"
            ),
        }
    )
    return action, info
