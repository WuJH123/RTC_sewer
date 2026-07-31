"""Synthetic Train1600 V3 sample manifest builder for V4 readiness tests.

The real frozen manifest is multi-GB and immutable; these helpers build a
small, structurally faithful stand-in so the pure readiness / authorization
functions can be unit-tested without touching the frozen evidence.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from sewerrtc.v4.dataset import TEMPORAL_RESIDUALS

# Frozen thresholds (never modified): margins are 0, dead-zones are the
# Project6 values.
FROZEN_MARGINS = {"pfv": 0.0, "tfv": 0.0, "peak": 0.0}
FROZEN_DEAD_ZONES = {"pfv": 1.0, "tfv": 1.0, "peak": 0.001}

_SPLIT_EVENTS = {
    "train": ["e_tr_0", "e_tr_1", "e_tr_2", "e_tr_3"],
    "calibration": ["e_cal_0"],
    "locked_validation": ["e_lk_0", "e_lk_1"],
}


def _residuals(rng: np.random.Generator, *, steps: int = 12) -> dict[str, str]:
    return {
        column: json.dumps([float(x) for x in rng.normal(size=steps)])
        for column in TEMPORAL_RESIDUALS
    }


def make_manifest(*, k_values: tuple[int, ...] = (4, 6, 8), seed: int = 7) -> pd.DataFrame:
    """Build a passing/conditional-pass Train1600 V3 stand-in manifest.

    Defaults mirror the real frozen evidence: K values contain no K=1/K=2, so
    the online generator (K in {1,2,4,6,8}) exceeds the training domain and the
    learnability verdict degrades to ``conditional_pass``.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []
    for split, events in _SPLIT_EVENTS.items():
        for event_id in events:
            for cp in range(2):  # two states per event
                checkpoint_id = f"{event_id}_cp{cp}"
                for cand in range(5):  # five candidates per state
                    k = int(k_values[cand % len(k_values)])
                    row = {
                        "event_id": event_id,
                        "checkpoint_id": checkpoint_id,
                        "split": split,
                        "k_target": k,
                        "k_actual": k,
                        "candidate_family": f"fam_{cand % 3}",
                        "predicted_stratum": [
                            "boundary",
                            "high",
                            "low",
                            "fallback",
                        ][cand % 4],
                        "hard_negative_type": (
                            "toward_no_control" if cand == 0 else ""
                        ),
                        "rainfall_sha256": f"sha_{event_id}",
                        "delta_pfv_h120_vs_no_control": float(rng.normal() * 20.0),
                        "delta_tfv_h120_vs_dynamic_internal": float(
                            rng.normal() * 5000.0
                        ),
                        "delta_peak_h120_vs_dynamic_internal": float(
                            rng.normal() * 20.0
                        ),
                        # Core labels: two-sided within every split.
                        "pfv_safe": cand % 2 == 0,
                        "tfv_improved": cand % 2 == 1,
                        "peak_noninferior": cand % 3 != 0,
                        "joint_noninferior": cand == 0,
                        "materially_beneficial": cand % 2 == 0,
                        "neutral": cand == 4 and split == "train",
                    }
                    row.update(_residuals(rng))
                    rows.append(row)
    return pd.DataFrame(rows)


def make_learnability_payload(audit: dict) -> dict:
    """Slice a full learnability audit into the persisted verdict payload
    consumed by ``evaluate_model_training_authorization_v4``."""
    return {
        "verdict": audit["verdict"],
        "feature_leakage_audit": audit["feature_leakage_audit"],
        "residual_schema_audit": audit["residual_schema_audit"],
        "train_core_two_sided": audit["train_core_two_sided"],
        "locked_power_report": audit["locked_power_report"],
    }
