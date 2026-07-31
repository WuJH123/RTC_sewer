"""Synthetic Train1600-shaped fixtures for the V4 model tests.

Builds a tiny manifest + checkpoint catalog with the same columns and contract
as the frozen Train1600 V3 evidence, so the loader / model / stage logic can be
exercised without touching the multi-GB frozen artifacts.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from sewerrtc.v4.train_v4_loader import (
    ACCEPTANCE_GATE_COLUMNS,
    ALLOWED_STATE_FEATURE_COLUMNS,
    CLASSIFICATION_TARGET_COLUMNS,
    CONTINUOUS_TARGET_COLUMNS,
    PROCESS_RESIDUAL_COLUMNS,
    N_FACILITIES,
)
from sewerrtc.v4.train1600_v4 import HORIZON_STEPS

SPLIT_EVENTS = {
    "train": ["te0", "te1", "te2", "te3"],
    "calibration": ["ce0"],
    "locked_validation": ["le0"],
}


def _schedule(rng) -> str:
    mat = (rng.random((HORIZON_STEPS, N_FACILITIES)) > 0.5).astype(float)
    # one variable-speed style continuous entry
    mat[:, 13] = rng.random(HORIZON_STEPS)
    return json.dumps(mat.tolist())


def _residual(rng) -> str:
    return json.dumps((rng.standard_normal(HORIZON_STEPS)).round(4).tolist())


def make_manifest(
    *, states_per_event: int = 3, candidates: int = 5, seed: int = 11
) -> pd.DataFrame:
    """Synthetic manifest; event/state fully isolated per split."""
    rng = np.random.default_rng(seed)
    rows = []
    ck = 0
    for split, events in SPLIT_EVENTS.items():
        for event in events:
            for _ in range(states_per_event):
                checkpoint_id = f"{event}__{ck}"
                ck += 1
                for c in range(candidates):
                    # PFV: ~half near-zero (inactive), half active (hurdle).
                    pfv = 0.0 if c % 2 == 0 else float(rng.normal(20, 8))
                    tfv = float(rng.normal(0, 5000))
                    peak = float(rng.normal(0, 20))
                    # ensure two-sided classification labels within each split
                    row = {
                        "event_id": event,
                        "checkpoint_id": checkpoint_id,
                        "case_id": f"{checkpoint_id}__{c}",
                        "split": split,
                        "status": "planned",
                        "full_event_eligible": False,
                        "label_validity_full": False,
                        "hard_negative_type": (
                            "Peak_hard_negative"
                            if (split == "train" and c == 0)
                            else ("TFV_hard_negative" if c == 1 else "")
                        ),
                        "k_actual": int(rng.choice([4, 6, 8])),
                        "k_target": int(rng.choice([4, 6, 8])),
                        "is_noop": bool(c == 0),
                        "action_cost": float(rng.random()),
                        "actual_action_distance": float(rng.random()),
                        "requested_schedule_json": _schedule(rng),
                        "projected_schedule_json": _schedule(rng),
                        "anchor_schedule_json": _schedule(rng),
                        "feasible_rank": c,
                        "regret_to_exact_best": float(rng.random()),
                        "pfv_safe": bool(c % 2),
                        "tfv_improved": bool(c % 2 == 0),
                        "peak_noninferior": bool(c < 3),
                        "joint_noninferior": bool(c == 4),
                        "delta_pfv_h120_vs_no_control": pfv,
                        "delta_tfv_h120_vs_dynamic_internal": tfv,
                        "delta_peak_h120_vs_dynamic_internal": peak,
                    }
                    for col in ACCEPTANCE_GATE_COLUMNS:
                        row[col] = True
                    for col in PROCESS_RESIDUAL_COLUMNS:
                        row[col] = _residual(rng)
                    rows.append(row)
    return pd.DataFrame(rows)


def make_catalog(manifest: pd.DataFrame, *, seed: int = 12) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ckpts = manifest["checkpoint_id"].unique()
    data = {"checkpoint_id": ckpts}
    for col in ALLOWED_STATE_FEATURE_COLUMNS:
        data[col] = rng.random(len(ckpts))
    return pd.DataFrame(data)
