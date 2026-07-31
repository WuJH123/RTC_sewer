"""Gate P3 feasibility state catalog and search-policy helpers.

Builds the 32-responsive-state catalog required by the P3 contract from the
immutable Pilot Dataset v2 sample manifest.  Read-only over v1/v2 evidence;
all outputs live under ``pilot_feasibility_p3/``.

The Exact-SWMM search is a development diagnostic: states outside
``pilot_train`` are flagged so their search results can never enter
training or tune candidate-generator thresholds (contract
``evaluation_state_policy``).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .pilot_extension import _bool, _facility_moved_mask

REFERENCE_BRANCHES = ("no_control", "dynamic_internal_rules", "hold_previous")

_DELTA_PFV = "delta_pfv_h120_vs_no_control"
_DELTA_TFV = "delta_tfv_h120_vs_dynamic_internal"
_DELTA_PEAK = "delta_peak_h120_vs_dynamic_internal"

_STATE_KEYS = ["event_id", "checkpoint_id"]

TRAIN_SPLIT = "pilot_train"


def _best_sample(
    group: pd.DataFrame, metric: str, mask: pd.Series | None = None
) -> tuple[str, float]:
    """(sample_id, value) of the minimum metric row, optionally masked."""
    values = pd.to_numeric(group[metric], errors="coerce")
    if mask is not None:
        values = values.where(mask)
    if values.notna().sum() == 0:
        return "", float("nan")
    index = values.idxmin()
    return str(group.loc[index, "sample_id"]), float(values.loc[index])


def build_feasibility_state_catalog(
    samples: pd.DataFrame,
    *,
    facility_ids: list[str],
    reference_root_rel: str = "pilot/references",
) -> pd.DataFrame:
    """One row per responsive Pilot v2 state with P3 search-planning evidence."""
    required = {
        "sample_id",
        "event_id",
        "checkpoint_id",
        "checkpoint_role",
        "checkpoint_state_sha256",
        "split",
        "pfv_safe",
        "tfv_noninferior",
        "peak_noninferior",
        "joint_noninferior",
        "candidate_family",
        _DELTA_PFV,
        _DELTA_TFV,
        _DELTA_PEAK,
    }
    missing = required - set(samples)
    if missing:
        raise ValueError(f"v2 sample manifest missing: {sorted(missing)}")
    responsive = samples[
        samples["checkpoint_role"].astype(str) == "responsive"
    ]
    width = len(facility_ids)
    rows: list[dict] = []
    for (event_id, checkpoint_id), group in responsive.groupby(_STATE_KEYS):
        joint = _bool(group.get("joint_noninferior"), group.index)
        pfv_safe = _bool(group.get("pfv_safe"), group.index)
        tfv_ok = _bool(group.get("tfv_noninferior"), group.index)
        peak_ok = _bool(group.get("peak_noninferior"), group.index)
        if joint.any():
            reason = ""
        elif not pfv_safe.any():
            reason = "no_pfv_safe_candidate"
        elif not tfv_ok.any():
            reason = "tfv_always_degraded"
        elif not peak_ok.any():
            reason = "peak_always_degraded"
        else:
            reason = "labels_never_jointly_noninferior"
        moved_union = np.zeros(width, dtype=bool)
        for _, sample in group.iterrows():
            mask = _facility_moved_mask(sample)
            if mask.size == width:
                moved_union |= mask
        active_ids = [
            facility_ids[i] for i in range(width) if moved_union[i]
        ]
        best_pfv_id, best_pfv = _best_sample(group, _DELTA_PFV, pfv_safe)
        best_tfv_id, best_tfv = _best_sample(group, _DELTA_TFV)
        best_peak_id, best_peak = _best_sample(group, _DELTA_PEAK)
        joint_families = sorted(
            group.loc[joint, "candidate_family"].astype(str).unique()
        )
        split = str(group["split"].iloc[0])
        is_train = split == TRAIN_SPLIT
        response = pd.to_numeric(
            group.get("local_response_magnitude"), errors="coerce"
        )
        rows.append(
            {
                "event_id": str(event_id),
                "checkpoint_id": str(checkpoint_id),
                "state_id": str(group["checkpoint_state_sha256"].iloc[0]),
                "split": split,
                "checkpoint_min": float(
                    pd.to_numeric(
                        group.get("checkpoint_min"), errors="coerce"
                    ).iloc[0]
                )
                if "checkpoint_min" in group
                else float("nan"),
                "accepted_samples": int(len(group)),
                "joint_count": int(joint.sum()),
                "pfv_safe_count": int(pfv_safe.sum()),
                "tfv_noninferior_count": int(tfv_ok.sum()),
                "peak_noninferior_count": int(peak_ok.sum()),
                "positive_control_state": bool(joint.any()),
                "joint_missing_state": not bool(joint.any()),
                "dominant_failure_reason": reason,
                "joint_family_count": int(len(joint_families)),
                "joint_families_json": json.dumps(joint_families),
                "best_pfv_safe_sample_id": best_pfv_id,
                "best_pfv_safe_delta_pfv": best_pfv,
                "best_tfv_sample_id": best_tfv_id,
                "best_tfv_delta_tfv": best_tfv,
                "best_peak_sample_id": best_peak_id,
                "best_peak_delta_peak": best_peak,
                "best_delta_pfv": float(
                    pd.to_numeric(group[_DELTA_PFV], errors="coerce").min()
                ),
                "best_delta_tfv": float(
                    pd.to_numeric(group[_DELTA_TFV], errors="coerce").min()
                ),
                "best_delta_peak": float(
                    pd.to_numeric(group[_DELTA_PEAK], errors="coerce").min()
                ),
                "max_local_response_magnitude": (
                    float(response.max()) if response.notna().any() else 0.0
                ),
                "active_facility_count": int(moved_union.sum()),
                "active_actuator_coverage": (
                    float(moved_union.sum()) / width if width else 0.0
                ),
                "active_facility_ids_json": json.dumps(active_ids),
                "is_pilot_train": is_train,
                "search_result_training_eligible": is_train,
                "oracle_revealed_flag_required": not is_train,
                **{
                    f"ref_{branch}_path": (
                        f"{reference_root_rel}/{event_id}/{checkpoint_id}/"
                        f"{branch}/detail.csv"
                    )
                    for branch in REFERENCE_BRANCHES
                },
            }
        )
    return pd.DataFrame(rows).sort_values(_STATE_KEYS).reset_index(drop=True)


def audit_feasibility_state_catalog(
    catalog: pd.DataFrame,
    *,
    expected_states: int = 32,
    expected_positive_controls: int = 9,
    expected_joint_missing: int = 23,
) -> dict:
    """Fail-closed consistency audit of the P3 state catalog."""
    positive = int(catalog["positive_control_state"].sum())
    missing_joint = int(catalog["joint_missing_state"].sum())
    checks = {
        "state_count_matches": int(len(catalog)) == expected_states,
        "positive_control_count_matches": positive
        == expected_positive_controls,
        "joint_missing_count_matches": missing_joint
        == expected_joint_missing,
        "partition_is_exact": bool(
            (
                catalog["positive_control_state"]
                ^ catalog["joint_missing_state"]
            ).all()
        ),
        "state_ids_unique": bool(catalog["state_id"].is_unique),
        "state_ids_nonempty": bool(
            catalog["state_id"].astype(str).str.len().gt(0).all()
        ),
        "positive_controls_have_no_failure_reason": bool(
            (
                catalog.loc[
                    catalog["positive_control_state"],
                    "dominant_failure_reason",
                ].astype(str)
                == ""
            ).all()
        ),
        "missing_states_have_failure_reason": bool(
            (
                catalog.loc[
                    catalog["joint_missing_state"], "dominant_failure_reason"
                ]
                .astype(str)
                .str.len()
                .gt(0)
            ).all()
        ),
        "non_train_states_flagged_oracle_revealed": bool(
            (
                catalog["oracle_revealed_flag_required"]
                == ~catalog["is_pilot_train"]
            ).all()
        ),
        "reference_paths_present": all(
            f"ref_{branch}_path" in catalog for branch in REFERENCE_BRANCHES
        ),
    }
    failure_counts = (
        catalog.loc[
            catalog["joint_missing_state"], "dominant_failure_reason"
        ]
        .astype(str)
        .value_counts()
        .to_dict()
    )
    return {
        "checks": checks,
        "states_total": int(len(catalog)),
        "positive_control_states": positive,
        "joint_missing_states": missing_joint,
        "dominant_failure_counts": failure_counts,
        "events": int(catalog["event_id"].nunique()),
        "splits": sorted(catalog["split"].astype(str).unique()),
        "status": "pass" if all(checks.values()) else "blocked",
    }
