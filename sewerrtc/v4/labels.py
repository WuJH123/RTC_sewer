from __future__ import annotations

import numpy as np
import pandas as pd

from sewerrtc.simulation.kpi_metrics import compute_kpis


def select_h120_window(
    detail: pd.DataFrame, checkpoint_min: float, horizon_min: float = 120.0
) -> pd.DataFrame:
    elapsed = pd.to_numeric(detail["elapsed_min"], errors="coerce")
    return detail[
        (elapsed > float(checkpoint_min))
        & (elapsed <= float(checkpoint_min) + float(horizon_min))
    ].copy()


def window_kpis(
    detail: pd.DataFrame,
    priority_nodes: list[str],
    checkpoint_min: float,
    *,
    dt_sec: int = 300,
) -> dict:
    window = select_h120_window(detail, checkpoint_min)
    result = compute_kpis(window, priority_nodes, dt_sec=dt_sec)
    result["steps"] = int(len(window))
    return result


def classify_labels(
    delta_pfv: float,
    delta_tfv: float,
    delta_peak: float,
    *,
    scientific_margin: dict[str, float],
    dead_zone: dict[str, float],
    action_cost: float,
    minimum_tfv_benefit: float = 25.0,
    minimum_ratio: float = 1.5,
) -> dict:
    pfv_safe = delta_pfv <= scientific_margin["pfv_m3"]
    tfv_noninferior = delta_tfv <= scientific_margin["tfv_m3"]
    peak_noninferior = delta_peak <= scientific_margin["peak_m3s"]
    joint = pfv_safe and tfv_noninferior and peak_noninferior
    benefit = max(0.0, -float(delta_tfv))
    ratio = benefit / max(float(action_cost), 1e-12)
    neutral = (
        abs(delta_pfv) <= dead_zone["pfv_m3"]
        and abs(delta_tfv) <= dead_zone["tfv_m3"]
        and abs(delta_peak) <= dead_zone["peak_m3s"]
    )
    hard_negative = ""
    if delta_tfv < -dead_zone["tfv_m3"] and not pfv_safe:
        hard_negative = "PFV_hard_negative"
    elif pfv_safe and not peak_noninferior:
        hard_negative = "Peak_hard_negative"
    elif peak_noninferior and not tfv_noninferior:
        hard_negative = "TFV_hard_negative"
    return {
        "pfv_safe": pfv_safe,
        "tfv_improved": delta_tfv < -dead_zone["tfv_m3"],
        "peak_noninferior": peak_noninferior,
        "joint_noninferior": joint,
        "materially_beneficial": (
            joint and benefit >= minimum_tfv_benefit and ratio >= minimum_ratio
        ),
        "neutral": neutral,
        "hard_negative_type": hard_negative,
        "benefit_cost_ratio": ratio,
    }


def add_ranking_labels(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    group_keys = ["event_id", "checkpoint_id"]
    result["feasible_rank"] = np.nan
    result["regret_to_exact_best"] = np.nan
    result["pairwise_preference"] = ""
    for _, group in result.groupby(group_keys):
        feasible = group[group["joint_noninferior"].astype(bool)]
        if feasible.empty:
            continue
        ordered = feasible.sort_values("delta_tfv_h120_vs_dynamic_internal")
        best = float(ordered["delta_tfv_h120_vs_dynamic_internal"].iloc[0])
        result.loc[ordered.index, "feasible_rank"] = np.arange(1, len(ordered) + 1)
        result.loc[ordered.index, "regret_to_exact_best"] = (
            ordered["delta_tfv_h120_vs_dynamic_internal"] - best
        )
        result.loc[ordered.index, "pairwise_preference"] = "preferred_over_infeasible"
    return result


def enforce_full_label_eligibility(frame: pd.DataFrame) -> pd.DataFrame:
    if "full_event_eligible" not in frame:
        raise ValueError("full_event_eligible is required")
    result = frame.copy()
    full_columns = [
        column
        for column in result
        if column.startswith("delta_") and "_full" in column
    ]
    result.loc[~result["full_event_eligible"].astype(bool), full_columns] = np.nan
    return result
