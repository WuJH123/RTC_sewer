"""PFV-relaxation versus TFV-benefit trade-off analysis for V4.2.

All functions operate on authoritative candidate outcomes.  They do not alter
online safety thresholds.  The purpose is to quantify how much extra TFV
benefit would become available if the PFV non-inferiority contract were relaxed
and to expose state-wise Pareto-efficient actions.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


TRADEOFF_CONTRACT = "V42_PFV_TFV_EXCHANGE_PARETO_V1"


@dataclass(frozen=True)
class PfvContract:
    relative_margin_fraction: float
    absolute_margin_m3: float

    def admits(self, pfv_candidate_m3, pfv_no_control_m3):
        candidate = np.asarray(pfv_candidate_m3, dtype=float)
        reference = np.asarray(pfv_no_control_m3, dtype=float)
        metric = candidate - (1.0 + float(self.relative_margin_fraction)) * reference
        return metric <= float(self.absolute_margin_m3) + 1.0e-9


def add_tradeoff_columns(frame: pd.DataFrame) -> pd.DataFrame:
    required = {
        "state_key",
        "candidate_action_sha256",
        "pfv_candidate_m3",
        "pfv_no_control_m3",
        "tfv_candidate_m3",
        "tfv_internal_m3",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"tradeoff table missing columns: {missing}")
    work = frame.copy()
    for column in required - {"state_key", "candidate_action_sha256"}:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    work = work[np.isfinite(work[["pfv_candidate_m3", "pfv_no_control_m3", "tfv_candidate_m3", "tfv_internal_m3"]]).all(axis=1)].copy()
    work["pfv_excess_vs_no_control_m3"] = work["pfv_candidate_m3"] - work["pfv_no_control_m3"]
    work["pfv_relative_excess_fraction"] = np.where(
        work["pfv_no_control_m3"].abs() > 1.0e-9,
        work["pfv_excess_vs_no_control_m3"] / work["pfv_no_control_m3"],
        np.nan,
    )
    work["tfv_benefit_m3"] = work["tfv_internal_m3"] - work["tfv_candidate_m3"]
    work["tfv_reduction_pct"] = np.where(
        work["tfv_internal_m3"].abs() > 1.0e-9,
        100.0 * work["tfv_benefit_m3"] / work["tfv_internal_m3"],
        np.nan,
    )
    return work


def state_pareto_frontier(frame: pd.DataFrame) -> pd.DataFrame:
    """Return actions not dominated in PFV cost and TFV benefit for each state.

    Lower PFV excess is better; higher TFV benefit is better.  Negative PFV
    excess is retained because it means the candidate also improves PFV.
    """
    work = add_tradeoff_columns(frame)
    output = []
    for state_key, group in work.groupby(work["state_key"].astype(str), sort=False):
        ordered = group.sort_values(
            ["pfv_excess_vs_no_control_m3", "tfv_benefit_m3", "candidate_action_sha256"],
            ascending=[True, False, True],
            kind="stable",
        )
        best_benefit = -np.inf
        for row in ordered.itertuples(index=False):
            benefit = float(row.tfv_benefit_m3)
            if benefit > best_benefit + 1.0e-9:
                payload = row._asdict()
                payload["state_key"] = str(state_key)
                payload["pareto_rank"] = len(output)
                output.append(payload)
                best_benefit = benefit
    return pd.DataFrame(output)


def pareto_exchange_rates(frontier: pd.DataFrame) -> pd.DataFrame:
    """Compute incremental TFV benefit gained per extra m3 of PFV cost."""
    if frontier.empty:
        return frontier.copy()
    rows = []
    for state_key, group in frontier.groupby(frontier["state_key"].astype(str), sort=False):
        ordered = group.sort_values("pfv_excess_vs_no_control_m3", kind="stable").reset_index(drop=True)
        previous = None
        for item in ordered.to_dict("records"):
            current = dict(item)
            if previous is None:
                current["delta_pfv_cost_m3"] = np.nan
                current["delta_tfv_benefit_m3"] = np.nan
                current["marginal_tfv_benefit_per_pfv_m3"] = np.nan
            else:
                dx = float(current["pfv_excess_vs_no_control_m3"] - previous["pfv_excess_vs_no_control_m3"])
                dy = float(current["tfv_benefit_m3"] - previous["tfv_benefit_m3"])
                current["delta_pfv_cost_m3"] = dx
                current["delta_tfv_benefit_m3"] = dy
                current["marginal_tfv_benefit_per_pfv_m3"] = dy / dx if dx > 1.0e-9 else np.nan
            rows.append(current)
            previous = current
    return pd.DataFrame(rows)


def contract_scan(
    frame: pd.DataFrame,
    *,
    relative_margins: Sequence[float] = (0.0, 0.025, 0.05, 0.075, 0.10, 0.15),
    absolute_margins_m3: Sequence[float] = (0.0, 100.0, 250.0, 500.0, 1000.0, 2000.0),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluate best authoritative TFV under a grid of PFV contracts.

    Returns (state_contract_rows, aggregate_rows).  Aggregate output reports
    both admitted-only summaries and all-state summaries where an unavailable
    safe action contributes zero controllable benefit, avoiding the misleading
    conditional-median shift seen in earlier audits.
    """
    work = add_tradeoff_columns(frame)
    state_keys = sorted(work["state_key"].astype(str).unique())
    state_rows = []
    aggregate_rows = []
    for relative in relative_margins:
        for absolute in absolute_margins_m3:
            contract = PfvContract(float(relative), float(absolute))
            admitted = contract.admits(work["pfv_candidate_m3"], work["pfv_no_control_m3"])
            subset = work[admitted].copy()
            best_by_state: dict[str, dict] = {}
            for state_key, group in subset.groupby(subset["state_key"].astype(str), sort=False):
                best = group.sort_values(["tfv_candidate_m3", "candidate_action_sha256"], kind="stable").iloc[0]
                payload = best.to_dict()
                payload.update(
                    {
                        "state_key": str(state_key),
                        "relative_margin_fraction": float(relative),
                        "absolute_margin_m3": float(absolute),
                        "contract_budget_metric_m3": float(
                            best["pfv_candidate_m3"] - (1.0 + float(relative)) * best["pfv_no_control_m3"]
                        ),
                    }
                )
                best_by_state[str(state_key)] = payload
                state_rows.append(payload)
            all_benefits_pct = []
            admitted_values = []
            improving = 0
            for state_key in state_keys:
                row = best_by_state.get(state_key)
                if row is None:
                    all_benefits_pct.append(0.0)
                    continue
                value = float(row["tfv_reduction_pct"])
                admitted_values.append(value)
                all_benefits_pct.append(max(value, 0.0))
                improving += int(value > 0.0)
            aggregate_rows.append(
                {
                    "relative_margin_fraction": float(relative),
                    "absolute_margin_m3": float(absolute),
                    "total_states": len(state_keys),
                    "admitted_states": len(best_by_state),
                    "admitted_fraction": len(best_by_state) / max(len(state_keys), 1),
                    "improving_states": improving,
                    "improving_fraction_all_states": improving / max(len(state_keys), 1),
                    "admitted_only_median_tfv_reduction_pct": float(np.median(admitted_values)) if admitted_values else np.nan,
                    "admitted_only_mean_tfv_reduction_pct": float(np.mean(admitted_values)) if admitted_values else np.nan,
                    "all_state_zero_if_unavailable_median_pct": float(np.median(all_benefits_pct)) if all_benefits_pct else np.nan,
                    "all_state_zero_if_unavailable_mean_pct": float(np.mean(all_benefits_pct)) if all_benefits_pct else np.nan,
                    "states_ge_5pct": int(sum(value >= 5.0 for value in all_benefits_pct)),
                    "states_ge_10pct": int(sum(value >= 10.0 for value in all_benefits_pct)),
                    "states_ge_20pct": int(sum(value >= 20.0 for value in all_benefits_pct)),
                }
            )
    return pd.DataFrame(state_rows), pd.DataFrame(aggregate_rows)


def select_knee_points(frontier: pd.DataFrame) -> pd.DataFrame:
    """Select one descriptive knee per state using normalized chord distance."""
    if frontier.empty:
        return frontier.copy()
    result = []
    for state_key, group in frontier.groupby(frontier["state_key"].astype(str), sort=False):
        ordered = group.sort_values("pfv_excess_vs_no_control_m3", kind="stable").copy()
        if len(ordered) <= 2:
            chosen = ordered.iloc[-1]
        else:
            x = ordered["pfv_excess_vs_no_control_m3"].to_numpy(float)
            y = ordered["tfv_benefit_m3"].to_numpy(float)
            xn = (x - x.min()) / max(x.max() - x.min(), 1.0e-9)
            yn = (y - y.min()) / max(y.max() - y.min(), 1.0e-9)
            # Distance above the straight line connecting the two extremes.
            x0, y0 = xn[0], yn[0]
            x1, y1 = xn[-1], yn[-1]
            denom = max(np.hypot(y1 - y0, x1 - x0), 1.0e-9)
            distance = np.abs((y1 - y0) * xn - (x1 - x0) * yn + x1 * y0 - y1 * x0) / denom
            chosen = ordered.iloc[int(np.argmax(distance))]
        payload = chosen.to_dict()
        payload["state_key"] = str(state_key)
        payload["knee_method"] = "normalized_chord_distance"
        result.append(payload)
    return pd.DataFrame(result)
