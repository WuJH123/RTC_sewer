from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

SURROGATE_ABLATIONS = {
    "A": ("true_state", "exact_evaluation"),
    "B": ("true_state", "v4_surrogate"),
    "C": ("gat_state", "exact_evaluation"),
    "D": ("gat_state", "v4_surrogate"),
}


def audit_closed_loop_order(exact_exit_code: int) -> dict:
    return {
        "status": "pass" if int(exact_exit_code) == 0 else "blocked",
        "exact_closed_loop_exit_code": int(exact_exit_code),
    }


def timing_budget_ok(total_time_sec: float, budget_sec: float = 600.0) -> bool:
    return float(total_time_sec) < float(budget_sec)


@dataclass
class CumulativeSafetyBudget:
    initial_margin_m3: float

    def __post_init__(self) -> None:
        self.remaining_m3 = float(self.initial_margin_m3)

    def consume(self, delta_pfv_m3: float) -> bool:
        cost = max(0.0, float(delta_pfv_m3))
        if cost > self.remaining_m3:
            return False
        self.remaining_m3 -= cost
        return True


def select_pfv_first_candidate(
    candidates: pd.DataFrame,
    *,
    pfv_margin: float = 0.0,
    peak_margin: float = 0.0,
    terminal_margin: float = 0.0,
    action_weight: float = 1.0,
    switching_weight: float = 1.0,
) -> dict:
    required = {
        "candidate_id",
        "delta_pfv",
        "delta_peak",
        "terminal_risk",
        "delta_tfv",
        "action_cost",
        "switching_cost",
    }
    missing = required - set(candidates)
    if missing:
        raise ValueError(f"candidate table missing: {sorted(missing)}")
    def gate_column(name: str) -> pd.Series:
        """Use a calibrated upper bound when supplied by the evaluator."""
        ucb = f"{name}_ucb"
        return candidates[ucb] if ucb in candidates else candidates[name]

    safe = candidates[
        (gate_column("delta_pfv") <= float(pfv_margin))
        & (gate_column("delta_peak") <= float(peak_margin))
        & (gate_column("delta_tfv") <= 0.0)
        & (candidates["terminal_risk"] <= float(terminal_margin))
    ].copy()
    if safe.empty:
        return {"candidate_id": "fallback", "fallback": True}
    safe["_objective"] = (
        safe["delta_tfv"]
        + float(action_weight) * safe["action_cost"]
        + float(switching_weight) * safe["switching_cost"]
    )
    selected = safe.sort_values(
        ["_objective", "candidate_id"]
    ).iloc[0].to_dict()
    selected["fallback"] = False
    return selected
