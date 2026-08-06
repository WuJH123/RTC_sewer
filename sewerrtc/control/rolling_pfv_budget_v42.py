"""Cumulative PFV non-inferiority accounting for rolling V4.2 MPC.

The event-level contract is

    PFV_candidate_total <= (1 + delta) * PFV_no_control_total + B.

A receding-horizon controller must therefore combine the *realised causal
prefix* with the UCB of the candidate future.  Re-applying the full allowance
at every 10-minute decision is not equivalent to the event-level contract.

This module is deliberately independent of SWMM and of the Step2 model.  The
plant runtime owns the authoritative interval increments and updates this
state; the selector only consumes the composed cumulative budget metric.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class RollingPfvBudgetState:
    """Causal prefix state for one event-level PFV non-inferiority budget."""

    relative_margin_fraction: float = 0.05
    absolute_margin_m3: float = 100.0
    realised_candidate_pfv_m3: float = 0.0
    realised_no_control_pfv_m3: float = 0.0
    update_count: int = 0

    def __post_init__(self) -> None:
        values = (
            self.relative_margin_fraction,
            self.absolute_margin_m3,
            self.realised_candidate_pfv_m3,
            self.realised_no_control_pfv_m3,
        )
        if not all(np.isfinite(float(value)) for value in values):
            raise ValueError("rolling PFV budget values must be finite")
        if self.relative_margin_fraction < 0.0:
            raise ValueError("relative PFV margin must be non-negative")
        if self.absolute_margin_m3 < 0.0:
            raise ValueError("absolute PFV margin must be non-negative")
        if self.realised_candidate_pfv_m3 < 0.0 or self.realised_no_control_pfv_m3 < 0.0:
            raise ValueError("realised PFV prefixes must be non-negative")
        if int(self.update_count) < 0:
            raise ValueError("update_count must be non-negative")

    @property
    def realised_prefix_budget_metric_m3(self) -> float:
        """Return candidate_prefix - (1 + delta) * no_control_prefix."""
        return float(
            self.realised_candidate_pfv_m3
            - (1.0 + self.relative_margin_fraction)
            * self.realised_no_control_pfv_m3
        )

    @property
    def remaining_future_allowance_m3(self) -> float:
        """Allowance available to the future budget metric at this decision."""
        return float(self.absolute_margin_m3 - self.realised_prefix_budget_metric_m3)

    def update(
        self,
        *,
        candidate_interval_pfv_m3: float,
        no_control_interval_pfv_m3: float,
    ) -> "RollingPfvBudgetState":
        """Append one authoritative, already-realised control interval."""
        candidate = float(candidate_interval_pfv_m3)
        no_control = float(no_control_interval_pfv_m3)
        if not np.isfinite(candidate) or not np.isfinite(no_control):
            raise ValueError("PFV interval increments must be finite")
        if candidate < -1.0e-9 or no_control < -1.0e-9:
            raise ValueError("PFV interval increments must be non-negative")
        return replace(
            self,
            realised_candidate_pfv_m3=self.realised_candidate_pfv_m3 + max(candidate, 0.0),
            realised_no_control_pfv_m3=self.realised_no_control_pfv_m3 + max(no_control, 0.0),
            update_count=self.update_count + 1,
        )

    def cumulative_budget_metric_ucb_m3(
        self,
        future_budget_metric_ucb_m3: float | np.ndarray | Iterable[float],
    ) -> np.ndarray:
        """Compose realised prefix and future UCB on the same event metric."""
        future = np.asarray(future_budget_metric_ucb_m3, dtype=float)
        if not np.all(np.isfinite(future)):
            raise ValueError("future PFV budget UCB must be finite")
        return future + self.realised_prefix_budget_metric_m3

    def admits(
        self,
        future_budget_metric_ucb_m3: float | np.ndarray | Iterable[float],
    ) -> np.ndarray:
        """Return event-level admission decisions without resetting allowance."""
        cumulative = self.cumulative_budget_metric_ucb_m3(
            future_budget_metric_ucb_m3
        )
        return cumulative <= self.absolute_margin_m3 + 1.0e-9

    def audit_payload(self) -> dict[str, float | int | str | bool]:
        return {
            "contract": "event_level_candidate_le_(1+delta)_no_control_plus_B",
            "relative_margin_fraction": float(self.relative_margin_fraction),
            "absolute_margin_m3": float(self.absolute_margin_m3),
            "realised_candidate_pfv_m3": float(self.realised_candidate_pfv_m3),
            "realised_no_control_pfv_m3": float(self.realised_no_control_pfv_m3),
            "realised_prefix_budget_metric_m3": float(
                self.realised_prefix_budget_metric_m3
            ),
            "remaining_future_allowance_m3": float(
                self.remaining_future_allowance_m3
            ),
            "update_count": int(self.update_count),
            "allowance_reinitialised_each_decision": False,
            "realised_prefix_included": True,
        }
