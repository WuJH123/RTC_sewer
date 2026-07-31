from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class EventPfvBudget:
    """Cumulative event-level PFV noninferiority budget.

    The budget is initialized once per rainfall event from the predicted
    same-state No-control PFV. Online selection then spends it with observed
    realized PFV increments plus conservative costs for actions whose effects
    have not fully cleared the hydraulic horizon.
    """

    predicted_event_no_control_pfv: float
    abs_margin_m3: float = 200.0
    rel_margin: float = 0.02
    spent_observed_m3: float = 0.0
    spent_inflight_m3: float = 0.0
    ledger: list[dict[str, float | str]] = field(default_factory=list)

    @property
    def initial_budget_m3(self) -> float:
        return max(
            float(self.abs_margin_m3),
            max(0.0, float(self.predicted_event_no_control_pfv)) * float(self.rel_margin),
        )

    @property
    def remaining_budget_m3(self) -> float:
        return float(self.initial_budget_m3) - float(self.spent_observed_m3) - float(self.spent_inflight_m3)

    def debit_observed(self, delta_pfv_m3: float, *, label: str = "observed") -> None:
        spend = max(0.0, float(delta_pfv_m3))
        self.spent_observed_m3 += spend
        self.ledger.append({"kind": "observed", "label": str(label), "debit_m3": spend})

    def set_inflight_conservative_cost(self, cost_m3: float, *, label: str = "inflight") -> None:
        spend = max(0.0, float(cost_m3))
        self.spent_inflight_m3 = spend
        self.ledger.append({"kind": "inflight", "label": str(label), "debit_m3": spend})

    def candidate_allowed(self, future_pfv_increment_ucb_m3: float) -> bool:
        return float(future_pfv_increment_ucb_m3) <= float(self.remaining_budget_m3)

    def audit_row(self) -> dict[str, float]:
        return {
            "predicted_event_no_control_pfv": float(self.predicted_event_no_control_pfv),
            "initial_budget_m3": float(self.initial_budget_m3),
            "spent_observed_m3": float(self.spent_observed_m3),
            "spent_inflight_m3": float(self.spent_inflight_m3),
            "remaining_budget_m3": float(self.remaining_budget_m3),
        }
