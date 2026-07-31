"""Same-state lexicographic ranking losses for V4.2 control selection.

Decision contract
-----------------
PFV is a safety constraint, not a continuous optimisation reward.  Candidate
ranking must therefore be lexicographic within the *same hydraulic state*:

1. PFV-safe beats PFV-unsafe.
2. Among PFV-safe candidates, Peak-safe beats Peak-unsafe.
3. Among candidates safe on both constraints, lower TFV is better.

Cross-event/cross-checkpoint pairs are never valid.  The old implementation
formed all-vs-all PFV pairs inside a minibatch, which mixed unrelated states
and encouraged the model to continuously minimise PFV even after the safety
constraint was satisfied.
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


class RankingLosses(nn.Module):
    """PFV-first, same-state pairwise ranking and safety losses."""

    def __init__(
        self,
        margin: float = 0.0,
        safety_margin: float = 0.0,
        regret_weight: float = 0.2,
    ):
        super().__init__()
        self.margin = float(margin)
        self.safety_margin = float(safety_margin)
        self.regret_weight = float(regret_weight)

    @staticmethod
    def _connected_zero(pred: dict[str, torch.Tensor]) -> torch.Tensor:
        return pred["pfv_delta"].sum() * 0.0

    @staticmethod
    def _pairwise_softplus(
        score: torch.Tensor,
        i: torch.Tensor,
        j: torch.Tensor,
        preference_i_over_j: torch.Tensor,
    ) -> torch.Tensor:
        """Smooth pairwise loss; larger score means better."""
        if i.numel() == 0:
            return score.sum() * 0.0
        diff = score[i] - score[j]
        return F.softplus(-preference_i_over_j * diff).mean()

    @staticmethod
    def _state_pairs(state_group: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return upper-triangle pairs restricted to equal state group IDs."""
        n = int(state_group.numel())
        if n < 2:
            empty = torch.empty(0, dtype=torch.long, device=state_group.device)
            return empty, empty
        ii, jj = torch.triu_indices(n, n, offset=1, device=state_group.device)
        same = state_group[ii] == state_group[jj]
        return ii[same], jj[same]

    def forward(
        self,
        pred: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        required_pred = ("pfv_delta", "tfv_delta", "peak_delta")
        missing_pred = [k for k in required_pred if k not in pred]
        if missing_pred:
            raise KeyError(f"RankingLosses missing predictions: {missing_pred}")

        zero = self._connected_zero(pred)
        device = pred["pfv_delta"].device
        losses: dict[str, torch.Tensor] = {}

        # All ranking pairs must belong to one event/checkpoint state.
        state_group = target.get("state_group_index")
        if state_group is None:
            losses["pairwise_ranking"] = zero
            losses["valid_pair_count"] = zero
            losses["pfv_safety_hinge"] = zero
            losses["peak_safety_hinge"] = zero
            losses["decision_regret"] = zero
            return losses
        state_group = state_group.to(device=device, dtype=torch.long)
        i, j = self._state_pairs(state_group)

        pfv_safe = target.get("pfv_safe_label")
        peak_safe = target.get("peak_noninferior_label")
        if pfv_safe is None or peak_safe is None:
            raise KeyError(
                "RankingLosses requires pfv_safe_label and peak_noninferior_label "
                "for PFV-first lexicographic ranking"
            )
        pfv_safe = pfv_safe.to(device=device) > 0.5
        peak_safe = peak_safe.to(device=device) > 0.5

        pair_losses: list[torch.Tensor] = []
        valid_pair_count = 0

        if i.numel() > 0:
            # Tier 1 — PFV safety dominates every lower-priority objective.
            pfv_diff = pfv_safe[i] != pfv_safe[j]
            if pfv_diff.any():
                pi, pj = i[pfv_diff], j[pfv_diff]
                preference = torch.where(
                    pfv_safe[pi],
                    torch.ones_like(pi, dtype=pred["pfv_delta"].dtype),
                    -torch.ones_like(pi, dtype=pred["pfv_delta"].dtype),
                )
                score = -pred["pfv_delta"]
                pair_losses.append(self._pairwise_softplus(score, pi, pj, preference))
                valid_pair_count += int(pi.numel())

            # Tier 2 — only when both candidates are PFV-safe.
            both_pfv_safe = pfv_safe[i] & pfv_safe[j]
            peak_diff = both_pfv_safe & (peak_safe[i] != peak_safe[j])
            if peak_diff.any():
                pi, pj = i[peak_diff], j[peak_diff]
                preference = torch.where(
                    peak_safe[pi],
                    torch.ones_like(pi, dtype=pred["peak_delta"].dtype),
                    -torch.ones_like(pi, dtype=pred["peak_delta"].dtype),
                )
                score = -pred["peak_delta"]
                pair_losses.append(self._pairwise_softplus(score, pi, pj, preference))
                valid_pair_count += int(pi.numel())

            # Tier 3 — TFV only inside the true safe set.
            both_safe = both_pfv_safe & peak_safe[i] & peak_safe[j]
            if "tfv_delta" in target:
                tfv_true = target["tfv_delta"].to(device=device)
                tfv_gap = (tfv_true[i] - tfv_true[j]).abs()
                tfv_valid = both_safe & (tfv_gap > self.margin)
                if tfv_valid.any():
                    pi, pj = i[tfv_valid], j[tfv_valid]
                    # Lower true TFV delta is better.
                    preference = torch.sign(tfv_true[pj] - tfv_true[pi]).to(
                        dtype=pred["tfv_delta"].dtype
                    )
                    score = -pred["tfv_delta"]
                    pair_losses.append(self._pairwise_softplus(score, pi, pj, preference))
                    valid_pair_count += int(pi.numel())

        losses["pairwise_ranking"] = (
            torch.stack(pair_losses).mean() if pair_losses else zero
        )
        # Logged as a tensor so existing training-history code can serialize it;
        # it is detached from optimisation by multiplying the graph-connected
        # zero and adding the scalar count.
        losses["valid_pair_count"] = zero + float(valid_pair_count)

        # One-sided safety losses.  KPI targets are z-score normalised during
        # training, so the raw safety boundary must be transformed per fold and
        # supplied by the trainer.
        pfv_boundary = target.get("pfv_boundary_norm")
        peak_boundary = target.get("peak_boundary_norm")
        if pfv_boundary is not None:
            losses["pfv_safety_hinge"] = F.softplus(
                pred["pfv_delta"] - pfv_boundary.to(device) + self.safety_margin
            ).mean()
        else:
            losses["pfv_safety_hinge"] = zero

        if peak_boundary is not None:
            losses["peak_safety_hinge"] = F.softplus(
                pred["peak_delta"] - peak_boundary.to(device) + self.safety_margin
            ).mean()
        else:
            losses["peak_safety_hinge"] = zero

        # Decision-regret surrogate is restricted to the true safe set and
        # penalises TFV ordering errors only.  Safety violations are accounted
        # for by the two safety losses above and must never be traded for TFV.
        if i.numel() > 0 and "tfv_delta" in target:
            tfv_true = target["tfv_delta"].to(device)
            safe_pair = pfv_safe[i] & pfv_safe[j] & peak_safe[i] & peak_safe[j]
            if safe_pair.any():
                pi, pj = i[safe_pair], j[safe_pair]
                truth_pref = torch.sign(tfv_true[pj] - tfv_true[pi])
                pred_gap = pred["tfv_delta"][pj] - pred["tfv_delta"][pi]
                losses["decision_regret"] = (
                    F.softplus(-truth_pref * pred_gap).mean() * self.regret_weight
                )
            else:
                losses["decision_regret"] = zero
        else:
            losses["decision_regret"] = zero

        return losses

    def total(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        """Sum differentiable ranking losses, excluding the diagnostic count."""
        return sum(v for k, v in losses.items() if k != "valid_pair_count")
