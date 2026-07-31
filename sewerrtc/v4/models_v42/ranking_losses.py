"""Ranking losses: pairwise ranking and safety hinge losses.

Includes:
  - Same-state pairwise ranking: better candidate → lower predicted KPI
  - PFV safety hinge: penalize when PFV_candidate > PFV_no_control
  - Peak safety hinge: similar
  - Decision regret loss
"""
from __future__ import annotations

import torch
from torch import nn


class RankingLosses(nn.Module):
    """Pairwise ranking and safety hinge losses.

    These losses enforce ordinal consistency: if action A is better than
    action B in simulation, the model should predict A having lower KPI.
    Safety hinges penalize predictions that exceed no-control baselines.
    """

    def __init__(
        self,
        margin: float = 0.1,
        safety_margin: float = 0.0,
        regret_weight: float = 0.2,
    ):
        super().__init__()
        self.margin = float(margin)
        self.safety_margin = float(safety_margin)
        self.regret_weight = float(regret_weight)

    def forward(
        self,
        pred: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        pred dict:
            pfv_delta : [B]
            tfv_delta : [B]
            peak_flood_rate : [B]

        target dict:
            pfv_gt       : [B]
            tfv_gt       : [B]
            peak_gt      : [B]
            pfv_no_ctrl  : [B]  (optional: no-control baseline PFV)
            peak_no_ctrl : [B]  (optional: no-control baseline peak)
            pair_mask    : [B]  (optional: 1 if this sample has a valid pair)
            pair_better  : [B]  (optional: 1 if candidate is better than pair)
        """
        losses: dict[str, torch.Tensor] = {}
        device = pred["pfv_delta"].device

        # 1. Same-state pairwise ranking
        #    Convention: pfv_delta = Candidate − Reference; negative = improvement.
        #    Define score = -pfv_delta so that "larger is better".  For a pair where
        #    sample A is truly better than B we want score_A > score_B.  Using
        #    softplus instead of hinge gives non-zero gradient even when the ranking
        #    is already correct, which prevents the loss from going flat during
        #    training (the old hinge had dloss/dscore=0 as soon as the margin was
        #    satisfied — this was the root cause of the "correct-sort loss ≈ 1.3,
        #    wrong-sort loss = 0" pathology flagged in the P0 audit).
        #
        #    Two modes:
        #      (a) explicit pair labels (pair_mask, pair_better) — use them;
        #      (b) otherwise — build all-vs-all pairs from the batch using the
        #          ground-truth pfv_delta to determine preference.  Pairs whose
        #          |gt delta| < dead_zone are ignored (neither sample is clearly
        #          better).  If no valid pairs exist we return 0 but the caller
        #          (AuditV42RankingPairs) is expected to fail-closed separately.
        if "pair_mask" in target and "pair_better" in target:
            mask = target["pair_mask"].bool()
            if mask.any():
                better = target["pair_better"][mask]  # [K]
                pfv_pred = pred["pfv_delta"][mask]
                score = -pfv_pred  # larger = better
                score_ref = score.mean()
                preference = 2.0 * better - 1.0  # +1 for better, -1 for worse
                ranking_loss = torch.nn.functional.softplus(
                    -preference * (score - score_ref)
                ).mean()
                losses["pairwise_ranking"] = ranking_loss
            else:
                losses["pairwise_ranking"] = torch.zeros((), device=device)
        elif "pfv_delta" in target and target["pfv_delta"].numel() >= 2:
            gt = target["pfv_delta"]
            score_pred = -pred["pfv_delta"]  # [B] larger = better
            # All-vs-all pairwise differences (vectorised).
            d_gt = gt.unsqueeze(0) - gt.unsqueeze(1)  # [B, B]  gt_i - gt_j
            d_pred = score_pred.unsqueeze(0) - score_pred.unsqueeze(1)  # [B, B]
            # Lower gt = better, so preference_ij = sign(gt_j - gt_i).
            preference = torch.sign(-d_gt)
            # Dead zone: ignore pairs whose |gt diff| is below margin.
            valid = d_gt.abs() > self.margin
            valid = valid & ~torch.eye(
                gt.numel(), dtype=torch.bool, device=device
            )
            if valid.any():
                ranking_loss = torch.nn.functional.softplus(
                    -preference[valid] * d_pred[valid]
                ).mean()
                losses["pairwise_ranking"] = ranking_loss
            else:
                losses["pairwise_ranking"] = torch.zeros((), device=device)
        else:
            losses["pairwise_ranking"] = torch.zeros((), device=device)

        # 2. PFV safety hinge: penalize when PFV_candidate > PFV_no_control
        if "pfv_no_ctrl" in target:
            # We want pfv_delta (candidate - reference) <= pfv_no_ctrl
            # i.e., the improvement should not exceed no-control baseline
            pfv_violation = pred["pfv_delta"] - target["pfv_no_ctrl"] + self.safety_margin
            losses["pfv_safety_hinge"] = torch.relu(pfv_violation).mean()
        else:
            losses["pfv_safety_hinge"] = torch.zeros((), device=device)

        # 3. Peak safety hinge
        if "peak_no_ctrl" in target:
            peak_violation = pred["peak_flood_rate"] - target["peak_no_ctrl"] + self.safety_margin
            losses["peak_safety_hinge"] = torch.relu(peak_violation).mean()
        else:
            losses["peak_safety_hinge"] = torch.zeros((), device=device)

        # 4. Decision regret loss
        #    If the model predicts improvement but simulation shows degradation
        #    Data key is "pfv_delta" (Candidate − Reference convention)
        if "pfv_delta" in target:
            # Predicted direction: sign of -pfv_delta (negative = improvement)
            pred_improve = -pred["pfv_delta"]
            # Actual direction: sign of -pfv_delta label
            actual_improve = -target["pfv_delta"]
            # Regret: when pred says improve but actual is worse
            regret = torch.relu(-pred_improve * torch.sign(actual_improve))
            losses["decision_regret"] = regret.mean() * self.regret_weight
        else:
            losses["decision_regret"] = torch.zeros((), device=device)

        return losses

    def total(self, losses: dict[str, torch.Tensor]) -> torch.Tensor:
        """Sum of all ranking losses."""
        return sum(losses.values())
