"""Experience-guided differentiable candidate search for V4.2 MPC.

This module proposes candidates only.  It never authorises an action.  Final
admission remains the frozen rolling PFV-UCB gate and final selection remains
minimum predicted TFV inside that admitted set.

Search structure
----------------
1. warm starts from authoritative development experience + current action;
2. state-adaptive active-facility selection from Step2 action gradients;
3. coarse H3-constant optimisation;
4. bounded H3 temporal refinement of the best few starts;
5. exact binary semantics and [0,1] projection;
6. return a small candidate population to the existing selector.
"""
from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Any, Iterable, Sequence

import numpy as np

from sewerrtc.control.authoritative_control_metrics_v42 import action_sha256
from sewerrtc.control.rolling_pfv_budget_v42 import RollingPfvBudgetState


HYBRID_SEARCH_CONTRACT = "V42_EXPERIENCE_GUIDED_DIFFERENTIABLE_SEARCH_V1"
BINARY_IDS = {"ADD301.2", "ADD301.3"}


@dataclass(frozen=True)
class DifferentiableSearchConfig:
    max_warm_starts: int = 8
    active_continuous_assets: int = 8
    temporal_refine_assets: int = 4
    spatial_steps: int = 24
    temporal_steps: int = 16
    learning_rate: float = 0.04
    barrier_weight: float = 25.0
    trust_region_weight: float = 0.05
    max_h3_change_from_seed: float = 0.30
    temporal_refine_top_k: int = 4
    max_return_candidates: int = 32
    include_all_binary_modes_for_current: bool = True


def _calibrated_budget_ucb(mean, std, calibration: dict[str, Any]):
    """Torch-compatible frozen one-sided PFV UCB."""
    import torch

    method = str(calibration.get("pfv_budget_metric_ucb_method", ""))
    margin = calibration.get("pfv_budget_metric_residual_margin_m3")
    try:
        margin_value = float(margin)
    except (TypeError, ValueError):
        margin_value = float("nan")
    if method == "absolute_residual_one_sided_conformal" and np.isfinite(margin_value) and margin_value >= 0.0:
        return mean + torch.as_tensor(margin_value, dtype=mean.dtype, device=mean.device)
    z = float(calibration.get("confidence_z", np.nan))
    if not np.isfinite(z) or z < 0.0:
        raise RuntimeError("Step2 PFV calibration is missing a finite one-sided margin")
    return mean + float(z) * std


def _tensor(value, *, device):
    import torch

    return torch.as_tensor(np.asarray(value, dtype=np.float32), device=device)


def _shared(value, batch: int, *, device):
    base = _tensor(value, device=device)
    return base.unsqueeze(0).expand(int(batch), *base.shape)


def _ensemble_objective(
    *,
    bundle: Any,
    candidate,
    state_history: np.ndarray,
    historical_actions: np.ndarray,
    rainfall_forecast: np.ndarray,
    current_action: np.ndarray,
    internal_current_action: np.ndarray,
    rolling_pfv_budget_state: RollingPfvBudgetState | None,
):
    """Return differentiable TFV objective and PFV UCB for a candidate batch."""
    import torch

    n = int(candidate.shape[0])
    device = candidate.device
    horizon = int(candidate.shape[1])
    facilities = int(candidate.shape[2])
    no_control = torch.ones((n, horizon, facilities), dtype=candidate.dtype, device=device)
    internal = _shared(
        np.repeat(np.asarray(internal_current_action, np.float32)[None, :], horizon, axis=0),
        n,
        device=device,
    )
    hold = _shared(
        np.repeat(np.asarray(current_action, np.float32)[None, :], horizon, axis=0),
        n,
        device=device,
    )
    state = _shared(np.asarray(state_history, np.float32), n, device=device)
    history_action = _shared(np.asarray(historical_actions, np.float32), n, device=device)
    rainfall = _shared(np.asarray(rainfall_forecast, np.float32)[:horizon], n, device=device)
    priority = torch.as_tensor(bundle.priority_indices, dtype=torch.long, device=device)

    tfv_members = []
    budget_members = []
    for model in bundle.step2_models:
        out = model(
            state_history=state,
            historical_actions=history_action,
            rainfall_forecast=rainfall,
            action_candidate=candidate,
            action_no_control=no_control,
            action_dynamic_internal=internal,
            action_hold_previous=hold,
            edge_index=bundle.edge_index,
            node_static=bundle.node_static,
            action_node_map=bundle.surrogate_action_node_map,
            priority_node_indices=priority,
        )
        pfv_delta = out["pfv_delta"]
        no_control_pfv = out["kpi_no_control"]["pfv_m3"]
        budget_metric = pfv_delta - 0.05 * torch.clamp(no_control_pfv, min=0.0)
        tfv_members.append(out["tfv_delta"])
        budget_members.append(budget_metric)
    tfv_stack = torch.stack(tfv_members, dim=0)
    budget_stack = torch.stack(budget_members, dim=0)
    tfv_mean = tfv_stack.mean(dim=0)
    budget_mean = budget_stack.mean(dim=0)
    if budget_stack.shape[0] > 1:
        budget_std = budget_stack.std(dim=0, unbiased=True)
    else:
        budget_std = torch.zeros_like(budget_mean)
    budget_ucb = _calibrated_budget_ucb(budget_mean, budget_std, bundle.step2_calibration)
    if rolling_pfv_budget_state is not None:
        prefix = float(rolling_pfv_budget_state.realised_prefix_budget_metric_m3)
        absolute = float(rolling_pfv_budget_state.absolute_margin_m3)
        cumulative_ucb = budget_ucb + prefix
        remaining = absolute
    else:
        cumulative_ucb = budget_ucb
        remaining = 100.0
    return tfv_mean, cumulative_ucb, float(remaining)


def _candidate_objective(
    *,
    tfv_mean,
    cumulative_budget_ucb,
    budget_limit: float,
    candidate,
    seed,
    config: DifferentiableSearchConfig,
):
    import torch

    tfv_scale = torch.clamp(tfv_mean.detach().abs().median(), min=100.0)
    violation = torch.relu(cumulative_budget_ucb - float(budget_limit))
    pfv_scale = torch.clamp(cumulative_budget_ucb.detach().abs().median(), min=100.0)
    trust = torch.mean((candidate[:, :3] - seed[:, :3]) ** 2, dim=(1, 2))
    return (
        tfv_mean / tfv_scale
        + float(config.barrier_weight) * (violation / pfv_scale) ** 2
        + float(config.trust_region_weight) * trust
    )


def _binary_indices(actuator_ids: Sequence[str]) -> list[int]:
    return [i for i, aid in enumerate(actuator_ids) if str(aid) in BINARY_IDS]


def _normalise_seed(sequence: np.ndarray, current: np.ndarray, horizon: int) -> np.ndarray:
    value = np.asarray(sequence, dtype=np.float32)
    if value.shape != (horizon, current.size):
        return np.repeat(current[None, :], horizon, axis=0).astype(np.float32)
    result = np.clip(value, 0.0, 1.0).astype(np.float32, copy=True)
    result[3:] = current[None, :]
    return result


def _dedupe(sequences: Iterable[np.ndarray]) -> list[np.ndarray]:
    unique: dict[str, np.ndarray] = {}
    for sequence in sequences:
        value = np.asarray(sequence, dtype=np.float32)
        unique.setdefault(action_sha256(value), value)
    return list(unique.values())


def _prepare_starts(
    *,
    current_action: np.ndarray,
    actuator_ids: Sequence[str],
    warm_starts: Sequence[np.ndarray],
    horizon: int,
    config: DifferentiableSearchConfig,
) -> list[np.ndarray]:
    current = np.asarray(current_action, dtype=np.float32).reshape(-1)
    base = np.repeat(current[None, :], horizon, axis=0)
    starts = [base]
    for sequence in list(warm_starts)[: max(0, int(config.max_warm_starts) - 1)]:
        starts.append(_normalise_seed(sequence, current, horizon))
    binary = _binary_indices(actuator_ids)
    if config.include_all_binary_modes_for_current and binary:
        for mode in itertools.product((0.0, 1.0), repeat=len(binary)):
            candidate = base.copy()
            for index, value in zip(binary, mode):
                candidate[:3, index] = float(value)
            starts.append(candidate)
    for candidate in starts:
        for index in binary:
            candidate[:, index] = np.where(candidate[:, index] >= 0.5, 1.0, 0.0)
        candidate[3:] = current[None, :]
    return _dedupe(starts)


def _gradient_importance(
    *,
    bundle: Any,
    starts: list[np.ndarray],
    actuator_ids: Sequence[str],
    state_history: np.ndarray,
    historical_actions: np.ndarray,
    rainfall_forecast: np.ndarray,
    current_action: np.ndarray,
    internal_current_action: np.ndarray,
    rolling_pfv_budget_state: RollingPfvBudgetState | None,
) -> np.ndarray:
    import torch

    device = bundle.device
    candidate = _tensor(np.stack(starts), device=device).clone().detach().requires_grad_(True)
    tfv, budget, limit = _ensemble_objective(
        bundle=bundle,
        candidate=candidate,
        state_history=state_history,
        historical_actions=historical_actions,
        rainfall_forecast=rainfall_forecast,
        current_action=current_action,
        internal_current_action=internal_current_action,
        rolling_pfv_budget_state=rolling_pfv_budget_state,
    )
    objective = tfv + 10.0 * torch.relu(budget - limit) ** 2
    grad = torch.autograd.grad(objective.sum(), candidate, retain_graph=False, create_graph=False)[0]
    importance = grad[:, :3].abs().mean(dim=(0, 1)).detach().cpu().numpy()
    for index in _binary_indices(actuator_ids):
        importance[int(index)] = -np.inf
    return importance


def _optimise_spatial(
    *,
    bundle: Any,
    starts: list[np.ndarray],
    active_indices: Sequence[int],
    state_history: np.ndarray,
    historical_actions: np.ndarray,
    rainfall_forecast: np.ndarray,
    current_action: np.ndarray,
    internal_current_action: np.ndarray,
    rolling_pfv_budget_state: RollingPfvBudgetState | None,
    config: DifferentiableSearchConfig,
) -> tuple[list[np.ndarray], list[float]]:
    import torch

    if not active_indices:
        return starts, [0.0] * len(starts)
    device = bundle.device
    seed = _tensor(np.stack(starts), device=device)
    active = torch.as_tensor(list(active_indices), dtype=torch.long, device=device)
    initial = seed[:, 0].index_select(1, active).clone().detach()
    parameter = initial.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([parameter], lr=float(config.learning_rate))
    lower = torch.clamp(initial - float(config.max_h3_change_from_seed), 0.0, 1.0)
    upper = torch.clamp(initial + float(config.max_h3_change_from_seed), 0.0, 1.0)
    last_score = None
    for _ in range(max(1, int(config.spatial_steps))):
        optimizer.zero_grad(set_to_none=True)
        candidate = seed.clone()
        expanded = parameter[:, None, :].expand(-1, 3, -1)
        candidate[:, :3, active] = expanded
        tfv, budget, limit = _ensemble_objective(
            bundle=bundle,
            candidate=candidate,
            state_history=state_history,
            historical_actions=historical_actions,
            rainfall_forecast=rainfall_forecast,
            current_action=current_action,
            internal_current_action=internal_current_action,
            rolling_pfv_budget_state=rolling_pfv_budget_state,
        )
        score = _candidate_objective(
            tfv_mean=tfv,
            cumulative_budget_ucb=budget,
            budget_limit=limit,
            candidate=candidate,
            seed=seed,
            config=config,
        )
        score.sum().backward()
        optimizer.step()
        with torch.no_grad():
            parameter.clamp_(0.0, 1.0)
            parameter.copy_(torch.maximum(torch.minimum(parameter, upper), lower))
        last_score = score.detach()
    candidate = seed.clone().detach()
    candidate[:, :3, active] = parameter.detach()[:, None, :].expand(-1, 3, -1)
    return [x.detach().cpu().numpy().astype(np.float32) for x in candidate], (
        last_score.detach().cpu().numpy().astype(float).tolist() if last_score is not None else [0.0] * len(starts)
    )


def _optimise_temporal(
    *,
    bundle: Any,
    starts: list[np.ndarray],
    active_indices: Sequence[int],
    state_history: np.ndarray,
    historical_actions: np.ndarray,
    rainfall_forecast: np.ndarray,
    current_action: np.ndarray,
    internal_current_action: np.ndarray,
    rolling_pfv_budget_state: RollingPfvBudgetState | None,
    config: DifferentiableSearchConfig,
) -> list[np.ndarray]:
    import torch

    if not starts or not active_indices:
        return []
    device = bundle.device
    seed = _tensor(np.stack(starts), device=device)
    active = torch.as_tensor(list(active_indices), dtype=torch.long, device=device)
    initial = seed[:, :3].index_select(2, active).clone().detach()
    parameter = initial.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([parameter], lr=float(config.learning_rate) * 0.75)
    lower = torch.clamp(initial - float(config.max_h3_change_from_seed) / 2.0, 0.0, 1.0)
    upper = torch.clamp(initial + float(config.max_h3_change_from_seed) / 2.0, 0.0, 1.0)
    for _ in range(max(1, int(config.temporal_steps))):
        optimizer.zero_grad(set_to_none=True)
        candidate = seed.clone()
        candidate[:, :3, active] = parameter
        tfv, budget, limit = _ensemble_objective(
            bundle=bundle,
            candidate=candidate,
            state_history=state_history,
            historical_actions=historical_actions,
            rainfall_forecast=rainfall_forecast,
            current_action=current_action,
            internal_current_action=internal_current_action,
            rolling_pfv_budget_state=rolling_pfv_budget_state,
        )
        score = _candidate_objective(
            tfv_mean=tfv,
            cumulative_budget_ucb=budget,
            budget_limit=limit,
            candidate=candidate,
            seed=seed,
            config=config,
        )
        score.sum().backward()
        optimizer.step()
        with torch.no_grad():
            parameter.clamp_(0.0, 1.0)
            parameter.copy_(torch.maximum(torch.minimum(parameter, upper), lower))
    candidate = seed.clone().detach()
    candidate[:, :3, active] = parameter.detach()
    return [x.detach().cpu().numpy().astype(np.float32) for x in candidate]


def generate_differentiable_candidates(
    *,
    bundle: Any,
    actuator_ids: Sequence[str],
    state_history: np.ndarray,
    historical_actions: np.ndarray,
    rainfall_forecast: np.ndarray,
    current_action: np.ndarray,
    internal_current_action: np.ndarray,
    warm_starts: Sequence[np.ndarray] = (),
    rolling_pfv_budget_state: RollingPfvBudgetState | None = None,
    config: DifferentiableSearchConfig = DifferentiableSearchConfig(),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Generate a compact, state-adaptive gradient candidate population."""
    horizon = 12
    current = np.asarray(current_action, dtype=np.float32).reshape(-1)
    if len(actuator_ids) != current.size:
        raise ValueError("actuator_ids/current_action size mismatch")
    starts = _prepare_starts(
        current_action=current,
        actuator_ids=actuator_ids,
        warm_starts=warm_starts,
        horizon=horizon,
        config=config,
    )
    importance = _gradient_importance(
        bundle=bundle,
        starts=starts,
        actuator_ids=actuator_ids,
        state_history=state_history,
        historical_actions=historical_actions,
        rainfall_forecast=rainfall_forecast,
        current_action=current,
        internal_current_action=internal_current_action,
        rolling_pfv_budget_state=rolling_pfv_budget_state,
    )
    order = np.argsort(-importance, kind="stable")
    active = [int(i) for i in order if np.isfinite(importance[int(i)])][: int(config.active_continuous_assets)]
    spatial, spatial_score = _optimise_spatial(
        bundle=bundle,
        starts=starts,
        active_indices=active,
        state_history=state_history,
        historical_actions=historical_actions,
        rainfall_forecast=rainfall_forecast,
        current_action=current,
        internal_current_action=internal_current_action,
        rolling_pfv_budget_state=rolling_pfv_budget_state,
        config=config,
    )
    ranked = [x for _, x in sorted(zip(spatial_score, spatial), key=lambda item: item[0])]
    temporal_assets = active[: int(config.temporal_refine_assets)]
    temporal = _optimise_temporal(
        bundle=bundle,
        starts=ranked[: int(config.temporal_refine_top_k)],
        active_indices=temporal_assets,
        state_history=state_history,
        historical_actions=historical_actions,
        rainfall_forecast=rainfall_forecast,
        current_action=current,
        internal_current_action=internal_current_action,
        rolling_pfv_budget_state=rolling_pfv_budget_state,
        config=config,
    )
    binary = _binary_indices(actuator_ids)
    combined = _dedupe([*spatial, *temporal])
    result: list[dict[str, Any]] = []
    for sequence in combined[: int(config.max_return_candidates)]:
        value = np.asarray(sequence, dtype=np.float32).copy()
        value = np.clip(value, 0.0, 1.0)
        value[3:] = current[None, :]
        for index in binary:
            value[:3, index] = np.where(value[:3, index] >= 0.5, 1.0, 0.0)
            value[3:, index] = current[index]
        result.append(
            {
                "label": f"gradient_refined|sha={action_sha256(value)[:12]}",
                "sequence": value,
                "candidate_family": "experience_guided_gradient",
                "target_actuators": ",".join(str(actuator_ids[i]) for i in active),
                "physical_rationale": "Multi-start differentiable Step2 refinement; final PFV-UCB admission is external.",
            }
        )
    diagnostics = {
        "contract": HYBRID_SEARCH_CONTRACT,
        "warm_start_count": len(starts),
        "active_indices": active,
        "active_actuator_ids": [str(actuator_ids[i]) for i in active],
        "active_gradient_importance": [float(importance[i]) for i in active],
        "spatial_candidate_count": len(spatial),
        "temporal_candidate_count": len(temporal),
        "returned_candidate_count": len(result),
        "binary_outer_semantics": "exact_0_1",
        "h4_h12": "current_readback",
        "search_is_safety_authority": False,
    }
    return result, diagnostics
