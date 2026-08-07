"""Corrected Formal V4.2 PFV-only / TFV-min runtime selector.

This module fixes two search-space errors in the first PFV-only implementation:

1. Global TFV minimisation must not be restricted to actuators in PFV_CORE8
   influence domains. Local priority-domain candidates are still useful, but
   every Engineering36 asset also receives global single-asset candidates.
2. The requested candidate budget is applied *after* the H3 executable-prefix
   projection and deduplication. Profiles that collapse to the same executed
   sequence can no longer consume the budget before scoring. If complete
   Engineering36 single-asset coverage itself is slightly larger than the
   requested budget, the effective budget is raised only enough to preserve
   that coverage.

It also evaluates the calibrated PFV safety statistic directly as

    PFV_candidate - 1.05 * PFV_no_control <= 100 m3,

using one ensemble scalar per seed before the one-sided UCB is formed. This
keeps uncertainty in the No-control reference inside the same calibrated safety
quantity instead of treating the 5% reference term as deterministic.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import torch

from sewerrtc.control.action_sequence_generator import generate_action_sequences
from sewerrtc.control.pfvfirst_mpc_v42 import (
    FrozenFallback,
    MPCandidate,
    MPCWeights,
    SafetyMargins,
    decide_pfvfirst_mpc,
)
from sewerrtc.control.rolling_pfv_budget_v42 import RollingPfvBudgetState
from sewerrtc.v4 import v42_formal_runtime as base_runtime


PFV_RELATIVE_ALLOWANCE_FRACTION = 0.05
GLOBAL_SINGLE_DELTA = 0.12
GLOBAL_SINGLE_EXTRA_DELTAS = (0.25, 0.50)
BINARY_IDS = {"ADD301.2", "ADD301.3"}


def _shared_batch_tensor(value: np.ndarray, batch_size: int, device: torch.device) -> torch.Tensor:
    """Copy one shared input to the device, then expose it as a batch view."""
    base = torch.as_tensor(np.asarray(value, dtype=np.float32), device=device)
    return base.unsqueeze(0).expand(int(batch_size), *base.shape)


def _pfv_budget_metric_ucb(
    mean: np.ndarray,
    std: np.ndarray,
    calibration: dict[str, Any],
) -> tuple[np.ndarray, str]:
    """Return the frozen one-sided PFV UCB without changing the 100 m3 gate."""
    raw_margin = calibration.get("pfv_budget_metric_residual_margin_m3", np.nan)
    try:
        margin = float(raw_margin)
    except (TypeError, ValueError):
        # Canonical standardized calibration records this optional field as
        # JSON null; null means the diagnostic alternative is disabled.
        margin = np.nan
    if (
        calibration.get("pfv_budget_metric_ucb_method")
        == "absolute_residual_one_sided_conformal"
        and np.isfinite(margin)
        and margin >= 0.0
    ):
        return np.asarray(mean, dtype=float) + margin, "absolute_residual_one_sided_conformal"
    z = float(calibration.get("confidence_z", np.nan))
    if not np.isfinite(z) or z < 0.0:
        raise RuntimeError("Formal Step2 PFV calibration margin is not finite")
    return np.asarray(mean, dtype=float) + z * np.asarray(std, dtype=float), "standardized_ensemble_conformal_legacy"


def _global_tfv_sequences(
    base: np.ndarray,
    actuators: pd.DataFrame,
) -> list[dict[str, Any]]:
    """Cover every Engineering36 asset independently of priority-node locality."""
    ids = actuators["actuator_id"].astype(str).tolist()
    reference = np.repeat(
        np.asarray(base, dtype=np.float32)[None, :],
        base_runtime.HORIZON_STEPS,
        axis=0,
    )
    prefix = min(base_runtime.CONTROLLABLE_PREFIX_STEPS, base_runtime.HORIZON_STEPS)
    result: list[dict[str, Any]] = []
    for idx, aid in enumerate(ids):
        current = float(reference[0, idx])
        if aid in BINARY_IDS:
            target = 0.0 if current >= 0.5 else 1.0
            seq = reference.copy()
            seq[:prefix, idx] = target
            result.append(
                {
                    "label": f"global_binary_toggle|actuator={aid}|target={int(target)}",
                    "sequence": seq,
                    "target_actuators": aid,
                    "physical_rationale": "Global TFV search: explicit executable binary transition.",
                }
            )
            continue
        for magnitude in (GLOBAL_SINGLE_DELTA, *GLOBAL_SINGLE_EXTRA_DELTAS):
            for direction, delta in (("decrease", -magnitude), ("increase", magnitude)):
                target = float(np.clip(current + delta, 0.0, 1.0))
                if abs(target - current) <= 1.0e-7:
                    continue
                seq = reference.copy()
                seq[:prefix, idx] = target
                label = f"global_tfv_single|actuator={aid}|direction={direction}"
                if magnitude != GLOBAL_SINGLE_DELTA:
                    label += f"|delta={magnitude:g}"
                result.append(
                    {
                        "label": label,
                        "sequence": seq,
                        "target_actuators": aid,
                        "physical_rationale": "Global TFV search candidate; PFV is checked only by the calibrated admission gate.",
                    }
                )
    return result


def _project_dedupe_and_cap(
    raw_candidates: list[dict[str, Any]],
    *,
    base: np.ndarray,
    actuators: pd.DataFrame,
    requested_cap: int,
) -> tuple[list[tuple[str, np.ndarray, Any, int, bool]], dict[str, int]]:
    """Project first, deduplicate executed H3 semantics, then apply the cap."""
    unique: list[tuple[str, np.ndarray, Any, int, bool]] = []
    seen: set[tuple[tuple[int, ...], bytes]] = set()
    for item in raw_candidates:
        seq, engineering, k_count, executable = base_runtime.project_candidate_sequence(
            np.asarray(item["sequence"], np.float32), base, actuators
        )
        rounded = np.round(seq, 6).astype(np.float32, copy=False)
        key = (tuple(rounded.shape), rounded.tobytes())
        if key in seen:
            continue
        seen.add(key)
        unique.append(
            (
                str(item.get("label", f"candidate_{len(unique)}")),
                seq,
                engineering,
                k_count,
                executable,
            )
        )

    def rank(item: tuple[str, np.ndarray, Any, int, bool]) -> tuple[int, str]:
        label = item[0]
        if label == "hold_native" or label == "hold_only":
            return (0, label)
        if label.startswith("global_binary_toggle"):
            return (1, label)
        if label.startswith("global_tfv_single"):
            return (2, label)
        if "binary_toggle" in label:
            return (3, label)
        if "priority_group" in label:
            return (4, label)
        return (5, label)

    ordered = sorted(unique, key=rank)
    mandatory = [
        item
        for item in ordered
        if item[0] in {"hold_native", "hold_only"}
        or item[0].startswith("global_binary_toggle")
        or (
            item[0].startswith("global_tfv_single")
            and "|delta=" not in item[0]
        )
    ]
    mandatory_keys = {item[0] for item in mandatory}
    requested = max(1, int(requested_cap))
    effective_cap = max(requested, len(mandatory))
    selected = list(mandatory)
    for item in ordered:
        if item[0] in mandatory_keys:
            continue
        if len(selected) >= effective_cap:
            break
        selected.append(item)
    return selected, {
        "raw_candidate_count": len(raw_candidates),
        "projected_unique_candidate_count": len(unique),
        "requested_candidate_cap": requested,
        "effective_candidate_cap": effective_cap,
        "global_coverage_candidate_count": len(mandatory),
    }


def predict_and_decide(
    *,
    bundle: base_runtime.FormalModelBundle,
    actuators: pd.DataFrame,
    state_history: np.ndarray,
    historical_actions: np.ndarray,
    rainfall_forecast: np.ndarray,
    current_action: np.ndarray,
    internal_current_action: np.ndarray,
    gat_ood_score: float,
    max_candidate_sequences: int = 64,
    rolling_pfv_budget_state: RollingPfvBudgetState | None = None,
    extra_candidate_sequences: list[dict[str, Any]] | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Run the corrected global candidate pool and calibrated PFV-budget selector."""
    ids = actuators["actuator_id"].astype(str).tolist()
    base = np.asarray(current_action, np.float32)

    # Global candidates are deliberately inserted before priority-local
    # heuristics. Projection deduplication keeps the first semantic equivalent;
    # putting global coverage first prevents a local alias from stealing the
    # mandatory rank and then being discarded by the post-projection cap.
    generated = _global_tfv_sequences(base, actuators)
    if extra_candidate_sequences:
        generated.extend(extra_candidate_sequences)
    generated.extend(
        generate_action_sequences(
            base,
            actuators,
            base_runtime.HORIZON_STEPS,
            max_delta=GLOBAL_SINGLE_DELTA,
            include_hold=True,
            max_sequences=0,
            group_limit=8,
            reference_sequence=np.repeat(
                base[None, :], base_runtime.HORIZON_STEPS, axis=0
            ),
            priority_to_actuators=bundle.priority_to_actuators,
        )
    )
    projected, pool_stats = _project_dedupe_and_cap(
        generated,
        base=base,
        actuators=actuators,
        requested_cap=max_candidate_sequences,
    )
    if not projected:
        seq = np.repeat(base[None, :], base_runtime.HORIZON_STEPS, axis=0)
        projected = [
            (
                "hold_only",
                seq,
                base_runtime.EngineeringStatus(True, True, True, True, True),
                0,
                True,
            )
        ]

    candidate = np.stack([x[1] for x in projected])
    n = len(candidate)
    # These inputs are identical for every candidate.  Keep one device copy
    # and broadcast it as a view; only the candidate tensor is materialized per
    # row.  This removes repeated CPU allocation and H2D transfer per decision.
    nc = np.ones((base_runtime.HORIZON_STEPS, len(ids)), np.float32)
    internal = np.repeat(
        np.asarray(internal_current_action, np.float32)[None, :],
        base_runtime.HORIZON_STEPS,
        axis=0,
    )
    hold = np.repeat(base[None, :], base_runtime.HORIZON_STEPS, axis=0)
    state_history_value = np.asarray(state_history, np.float32)
    historical_actions_value = np.asarray(historical_actions, np.float32)
    rainfall_value = np.asarray(rainfall_forecast, np.float32)[: base_runtime.HORIZON_STEPS]
    priority = torch.as_tensor(
        bundle.priority_indices, dtype=torch.long, device=bundle.device
    )
    state_history_tensor = _shared_batch_tensor(
        state_history_value, n, bundle.device
    )
    historical_actions_tensor = _shared_batch_tensor(
        historical_actions_value, n, bundle.device
    )
    rainfall_tensor = _shared_batch_tensor(rainfall_value, n, bundle.device)
    candidate_tensor = torch.as_tensor(candidate, device=bundle.device)
    no_control_tensor = _shared_batch_tensor(nc, n, bundle.device)
    internal_tensor = _shared_batch_tensor(internal, n, bundle.device)
    hold_tensor = _shared_batch_tensor(hold, n, bundle.device)

    predictions: list[dict[str, np.ndarray]] = []
    with torch.inference_mode():
        for model in bundle.step2_models:
            out = model(
                state_history=state_history_tensor,
                historical_actions=historical_actions_tensor,
                rainfall_forecast=rainfall_tensor,
                action_candidate=candidate_tensor,
                action_no_control=no_control_tensor,
                action_dynamic_internal=internal_tensor,
                action_hold_previous=hold_tensor,
                edge_index=bundle.edge_index,
                node_static=bundle.node_static,
                action_node_map=bundle.surrogate_action_node_map,
                priority_node_indices=priority,
            )
            pfv_delta = out["pfv_delta"].detach().cpu().numpy()
            no_control_pfv = (
                out["kpi_no_control"]["pfv_m3"].detach().cpu().numpy()
            )
            predictions.append(
                {
                    "pfv_delta": pfv_delta,
                    "pfv_budget_metric": pfv_delta
                    - PFV_RELATIVE_ALLOWANCE_FRACTION
                    * np.maximum(no_control_pfv, 0.0),
                    "tfv_delta": out["tfv_delta"].detach().cpu().numpy(),
                    "peak_delta": out["peak_delta"].detach().cpu().numpy(),
                    "no_control_pfv": no_control_pfv,
                    "depth": out["branches"]["candidate"]["node_depth"][:, :, priority]
                    .detach()
                    .cpu()
                    .numpy(),
                }
            )

    stack = {
        key: np.stack([prediction[key] for prediction in predictions], axis=0)
        for key in predictions[0]
    }
    mean = {key: value.mean(axis=0) for key, value in stack.items()}
    std = {key: value.std(axis=0, ddof=1) for key, value in stack.items()}
    z = float(bundle.step2_calibration.get("confidence_z", np.nan))
    if not np.isfinite(z):
        raise RuntimeError("Formal Step2 calibration confidence_z is not finite")
    if bundle.step2_calibration.get("pfv_safety_statistic") != "candidate_minus_1p05_no_control":
        raise RuntimeError(
            "Formal runtime requires PFV calibration of candidate_minus_1p05_no_control"
        )

    # Independent uncertainty/OOD gates remain diagnostic only. The calibrated
    # scalar PFV budget metric is the sole hydraulic admission statistic.
    budget_metric_std = std["pfv_budget_metric"]
    budget_metric_ucb, pfv_ucb_method = _pfv_budget_metric_ucb(
        mean["pfv_budget_metric"], budget_metric_std, bundle.step2_calibration
    )
    uncertainty_scale = float(
        bundle.step2_calibration.get("pfv_budget_metric_std_scale", 1.0)
    )
    uncertainty_score = np.abs(budget_metric_std / max(uncertainty_scale, 1.0e-6))
    uncertainty_limit = float(
        bundle.step2_calibration.get("uncertainty_limit_99", np.nan)
    )
    ood_limit = float(bundle.step1_calibration.get("ood_limit_99", np.nan))

    candidates: list[MPCandidate] = []
    for i, (label, seq, engineering, k_count, executable) in enumerate(projected):
        depth_ucb = mean["depth"][i] + z * std["depth"][i]
        candidates.append(
            MPCandidate(
                candidate_id=label,
                action_sequence=seq,
                pfv_delta_ucb_m3=float(
                    mean["pfv_delta"][i] + z * std["pfv_delta"][i]
                ),
                peak_delta_ucb_m3s=float(
                    mean["peak_delta"][i] + z * std["peak_delta"][i]
                ),
                tfv_delta_di_m3=float(mean["tfv_delta"][i]),
                action_cost=float(
                    np.mean(
                        np.abs(
                            seq[: base_runtime.CONTROLLABLE_PREFIX_STEPS]
                            - base[None, :]
                        )
                    )
                ),
                terminal_cost=float(
                    np.mean(
                        np.abs(
                            seq[base_runtime.CONTROLLABLE_PREFIX_STEPS - 1] - base
                        )
                    )
                ),
                uncertainty_cost=float(uncertainty_score[i]),
                changed_facilities=k_count,
                engineering=engineering,
                uncertainty_pass=True,
                ood_pass=True,
                executable=executable,
                pfv_no_control_m3=float(mean["no_control_pfv"][i]),
                priority_depth_ucb_m=tuple(
                    depth_ucb.reshape(-1).astype(float)
                ),
                priority_depth_limit_m=tuple(
                    np.tile(
                        bundle.priority_depth_limits,
                        base_runtime.HORIZON_STEPS,
                    ).astype(float)
                ),
                metadata={
                    "ensemble_seed_count": len(bundle.step2_models),
                    "confidence_z": z,
                    "pfv_budget_metric_ucb_method": pfv_ucb_method,
                    "dynamic_internal_action_forecast": "causal_current_native_rule_setting_persistence",
                    "no_control_action_forecast": "all_engineering36_open_training_contract",
                    "hold_action_forecast": "current_readback_persistence",
                    "candidate_search_scope": "global_engineering36_plus_priority_local",
                },
                pfv_budget_metric_ucb_m3=float(budget_metric_ucb[i]),
            )
        )

    fallback_seq = np.repeat(
        base[None, :], base_runtime.HORIZON_STEPS, axis=0
    )
    decision = decide_pfvfirst_mpc(
        candidates=candidates,
        fallback=FrozenFallback(
            fallback_id="frozen_hold_readback",
            action_sequence=fallback_seq,
            contract_hash=bundle.fallback_contract_sha256,
            legal=True,
        ),
        margins=SafetyMargins(),
        weights=MPCWeights(),
        expected_fallback_contract_hash=bundle.fallback_contract_sha256,
        rolling_pfv_budget_state=rolling_pfv_budget_state,
    )
    audit_rows = [
        {
            "candidate_id": audit.candidate_id,
            "safe": audit.safe,
            "rejection_reasons": list(audit.rejection_reasons),
            "objective": audit.objective,
            "pfv_allowance_m3": audit.pfv_allowance_m3,
            "pfv_budget_metric_ucb_m3": audit.pfv_budget_metric_ucb_m3,
            "maximum_priority_depth_exceedance_m": audit.maximum_priority_depth_exceedance_m,
        }
        for audit in decision.audits
    ]
    info: dict[str, Any] = {
        "selected_id": decision.selected_id,
        "used_fallback": decision.used_fallback,
        "reason": decision.reason,
        "selected_objective_score": decision.objective,
        "candidate_audits": audit_rows,
        "gat_ood_score": float(gat_ood_score),
        "ood_limit_99": ood_limit,
        "uncertainty_limit_99": uncertainty_limit,
        "canonical_pfvfirst_mpc_v42": True,
        "control_objective_contract": base_runtime.FORMAL_OBJECTIVE_CONTRACT,
        "pfv_budget_applied": True,
        "rolling_pfv_budget_applied": rolling_pfv_budget_state is not None,
        "rolling_pfv_allowance_reinitialised_each_decision": False,
        "pfv_safety_statistic": "candidate_minus_1p05_no_control",
        "uncertainty_used_for_pfv_ucb": True,
        "priority_depth_hard_gate": False,
        "global_peak_hard_gate": False,
        "global_peak_objective_term": False,
        "peak_penalty_weight": 0.0,
        "action_penalty_weight": 0.0,
        "terminal_penalty_weight": 0.0,
        "uncertainty_penalty_weight": 0.0,
        "independent_OOD_gate": False,
        "independent_uncertainty_gate": False,
        "objective": "minimize_TFV_subject_to_PFV_budget",
        "candidate_search_scope": "global_engineering36_plus_priority_local",
        "candidate_cap_applied_after_projection": True,
        "future_hydraulic_truth_used_online": False,
        "realized_future_rainfall_used_online": False,
        "dynamic_internal_future_truth_used_online": False,
    }
    info.update(pool_stats)
    return decision.execute_action.astype(np.float32), info
