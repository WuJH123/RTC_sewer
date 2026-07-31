from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from .action_sequence_generator import generate_action_sequences
from .horizon_objective import score_horizon_sequence, score_horizon_system_repair_sequence


HorizonPredictor = Callable[[np.ndarray, dict[str, Any]], dict[str, np.ndarray]]


@dataclass(frozen=True)
class GenericGATMPCConfig:
    horizon_steps: int = 6
    max_candidate_delta: float = 0.08
    max_candidate_sequences: int = 512
    candidate_group_limit: int = 12
    smooth_weight: float = 0.05
    violation_penalty: float = 1.0e6
    min_pfv_improvement_abs: float = 1.0
    min_pfv_improvement_frac: float = 0.0
    tfv_tolerance_abs: float = 0.0
    tfv_tolerance_frac: float = 0.0
    peak_tolerance_abs: float = 0.0
    peak_tolerance_frac: float = 0.0
    pfv_tolerance_abs: float = 0.0
    pfv_tolerance_frac: float = 0.0
    tfv_required_reduction_abs: float = 0.0
    tfv_required_reduction_frac: float = 0.0
    tfv_required_reduction_dry_multiplier: float = 1.0
    tfv_hard_constraint: bool = True
    dry_rain_threshold: float = 0.10
    peak_weight: float = 1.0
    pfv_weight: float = 1.0
    adaptive_delta_enabled: bool = False
    low_risk_max_candidate_delta: float = 0.08
    high_risk_max_candidate_delta: float = 0.03
    pfv_high_risk_horizon_threshold: float = 1000.0
    pfv_low_risk_horizon_threshold: float = 100.0
    max_first_step_delta: float = 1.0
    per_actuator_max_delta: tuple[tuple[str, float], ...] = ()
    min_hold_steps_by_actuator: tuple[tuple[str, int], ...] = ()
    objective_mode: str = "pfv_first"
    allowed_actuator_ids: tuple[str, ...] = ()
    blocked_actuator_ids: tuple[str, ...] = ()
    empirical_single_action_gate: bool = False
    empirical_hold_steps: int = 2
    pump_control_mode: str = "continuous"
    variable_speed_pump_ids: tuple[str, ...] = ()


def _parse_id_filter(value) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        values = value.replace(";", ",").split(",")
    else:
        values = value
    out = []
    for item in values:
        text = str(item).strip()
        if text and text not in out:
            out.append(text)
    return tuple(out)


def _as_horizon_array(values, horizon_steps: int, default: float = 0.0) -> np.ndarray:
    arr = np.asarray(values if values is not None else [], dtype=float).reshape(-1)
    if arr.size == 0:
        arr = np.full(int(horizon_steps), float(default), dtype=float)
    if arr.size < int(horizon_steps):
        arr = np.pad(arr, (0, int(horizon_steps) - arr.size), mode="edge")
    return np.nan_to_num(arr[: int(horizon_steps)], nan=float(default)).astype(float)


class GenericGATMPCController:
    """Discrete rolling-horizon GAT-MPC controller independent of native rules.

    The controller receives reconstructed hydraulic state and a future rainfall
    window, generates bounded multi-step action sequences over influence-domain
    assets, scores each sequence with a horizon predictor, and returns only the
    first action for real-time receding-horizon execution.
    """

    def __init__(
        self,
        actuators: pd.DataFrame,
        *,
        horizon_steps: int = 6,
        max_candidate_delta: float = 0.08,
        priority_to_actuators: pd.DataFrame | None = None,
        horizon_predictor: HorizonPredictor | None = None,
        predictor_source: str = "",
        smooth_weight: float = 0.05,
        violation_penalty: float = 1.0e6,
        min_pfv_improvement_abs: float = 1.0,
        min_pfv_improvement_frac: float = 0.0,
        max_candidate_sequences: int = 512,
        candidate_group_limit: int = 12,
        tfv_tolerance_abs: float = 0.0,
        tfv_tolerance_frac: float = 0.0,
        peak_tolerance_abs: float = 0.0,
        peak_tolerance_frac: float = 0.0,
        pfv_tolerance_abs: float = 0.0,
        pfv_tolerance_frac: float = 0.0,
        tfv_required_reduction_abs: float = 0.0,
        tfv_required_reduction_frac: float = 0.0,
        tfv_required_reduction_dry_multiplier: float = 1.0,
        tfv_hard_constraint: bool = True,
        dry_rain_threshold: float = 0.10,
        peak_weight: float = 1.0,
        pfv_weight: float = 1.0,
        adaptive_delta_enabled: bool = False,
        low_risk_max_candidate_delta: float = 0.08,
        high_risk_max_candidate_delta: float = 0.03,
        pfv_high_risk_horizon_threshold: float = 1000.0,
        pfv_low_risk_horizon_threshold: float = 100.0,
        max_first_step_delta: float = 1.0,
        per_actuator_max_delta: dict[str, float] | None = None,
        min_hold_steps_by_actuator: dict[str, int] | None = None,
        objective_mode: str = "pfv_first",
        allowed_actuator_ids: set[str] | list[str] | tuple[str, ...] | str | None = None,
        blocked_actuator_ids: set[str] | list[str] | tuple[str, ...] | str | None = None,
        allowed_action_directions: dict[str, set[str] | list[str] | tuple[str, ...]] | None = None,
        empirical_single_action_gate: bool = False,
        empirical_hold_steps: int = 2,
        pump_control_mode: str = "continuous",
        variable_speed_pump_ids: set[str] | list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self.actuators = actuators.reset_index(drop=True).copy()
        self.config = GenericGATMPCConfig(
            horizon_steps=max(1, int(horizon_steps)),
            max_candidate_delta=abs(float(max_candidate_delta)),
            max_candidate_sequences=max(0, int(max_candidate_sequences)),
            candidate_group_limit=max(1, int(candidate_group_limit)),
            smooth_weight=float(smooth_weight),
            violation_penalty=float(violation_penalty),
            min_pfv_improvement_abs=max(0.0, float(min_pfv_improvement_abs)),
            min_pfv_improvement_frac=max(0.0, float(min_pfv_improvement_frac)),
            tfv_tolerance_abs=max(0.0, float(tfv_tolerance_abs)),
            tfv_tolerance_frac=max(0.0, float(tfv_tolerance_frac)),
            peak_tolerance_abs=max(0.0, float(peak_tolerance_abs)),
            peak_tolerance_frac=max(0.0, float(peak_tolerance_frac)),
            pfv_tolerance_abs=max(0.0, float(pfv_tolerance_abs)),
            pfv_tolerance_frac=max(0.0, float(pfv_tolerance_frac)),
            tfv_required_reduction_abs=max(0.0, float(tfv_required_reduction_abs)),
            tfv_required_reduction_frac=max(0.0, float(tfv_required_reduction_frac)),
            tfv_required_reduction_dry_multiplier=max(1.0, float(tfv_required_reduction_dry_multiplier)),
            tfv_hard_constraint=bool(tfv_hard_constraint),
            dry_rain_threshold=max(0.0, float(dry_rain_threshold)),
            peak_weight=max(0.0, float(peak_weight)),
            pfv_weight=max(0.0, float(pfv_weight)),
            adaptive_delta_enabled=bool(adaptive_delta_enabled),
            low_risk_max_candidate_delta=max(0.0, float(low_risk_max_candidate_delta)),
            high_risk_max_candidate_delta=max(0.0, float(high_risk_max_candidate_delta)),
            pfv_high_risk_horizon_threshold=max(1.0e-9, float(pfv_high_risk_horizon_threshold)),
            pfv_low_risk_horizon_threshold=max(0.0, float(pfv_low_risk_horizon_threshold)),
            max_first_step_delta=max(0.0, float(max_first_step_delta)),
            per_actuator_max_delta=tuple(
                (str(key), max(0.0, float(value)))
                for key, value in (per_actuator_max_delta or {}).items()
                if str(key).strip()
            ),
            min_hold_steps_by_actuator=tuple(
                (str(key), max(0, int(value)))
                for key, value in (min_hold_steps_by_actuator or {}).items()
                if str(key).strip()
            ),
            objective_mode=str(objective_mode or "pfv_first").strip().lower(),
            allowed_actuator_ids=_parse_id_filter(allowed_actuator_ids),
            blocked_actuator_ids=_parse_id_filter(blocked_actuator_ids),
            empirical_single_action_gate=bool(empirical_single_action_gate),
            empirical_hold_steps=max(1, int(empirical_hold_steps)),
            pump_control_mode=str(pump_control_mode or "continuous").strip().lower(),
            variable_speed_pump_ids=tuple(
                str(x).strip() for x in (variable_speed_pump_ids or []) if str(x).strip()
            ),
        )
        self.priority_to_actuators = priority_to_actuators.copy() if priority_to_actuators is not None else None
        self.allowed_action_directions = dict(allowed_action_directions or {})
        self.runtime_allowed_actuator_ids: tuple[str, ...] | None = None
        self.runtime_allowed_action_directions: dict[str, set[str]] | None = None
        self.runtime_allowed_action_delta_limits: dict[str, dict[str, float]] | None = None
        self.runtime_verified_action_effects: dict[str, dict[str, list[dict[str, float]]]] | None = None
        self.runtime_candidate_group_limit: int | None = None
        self.horizon_predictor = horizon_predictor or self._proxy_predictor
        self.predictor_source = str(predictor_source or ("proxy_hydraulic_heuristic" if horizon_predictor is None else "learned_horizon_surrogate"))
        self._decision_index = 0
        self._last_change_index = np.full(len(self.actuators), -10**9, dtype=int)

    def set_runtime_action_filter(
        self,
        actuator_ids: set[str] | list[str] | tuple[str, ...] | None,
        directions: dict[str, set[str] | list[str] | tuple[str, ...]] | None,
        candidate_group_limit: int | None = None,
        action_delta_limits: dict[str, dict[str, float]] | None = None,
        verified_action_effects: dict[str, dict[str, list[dict[str, float]]]] | None = None,
    ) -> None:
        """Apply a phase-specific empirical filter for the next MPC decision."""
        parsed_ids = None if actuator_ids is None else _parse_id_filter(actuator_ids)
        # ``generate_action_sequences`` treats an empty collection as no
        # filter. For an empirical phase gate, no locally verified actuator
        # must instead mean deny-all (while the hold sequence remains valid).
        self.runtime_allowed_actuator_ids = (
            None
            if parsed_ids is None
            else (parsed_ids if parsed_ids else ("__NO_RELIABLE_ACTUATOR__",))
        )
        self.runtime_allowed_action_directions = (
            None
            if directions is None
            else {
                str(aid): {str(direction).strip().lower() for direction in values}
                for aid, values in directions.items()
                if str(aid).strip()
            }
        )
        self.runtime_allowed_action_delta_limits = (
            None
            if action_delta_limits is None
            else {
                str(aid): {
                    str(direction).strip().lower(): max(0.0, float(limit))
                    for direction, limit in (limits or {}).items()
                }
                for aid, limits in action_delta_limits.items()
                if str(aid).strip()
            }
        )
        self.runtime_verified_action_effects = verified_action_effects
        self.runtime_candidate_group_limit = (
            None if candidate_group_limit is None else max(1, int(candidate_group_limit))
        )

    def _project_execution_sequence(self, sequence: np.ndarray, last_action: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
        """Apply device-level ramp and dwell constraints before scoring a candidate."""
        seq = np.asarray(sequence, dtype=np.float32).copy()
        if seq.ndim != 2 or seq.shape[1] != len(self.actuators):
            seq = np.resize(seq, (self.config.horizon_steps, len(self.actuators))).astype(np.float32)
        previous = np.asarray(last_action, dtype=np.float32).reshape(-1)
        if previous.size != len(self.actuators):
            previous = np.resize(previous, len(self.actuators)).astype(np.float32)
        id_to_index = {str(aid): idx for idx, aid in enumerate(self.actuators["actuator_id"].astype(str))}
        ramp_map = dict(self.config.per_actuator_max_delta)
        hold_map = dict(self.config.min_hold_steps_by_actuator)
        virtual_last_change = self._last_change_index.copy()
        ramp_clipped = 0
        dwell_held = 0
        for t in range(seq.shape[0]):
            base = previous if t == 0 else seq[t - 1]
            for aid, idx in id_to_index.items():
                requested = float(seq[t, idx])
                dwell = int(hold_map.get(aid, 0))
                # The horizon must itself be executable.  A sequence that
                # changes a pump at step 0 and reverses it at step 1 should
                # not be scored as feasible when the pump has a two-step
                # dwell constraint.
                virtual_step = int(self._decision_index + t)
                if dwell > 0 and virtual_step - int(virtual_last_change[idx]) < dwell:
                    if abs(requested - float(base[idx])) > 1.0e-7:
                        dwell_held += 1
                    seq[t, idx] = base[idx]
                    continue
                limit = float(ramp_map.get(aid, self.config.max_first_step_delta))
                bounded = float(np.clip(requested, float(base[idx]) - limit, float(base[idx]) + limit))
                if abs(bounded - requested) > 1.0e-7:
                    ramp_clipped += 1
                seq[t, idx] = bounded
                if abs(float(seq[t, idx]) - float(base[idx])) > 1.0e-7:
                    virtual_last_change[idx] = virtual_step
            seq[t] = np.clip(seq[t], 0.0, 1.0)
        # Apply the same physical pump projection used immediately before
        # PySWMM writes target_setting. Otherwise a fractional candidate can
        # be scored by the surrogate but executed as an ON/OFF pump action.
        mode = str(self.config.pump_control_mode or "continuous").lower()
        vfd_ids = set(self.config.variable_speed_pump_ids)
        if mode in {"binary", "binary_unless_verified", "on_off"} or (
            mode == "variable_speed" and vfd_ids
        ):
            link_types = (
                self.actuators.set_index("actuator_id")["link_type"].astype(str).str.lower().to_dict()
                if "actuator_id" in self.actuators and "link_type" in self.actuators
                else {}
            )
            for idx, aid in enumerate(self.actuators["actuator_id"].astype(str).tolist()):
                if link_types.get(aid, "") == "pump" and aid not in vfd_ids:
                    seq[:, idx] = (seq[:, idx] >= 0.5).astype(np.float32)
        return seq.astype(np.float32), {"ramp_clipped_values": ramp_clipped, "dwell_held_values": dwell_held}

    def _record_executed_action(self, action: np.ndarray, previous_action: np.ndarray) -> None:
        action = np.asarray(action, dtype=float).reshape(-1)
        previous = np.asarray(previous_action, dtype=float).reshape(-1)
        if action.size == previous.size:
            self._last_change_index[np.flatnonzero(np.abs(action - previous) > 1.0e-7)] = int(self._decision_index)
        self._decision_index += 1

    def _proxy_predictor(self, sequence: np.ndarray, context: dict[str, Any]) -> dict[str, np.ndarray]:
        """Fallback deterministic predictor used only when no learned model is wired.

        It keeps the generic controller executable for smoke tests and plumbing
        validation. Formal experiments should replace it with a trained horizon
        surrogate.
        """
        horizon_steps = self.config.horizon_steps
        state = np.asarray(context.get("reconstructed_state", []), dtype=float)
        rain = _as_horizon_array(context.get("rainfall_window"), horizon_steps)
        current = np.asarray(context.get("current_action", np.ones(sequence.shape[1])), dtype=float)
        seq = np.asarray(sequence, dtype=float)
        priority_depth = float(np.nanmax(state)) if state.size else 0.0
        baseline_risk = np.maximum(0.0, 10.0 * priority_depth + 0.25 * rain)
        action_delta = current[None, :] - seq
        retain_effect = np.maximum(0.0, action_delta).mean(axis=1)
        release_effect = np.maximum(0.0, -action_delta).mean(axis=1)
        pfv = np.maximum(0.0, baseline_risk * (1.0 - 0.25 * retain_effect + 0.10 * release_effect))
        tfv = np.maximum(0.0, np.asarray(context.get("reference_tfv", np.ones(horizon_steps)), dtype=float))
        peak = np.maximum(0.0, np.asarray(context.get("reference_peak", np.ones(horizon_steps)), dtype=float))
        return {
            "pfv": _as_horizon_array(pfv, horizon_steps),
            "tfv": _as_horizon_array(tfv, horizon_steps),
            "peak_tfv_rate": _as_horizon_array(peak, horizon_steps),
        }

    @staticmethod
    def _action_change(sequence: np.ndarray, current_action: np.ndarray) -> np.ndarray:
        seq = np.asarray(sequence, dtype=float)
        current = np.asarray(current_action, dtype=float).reshape(1, -1)
        first = np.nanmean(np.abs(seq[:1] - current), axis=1)
        if len(seq) <= 1:
            return first
        later = np.nanmean(np.abs(np.diff(seq, axis=0)), axis=1)
        return np.concatenate([first, later])

    def choose(
        self,
        *,
        reconstructed_state: np.ndarray,
        rainfall_window: np.ndarray,
        current_action: np.ndarray,
        reference_pfv: np.ndarray | None = None,
        reference_tfv: np.ndarray | None = None,
        reference_peak: np.ndarray | None = None,
        reference_action_sequence: np.ndarray | None = None,
        last_executed_action: np.ndarray | None = None,
        elapsed_min: float = 0.0,
        phase: str = "",
        empirical_local_verified: bool = False,
        extra_predictor_context: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        current = np.asarray(current_action, dtype=np.float32).reshape(-1)
        if len(current) != len(self.actuators):
            current = np.resize(current, len(self.actuators)).astype(np.float32)
        current = np.clip(current, 0.0, 1.0)
        last = np.asarray(last_executed_action if last_executed_action is not None else current, dtype=np.float32).reshape(-1)
        if len(last) != len(self.actuators):
            last = np.resize(last, len(self.actuators)).astype(np.float32)
        last = np.clip(last, 0.0, 1.0)
        horizon_steps = self.config.horizon_steps
        rainfall = _as_horizon_array(rainfall_window, horizon_steps)
        if reference_action_sequence is None:
            reference_actions = np.repeat(current[None, :], horizon_steps, axis=0).astype(np.float32)
        else:
            reference_actions = np.asarray(reference_action_sequence, dtype=np.float32)
            if reference_actions.ndim == 1:
                reference_actions = np.repeat(reference_actions[None, :], horizon_steps, axis=0)
            if reference_actions.shape != (horizon_steps, len(self.actuators)):
                reference_actions = np.resize(reference_actions, (horizon_steps, len(self.actuators))).astype(np.float32)
            reference_actions = np.clip(reference_actions, 0.0, 1.0)
            current = reference_actions[0].copy()
        reference_source = "provided_external_arrays"
        if reference_pfv is None or reference_tfv is None or reference_peak is None:
            default_sequence = reference_actions
            reference_context = {
                "label": "online_default_reference",
                "elapsed_min": float(elapsed_min),
                "phase": str(phase),
                "reconstructed_state": np.asarray(reconstructed_state, dtype=np.float32),
                "rainfall_window": rainfall,
                "current_action": current,
                "reference_action_sequence": reference_actions,
                "reference_pfv": None,
                "reference_tfv": None,
                "reference_peak": None,
            }
            if extra_predictor_context:
                reference_context.update(extra_predictor_context)
            reference_prediction = self.horizon_predictor(default_sequence, reference_context)
            # Candidate and no-control sequences are evaluated by the same
            # surrogate.  Compare the candidate upper envelope with the
            # reference upper envelope, rather than charging the candidate
            # for an uncertainty margin that is also present in the reference
            # prediction.  This preserves the safety gate while avoiding the
            # systematic false rejection seen in low-risk windows.
            reference_pfv = reference_prediction.get("pfv_upper", reference_prediction.get("pfv"))
            reference_tfv = reference_prediction.get("tfv_upper", reference_prediction.get("tfv"))
            reference_peak = reference_prediction.get(
                "peak_tfv_rate_upper", reference_prediction.get("peak_tfv_rate")
            )
            reference_source = "online_surrogate_default_sequence"
        ref_pfv = _as_horizon_array(reference_pfv, horizon_steps) if reference_pfv is not None else None
        ref_tfv = _as_horizon_array(reference_tfv, horizon_steps)
        ref_peak = _as_horizon_array(reference_peak, horizon_steps)
        reference_tfv_total = float(np.sum(np.maximum(0.0, ref_tfv)))
        tfv_tolerance = max(
            float(self.config.tfv_tolerance_abs),
            float(self.config.tfv_tolerance_frac) * reference_tfv_total,
        )
        reference_peak_value = float(np.max(np.maximum(0.0, ref_peak))) if ref_peak.size else 0.0
        peak_tolerance = max(
            float(self.config.peak_tolerance_abs),
            float(self.config.peak_tolerance_frac) * reference_peak_value,
        )
        ref_pfv_total = 0.0
        if ref_pfv is not None:
            ref_pfv_total = float(np.sum(np.maximum(0.0, ref_pfv)))
            required_pfv_improvement = max(
                float(self.config.min_pfv_improvement_abs),
                float(self.config.min_pfv_improvement_frac) * ref_pfv_total,
            )
            pfv_tolerance = max(
                float(self.config.pfv_tolerance_abs),
                float(self.config.pfv_tolerance_frac) * ref_pfv_total,
            )
        else:
            required_pfv_improvement = 0.0
            pfv_tolerance = 0.0
        required_tfv_improvement = max(
            float(self.config.tfv_required_reduction_abs),
            float(self.config.tfv_required_reduction_frac) * reference_tfv_total,
        )
        rainfall_sum = float(np.sum(np.maximum(0.0, rainfall))) if rainfall.size else 0.0
        rainfall_max = float(np.max(np.maximum(0.0, rainfall))) if rainfall.size else 0.0
        tfv_repair_multiplier = 1.0
        if rainfall_sum <= float(self.config.dry_rain_threshold) or rainfall_max <= float(self.config.dry_rain_threshold):
            tfv_repair_multiplier = float(self.config.tfv_required_reduction_dry_multiplier)
            required_tfv_improvement *= tfv_repair_multiplier
        adaptive_delta = float(self.config.max_candidate_delta)
        if self.config.adaptive_delta_enabled:
            low = float(self.config.pfv_low_risk_horizon_threshold)
            high = max(low + 1.0e-9, float(self.config.pfv_high_risk_horizon_threshold))
            if ref_pfv_total <= low:
                adaptive_delta = float(self.config.low_risk_max_candidate_delta)
            elif ref_pfv_total >= high:
                adaptive_delta = float(self.config.high_risk_max_candidate_delta)
            else:
                ratio = (ref_pfv_total - low) / (high - low)
                adaptive_delta = (
                    (1.0 - ratio) * float(self.config.low_risk_max_candidate_delta)
                    + ratio * float(self.config.high_risk_max_candidate_delta)
                )
            adaptive_delta = max(0.0, adaptive_delta)
        sequences = generate_action_sequences(
            current,
            self.actuators,
            horizon_steps=horizon_steps,
            max_delta=adaptive_delta,
            include_hold=True,
            priority_to_actuators=self.priority_to_actuators,
            max_sequences=self.config.max_candidate_sequences,
            group_limit=(
                self.runtime_candidate_group_limit
                if self.runtime_candidate_group_limit is not None
                else self.config.candidate_group_limit
            ),
            allowed_actuator_ids=(
                self.runtime_allowed_actuator_ids
                if self.runtime_allowed_actuator_ids is not None
                else self.config.allowed_actuator_ids
            ),
            blocked_actuator_ids=self.config.blocked_actuator_ids,
            reference_sequence=reference_actions,
            allowed_action_directions=(
                self.runtime_allowed_action_directions
                if self.runtime_allowed_action_directions is not None
                else self.allowed_action_directions
            ),
            empirical_hold_steps=(
                self.config.empirical_hold_steps
                if self.config.empirical_single_action_gate
                else 0
            ),
            allowed_action_delta_limits=self.runtime_allowed_action_delta_limits,
        )
        pending: list[dict[str, Any]] = []
        for candidate in sequences:
            seq, execution_projection = self._project_execution_sequence(candidate["sequence"], last)
            label = str(candidate.get("label", ""))
            first_step_delta = float(np.max(np.abs(seq[0] - last))) if seq.size and last.size else 0.0
            if label != "hold_native" and first_step_delta > float(self.config.max_first_step_delta) + 1e-9:
                continue
            context = {
                "label": label,
                "elapsed_min": float(elapsed_min),
                "phase": str(phase),
                "target_actuators": str(candidate.get("target_actuators", "")),
                "reconstructed_state": np.asarray(reconstructed_state, dtype=np.float32),
                "rainfall_window": rainfall,
                "current_action": current,
                "reference_action_sequence": reference_actions,
                "reference_pfv": ref_pfv,
                "reference_tfv": ref_tfv,
                "reference_peak": ref_peak,
            }
            if extra_predictor_context:
                context.update(extra_predictor_context)
            pending.append(
                {
                    "candidate": candidate,
                    "sequence": seq,
                    "label": label,
                    "first_step_delta": first_step_delta,
                    "execution_projection": execution_projection,
                    "context": context,
                }
            )
        if not pending:
            return current, {
                "policy_id": "proposed_gat_mpc",
                "constraint_reference_source": reference_source,
                "objective_mode": self.config.objective_mode,
                "fallback_to_default": True,
                "intervention_reason": "no_action_sequences",
                "candidate_sequence_count": 0,
                "candidate_sequence_cap": int(self.config.max_candidate_sequences),
                "candidate_group_limit": int(self.config.candidate_group_limit),
            }

        predict_many = getattr(self.horizon_predictor, "predict_many", None)
        if callable(predict_many):
            preds = predict_many(
                [item["sequence"] for item in pending],
                [item["context"] for item in pending],
            )
        else:
            preds = [self.horizon_predictor(item["sequence"], item["context"]) for item in pending]

        scored: list[dict[str, Any]] = []
        for item, pred in zip(pending, preds):
            seq = item["sequence"]
            candidate = item["candidate"]
            label = str(item["label"])
            if self.config.objective_mode == "pfv_preserving_system_repair":
                score = score_horizon_system_repair_sequence(
                    pfv=pred.get("pfv_upper", pred.get("pfv")),
                    tfv=pred.get("tfv_upper", pred.get("tfv")),
                    peak_tfv_rate=pred.get("peak_tfv_rate_upper", pred.get("peak_tfv_rate")),
                    action_change=self._action_change(seq, last),
                    reference_pfv=ref_pfv,
                    reference_tfv=ref_tfv,
                    reference_peak=ref_peak,
                    smooth_weight=self.config.smooth_weight,
                    violation_penalty=self.config.violation_penalty,
                    pfv_tolerance=pfv_tolerance,
                    tfv_required_improvement=required_tfv_improvement,
                    tfv_tolerance=tfv_tolerance,
                    peak_tolerance=peak_tolerance,
                    peak_weight=self.config.peak_weight,
                    pfv_weight=self.config.pfv_weight,
                    tfv_hard_constraint=self.config.tfv_hard_constraint,
                )
            else:
                score = score_horizon_sequence(
                    pfv=pred.get("pfv_upper", pred.get("pfv")),
                    tfv=pred.get("tfv_upper", pred.get("tfv")),
                    peak_tfv_rate=pred.get("peak_tfv_rate_upper", pred.get("peak_tfv_rate")),
                    action_change=self._action_change(seq, last),
                    reference_pfv=ref_pfv,
                    reference_tfv=ref_tfv,
                    reference_peak=ref_peak,
                    smooth_weight=self.config.smooth_weight,
                    violation_penalty=self.config.violation_penalty,
                    pfv_required_improvement=required_pfv_improvement,
                    tfv_tolerance=tfv_tolerance,
                    peak_tolerance=peak_tolerance,
                )
            gate_mode = "uncertainty_upper_bound"
            target_text = str(candidate.get("target_actuators", ""))
            target_count = len([x for x in target_text.split(",") if x.strip()])
            # Exact local replay proves only a single actuator/direction.  It
            # may therefore relax the global effect margin for a one-actuator
            # candidate, but never for an unverified joint action.
            if (
                self.config.empirical_single_action_gate
                and bool(empirical_local_verified)
                and target_count == 1
                and label.startswith("empirical_relative_hold|")
            ):
                point_score = (
                    score_horizon_system_repair_sequence(
                        pfv=pred.get("pfv"),
                        tfv=pred.get("tfv"),
                        peak_tfv_rate=pred.get("peak_tfv_rate"),
                        action_change=self._action_change(seq, last),
                        reference_pfv=ref_pfv,
                        reference_tfv=ref_tfv,
                        reference_peak=ref_peak,
                        smooth_weight=self.config.smooth_weight,
                        violation_penalty=self.config.violation_penalty,
                        pfv_tolerance=pfv_tolerance,
                        tfv_required_improvement=required_tfv_improvement,
                        tfv_tolerance=tfv_tolerance,
                        peak_tolerance=peak_tolerance,
                        peak_weight=self.config.peak_weight,
                        pfv_weight=self.config.pfv_weight,
                        tfv_hard_constraint=self.config.tfv_hard_constraint,
                    )
                    if self.config.objective_mode == "pfv_preserving_system_repair"
                    else score_horizon_sequence(
                        pfv=pred.get("pfv"),
                        tfv=pred.get("tfv"),
                        peak_tfv_rate=pred.get("peak_tfv_rate"),
                        action_change=self._action_change(seq, last),
                        reference_pfv=ref_pfv,
                        reference_tfv=ref_tfv,
                        reference_peak=ref_peak,
                        smooth_weight=self.config.smooth_weight,
                        violation_penalty=self.config.violation_penalty,
                        pfv_required_improvement=required_pfv_improvement,
                        tfv_tolerance=tfv_tolerance,
                        peak_tolerance=peak_tolerance,
                    )
                )
                # The local point gate is used to recover effective actions,
                # but it must not spend the deployment PFV tolerance. The
                # tolerance remains available to the joint uncertainty gate;
                # empirical single-action acceptance requires strict PFV
                # non-inferiority against the same online reference.
                strict_pfv_noninferior = (
                    float(point_score.pfv_total)
                    <= float(point_score.reference_pfv_total) + 1.0e-9
                )
                if point_score.gate_pass and strict_pfv_noninferior:
                    score = point_score
                    gate_mode = "empirical_single_point_effect"
                elif label.startswith("empirical_relative_hold|") and self.runtime_verified_action_effects:
                    parts = {
                        piece.split("=", 1)[0]: piece.split("=", 1)[1]
                        for piece in label.split("|")
                        if "=" in piece
                    }
                    aid = str(parts.get("actuator", ""))
                    try:
                        signed_delta = float(parts.get("d", "0"))
                    except ValueError:
                        signed_delta = 0.0
                    direction = "increase" if signed_delta > 0.0 else "decrease"
                    records = (
                        (self.runtime_verified_action_effects.get(aid, {}) or {}).get(direction, [])
                    )
                    exact = next(
                        (
                            item for item in records
                            if abs(float(item.get("delta_abs", 0.0)) - abs(signed_delta)) <= 1.0e-5
                        ),
                        None,
                    )
                    if exact is not None:
                        local_pfv = float(ref_pfv_total) + float(exact["effect_PFV_H"])
                        local_tfv = float(reference_tfv_total) + float(exact["effect_TFV_H"])
                        local_peak = float(reference_peak_value) + float(exact["effect_peak_TFV_rate_H"])
                        local_pfv_ok = local_pfv <= float(ref_pfv_total) + 1.0e-9
                        local_tfv_ok = local_tfv <= float(reference_tfv_total) + float(tfv_tolerance) + 1.0e-9
                        local_peak_ok = local_peak <= float(reference_peak_value) + float(peak_tolerance) + 1.0e-9
                        if local_pfv_ok and local_tfv_ok and local_peak_ok:
                            score = replace(
                                point_score,
                                score=float(
                                    self.config.pfv_weight * local_pfv
                                    + local_tfv
                                    + self.config.peak_weight * local_peak
                                    + point_score.smooth_term
                                ),
                                gate_pass=True,
                                pfv_total=local_pfv,
                                tfv_total=local_tfv,
                                peak_tfv_rate=local_peak,
                                pfv_violation=0.0,
                                tfv_violation=0.0,
                                peak_violation=0.0,
                                penalty_term=0.0,
                            )
                            gate_mode = "empirical_exact_replay_effect"
            online_future_hydraulics_used = bool(
                float(np.asarray(pred.get("online_future_hydraulics_used", [0.0]), dtype=float).reshape(-1)[0]) > 0.5
            )
            first_delta_vector = np.asarray(seq[0], dtype=float) - np.asarray(reference_actions[0], dtype=float)
            deadband = max(0.0, float((extra_predictor_context or {}).get("action_setting_deadband", 0.0)))
            active_mask = np.abs(first_delta_vector) > deadband
            changed_facilities = int(active_mask.sum())
            total_variation = float(np.sum(np.abs(first_delta_vector[active_mask]))) if changed_facilities else 0.0
            seq_delta = np.asarray(seq, dtype=float) - np.asarray(reference_actions, dtype=float)
            signs = np.sign(np.where(np.abs(seq_delta) > deadband, seq_delta, 0.0))
            reversal_count = 0
            if signs.shape[0] > 1:
                reversal_count = int(np.sum((signs[1:] * signs[:-1]) < 0.0))
            changed_penalty = max(0.0, float((extra_predictor_context or {}).get("changed_facility_penalty", 1.0)))
            variation_penalty = max(0.0, float((extra_predictor_context or {}).get("variation_penalty", 1.0)))
            reversal_penalty = max(0.0, float((extra_predictor_context or {}).get("reversal_penalty", 5.0)))
            action_cost = (
                changed_penalty * float(changed_facilities)
                + variation_penalty * total_variation
                + reversal_penalty * float(reversal_count)
            )
            tfv_benefit = max(0.0, float(score.reference_tfv_total) - float(score.tfv_total))
            peak_benefit = max(0.0, float(score.reference_peak_tfv_rate) - float(score.peak_tfv_rate))
            material_benefit = tfv_benefit + max(0.0, float(self.config.peak_weight)) * peak_benefit
            benefit_cost_ratio = material_benefit / max(action_cost, 1.0e-12) if action_cost > 0.0 else (float("inf") if material_benefit > 0.0 else 0.0)
            min_benefit = max(0.0, float((extra_predictor_context or {}).get("minimum_material_benefit", 0.0)))
            min_ratio = max(0.0, float((extra_predictor_context or {}).get("minimum_benefit_cost_ratio", 0.0)))
            adaptive_k_limit = int((extra_predictor_context or {}).get("adaptive_k_limit", self.config.candidate_group_limit))
            hard_reasons: list[str] = []
            is_hold = label == "hold_native"
            if not is_hold:
                if online_future_hydraulics_used:
                    hard_reasons.append("online_future_hydraulic_truth_forbidden")
                if changed_facilities > max(0, adaptive_k_limit):
                    hard_reasons.append("adaptive_k_exceeded")
                if material_benefit + 1.0e-12 < min_benefit:
                    hard_reasons.append("benefit_below_material_threshold")
                if action_cost > 0.0 and benefit_cost_ratio + 1.0e-12 < min_ratio:
                    hard_reasons.append("benefit_cost_ratio_too_low")
            v4_hard_gate_pass = is_hold or not hard_reasons
            scored.append(
                {
                    "label": label,
                    "sequence": seq,
                    "score": score,
                    "physical_rationale": str(candidate.get("physical_rationale", "")),
                    "target_actuators": str(candidate.get("target_actuators", "")),
                    "first_step_delta": float(item["first_step_delta"]),
                    "execution_projection": dict(item["execution_projection"]),
                    "uncertainty_margin": np.asarray(pred.get("uncertainty_margin", []), dtype=float),
                    "event_pfv_upper": float(np.asarray(pred.get("event_pfv_upper", [np.nan]), dtype=float).reshape(-1)[0]),
                    "event_pfv_mean": float(np.asarray(pred.get("event_pfv", [np.nan]), dtype=float).reshape(-1)[0]),
                    "online_future_hydraulics_used": online_future_hydraulics_used,
                    "adaptive_k_limit": int(adaptive_k_limit),
                    "changed_facilities": int(changed_facilities),
                    "total_absolute_setting_variation": float(total_variation),
                    "reversal_count": int(reversal_count),
                    "predicted_material_benefit": float(material_benefit),
                    "predicted_action_cost": float(action_cost),
                    "predicted_benefit_cost_ratio": float(benefit_cost_ratio),
                    "v4_hard_gate_pass": bool(v4_hard_gate_pass),
                    "v4_hard_gate_reasons": hard_reasons,
                    "gate_mode": gate_mode,
                }
            )
        hold = next((item for item in scored if str(item.get("label", "")) == "hold_native"), None)
        safe_scored = [
            item for item in scored
            if bool(item["score"].gate_pass) and bool(item.get("v4_hard_gate_pass", True))
        ]
        if safe_scored:
            best = min(safe_scored, key=lambda item: item["score"].score)
            no_safe_sequence = False
        else:
            best = hold if hold is not None else min(scored, key=lambda item: item["score"].score)
            no_safe_sequence = True
        non_hold = [item for item in scored if str(item.get("label", "")) != "hold_native"]
        best_non_hold = min(non_hold, key=lambda item: item["score"].score) if non_hold else None
        best_score = best["score"]
        action = np.asarray(best["sequence"][0], dtype=np.float32)
        self._record_executed_action(action, last)
        fallback = no_safe_sequence or str(best["label"]) == "hold_native"
        info = {
            "policy_id": "proposed_gat_mpc",
            "constraint_reference_source": reference_source,
            "objective_mode": self.config.objective_mode,
            "fallback_to_default": bool(fallback),
            "intervention_reason": "no_safe_sequence"
            if no_safe_sequence
            else ("hold_current_action" if fallback else "sequence_passed_horizon_objective"),
            "selected_sequence_label": str(best["label"]),
            "selected_action_semantics": str((extra_predictor_context or {}).get("action_semantics", "absolute_from_reference_sequence")),
            "selected_residual_action_max": float(np.max(np.abs(best["sequence"] - reference_actions))),
            "selected_sequence_first_action": action.astype(float).tolist(),
            "candidate_sequence_count": int(len(scored)),
            "candidate_sequence_cap": int(self.config.max_candidate_sequences),
            "candidate_group_limit": int(self.config.candidate_group_limit),
            "adaptive_candidate_delta": float(adaptive_delta),
            "adaptive_delta_enabled": bool(self.config.adaptive_delta_enabled),
            "selected_horizon_objective_score": float(best_score.score),
            "selected_gate_pass": bool(best_score.gate_pass and best.get("v4_hard_gate_pass", True)),
            "selected_v4_hard_gate_pass": bool(best.get("v4_hard_gate_pass", True)),
            "selected_v4_hard_gate_reasons": ";".join(best.get("v4_hard_gate_reasons", [])),
            "selected_adaptive_k_limit": int(best.get("adaptive_k_limit", self.config.candidate_group_limit)),
            "selected_changed_facilities": int(best.get("changed_facilities", 0)),
            "selected_total_absolute_setting_variation": float(best.get("total_absolute_setting_variation", 0.0)),
            "selected_reversal_count": int(best.get("reversal_count", 0)),
            "selected_predicted_material_benefit": float(best.get("predicted_material_benefit", 0.0)),
            "selected_predicted_action_cost": float(best.get("predicted_action_cost", 0.0)),
            "selected_predicted_benefit_cost_ratio": float(best.get("predicted_benefit_cost_ratio", 0.0)),
            "selected_pfv_horizon": float(best_score.pfv_total),
            "selected_reference_pfv_horizon": float(best_score.reference_pfv_total),
            "selected_tfv_horizon": float(best_score.tfv_total),
            "selected_reference_tfv_horizon": float(best_score.reference_tfv_total),
            "selected_tfv_tolerance": float(tfv_tolerance),
            "selected_peak_tfv_rate": float(best_score.peak_tfv_rate),
            "selected_reference_peak_tfv_rate": float(best_score.reference_peak_tfv_rate),
            "selected_pfv_violation": float(best_score.pfv_violation),
            "required_pfv_improvement": float(required_pfv_improvement),
            "pfv_tolerance": float(pfv_tolerance),
            "required_tfv_improvement": float(required_tfv_improvement),
            "tfv_hard_constraint": bool(self.config.tfv_hard_constraint),
            "tfv_repair_multiplier": float(tfv_repair_multiplier),
            "rainfall_window_sum": float(rainfall_sum),
            "rainfall_window_max": float(rainfall_max),
            "selected_tfv_violation": float(best_score.tfv_violation),
            "selected_peak_violation": float(best_score.peak_violation),
            "selected_pfv_improvement": float(best_score.reference_pfv_total - best_score.pfv_total),
            "selected_peak_tolerance": float(peak_tolerance),
            "selected_event_pfv_upper": float(best.get("event_pfv_upper", np.nan)),
            "selected_event_pfv_mean": float(best.get("event_pfv_mean", np.nan)),
            "online_future_hydraulics_used": bool(best.get("online_future_hydraulics_used", False)),
            "selected_first_step_delta": float(best.get("first_step_delta", 0.0)),
            "selected_ramp_clipped_values": int(best.get("execution_projection", {}).get("ramp_clipped_values", 0)),
            "selected_dwell_held_values": int(best.get("execution_projection", {}).get("dwell_held_values", 0)),
            "selected_action_change_penalty": float(best_score.smooth_term),
            "value_model_source": self.predictor_source,
            "selected_gate_mode": str(best.get("gate_mode", "uncertainty_upper_bound")),
            "target_actuators": str(best.get("target_actuators", "")),
            "uncertainty_margin_max": float(np.max(best["uncertainty_margin"]))
            if np.asarray(best.get("uncertainty_margin", [])).size
            else 0.0,
        }
        if best_non_hold is None:
            info.update(
                {
                    "best_non_hold_sequence_label": "",
                    "best_non_hold_gate_pass": False,
                    "best_non_hold_horizon_objective_score": np.nan,
                    "best_non_hold_pfv_horizon": np.nan,
                    "best_non_hold_reference_pfv_horizon": np.nan,
                    "best_non_hold_tfv_horizon": np.nan,
                    "best_non_hold_reference_tfv_horizon": np.nan,
                    "best_non_hold_peak_tfv_rate": np.nan,
                    "best_non_hold_reference_peak_tfv_rate": np.nan,
                    "best_non_hold_pfv_violation": np.nan,
                    "best_non_hold_tfv_violation": np.nan,
                    "best_non_hold_peak_violation": np.nan,
                    "best_non_hold_pfv_improvement": np.nan,
                    "best_non_hold_first_step_delta": np.nan,
                    "best_non_hold_action_change_penalty": np.nan,
                    "best_non_hold_target_actuators": "",
                }
            )
        else:
            non_hold_score = best_non_hold["score"]
            info.update(
                {
                    "best_non_hold_sequence_label": str(best_non_hold.get("label", "")),
                    "best_non_hold_gate_pass": bool(non_hold_score.gate_pass),
                    "best_non_hold_horizon_objective_score": float(non_hold_score.score),
                    "best_non_hold_pfv_horizon": float(non_hold_score.pfv_total),
                    "best_non_hold_reference_pfv_horizon": float(non_hold_score.reference_pfv_total),
                    "best_non_hold_tfv_horizon": float(non_hold_score.tfv_total),
                    "best_non_hold_reference_tfv_horizon": float(non_hold_score.reference_tfv_total),
                    "best_non_hold_peak_tfv_rate": float(non_hold_score.peak_tfv_rate),
                    "best_non_hold_reference_peak_tfv_rate": float(non_hold_score.reference_peak_tfv_rate),
                    "best_non_hold_pfv_violation": float(non_hold_score.pfv_violation),
                    "best_non_hold_tfv_violation": float(non_hold_score.tfv_violation),
                    "best_non_hold_peak_violation": float(non_hold_score.peak_violation),
                    "best_non_hold_pfv_improvement": float(non_hold_score.reference_pfv_total - non_hold_score.pfv_total),
                    "best_non_hold_first_step_delta": float(best_non_hold.get("first_step_delta", 0.0)),
                    "best_non_hold_action_change_penalty": float(non_hold_score.smooth_term),
                    "best_non_hold_target_actuators": str(best_non_hold.get("target_actuators", "")),
                }
            )
        return action, info
