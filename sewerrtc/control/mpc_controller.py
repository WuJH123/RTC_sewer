from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch

from sewerrtc.models.graph_surrogate import PhysicsGuidedTemporalGraphSurrogate
from sewerrtc.models.residual_value import ResidualValuePredictor, build_residual_feature_dict
from .candidate_generator import candidate_metadata_features, generate_labeled_candidates, parse_candidate_label
from .horizon_objective import score_horizon_candidate
from .nominal_policy import nominal_safe_action
from .policy_base import risk_class_from_pfv
from .safety_guards import (
    candidate_boundary_decision,
    should_block_low_risk_takeover,
    should_cancel_held_action_in_low_risk,
)


def _delta_tier(delta: float) -> str:
    d = abs(float(delta))
    if d <= 0.080001:
        return "small"
    if d <= 0.160001:
        return "medium"
    return "large"


def _boolish(value) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "y", "allow", "allowed"}


def _template_family(template: str) -> str:
    t = str(template).lower()
    if "boost" in t or "release" in t or "empty" in t:
        return "release_or_boost"
    if "throttle" in t or "restrict" in t or "retain" in t:
        return "protective"
    return "neutral"


def _parse_int_tuple(value, default: tuple[int, ...]) -> tuple[int, ...]:
    if value is None:
        return default
    if isinstance(value, str):
        raw = [x.strip() for x in value.split(",")]
    else:
        raw = list(value)
    vals = []
    for item in raw:
        try:
            iv = int(float(item))
        except Exception:
            continue
        if iv > 0 and iv not in vals:
            vals.append(iv)
    return tuple(vals) or default


class PFVFirstMPC:
    def __init__(
        self,
        model_path: str | Path | None,
        actuators: pd.DataFrame,
        n_nodes: int,
        device: str = "cpu",
        pfv_min_improve: float = 1.0,
        tfv_guard_pct: float = 0.005,
        peak_guard_pct: float = 0.010,
        priority_node_indices: Optional[list[int]] = None,
        priority_depth_trigger: float = 2.00,
        rain_priority_depth_trigger: float = 2.00,
        rain_trigger_mm_h: float = 20.0,
        pfv_prob_min: float = 0.55,
        safe_prob_min: float = 0.60,
        pfv_nonzero_prob_min: float = 0.40,
        max_candidate_delta: float = 0.08,
        min_control_interval_steps: int = 2,
        residual_value_path: str | Path | None = None,
        residual_pfv_prob_min: float = 0.60,
        residual_safe_prob_min: float = 0.70,
        residual_nonzero_prob_min: float = 0.45,
        residual_peak_prob_min: float = 0.60,
        topk_log_count: int = 8,
        max_candidate_count: int = 96,
        candidate_hold_steps: tuple[int, ...] | list[int] | str | None = None,
        allowed_candidate_templates: set[str] | list[str] | tuple[str, ...] | str | None = None,
        blocked_candidate_templates: set[str] | list[str] | tuple[str, ...] | str | None = None,
        allowed_candidate_scopes_by_template: dict[str, set[str] | list[str] | tuple[str, ...] | str] | None = None,
        priority_upstream_nodes: Optional[set[str]] = None,
        priority_downstream_nodes: Optional[set[str]] = None,
        empirical_guard_path: str | Path | None = None,
        empirical_guard_unknown_allow: bool = True,
        boost_safe_prob_extra: float = 0.12,
        boost_peak_prob_extra: float = 0.10,
        protective_safe_prob_relief: float = 0.05,
        release_peak_hold_max: int = 1,
        event_id: str = "",
        nominal_pfv_reference: float = 0.0,
        low_risk_pfv_threshold: float = 1000.0,
        high_risk_pfv_threshold: float = 20000.0,
        release_recession_pfv_min: float = 500.0,
        release_recession_priority_depth_min: float = 1.0,
        strict_guard_return_period_max: int = 15,
        strict_guard_patterns: str | list[str] | tuple[str, ...] = "chicago_late,block,double_peak",
        strict_guard_prob_extra: float = 0.10,
        horizon_smooth_weight: float = 0.05,
        horizon_violation_penalty: float = 1.0e6,
    ):
        self.actuators = actuators.reset_index(drop=True)
        self.device = torch.device(device if device == "cuda" and torch.cuda.is_available() else "cpu")
        self.model = None
        self.node_static = None
        self.edge_index = None
        self.action_node_map = None
        self.risk_delta_scale = None
        self.horizon_steps = 1
        self.n_nodes = n_nodes
        self.pfv_min_improve = pfv_min_improve
        self.tfv_guard_pct = tfv_guard_pct
        self.peak_guard_pct = peak_guard_pct
        self.priority_node_indices = list(priority_node_indices or [])
        self.priority_depth_trigger = float(priority_depth_trigger)
        self.rain_priority_depth_trigger = float(rain_priority_depth_trigger)
        self.rain_trigger_mm_h = float(rain_trigger_mm_h)
        self.pfv_prob_min = float(pfv_prob_min)
        self.safe_prob_min = float(safe_prob_min)
        self.pfv_nonzero_prob_min = float(pfv_nonzero_prob_min)
        self.max_candidate_delta = float(max_candidate_delta)
        self.min_control_interval_steps = max(0, int(min_control_interval_steps))
        self.residual_predictor = None
        self.residual_pfv_prob_min = float(residual_pfv_prob_min)
        self.residual_safe_prob_min = float(residual_safe_prob_min)
        self.residual_nonzero_prob_min = float(residual_nonzero_prob_min)
        self.residual_peak_prob_min = float(residual_peak_prob_min)
        self.topk_log_count = max(0, int(topk_log_count))
        self.max_candidate_count = max(1, int(max_candidate_count))
        self.candidate_hold_steps = _parse_int_tuple(candidate_hold_steps, (1, 2, 3))
        self.allowed_candidate_templates = allowed_candidate_templates
        self.blocked_candidate_templates = blocked_candidate_templates
        self.allowed_candidate_scopes_by_template = allowed_candidate_scopes_by_template
        self.priority_upstream_nodes = set(priority_upstream_nodes or set())
        self.priority_downstream_nodes = set(priority_downstream_nodes or set())
        self.empirical_guard = None
        self.empirical_guard_unknown_allow = bool(empirical_guard_unknown_allow)
        self.boost_safe_prob_extra = float(boost_safe_prob_extra)
        self.boost_peak_prob_extra = float(boost_peak_prob_extra)
        self.protective_safe_prob_relief = float(protective_safe_prob_relief)
        self.release_peak_hold_max = max(1, int(release_peak_hold_max))
        self.event_id = str(event_id or "")
        self.nominal_pfv_reference = float(nominal_pfv_reference or 0.0)
        self.low_risk_pfv_threshold = float(low_risk_pfv_threshold)
        self.high_risk_pfv_threshold = float(high_risk_pfv_threshold)
        self.release_recession_pfv_min = float(release_recession_pfv_min)
        self.release_recession_priority_depth_min = float(release_recession_priority_depth_min)
        self.strict_guard_return_period_max = int(strict_guard_return_period_max)
        if isinstance(strict_guard_patterns, str):
            self.strict_guard_patterns = tuple(p.strip().lower() for p in strict_guard_patterns.split(",") if p.strip())
        else:
            self.strict_guard_patterns = tuple(str(p).strip().lower() for p in strict_guard_patterns if str(p).strip())
        self.strict_guard_prob_extra = float(strict_guard_prob_extra)
        self.horizon_smooth_weight = float(horizon_smooth_weight)
        self.horizon_violation_penalty = float(horizon_violation_penalty)
        self._cooldown = 0
        self._hold_remaining = 0
        self._hold_action: np.ndarray | None = None
        self._hold_label = ""
        if empirical_guard_path and Path(empirical_guard_path).exists():
            guard = pd.read_csv(empirical_guard_path)
            for col in ["template_name", "candidate_scope", "residual_delta_tier", "group_level"]:
                if col not in guard:
                    guard[col] = "*"
                guard[col] = guard[col].fillna("*").astype(str)
            if "hold_steps" not in guard:
                guard["hold_steps"] = "*"
            guard["hold_steps_norm"] = guard["hold_steps"].map(self._normalise_hold_key)
            if "empirical_allow" not in guard:
                guard["empirical_allow"] = True
            self.empirical_guard = guard
        if residual_value_path and Path(residual_value_path).exists():
            self.residual_predictor = ResidualValuePredictor(residual_value_path, device=str(self.device))
            # Do not force the validation-calibrated threshold onto online
            # closed-loop candidates. In practice the online candidate feature
            # distribution can be more conservative than the residual training
            # set; forcing the calibrated threshold can silently reject every
            # action. The caller controls the deployment threshold explicitly.
            self.residual_calibrated_safe_threshold = float(
                getattr(self.residual_predictor, "safe_threshold", self.residual_safe_prob_min)
            )
        if model_path and Path(model_path).exists():
            ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
            n_actions = int(ckpt.get("n_actions", len(actuators)))
            hidden = int(ckpt.get("hidden_dim", 256))
            static_dim = int(ckpt.get("static_dim", 7))
            self.horizon_steps = int(ckpt.get("horizon_steps", 1))
            heads = int(ckpt.get("gat_heads", 4))
            self.model = PhysicsGuidedTemporalGraphSurrogate(n_nodes, n_actions, static_dim, self.horizon_steps, hidden, heads).to(self.device)
            self.model.load_state_dict(ckpt["model"])
            self.model.eval()
            self.node_static = torch.tensor(ckpt["node_static"], dtype=torch.float32, device=self.device)
            self.edge_index = torch.tensor(ckpt["edge_index"], dtype=torch.long, device=self.device)
            self.action_node_map = torch.tensor(ckpt["action_node_map"], dtype=torch.float32, device=self.device)
            self.risk_delta_scale = torch.tensor(
                ckpt.get("risk_delta_scale", np.ones(3, dtype=np.float32)),
                dtype=torch.float32,
                device=self.device,
            )

    def _current_priority_risk_class(self, priority_depth_max: float, rainfall_mm_h: float) -> str:
        depth = float(priority_depth_max or 0.0)
        rain = float(rainfall_mm_h or 0.0)
        if depth >= self.priority_depth_trigger:
            return "high_risk_state"
        if rain >= self.rain_trigger_mm_h and depth >= self.rain_priority_depth_trigger:
            return "high_risk_state"
        if depth >= 0.5 * self.priority_depth_trigger or rain >= 0.5 * self.rain_trigger_mm_h:
            return "medium_risk_state"
        return "low_risk_state"

    def _event_risk_class(self) -> str:
        return risk_class_from_pfv(
            self.nominal_pfv_reference,
            low_threshold=self.low_risk_pfv_threshold,
            high_threshold=self.high_risk_pfv_threshold,
        )

    def _history_meta(
        self,
        *,
        event_risk_class: str,
        current_priority_risk_class: str,
        intervention_allowed: bool,
        intervention_reason: str,
        predicted_pfv_gain: float = 0.0,
        predicted_pfv_gain_abs: float = 0.0,
        native_expected_pfv: float | None = None,
        candidate_expected_pfv: float | None = None,
        low_risk_false_intervention: bool = False,
        false_intervention_reason: str = "",
        delta_PFV_p50: float | None = None,
        delta_PFV_p90: float | None = None,
        delta_TFV_p90: float | None = None,
        delta_peak_p90: float | None = None,
        uncertainty_score: float | None = None,
        uncertainty_gate_pass: bool = False,
        uncertainty_gate_reason: str = "not_evaluated",
    ) -> dict:
        return {
            "event_risk_class": str(event_risk_class),
            "current_priority_risk_class": str(current_priority_risk_class),
            "intervention_allowed": bool(intervention_allowed),
            "intervention_reason": str(intervention_reason),
            "predicted_pfv_gain": float(predicted_pfv_gain or 0.0),
            "predicted_pfv_gain_abs": float(predicted_pfv_gain_abs or 0.0),
            "native_expected_pfv": float(native_expected_pfv) if native_expected_pfv is not None else float("nan"),
            "candidate_expected_pfv": float(candidate_expected_pfv) if candidate_expected_pfv is not None else float("nan"),
            "low_risk_false_intervention": bool(low_risk_false_intervention),
            "false_intervention_reason": str(false_intervention_reason),
            "delta_PFV_p50": float(delta_PFV_p50) if delta_PFV_p50 is not None else float("nan"),
            "delta_PFV_p90": float(delta_PFV_p90) if delta_PFV_p90 is not None else float("nan"),
            "delta_TFV_p90": float(delta_TFV_p90) if delta_TFV_p90 is not None else float("nan"),
            "delta_peak_p90": float(delta_peak_p90) if delta_peak_p90 is not None else float("nan"),
            "uncertainty_score": float(uncertainty_score) if uncertainty_score is not None else float("nan"),
            "uncertainty_gate_pass": bool(uncertainty_gate_pass),
            "uncertainty_gate_reason": str(uncertainty_gate_reason),
        }

    @staticmethod
    def _normalise_hold_key(value) -> str:
        text = str(value).strip()
        if text in {"", "*", "nan", "None"}:
            return "*"
        try:
            return str(int(round(float(text))))
        except Exception:
            return text

    def _empirical_guard_check(self, candidate_label: str) -> tuple[bool, str, dict]:
        if self.empirical_guard is None or self.empirical_guard.empty:
            return True, "", {}
        meta = parse_candidate_label(candidate_label)
        template = str(meta.get("template", "unknown"))
        scope = str(meta.get("scope", "all"))
        delta = float(meta.get("delta", 0.0) or 0.0)
        tier = _delta_tier(delta)
        hold = str(int(meta.get("hold_steps", 1) or 1))
        lookups = [
            ("template_scope_tier_hold", template, scope, tier, hold),
            ("template_scope_tier", template, scope, tier, "*"),
            ("template_tier", template, "*", tier, "*"),
            ("template", template, "*", "*", "*"),
        ]
        guard = self.empirical_guard
        for level, tmpl, scp, tr, hld in lookups:
            m = (
                guard["group_level"].eq(level)
                & guard["template_name"].eq(tmpl)
                & guard["candidate_scope"].eq(scp)
                & guard["residual_delta_tier"].eq(tr)
                & guard["hold_steps_norm"].eq(hld)
            )
            if not m.any():
                continue
            row = guard.loc[m].sort_values(["n", "pfv_improve_safe_frac"], ascending=[False, False]).iloc[0]
            allow = _boolish(row.get("empirical_allow", True))
            stats = {
                "empirical_group_level": str(row.get("group_level", "")),
                "empirical_n": float(row.get("n", 0.0) or 0.0),
                "empirical_events": float(row.get("events", 0.0) or 0.0),
                "empirical_pfv_improve_safe_frac": float(row.get("pfv_improve_safe_frac", 0.0) or 0.0),
                "empirical_pfv_worse_frac": float(row.get("pfv_worse_frac", 1.0) or 1.0),
                "empirical_safe_guarded_frac": float(row.get("safe_guarded_frac", 0.0) or 0.0),
                "empirical_peak_worse_frac": float(row.get("peak_worse_frac", 1.0) or 1.0),
            }
            if allow:
                return True, "", stats
            reason = str(row.get("empirical_block_reason", "empirical_guard"))
            return False, reason or "empirical_guard", stats
        return self.empirical_guard_unknown_allow, "empirical_unknown" if not self.empirical_guard_unknown_allow else "", {}

    def _candidate_thresholds(
        self,
        candidate_label: str,
        phase: str,
        residual_mode: bool,
        priority_depth_max: float,
        rainfall_mm_h: float,
    ) -> tuple[float, float, float, float, bool, str, str, dict]:
        if residual_mode:
            pfv_prob_min = self.residual_pfv_prob_min
            safe_prob_min = self.residual_safe_prob_min
            pfv_nonzero_prob_min = self.residual_nonzero_prob_min
            peak_prob_min = self.residual_peak_prob_min
        else:
            pfv_prob_min = self.pfv_prob_min
            safe_prob_min = self.safe_prob_min
            pfv_nonzero_prob_min = self.pfv_nonzero_prob_min
            peak_prob_min = 0.0
        meta = parse_candidate_label(candidate_label)
        template = str(meta.get("template", "unknown"))
        hold_steps = int(meta.get("hold_steps", 1) or 1)
        family = _template_family(template)
        phase_rule_ok = True
        phase_rule_reason = ""
        if family == "release_or_boost":
            safe_prob_min = min(0.98, safe_prob_min + self.boost_safe_prob_extra)
            peak_prob_min = min(0.98, peak_prob_min + self.boost_peak_prob_extra)
            if str(phase) != "recession" and hold_steps > self.release_peak_hold_max:
                phase_rule_ok = False
                phase_rule_reason = f"release_boost_hold>{self.release_peak_hold_max}_outside_recession"
        elif family == "protective":
            safe_prob_min = max(0.0, safe_prob_min - self.protective_safe_prob_relief)
            peak_prob_min = max(0.0, peak_prob_min - self.protective_safe_prob_relief)
        boundary = candidate_boundary_decision(
            candidate_label=candidate_label,
            event_id=self.event_id,
            phase=phase,
            nominal_pfv_reference=self.nominal_pfv_reference,
            priority_depth_max=float(priority_depth_max),
            rainfall_mm_h=float(rainfall_mm_h),
            release_recession_pfv_min=self.release_recession_pfv_min,
            release_recession_priority_depth_min=self.release_recession_priority_depth_min,
            strict_guard_return_period_max=self.strict_guard_return_period_max,
            strict_guard_patterns=self.strict_guard_patterns,
            strict_guard_prob_extra=self.strict_guard_prob_extra,
        )
        safe_prob_min = min(0.98, safe_prob_min + float(boundary.safe_prob_extra))
        peak_prob_min = min(0.98, peak_prob_min + float(boundary.peak_prob_extra))
        if not boundary.allowed:
            phase_rule_ok = False
            phase_rule_reason = boundary.reason or "action_boundary"
        return (
            float(pfv_prob_min),
            float(safe_prob_min),
            float(pfv_nonzero_prob_min),
            float(peak_prob_min),
            phase_rule_ok,
            phase_rule_reason,
            family,
            {
                "boundary_cautious_event": bool(boundary.cautious_event),
                "boundary_safe_prob_extra": float(boundary.safe_prob_extra),
                "boundary_peak_prob_extra": float(boundary.peak_prob_extra),
                "boundary_reason": boundary.reason,
            },
        )

    def choose(
        self,
        state: np.ndarray,
        rainfall_mm_h: float,
        phase: str,
        baseline_tfv_rate: float = 1.0,
        baseline_peak: float = 1.0,
        nominal_action: Optional[np.ndarray] = None,
        elapsed_min: float = 0.0,
    ) -> tuple[np.ndarray, dict]:
        if nominal_action is None:
            nominal = nominal_safe_action(self.actuators, phase, rainfall_mm_h)
            nominal_source = "generic_nominal"
        else:
            nominal = np.asarray(nominal_action, dtype=np.float32).reshape(-1)
            if len(nominal) != len(self.actuators):
                nominal = np.resize(nominal, len(self.actuators)).astype(np.float32)
            nominal = np.clip(nominal, 0.0, 1.0)
            nominal_source = "internal_rule_nominal"
        if self.priority_node_indices:
            pdepth = np.asarray(state, dtype=float)[self.priority_node_indices]
            priority_depth_max = float(np.nanmax(pdepth)) if pdepth.size else 0.0
        else:
            priority_depth_max = float(np.nanpercentile(np.asarray(state, dtype=float), 95))
        risk_triggered = (
            priority_depth_max >= self.priority_depth_trigger
            or (float(rainfall_mm_h) >= self.rain_trigger_mm_h and priority_depth_max >= self.rain_priority_depth_trigger)
        )
        event_risk_class = self._event_risk_class()
        current_priority_risk_class = self._current_priority_risk_class(priority_depth_max, rainfall_mm_h)
        residual_mode = self.residual_predictor is not None and nominal_source == "internal_rule_nominal"
        low_risk_blocked, low_risk_reason = should_block_low_risk_takeover(
            self.nominal_pfv_reference,
            self.low_risk_pfv_threshold,
            use_native_shield=nominal_source == "internal_rule_nominal",
        )
        base_meta = self._history_meta(
            event_risk_class=event_risk_class,
            current_priority_risk_class=current_priority_risk_class,
            intervention_allowed=False,
            intervention_reason="not_evaluated",
            native_expected_pfv=self.nominal_pfv_reference,
        )
        if self._hold_remaining > 0 and self._hold_action is not None:
            cancel_hold, cancel_hold_reason = should_cancel_held_action_in_low_risk(
                low_risk_blocked,
                current_priority_risk_class,
            )
            if cancel_hold:
                self._hold_remaining = 0
                self._hold_action = None
                self._hold_label = ""
                return nominal, {
                    **{
                        **base_meta,
                        "intervention_reason": low_risk_reason or cancel_hold_reason,
                        "false_intervention_reason": low_risk_reason or cancel_hold_reason,
                    },
                    "candidate_count": 0,
                    "valid_candidate_count": 0,
                    "fallback_to_nominal": True,
                    "priority_depth_max": priority_depth_max,
                    "risk_triggered": bool(risk_triggered),
                    "nominal_source": nominal_source,
                    "value_model_source": "held_residual_action",
                    "holding_action": False,
                    "cancelled_held_action": True,
                    "low_risk_fallback": True,
                    "low_risk_reason": low_risk_reason or cancel_hold_reason,
                }
            self._hold_remaining -= 1
            action = np.asarray(self._hold_action, dtype=np.float32).copy()
            hold_meta = self._history_meta(
                event_risk_class=event_risk_class,
                current_priority_risk_class=current_priority_risk_class,
                intervention_allowed=True,
                intervention_reason="continue_held_action",
                predicted_pfv_gain=float("nan"),
                predicted_pfv_gain_abs=float("nan"),
                native_expected_pfv=self.nominal_pfv_reference,
                low_risk_false_intervention=event_risk_class == "low_risk_event",
                false_intervention_reason="held_action_in_low_risk" if event_risk_class == "low_risk_event" else "",
            )
            return action, {
                **hold_meta,
                "candidate_count": 0,
                "valid_candidate_count": 1,
                "fallback_to_nominal": False,
                "priority_depth_max": priority_depth_max,
                "risk_triggered": risk_triggered,
                "nominal_source": nominal_source,
                "value_model_source": "held_residual_action",
                "selected_candidate_label": self._hold_label,
                "holding_action": True,
                "hold_remaining_after_step": int(self._hold_remaining),
            }
        if self._cooldown > 0:
            self._cooldown -= 1
            return nominal, {
                **{
                    **base_meta,
                    "intervention_reason": "cooldown_fallback",
                },
                "candidate_count": 0,
                "valid_candidate_count": 0,
                "fallback_to_nominal": True,
                "priority_depth_max": priority_depth_max,
                "risk_triggered": risk_triggered,
                "nominal_source": nominal_source,
                "value_model_source": "residual_action_value" if residual_mode else "hydraulic_surrogate",
                "cooldown_fallback": True,
            }
        if not risk_triggered:
            return nominal, {
                **{
                    **base_meta,
                    "intervention_reason": "current_priority_risk_not_triggered",
                },
                "candidate_count": 0,
                "valid_candidate_count": 0,
                "fallback_to_nominal": True,
                "priority_depth_max": priority_depth_max,
                "risk_triggered": False,
                "nominal_source": nominal_source,
                "value_model_source": "residual_action_value" if residual_mode else "hydraulic_surrogate",
            }
        if low_risk_blocked:
            return nominal, {
                **{
                    **base_meta,
                    "intervention_reason": low_risk_reason or "low_risk_event_fallback",
                    "false_intervention_reason": low_risk_reason or "low_risk_event_fallback",
                },
                "candidate_count": 0,
                "valid_candidate_count": 0,
                "fallback_to_nominal": True,
                "priority_depth_max": priority_depth_max,
                "risk_triggered": bool(risk_triggered),
                "nominal_source": nominal_source,
                "value_model_source": "residual_action_value" if residual_mode else "hydraulic_surrogate",
                "low_risk_fallback": True,
                "low_risk_reason": low_risk_reason,
                "nominal_pfv_reference": float(self.nominal_pfv_reference),
            }
        labeled_candidates = generate_labeled_candidates(
            nominal,
            self.actuators,
            phase,
            max_delta=self.max_candidate_delta,
            include_nominal=False,
            state=state,
            priority_upstream_nodes=self.priority_upstream_nodes,
            priority_downstream_nodes=self.priority_downstream_nodes,
            max_candidates=self.max_candidate_count,
            hold_steps=self.candidate_hold_steps,
            allowed_templates=self.allowed_candidate_templates,
            blocked_templates=self.blocked_candidate_templates,
            allowed_scopes_by_template=self.allowed_candidate_scopes_by_template,
        )
        candidates = [u for _, u in labeled_candidates]
        if self.model is None and self.residual_predictor is None:
            return nominal, {
                **{
                    **base_meta,
                    "intervention_reason": "no_value_model_available",
                },
                "candidate_count": len(candidates),
                "valid_candidate_count": 0,
                "fallback_to_nominal": True,
                "priority_depth_max": priority_depth_max,
                "risk_triggered": True,
                "nominal_source": nominal_source,
                "value_model_source": "none",
            }
        s = torch.tensor(state[None, :], dtype=torch.float32, device=self.device)
        rain = torch.tensor([[rainfall_mm_h]], dtype=torch.float32, device=self.device)
        valid = []
        all_scores = []
        pfv_improve_count = 0
        tfv_safe_count = 0
        peak_safe_count = 0
        pfv_prob_count = 0
        safe_prob_count = 0
        nonzero_prob_count = 0
        zero_delta_candidate_count = 0
        best_pred = {
            "delta_pfv": 0.0,
            "delta_tfv": 0.0,
            "delta_peak": 0.0,
            "pfv_prob": 0.0,
            "safe_prob": 0.0,
            "nonzero_prob": 0.0,
            "peak_prob": 0.0,
        }
        candidate_records: list[dict] = []
        tfv_guard = self.tfv_guard_pct * baseline_tfv_rate if baseline_tfv_rate > 1.1 else 0.0
        peak_guard = self.peak_guard_pct * baseline_peak if baseline_peak > 1.1 else 0.0
        with torch.no_grad():
            for candidate_label, u in labeled_candidates:
                if candidate_label == "nominal" or np.nanmax(np.abs(np.asarray(u) - nominal)) <= 1e-6:
                    zero_delta_candidate_count += 1
                    continue
                value_source = "hydraulic_surrogate"
                if residual_mode:
                    feat = build_residual_feature_dict(
                        self.actuators,
                        nominal,
                        u,
                        phase,
                        float(rainfall_mm_h),
                        float(priority_depth_max),
                        float(elapsed_min),
                    )
                    feat.update(candidate_metadata_features(candidate_label))
                    rv = self.residual_predictor.predict_one(feat)
                    delta_pfv = float(rv["delta_pfv"])
                    delta_tfv = float(rv["delta_tfv"])
                    delta_peak = float(rv["delta_peak"])
                    pfv_prob = float(rv["pfv_improve_prob"])
                    safe_prob = float(rv["safe_prob"])
                    pfv_nonzero_prob = float(rv["pfv_nonzero_prob"])
                    peak_nonworse_prob = float(rv.get("peak_nonworse_prob", safe_prob))
                    value_source = "residual_action_value"
                else:
                    if self.model is None:
                        continue
                    aseq = torch.tensor(u[None, None, :], dtype=torch.float32, device=self.device).expand(1, self.horizon_steps, -1)
                    rseq = torch.tensor([[[rainfall_mm_h]]], dtype=torch.float32, device=self.device).expand(1, self.horizon_steps, -1)
                    out = self.model(s, aseq, rseq, self.node_static, self.edge_index, self.action_node_map)
                    delta = out["risk_delta"]
                    if self.risk_delta_scale is not None:
                        delta = delta * self.risk_delta_scale[None, :]
                    d = delta.cpu().numpy()[0]
                    logits = out.get("logits")
                    if logits is not None:
                        probs = torch.sigmoid(logits).detach().cpu().numpy()[0]
                    else:
                        probs = np.ones(3, dtype=np.float32)
                    delta_pfv = float(d[0])
                    delta_tfv = float(d[1])
                    delta_peak = float(d[2])
                    pfv_prob = float(probs[0]) if len(probs) > 0 else 1.0
                    safe_prob = float(probs[1]) if len(probs) > 1 else 1.0
                    pfv_nonzero_prob = float(probs[2]) if len(probs) > 2 else 1.0
                    peak_nonworse_prob = float(probs[4]) if len(probs) > 4 else 1.0
                all_scores.append((delta_pfv, delta_tfv, delta_peak))
                if len(all_scores) == 1 or delta_pfv < best_pred["delta_pfv"]:
                    best_pred = {
                        "delta_pfv": delta_pfv,
                        "delta_tfv": delta_tfv,
                        "delta_peak": delta_peak,
                        "pfv_prob": pfv_prob,
                        "safe_prob": safe_prob,
                        "nonzero_prob": pfv_nonzero_prob,
                        "peak_prob": peak_nonworse_prob,
                    }
                (
                    pfv_prob_min,
                    safe_prob_min,
                    pfv_nonzero_prob_min,
                    peak_prob_min,
                    phase_rule_ok,
                    phase_rule_reason,
                    action_family,
                    boundary_info,
                ) = self._candidate_thresholds(candidate_label, phase, residual_mode, priority_depth_max, rainfall_mm_h)
                empirical_ok, empirical_reason, empirical_stats = self._empirical_guard_check(candidate_label)
                pfv_ok = delta_pfv < -self.pfv_min_improve
                tfv_ok = delta_tfv <= tfv_guard
                peak_ok = delta_peak <= peak_guard
                action_change_penalty = float(np.nanmean(np.abs(np.asarray(u, dtype=float) - nominal)))
                objective = score_horizon_candidate(
                    delta_pfv=delta_pfv,
                    delta_tfv=delta_tfv,
                    delta_peak=delta_peak,
                    action_change_penalty=action_change_penalty,
                    baseline_tfv=baseline_tfv_rate,
                    baseline_peak=baseline_peak,
                    tfv_guard_pct=self.tfv_guard_pct,
                    peak_guard_pct=self.peak_guard_pct,
                    smooth_weight=self.horizon_smooth_weight,
                    violation_penalty=self.horizon_violation_penalty,
                )
                pfv_prob_ok = pfv_prob >= pfv_prob_min
                safe_prob_ok = safe_prob >= safe_prob_min
                nonzero_prob_ok = pfv_nonzero_prob >= pfv_nonzero_prob_min
                peak_prob_ok = peak_nonworse_prob >= peak_prob_min
                rejection = []
                if not pfv_ok:
                    rejection.append("delta_pfv")
                if not tfv_ok:
                    rejection.append("delta_tfv")
                if not peak_ok:
                    rejection.append("delta_peak")
                if not pfv_prob_ok:
                    rejection.append("pfv_prob")
                if not safe_prob_ok:
                    rejection.append("safe_prob")
                if not nonzero_prob_ok:
                    rejection.append("nonzero_prob")
                if not peak_prob_ok:
                    rejection.append("peak_prob")
                if not empirical_ok:
                    rejection.append(f"empirical:{empirical_reason}")
                if not phase_rule_ok:
                    rejection.append(phase_rule_reason)
                candidate_records.append(
                    {
                        "candidate_label": candidate_label,
                        "action_family": action_family,
                        "delta_pfv": float(delta_pfv),
                        "delta_tfv": float(delta_tfv),
                        "delta_peak": float(delta_peak),
                        "pfv_prob": float(pfv_prob),
                        "safe_prob": float(safe_prob),
                        "pfv_nonzero_prob": float(pfv_nonzero_prob),
                        "peak_nonworse_prob": float(peak_nonworse_prob),
                        "action_change_penalty": float(action_change_penalty),
                        "horizon_objective_score": float(objective.score),
                        "horizon_tfv_violation": float(objective.tfv_violation),
                        "horizon_peak_violation": float(objective.peak_violation),
                        "pfv_prob_min": float(pfv_prob_min),
                        "safe_prob_min": float(safe_prob_min),
                        "pfv_nonzero_prob_min": float(pfv_nonzero_prob_min),
                        "peak_prob_min": float(peak_prob_min),
                        "empirical_ok": bool(empirical_ok),
                        "phase_rule_ok": bool(phase_rule_ok),
                        **boundary_info,
                        **empirical_stats,
                        "passes_all": bool(
                            pfv_ok
                            and tfv_ok
                            and peak_ok
                            and pfv_prob_ok
                            and safe_prob_ok
                            and nonzero_prob_ok
                            and peak_prob_ok
                            and empirical_ok
                            and phase_rule_ok
                        ),
                        "rejection_reason": ",".join(rejection) if rejection else "",
                        "value_source": value_source,
                    }
                )
                pfv_improve_count += int(pfv_ok)
                tfv_safe_count += int(pfv_ok and tfv_ok)
                peak_safe_count += int(pfv_ok and tfv_ok and peak_ok)
                pfv_prob_count += int(pfv_ok and tfv_ok and peak_ok and pfv_prob_ok)
                safe_prob_count += int(pfv_ok and tfv_ok and peak_ok and pfv_prob_ok and safe_prob_ok)
                nonzero_prob_count += int(pfv_ok and tfv_ok and peak_ok and pfv_prob_ok and safe_prob_ok and nonzero_prob_ok)
                if (
                    pfv_ok
                    and tfv_ok
                    and peak_ok
                    and pfv_prob_ok
                    and safe_prob_ok
                    and nonzero_prob_ok
                    and peak_prob_ok
                    and empirical_ok
                    and phase_rule_ok
                ):
                    valid.append(
                        (
                            u,
                            delta_pfv,
                            delta_tfv,
                            delta_peak,
                            pfv_prob,
                            safe_prob,
                            pfv_nonzero_prob,
                            value_source,
                            candidate_label,
                            peak_nonworse_prob,
                            objective.score,
                            action_change_penalty,
                        )
                    )
        candidate_records = sorted(
            candidate_records,
            key=lambda r: (
                not bool(r.get("passes_all", False)),
                float(r.get("horizon_objective_score", r.get("delta_pfv", 0.0))),
                float(r.get("delta_pfv", 0.0)),
                -float(r.get("safe_prob", 0.0)),
            ),
        )
        topk_json = json.dumps(candidate_records[: self.topk_log_count], ensure_ascii=False)
        if not valid:
            return nominal, {
                **{
                    **base_meta,
                    "intervention_reason": "no_candidate_passed_safety_gates",
                    "predicted_pfv_gain": max(0.0, -float(best_pred["delta_pfv"])),
                    "predicted_pfv_gain_abs": max(0.0, -float(best_pred["delta_pfv"])),
                    "candidate_expected_pfv": self.nominal_pfv_reference + float(best_pred["delta_pfv"]),
                    "delta_PFV_p50": float(best_pred["delta_pfv"]),
                    "delta_PFV_p90": float(best_pred["delta_pfv"]),
                    "delta_TFV_p90": float(best_pred["delta_tfv"]),
                    "delta_peak_p90": float(best_pred["delta_peak"]),
                    "uncertainty_score": 0.0,
                    "uncertainty_gate_pass": False,
                    "uncertainty_gate_reason": "no_candidate_passed_safety_gates",
                },
                "candidate_count": len(candidates),
                "valid_candidate_count": 0,
                "zero_delta_candidate_count": zero_delta_candidate_count,
                "fallback_to_nominal": True,
                "priority_depth_max": priority_depth_max,
                "risk_triggered": True,
                "nominal_source": nominal_source,
                "value_model_source": "residual_action_value" if residual_mode else "hydraulic_surrogate",
                "tfv_guard": tfv_guard,
                "peak_guard": peak_guard,
                "pfv_improve_count": pfv_improve_count,
                "tfv_safe_count": tfv_safe_count,
                "peak_safe_count": peak_safe_count,
                "pfv_prob_count": pfv_prob_count,
                "safe_prob_count": safe_prob_count,
                "nonzero_prob_count": nonzero_prob_count,
                "best_pred_delta_pfv": best_pred["delta_pfv"],
                "best_pred_delta_tfv": best_pred["delta_tfv"],
                "best_pred_delta_peak": best_pred["delta_peak"],
                "best_pred_pfv_prob": best_pred["pfv_prob"],
                "best_pred_safe_prob": best_pred["safe_prob"],
                "best_pred_pfv_nonzero_prob": best_pred["nonzero_prob"],
                "best_pred_peak_nonworse_prob": best_pred["peak_prob"],
                "topk_candidates_json": topk_json,
                "topk_candidate_count": min(len(candidate_records), self.topk_log_count),
            }
        best = min(valid, key=lambda x: x[10])
        predicted_gain = max(0.0, -float(best[1]))
        hold_steps = int(parse_candidate_label(best[8]).get("hold_steps", 1) or 1)
        self._hold_remaining = max(0, hold_steps - 1)
        self._hold_action = np.asarray(best[0], dtype=np.float32).copy() if self._hold_remaining > 0 else None
        self._hold_label = str(best[8]) if self._hold_remaining > 0 else ""
        self._cooldown = max(0, self.min_control_interval_steps - hold_steps)
        return best[0], {
            **self._history_meta(
                event_risk_class=event_risk_class,
                current_priority_risk_class=current_priority_risk_class,
                intervention_allowed=True,
                intervention_reason="candidate_passed_all_gates",
                predicted_pfv_gain=predicted_gain,
                predicted_pfv_gain_abs=predicted_gain,
                native_expected_pfv=self.nominal_pfv_reference,
                candidate_expected_pfv=self.nominal_pfv_reference + float(best[1]),
                low_risk_false_intervention=event_risk_class == "low_risk_event",
                false_intervention_reason="selected_action_in_low_risk" if event_risk_class == "low_risk_event" else "",
                delta_PFV_p50=float(best[1]),
                delta_PFV_p90=float(best[1]),
                delta_TFV_p90=float(best[2]),
                delta_peak_p90=float(best[3]),
                uncertainty_score=0.0,
                uncertainty_gate_pass=True,
                uncertainty_gate_reason="residual_value_no_quantile_head" if best[7] == "residual_action_value" else "deterministic_surrogate_proxy",
            ),
            "candidate_count": len(candidates),
            "valid_candidate_count": len(valid),
            "zero_delta_candidate_count": zero_delta_candidate_count,
            "fallback_to_nominal": False,
            "priority_depth_max": priority_depth_max,
            "risk_triggered": True,
            "nominal_source": nominal_source,
            "value_model_source": best[7],
            "selected_candidate_label": best[8],
            "selected_hold_steps": hold_steps,
            "hold_remaining_after_step": int(self._hold_remaining),
            "holding_action": False,
            "tfv_guard": tfv_guard,
            "peak_guard": peak_guard,
            "pfv_improve_count": pfv_improve_count,
            "tfv_safe_count": tfv_safe_count,
            "peak_safe_count": peak_safe_count,
            "pfv_prob_count": pfv_prob_count,
            "safe_prob_count": safe_prob_count,
            "nonzero_prob_count": nonzero_prob_count,
            "selected_pred_delta_pfv": best[1],
            "selected_pred_delta_tfv": best[2],
            "selected_pred_delta_peak": best[3],
            "selected_pfv_prob": best[4],
            "selected_safe_prob": best[5],
            "selected_pfv_nonzero_prob": best[6],
            "selected_peak_nonworse_prob": best[9],
            "selected_horizon_objective_score": best[10],
            "selected_action_change_penalty": best[11],
            "topk_candidates_json": topk_json,
            "topk_candidate_count": min(len(candidate_records), self.topk_log_count),
        }
