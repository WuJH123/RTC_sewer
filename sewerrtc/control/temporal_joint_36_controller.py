from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any, Protocol, Sequence
import re

import numpy as np
import pandas as pd

from .temporal_joint_candidate_search import (
    TemporalJointCandidateConfig,
    generate_temporal_joint_candidates,
    validate_candidate_sequence,
)
from .temporal_joint_safety import (
    JointCandidatePrediction,
    JointSafetyConfig,
    select_lexicographic_candidate,
)


class RawJointPredictor(Protocol):
    def predict_many(self, **kwargs: Any) -> dict[str, np.ndarray]: ...


class LegacyHorizonPredictor(Protocol):
    def predict_many(self, sequences: list[np.ndarray], contexts: list[dict[str, Any]]) -> list[dict[str, np.ndarray]]: ...


class TemporalJoint36Controller:
    """Receding-horizon controller for the isolated 36-actuator study line.

    ``choose`` is intentionally stateless with respect to future hydraulics:
    every call receives a fresh reconstructed state and forecast, scores raw
    ``[B,H,A]`` candidates against an online default sequence, and returns only
    the selected first row for PySWMM execution.
    """

    def __init__(
        self,
        *,
        actuators: pd.DataFrame,
        predictor: RawJointPredictor,
        candidate_config: TemporalJointCandidateConfig,
        safety_config: JointSafetyConfig,
        legacy_groups: Sequence[Sequence[str]] = (),
        paired_groups: Sequence[Sequence[str]] = (),
        legacy_predictor: LegacyHorizonPredictor | None = None,
        residual_actuator_ids: Sequence[str] = (),
        residual_enabled: bool = True,
        residual_status_reason: str = "configured",
        event_id: str = "",
        deployment_allowed_patterns: Sequence[str] = (),
        deployment_return_period_min: int = 0,
        deployment_return_period_max: int = 0,
        deployment_evidence_csv: str = "",
        deployment_time_tolerance_min: float = 0.1,
        deployment_require_evidence: bool = False,
        deployment_stratum_rules: Sequence[dict[str, object]] = (),
        template_reliability_csv: str = "",
        template_reliability_default_allow: bool = True,
        template_reliability_stress_return_period_min: int = 75,
        template_reliability_stress_default_allow: bool | None = None,
        template_reliability_block_strong_patterns: Sequence[str] = (),
        prediction_horizon_steps: int | None = None,
        terminal_return_steps: int = 6,
    ) -> None:
        self.actuators = actuators.reset_index(drop=True).copy()
        if len(self.actuators) != 36:
            raise ValueError(f"TemporalJoint36Controller requires 36 canonical actuators, got {len(self.actuators)}")
        self.predictor = predictor
        self.candidate_config = candidate_config
        self.safety_config = safety_config
        self.legacy_groups = [list(group) for group in legacy_groups]
        self.paired_groups = [list(group) for group in paired_groups]
        self.legacy_predictor = legacy_predictor
        self.residual_actuator_ids = tuple(str(value) for value in residual_actuator_ids)
        self.residual_enabled = bool(residual_enabled)
        self.residual_status_reason = str(residual_status_reason)
        self.event_id = str(event_id)
        self.deployment_allowed_patterns = tuple(str(value) for value in deployment_allowed_patterns if str(value))
        self.deployment_return_period_min = int(deployment_return_period_min)
        self.deployment_return_period_max = int(deployment_return_period_max)
        self.deployment_time_tolerance_min = float(deployment_time_tolerance_min)
        self.deployment_require_evidence = bool(deployment_require_evidence)
        self.deployment_evidence = self._load_deployment_evidence(deployment_evidence_csv)
        self.deployment_stratum_rules = tuple(dict(rule) for rule in deployment_stratum_rules)
        self.template_reliability = self._load_template_reliability(template_reliability_csv)
        self.template_reliability_default_allow = bool(template_reliability_default_allow)
        self.template_reliability_stress_return_period_min = int(template_reliability_stress_return_period_min)
        self.template_reliability_stress_default_allow = (
            bool(template_reliability_stress_default_allow)
            if template_reliability_stress_default_allow is not None
            else bool(template_reliability_default_allow)
        )
        self.template_reliability_block_strong_patterns = tuple(
            str(value) for value in template_reliability_block_strong_patterns if str(value)
        )
        self.prediction_horizon_steps = max(
            int(candidate_config.horizon_steps),
            int(prediction_horizon_steps or candidate_config.horizon_steps),
        )
        self.terminal_return_steps = max(0, int(terminal_return_steps))
        self.decision_index = 0
        self._pump_last_change = {str(aid): -10_000 for aid in candidate_config.binary_pump_ids}
        self._pump_switch_count = {str(aid): 0 for aid in candidate_config.binary_pump_ids}
        self._event_reference_pfv_estimate = 0.0
        self._event_budget_realized_pfv_cost = 0.0
        self._event_budget_pending_pfv_costs = [0.0] * max(1, self.prediction_horizon_steps)

    def _load_deployment_evidence(self, path: str) -> pd.DataFrame | None:
        if not path:
            return None
        try:
            evidence = pd.read_csv(path)
        except Exception:
            return None
        required = {"event_id", "elapsed_min"}
        if not required.issubset(set(evidence.columns)):
            return None
        evidence = evidence.copy()
        evidence["event_id"] = evidence["event_id"].astype(str)
        evidence["elapsed_min"] = pd.to_numeric(evidence["elapsed_min"], errors="coerce")
        evidence = evidence[evidence["elapsed_min"].notna()]
        return evidence if not evidence.empty else None

    def _load_template_reliability(self, path: str) -> pd.DataFrame | None:
        if not path:
            return None
        try:
            table = pd.read_csv(path)
        except Exception:
            return None
        required = {"allowed", "reason"}
        if not required.issubset(set(table.columns)):
            return None
        table = table.copy()
        for col in ["match_level", "selected_sequence_label", "label_family", "rain_id", "pattern", "duration_min", "phase", "reason"]:
            if col not in table:
                table[col] = ""
            table[col] = table[col].fillna("").astype(str)
        table["allowed"] = table["allowed"].astype(str).str.lower().isin(["true", "1", "yes"])
        return table if not table.empty else None

    def _label_family(self, label: object) -> str:
        text = str(label or "")
        if text.startswith("engineered_"):
            text = text[len("engineered_"):]
        if text.startswith("tier2_binary_pump_"):
            return "tier2_binary_pump"
        if text.startswith("tier2_") and "RTC_IN" in text:
            return "tier2_storage_inlet"
        if text.startswith("tier2_") and "RTC_OUT" in text:
            return "tier2_storage_outlet"
        if text.startswith("tier1_ramp_dw3700.1"):
            return "tier1_dw3700"
        if "MH0200773_8" in text:
            return "v8_mh0200773_group"
        if "MSLBZW001_8" in text:
            return "v8_mslbzw001_group"
        if "HS2529198_8" in text:
            return "v8_hs2529198_group"
        return text.split("|", 1)[0].split("_0.", 1)[0]

    def _event_return_period(self) -> int:
        match = re.match(r"^T(\d+)_", self.event_id)
        return int(match.group(1)) if match else 0

    def _event_rain_pattern(self) -> str:
        match = re.match(r"^T\d+_D\d+_(.+)$", self.event_id)
        return match.group(1) if match else ""

    def _event_duration_min(self) -> int:
        match = re.search(r"_D(\d+)_", self.event_id)
        return int(match.group(1)) if match else 0

    def _template_reliability_allowed(self, item: dict[str, object], phase: str) -> tuple[bool, str]:
        if self.template_reliability is None:
            return True, "template_reliability_not_configured"
        label = str(item.get("label", ""))
        if int(item.get("tier", 0) or 0) == 0:
            return True, "reference_always_allowed"
        # Only manage learned/engineered high-impact templates. Ordinary
        # candidates remain available unless the family table explicitly blocks
        # them, which avoids turning reliability filtering into global fallback.
        family = self._label_family(label)
        rain_id = f"T{self._event_return_period()}" if self._event_return_period() else ""
        pattern = self._event_rain_pattern()
        duration = str(self._event_duration_min())
        phase_text = str(phase or "")
        table = self.template_reliability
        strong_families = {
            "tier1_dw3700",
            "v8_mh0200773_group",
            "v8_mslbzw001_group",
            "v8_hs2529198_group",
        }
        if family in strong_families and pattern in self.template_reliability_block_strong_patterns:
            return False, "strong_template_blocked_for_rain_pattern"

        def score_matches(candidates: pd.DataFrame) -> pd.DataFrame:
            if candidates.empty:
                return candidates
            work = candidates.copy()
            score = np.zeros(len(work), dtype=int)
            for col, value, weight in [
                # Rain-pattern and phase rules are deployment safety rules:
                # block/double_peak prohibitions must override broad T100
                # allowances learned across all patterns.
                ("pattern", pattern, 16),
                ("rain_id", rain_id, 8),
                ("duration_min", duration, 2),
                ("phase", phase_text, 4),
            ]:
                text = work[col].fillna("").astype(str)
                exact = text.eq(str(value))
                wildcard = text.eq("")
                keep = exact | wildcard
                work = work[keep].copy()
                score = score[keep.to_numpy()]
                score += exact[keep].to_numpy(dtype=int) * weight
            if work.empty:
                return work
            work["_match_score"] = score
            return work.sort_values("_match_score", ascending=False)

        exact = table[
            table["match_level"].astype(str).eq("label")
            & table["selected_sequence_label"].astype(str).eq(label)
        ]
        exact = score_matches(exact)
        if not exact.empty:
            row = exact.iloc[0]
            return bool(row["allowed"]), str(row.get("reason", "template_reliability_label_rule"))
        family_rows = table[
            ~table["match_level"].astype(str).eq("label")
            & table["label_family"].astype(str).eq(family)
        ]
        family_rows = score_matches(family_rows)
        if not family_rows.empty:
            row = family_rows.iloc[0]
            return bool(row["allowed"]), str(row.get("reason", "template_reliability_family_rule"))
        if (
            self.template_reliability_stress_return_period_min > 0
            and self._event_return_period() >= self.template_reliability_stress_return_period_min
        ):
            return bool(self.template_reliability_stress_default_allow), "stress_template_reliability_no_matching_rule"
        return bool(self.template_reliability_default_allow), "template_reliability_no_matching_rule"

    def _filter_by_template_reliability(self, candidates: list[dict[str, object]], phase: str) -> tuple[list[dict[str, object]], dict[str, int]]:
        kept: list[dict[str, object]] = []
        blocked: dict[str, int] = {}
        for item in candidates:
            allowed, reason = self._template_reliability_allowed(item, phase)
            if allowed:
                kept.append(item)
            else:
                blocked[reason] = blocked.get(reason, 0) + 1
        if not kept:
            reference = [item for item in candidates if int(item.get("tier", 0) or 0) == 0]
            kept = reference[:1] if reference else candidates[:1]
        return kept, blocked

    def _deployment_allowed(self, elapsed_min: float | None = None) -> tuple[bool, str]:
        pattern = self._event_rain_pattern()
        return_period = self._event_return_period()
        duration_min = self._event_duration_min()
        if self.deployment_evidence is not None:
            if elapsed_min is not None:
                event_evidence = self.deployment_evidence[self.deployment_evidence["event_id"].eq(self.event_id)]
                if not event_evidence.empty:
                    distance = (event_evidence["elapsed_min"].astype(float) - float(elapsed_min)).abs()
                    if bool((distance <= max(0.0, self.deployment_time_tolerance_min)).any()):
                        return True, "deployment_evidence_matched"
        for index, rule in enumerate(self.deployment_stratum_rules):
            patterns = {str(item) for item in rule.get("patterns", [])}
            if patterns and pattern not in patterns:
                continue
            return_periods = {int(item) for item in rule.get("return_periods", []) if str(item)}
            if return_periods and return_period not in return_periods:
                continue
            rp_min = int(rule.get("return_period_min", 0) or 0)
            rp_max = int(rule.get("return_period_max", 0) or 0)
            if rp_min > 0 and return_period < rp_min:
                continue
            if rp_max > 0 and return_period > rp_max:
                continue
            duration_rule = {int(item) for item in rule.get("durations_min", []) if str(item)}
            if duration_rule and duration_min not in duration_rule:
                continue
            duration_min_rule = int(rule.get("duration_min", 0) or 0)
            duration_max_rule = int(rule.get("duration_max", 0) or 0)
            if duration_min_rule > 0 and duration_min < duration_min_rule:
                continue
            if duration_max_rule > 0 and duration_min > duration_max_rule:
                continue
            return True, f"deployment_stratum_rule_{index}"
        if self.deployment_allowed_patterns and pattern not in self.deployment_allowed_patterns:
            return False, f"deployment_pattern_not_reliable:{pattern}"
        if self.deployment_return_period_min > 0 and return_period < self.deployment_return_period_min:
            return False, f"deployment_return_period_below_min:T{return_period}"
        if self.deployment_return_period_max > 0 and return_period > self.deployment_return_period_max:
            return False, f"deployment_return_period_above_max:T{return_period}"
        if self.deployment_require_evidence:
            return False, "deployment_evidence_or_stratum_required"
        return True, "deployment_reliable"

    def _pump_runtime_allowed(self, sequence: np.ndarray, reference: np.ndarray) -> bool:
        ids = self.actuators["actuator_id"].astype(str).tolist()
        dwell = max(0, int(self.candidate_config.binary_pump_min_dwell_steps))
        for aid in self.candidate_config.binary_pump_ids:
            if aid not in ids:
                continue
            index = ids.index(aid)
            changing_now = abs(float(sequence[0, index]) - float(reference[0, index])) > 1.0e-7
            if not changing_now:
                continue
            if self.decision_index - int(self._pump_last_change.get(aid, -10_000)) < dwell:
                return False
            if int(self._pump_switch_count.get(aid, 0)) >= int(self.candidate_config.max_pump_switches_per_event):
                return False
        return True

    def _complexity(self, sequence: np.ndarray, reference: np.ndarray) -> tuple[int, float, int]:
        residual = np.asarray(sequence) - np.asarray(reference)
        simultaneous = int((np.abs(residual) > 1.0e-7).sum(axis=1).max(initial=0))
        ids = self.actuators["actuator_id"].astype(str).tolist()
        pump_switches = 0
        for actuator_id in self.candidate_config.binary_pump_ids:
            if actuator_id in ids:
                values = np.asarray(sequence)[:, ids.index(actuator_id)]
                pump_switches += int(np.sum(np.abs(np.diff(values)) > 1.0e-7))
        return simultaneous, float(np.abs(residual).sum()), pump_switches

    def _score_legacy(
        self,
        candidates: list[dict[str, object]],
        *,
        state: np.ndarray,
        rain: np.ndarray,
        reference: np.ndarray,
        phase: str,
    ) -> tuple[list[JointCandidatePrediction], float]:
        reference = np.asarray(reference, dtype=np.float32)
        raw_sequences = [np.asarray(item["candidate_action_seq"], dtype=np.float32) for item in candidates]
        sequences = [
            self._extend_for_prediction(sequence, reference)
            if len(reference) != len(sequence)
            else sequence
            for sequence in raw_sequences
        ]
        contexts = [{
            "label": str(item["label"]),
            "reconstructed_state": np.asarray(state, dtype=np.float32),
            "rainfall_window": np.asarray(rain, dtype=np.float32),
            "current_action": reference[0],
            "reference_action_sequence": reference,
            "phase": str(phase),
        } for item in candidates]
        rows = self.legacy_predictor.predict_many(sequences, contexts)  # type: ignore[union-attr]

        def total(row: dict[str, np.ndarray], upper: str, base: str, peak: bool = False) -> float:
            values = np.asarray(row.get(upper, row.get(base, np.zeros(1))), dtype=float)
            return float(np.max(values) if peak else np.sum(values))

        reference_pfv = total(rows[0], "pfv_upper", "pfv")
        reference_tfv = total(rows[0], "tfv_upper", "tfv")
        reference_peak = total(rows[0], "peak_tfv_rate_upper", "peak_tfv_rate", peak=True)
        predictions = []
        for item, sequence, row in zip(candidates, sequences, rows):
            simultaneous, action_l1, pump_switches = self._complexity(sequence, reference)
            predictions.append(JointCandidatePrediction(
                label=str(item["label"]),
                delta_pfv=total(row, "pfv_upper", "pfv") - reference_pfv,
                delta_tfv=total(row, "tfv_upper", "tfv") - reference_tfv,
                delta_peak=total(row, "peak_tfv_rate_upper", "peak_tfv_rate", peak=True) - reference_peak,
                sigma_pfv=0.0,
                sigma_tfv=0.0,
                sigma_peak=0.0,
                simultaneous_actuators=simultaneous,
                action_l1=action_l1,
                pump_switches=pump_switches,
            ))
        return predictions, reference_pfv

    def _score_raw(
        self,
        candidates: list[dict[str, object]],
        *,
        state: np.ndarray,
        rain: np.ndarray,
        no_control_reference: np.ndarray,
        phase: str,
        actuator_mask: np.ndarray | None,
        storage_state: dict[str, float] | None,
        downstream_headroom: dict[str, float] | None,
    ) -> tuple[list[JointCandidatePrediction], dict[str, np.ndarray]]:
        candidate_batch = np.stack([np.asarray(item["candidate_action_seq"], dtype=np.float32) for item in candidates])
        reference_batch = np.repeat(no_control_reference[None, :, :], len(candidates), axis=0)
        mask = np.ones((len(candidates), len(self.actuators)), dtype=np.float32)
        if actuator_mask is not None:
            supplied = np.asarray(actuator_mask, dtype=np.float32).reshape(1, -1)
            if supplied.shape[1] != len(self.actuators):
                raise ValueError("actuator_mask does not match canonical action order")
            mask = np.repeat(supplied, len(candidates), axis=0)
        outputs = self.predictor.predict_many(
            state=np.asarray(state, dtype=np.float32),
            rain_seq=np.asarray(rain, dtype=np.float32),
            candidate_action_seq=candidate_batch,
            reference_action_seq=reference_batch,
            actuator_mask=mask,
            context={
                "phase": str(phase),
                "storage_state": dict(storage_state or {}),
                "downstream_headroom": dict(downstream_headroom or {}),
                "decision_index": int(self.decision_index),
            },
        )
        required = (
            "reference_PFV_H", "delta_PFV_H", "delta_TFV_H", "delta_peak",
            "delta_PFV_sigma", "delta_TFV_sigma", "delta_peak_sigma",
        )
        missing = [key for key in required if key not in outputs]
        if missing:
            raise KeyError(f"raw joint predictor is missing outputs: {missing}")

        def values(name: str, fallback: str, default: float = 1.0) -> np.ndarray:
            if name in outputs:
                return np.asarray(outputs[name], dtype=float).reshape(-1)
            if fallback in outputs:
                return np.asarray(outputs[fallback], dtype=float).reshape(-1)
            return np.full(len(candidates), default, dtype=float)

        pfv_probability = values("PFV_noninferiority_classifier_probability", "PFV_noninferiority_probability")
        tfv_probability = values("TFV_improvement_classifier_probability", "TFV_improvement_probability")
        peak_probability = values("peak_safe_classifier_probability", "peak_safe_probability")
        pfv_threshold = values("PFV_noninferiority_classifier_threshold", "", 0.0)
        tfv_threshold = values("TFV_improvement_classifier_threshold", "", 0.0)
        peak_threshold = values("peak_safe_classifier_threshold", "", 0.0)
        predictions = []
        for index, item in enumerate(candidates):
            simultaneous, action_l1, pump_switches = self._complexity(candidate_batch[index], no_control_reference)
            predictions.append(JointCandidatePrediction(
                label=str(item["label"]),
                delta_pfv=float(outputs["delta_PFV_H"][index]),
                delta_tfv=float(outputs["delta_TFV_H"][index]),
                delta_peak=float(outputs["delta_peak"][index]),
                sigma_pfv=float(outputs["delta_PFV_sigma"][index]),
                sigma_tfv=float(outputs["delta_TFV_sigma"][index]),
                sigma_peak=float(outputs["delta_peak_sigma"][index]),
                simultaneous_actuators=simultaneous,
                action_l1=action_l1,
                pump_switches=pump_switches,
                pfv_noninferiority_probability=float(pfv_probability[index]),
                tfv_improvement_probability=float(tfv_probability[index]),
                peak_safe_probability=float(peak_probability[index]),
                pfv_classifier_threshold=float(pfv_threshold[index]),
                tfv_classifier_threshold=float(tfv_threshold[index]),
                peak_classifier_threshold=float(peak_threshold[index]),
            ))
        return predictions, outputs

    def _record_pump_changes(self, first_action: np.ndarray, reference: np.ndarray) -> None:
        ids = self.actuators["actuator_id"].astype(str).tolist()
        for actuator_id in self.candidate_config.binary_pump_ids:
            if actuator_id not in ids:
                continue
            index = ids.index(actuator_id)
            if abs(float(first_action[index]) - float(reference[0, index])) > 1.0e-7:
                self._pump_last_change[actuator_id] = int(self.decision_index)
                self._pump_switch_count[actuator_id] = int(self._pump_switch_count.get(actuator_id, 0)) + 1

    def _advance_event_budget(self) -> None:
        if not bool(self.safety_config.event_pfv_budget_enabled):
            return
        if self.decision_index <= 0:
            return
        if not self._event_budget_pending_pfv_costs:
            return
        self._event_budget_realized_pfv_cost += max(0.0, float(self._event_budget_pending_pfv_costs.pop(0)))
        self._event_budget_pending_pfv_costs.append(0.0)

    def _event_pfv_margin(self, reference_pfv: float) -> float:
        self._event_reference_pfv_estimate = max(
            float(self._event_reference_pfv_estimate),
            max(0.0, float(reference_pfv)),
        )
        return max(
            float(self.safety_config.event_pfv_abs_margin_m3),
            float(self.safety_config.event_pfv_rel_margin) * self._event_reference_pfv_estimate,
        )

    def _event_budget_state(self, reference_pfv: float) -> dict[str, float]:
        margin = self._event_pfv_margin(reference_pfv)
        committed = float(sum(max(0.0, cost) for cost in self._event_budget_pending_pfv_costs))
        remaining = margin - float(self._event_budget_realized_pfv_cost) - committed
        return {
            "event_pfv_budget_margin": float(margin),
            "event_pfv_budget_realized_cost": float(self._event_budget_realized_pfv_cost),
            "event_pfv_budget_committed_cost": float(committed),
            "event_pfv_budget_remaining": float(remaining),
            "event_reference_pfv_estimate": float(self._event_reference_pfv_estimate),
        }

    def _budgeted_config(self, reference_pfv: float) -> JointSafetyConfig:
        if not bool(self.safety_config.event_pfv_budget_enabled):
            return self.safety_config
        state = self._event_budget_state(reference_pfv)
        return replace(
            self.safety_config,
            pfv_abs_margin_m3=max(0.0, float(state["event_pfv_budget_remaining"])),
            pfv_rel_margin=0.0,
        )

    def _pfv_ucb(self, prediction: JointCandidatePrediction) -> float:
        z = max(0.0, float(self.safety_config.uncertainty_z))
        return max(0.0, float(prediction.delta_pfv) + z * max(0.0, float(prediction.sigma_pfv)))

    def _commit_event_pfv_cost(self, prediction: JointCandidatePrediction) -> None:
        if not bool(self.safety_config.event_pfv_budget_enabled):
            return
        cost = self._pfv_ucb(prediction)
        if cost <= 0.0:
            return
        steps = max(1, min(len(self._event_budget_pending_pfv_costs), self.prediction_horizon_steps))
        per_step = cost / float(steps)
        for index in range(steps):
            self._event_budget_pending_pfv_costs[index] = max(
                float(self._event_budget_pending_pfv_costs[index]),
                float(per_step),
            )

    def _extend_for_prediction(self, sequence: np.ndarray, reference_long: np.ndarray) -> np.ndarray:
        """Map a 30-minute free sequence to the 120-minute prediction grid.

        The first ``H_free`` rows are the candidate sequence scored by the raw
        joint model. The middle horizon holds the last free action. The terminal
        tail returns to the online No-control/default reference, preventing a
        candidate from looking safe only because its release/recovery phase is
        outside the modelled horizon.
        """

        seq = np.asarray(sequence, dtype=np.float32)
        ref = np.asarray(reference_long, dtype=np.float32)
        if len(ref) <= len(seq):
            return seq[: len(ref)].copy()
        out = ref.copy()
        out[: len(seq)] = seq
        tail = min(self.terminal_return_steps, max(0, len(ref) - len(seq)))
        hold_end = len(ref) - tail
        if hold_end > len(seq):
            out[len(seq) : hold_end] = seq[-1]
        return out

    def _choose_hierarchical(
        self,
        *,
        reconstructed_state: np.ndarray,
        rainfall_window: np.ndarray,
        reference: np.ndarray,
        phase: str,
        actuator_mask: np.ndarray | None,
        storage_state: dict[str, float] | None,
        downstream_headroom: dict[str, float] | None,
        rainfall_long_window: np.ndarray | None,
        reference_long: np.ndarray | None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self._advance_event_budget()
        reference_for_legacy = np.asarray(reference_long, dtype=np.float32) if reference_long is not None else reference
        rain_for_legacy = np.asarray(rainfall_long_window, dtype=np.float32) if rainfall_long_window is not None else rainfall_window
        all_candidates = generate_temporal_joint_candidates(
            reference_action_seq=reference,
            actuators=self.actuators,
            legacy_groups=self.legacy_groups,
            paired_groups=(),
            phase=phase,
            config=self.candidate_config,
        )
        all_candidates, reliability_blocked = self._filter_by_template_reliability(all_candidates, phase)
        tier1_candidates = [
            item for item in all_candidates
            if int(item["tier"]) == 0
            or (
                int(item["tier"]) == 1
                and np.any(
                    np.abs(np.asarray(item["candidate_action_seq"])[0] - reference[0]) > 1.0e-7
                )
            )
        ]
        tier1_predictions, reference_pfv = self._score_legacy(
            tier1_candidates,
            state=reconstructed_state,
            rain=rain_for_legacy,
            reference=reference_for_legacy,
            phase=phase,
        )
        tier1_budget_config = self._budgeted_config(reference_pfv)
        selected_tier1, tier1_audit = select_lexicographic_candidate(
            tier1_predictions, reference_pfv=reference_pfv, config=tier1_budget_config
        )
        tier1_index = next(index for index, row in enumerate(tier1_predictions) if row.label == selected_tier1.label)
        tier1_item = tier1_candidates[tier1_index]
        tier1_sequence = np.asarray(tier1_item["candidate_action_seq"], dtype=np.float32)
        selected_sequence = tier1_sequence
        selected_item = tier1_item
        tier2_audit: dict[str, dict[str, float | bool | str]] = {}
        tier2_candidate_count = 0
        fallback_path = "no_control" if int(tier1_item["tier"]) == 0 else "tier1"

        if int(tier1_item["tier"]) == 1 and self.residual_enabled and self.residual_actuator_ids:
            residual_config = replace(
                self.candidate_config,
                allowed_candidate_ids=tuple(self.residual_actuator_ids),
            )
            residual_candidates = generate_temporal_joint_candidates(
                reference_action_seq=tier1_sequence,
                actuators=self.actuators,
                legacy_groups=(),
                paired_groups=self.paired_groups,
                phase=phase,
                config=residual_config,
            )
            residual_candidates = [
                item for item in residual_candidates
                if (
                    int(item["tier"]) == 0
                    or (
                        int(item["tier"]) == 2
                        and np.any(
                            np.abs(np.asarray(item["candidate_action_seq"])[0] - tier1_sequence[0]) > 1.0e-7
                        )
                        and self._pump_runtime_allowed(np.asarray(item["candidate_action_seq"]), tier1_sequence)
                    )
                )
                and validate_candidate_sequence(
                    np.asarray(item["candidate_action_seq"]), reference, self.actuators, self.candidate_config
                ).get("valid", False)
            ]
            tier2_candidate_count = max(0, len(residual_candidates) - 1)
            residual_candidates[0] = {**residual_candidates[0], "label": "tier1_base", "tier": 1}
            tier2_predictions, raw_outputs = self._score_raw(
                residual_candidates,
                state=reconstructed_state,
                rain=rainfall_window,
                no_control_reference=reference,
                phase=phase,
                actuator_mask=actuator_mask,
                storage_state=storage_state,
                downstream_headroom=downstream_headroom,
            )
            tier2_budget_config = self._budgeted_config(
                float(np.asarray(raw_outputs["reference_PFV_H"]).reshape(-1)[0])
            )
            selected_tier2, tier2_audit = select_lexicographic_candidate(
                tier2_predictions,
                reference_pfv=float(np.asarray(raw_outputs["reference_PFV_H"]).reshape(-1)[0]),
                config=tier2_budget_config,
            )
            tier2_index = next(index for index, row in enumerate(tier2_predictions) if row.label == selected_tier2.label)
            if int(residual_candidates[tier2_index]["tier"]) == 2:
                selected_item = residual_candidates[tier2_index]
                selected_sequence = np.asarray(selected_item["candidate_action_seq"], dtype=np.float32)
                fallback_path = "tier2"

        first_action = selected_sequence[0].copy()
        final_prediction = (
            next((row for row in tier2_predictions if row.label == str(selected_item["label"])), None)
            if fallback_path == "tier2"
            else next((row for row in tier1_predictions if row.label == str(selected_item["label"])), None)
        )
        if final_prediction is not None and fallback_path != "no_control":
            self._commit_event_pfv_cost(final_prediction)
        budget_state = self._event_budget_state(reference_pfv)
        self._record_pump_changes(first_action, reference)
        info = {
            "policy_id": "proposed_hierarchical_v8_residual_36",
            "decision_index": int(self.decision_index),
            "selected_label": str(selected_item["label"]),
            "selected_tier": int(selected_item["tier"]),
            "fallback_path": fallback_path,
            "fallback_to_no_control": fallback_path == "no_control",
            # Keep the established closed-loop evaluation schema compatible.
            "fallback_to_default": fallback_path == "no_control",
            "fallback_to_nominal": fallback_path == "no_control",
            "skip_action_write": fallback_path == "no_control",
            "selected_gate_pass": fallback_path != "no_control",
            "residual_enabled": bool(self.residual_enabled),
            "residual_status_reason": self.residual_status_reason,
            "selected_tier1_sequence": tier1_sequence.astype(float).tolist(),
            "selected_action_sequence": selected_sequence.astype(float).tolist(),
            "executed_first_action": first_action.astype(float).tolist(),
            "target_actuators": list(selected_item.get("target_actuators", [])),
            "simultaneous_actuator_count": self._complexity(selected_sequence, reference)[0],
            "candidate_count": len(tier1_candidates) + tier2_candidate_count,
            "template_reliability_blocked": reliability_blocked,
            "tier1_gate_audit": tier1_audit,
            "tier2_gate_audit": tier2_audit,
            "candidate_tensor_shape": [len(tier1_candidates), *reference.shape],
            "reference_tensor_shape": [len(tier1_candidates), *reference.shape],
            "prediction_horizon_steps": int(len(reference_for_legacy)),
            "move_horizon_steps": int(reference.shape[0]),
            **budget_state,
        }
        self.decision_index += 1
        return first_action, info

    def choose(
        self,
        *,
        reconstructed_state: np.ndarray,
        rainfall_window: np.ndarray,
        reference_action_sequence: np.ndarray,
        rainfall_long_window: np.ndarray | None = None,
        reference_action_sequence_long: np.ndarray | None = None,
        phase: str,
        actuator_mask: np.ndarray | None = None,
        storage_state: dict[str, float] | None = None,
        downstream_headroom: dict[str, float] | None = None,
        elapsed_min: float | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        reference = np.asarray(reference_action_sequence, dtype=np.float32)
        deployment_ok, deployment_reason = self._deployment_allowed(elapsed_min)
        if not deployment_ok:
            first_action = reference[0].copy()
            info = {
                "policy_id": "proposed_hierarchical_v8_residual_36" if self.legacy_predictor is not None else "proposed_temporal_joint_36",
                "decision_index": int(self.decision_index),
                "selected_label": "deployment_reliability_no_control",
                "selected_tier": 0,
                "fallback_path": "no_control",
                "fallback_to_no_control": True,
                "fallback_to_default": True,
                "fallback_to_nominal": True,
                "skip_action_write": True,
                "selected_gate_pass": False,
                "deployment_reliability_allowed": False,
                "deployment_reliability_reason": deployment_reason,
                "selected_action_sequence": reference.astype(float).tolist(),
                "executed_first_action": first_action.astype(float).tolist(),
                "target_actuators": [],
                "simultaneous_actuator_count": 0,
                "candidate_count": 1,
                "candidate_tensor_shape": [1, *reference.shape],
                "reference_tensor_shape": [1, *reference.shape],
            }
            self.decision_index += 1
            return first_action, info
        if self.legacy_predictor is not None:
            action, info = self._choose_hierarchical(
                reconstructed_state=reconstructed_state,
                rainfall_window=rainfall_window,
                reference=reference,
                phase=phase,
                actuator_mask=actuator_mask,
                storage_state=storage_state,
                downstream_headroom=downstream_headroom,
                rainfall_long_window=rainfall_long_window,
                reference_long=reference_action_sequence_long,
            )
            info["deployment_reliability_allowed"] = True
            info["deployment_reliability_reason"] = deployment_reason
            return action, info
        candidates = generate_temporal_joint_candidates(
            reference_action_seq=reference,
            actuators=self.actuators,
            legacy_groups=self.legacy_groups,
            paired_groups=self.paired_groups,
            phase=phase,
            config=self.candidate_config,
        )
        candidates, reliability_blocked = self._filter_by_template_reliability(candidates, phase)
        candidates = [
            item for item in candidates
            if int(item["tier"]) == 0
            or self._pump_runtime_allowed(np.asarray(item["candidate_action_seq"]), reference)
        ]
        candidate_batch = np.stack([np.asarray(item["candidate_action_seq"], dtype=np.float32) for item in candidates])
        reference_batch = np.repeat(reference[None, :, :], len(candidates), axis=0)
        mask = np.ones((len(candidates), len(self.actuators)), dtype=np.float32)
        if actuator_mask is not None:
            supplied = np.asarray(actuator_mask, dtype=np.float32).reshape(1, -1)
            if supplied.shape[1] != len(self.actuators):
                raise ValueError("actuator_mask does not match canonical action order")
            mask = np.repeat(supplied, len(candidates), axis=0)
        outputs = self.predictor.predict_many(
            state=np.asarray(reconstructed_state, dtype=np.float32),
            rain_seq=np.asarray(rainfall_window, dtype=np.float32),
            candidate_action_seq=candidate_batch,
            reference_action_seq=reference_batch,
            actuator_mask=mask,
            context={
                "phase": str(phase),
                "storage_state": dict(storage_state or {}),
                "downstream_headroom": dict(downstream_headroom or {}),
                "decision_index": int(self.decision_index),
            },
        )
        required = (
            "reference_PFV_H", "delta_PFV_H", "delta_TFV_H", "delta_peak",
            "delta_PFV_sigma", "delta_TFV_sigma", "delta_peak_sigma",
        )
        missing = [key for key in required if key not in outputs]
        if missing:
            raise KeyError(f"raw joint predictor is missing outputs: {missing}")
        predictions: list[JointCandidatePrediction] = []
        def probability(name: str, fallback: str, *, default: float = 1.0) -> np.ndarray:
            if name in outputs:
                return np.asarray(outputs[name], dtype=np.float64).reshape(-1)
            if fallback in outputs:
                return np.asarray(outputs[fallback], dtype=np.float64).reshape(-1)
            return np.full(len(candidates), float(default), dtype=np.float64)

        pfv_probability = probability(
            "PFV_noninferiority_classifier_probability", "PFV_noninferiority_probability"
        )
        tfv_probability = probability(
            "TFV_improvement_classifier_probability", "TFV_improvement_probability"
        )
        peak_probability = probability("peak_safe_classifier_probability", "peak_safe_probability")
        pfv_threshold = probability("PFV_noninferiority_classifier_threshold", "", default=0.0)
        tfv_threshold = probability("TFV_improvement_classifier_threshold", "", default=0.0)
        peak_threshold = probability("peak_safe_classifier_threshold", "", default=0.0)
        for index, item in enumerate(candidates):
            residual = candidate_batch[index] - reference
            simultaneous = int((np.abs(residual) > 1.0e-7).sum(axis=1).max(initial=0))
            ids = self.actuators["actuator_id"].astype(str).tolist()
            pump_switches = 0
            for aid in self.candidate_config.binary_pump_ids:
                if aid in ids:
                    values = candidate_batch[index, :, ids.index(aid)]
                    pump_switches += int(np.sum(np.abs(np.diff(values)) > 1.0e-7))
            predictions.append(
                JointCandidatePrediction(
                    label=str(item["label"]),
                    delta_pfv=float(outputs["delta_PFV_H"][index]),
                    delta_tfv=float(outputs["delta_TFV_H"][index]),
                    delta_peak=float(outputs["delta_peak"][index]),
                    sigma_pfv=float(outputs["delta_PFV_sigma"][index]),
                    sigma_tfv=float(outputs["delta_TFV_sigma"][index]),
                    sigma_peak=float(outputs["delta_peak_sigma"][index]),
                    simultaneous_actuators=simultaneous,
                    action_l1=float(np.abs(residual).sum()),
                    pump_switches=pump_switches,
                    pfv_noninferiority_probability=float(pfv_probability[index]),
                    tfv_improvement_probability=float(tfv_probability[index]),
                    peak_safe_probability=float(peak_probability[index]),
                    pfv_classifier_threshold=float(pfv_threshold[index]),
                    tfv_classifier_threshold=float(tfv_threshold[index]),
                    peak_classifier_threshold=float(peak_threshold[index]),
                )
            )
        selected, gate_audit = select_lexicographic_candidate(
            predictions,
            reference_pfv=float(np.asarray(outputs["reference_PFV_H"]).reshape(-1)[0]),
            config=self.safety_config,
        )
        selected_index = next(index for index, row in enumerate(predictions) if row.label == selected.label)
        selected_item = candidates[selected_index]
        selected_sequence = candidate_batch[selected_index]
        first_action = selected_sequence[0].copy()
        ids = self.actuators["actuator_id"].astype(str).tolist()
        for aid in self.candidate_config.binary_pump_ids:
            if aid not in ids:
                continue
            index = ids.index(aid)
            if abs(float(first_action[index]) - float(reference[0, index])) > 1.0e-7:
                self._pump_last_change[aid] = int(self.decision_index)
                self._pump_switch_count[aid] = int(self._pump_switch_count.get(aid, 0)) + 1
        info = {
            "policy_id": "proposed_temporal_joint_36",
            "decision_index": int(self.decision_index),
            "selected_label": selected.label,
            "selected_tier": int(selected_item["tier"]),
            "fallback_to_no_control": int(selected_item["tier"]) == 0,
            "skip_action_write": int(selected_item["tier"]) == 0,
            "selected_action_sequence": selected_sequence.astype(float).tolist(),
            "executed_first_action": first_action.astype(float).tolist(),
            "target_actuators": list(selected_item["target_actuators"]),
            "simultaneous_actuator_count": int(selected.simultaneous_actuators),
            "candidate_count": len(candidates),
            "template_reliability_blocked": reliability_blocked,
            "candidate_tensor_shape": list(candidate_batch.shape),
            "reference_tensor_shape": list(reference_batch.shape),
            "gate_audit": gate_audit,
            "candidate_config": asdict(self.candidate_config),
            "safety_config": asdict(self.safety_config),
        }
        self.decision_index += 1
        return first_action, info
