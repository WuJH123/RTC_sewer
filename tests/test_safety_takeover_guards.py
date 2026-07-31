import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_low_risk_internal_event_blocks_takeover():
    from sewerrtc.control.safety_guards import should_block_low_risk_takeover

    blocked, reason = should_block_low_risk_takeover(
        nominal_pfv_reference=0.0,
        threshold=100.0,
        use_native_shield=True,
    )

    assert blocked is True
    assert reason == "low_internal_pfv"


def test_held_action_is_cancelled_after_risk_recedes():
    from sewerrtc.control.safety_guards import should_cancel_held_action_in_low_risk

    blocked, reason = should_cancel_held_action_in_low_risk(
        low_risk_blocked=False,
        current_priority_risk_class="low_risk_state",
    )

    assert blocked is True
    assert reason == "current_priority_low_risk_cancel_held_action"


def test_held_action_is_cancelled_for_low_internal_pfv_event():
    from sewerrtc.control.safety_guards import should_cancel_held_action_in_low_risk

    blocked, reason = should_cancel_held_action_in_low_risk(
        low_risk_blocked=True,
        current_priority_risk_class="medium_risk_state",
    )

    assert blocked is True
    assert reason == "low_internal_pfv_cancel_held_action"


def test_held_action_can_continue_when_risk_remains_high():
    from sewerrtc.control.safety_guards import should_cancel_held_action_in_low_risk

    blocked, reason = should_cancel_held_action_in_low_risk(
        low_risk_blocked=False,
        current_priority_risk_class="high_risk_state",
    )

    assert blocked is False
    assert reason == ""


def test_recession_release_plus_pump_boost_is_blocked_for_low_risk_event():
    from sewerrtc.control.safety_guards import candidate_boundary_decision

    decision = candidate_boundary_decision(
        candidate_label="release_plus_pump_boost:priority_upstream:d=0.08:hold=1",
        event_id="T3_D105_block",
        phase="recession",
        nominal_pfv_reference=0.0,
        priority_depth_max=0.5,
        rainfall_mm_h=0.0,
        release_recession_pfv_min=500.0,
        release_recession_priority_depth_min=1.0,
        strict_guard_return_period_max=15,
        strict_guard_patterns=("chicago_late", "block", "double_peak"),
    )

    assert decision.allowed is False
    assert decision.reason in {
        "release_recession_low_internal_pfv",
        "release_recession_cautious_event_low_priority_depth",
    }


def test_recession_release_plus_pump_boost_can_pass_high_risk_boundary():
    from sewerrtc.control.safety_guards import candidate_boundary_decision

    decision = candidate_boundary_decision(
        candidate_label="release_plus_pump_boost:priority_upstream:d=0.08:hold=1",
        event_id="T75_D105_chicago_center",
        phase="recession",
        nominal_pfv_reference=2000.0,
        priority_depth_max=1.5,
        rainfall_mm_h=0.0,
        release_recession_pfv_min=500.0,
        release_recession_priority_depth_min=1.0,
        strict_guard_return_period_max=15,
        strict_guard_patterns=("chicago_late", "block", "double_peak"),
    )

    assert decision.allowed is True
    assert decision.safe_prob_extra == 0.0
    assert decision.peak_prob_extra == 0.0


def test_strict_event_gets_stronger_probability_guard():
    from sewerrtc.control.safety_guards import candidate_boundary_decision

    decision = candidate_boundary_decision(
        candidate_label="storage_retain_pump_throttle:priority_upstream:d=0.08:hold=2",
        event_id="T15_D210_chicago_late",
        phase="peak",
        nominal_pfv_reference=1000.0,
        priority_depth_max=2.5,
        rainfall_mm_h=30.0,
        strict_guard_return_period_max=15,
        strict_guard_patterns=("chicago_late", "block", "double_peak"),
        strict_guard_prob_extra=0.1,
    )

    assert decision.allowed is True
    assert decision.safe_prob_extra == 0.1
    assert decision.peak_prob_extra == 0.1


def test_gate_fails_when_tfv_or_peak_worse_fraction_is_high():
    from sewerrtc.evaluation.evaluate_closed_loop import gate_summary

    proposed_vs_internal = {
        "PFV_median_reduction_pct": 10.0,
        "PFV_worse_frac": 0.0,
        "TFV_mean_reduction_pct": 1.0,
        "peak_TFV_rate_mean_reduction_pct": 1.0,
        "TFV_worse_frac": 0.45,
        "peak_worse_frac": 0.05,
    }

    summary = gate_summary(proposed_vs_internal)

    assert summary["passed"] is False
    assert "TFV_worse_frac" in ";".join(summary["reasons"])
