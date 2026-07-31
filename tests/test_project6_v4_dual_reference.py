from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sewerrtc.control.dual_reference_v4 import (
    CandidateEnvelopePrediction,
    DualReferenceLimits,
    EventQuantilePfvBudget,
    HydraulicPhase,
    ReferenceEnvelope,
    adaptive_k,
    choose_phase_aware_fallback,
    enforce_final_readback,
    evaluate_candidate,
)
from sewerrtc.io.safe_paths import path_budget_check, short_run_tag, single_writer_lease
from sewerrtc.prompt3.action_effect_v4 import (
    ACTION_FEATURE_NAMES, CONTEXT_FEATURE_NAMES, REFERENCE_LABELS, RESIDUAL_LABELS,
    V4_LABELS, _event_balanced_sample, materialize_v4_row,
    CAUSAL_FEATURE_NAMES, FutureHydraulicLeakageError, causal_context_features,
)
from sewerrtc.simulation.pyswmm_runner import _make_dual_reference_action_effect_predictor


def test_rising_phase_freezes_passive() -> None:
    decision = choose_phase_aware_fallback(
        phase=HydraulicPhase.RISING, pfv_budget_remaining_m3=50,
        internal_predicted_pfv_quantile_m3=10, pfv_cap_m3=20,
        internal_legal=True, passive_legal=True,
    )
    assert decision.fallback_id == "passive_anchor"


def test_peak_phase_uses_internal_only_with_headroom() -> None:
    decision = choose_phase_aware_fallback(
        phase=HydraulicPhase.PEAK, pfv_budget_remaining_m3=50,
        internal_predicted_pfv_quantile_m3=10, pfv_cap_m3=20,
        internal_legal=True, passive_legal=True,
    )
    assert decision.fallback_id == "internal_rules"


def test_peak_phase_rejects_internal_when_pfv_tight() -> None:
    decision = choose_phase_aware_fallback(
        phase=HydraulicPhase.PEAK, pfv_budget_remaining_m3=0,
        internal_predicted_pfv_quantile_m3=30, pfv_cap_m3=20,
        internal_legal=True, passive_legal=True,
    )
    assert decision.fallback_id == "passive_anchor"


def test_candidate_gate_uses_safety_and_performance_references() -> None:
    refs = ReferenceEnvelope(no_control_pfv=10, passive_pfv=12, internal_tfv=100, internal_peak=5)
    budget = EventQuantilePfvBudget(10, 12)
    good = CandidateEnvelopePrediction("good", 10, 99, 4.9, 2, 0.1)
    result = evaluate_candidate(
        good, references=refs, limits=DualReferenceLimits(), event_budget=budget,
        minimum_material_benefit=0, minimum_benefit_cost_ratio=0,
        changed_facility_penalty=1, variation_penalty=1, reversal_penalty=1,
    )
    assert result.accepted
    bad_pfv = CandidateEnvelopePrediction("bad", 10.1, 99, 4.9, 2, 0.1)
    result = evaluate_candidate(
        bad_pfv, references=refs, limits=DualReferenceLimits(), event_budget=budget,
        minimum_material_benefit=0, minimum_benefit_cost_ratio=0,
        changed_facility_penalty=1, variation_penalty=1, reversal_penalty=1,
    )
    assert "pfv_quantile_worse_than_no_control_passive_envelope" in result.reasons


def test_readback_is_hard_constraint() -> None:
    audit = enforce_final_readback(
        requested=[1, 0.5, 0], projected=[1, 0.5, 0], readback=[0.9, 0.5, 0],
        anchor=[0, 0.5, 0], actuator_ids=["ADD301.2", "add350.1", "X"],
        binary_pump_ids={"ADD301.2", "ADD301.3"}, max_k=1,
        tolerance=1e-4, deadband=0.02,
    )
    assert not audit["passed"]
    assert "write_readback_mismatch" in audit["reasons"]
    assert "binary_readback_violation" in audit["reasons"]


def test_adaptive_k_is_conservative_under_uncertainty() -> None:
    assert adaptive_k(phase=HydraulicPhase.RISING, pfv_headroom_fraction=0.01, uncertainty_score=0.2) == 0
    assert adaptive_k(phase=HydraulicPhase.RISING, pfv_headroom_fraction=0.5, uncertainty_score=0.2) == 2


def test_v4_row_materializes_metric_aligned_two_layer_heads() -> None:
    row = {
        "runtime_executed": "true", "truth_leakage": "0", "selected_fallback": "internal_rules",
        "candidate_then_internal_PFV_H120": "20", "no_control_PFV_H120": "10",
        "executable_passive_PFV_H120": "12", "internal_rules_PFV_H120": "11",
        "candidate_then_internal_TFV_H120": "90", "internal_rules_TFV_H120": "100",
        "candidate_then_internal_peak_TFV_rate_H120": "4", "internal_rules_peak_TFV_rate_H120": "5",
        "candidate_then_internal_PFV_full_recovery": "24", "no_control_PFV_full_recovery": "14",
        "executable_passive_PFV_full_recovery": "16", "internal_rules_PFV_full_recovery": "15",
    }
    row.update({f"v4_ctx_{name}": "0" for name in CONTEXT_FEATURE_NAMES})
    out = materialize_v4_row(row)
    assert out is not None
    assert [out[k] for k in REFERENCE_LABELS] == [10.0, 12.0, 11.0, 100.0, 5.0, 14.0, 16.0, 15.0]
    assert [out[k] for k in RESIDUAL_LABELS] == [10.0, 8.0, -10.0, -1.0, 10.0, 8.0]
    assert out["v4_online_future_hydraulics_used"] == "false"


def test_v4_predictor_uses_causal_reference_and_never_rebases_pfv_to_internal(tmp_path: Path) -> None:
    members = 2
    ref_weights = np.zeros((members, len(CONTEXT_FEATURE_NAMES) + 1, len(REFERENCE_LABELS)), dtype=float)
    ref_intercepts = np.asarray([5.0, 7.0, 6.0, 30.0, 5.0, 10.0, 12.0, 11.0])
    ref_weights[:, 0, :] = ref_intercepts
    res_weights = np.zeros((members, len(CONTEXT_FEATURE_NAMES) + len(ACTION_FEATURE_NAMES) + 1, len(RESIDUAL_LABELS)), dtype=float)
    path = tmp_path / "m.npz"
    np.savez(
        path,
        reference_weights=ref_weights,
        reference_feature_mean=np.zeros((members, len(CONTEXT_FEATURE_NAMES))),
        reference_feature_scale=np.ones((members, len(CONTEXT_FEATURE_NAMES))),
        residual_weights=res_weights,
        residual_feature_mean=np.zeros((members, len(CONTEXT_FEATURE_NAMES) + len(ACTION_FEATURE_NAMES))),
        residual_feature_scale=np.ones((members, len(CONTEXT_FEATURE_NAMES) + len(ACTION_FEATURE_NAMES))),
        reference_labels=np.asarray(REFERENCE_LABELS), residual_labels=np.asarray(RESIDUAL_LABELS),
        context_feature_names=np.asarray(CONTEXT_FEATURE_NAMES), action_feature_names=np.asarray(ACTION_FEATURE_NAMES),
        reference_conformal=np.zeros(len(REFERENCE_LABELS)), residual_conformal=np.zeros(len(RESIDUAL_LABELS)),
        quantile=np.asarray([0.95]),
    )
    predictor = _make_dual_reference_action_effect_predictor(path, 2)
    context = {
        "elapsed_min": 10.0, "phase": "rising",
        "reconstructed_state": np.asarray([0.1, 0.2]), "rainfall_window": np.asarray([1.0, 2.0]),
        "current_action": np.zeros(2), "reference_action_sequence": np.zeros((2, 2)),
        "selected_fallback_id": "passive_anchor",
    }
    out = predictor(np.zeros((2, 2)), context)
    assert np.isclose(out["pfv"].sum(), 7.0)  # conservative max(No-control=5, Passive=7)
    assert np.isclose(out["tfv"].sum(), 30.0)  # Internal reference
    assert np.isclose(out["peak_tfv_rate"].max(), 5.0)
    assert np.isclose(out["event_pfv_upper"][0], 12.0)
    assert out["online_future_hydraulics_used"][0] == 0.0


def test_short_path_and_writer_lease(tmp_path: Path) -> None:
    tag = short_run_tag("x" * 300, max_length=24)
    assert len(tag) <= 24
    assert path_budget_check(tmp_path / tag)["within_budget"]
    with single_writer_lease(tmp_path / "out", owner="one"):
        with pytest.raises(RuntimeError):
            with single_writer_lease(tmp_path / "out", owner="two"):
                pass


def test_v33_placeholder_patterns_not_in_v4_module() -> None:
    source = Path(__file__).resolve().parents[1] / "sewerrtc/prompt3/action_effect_v4.py"
    text = source.read_text(encoding="utf-8")
    assert "shutil.copyfile" not in text
    assert '"runtime_executed": "true"' not in text
    assert "candidate_executed_count\": 4" not in text


def test_post_projection_adaptive_k_and_benefit_cost_are_binding() -> None:
    import pandas as pd
    from sewerrtc.control.generic_gat_mpc import GenericGATMPCController

    actuators = pd.DataFrame([
        {"actuator_id": "A1", "link_type": "orifice", "asset_role": "orifice"},
        {"actuator_id": "A2", "link_type": "orifice", "asset_role": "orifice"},
    ])

    def predictor(sequence, context):
        # All candidates are hydraulically non-inferior but have no material
        # TFV/Peak benefit; the action-cost hard gate must select hold.
        h = len(sequence)
        return {
            "pfv": np.zeros(h), "tfv": np.ones(h) * 5.0,
            "peak_tfv_rate": np.ones(h),
            "pfv_upper": np.zeros(h), "tfv_upper": np.ones(h) * 5.0,
            "peak_tfv_rate_upper": np.ones(h),
            "event_pfv": np.asarray([0.0]), "event_pfv_upper": np.asarray([0.0]),
            "online_future_hydraulics_used": np.asarray([0.0]),
        }

    controller = GenericGATMPCController(
        actuators, horizon_steps=2, max_candidate_delta=0.1,
        horizon_predictor=predictor, objective_mode="pfv_preserving_system_repair",
        min_pfv_improvement_abs=0.0, candidate_group_limit=2,
        tfv_hard_constraint=True,
    )
    action, info = controller.choose(
        reconstructed_state=np.asarray([0.2, 0.3]), rainfall_window=np.asarray([1.0, 1.0]),
        current_action=np.asarray([0.5, 0.5]), reference_action_sequence=np.full((2, 2), 0.5),
        reference_pfv=np.zeros(2), reference_tfv=np.ones(2) * 5.0,
        reference_peak=np.ones(2),
        extra_predictor_context={
            "adaptive_k_limit": 0,
            "action_setting_deadband": 0.02,
            "minimum_material_benefit": 25.0,
            "minimum_benefit_cost_ratio": 1.5,
            "changed_facility_penalty": 1.0,
            "variation_penalty": 1.0,
            "reversal_penalty": 5.0,
        },
    )
    assert np.allclose(action, [0.5, 0.5])
    assert info["selected_sequence_label"] == "hold_native"
    assert info["fallback_to_default"] is True


def test_v4_smoke_sampling_preserves_event_diversity() -> None:
    frame = pd.DataFrame(
        {
            "event_id": (
                ["event_a"] * 100
                + ["event_b"] * 100
                + ["event_c"] * 100
            ),
            "sample_id": list(range(300)),
        }
    )

    sampled = _event_balanced_sample(
        frame,
        event_column="event_id",
        max_samples=64,
        minimum_events=2,
        random_seed=20260723,
    )

    assert len(sampled) == 64
    assert sampled["event_id"].nunique() >= 2
    assert sampled["sample_id"].nunique() == 64


def test_v4_smoke_sampling_is_reproducible() -> None:
    frame = pd.DataFrame(
        {
            "event_id": ["event_a"] * 50 + ["event_b"] * 50,
            "sample_id": list(range(100)),
        }
    )

    first = _event_balanced_sample(
        frame,
        event_column="event_id",
        max_samples=32,
        minimum_events=2,
        random_seed=20260723,
    )

    second = _event_balanced_sample(
        frame,
        event_column="event_id",
        max_samples=32,
        minimum_events=2,
        random_seed=20260723,
    )

    pd.testing.assert_frame_equal(first, second)


def test_v4_smoke_does_not_fabricate_second_event() -> None:
    frame = pd.DataFrame(
        {
            "event_id": ["event_a"] * 100,
            "sample_id": list(range(100)),
        }
    )

    sampled = _event_balanced_sample(
        frame,
        event_column="event_id",
        max_samples=64,
        minimum_events=2,
        random_seed=20260723,
    )

    assert len(sampled) == 64
    assert sampled["event_id"].nunique() == 1


def _prefix_rows() -> list[dict]:
    rows = []
    for step in range(13):  # 0..120 min at 10-min cadence
        elapsed = step * 10.0
        rows.append({
            "elapsed_min": elapsed,
            "rainfall_mm_h": 30.0 if elapsed <= 60.0 else 0.0,
            "flood:P1": 2.0 if 40.0 <= elapsed <= 80.0 else 0.0,
            "flood:N9": 1.0 if elapsed == 50.0 else 0.0,
            "h:P1": 1.5, "h:S1": 3.0, "h:D1": 0.5,
            "setting:ADD301.2": 1.0 if elapsed < 90.0 else 0.0,
            "selected_fallback": "passive_anchor" if elapsed < 90.0 else "internal_rules",
        })
    return rows


def test_causal_features_reject_future_hydraulics() -> None:
    rows = _prefix_rows() + [{"elapsed_min": 130.0, "flood:P1": 9.0, "h:P1": 5.0}]
    with pytest.raises(FutureHydraulicLeakageError):
        causal_context_features(
            rows, checkpoint_elapsed_min=120.0, event_duration_min=150.0,
            rainfall_forecast=[(0.0, 30.0), (60.0, 30.0), (90.0, 10.0)],
            priority_nodes=["P1"], storage_nodes=["S1"], downstream_nodes=["D1"],
        )


def test_causal_features_are_path_dependent_and_leakage_free() -> None:
    feats = causal_context_features(
        _prefix_rows(), checkpoint_elapsed_min=120.0, event_duration_min=150.0,
        rainfall_forecast=[(0.0, 30.0), (60.0, 30.0), (90.0, 10.0), (120.0, 5.0), (140.0, 20.0)],
        priority_nodes=["P1"], storage_nodes=["S1"], downstream_nodes=["D1"],
        storage_capacity={"S1": 5.0}, node_freeboard={"D1": 2.0},
        controller_memory={"override_active": True},
    )
    assert feats.shape == (len(CAUSAL_FEATURE_NAMES),)
    values = dict(zip(CAUSAL_FEATURE_NAMES, feats))
    # Priority flooding accumulated (P1 floods 40..80 -> five 10-min steps).
    assert values["cumulative_PFV_before_checkpoint"] > 0.0
    assert values["cumulative_priority_duration_before_checkpoint"] > 0.0
    # Total flooding is at least the priority flooding.
    assert values["cumulative_TFV_before_checkpoint"] >= values["cumulative_PFV_before_checkpoint"]
    # Only the post-checkpoint hyetograph tail (t=140) feeds the remaining peak.
    assert values["operational_forecast_remaining_peak"] == 20.0
    assert values["operational_forecast_time_to_peak"] == pytest.approx(20.0)
    assert values["elapsed_fraction"] == pytest.approx(120.0 / 150.0)
    assert values["storage_headroom_mean"] == pytest.approx(2.0)
    assert values["controller_memory_override_active"] == 1.0
    # A recession fallback switch happened inside the trailing 60-min window.
    assert values["previous_60min_candidate_fallback_switches"] >= 1.0


def test_causal_feature_names_are_distinct_from_h120_local_context() -> None:
    assert len(set(CAUSAL_FEATURE_NAMES)) == len(CAUSAL_FEATURE_NAMES)
    # The causal features add path-dependent signal beyond the local H120 context.
    # They are namespaced separately (v4_causal_* vs v4_ctx_*), so the only
    # allowed shared name is the intentional elapsed_fraction anchor.
    assert (set(CAUSAL_FEATURE_NAMES) & set(CONTEXT_FEATURE_NAMES)) == {"elapsed_fraction"}

