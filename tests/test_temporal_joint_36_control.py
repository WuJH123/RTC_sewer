from __future__ import annotations

import numpy as np
import pandas as pd


def _actuators() -> pd.DataFrame:
    rows = []
    for index in range(36):
        actuator_id = f"R{index:02d}"
        link_type = "orifice"
        role = "regulator"
        if index in (34, 35):
            actuator_id = f"ADD301.{index - 32}"
            link_type = "pump"
            role = "pump"
        elif index in (26, 28, 30):
            actuator_id = f"RTC_IN_{(index - 24) // 2:02d}"
            role = "storage_inlet"
        elif index in (27, 29, 31):
            actuator_id = f"RTC_OUT_{(index - 25) // 2:02d}"
            role = "storage_outlet"
        rows.append(
            {
                "actuator_id": actuator_id,
                "link_type": link_type,
                "asset_role": role,
                "storage_control_type": role if role.startswith("storage_") else "",
                "is_legacy_v8": index < 26,
                "retrofit_storage_group": (
                    f"S{(index - 26) // 2}" if 26 <= index <= 31 else ""
                ),
            }
        )
    return pd.DataFrame(rows)


def test_hierarchical_generator_retains_raw_horizon_and_actuator_axes():
    from sewerrtc.control.temporal_joint_candidate_search import (
        TemporalJointCandidateConfig,
        generate_temporal_joint_candidates,
    )

    actuators = _actuators()
    reference = np.ones((6, 36), dtype=np.float32)
    reference[:, -2:] = 0.0
    candidates = generate_temporal_joint_candidates(
        reference_action_seq=reference,
        actuators=actuators,
        legacy_groups=[["R00", "R01", "R02"]],
        paired_groups=[["R03", "RTC_OUT_01"]],
        phase="rising",
        config=TemporalJointCandidateConfig(
            horizon_steps=6,
            max_candidates=128,
            max_simultaneous_changes=6,
            continuous_max_delta=0.10,
            max_change_points=2,
            binary_pump_ids=("ADD301.2", "ADD301.3"),
            binary_pump_min_dwell_steps=2,
        ),
    )

    stack = np.stack([item["candidate_action_seq"] for item in candidates])
    assert stack.ndim == 3
    assert stack.shape[1:] == (6, 36)
    assert np.array_equal(stack[0], reference)
    assert any(item["tier"] == 1 and len(item["target_actuators"]) > 1 for item in candidates)
    assert any(item["tier"] == 2 for item in candidates)
    assert any(item["tier"] == 3 for item in candidates)
    assert any(
        not np.array_equal(item["candidate_action_seq"][0], item["candidate_action_seq"][1])
        for item in candidates[1:]
    )


def test_retrofit_manifest_enrichment_preserves_legacy_and_storage_pair_identity():
    from sewerrtc.control.actuator_scope import enrich_temporal_joint_actuator_semantics

    actuators = pd.DataFrame({
        "actuator_id": ["cc006.1", "RTC_IN_01", "RTC_OUT_01"],
        "storage_control_type": ["not_storage", "storage_inlet", "storage_outlet"],
    })
    manifest = pd.DataFrame({
        "actuator_id": ["RTC_IN_01", "RTC_OUT_01"],
        "asset_class": ["storage_inlet", "storage_outlet"],
        "storage_node": ["RTC_ST_01", "RTC_ST_01"],
        "inlet_or_outlet": ["inlet", "outlet"],
    })
    enriched = enrich_temporal_joint_actuator_semantics(actuators, manifest).set_index("actuator_id")

    assert bool(enriched.loc["cc006.1", "is_legacy_v8"])
    assert not bool(enriched.loc["RTC_IN_01", "is_legacy_v8"])
    assert enriched.loc["RTC_IN_01", "retrofit_storage_group"] == "RTC_ST_01"
    assert enriched.loc["RTC_OUT_01", "retrofit_storage_group"] == "RTC_ST_01"


def test_generator_enforces_binary_pumps_limits_and_storage_interlock():
    from sewerrtc.control.temporal_joint_candidate_search import (
        TemporalJointCandidateConfig,
        generate_temporal_joint_candidates,
        validate_candidate_sequence,
    )

    actuators = _actuators()
    reference = np.ones((6, 36), dtype=np.float32)
    reference[:, -2:] = 0.0
    config = TemporalJointCandidateConfig(
        horizon_steps=6,
        max_candidates=256,
        max_simultaneous_changes=4,
        continuous_max_delta=0.10,
        binary_pump_ids=("ADD301.2", "ADD301.3"),
        binary_pump_min_dwell_steps=2,
        storage_interlock=True,
    )
    candidates = generate_temporal_joint_candidates(
        reference_action_seq=reference,
        actuators=actuators,
        legacy_groups=[["R00", "R01", "R02", "R03"]],
        paired_groups=[["RTC_IN_01", "RTC_OUT_01"]],
        phase="peak",
        config=config,
    )
    ids = actuators["actuator_id"].tolist()
    pump_indices = [ids.index("ADD301.2"), ids.index("ADD301.3")]
    for item in candidates:
        seq = item["candidate_action_seq"]
        assert set(np.unique(seq[:, pump_indices])).issubset({0.0, 1.0})
        report = validate_candidate_sequence(seq, reference, actuators, config)
        assert report["valid"], report
        assert report["max_simultaneous_changes"] <= 4
        assert report["pump_dwell_violations"] == 0
        assert report["storage_interlock_violations"] == 0


def test_lexicographic_gate_uses_pfv_peak_then_maximizes_tfv_lcb():
    from sewerrtc.control.temporal_joint_safety import (
        JointCandidatePrediction,
        JointSafetyConfig,
        select_lexicographic_candidate,
    )

    predictions = [
        JointCandidatePrediction("reference", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0),
        JointCandidatePrediction("unsafe_pfv", 120.0, -900.0, -1.0, 0.0, 0.0, 0.0, 2, 0.1, 0),
        JointCandidatePrediction("unsafe_peak", 20.0, -1000.0, 0.8, 0.0, 0.0, 0.0, 2, 0.1, 0),
        JointCandidatePrediction("safe_small", 20.0, -400.0, -0.2, 10.0, 80.0, 0.1, 1, 0.05, 0),
        JointCandidatePrediction("safe_best", 40.0, -700.0, -0.1, 10.0, 80.0, 0.1, 3, 0.15, 1),
    ]
    selected, audit = select_lexicographic_candidate(
        predictions,
        reference_pfv=1000.0,
        config=JointSafetyConfig(
            pfv_abs_margin_m3=100.0,
            pfv_rel_margin=0.005,
            peak_margin=0.5,
            uncertainty_z=1.0,
            min_tfv_lcb_reduction=0.0,
        ),
    )

    assert selected.label == "safe_best"
    assert audit["unsafe_pfv"]["rejection_reason"] == "pfv_noninferiority"
    assert audit["unsafe_peak"]["rejection_reason"] == "peak_safety"


def test_no_reliable_tfv_benefit_falls_back_to_reference():
    from sewerrtc.control.temporal_joint_safety import (
        JointCandidatePrediction,
        JointSafetyConfig,
        select_lexicographic_candidate,
    )

    reference = JointCandidatePrediction("reference", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0)
    uncertain = JointCandidatePrediction("uncertain", 0.0, -100.0, 0.0, 1.0, 150.0, 0.1, 1, 0.1, 0)
    selected, audit = select_lexicographic_candidate(
        [reference, uncertain],
        reference_pfv=5000.0,
        config=JointSafetyConfig(peak_margin=0.5, min_tfv_lcb_reduction=10.0),
    )
    assert selected.label == "reference"
    assert audit["uncertain"]["rejection_reason"] == "insufficient_tfv_lcb"


def test_pfv_noninferiority_uses_two_percent_relative_margin_when_larger_than_absolute():
    from sewerrtc.control.temporal_joint_safety import (
        JointCandidatePrediction,
        JointSafetyConfig,
        select_lexicographic_candidate,
    )

    reference = JointCandidatePrediction("reference", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0)
    within = JointCandidatePrediction("within", 150.0, -500.0, -0.1, 0.0, 0.0, 0.0, 1, 0.1, 0)
    beyond = JointCandidatePrediction("beyond", 210.0, -900.0, -0.1, 0.0, 0.0, 0.0, 1, 0.1, 0)

    selected, audit = select_lexicographic_candidate(
        [reference, within, beyond],
        reference_pfv=10_000.0,
        config=JointSafetyConfig(pfv_abs_margin_m3=100.0, pfv_rel_margin=0.02),
    )

    assert selected.label == "within"
    assert audit["within"]["pfv_margin"] == 200.0
    assert audit["beyond"]["rejection_reason"] == "pfv_noninferiority"


def test_classifier_probabilities_are_hard_safety_evidence_before_tfv_ranking():
    from sewerrtc.control.temporal_joint_safety import (
        JointCandidatePrediction,
        JointSafetyConfig,
        select_lexicographic_candidate,
    )

    reference = JointCandidatePrediction("reference", 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0)
    false_safe = JointCandidatePrediction(
        "low_probability", 10.0, -2000.0, -0.2, 1.0, 10.0, 0.01, 1, 0.1, 0,
        pfv_noninferiority_probability=0.60,
        tfv_improvement_probability=0.95,
        peak_safe_probability=0.95,
    )
    reliable = JointCandidatePrediction(
        "reliable", 20.0, -700.0, -0.1, 1.0, 10.0, 0.01, 1, 0.1, 0,
        pfv_noninferiority_probability=0.95,
        tfv_improvement_probability=0.80,
        peak_safe_probability=0.90,
    )

    selected, audit = select_lexicographic_candidate(
        [reference, false_safe, reliable],
        reference_pfv=5000.0,
        config=JointSafetyConfig(
            pfv_rel_margin=0.02,
            min_pfv_noninferiority_probability=0.90,
            min_tfv_improvement_probability=0.60,
            min_peak_safe_probability=0.85,
        ),
    )

    assert selected.label == "reliable"
    assert audit["low_probability"]["rejection_reason"] == "pfv_classifier"


def test_temporal_joint_controller_scores_raw_batch_and_executes_only_first_row():
    from sewerrtc.control.temporal_joint_36_controller import TemporalJoint36Controller
    from sewerrtc.control.temporal_joint_candidate_search import TemporalJointCandidateConfig
    from sewerrtc.control.temporal_joint_safety import JointSafetyConfig

    class FakeRawPredictor:
        def __init__(self):
            self.calls = []

        def predict_many(self, **kwargs):
            self.calls.append(kwargs)
            candidates = kwargs["candidate_action_seq"]
            residual = candidates - kwargs["reference_action_seq"]
            magnitude = np.abs(residual).sum(axis=(1, 2))
            # Every non-reference action is predicted PFV/peak safe and TFV
            # improving. The largest sparse action is selected by TFV LCB.
            return {
                "reference_PFV_H": np.full(len(candidates), 1000.0),
                "delta_PFV_H": np.where(magnitude > 0, 10.0, 0.0),
                "delta_TFV_H": -100.0 * magnitude,
                "delta_peak": np.where(magnitude > 0, -0.1, 0.0),
                "delta_PFV_sigma": np.zeros(len(candidates)),
                "delta_TFV_sigma": np.zeros(len(candidates)),
                "delta_peak_sigma": np.zeros(len(candidates)),
            }

    predictor = FakeRawPredictor()
    actuators = _actuators()
    controller = TemporalJoint36Controller(
        actuators=actuators,
        predictor=predictor,
        candidate_config=TemporalJointCandidateConfig(max_candidates=64, max_simultaneous_changes=4),
        safety_config=JointSafetyConfig(peak_margin=0.0),
        legacy_groups=[["R00", "R01", "R02"]],
        paired_groups=[["R03", "R04"]],
    )
    reference = np.ones((6, 36), dtype=np.float32)
    reference[:, -2:] = 0.0
    first, first_info = controller.choose(
        reconstructed_state=np.zeros(932, dtype=np.float32),
        rainfall_window=np.ones((6, 1), dtype=np.float32),
        reference_action_sequence=reference,
        phase="rising",
    )
    second, second_info = controller.choose(
        reconstructed_state=np.ones(932, dtype=np.float32),
        rainfall_window=np.zeros((6, 1), dtype=np.float32),
        reference_action_sequence=reference,
        phase="peak",
    )

    assert predictor.calls[0]["candidate_action_seq"].ndim == 3
    assert predictor.calls[0]["candidate_action_seq"].shape[1:] == (6, 36)
    assert predictor.calls[0]["reference_action_seq"].shape == predictor.calls[0]["candidate_action_seq"].shape
    assert first.shape == (36,)
    assert np.array_equal(first, np.asarray(first_info["selected_action_sequence"])[0])
    assert len(predictor.calls) == 2
    assert first_info["decision_index"] == 0
    assert second_info["decision_index"] == 1
    assert first_info["simultaneous_actuator_count"] <= 4
    assert second.shape == (36,)


class _FakeLegacyHorizonPredictor:
    def __init__(self, *, safe: bool = True):
        self.safe = safe
        self.calls = []

    def predict_many(self, sequences, contexts):
        self.calls.append((sequences, contexts))
        out = []
        reference = np.asarray(sequences[0])
        for sequence in sequences:
            changed = float(np.abs(np.asarray(sequence) - reference).sum())
            out.append({
                "pfv_upper": np.full(6, (1000.0 + (10.0 if self.safe else 25_000.0) * changed) / 6.0),
                "tfv_upper": np.full(6, (5000.0 - 100.0 * changed) / 6.0),
                "peak_tfv_rate_upper": np.full(6, 10.0 - 0.1 * changed),
            })
        return out


class _FakeResidualPredictor:
    def __init__(self, *, safe: bool = True):
        self.safe = safe
        self.calls = []

    def predict_many(self, **kwargs):
        self.calls.append(kwargs)
        candidate = kwargs["candidate_action_seq"]
        reference = kwargs["reference_action_seq"]
        residual = np.abs(candidate - reference).sum(axis=(1, 2))
        return {
            "reference_PFV_H": np.full(len(candidate), 1000.0),
            "delta_PFV_H": np.where(residual > 0, 20.0 if self.safe else 300.0, 0.0),
            "delta_TFV_H": -200.0 * residual,
            "delta_peak": np.where(residual > 0, -0.2, 0.0),
            "delta_PFV_sigma": np.zeros(len(candidate)),
            "delta_TFV_sigma": np.zeros(len(candidate)),
            "delta_peak_sigma": np.zeros(len(candidate)),
        }


def _hierarchical_controller(*, legacy_safe=True, residual_safe=True, residual_enabled=True):
    from sewerrtc.control.temporal_joint_36_controller import TemporalJoint36Controller
    from sewerrtc.control.temporal_joint_candidate_search import TemporalJointCandidateConfig
    from sewerrtc.control.temporal_joint_safety import JointSafetyConfig

    legacy = _FakeLegacyHorizonPredictor(safe=legacy_safe)
    residual = _FakeResidualPredictor(safe=residual_safe)
    controller = TemporalJoint36Controller(
        actuators=_actuators(),
        predictor=residual,
        legacy_predictor=legacy,
        residual_actuator_ids=["RTC_IN_01", "RTC_OUT_01", "ADD301.2", "ADD301.3"],
        residual_enabled=residual_enabled,
        candidate_config=TemporalJointCandidateConfig(
            max_candidates=96,
            max_simultaneous_changes=6,
            continuous_max_delta=0.10,
            binary_pump_ids=("ADD301.2", "ADD301.3"),
        ),
        safety_config=JointSafetyConfig(pfv_abs_margin_m3=100.0, pfv_rel_margin=0.02),
        legacy_groups=[["R00", "R01", "R02"]],
    )
    return controller, legacy, residual


def test_hierarchical_controller_executes_tier1_when_residual_is_disabled():
    controller, legacy, residual = _hierarchical_controller(residual_enabled=False)
    reference = np.ones((6, 36), dtype=np.float32)
    reference[:, -2:] = 0.0
    action, info = controller.choose(
        reconstructed_state=np.zeros(932, dtype=np.float32),
        rainfall_window=np.ones((6, 1), dtype=np.float32),
        reference_action_sequence=reference,
        phase="peak",
    )

    assert legacy.calls
    assert not residual.calls
    scored = np.asarray(legacy.calls[0][0])
    assert np.all(np.any(np.abs(scored[1:, 0, :] - reference[None, 0, :]) > 1.0e-7, axis=1))
    assert info["selected_tier"] == 1
    assert info["fallback_path"] == "tier1"
    assert info["fallback_to_default"] is False
    assert info["selected_gate_pass"] is True
    assert info["candidate_count"] > 1
    assert not np.array_equal(action, reference[0])


def test_hierarchical_tier2_changes_only_residual_assets_on_top_of_tier1():
    controller, _, residual = _hierarchical_controller()
    reference = np.ones((6, 36), dtype=np.float32)
    reference[:, -2:] = 0.0
    _, info = controller.choose(
        reconstructed_state=np.zeros(932, dtype=np.float32),
        rainfall_window=np.ones((6, 1), dtype=np.float32),
        reference_action_sequence=reference,
        phase="recession",
    )

    assert residual.calls
    call = residual.calls[0]
    base = np.asarray(info["selected_tier1_sequence"])
    candidates = call["candidate_action_seq"]
    assert np.all(np.any(np.abs(candidates[1:, 0, :] - base[None, 0, :]) > 1.0e-7, axis=1))
    ids = _actuators()["actuator_id"].tolist()
    residual_indices = {ids.index(aid) for aid in ("RTC_IN_01", "RTC_OUT_01", "ADD301.2", "ADD301.3")}
    changed = np.any(np.abs(candidates - base[None, :, :]) > 1.0e-7, axis=(0, 1))
    assert set(np.flatnonzero(changed)).issubset(residual_indices)
    assert info["selected_tier"] == 2
    assert info["fallback_path"] == "tier2"


def test_hierarchical_fallback_is_tier2_then_tier1_then_no_control():
    reference = np.ones((6, 36), dtype=np.float32)
    reference[:, -2:] = 0.0
    kwargs = dict(
        reconstructed_state=np.zeros(932, dtype=np.float32),
        rainfall_window=np.ones((6, 1), dtype=np.float32),
        reference_action_sequence=reference,
        phase="peak",
    )

    tier1_controller, _, _ = _hierarchical_controller(residual_safe=False)
    _, tier1_info = tier1_controller.choose(**kwargs)
    assert tier1_info["selected_tier"] == 1
    assert tier1_info["fallback_path"] == "tier1"

    no_control_controller, _, _ = _hierarchical_controller(legacy_safe=False)
    action, no_control_info = no_control_controller.choose(**kwargs)
    assert no_control_info["selected_tier"] == 0
    assert no_control_info["fallback_path"] == "no_control"
    assert no_control_info["fallback_to_default"] is True
    assert no_control_info["selected_gate_pass"] is False
    assert np.array_equal(action, reference[0])


def test_hierarchical_config_uses_v6_model_and_explicit_tier_sets():
    from sewerrtc.io.project_paths import load_config

    cfg = load_config("configs/wuhan_project6_36_hierarchical_residual_v1.yaml")
    temporal = cfg["controller"]["temporal_joint"]
    hierarchical = temporal["hierarchical"]

    assert temporal["model_path"].endswith(
        "models_temporal_joint_36_causal_effect_boundary_v6_round2/"
        "raw_joint_36_causal_effect_boundary_v6_round2.pt"
    )
    assert len(temporal["legacy_groups"]) == 3
    assert hierarchical["enabled"] is True
    assert hierarchical["require_residual_validation"] is True
    assert len(hierarchical["residual_actuator_ids"]) == 10
    assert set(hierarchical["residual_actuator_ids"]).isdisjoint(
        {actuator for group in temporal["legacy_groups"] for actuator in group}
    )


def test_closed_loop_runner_accepts_raw_joint_runtime_arguments():
    import inspect

    from sewerrtc.simulation.pyswmm_runner import run_swmm_mpc_closed_loop

    parameters = inspect.signature(run_swmm_mpc_closed_loop).parameters
    assert "raw_joint_model_path" in parameters
    assert "temporal_joint_config" in parameters
