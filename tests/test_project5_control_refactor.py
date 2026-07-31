import sys
from pathlib import Path
import subprocess

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def test_paper_policy_set_keeps_main_table_focused():
    from sewerrtc.evaluation.policy_sets import (
        paper_baseline_policy_ids,
        paper_policy_ids,
        split_policy_set_frames,
    )

    cfg = {
        "evaluation": {
            "paper_policy_set": [
                "proposed_gat_mpc",
                "internal_rules",
                "efd_storage_priority",
                "auto_rbc",
                "no_control_diagnostic",
            ],
            "diagnostic_policy_set": ["all_open", "random_safe", "proposed_native_shield"],
        }
    }
    df = pd.DataFrame(
        {
            "policy_id": [
                "proposed_pfv_first_mpc",
                "proposed_native_shield",
                "internal_rules",
                "efd_storage_priority",
                "auto_rbc",
                "no_control",
                "all_open",
                "random_safe",
            ],
            "PFV": [7, 8, 10, 9, 0, 6, 0, 1],
        }
    )

    main, diagnostic = split_policy_set_frames(df, cfg)

    assert paper_policy_ids(cfg)[0] == "proposed_gat_mpc"
    assert paper_baseline_policy_ids(cfg) == ["internal_rules", "efd_storage_priority", "auto_rbc", "no_control"]
    assert set(main["policy_id"]) == {
        "proposed_gat_mpc",
        "internal_rules",
        "efd_storage_priority",
        "auto_rbc",
        "no_control",
    }
    assert set(diagnostic["policy_id"]) == {"all_open", "random_safe", "proposed_native_shield"}


def test_action_sequence_generator_creates_multistep_influence_sequences():
    from sewerrtc.control.action_sequence_generator import generate_action_sequences

    actuators = pd.DataFrame(
        {
            "actuator_id": ["P1", "IN1", "OUT1"],
            "link_type": ["pump", "orifice", "orifice"],
            "storage_control_type": ["", "storage_inlet", "storage_outlet"],
            "from_node": ["U1", "U2", "S1"],
            "to_node": ["D1", "S1", "D2"],
        }
    )
    influence = pd.DataFrame(
        {
            "actuator_id": ["IN1", "OUT1"],
            "priority_node": ["PX", "PX"],
            "asset_role": ["storage_inlet", "storage_outlet"],
            "influence_path_length": [1, 2],
        }
    )

    seqs = generate_action_sequences(
        np.full(3, 0.5, dtype=np.float32),
        actuators,
        horizon_steps=4,
        max_delta=0.08,
        priority_to_actuators=influence,
        include_hold=True,
    )

    assert seqs
    assert all(s["sequence"].shape == (4, 3) for s in seqs)
    labels = {s["label"] for s in seqs}
    assert any("restrict_then_release" in label for label in labels)
    assert any("retain_through_peak" in label for label in labels)
    assert all("internal" not in s.get("physical_rationale", "").lower() for s in seqs)


def test_action_sequence_generator_controls_influence_orifices_and_weirs():
    from sewerrtc.control.action_sequence_generator import generate_action_sequences

    actuators = pd.DataFrame(
        {
            "actuator_id": ["OR1", "WR1", "P1"],
            "link_type": ["orifice", "weir", "pump"],
            "storage_control_type": ["not_storage", "not_storage", ""],
            "from_node": ["U1", "U2", "U3"],
            "to_node": ["D1", "D2", "D3"],
        }
    )
    influence = pd.DataFrame(
        {
            "actuator_id": ["OR1", "WR1"],
            "priority_node": ["PX", "PX"],
            "asset_role": ["orifice", "weir"],
            "influence_path_length": [5, 6],
        }
    )

    seqs = generate_action_sequences(
        np.full(3, 0.5, dtype=np.float32),
        actuators,
        horizon_steps=4,
        max_delta=0.10,
        priority_to_actuators=influence,
        include_hold=True,
    )
    labels = {s["label"] for s in seqs}

    assert any("regulator_restrict_then_restore" in label and "OR1" in label for label in labels)
    assert any("regulator_release_if_safe" in label and "WR1" in label for label in labels)


def test_action_sequence_generator_adds_temporal_profiles_and_caps_pool():
    from sewerrtc.control.action_sequence_generator import generate_action_sequences

    actuators = pd.DataFrame(
        {
            "actuator_id": ["IN1", "OUT1", "OR1", "WR1", "P1", "P2"],
            "link_type": ["orifice", "orifice", "orifice", "weir", "pump", "pump"],
            "storage_control_type": ["storage_inlet", "storage_outlet", "not_storage", "not_storage", "", ""],
            "from_node": ["U1", "S1", "U2", "U3", "U4", "U5"],
            "to_node": ["S1", "D1", "D2", "D3", "D4", "D5"],
        }
    )
    influence = pd.DataFrame(
        {
            "actuator_id": ["IN1", "OUT1", "OR1", "WR1", "P1", "P2"],
            "priority_node": ["PX"] * 6,
            "asset_role": ["storage_inlet", "storage_outlet", "orifice", "weir", "pump", "pump"],
            "influence_path_length": [1, 1, 2, 2, 3, 4],
        }
    )

    seqs = generate_action_sequences(
        np.full(6, 0.5, dtype=np.float32),
        actuators,
        horizon_steps=6,
        max_delta=0.10,
        priority_to_actuators=influence,
        include_hold=True,
        max_sequences=12,
        group_limit=4,
    )
    labels = {s["label"] for s in seqs}

    assert 1 < len(seqs) <= 12
    assert any("ramp" in label for label in labels)
    assert any("pulse" in label for label in labels)
    assert any("priority_group_regulator" in label for label in labels)
    assert all(s["sequence"].shape == (6, 6) for s in seqs)


def test_influence_domain_expands_to_storage_regulator_and_pump_role_quotas():
    from sewerrtc.network.influence_domain import build_priority_influence_domains

    link_table = pd.DataFrame(
        {
            "link_id": ["L1", "L2", "L3"],
            "from_node": ["P1", "N1", "N2"],
            "to_node": ["N1", "N2", "N3"],
        }
    )
    actuator_table = pd.DataFrame(
        {
            "actuator_id": [
                "RTC_IN_01",
                "RTC_OUT_01",
                "RTC_IN_02",
                "RTC_OUT_02",
                "OR1",
                "WR1",
                "PUMP1",
                "PUMP2",
            ],
            "link_type": ["orifice", "orifice", "orifice", "orifice", "orifice", "weir", "pump", "pump"],
            "storage_control_type": [
                "storage_inlet",
                "storage_outlet",
                "storage_inlet",
                "storage_outlet",
                "not_storage",
                "not_storage",
                "not_storage",
                "not_storage",
            ],
            "from_node": ["FAR1", "S1", "FAR2", "S2", "FAR3", "FAR4", "FAR5", "FAR6"],
            "to_node": ["S1", "FAR1", "S2", "FAR2", "FAR7", "FAR8", "FAR9", "FAR10"],
        }
    )

    _, candidates = build_priority_influence_domains(
        link_table,
        actuator_table,
        ["P1"],
        k=1,
        fallback_k=1,
        max_candidates_per_priority=20,
        include_global_storage_controls=True,
        include_global_regulators=True,
        include_global_pumps=True,
        max_storage_controls_per_priority=4,
        max_regulators_per_priority=2,
        max_pumps_per_priority=2,
    )

    by_role = candidates.groupby("asset_role")["actuator_id"].nunique().to_dict()
    assert by_role.get("storage_inlet", 0) == 2
    assert by_role.get("storage_outlet", 0) == 2
    assert by_role.get("orifice", 0) >= 1
    assert by_role.get("weir", 0) >= 1
    assert by_role.get("pump", 0) == 2
    assert "global_storage_control_pool" in set(candidates["candidate_source"])


def test_horizon_objective_scores_predicted_sequences_with_hard_safety_penalty():
    from sewerrtc.control.horizon_objective import score_horizon_sequence

    safe = score_horizon_sequence(
        pfv=[20.0, 15.0, 5.0],
        tfv=[100.0, 100.0, 100.0],
        peak_tfv_rate=[40.0, 35.0, 20.0],
        action_change=[0.02, 0.02, 0.01],
        reference_tfv=[100.0, 100.0, 100.0],
        reference_peak=[45.0, 40.0, 25.0],
        smooth_weight=0.1,
    )
    unsafe = score_horizon_sequence(
        pfv=[5.0, 5.0, 5.0],
        tfv=[100.0, 120.0, 100.0],
        peak_tfv_rate=[40.0, 60.0, 20.0],
        action_change=[0.02, 0.02, 0.01],
        reference_tfv=[100.0, 100.0, 100.0],
        reference_peak=[45.0, 40.0, 25.0],
        smooth_weight=0.1,
    )

    assert safe.gate_pass is True
    assert unsafe.gate_pass is False
    assert safe.score < unsafe.score


def test_horizon_objective_can_require_pfv_below_reference():
    from sewerrtc.control.horizon_objective import score_horizon_sequence

    improved = score_horizon_sequence(
        pfv=[8.0, 8.0, 8.0],
        tfv=[100.0, 100.0, 100.0],
        peak_tfv_rate=[30.0, 30.0, 30.0],
        action_change=[0.02, 0.02, 0.01],
        reference_pfv=[10.0, 10.0, 10.0],
        reference_tfv=[100.0, 100.0, 100.0],
        reference_peak=[35.0, 35.0, 35.0],
    )
    pfv_worse = score_horizon_sequence(
        pfv=[12.0, 12.0, 12.0],
        tfv=[100.0, 100.0, 100.0],
        peak_tfv_rate=[30.0, 30.0, 30.0],
        action_change=[0.02, 0.02, 0.01],
        reference_pfv=[10.0, 10.0, 10.0],
        reference_tfv=[100.0, 100.0, 100.0],
        reference_peak=[35.0, 35.0, 35.0],
    )

    assert improved.gate_pass is True
    assert pfv_worse.gate_pass is False
    assert pfv_worse.pfv_violation > 0


def test_horizon_objective_can_require_strict_pfv_improvement_margin():
    from sewerrtc.control.horizon_objective import score_horizon_sequence

    barely_equal = score_horizon_sequence(
        pfv=[10.0, 10.0, 10.0],
        tfv=[100.0, 100.0, 100.0],
        peak_tfv_rate=[30.0, 30.0, 30.0],
        action_change=[0.0, 0.0, 0.0],
        reference_pfv=[10.0, 10.0, 10.0],
        reference_tfv=[100.0, 100.0, 100.0],
        reference_peak=[35.0, 35.0, 35.0],
        pfv_required_improvement=1.0,
    )

    assert barely_equal.gate_pass is False
    assert barely_equal.pfv_violation >= 1.0


def test_horizon_objective_allows_configured_tfv_tradeoff_for_pfv_gain():
    from sewerrtc.control.horizon_objective import score_horizon_sequence

    accepted = score_horizon_sequence(
        pfv=[7.0, 7.0, 7.0],
        tfv=[101.0, 101.0, 101.0],
        peak_tfv_rate=[30.0, 30.0, 30.0],
        action_change=[0.02, 0.02, 0.01],
        reference_pfv=[10.0, 10.0, 10.0],
        reference_tfv=[100.0, 100.0, 100.0],
        reference_peak=[35.0, 35.0, 35.0],
        pfv_required_improvement=1.0,
        tfv_tolerance=3.0,
    )
    rejected = score_horizon_sequence(
        pfv=[7.0, 7.0, 7.0],
        tfv=[102.0, 102.0, 102.0],
        peak_tfv_rate=[30.0, 30.0, 30.0],
        action_change=[0.02, 0.02, 0.01],
        reference_pfv=[10.0, 10.0, 10.0],
        reference_tfv=[100.0, 100.0, 100.0],
        reference_peak=[35.0, 35.0, 35.0],
        pfv_required_improvement=1.0,
        tfv_tolerance=3.0,
    )

    assert accepted.gate_pass is True
    assert accepted.tfv_violation == 0.0
    assert rejected.gate_pass is False
    assert rejected.tfv_violation > 0.0


def test_generic_gat_mpc_executes_first_step_of_best_sequence():
    from sewerrtc.control.generic_gat_mpc import GenericGATMPCController

    actuators = pd.DataFrame(
        {
            "actuator_id": ["IN1", "OUT1"],
            "link_type": ["orifice", "orifice"],
            "storage_control_type": ["storage_inlet", "storage_outlet"],
            "from_node": ["U1", "S1"],
            "to_node": ["S1", "D1"],
        }
    )
    influence = pd.DataFrame(
        {
            "actuator_id": ["IN1"],
            "priority_node": ["P1"],
            "asset_role": ["storage_inlet"],
            "influence_path_length": [1],
        }
    )

    def predictor(sequence, context):
        label = context["label"]
        if "restrict_then_release" in label:
            return {
                "pfv": np.asarray([8.0, 6.0, 4.0]),
                "tfv": np.asarray([100.0, 100.0, 100.0]),
                "peak_tfv_rate": np.asarray([30.0, 30.0, 30.0]),
            }
        return {
            "pfv": np.asarray([20.0, 20.0, 20.0]),
            "tfv": np.asarray([100.0, 100.0, 100.0]),
            "peak_tfv_rate": np.asarray([30.0, 30.0, 30.0]),
        }

    ctrl = GenericGATMPCController(
        actuators,
        horizon_steps=3,
        max_candidate_delta=0.08,
        priority_to_actuators=influence,
        horizon_predictor=predictor,
    )
    current_action = np.full(2, 0.5, dtype=np.float32)
    action, info = ctrl.choose(
        reconstructed_state=np.asarray([1.2, 0.6], dtype=np.float32),
        rainfall_window=np.asarray([20.0, 15.0, 5.0], dtype=np.float32),
        current_action=current_action,
        reference_pfv=np.asarray([25.0, 25.0, 25.0]),
        reference_tfv=np.asarray([100.0, 100.0, 100.0]),
        reference_peak=np.asarray([35.0, 35.0, 35.0]),
        elapsed_min=30.0,
    )

    assert info["policy_id"] == "proposed_gat_mpc"
    assert info["fallback_to_default"] is False
    assert info["selected_sequence_label"].startswith("restrict_then_release")
    assert info["best_non_hold_sequence_label"].startswith("restrict_then_release")
    assert np.allclose(action, info["selected_sequence_first_action"])
    assert action[0] < current_action[0]


def test_generic_gat_mpc_falls_back_when_no_candidate_passes_safety_gate():
    from sewerrtc.control.generic_gat_mpc import GenericGATMPCController

    actuators = pd.DataFrame(
        {
            "actuator_id": ["IN1"],
            "link_type": ["orifice"],
            "storage_control_type": ["storage_inlet"],
            "from_node": ["U1"],
            "to_node": ["S1"],
        }
    )
    influence = pd.DataFrame(
        {
            "actuator_id": ["IN1"],
            "priority_node": ["P1"],
            "asset_role": ["storage_inlet"],
            "influence_path_length": [1],
        }
    )

    def predictor(sequence, context):
        label = context["label"]
        pfv = 5.0 if "restrict_then_release" in label else 20.0
        return {
            "pfv": np.asarray([pfv, pfv, pfv]),
            "tfv": np.asarray([200.0, 200.0, 200.0]),
            "peak_tfv_rate": np.asarray([80.0, 80.0, 80.0]),
        }

    ctrl = GenericGATMPCController(
        actuators,
        horizon_steps=3,
        max_candidate_delta=0.08,
        priority_to_actuators=influence,
        horizon_predictor=predictor,
    )
    current_action = np.asarray([0.5], dtype=np.float32)

    action, info = ctrl.choose(
        reconstructed_state=np.asarray([1.0], dtype=np.float32),
        rainfall_window=np.asarray([20.0, 20.0, 20.0], dtype=np.float32),
        current_action=current_action,
        reference_pfv=np.asarray([30.0, 30.0, 30.0]),
        reference_tfv=np.asarray([100.0, 100.0, 100.0]),
        reference_peak=np.asarray([40.0, 40.0, 40.0]),
        elapsed_min=30.0,
    )

    assert info["fallback_to_default"] is True
    assert info["intervention_reason"] == "no_safe_sequence"
    assert info["selected_sequence_label"] == "hold_native"
    assert info["best_non_hold_gate_pass"] is False
    assert np.allclose(action, current_action)


def test_generic_gat_mpc_accepts_pfv_gain_with_bounded_tfv_tradeoff():
    from sewerrtc.control.generic_gat_mpc import GenericGATMPCController

    actuators = pd.DataFrame(
        {
            "actuator_id": ["IN1"],
            "link_type": ["orifice"],
            "storage_control_type": ["storage_inlet"],
            "from_node": ["U1"],
            "to_node": ["S1"],
        }
    )
    influence = pd.DataFrame(
        {
            "actuator_id": ["IN1"],
            "priority_node": ["P1"],
            "asset_role": ["storage_inlet"],
            "influence_path_length": [1],
        }
    )

    def predictor(sequence, context):
        label = context["label"]
        if "restrict_then_release" in label:
            return {
                "pfv": np.asarray([7.0, 7.0, 7.0]),
                "tfv": np.asarray([101.0, 101.0, 101.0]),
                "peak_tfv_rate": np.asarray([30.0, 30.0, 30.0]),
            }
        return {
            "pfv": np.asarray([10.0, 10.0, 10.0]),
            "tfv": np.asarray([100.0, 100.0, 100.0]),
            "peak_tfv_rate": np.asarray([30.0, 30.0, 30.0]),
        }

    ctrl = GenericGATMPCController(
        actuators,
        horizon_steps=3,
        max_candidate_delta=0.08,
        priority_to_actuators=influence,
        horizon_predictor=predictor,
        min_pfv_improvement_abs=1.0,
        tfv_tolerance_frac=0.01,
    )
    current_action = np.asarray([0.5], dtype=np.float32)
    action, info = ctrl.choose(
        reconstructed_state=np.asarray([1.0], dtype=np.float32),
        rainfall_window=np.asarray([20.0, 20.0, 20.0], dtype=np.float32),
        current_action=current_action,
        reference_pfv=np.asarray([10.0, 10.0, 10.0]),
        reference_tfv=np.asarray([100.0, 100.0, 100.0]),
        reference_peak=np.asarray([35.0, 35.0, 35.0]),
        elapsed_min=30.0,
    )

    assert info["fallback_to_default"] is False
    assert info["selected_sequence_label"].startswith("restrict_then_release")
    assert info["selected_tfv_tolerance"] == 3.0
    assert info["selected_tfv_violation"] == 0.0
    assert action[0] < current_action[0]


def test_generic_gat_mpc_caps_expanded_candidate_sequences():
    from sewerrtc.control.generic_gat_mpc import GenericGATMPCController

    actuators = pd.DataFrame(
        {
            "actuator_id": ["IN1", "OUT1", "OR1", "WR1", "P1", "P2"],
            "link_type": ["orifice", "orifice", "orifice", "weir", "pump", "pump"],
            "storage_control_type": ["storage_inlet", "storage_outlet", "not_storage", "not_storage", "", ""],
            "near_storage": [True, True, False, False, False, False],
        }
    )
    influence = pd.DataFrame(
        {
            "actuator_id": ["IN1", "OUT1", "OR1", "WR1", "P1", "P2"],
            "priority_node": ["P1"] * 6,
            "asset_role": ["storage_inlet", "storage_outlet", "orifice", "weir", "pump", "pump"],
            "influence_path_length": [1, 1, 2, 2, 3, 4],
        }
    )

    def predictor(sequence, context):
        return {
            "pfv": np.asarray([5.0, 5.0, 5.0]),
            "tfv": np.asarray([90.0, 90.0, 90.0]),
            "peak_tfv_rate": np.asarray([20.0, 20.0, 20.0]),
        }

    ctrl = GenericGATMPCController(
        actuators,
        horizon_steps=3,
        max_candidate_delta=0.08,
        priority_to_actuators=influence,
        horizon_predictor=predictor,
        max_candidate_sequences=5,
        candidate_group_limit=3,
    )
    _, info = ctrl.choose(
        reconstructed_state=np.asarray([0.5], dtype=np.float32),
        rainfall_window=np.asarray([10.0, 10.0, 10.0], dtype=np.float32),
        current_action=np.full(6, 0.5, dtype=np.float32),
        reference_pfv=np.asarray([20.0, 20.0, 20.0]),
        reference_tfv=np.asarray([100.0, 100.0, 100.0]),
        reference_peak=np.asarray([30.0, 30.0, 30.0]),
    )

    assert info["candidate_sequence_count"] <= 5
    assert info["candidate_sequence_cap"] == 5


def test_project5_candidate_filters_block_broad_actions():
    from sewerrtc.control.candidate_generator import generate_candidate_specs

    actuators = pd.DataFrame(
        {
            "actuator_id": ["PUMP1", "IN1", "OUT1", "L1"],
            "link_type": ["pump", "orifice", "orifice", "conduit"],
            "storage_control_type": ["", "storage_inlet", "storage_outlet", ""],
            "near_storage": [False, True, True, False],
            "from_node": ["N1", "P1", "P1", "X1"],
            "to_node": ["N2", "S1", "D1", "X2"],
            "from_index": [0, 1, 2, 3],
            "to_index": [0, 1, 2, 3],
            "has_internal_rule": [True, True, True, False],
        }
    )
    state = np.asarray([2.0, 0.8, 0.7, 0.1], dtype=np.float32)

    specs = generate_candidate_specs(
        np.full(len(actuators), 0.5, dtype=np.float32),
        actuators,
        "recession",
        max_delta=0.08,
        state=state,
        hold_steps=(1, 2),
        priority_upstream_nodes={"P1"},
        priority_downstream_nodes={"D1"},
        allowed_templates={"pump_boost", "storage_outlet_release"},
        allowed_scopes_by_template={
            "pump_boost": {"hot_local"},
            "storage_outlet_release": {"priority_corridor"},
        },
        blocked_templates={"release_plus_pump_boost", "pump_throttle"},
    )

    assert specs
    assert {s.template for s in specs} <= {"pump_boost", "storage_outlet_release"}
    assert "release_plus_pump_boost" not in {s.template for s in specs}
    assert "pump_throttle" not in {s.template for s in specs}
    assert {s.hold_steps for s in specs} <= {1, 2}
    assert all(abs(s.delta) <= 0.080001 for s in specs)
    assert all(s.scope == "hot_local" for s in specs if s.template == "pump_boost")
    assert all(s.scope == "priority_corridor" for s in specs if s.template == "storage_outlet_release")


def test_training_action_policies_include_facility_sweep_counterfactuals():
    from sewerrtc.simulation.action_policies import GenericActionPolicy, PolicyContext

    actuators = pd.DataFrame(
        {
            "actuator_id": ["IN1", "OUT1", "OR1", "WR1", "P1"],
            "link_type": ["orifice", "orifice", "orifice", "weir", "pump"],
            "storage_control_type": ["storage_inlet", "storage_outlet", "not_storage", "not_storage", ""],
            "near_storage": [True, True, False, False, False],
        }
    )
    ctx = PolicyContext(30.0, 120, 50.0, "peak", np.ones(5, dtype=np.float32))

    regulator = GenericActionPolicy("regulator_restrict_wave", actuators).action(ctx)
    storage = GenericActionPolicy("storage_inlet_outlet_sweep", actuators).action(ctx)
    pump = GenericActionPolicy("pump_station_wave", actuators).action(ctx)

    assert regulator[2] < 1.0 or regulator[3] < 1.0
    assert storage[0] != storage[1]
    assert pump[4] < 1.0


def test_horizon_objective_prefers_safe_pfv_gain_over_unsafe_larger_gain():
    from sewerrtc.control.horizon_objective import score_horizon_candidate

    safe = score_horizon_candidate(
        delta_pfv=-10.0,
        delta_tfv=0.0,
        delta_peak=0.0,
        action_change_penalty=0.05,
        baseline_tfv=1000.0,
        baseline_peak=100.0,
        tfv_guard_pct=0.005,
        peak_guard_pct=0.010,
    )
    unsafe = score_horizon_candidate(
        delta_pfv=-50.0,
        delta_tfv=20.0,
        delta_peak=5.0,
        action_change_penalty=0.05,
        baseline_tfv=1000.0,
        baseline_peak=100.0,
        tfv_guard_pct=0.005,
        peak_guard_pct=0.010,
    )

    assert safe.gate_pass is True
    assert unsafe.gate_pass is False
    assert safe.score < unsafe.score


def test_project5_formal_gate_enforces_pfv_and_safety_thresholds():
    from sewerrtc.evaluation.project5_formal_gate import evaluate_project5_gate

    event_ids = [f"E{i:02d}" for i in range(20)]
    rows = []
    for event_id in event_ids:
        rows.append(
            {
                "event_id": event_id,
                "event_risk_class": "high_risk_event",
                "policy_id": "no_control",
                "project5_priority_PFV": 220.0,
                "TFV": 1005.0,
                "peak_TFV_rate": 100.5,
                "action_changes": 0.0,
            }
        )
        rows.append(
            {
                "event_id": event_id,
                "event_risk_class": "high_risk_event",
                "policy_id": "auto_rbc",
                "project5_priority_PFV": 224.0,
                "TFV": 1008.0,
                "peak_TFV_rate": 100.8,
                "action_changes": 8.0,
            }
        )
        rows.append(
            {
                "event_id": event_id,
                "event_risk_class": "high_risk_event",
                "policy_id": "internal_rules",
                "project5_priority_PFV": 200.0,
                "TFV": 1000.0,
                "peak_TFV_rate": 100.0,
                "action_changes": 10.0,
            }
        )
        rows.append(
            {
                "event_id": event_id,
                "event_risk_class": "high_risk_event",
                "policy_id": "efd_storage_priority",
                "project5_priority_PFV": 210.0,
                "TFV": 1010.0,
                "peak_TFV_rate": 101.0,
                "action_changes": 12.0,
            }
        )
        rows.append(
            {
                "event_id": event_id,
                "event_risk_class": "high_risk_event",
                "policy_id": "proposed_gat_mpc",
                "project5_priority_PFV": 170.0,
                "TFV": 995.0,
                "peak_TFV_rate": 99.0,
                "action_changes": 9.0,
            }
        )
    paired = pd.DataFrame(rows)
    residual = pd.DataFrame(
        [
            {
                "PFV_direction_accuracy": 0.72,
                "safe_precision": 0.82,
                "peak_direction_accuracy": 0.83,
                "score": 1.0,
            }
        ]
    )
    guard = pd.DataFrame(
        [
            {
                "empirical_allow": True,
                "events": 12,
                "template_name": "storage_inlet_restrict",
            }
        ]
    )

    passed = evaluate_project5_gate(paired, residual, guard)
    assert passed["passed"] is True
    assert {row["baseline_policy"] for row in passed["baseline_comparisons"]} == {
        "no_control",
        "auto_rbc",
        "internal_rules",
        "efd_storage_priority",
    }

    weak = paired.copy()
    weak.loc[weak["policy_id"].eq("proposed_gat_mpc"), "project5_priority_PFV"] = 199.0
    failed = evaluate_project5_gate(weak, residual, guard)
    assert failed["passed"] is False
    assert any("PFV" in reason for reason in failed["reasons"])


def test_generic_gat_mpc_gate_does_not_require_legacy_residual_guard():
    from sewerrtc.evaluation.project5_formal_gate import evaluate_project5_gate

    rows = []
    for idx in range(20):
        event_id = f"E{idx:02d}"
        for policy in ["no_control", "efd_storage_priority", "auto_rbc"]:
            rows.append(
                {
                    "event_id": event_id,
                    "event_risk_class": "high_risk_event",
                    "policy_id": policy,
                    "project5_priority_PFV": 200.0,
                    "TFV": 1000.0,
                    "peak_TFV_rate": 100.0,
                    "action_changes": 0.0 if policy == "no_control" else 8.0,
                }
            )
        rows.append(
            {
                "event_id": event_id,
                "event_risk_class": "high_risk_event",
                "policy_id": "internal_rules",
                "project5_priority_PFV": 200.0,
                "TFV": 1000.0,
                "peak_TFV_rate": 100.0,
                "action_changes": 10.0,
            }
        )
        rows.append(
            {
                "event_id": event_id,
                "event_risk_class": "high_risk_event",
                "policy_id": "proposed_gat_mpc",
                "project5_priority_PFV": 160.0,
                "TFV": 995.0,
                "peak_TFV_rate": 98.0,
                "action_changes": 9.0,
            }
        )

    report = evaluate_project5_gate(pd.DataFrame(rows), pd.DataFrame(), pd.DataFrame())

    assert report["proposed_policy"] == "proposed_gat_mpc"
    assert report["legacy_residual_gate_applied"] is False
    assert report["passed"] is True


def test_project5_formal_gate_excludes_near_zero_reference_pfv_from_percentage_stats():
    from sewerrtc.evaluation.project5_formal_gate import Project5GateThresholds, evaluate_project5_gate

    rows = []
    required = ["internal_rules", "no_control", "efd_storage_priority", "auto_rbc"]
    for idx in range(20):
        event_id = f"E{idx:02d}"
        for policy in required:
            rows.append(
                {
                    "event_id": event_id,
                    "event_risk_class": "high_risk_event",
                    "policy_id": policy,
                    "project5_priority_PFV": 0.0 if policy == "no_control" and idx < 5 else 100.0,
                    "TFV": 1000.0,
                    "peak_TFV_rate": 100.0,
                    "action_changes": 1.0,
                }
            )
        rows.append(
            {
                "event_id": event_id,
                "event_risk_class": "high_risk_event",
                "policy_id": "all_open",
                "project5_priority_PFV": 1.0,
                "TFV": 2000.0,
                "peak_TFV_rate": 200.0,
                "action_changes": 1.0,
            }
        )
        rows.append(
            {
                "event_id": event_id,
                "event_risk_class": "high_risk_event",
                "policy_id": "proposed_gat_mpc",
                "project5_priority_PFV": 80.0,
                "TFV": 1000.0,
                "peak_TFV_rate": 100.0,
                "action_changes": 1.0,
            }
        )

    report = evaluate_project5_gate(
        pd.DataFrame(rows),
        pd.DataFrame(),
        pd.DataFrame(),
        thresholds=Project5GateThresholds(near_zero_pfv_epsilon=1.0),
    )
    no_control = [r for r in report["baseline_comparisons"] if r["baseline_policy"] == "no_control"][0]

    assert "all_open" not in {r["baseline_policy"] for r in report["baseline_comparisons"]}
    assert no_control["paired_events"] == 20
    assert no_control["near_zero_reference_events"] == 5
    assert no_control["PFV_percent_stat_events"] == 15
    assert np.isclose(no_control["PFV_mean_reduction_pct"], 20.0)


def test_project5_priority_config_separates_pfv_core_from_sentinels():
    from sewerrtc.io.priority_config import (
        combined_priority_depth_nodes,
        configured_priority_nodes,
        configured_priority_sentinel_nodes,
    )
    from sewerrtc.io.project_paths import load_config

    cfg = load_config(ROOT / "configs" / "wuhan.yaml")
    core = configured_priority_nodes(cfg)
    sentinels = configured_priority_sentinel_nodes(cfg)

    assert core == [
        "MSLBZW001",
        "HS1316314",
        "YS2530050",
        "HS2529198",
        "MH0200773",
        "HS1330349",
        "HS2529139",
        "HS2529052",
    ]
    assert sentinels == ["MH0200770", "HS1355904"]
    assert combined_priority_depth_nodes(cfg) == core + sentinels


def test_sensor_selection_can_disable_forced_priority_observation():
    from sewerrtc.graph.sensor_selection import select_sensors

    nodes = pd.DataFrame(
        {
            "node_id": ["P1", "N1", "N2", "N3"],
            "node_type": ["junction", "junction", "junction", "junction"],
            "max_depth": [1.0, 1.0, 1.0, 1.0],
            "degree_in": [0, 5, 4, 3],
            "degree_out": [0, 5, 4, 3],
        }
    )

    forced = select_sensors(nodes, ["P1"], sensor_ratio=0.25, seed=1, include_priority_nodes=True)
    unforced = select_sensors(nodes, ["P1"], sensor_ratio=0.25, seed=1, include_priority_nodes=False)

    assert forced["node_id"].tolist() == ["P1"]
    assert unforced["node_id"].tolist() == ["N1"]


def test_closed_loop_cli_exposes_generic_gat_mpc_controller():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "08_run_closed_loop.py"), "--help"],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--proposed-controller" in proc.stdout
    assert "generic_gat_mpc" in proc.stdout


def test_horizon_surrogate_training_cli_marks_ridge_as_baseline():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "43_train_horizon_surrogate.py"), "--help"],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--model-kind" in proc.stdout
    assert "ridge_baseline" in proc.stdout
    assert "temporal_gnn" in proc.stdout


def test_temporal_horizon_surrogate_round_trips_and_learns_action_effect(tmp_path):
    from sewerrtc.models.temporal_graph_surrogate import (
        TARGET_COLUMNS,
        TemporalGraphHorizonSurrogate,
        load_horizon_surrogate,
    )

    rows = []
    for i in range(80):
        action = 0.15 if i % 2 else 0.90
        rain = 10.0 + (i % 5)
        rows.append(
            {
                "rain_forecast_mean": rain,
                "priority_depth_max": 0.4,
                "priority_action_mean": action,
                "sequence_delta_abs_mean": abs(action - 0.5),
                "PFV_H": 10.0 if action < 0.3 else 100.0,
                "TFV_H": 500.0,
                "peak_TFV_rate_H": 50.0,
            }
        )
    df = pd.DataFrame(rows)
    feature_columns = [
        "rain_forecast_mean",
        "priority_depth_max",
        "priority_action_mean",
        "sequence_delta_abs_mean",
    ]
    model = TemporalGraphHorizonSurrogate(hidden_dim=16, layers=2, dropout=0.0, seed=7)
    model.fit(
        df,
        feature_columns,
        TARGET_COLUMNS,
        epochs=8,
        batch_size=16,
        lr=0.01,
        val_df=df,
        device="cpu",
        patience=4,
    )
    path = tmp_path / "horizon_temporal_gnn.pt"
    model.save(path)

    loaded = load_horizon_surrogate(path)
    probe = pd.DataFrame(
        [
            {
                "rain_forecast_mean": 12.0,
                "priority_depth_max": 0.4,
                "priority_action_mean": 0.90,
                "sequence_delta_abs_mean": 0.40,
            },
            {
                "rain_forecast_mean": 12.0,
                "priority_depth_max": 0.4,
                "priority_action_mean": 0.15,
                "sequence_delta_abs_mean": 0.35,
            },
        ]
    )
    pred = loaded.predict(probe)

    assert path.exists()
    assert set(pred.columns) == {f"pred_{c}" for c in TARGET_COLUMNS}
    assert float(pred.loc[1, "pred_PFV_H"]) < float(pred.loc[0, "pred_PFV_H"])


def test_gat_training_cli_exposes_staged_training_controls():
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "05_train_gat.py"), "--help"],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        check=True,
    )

    assert "--max-train-samples-per-epoch" in proc.stdout
    assert "--eval-every" in proc.stdout
    assert "--patience" in proc.stdout
    assert "--score-full-weight" in proc.stdout
    assert "--score-priority-weight" in proc.stdout


def test_generic_gat_mpc_pipeline_param_block_is_first_statement():
    script = ROOT / "scripts" / "project6_runs" / "RUN_PROJECT6_NO_CONTROL_REPAIR_PIPELINE.ps1"
    meaningful_lines = [
        line.strip()
        for line in script.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]

    assert meaningful_lines[0].lower().startswith("param(")


def test_generic_gat_mpc_pipeline_uses_fail_fast_python_wrapper():
    script = ROOT / "scripts" / "project6_runs" / "RUN_PROJECT6_NO_CONTROL_REPAIR_PIPELINE.ps1"
    text = script.read_text(encoding="utf-8")

    assert "function Invoke-PythonStep" in text
    assert "$LASTEXITCODE" in text
    assert "throw \"Python step failed" in text
    assert "& $Python scripts\\" not in text


def test_generic_gat_mpc_pipeline_passes_gat_staged_training_controls():
    script = ROOT / "scripts" / "project6_runs" / "RUN_PROJECT6_NO_CONTROL_REPAIR_PIPELINE.ps1"
    text = script.read_text(encoding="utf-8")

    assert "--max-train-samples-per-epoch" in text
    assert "--eval-every" in text
    assert "--patience" in text
    assert "--score-full-weight" in text
    assert "--score-priority-weight" in text


def test_horizon_dataset_source_scope_collects_only_generic_trajectories(tmp_path):
    import importlib.util

    script_path = ROOT / "scripts" / "42_build_horizon_surrogate_dataset.py"
    spec = importlib.util.spec_from_file_location("build_horizon_dataset", script_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    generic = tmp_path / "outputs" / "data_bank_train_paired_no_controls" / "trajectories"
    old_prop = tmp_path / "outputs" / "closed_loop_paired_no_controls" / "formal" / "old" / "proposed"
    old_base = tmp_path / "outputs" / "closed_loop_paired_no_controls" / "formal" / "old" / "baselines" / "auto_rbc"
    generic.mkdir(parents=True)
    old_prop.mkdir(parents=True)
    old_base.mkdir(parents=True)
    (generic / "E1__auto_rbc_detail.csv").write_text("event_id,policy_id\nE1,auto_rbc\n", encoding="utf-8")
    (old_prop / "E_old__proposed_detail.csv").write_text("event_id,policy_id\nE_old,proposed\n", encoding="utf-8")
    (old_base / "E_old__auto_rbc_detail.csv").write_text("event_id,policy_id\nE_old,auto_rbc\n", encoding="utf-8")

    files = mod._collect_detail_files(tmp_path, source_scope="generic_trajectories")

    assert [p.name for p in files] == ["E1__auto_rbc_detail.csv"]


def test_horizon_dataset_collect_filters_to_current_rainfall_event_ids(tmp_path):
    import importlib.util

    script_path = ROOT / "scripts" / "42_build_horizon_surrogate_dataset.py"
    spec = importlib.util.spec_from_file_location("build_horizon_dataset", script_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    generic = tmp_path / "outputs" / "data_bank_train_paired_no_controls" / "trajectories"
    generic.mkdir(parents=True)
    (generic / "T75_D75_chicago_center__auto_rbc_detail.csv").write_text("event_id,policy_id\nT75_D75_chicago_center,auto_rbc\n", encoding="utf-8")
    (generic / "T75_D90_old_project__auto_rbc_detail.csv").write_text("event_id,policy_id\nT75_D90_old_project,auto_rbc\n", encoding="utf-8")

    files = mod._collect_detail_files(
        tmp_path,
        source_scope="generic_trajectories",
        allowed_event_ids={"T75_D75_chicago_center"},
    )

    assert [p.name for p in files] == ["T75_D75_chicago_center__auto_rbc_detail.csv"]


def test_horizon_surrogate_training_uses_grouped_event_split():
    import importlib.util

    script_path = ROOT / "scripts" / "43_train_horizon_surrogate.py"
    spec = importlib.util.spec_from_file_location("train_horizon_surrogate", script_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    rows = []
    for event_id in ["E1", "E2", "E3", "E4"]:
        for step in range(3):
            rows.append(
                {
                    "event_id": event_id,
                    "x": float(step),
                    "PFV_H": 10.0,
                    "TFV_H": 100.0,
                    "peak_TFV_rate_H": 5.0,
                }
            )
    train, val, split = mod._event_grouped_split(pd.DataFrame(rows), val_fraction=0.25, seed=7)

    assert split["split_strategy"] == "event_id_grouped"
    assert set(train["event_id"]).isdisjoint(set(val["event_id"]))
    assert set(split["train_events"]).isdisjoint(set(split["val_events"]))
    assert len(split["val_events"]) == 1


def test_horizon_dataset_resume_prefers_csv_chunk_when_parquet_is_unreadable(tmp_path):
    import importlib.util

    script_path = ROOT / "scripts" / "42_build_horizon_surrogate_dataset.py"
    spec = importlib.util.spec_from_file_location("build_horizon_dataset", script_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    parquet = tmp_path / "chunk_00000.parquet"
    csv = tmp_path / "chunk_00000.csv"
    parquet.write_bytes(b"stale parquet placeholder")
    csv.write_text("event_id,action_mean\nE1,0.5\n", encoding="utf-8")

    assert mod._preferred_existing_chunk_path(parquet) == csv


def test_horizon_dataset_resume_rejects_chunk_with_stale_event_ids():
    import importlib.util

    script_path = ROOT / "scripts" / "42_build_horizon_surrogate_dataset.py"
    spec = importlib.util.spec_from_file_location("build_horizon_dataset", script_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    reused = pd.DataFrame(
        {
            "event_id": ["T75_D75_chicago_center", "T75_D90_old_project"],
            "PFV_H": [1.0, 2.0],
        }
    )

    ok, reason = mod._chunk_event_filter_ok(reused, {"T75_D75_chicago_center"})

    assert ok is False
    assert "stale_event_ids" in reason


def test_horizon_dataset_treats_parquet_engine_fallback_as_warning():
    import importlib.util

    script_path = ROOT / "scripts" / "42_build_horizon_surrogate_dataset.py"
    spec = importlib.util.spec_from_file_location("build_horizon_dataset", script_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    issue = mod._classify_write_issue(
        Path("chunk_00000.parquet"),
        "ImportError(\"Unable to find a usable engine; tried using: 'pyarrow', 'fastparquet'.\")",
        chunk_index=0,
    )

    assert issue["severity"] == "warning"
    assert issue["category"] == "parquet_engine_missing_csv_fallback"
    assert issue["chunk_index"] == 0


def test_transition_cache_filters_current_events_and_uses_candidate_state(tmp_path):
    from sewerrtc.data.tensor_cache import build_transition_cache, load_npz

    traj = tmp_path / "trajectories"
    traj.mkdir()

    def write_detail(event_id: str, policy_id: str, depth_base: float, flood_base: float) -> None:
        rows = []
        for t in range(5):
            rows.append(
                {
                    "event_id": event_id,
                    "policy_id": policy_id,
                    "elapsed_min": float(t * 5),
                    "rainfall_mm_h": float(10 + t),
                    "h:P1": depth_base + t,
                    "h:N1": depth_base + 10 + t,
                    "a:A1": 0.2 if policy_id == "no_control" else 0.8,
                    "flood:P1": flood_base + t,
                    "flood:N1": flood_base + 10 + t,
                }
            )
        pd.DataFrame(rows).to_csv(traj / f"{event_id}__{policy_id}_detail.csv", index=False)

    write_detail("E1", "no_control", depth_base=1.0, flood_base=0.0)
    write_detail("E1", "auto_rbc", depth_base=100.0, flood_base=100.0)
    write_detail("E_old", "no_control", depth_base=50.0, flood_base=50.0)
    write_detail("E_old", "auto_rbc", depth_base=60.0, flood_base=60.0)

    out = tmp_path / "transition_cache.npz"
    meta = build_transition_cache(
        traj,
        out,
        horizon_steps=2,
        priority_nodes=["P1"],
        dt_sec=300,
        allowed_event_ids={"E1"},
        reference_policies=["no_control", "auto_rbc"],
    )
    cache = load_npz(out, keys=["state", "sources", "event_ids", "policy_ids", "risk_delta"])
    sources = [str(x) for x in cache["sources"]]
    no_control_idx = next(i for i, s in enumerate(sources) if "E1__no_control" in s)

    assert meta["files_seen"] == 4
    assert meta["files_used"] == 2
    assert meta["skipped_stale_detail_files"] == 2
    assert meta["paired_events"] == 1
    assert meta["baseline_policy"] == "multi_reference"
    assert meta["reference_policies"] == ["no_control", "auto_rbc"]
    assert set(cache["event_ids"].astype(str)) == {"E1"}
    assert set(cache["policy_ids"].astype(str)) == {"no_control", "auto_rbc"}
    assert np.isclose(cache["state"][no_control_idx, 0], 1.0)
    assert np.isclose(cache["risk_delta"][no_control_idx, 0], 0.0)


def test_load_npz_can_load_selected_keys(tmp_path):
    from sewerrtc.data.tensor_cache import load_npz

    path = tmp_path / "small.npz"
    np.savez(path, keep=np.asarray([1, 2, 3]), skip=np.asarray([4, 5, 6]))

    loaded = load_npz(path, keys=["keep"])

    assert set(loaded) == {"keep"}
    assert loaded["keep"].tolist() == [1, 2, 3]


def test_surrogate_numpy_batcher_materializes_only_current_batch():
    import importlib.util

    script_path = ROOT / "scripts" / "06_train_surrogate.py"
    spec = importlib.util.spec_from_file_location("train_surrogate", script_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    arr = np.arange(40, dtype=np.float32).reshape(10, 4)
    idx = np.arange(10, dtype=np.int64)
    batches = list(mod._fast_numpy_batches([arr], idx, batch_size=3, sample_weights=None, max_samples=5, seed=1))

    assert len(batches) == 2
    assert batches[0][0].shape == (3, 4)
    assert batches[1][0].shape == (2, 4)


def test_horizon_rollout_adds_role_aware_action_features(tmp_path):
    from sewerrtc.control.horizon_rollout import build_horizon_samples_from_detail

    detail = pd.DataFrame(
        {
            "event_id": ["E1"] * 8,
            "policy_id": ["synthetic"] * 8,
            "elapsed_min": np.arange(8) * 5.0,
            "rainfall_mm_h": np.linspace(0.0, 20.0, 8),
            "phase": ["pre_peak"] * 4 + ["peak"] * 4,
            "h:P1": np.linspace(0.1, 0.8, 8),
            "h:N1": np.linspace(0.0, 0.4, 8),
            "flood:P1": np.linspace(0.0, 2.0, 8),
            "flood:N1": np.linspace(0.0, 3.0, 8),
            "a:PUMP1": [0.2, 0.2, 0.3, 0.4, 0.5, 0.8, 0.9, 1.0],
            "a:IN1": [0.9, 0.8, 0.5, 0.4, 0.3, 0.3, 0.6, 0.8],
            "a:OUT1": [0.1, 0.2, 0.2, 0.4, 0.7, 0.9, 0.9, 1.0],
        }
    )
    detail_path = tmp_path / "E1__synthetic_detail.csv"
    detail.to_csv(detail_path, index=False)
    actuators = pd.DataFrame(
        {
            "actuator_id": ["PUMP1", "IN1", "OUT1"],
            "link_type": ["pump", "orifice", "orifice"],
            "storage_control_type": ["", "storage_inlet", "storage_outlet"],
            "near_storage": [False, True, True],
        }
    )
    priority_to_actuators = pd.DataFrame(
        {
            "priority_node": ["P1", "P1"],
            "actuator_id": ["IN1", "OUT1"],
            "asset_role": ["storage_inlet", "storage_outlet"],
            "influence_path_length": [1, 2],
        }
    )

    samples = build_horizon_samples_from_detail(
        detail_path,
        ["P1"],
        horizon_steps=3,
        history_steps=2,
        dt_sec=300,
        stride=1,
        actuators=actuators,
        priority_to_actuators=priority_to_actuators,
    )

    expected = {
        "pump_action_mean",
        "storage_inlet_action_mean",
        "storage_outlet_action_mean",
        "priority_action_mean",
        "priority_sequence_action_mean",
        "sequence_delta_abs_mean",
        "retain_fraction",
        "release_fraction",
    }
    assert expected <= set(samples.columns)
    first = samples.iloc[0]
    assert np.isclose(first["pump_action_mean"], 0.2)
    assert np.isclose(first["storage_inlet_action_mean"], 0.9)
    assert np.isclose(first["storage_outlet_action_mean"], 0.1)
    assert np.isclose(first["priority_action_mean"], 0.5)
    depth_targets = {
        "full_depth_mean_H",
        "full_depth_p95_H",
        "full_depth_max_H",
        "priority_depth_mean_H",
        "priority_depth_p95_H",
        "priority_peak_depth_H",
    }
    assert depth_targets <= set(samples.columns)
    assert float(first["full_depth_max_H"]) > float(first["current_depth_max"])
    assert float(first["priority_depth_mean_H"]) > float(first["priority_depth_mean"])


def test_horizon_action_features_encode_temporal_action_profiles():
    from sewerrtc.control.horizon_action_features import build_action_feature_map

    actuators = pd.DataFrame(
        {
            "actuator_id": ["PUMP1", "IN1", "OR1"],
            "link_type": ["pump", "orifice", "orifice"],
            "storage_control_type": ["not_storage", "storage_inlet", "not_storage"],
            "near_storage": [False, True, False],
        }
    )
    priority_to_actuators = pd.DataFrame(
        {
            "priority_node": ["PX", "PX"],
            "actuator_id": ["IN1", "OR1"],
        }
    )
    sequence = np.asarray(
        [
            [0.5, 0.8, 0.9],
            [0.5, 0.6, 0.7],
            [0.7, 0.4, 0.5],
            [0.9, 0.8, 0.9],
            [0.9, 0.8, 0.9],
            [0.6, 0.7, 0.8],
        ],
        dtype=np.float32,
    )

    features = build_action_feature_map(
        ["PUMP1", "IN1", "OR1"],
        sequence[0],
        sequence=sequence,
        reference_action=np.asarray([0.5, 0.8, 0.9], dtype=np.float32),
        actuators=actuators,
        priority_to_actuators=priority_to_actuators,
    )

    expected = {
        "sequence_early_action_mean",
        "sequence_mid_action_mean",
        "sequence_late_action_mean",
        "sequence_early_delta_mean",
        "sequence_mid_delta_mean",
        "sequence_late_delta_mean",
        "pump_sequence_early_action_mean",
        "storage_sequence_mid_action_mean",
        "regulator_sequence_late_action_mean",
        "priority_sequence_mid_delta_mean",
    }
    assert expected <= set(features)
    assert features["sequence_early_action_mean"] != features["sequence_mid_action_mean"]
    assert features["priority_sequence_mid_delta_mean"] < 0.0


def test_temporal_surrogate_targets_include_full_and_priority_depth_effects():
    from sewerrtc.models.temporal_graph_surrogate import TARGET_COLUMNS

    expected = {
        "PFV_H",
        "TFV_H",
        "peak_TFV_rate_H",
        "priority_peak_depth_H",
        "high_risk_exposure_time_H",
        "full_depth_mean_H",
        "full_depth_p95_H",
        "full_depth_max_H",
        "priority_depth_mean_H",
        "priority_depth_p95_H",
    }

    assert expected <= set(TARGET_COLUMNS)


def test_horizon_ridge_predictor_uses_role_aware_sequence_features(tmp_path):
    from sewerrtc.models.temporal_graph_surrogate import HorizonRidgeSurrogate
    from sewerrtc.simulation.pyswmm_runner import _make_horizon_ridge_predictor

    feature_columns = [
        "current_depth_mean",
        "priority_depth_max",
        "rain_forecast_mean",
        "action_mean",
        "priority_action_mean",
        "priority_sequence_action_mean",
        "sequence_delta_abs_mean",
    ]
    train = pd.DataFrame(
        [
            {
                "current_depth_mean": 0.1,
                "priority_depth_max": 0.2,
                "rain_forecast_mean": 10.0,
                "action_mean": 0.5,
                "priority_action_mean": 0.9,
                "priority_sequence_action_mean": 0.9,
                "sequence_delta_abs_mean": 0.0,
                "PFV_H": 100.0,
                "TFV_H": 500.0,
                "peak_TFV_rate_H": 50.0,
            },
            {
                "current_depth_mean": 0.1,
                "priority_depth_max": 0.2,
                "rain_forecast_mean": 10.0,
                "action_mean": 0.5,
                "priority_action_mean": 0.2,
                "priority_sequence_action_mean": 0.2,
                "sequence_delta_abs_mean": 0.7,
                "PFV_H": 10.0,
                "TFV_H": 500.0,
                "peak_TFV_rate_H": 50.0,
            },
        ]
    )
    model = HorizonRidgeSurrogate(alpha=1e-6).fit(train, feature_columns)
    model_path = tmp_path / "horizon_ridge_surrogate.npz"
    model.save(model_path)
    actuators = pd.DataFrame(
        {
            "actuator_id": ["P1", "P2"],
            "link_type": ["orifice", "pump"],
            "storage_control_type": ["storage_inlet", ""],
            "near_storage": [True, False],
        }
    )
    priority_to_actuators = pd.DataFrame({"actuator_id": ["P1"], "priority_node": ["PX"]})
    predictor = _make_horizon_ridge_predictor(
        model_path,
        3,
        priority_indices=[0],
        actuators=actuators,
        priority_to_actuators=priority_to_actuators,
    )
    context = {
        "reconstructed_state": np.asarray([0.2, 0.1], dtype=np.float32),
        "rainfall_window": np.asarray([10.0, 10.0, 10.0], dtype=np.float32),
        "current_action": np.asarray([0.9, 0.5], dtype=np.float32),
    }

    hold = predictor(np.tile(np.asarray([[0.9, 0.5]], dtype=np.float32), (3, 1)), context)
    restrict = predictor(np.tile(np.asarray([[0.2, 0.5]], dtype=np.float32), (3, 1)), context)

    assert float(np.sum(restrict["pfv"])) < float(np.sum(hold["pfv"]))


def test_reference_horizon_arrays_use_current_time_window_not_event_totals():
    from sewerrtc.simulation.pyswmm_runner import _reference_horizon_arrays_from_detail

    detail = pd.DataFrame(
        {
            "elapsed_min": [0.0, 5.0, 10.0, 15.0, 20.0],
            "flood:P1": [0.0, 1.0, 2.0, 4.0, 8.0],
            "flood:N1": [0.0, 10.0, 20.0, 40.0, 80.0],
        }
    )
    ref = _reference_horizon_arrays_from_detail(
        detail,
        elapsed_min=5.0,
        horizon_steps=2,
        dt_sec=300,
        priority_nodes=["P1"],
    )

    assert np.allclose(ref["pfv"], [600.0, 1200.0])
    assert np.allclose(ref["tfv"], [6600.0, 13200.0])
    assert np.allclose(ref["peak_tfv_rate"], [22.0, 44.0])


def test_horizon_surrogate_validation_adapts_high_risk_threshold_when_absolute_is_too_high():
    import importlib.util

    script_path = ROOT / "scripts" / "44_validate_horizon_surrogate.py"
    spec = importlib.util.spec_from_file_location("validate_horizon_surrogate", script_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)

    true_pfv = pd.Series([0.0, 5.0, 10.0, 50.0, 100.0])
    pred_pfv = pd.Series([0.0, 4.0, 12.0, 55.0, 90.0])

    threshold_info = mod._resolve_high_risk_threshold(
        true_pfv,
        pred_pfv,
        requested_threshold=1000.0,
        quantile=0.80,
        min_true_count=1,
    )

    assert threshold_info["mode"] == "adaptive_quantile"
    assert threshold_info["requested_true_count"] == 0
    assert threshold_info["true_count"] > 0
    assert threshold_info["threshold"] <= float(true_pfv.max())


def test_formal_gate_cli_rejects_explicit_missing_paired_metrics(tmp_path):
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "63_project5_formal_gate.py"),
            "--config",
            str(ROOT / "configs" / "wuhan.yaml"),
            "--paired-metrics",
            str(tmp_path / "missing_paired_metrics.csv"),
            "--out-dir",
            str(tmp_path / "gate"),
        ],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
    )

    assert proc.returncode != 0
    assert "Missing paired metrics" in (proc.stderr + proc.stdout)


def test_formal_rainfall_config_uses_multiple_patterns_and_long_durations():
    from sewerrtc.io.project_paths import load_config

    cfg = load_config(ROOT / "configs" / "wuhan.yaml")
    rain = cfg["rainfall"]

    assert len(rain.get("formal_patterns", [])) >= 3
    assert max(int(x) for x in rain["formal_durations"]) >= 300
    assert "T75" in set(rain["formal_rain_ids"])
    assert rain.get("representative_event_ids")


def test_sensor_selection_config_forces_priority_domain_observation():
    from sewerrtc.io.project_paths import load_config

    cfg = load_config(ROOT / "configs" / "wuhan.yaml")
    sensor_cfg = cfg.get("sensor_selection", {}) or {}

    assert sensor_cfg.get("include_priority_nodes") is True
    assert sensor_cfg.get("include_sentinel_nodes") is True
    assert float(sensor_cfg.get("priority_sensor_fraction", 0.0)) >= 0.20


def test_generic_closed_loop_wires_trained_horizon_surrogate_predictor():
    runner = ROOT / "sewerrtc" / "simulation" / "pyswmm_runner.py"
    text = runner.read_text(encoding="utf-8")

    assert "load_horizon_surrogate" in text
    assert "horizon_predictor=horizon_predictor" in text
    assert '"horizon_ridge_surrogate.npz"' in text
    assert '"horizon_temporal_gnn.pt"' in text


def test_generic_closed_loop_passes_future_rainfall_window_to_controller():
    script = ROOT / "scripts" / "08_run_closed_loop.py"
    runner = ROOT / "sewerrtc" / "simulation" / "pyswmm_runner.py"
    script_text = script.read_text(encoding="utf-8")
    runner_text = runner.read_text(encoding="utf-8")

    assert '"rainfall_csv": str(ev["rainfall_csv"])' in script_text
    assert "def _load_rainfall_forecast" in runner_text
    assert "_rainfall_window_at(" in runner_text
    assert "rainfall_window=rainfall_window" in runner_text


def test_generic_closed_loop_initial_action_uses_swmm_current_settings_not_all_open():
    runner = ROOT / "sewerrtc" / "simulation" / "pyswmm_runner.py"
    text = runner.read_text(encoding="utf-8")

    assert "def _initial_action_from_links" in text
    assert 'for attr in ("current_setting", "target_setting")' in text
    assert "previous_action = _observed_action_from_links" in text
    assert "previous_action = np.ones(len(actuator_ids)" not in text


def test_generic_closed_loop_uses_configured_generic_default_policy_not_internal_nominal():
    script = ROOT / "scripts" / "08_run_closed_loop.py"
    runner = ROOT / "sewerrtc" / "simulation" / "pyswmm_runner.py"
    cfg = (ROOT / "configs" / "wuhan.yaml").read_text(encoding="utf-8")
    script_text = script.read_text(encoding="utf-8")
    runner_text = runner.read_text(encoding="utf-8")

    assert "default_action_policy: no_control" in cfg
    assert "generic_default_policy_id" in script_text
    assert "generic_default_policy_id" in runner_text
    assert "GenericActionPolicy(generic_default_policy_name" in runner_text
    assert "default_action = generic_default_policy.action" in runner_text


def test_generic_closed_loop_uses_explicit_no_control_twin_reference():
    runner = ROOT / "sewerrtc" / "simulation" / "pyswmm_runner.py"
    config = ROOT / "configs" / "wuhan_project6.yaml"
    text = runner.read_text(encoding="utf-8")
    cfg_text = config.read_text(encoding="utf-8")

    assert "reference_policy_for_constraints: precomputed_no_control_twin_horizon" in cfg_text
    generic_block = text[text.index('if proposed_controller == "generic_gat_mpc":', text.index("for _ in sim:")) :]
    assert 'reference_pfv=reference_window["pfv"]' in generic_block
    assert 'reference_tfv=reference_window["tfv"]' in generic_block
    assert 'reference_peak=reference_window["peak_tfv_rate"]' in generic_block
