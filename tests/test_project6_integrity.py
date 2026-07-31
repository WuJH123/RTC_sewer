from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest
import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_horizon_volume_labels_integrate_flow_rates_in_seconds(tmp_path: Path):
    from sewerrtc.control.horizon_rollout import build_horizon_samples_from_detail

    detail = pd.DataFrame(
        {
            "event_id": ["E1"] * 4,
            "policy_id": ["no_control"] * 4,
            "elapsed_min": [0.0, 5.0, 10.0, 15.0],
            "rainfall_mm_h": [0.0] * 4,
            "h:P": [0.0, 0.0, 0.0, 0.0],
            "flood:P": [0.0, 1.0, 1.0, 0.0],
            "a:A": [1.0] * 4,
        }
    )
    path = tmp_path / "E1__no_control_detail.csv"
    detail.to_csv(path, index=False)

    samples = build_horizon_samples_from_detail(
        path,
        priority_nodes=["P"],
        horizon_steps=2,
        history_steps=1,
        dt_sec=300,
    )

    assert samples.loc[0, "PFV_H"] == pytest.approx(600.0)
    assert samples.loc[0, "TFV_H"] == pytest.approx(600.0)


def test_dataset_source_fingerprint_changes_when_a_detail_file_changes(tmp_path: Path):
    from sewerrtc.data.dataset_fingerprint import source_file_fingerprint

    detail = tmp_path / "E1__no_control_detail.csv"
    detail.write_text("event_id,value\nE1,1\n", encoding="utf-8")
    initial = source_file_fingerprint([detail])

    detail.write_text("event_id,value\nE1,2\n", encoding="utf-8")
    changed = source_file_fingerprint([detail])

    assert initial != changed


def test_horizon_resume_rejects_a_chunk_when_its_source_content_changed(tmp_path: Path):
    from sewerrtc.data.dataset_fingerprint import source_file_fingerprint

    builder = _load_script("42_build_horizon_surrogate_dataset.py")
    detail = tmp_path / "E1__no_control_detail.csv"
    detail.write_text("event_id,value\nE1,1\n", encoding="utf-8")
    recorded = source_file_fingerprint([detail])

    detail.write_text("event_id,value\nE1,changed\n", encoding="utf-8")

    assert builder._chunk_source_fingerprint_matches(recorded, [detail]) is False


def test_no_control_repair_gate_rejects_metrics_without_risk_class():
    gate = _load_script("75_no_control_repair_gate.py")
    metrics = pd.DataFrame(
        {
            "event_id": ["E1"],
            "policy_id": ["proposed_gat_mpc"],
            "PFV": [1.0],
            "TFV": [1.0],
            "peak_TFV_rate": [1.0],
        }
    )

    report = gate.evaluate_repair_gate(metrics, {})

    assert report["passed"] is False
    assert "event_risk_class missing" in report["reasons"]


def test_recalculated_event_policy_metrics_carry_the_baseline_risk_class():
    recalc = _load_script("61_recalculate_project2_priority_zone_metrics.py")
    policy = pd.DataFrame(
        {
            "event_id": ["E1", "E1"],
            "policy_id": ["no_control", "proposed_gat_mpc"],
        }
    )
    event_table = pd.DataFrame(
        {
            "event_id": ["E1"],
            "event_risk_class": ["high_risk_event"],
            "is_near_zero_pfv": [False],
        }
    )

    merged = recalc._attach_event_risk_class(policy, event_table)

    assert set(merged["event_risk_class"]) == {"high_risk_event"}


def test_project6_default_config_has_no_external_reuse_dependency():
    from sewerrtc.io.project_paths import load_config

    cfg = load_config(ROOT / "configs" / "wuhan_project6.yaml")

    assert Path(cfg["project_root"]).resolve() == ROOT.resolve()
    assert not cfg.get("reuse_sources")
    assert cfg["controller"]["repair_reliable_action_filter"]["enabled"] is True


def test_project6_pipeline_defaults_to_the_project6_config():
    text = (ROOT / "scripts" / "project6_runs" / "RUN_PROJECT6_NO_CONTROL_REPAIR_PIPELINE.ps1").read_text(encoding="utf-8")

    assert '[string]$Root = "E:\\RTC_sewer\\Project6"' in text
    assert '[string]$Config = "configs\\wuhan_project6.yaml"' in text


def test_action_features_distinguish_different_actuator_priority_paths():
    from sewerrtc.control.horizon_action_features import ACTION_FEATURE_COLUMNS, build_action_feature_map

    actuators = pd.DataFrame(
        {
            "actuator_id": ["PUMP_A", "PUMP_B"],
            "link_type": ["pump", "pump"],
            "asset_role": ["pump", "pump"],
            "storage_control_type": ["not_storage", "not_storage"],
            "near_storage": [False, False],
        }
    )
    paths = pd.DataFrame(
        {
            "priority_node": ["PRIORITY_A", "PRIORITY_B"],
            "actuator_id": ["PUMP_A", "PUMP_B"],
            "influence_path_length": [1, 1],
            "direction": ["upstream", "upstream"],
        }
    )
    reference = np.ones(2, dtype=float)
    seq_a = np.tile(reference, (6, 1))
    seq_b = np.tile(reference, (6, 1))
    seq_a[:, 0] = 0.8
    seq_b[:, 1] = 0.8

    feat_a = build_action_feature_map(
        ["PUMP_A", "PUMP_B"], seq_a[0], sequence=seq_a, reference_action=reference,
        actuators=actuators, priority_to_actuators=paths,
    )
    feat_b = build_action_feature_map(
        ["PUMP_A", "PUMP_B"], seq_b[0], sequence=seq_b, reference_action=reference,
        actuators=actuators, priority_to_actuators=paths,
    )

    signature_cols = [c for c in ACTION_FEATURE_COLUMNS if c.startswith(("actuator_hash_", "path_hash_"))]
    assert signature_cols
    assert any(feat_a[c] != pytest.approx(feat_b[c]) for c in signature_cols)


def test_action_sequence_pool_deduplicates_identical_numeric_sequences():
    from sewerrtc.control.action_sequence_generator import _dedupe_and_cap_sequences

    sequence = np.ones((3, 2), dtype=np.float32)
    candidates = [
        {"label": "first", "target_actuators": "A", "sequence": sequence.copy()},
        {"label": "same_numbers_different_label", "target_actuators": "B", "sequence": sequence.copy()},
    ]

    deduplicated = _dedupe_and_cap_sequences(candidates)

    assert len(deduplicated) == 1


def test_existing_rtc_scope_excludes_uninstrumented_retrofit_assets():
    from sewerrtc.control.actuator_scope import select_actuators_for_scope

    actuators = pd.DataFrame(
        {
            "actuator_id": ["EXISTING", "RETROFIT"],
            "has_internal_rule": [True, False],
            "is_physically_controllable": [True, True],
        }
    )

    existing = select_actuators_for_scope(actuators, "existing_rtc")
    expanded = select_actuators_for_scope(actuators, "existing_plus_retrofit")

    assert existing["actuator_id"].tolist() == ["EXISTING"]
    assert expanded["actuator_id"].tolist() == ["EXISTING", "RETROFIT"]


def test_storage_retrofit_manifest_requires_engineering_balanced_action_set():
    from sewerrtc.control.retrofit_assets import validate_retrofit_asset_mix

    assets = pd.DataFrame(
        {
            "actuator_id": [
                "IN_1", "OUT_1", "IN_2", "OUT_2", "IN_3", "OUT_3",
                "REG_1", "WEIR_1", "PUMP_1", "PUMP_2",
            ],
            "asset_class": [
                "storage_inlet", "storage_outlet", "storage_inlet", "storage_outlet", "storage_inlet", "storage_outlet",
                "downstream_regulator", "downstream_regulator", "pump", "pump",
            ],
            "action_index": list(range(10)),
            "link_type": ["orifice"] * 8 + ["pump", "pump"],
        }
    )

    report = validate_retrofit_asset_mix(assets, action_dim=109)

    assert report["selected_assets"] == 10
    assert report["storage_linked_assets"] == 6
    assert report["downstream_regulators"] == 2
    assert report["pumps"] == 2


def test_storage_retrofit_manifest_rejects_priority_direct_discharge_pumps():
    from sewerrtc.control.retrofit_assets import validate_retrofit_asset_mix

    assets = pd.DataFrame(
        {
            "actuator_id": [
                "IN_1", "OUT_1", "IN_2", "OUT_2", "IN_3", "OUT_3",
                "REG_1", "WEIR_1", "PUMP_1", "MSL010.9",
            ],
            "asset_class": [
                "storage_inlet", "storage_outlet", "storage_inlet", "storage_outlet", "storage_inlet", "storage_outlet",
                "downstream_regulator", "downstream_regulator", "pump", "pump",
            ],
            "action_index": list(range(10)),
            "link_type": ["orifice"] * 8 + ["pump", "pump"],
        }
    )

    with pytest.raises(ValueError, match="blocked actuator"):
        validate_retrofit_asset_mix(assets, action_dim=109)


def test_audit_marks_control_enabled_without_reindexing_historical_action_space():
    audit = _load_script("00_audit_inp.py")
    actuators = pd.DataFrame(
        {
            "actuator_id": ["A", "B", "C"],
            "actuator_index": [0, 1, 2],
            "has_internal_rule": [False, False, False],
        }
    )

    marked = audit._annotate_control_enabled(actuators, ["B"])

    assert marked["actuator_index"].tolist() == [0, 1, 2]
    assert marked["control_enabled"].tolist() == [False, True, False]


def test_control_enabled_scope_uses_only_explicitly_deployed_physical_assets():
    from sewerrtc.control.actuator_scope import select_actuators_for_scope

    table = pd.DataFrame(
        {
            "actuator_id": ["old", "retrofit", "undeployed"],
            "control_enabled": [True, True, False],
            "is_physically_controllable": [True, True, True],
            "is_existing_rtc": [True, False, False],
        }
    )
    selected = select_actuators_for_scope(table, "control_enabled")
    assert selected["actuator_id"].tolist() == ["old", "retrofit"]


def test_horizon_features_keep_deployed_asset_identity_and_action_direction():
    from sewerrtc.control.horizon_action_features import build_action_feature_map

    features = build_action_feature_map(
        ["RTC_IN_01", "RTC_OUT_01"],
        np.asarray([0.2, 0.8], dtype=float),
        sequence=np.asarray([[0.2, 0.8], [0.3, 0.7]], dtype=float),
        reference_action=np.asarray([[0.5, 0.5], [0.5, 0.5]], dtype=float),
    )
    assert features["asset_RTC_IN_01_action"] == pytest.approx(0.2)
    assert features["asset_RTC_IN_01_delta"] == pytest.approx(-0.3)
    assert features["asset_RTC_OUT_01_delta"] == pytest.approx(0.3)
    assert features["asset_RTC_IN_01_early_delta"] != features["asset_RTC_OUT_01_early_delta"]


def test_binary_pump_semantics_project_fractional_pump_actions_but_keep_orifices_continuous():
    from sewerrtc.simulation.pyswmm_runner import _enforce_actuator_semantics

    actuators = pd.DataFrame({"actuator_id": ["P", "O"], "link_type": ["pump", "orifice"]})
    action = _enforce_actuator_semantics(np.asarray([0.49, 0.49]), ["P", "O"], actuators, "binary_unless_verified")
    assert action.tolist() == pytest.approx([0.0, 0.49])
    action = _enforce_actuator_semantics(np.asarray([0.50, 0.20]), ["P", "O"], actuators, "binary_unless_verified")
    assert action.tolist() == pytest.approx([1.0, 0.20])


def test_variable_speed_pump_semantics_keep_continuous_commands():
    from sewerrtc.simulation.pyswmm_runner import _enforce_actuator_semantics

    actuators = pd.DataFrame({"actuator_id": ["P"], "link_type": ["pump"]})
    action = _enforce_actuator_semantics(np.asarray([0.35]), ["P"], actuators, "variable_speed")
    assert action.tolist() == pytest.approx([0.35])


def test_verified_vfd_registry_keeps_only_declared_pump_continuous():
    from sewerrtc.simulation.pyswmm_runner import _enforce_actuator_semantics

    actuators = pd.DataFrame({"actuator_id": ["VFD", "BINARY"], "link_type": ["pump", "pump"]})
    action = _enforce_actuator_semantics(
        np.asarray([0.35, 0.35]),
        ["VFD", "BINARY"],
        actuators,
        "binary_unless_verified",
        variable_speed_pump_ids=["VFD"],
    )
    assert action.tolist() == pytest.approx([0.35, 0.0])


def test_variable_speed_controller_applies_pump_ramp_and_dwell_constraints():
    from sewerrtc.control.generic_gat_mpc import GenericGATMPCController

    actuators = pd.DataFrame(
        [{"actuator_id": "VFD_PUMP", "link_type": "pump", "asset_role": "pump"}]
    )
    controller = GenericGATMPCController(
        actuators,
        horizon_steps=2,
        max_first_step_delta=1.0,
        per_actuator_max_delta={"VFD_PUMP": 0.15},
        min_hold_steps_by_actuator={"VFD_PUMP": 2},
    )
    projected, audit = controller._project_execution_sequence(
        np.asarray([[1.0], [1.0]], dtype=np.float32), np.asarray([0.5], dtype=np.float32)
    )
    assert projected[0, 0] == pytest.approx(0.65)
    assert projected[1, 0] == pytest.approx(0.65)
    assert audit["ramp_clipped_values"] >= 1

    controller._record_executed_action(projected[0], np.asarray([0.5], dtype=np.float32))
    held, held_audit = controller._project_execution_sequence(
        np.asarray([[0.2], [0.2]], dtype=np.float32), projected[0]
    )
    assert held[0, 0] == pytest.approx(projected[0, 0])
    assert held_audit["dwell_held_values"] == 1


def test_runtime_empirical_filter_can_forbid_unverified_joint_actions():
    from sewerrtc.control.generic_gat_mpc import GenericGATMPCController

    actuators = pd.DataFrame(
        [
            {"actuator_id": "A", "link_type": "orifice", "asset_role": "regulator"},
            {"actuator_id": "B", "link_type": "orifice", "asset_role": "regulator"},
        ]
    )
    controller = GenericGATMPCController(actuators, candidate_group_limit=2)
    controller.set_runtime_action_filter(["A", "B"], {"A": ["decrease"], "B": ["decrease"]}, candidate_group_limit=1)
    assert controller.runtime_candidate_group_limit == 1


def test_first_no_control_reference_is_not_ramp_limited_by_swmm_initial_readback():
    from sewerrtc.control.generic_gat_mpc import GenericGATMPCController

    actuators = pd.DataFrame([{"actuator_id": "VFD", "link_type": "pump", "asset_role": "pump"}])
    controller = GenericGATMPCController(
        actuators, horizon_steps=2, per_actuator_max_delta={"VFD": 0.15}
    )
    projected, _ = controller._project_execution_sequence(
        np.asarray([[0.10], [0.10]], dtype=np.float32), np.asarray([0.0], dtype=np.float32)
    )
    assert projected[0, 0] == pytest.approx(0.10)


def test_targeted_ablation_uses_a_unique_swmm_input_per_case():
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "single_ablation", Path("scripts/76_generate_no_control_single_actuator_ablation.py")
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert "case_inp" in module._run_case.__code__.co_varnames


def test_policy_mask_keeps_disabled_assets_at_default_setting():
    from sewerrtc.simulation.action_policies import GenericActionPolicy, PolicyContext

    actuators = pd.DataFrame(
        {
            "actuator_id": ["ENABLED", "DISABLED"],
            "link_type": ["orifice", "orifice"],
            "near_storage": [False, False],
            "storage_control_type": ["not_storage", "not_storage"],
            "control_enabled": [True, False],
        }
    )
    ctx = PolicyContext(
        elapsed_min=20.0,
        duration_min=100,
        rainfall_mm_h=50.0,
        phase="peak",
        previous_action=np.asarray([0.6, 0.6], dtype=np.float32),
    )

    action = GenericActionPolicy("all_closed_safe", actuators).action(ctx)

    assert action[0] == pytest.approx(0.15)
    assert action[1] == pytest.approx(1.0)


def test_online_reference_mode_does_not_request_a_true_future_no_control_run():
    closed_loop = _load_script("08_run_closed_loop.py")

    assert closed_loop._needs_no_control_reference("online_predicted_default") is False
    assert closed_loop._needs_no_control_reference("no_control") is True


def test_closed_loop_report_uses_the_cli_controller_value():
    source = (ROOT / "scripts" / "08_run_closed_loop.py").read_text(encoding="utf-8")

    assert 'if args.proposed_controller != "generic_gat_mpc":' in source


def test_closed_loop_report_distinguishes_action_space_from_enabled_controls():
    source = (ROOT / "scripts" / "08_run_closed_loop.py").read_text(encoding="utf-8")

    assert '"action_space_actuator_count": int(len(actuators))' in source
    assert '"control_enabled_actuator_count"' in source


def test_online_reference_mode_never_converts_missing_oracle_detail_to_zero_risk():
    from sewerrtc.simulation.pyswmm_runner import _constraint_reference_window

    reference, mode = _constraint_reference_window(
        None,
        elapsed_min=10.0,
        horizon_steps=3,
        dt_sec=300,
        priority_nodes=["P"],
    )

    assert reference is None
    assert mode == "online_predicted_no_control_sequence"


def test_observed_no_control_action_uses_safe_pump_default_when_setting_is_unavailable():
    from sewerrtc.simulation.pyswmm_runner import _observed_action_from_links

    actuators = pd.DataFrame({"actuator_id": ["O", "P"], "link_type": ["orifice", "pump"]})
    action = _observed_action_from_links({}, ["O", "P"], actuators)

    assert action.tolist() == pytest.approx([1.0, 0.0])


def test_storage_retrofit_event_split_is_balanced_and_disjoint():
    builder = _load_script("84_build_v8_storage_retrofit_scenario.py")
    rows = []
    return_periods = ["T5", "T10", "T20", "T30", "T50", "T75", "T100"]
    patterns = ["chicago_center", "chicago_early", "chicago_late", "block", "double_peak"]
    for rp in return_periods:
        for duration in [75, 105, 150, 210, 240, 300]:
            for pattern in patterns:
                rows.append({"event_id": f"{rp}_D{duration}_{pattern}", "return_period": rp, "duration_min": duration, "pattern": pattern})
    calibration, formal = builder.build_event_splits(pd.DataFrame(rows))

    assert len(calibration) == 14
    assert len(formal) == 35
    assert set(calibration.event_id).isdisjoint(set(formal.event_id))
    assert calibration.groupby("return_period").size().eq(2).all()
    assert formal.groupby("return_period").size().eq(5).all()


def test_storage_retrofit_extended_split_has_two_events_per_return_period_pattern():
    builder = _load_script("84_build_v8_storage_retrofit_scenario.py")
    rows = []
    for rp in ["T5", "T10", "T20", "T30", "T50", "T75", "T100"]:
        for duration in [75, 105, 150, 210, 240, 300]:
            for pattern in ["chicago_center", "chicago_early", "chicago_late", "block", "double_peak"]:
                rows.append({"event_id": f"{rp}_D{duration}_{pattern}", "rain_id": rp, "duration_min": duration, "pattern": pattern})
    calibration, formal70 = builder.build_extended_formal_split(pd.DataFrame(rows))

    assert len(calibration) == 14
    assert len(formal70) == 70
    assert set(calibration.event_id).isdisjoint(set(formal70.event_id))
    assert formal70.groupby(["return_period", "pattern"]).size().eq(2).all()


def test_audit_marks_rule_or_explicit_assets_as_existing_rtc():
    audit = _load_script("00_audit_inp.py")
    actuators = pd.DataFrame(
        {
            "actuator_id": ["RULE", "EXPLICIT", "RETROFIT"],
            "has_internal_rule": [True, False, False],
        }
    )

    marked = audit._annotate_actuator_scope(actuators, ["EXPLICIT"])

    assert marked.set_index("actuator_id")["is_existing_rtc"].to_dict() == {
        "RULE": True,
        "EXPLICIT": True,
        "RETROFIT": False,
    }
    assert marked["is_physically_controllable"].all()


def test_efd_storage_priority_controls_regulators_under_equal_permission():
    from sewerrtc.simulation.action_policies import GenericActionPolicy, PolicyContext

    actuators = pd.DataFrame(
        {
            "actuator_id": ["STORAGE_OUT", "ORDINARY_ORIFICE"],
            "link_type": ["orifice", "orifice"],
            "from_node": ["S1", "J1"],
            "to_node": ["J2", "J2"],
            "near_storage": [True, False],
            "storage_control_type": ["storage_outlet", "not_storage"],
            "storage_node_max_depth": [2.0, np.nan],
            "efd_reference_node": ["S1", "J1"],
            "efd_reference_max_depth": [2.0, 1.0],
        }
    )
    previous = np.asarray([0.7, 0.9], dtype=np.float32)
    ctx = PolicyContext(
        elapsed_min=60.0,
        duration_min=120,
        rainfall_mm_h=50.0,
        phase="peak",
        previous_action=previous,
        node_depths={"S1": 1.5, "J1": 1.0},
        node_max_depths={"S1": 2.0, "J1": 1.0},
    )

    action = GenericActionPolicy("efd_storage_priority", actuators).action(ctx)

    assert action[1] != pytest.approx(previous[1])
    assert 0.0 <= action[1] <= 1.0


def test_auto_rbc_uses_local_fill_instead_of_one_global_pump_setting():
    from sewerrtc.simulation.action_policies import GenericActionPolicy, PolicyContext

    actuators = pd.DataFrame(
        {
            "actuator_id": ["P_LOW", "P_HIGH"],
            "link_type": ["pump", "pump"],
            "from_node": ["LOW", "HIGH"],
            "to_node": ["OUT", "OUT"],
            "near_storage": [False, False],
            "storage_control_type": ["not_storage", "not_storage"],
            "efd_reference_max_depth": [2.0, 2.0],
        }
    )
    ctx = PolicyContext(
        elapsed_min=60.0,
        duration_min=120,
        rainfall_mm_h=50.0,
        phase="peak",
        previous_action=np.asarray([0.5, 0.5], dtype=np.float32),
        node_depths={"LOW": 0.2, "HIGH": 1.8},
        node_max_depths={"LOW": 2.0, "HIGH": 2.0},
    )

    action = GenericActionPolicy("auto_rbc", actuators).action(ctx)

    assert action[1] > action[0]


def _project6_like_actuators() -> "pd.DataFrame":
    # Mirrors the real Project6 schema: no legacy from_node/to_node columns,
    # storage topology only via upstream/downstream/storage_node, and a mix of
    # regulators and pumps. This is exactly the frame that used to collapse
    # Auto-RBC and EFD onto identical zero-depth behaviour.
    return pd.DataFrame(
        {
            "actuator_id": [
                "RTC_IN_01",
                "RTC_OUT_01",
                "HS2512760.1",
                "ADD301.2",
            ],
            "link_type": ["orifice", "orifice", "orifice", "pump"],
            "storage_control_type": [
                "storage_inlet",
                "storage_outlet",
                "downstream_regulator",
                "none",
            ],
            "storage_node": ["TANK1", "TANK1", "", ""],
            "upstream_node": ["J_IN", "TANK1", "J_REG", "J_PUMP"],
            "downstream_node": ["TANK1", "J_OUT", "J_DOWN", "J_OUTFALL"],
        }
    )


def _project6_like_context(phase: str = "peak") -> "PolicyContext":
    from sewerrtc.simulation.action_policies import PolicyContext

    return PolicyContext(
        elapsed_min=60.0,
        duration_min=120,
        rainfall_mm_h=50.0,
        phase=phase,
        previous_action=np.asarray([0.5, 0.5, 0.5, 0.5], dtype=np.float32),
        node_depths={
            "TANK1": 1.8,
            "J_IN": 0.4,
            "J_OUT": 0.3,
            "J_REG": 1.2,
            "J_PUMP": 1.5,
        },
        node_max_depths={
            "TANK1": 2.0,
            "J_IN": 2.0,
            "J_OUT": 2.0,
            "J_REG": 2.0,
            "J_PUMP": 2.0,
        },
    )


def test_reference_node_resolves_without_legacy_from_to_columns():
    from sewerrtc.simulation.action_policies import _reference_node_for_row

    frame = _project6_like_actuators()
    inlet = frame.iloc[0]
    outlet = frame.iloc[1]
    regulator = frame.iloc[2]
    # storage inlet fills the downstream tank; outlet drains the upstream tank.
    assert _reference_node_for_row(inlet, "storage_inlet") == "TANK1"
    assert _reference_node_for_row(outlet, "storage_outlet") == "TANK1"
    # regulators/pumps fall back to their upstream node.
    assert _reference_node_for_row(regulator, "downstream_regulator") == "J_REG"


def test_auto_rbc_and_efd_are_not_identical_under_project6_schema():
    from sewerrtc.simulation.action_policies import GenericActionPolicy

    frame = _project6_like_actuators()
    ctx = _project6_like_context()

    auto = GenericActionPolicy("auto_rbc", frame).action(ctx)
    efd = GenericActionPolicy("efd_storage_priority", frame).action(ctx)

    assert auto.shape == efd.shape == (len(frame),)
    # The historic bug made these byte-for-byte identical because EFD silently
    # delegated to Auto-RBC. They must now differ on at least one facility.
    assert not np.allclose(auto, efd), (auto.tolist(), efd.tolist())


def test_attach_reference_nodes_fills_from_inp_topology(tmp_path: "Path"):
    from sewerrtc.simulation.action_policies import attach_reference_nodes

    inp = tmp_path / "tiny.inp"
    inp.write_text(
        "\n".join(
            [
                "[ORIFICES]",
                "REG1 UP1 DOWN1 SIDE 0 0",
                "[PUMPS]",
                "PMP1 UP2 DOWN2 CURVE1 ON 0 0",
                "",
            ]
        ),
        encoding="utf-8",
    )
    frame = pd.DataFrame(
        {
            "actuator_id": ["REG1", "PMP1"],
            "link_type": ["orifice", "pump"],
            "storage_control_type": ["none", "none"],
        }
    )

    out = attach_reference_nodes(frame, inp)

    assert list(out.loc[out["actuator_id"] == "REG1", "from_node"]) == ["UP1"]
    assert list(out.loc[out["actuator_id"] == "REG1", "to_node"]) == ["DOWN1"]
    assert list(out.loc[out["actuator_id"] == "PMP1", "from_node"]) == ["UP2"]
    assert list(out.loc[out["actuator_id"] == "PMP1", "to_node"]) == ["DOWN2"]



def test_generic_mpc_predicts_online_reference_when_oracle_arrays_are_absent():
    from sewerrtc.control.generic_gat_mpc import GenericGATMPCController

    actuators = pd.DataFrame(
        {
            "actuator_id": ["A"],
            "link_type": ["orifice"],
            "from_node": ["J1"],
            "to_node": ["J2"],
            "storage_control_type": ["not_storage"],
        }
    )

    class Predictor:
        def __init__(self):
            self.calls = 0

        def __call__(self, sequence, context):
            self.calls += 1
            h = len(sequence)
            value = float(np.mean(sequence))
            return {
                "pfv": np.full(h, 2.0 + value),
                "tfv": np.full(h, 5.0 + value),
                "peak_tfv_rate": np.full(h, 1.0 + value),
            }

    predictor = Predictor()
    controller = GenericGATMPCController(
        actuators,
        horizon_steps=3,
        max_candidate_sequences=8,
        horizon_predictor=predictor,
        min_pfv_improvement_abs=0.0,
    )
    _, info = controller.choose(
        reconstructed_state=np.asarray([0.2], dtype=np.float32),
        rainfall_window=np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
        current_action=np.asarray([1.0], dtype=np.float32),
        reference_pfv=None,
        reference_tfv=None,
        reference_peak=None,
    )

    assert predictor.calls > 1
    assert info["constraint_reference_source"] == "online_surrogate_default_sequence"
    assert np.isfinite(info["selected_reference_tfv_horizon"])


def test_closed_loop_uses_oracle_reference_only_when_a_true_twin_detail_exists():
    source = (ROOT / "sewerrtc" / "simulation" / "pyswmm_runner.py").read_text(encoding="utf-8")
    generic_block = source[source.index("if proposed_controller == \"generic_gat_mpc\":", source.index("for _ in sim:")) :]

    assert "reference_window, reference_mode = _constraint_reference_window" in generic_block
    assert "reference_pfv=reference_window[\"pfv\"] if reference_window is not None else None" in generic_block
    assert "reference_tfv=reference_window[\"tfv\"] if reference_window is not None else None" in generic_block
    assert "reference_peak=reference_window[\"peak_tfv_rate\"] if reference_window is not None else None" in generic_block
    assert "reference_action_sequence=reference_action_sequence" in generic_block


def test_action_features_use_time_varying_reference_sequence():
    from sewerrtc.control.horizon_action_features import build_action_feature_map

    reference = np.asarray([[0.2, 0.8], [0.4, 0.7], [0.6, 0.6]], dtype=float)
    candidate = reference.copy()
    candidate[1, 0] += 0.1

    same = build_action_feature_map(
        ["A", "B"], reference[0], sequence=reference, reference_action=reference
    )
    changed = build_action_feature_map(
        ["A", "B"], candidate[0], sequence=candidate, reference_action=reference
    )

    assert same["sequence_delta_abs_max"] == pytest.approx(0.0)
    assert changed["sequence_delta_abs_max"] == pytest.approx(0.1)


def test_generated_candidates_expose_absolute_and_residual_actions():
    from sewerrtc.control.action_sequence_generator import generate_action_sequences

    actuators = pd.DataFrame(
        {
            "actuator_id": ["A"],
            "link_type": ["orifice"],
            "asset_role": ["regulator"],
            "storage_control_type": ["not_storage"],
        }
    )
    influence = pd.DataFrame(
        {
            "priority_node": ["P"],
            "actuator_id": ["A"],
            "asset_role": ["regulator"],
            "influence_path_length": [1],
        }
    )
    reference = np.asarray([[0.6], [0.7], [0.8]], dtype=np.float32)

    candidates = generate_action_sequences(
        reference[0], actuators, 3, priority_to_actuators=influence,
        reference_sequence=reference,
    )

    assert candidates
    for candidate in candidates:
        absolute = np.asarray(candidate["absolute_sequence"])
        residual = np.asarray(candidate["residual_sequence"])
        assert candidate["action_semantics"] == "absolute_from_no_control_reference"
        assert absolute == pytest.approx(reference + residual)


def test_empty_reliability_filter_cannot_reopen_all_actuators():
    from sewerrtc.control.action_sequence_generator import generate_action_sequences

    actuators = pd.DataFrame(
        {
            "actuator_id": ["A", "B"],
            "link_type": ["orifice", "pump"],
            "asset_role": ["regulator", "pump"],
        }
    )
    influence = pd.DataFrame(
        {
            "priority_node": ["P", "P"],
            "actuator_id": ["A", "B"],
            "asset_role": ["regulator", "pump"],
            "influence_path_length": [1, 1],
        }
    )
    candidates = generate_action_sequences(
        np.ones(2), actuators, 3,
        priority_to_actuators=influence,
        allowed_actuator_ids=["__NO_RELIABLE_ACTUATOR__"],
    )

    assert [item["label"] for item in candidates] == ["hold_native"]


def test_horizon_samples_are_labeled_against_same_time_no_control_reference(tmp_path: Path):
    from sewerrtc.control.horizon_rollout import build_horizon_samples_from_detail

    base = pd.DataFrame(
        {
            "event_id": ["E1"] * 5,
            "policy_id": ["no_control"] * 5,
            "elapsed_min": [0, 5, 10, 15, 20],
            "rainfall_mm_h": [0] * 5,
            "h:P": [0.1, 0.2, 0.3, 0.4, 0.5],
            "flood:P": [0.0, 1.0, 1.0, 0.0, 0.0],
            "a:A": [0.2, 0.3, 0.4, 0.5, 0.6],
        }
    )
    candidate = base.copy()
    candidate["policy_id"] = "candidate"
    candidate["h:P"] = [9.0] * 5
    candidate["flood:P"] = [0.0, 0.5, 0.5, 0.0, 0.0]
    candidate["a:A"] = [0.2, 0.4, 0.5, 0.5, 0.6]
    ref_path = tmp_path / "E1__no_control_detail.csv"
    cand_path = tmp_path / "E1__candidate_detail.csv"
    base.to_csv(ref_path, index=False)
    candidate.to_csv(cand_path, index=False)

    samples = build_horizon_samples_from_detail(
        cand_path, ["P"], 2, 1, 300, reference_detail_path=ref_path
    )

    assert samples.loc[0, "reference_PFV_H"] == pytest.approx(600.0)
    assert samples.loc[0, "effect_PFV_H"] == pytest.approx(-300.0)
    assert samples.loc[0, "current_depth_mean"] == pytest.approx(0.1)
    assert samples.loc[0, "action_semantics"] == "absolute_from_no_control_reference"
    assert samples.loc[0, "effect_label_mode"] == "paired_no_control_same_time"


def test_multiscale_action_designs_cover_absolute_targets_without_noops():
    ablation = _load_script("76_generate_no_control_single_actuator_ablation.py")

    designs = ablation._effective_action_designs(
        0.40,
        delta_levels=[0.05, 0.20],
        absolute_levels=[0.0, 0.25, 0.50, 0.75, 1.0],
    )

    targets = {round(float(row["target_setting"]), 6) for row in designs}
    assert 0.0 in targets and 1.0 in targets
    assert 0.40 not in targets
    assert all(abs(float(row["effective_delta"])) > 1.0e-9 for row in designs)
    assert {row["action_direction"] for row in designs} == {"increase", "decrease"}


def test_dynamic_reliability_keeps_amplitude_pattern_and_phase_separate():
    ablation = _load_script("76_generate_no_control_single_actuator_ablation.py")
    frame = pd.DataFrame(
        {
            "case_id": ["a", "b"],
            "event_id": ["E1", "E1"],
            "actuator_id": ["A", "A"],
            "action_direction": ["decrease", "decrease"],
            "action_amplitude": [0.05, 0.20],
            "amplitude_tier": ["d_0p050", "d_0p200"],
            "action_design": ["relative", "absolute"],
            "pattern": ["block", "block"],
            "phase": ["peak", "peak"],
            "reference_PFV_H": [1000.0, 1000.0],
            "reference_peak_TFV_rate_H": [10.0, 10.0],
            "effect_PFV_H": [-5.0, 400.0],
            "effect_TFV_H": [-10.0, 100.0],
            "effect_peak_TFV_rate_H": [-1.0, 2.0],
            "sequence_delta_abs_max": [0.05, 0.20],
        }
    )

    dynamic = ablation._dynamic_reliability(frame, {})

    assert len(dynamic) == 2
    assert {"action_amplitude", "pattern", "phase", "amplitude_tier"}.issubset(dynamic.columns)


def test_joint_candidate_builder_only_uses_direction_safe_actions():
    joint = _load_script("77_generate_no_control_joint_action_ablation.py")
    reliable = pd.DataFrame(
        {
            "actuator_id": ["A", "B", "C"],
            "action_direction": ["decrease", "increase", "decrease"],
            "action_amplitude": [0.10, 0.10, 0.10],
            "repair_safe_frac": [1.0, 1.0, 0.2],
            "tfv_improved_frac": [1.0, 1.0, 0.2],
            "peak_safe_frac": [1.0, 1.0, 0.2],
            "events": [4, 4, 4],
            "rows": [4, 4, 4],
        }
    )
    actions = joint._joint_action_designs(
        np.asarray([0.5, 0.5, 0.5]), ["A", "B", "C"], reliable,
        max_group_size=2, max_combinations=4,
    )

    assert actions
    assert all("C" not in item["target_settings"] for item in actions)
    assert any(set(item["target_settings"]) == {"A", "B"} for item in actions)


def test_gat_checkpoint_node_order_rebuilds_all_node_indices():
    source = (ROOT / "sewerrtc" / "simulation" / "pyswmm_runner.py").read_text(encoding="utf-8")
    checkpoint_end = source.index("sensor_idx =", source.index("if gat_model_path"))
    checkpoint_block = source[source.index("if gat_model_path") : checkpoint_end]

    assert "node_index = {n: i for i, n in enumerate(node_order)}" in checkpoint_block
    assert "priority_idx = [node_index[n]" in checkpoint_block


def test_horizon_model_persists_conformal_calibration_margins(tmp_path: Path):
    from sewerrtc.models.temporal_graph_surrogate import HorizonRidgeSurrogate

    model = HorizonRidgeSurrogate()
    model.calibration_margins = {"PFV_H": 3.0, "TFV_H": 7.0, "peak_TFV_rate_H": 0.5}
    model.feature_columns = ["x"]
    model.target_columns = ["PFV_H"]
    model.x_mean = np.asarray([0.0])
    model.x_std = np.asarray([1.0])
    model.y_mean = np.asarray([0.0])
    model.y_std = np.asarray([1.0])
    model.coef = np.zeros((2, 1))
    path = tmp_path / "ridge.npz"

    model.save(path)
    loaded = HorizonRidgeSurrogate.load(path)

    assert loaded.calibration_margins["PFV_H"] == pytest.approx(3.0)
    assert loaded.calibration_margins["peak_TFV_rate_H"] == pytest.approx(0.5)


def test_owned_torch_checkpoints_explicitly_disable_weights_only_mode():
    paths = [
        ROOT / "sewerrtc" / "simulation" / "pyswmm_runner.py",
        ROOT / "sewerrtc" / "control" / "mpc_controller.py",
        ROOT / "sewerrtc" / "models" / "residual_value.py",
        ROOT / "scripts" / "06_train_surrogate.py",
        ROOT / "scripts" / "12_train_residual_action_value.py",
    ]
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if "torch.load(" in line:
                assert "weights_only=False" in line, f"unsafe/incompatible torch.load in {path}: {line}"


def test_formal_project6_entrypoints_have_no_sibling_project_fallbacks():
    priority_script = (ROOT / "scripts" / "61_recalculate_project2_priority_zone_metrics.py").read_text(encoding="utf-8")
    pipeline = (ROOT / "scripts" / "project6_runs" / "RUN_PROJECT6_NO_CONTROL_REPAIR_PIPELINE.ps1").read_text(encoding="utf-8")

    assert 'project_root.parent / "Project2"' not in priority_script
    assert 'project_root.parent / "Project3"' not in priority_script
    assert "E:\\RTC_sewer\\Project5" not in pipeline
    assert "E:\\RTC_sewer\\Project\\env" not in pipeline
    assert "torch.cuda.is_available()" in pipeline
    assert "build_gat_reconstructed_feature_cache" in pipeline
    assert '"--require-gat-features"' in pipeline


def test_horizon_dataset_replaces_true_state_features_with_fingerprinted_gat_cache(tmp_path: Path):
    builder = _load_script("42_build_horizon_surrogate_dataset.py")
    from sewerrtc.data.dataset_fingerprint import source_file_fingerprint
    from sewerrtc.data.gat_feature_cache import gat_feature_cache_path

    detail = tmp_path / "E1__policy_detail.csv"
    detail.write_text("event_id,row\nE1,0\nE1,1\nE1,2\n", encoding="utf-8")
    samples = pd.DataFrame({"row_index": [1, 2], "current_depth_mean": [99.0, 99.0]})
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    np.savez_compressed(
        gat_feature_cache_path(cache_dir, detail),
        source_fingerprint=np.asarray(source_file_fingerprint([detail])),
        gat_fingerprint=np.asarray("model"),
        row_count=np.asarray(3),
        current_depth_mean=np.asarray([1.0, 2.0, 3.0]),
        current_depth_p95=np.asarray([1.1, 2.1, 3.1]),
        current_depth_max=np.asarray([1.2, 2.2, 3.2]),
        priority_depth_mean=np.asarray([0.1, 0.2, 0.3]),
        priority_depth_max=np.asarray([0.2, 0.4, 0.8]),
    )

    result = builder._apply_gat_features(samples, detail, cache_dir, history_steps=2, require=True)

    assert result["current_depth_mean"].tolist() == [2.0, 3.0]
    assert result["priority_depth_trend"].tolist() == pytest.approx([0.2, 0.4])
    assert result["state_feature_source"].eq("gat_reconstructed_sparse_sensors").all()

    trusted = builder._apply_gat_features(
        samples.copy(), detail, cache_dir, history_steps=2, require=True,
        verify_source_fingerprint=False,
    )
    assert trusted["state_feature_source"].eq("gat_reconstructed_sparse_sensors").all()


def test_high_risk_surrogate_validation_measures_paired_action_direction():
    validator = _load_script("44_validate_horizon_surrogate.py")
    frame = pd.DataFrame(
        {
            "event_id": ["E1", "E1", "E1", "E1"],
            "row_index": [0, 0, 1, 1],
            "policy_id": ["no_control", "candidate", "no_control", "candidate"],
            "PFV_H": [100.0, 80.0, 120.0, 140.0],
            "TFV_H": [200.0, 180.0, 220.0, 240.0],
            "peak_TFV_rate_H": [10.0, 9.0, 11.0, 12.0],
        }
    )
    pred = pd.DataFrame(
        {
            "pred_PFV_H": [105.0, 85.0, 115.0, 130.0],
            "pred_TFV_H": [205.0, 185.0, 215.0, 230.0],
            "pred_peak_TFV_rate_H": [10.5, 9.5, 10.5, 11.5],
        }
    )

    paired, summary = validator._paired_direction_report(frame, pred, high_risk_threshold=90.0)

    assert len(paired) == 2
    assert summary["PFV_direction_accuracy"] == pytest.approx(1.0)
    assert summary["joint_safe_precision"] == pytest.approx(1.0)


def test_paired_direction_validation_does_not_collide_with_dataset_reference_labels():
    validator = _load_script("44_validate_horizon_surrogate.py")
    frame = pd.DataFrame(
        {
            "event_id": ["E1", "E1"],
            "row_index": [0, 0],
            "policy_id": ["no_control", "candidate"],
            "PFV_H": [100.0, 80.0],
            "TFV_H": [200.0, 190.0],
            "peak_TFV_rate_H": [10.0, 9.0],
            # These are ordinary dataset effect-label columns. The paired
            # audit must not overwrite or suffix them during its merge.
            "reference_PFV_H": [100.0, 100.0],
            "reference_TFV_H": [200.0, 200.0],
            "reference_peak_TFV_rate_H": [10.0, 10.0],
        }
    )
    pred = pd.DataFrame(
        {
            "pred_PFV_H": [100.0, 85.0],
            "pred_TFV_H": [200.0, 195.0],
            "pred_peak_TFV_rate_H": [10.0, 9.5],
        }
    )

    paired, summary = validator._paired_direction_report(frame, pred, high_risk_threshold=50.0)

    assert len(paired) == 1
    assert "paired_reference_true_PFV_H" in paired
    assert summary["paired_rows"] == 1


def test_gat_validation_split_keeps_events_disjoint():
    trainer = _load_script("05_train_gat.py")
    events = np.asarray(["E1"] * 4 + ["E2"] * 4 + ["E3"] * 4 + ["E4"] * 4)

    train_idx, val_idx, train_events, val_events = trainer._event_grouped_indices(events, 0.25, 2026)

    assert set(train_events).isdisjoint(val_events)
    assert set(events[train_idx]).isdisjoint(set(events[val_idx]))
    assert len(train_idx) + len(val_idx) == len(events)


def test_gat_reports_reconstruction_on_unsensed_nodes_separately():
    trainer = _load_script("05_train_gat.py")
    truth = np.asarray([[1.0, 2.0, 3.0, 4.0], [2.0, 3.0, 4.0, 5.0]])
    pred = truth.copy()
    pred[:, 1] += 1.0
    pred[:, 3] -= 1.0
    metrics = {}

    trainer._add_unseen_metrics(metrics, pred, truth, np.asarray([1.0, 0.0, 1.0, 0.0]), [0, 1])

    assert metrics["unsensed_node_count"] == 2
    assert metrics["priority_unsensed_node_count"] == 1
    assert "unsensed_NSE" in metrics
    assert "priority_unsensed_RMSE" in metrics


def test_effect_prediction_is_added_to_same_time_no_control_absolute_prediction():
    validator = _load_script("44_validate_horizon_surrogate.py")
    frame = pd.DataFrame(
        {
            "event_id": ["E1", "E1"],
            "row_index": [0, 0],
            "policy_id": ["no_control", "candidate"],
        }
    )
    absolute = pd.DataFrame({"pred_PFV_H": [100.0, 130.0]})
    effect = pd.DataFrame({"pred_PFV_H": [0.0, -20.0]})

    combined = validator._combine_effect_predictions(frame, absolute, effect)

    assert combined["pred_PFV_H"].tolist() == [100.0, 80.0]


def test_formal_trajectory_schedule_balances_exploration_without_repeating_all_policies():
    generator = _load_script("03_generate_generic_trajectories.py")
    all_policies = ["no_control", "auto_rbc"] + [f"explore_{i}" for i in range(12)]
    schedules = [
        generator._balanced_policies_for_event(all_policies, ["no_control", "auto_rbc"], 4, event_index)
        for event_index in range(12)
    ]

    assert all(schedule[:2] == ["no_control", "auto_rbc"] for schedule in schedules)
    assert all(len(schedule) == 6 for schedule in schedules)
    exploration_counts = {
        policy: sum(policy in schedule for schedule in schedules) for policy in all_policies[2:]
    }
    assert max(exploration_counts.values()) - min(exploration_counts.values()) <= 1


def test_trajectory_scope_manifest_blocks_incompatible_resume(tmp_path: Path):
    generator = _load_script("03_generate_generic_trajectories.py")
    old = generator._scope_manifest("existing_rtc", pd.DataFrame({"actuator_id": ["A"]}))
    new = generator._scope_manifest("existing_plus_retrofit", pd.DataFrame({"actuator_id": ["A", "B"]}))

    generator._validate_or_write_scope_manifest(tmp_path, old, resume=False)
    with pytest.raises(ValueError, match="action schema differs"):
        generator._validate_or_write_scope_manifest(tmp_path, new, resume=True)


def test_project5_importer_uses_explicit_source_and_exact_action_headers():
    importer = _load_script("65_import_verified_project5_trajectories.py")
    actuators = pd.DataFrame({"actuator_id": ["A", "B"]})

    assert importer._action_columns(actuators) == ["a:A", "a:B"]
    with pytest.raises(ValueError, match="event/policy separator"):
        importer._event_policy(Path("invalid_detail.csv"))


def test_gat_checkpoint_contract_rejects_changed_graph_or_sensor_count(tmp_path: Path):
    trainer = _load_script("05_train_gat.py")
    from sewerrtc.models.gat_reconstructor import SparseGATReconstructor

    model = SparseGATReconstructor(2, 1, 4, 1)
    checkpoint = {
        "model": model.state_dict(),
        "node_ids": ["A", "B"],
        "n_nodes": 2,
        "static_dim": 1,
        "hidden_dim": 4,
        "gat_heads": 1,
        "sensor_count": 1,
        "node_static": np.zeros((2, 1), dtype=np.float32),
        "edge_index": np.asarray([[0, 1], [1, 0]], dtype=np.int64),
    }
    path = tmp_path / "gat.pt"
    torch.save(checkpoint, path)

    trainer._load_verified_initial_checkpoint(
        path, model, node_ids=["A", "B"], node_static=checkpoint["node_static"],
        edge_index=checkpoint["edge_index"], sensor_count=1, hidden_dim=4, gat_heads=1,
    )
    with pytest.raises(ValueError, match="sensor_count"):
        trainer._load_verified_initial_checkpoint(
            path, model, node_ids=["A", "B"], node_static=checkpoint["node_static"],
            edge_index=checkpoint["edge_index"], sensor_count=2, hidden_dim=4, gat_heads=1,
        )


def test_project6_config_isolates_109_action_gat_feature_cache():
    from sewerrtc.io.project_paths import load_config

    cfg = load_config(ROOT / "configs" / "wuhan_project6.yaml")
    assert cfg["controller"]["actuator_scope"] == "existing_plus_retrofit"
    assert cfg["outputs"]["gat_features"] == "outputs/gat_reconstructed_features_all109"


def test_online_default_reference_does_not_enable_no_control_oracle():
    closed_loop = _load_script("08_run_closed_loop.py")

    assert closed_loop._needs_no_control_reference("online_predicted_default") is False
    assert closed_loop._needs_no_control_reference("precomputed_no_control_twin_horizon") is True


def test_effect_uncertainty_preserves_signed_candidate_minus_reference_values():
    from sewerrtc.models.uncertainty import ResidualQuantileUncertainty

    uncertainty = ResidualQuantileUncertainty(
        ["PFV_H", "TFV_H", "peak_TFV_rate_H"],
        q50=np.asarray([0.0, 0.0, 0.0]),
        q90=np.asarray([1.0, 2.0, 3.0]),
    )
    pred = pd.DataFrame(
        {
            "pred_PFV_H": [-10.0],
            "pred_TFV_H": [-5.0],
            "pred_peak_TFV_rate_H": [-2.0],
        }
    )

    signed = uncertainty.predict_quantiles(pred, clip_lower=False)
    absolute = uncertainty.predict_quantiles(pred, clip_lower=True)

    assert signed.loc[0, "PFV_H_p50"] == -10.0
    assert signed.loc[0, "TFV_H_p90"] == -3.0
    assert absolute.loc[0, "PFV_H_p50"] == 0.0


def test_online_effect_predictor_forces_zero_for_the_reference_sequence():
    from sewerrtc.simulation.pyswmm_runner import _zero_effect_for_reference_sequences

    reference = np.asarray([[0.85, 1.0], [0.85, 1.0]], dtype=np.float32)
    effects = pd.DataFrame({"pred_PFV_H": [3.0, -2.0], "pred_TFV_H": [50.0, -10.0]})
    out = _zero_effect_for_reference_sequences(
        effects,
        [reference.copy(), np.asarray([[0.70, 1.0], [0.70, 1.0]], dtype=np.float32)],
        [{"reference_action_sequence": reference}, {"reference_action_sequence": reference}],
    )

    assert out.loc[0, "pred_PFV_H"] == 0.0
    assert out.loc[0, "pred_TFV_H"] == 0.0
    assert out.loc[1, "pred_PFV_H"] == -2.0


def test_phase_reliability_denies_actions_without_exact_local_evidence():
    from sewerrtc.simulation.pyswmm_runner import _phase_reliable_action_filter

    table = pd.DataFrame(
        {
            "pattern": ["block"], "phase": ["peak"], "actuator_id": ["A"],
            "action_direction": ["decrease"], "repair_safe_frac": [1.0],
            "pfv_noninferior_frac": [1.0], "tfv_improved_frac": [1.0],
            "peak_safe_frac": [1.0], "rows": [1],
        }
    )

    ids, directions, source, state_threshold = _phase_reliable_action_filter(
        table, pattern="block", phase="peak"
    )
    assert ids == ["A"]
    assert directions == {"A": ["decrease"]}
    assert source == "phase_reliability_exact_local"
    assert state_threshold is None
    assert _phase_reliable_action_filter(table, pattern="block", phase="recession")[0] == []


def test_runtime_empty_phase_filter_is_deny_all_not_unfiltered():
    from sewerrtc.control.generic_gat_mpc import GenericGATMPCController

    controller = GenericGATMPCController(pd.DataFrame({"actuator_id": ["A"]}))
    controller.set_runtime_action_filter([], {})

    assert controller.runtime_allowed_actuator_ids == ("__NO_RELIABLE_ACTUATOR__",)
