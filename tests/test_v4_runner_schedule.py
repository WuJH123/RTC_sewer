from __future__ import annotations

import numpy as np
import inspect
import pandas as pd
from pathlib import Path

from sewerrtc.contracts.swmm_control_parser import parse_swmm_controls
from sewerrtc.io.swmm_mutation import mutate_inp_for_event
from sewerrtc.simulation.kpi_metrics import compute_window_kpis
from sewerrtc.simulation.pyswmm_runner import run_swmm_fixed_action, select_post_action


def test_select_post_action_repeats_each_ten_minute_action_for_two_samples() -> None:
    schedule = np.arange(24, dtype=float).reshape(12, 2)

    assert np.array_equal(
        select_post_action(schedule, elapsed_min=60.0, override_start_min=60.0, decision_interval_sec=600),
        schedule[0],
    )
    assert np.array_equal(
        select_post_action(schedule, elapsed_min=65.0, override_start_min=60.0, decision_interval_sec=600),
        schedule[0],
    )
    assert np.array_equal(
        select_post_action(schedule, elapsed_min=70.0, override_start_min=60.0, decision_interval_sec=600),
        schedule[1],
    )


def test_select_post_action_holds_last_step_after_schedule_end() -> None:
    schedule = np.array([[0.0], [1.0]])

    selected = select_post_action(
        schedule,
        elapsed_min=200.0,
        override_start_min=0.0,
        decision_interval_sec=600,
    )

    assert selected.tolist() == [1.0]


def test_fixed_action_signature_allows_native_prefix() -> None:
    signature = inspect.signature(run_swmm_fixed_action)
    annotation = str(signature.parameters["prefix_schedule"].annotation)
    assert "None" in annotation
    assert signature.parameters["prefix_history_min"].default is None
    assert signature.parameters["record_node_ids"].default is None
    assert signature.parameters["hydraulic_summary_start_min"].default is None


def test_tiny_swmm_native_prefix_readback_and_peak_units(tmp_path) -> None:
    inp = Path(__file__).parent / "fixtures" / "v4_tiny_network" / "tiny.inp"
    actuators = pd.DataFrame(
        [{"actuator_id": "P1", "facility_id": "P1", "link_type": "pump"}]
    )
    details = []
    for name, schedule in (
        ("closed", np.asarray([[0.0], [0.0]], dtype=float)),
        ("open", np.asarray([[1.0], [1.0]], dtype=float)),
    ):
        output = tmp_path / f"{name}.csv"
        run_swmm_fixed_action(
            inp_path=inp,
            actuators=actuators,
            priority_nodes=["J1"],
            out_detail_csv=output,
            event_id="tiny",
            duration_min=30,
            prefix_schedule=None,
            override_start_min=10.0,
            post_action=schedule,
            control_step_sec=300,
            decision_interval_sec=600,
            stop_after_override_min=20,
            prefix_history_min=10,
            simulation_duration_min=30,
            policy_id=name,
            cleanup_swmm_artifacts=True,
            hydraulic_summary_start_min=0.0,
        )
        details.append(pd.read_csv(output))

    closed, opened = details
    prefix_columns = ["h:J1", "h:J2", "a:P1"]
    assert np.allclose(
        closed.loc[closed["elapsed_min"] < 10, prefix_columns],
        opened.loc[opened["elapsed_min"] < 10, prefix_columns],
    )
    for detail in details:
        post = detail[detail["elapsed_min"] >= 10]
        assert np.allclose(post["requested_setting:P1"], post["a:P1"])

    kpi = compute_window_kpis(closed, ["J1"], 10.0, 20.0, 300)
    expected_peak = float(
        closed.loc[
            (closed["elapsed_min"] >= 10) & (closed["elapsed_min"] < 30),
            ["flood:J1", "flood:J2"],
        ].sum(axis=1).max()
    )
    assert kpi["peak_TFV_rate"] == expected_peak


def test_tiny_swmm_targeted_control_removal_makes_external_action_hydraulic(
    tmp_path,
) -> None:
    source = Path(__file__).parent / "fixtures" / "v4_tiny_network" / "tiny.inp"
    rain = tmp_path / "rain.csv"
    pd.DataFrame(
        {
            "elapsed_min": [0, 5, 10, 15, 20, 25, 30],
            "intensity_mm_h": [0, 2, 4, 6, 3, 1, 0],
        }
    ).to_csv(rain, index=False)
    managed = mutate_inp_for_event(
        source,
        rain,
        tmp_path / "managed.inp",
        30,
        disabled_control_targets=["P1"],
    )
    assert "P1 SETTING" not in managed.read_text(encoding="utf-8")
    actuators = pd.DataFrame(
        [{"actuator_id": "P1", "facility_id": "P1", "link_type": "pump"}]
    )
    flows = {}
    for name, setting in (("closed", 0.0), ("open", 1.0)):
        output = tmp_path / f"managed_{name}.csv"
        run_swmm_fixed_action(
            inp_path=managed,
            actuators=actuators,
            priority_nodes=["J1"],
            out_detail_csv=output,
            event_id="tiny",
            duration_min=30,
            prefix_schedule={5.0: np.asarray([1.0])},
            override_start_min=10.0,
            post_action=np.asarray([[setting], [setting]]),
            control_step_sec=300,
            decision_interval_sec=600,
            stop_after_override_min=20,
            prefix_history_min=10,
            simulation_duration_min=30,
            policy_id=name,
            cleanup_swmm_artifacts=True,
            hydraulic_summary_start_min=0.0,
        )
        detail = pd.read_csv(output)
        flows[name] = detail.loc[detail["elapsed_min"] >= 15, "flow:P1"].to_numpy()

    assert np.max(np.abs(flows["open"] - flows["closed"])) > 0.1


def test_targeted_control_removal_preserves_background_rules(tmp_path) -> None:
    source = Path(__file__).parent / "fixtures" / "v4_tiny_network" / "tiny.inp"
    expanded = tmp_path / "two_rules.inp"
    expanded.write_text(
        source.read_text(encoding="utf-8").replace(
            "[REPORT]",
            "RULE 2\nIF NODE J2 DEPTH > 0.1\n"
            "THEN LINK C1 SETTING = 1.0\nPRIORITY 1\n\n[REPORT]",
        ),
        encoding="utf-8",
    )
    rain = tmp_path / "rain.csv"
    pd.DataFrame(
        {"elapsed_min": [0, 5], "intensity_mm_h": [0.0, 0.0]}
    ).to_csv(rain, index=False)

    managed = mutate_inp_for_event(
        expanded,
        rain,
        tmp_path / "background_preserved.inp",
        30,
        disabled_control_targets=["P1"],
    )
    targets = {
        action["actuator_id"]
        for action in parse_swmm_controls(managed)["actions"]
    }

    assert "P1" not in targets
    assert "C1" in targets
