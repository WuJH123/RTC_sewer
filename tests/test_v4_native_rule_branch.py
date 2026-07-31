from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sewerrtc.io.swmm_mutation import inject_time_gated_control_schedule
from sewerrtc.prompt3.gate5r_pipeline import (
    branch_state_hashes,
    hashes_match_across_branches,
)
from sewerrtc.simulation.pyswmm_runner import managed_setting_write_required


def test_injects_inactive_prefix_and_twelve_ten_minute_priority_rules(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source.inp"
    source.write_text(
        "\n".join(
            [
                "[PUMPS]",
                "p1 n1 n2 curve1",
                "",
                "[ORIFICES]",
                "o1 n1 n2 BOTTOM 0 1 NO 0",
                "",
                "[CONTROLS]",
                "RULE native",
                "IF NODE n1 DEPTH > 1",
                "THEN ORIFICE o1 SETTING = 1",
                "",
                "[REPORT]",
                "INPUT YES",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    semantics = pd.DataFrame(
        {
            "facility_id": ["o1", "p1"],
            "actuator_type": ["weir", "pump"],
            "binary_or_continuous": ["continuous", "binary"],
        }
    )
    schedule = np.asarray([[0.25, 1.0]] * 12, dtype=float)
    output = tmp_path / "candidate.inp"

    inject_time_gated_control_schedule(
        source,
        output,
        semantics,
        schedule,
        checkpoint_min=360.0,
        decision_interval_sec=600,
        priority=100,
        rule_prefix="G5R_CAND",
    )

    text = output.read_text(encoding="utf-8")
    assert text.count("RULE G5R_CAND_STEP_") == 12
    assert "IF SIMULATION TIME >= 06:00:00" in text
    assert "AND SIMULATION TIME < 06:10:00" in text
    assert "THEN ORIFICE o1 SETTING = 0.250000" in text
    assert "AND PUMP p1 SETTING = 1.000000" in text
    assert text.count("PRIORITY 100") == 12
    assert text.index("RULE native") < text.index("RULE G5R_CAND_STEP_00")
    assert text.index("RULE G5R_CAND_STEP_11") < text.index("[REPORT]")


def test_branch_hashes_cover_full_prefix_checkpoint_and_post_schedule() -> None:
    frame = pd.DataFrame(
        {
            "elapsed_min": [300.0 + 5.0 * index for index in range(37)],
            "h:n": np.arange(37, dtype=float),
            "flow:a": np.arange(37, dtype=float) / 10.0,
            "actual_setting:a": [0.0] * 12 + [1.0] * 25,
            "readback_setting:a": [0.0] * 12 + [1.0] * 25,
        }
    )

    first = branch_state_hashes(frame, checkpoint_min=360.0, facility_ids=["a"])
    second = branch_state_hashes(
        frame.copy(), checkpoint_min=360.0, facility_ids=["a"]
    )

    assert first["prefix_history_rows"] == 12
    assert first["prefix_history_sha256"] == second["prefix_history_sha256"]
    assert first["checkpoint_pre_action_sha256"] == second[
        "checkpoint_pre_action_sha256"
    ]
    assert first["post_actual_schedule_sha256"] == second[
        "post_actual_schedule_sha256"
    ]
    endpoint_changed = frame.copy()
    endpoint_changed.loc[
        np.isclose(endpoint_changed["elapsed_min"], 480.0),
        ["actual_setting:a", "readback_setting:a"],
    ] = 0.0
    endpoint_hashes = branch_state_hashes(
        endpoint_changed, checkpoint_min=360.0, facility_ids=["a"]
    )
    # The 480 min sample is after the [360, 480) H120 control window:
    # the time-gated rule has expired and native control may resume.
    assert endpoint_hashes["post_actual_schedule_sha256"] == first[
        "post_actual_schedule_sha256"
    ]
    assert endpoint_hashes["post_readback_schedule_sha256"] == first[
        "post_readback_schedule_sha256"
    ]
    assert hashes_match_across_branches(
        {"candidate": first, "no_control": second},
        keys=("prefix_history_sha256", "checkpoint_pre_action_sha256"),
    )


def test_native_rule_branch_never_overwrites_managed_settings() -> None:
    assert not managed_setting_write_required(
        in_prefix=True,
        prefix_schedule_is_none=True,
        post_control_mode="native_rules",
    )
    assert not managed_setting_write_required(
        in_prefix=False,
        prefix_schedule_is_none=True,
        post_control_mode="native_rules",
    )
    assert managed_setting_write_required(
        in_prefix=False,
        prefix_schedule_is_none=True,
        post_control_mode="external_override",
    )
