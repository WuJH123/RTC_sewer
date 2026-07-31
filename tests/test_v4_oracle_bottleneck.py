#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Regression tests for V4 Oracle bottleneck diagnosis (Gate 0/1/2)."""

from __future__ import annotations

import importlib.util as _ilu
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load 206
_206 = _ilu.spec_from_file_location("_oracle206_test", str(PROJECT_ROOT / "scripts" / "206_oracle_pareto_v4.py"))
_o206 = _ilu.module_from_spec(_206)
sys.modules["_oracle206_test"] = _o206  # required for dataclass on py3.9
_206.loader.exec_module(_o206)  # type: ignore[union-attr]

from sewerrtc.prompt3.oracle_constraint_ablation_v4 import (
    ABLATION_MODES,
    project_schedule_ablation,
    ablation_mode_for_constraint_mode,
    constraint_mode_for_ablation,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def small_actuators() -> pd.DataFrame:
    """Minimal 4-actuator table (2 binary pumps, 2 variable)."""
    return pd.DataFrame({
        "actuator_id": ["P1", "P2", "V1", "V2"],
        "link_type": ["pump", "pump", "pump", "pump"],
        "storage_control_type": ["", "", "", ""],
        "storage_node": ["", "", "", ""],
    })


@pytest.fixture()
def small_cfg() -> dict:
    return {
        "controller": {
            "per_actuator_max_delta": {"P1": 1.0, "P2": 1.0, "V1": 0.1, "V2": 0.1},
            "max_first_step_delta": 1.0,
            "min_hold_steps_by_actuator": {"P1": 2, "P2": 2},
            "variable_speed_pump_ids": ["V1", "V2"],
            "storage_retrofit": {"inlet_outlet_incompatible_action_constraint": False},
        },
    }


@pytest.fixture()
def small_eng_cfg() -> dict:
    return {
        "engineering36": {"variable_speed_pump_ids": ["V1", "V2"]},
    }


@pytest.fixture()
def small_matrices():
    T, N = 10, 4
    rng = np.random.default_rng(42)
    matrix = rng.uniform(0, 1, size=(T, N))
    anchor = np.zeros((T, N))
    anchor[:, :2] = 0.0  # binary pumps off
    anchor[:, 2:] = 0.3  # variable pumps at 0.3
    return matrix, anchor


# ---------------------------------------------------------------------------
# Tests: ABLATION_MODES table
# ---------------------------------------------------------------------------

class TestAblationModes:
    def test_nine_modes(self):
        assert len(ABLATION_MODES) == 9

    def test_a0_all_on(self):
        m = ABLATION_MODES["A0_full_constraints"]
        assert all(m.values())

    def test_a8_all_off(self):
        m = ABLATION_MODES["A8_operational_relaxed"]
        assert not any(m.values())

    def test_all_modes_have_four_keys(self):
        expected = {"rate", "dwell", "topk", "interlock"}
        for name, mask in ABLATION_MODES.items():
            assert set(mask.keys()) == expected, f"Mode {name} missing keys"


# ---------------------------------------------------------------------------
# Tests: project_schedule_ablation
# ---------------------------------------------------------------------------

class TestProjectScheduleAblation:
    def test_output_shape(self, small_actuators, small_cfg, small_eng_cfg, small_matrices):
        matrix, anchor = small_matrices
        for mode_name, mask in ABLATION_MODES.items():
            out = project_schedule_ablation(
                matrix, anchor=anchor, actuators=small_actuators,
                cfg=small_cfg, engineering_cfg=small_eng_cfg,
                constraint_mask=mask, max_k=4,
            )
            assert out.shape == matrix.shape, f"Shape mismatch for {mode_name}"

    def test_output_range(self, small_actuators, small_cfg, small_eng_cfg, small_matrices):
        matrix, anchor = small_matrices
        for mode_name, mask in ABLATION_MODES.items():
            out = project_schedule_ablation(
                matrix, anchor=anchor, actuators=small_actuators,
                cfg=small_cfg, engineering_cfg=small_eng_cfg,
                constraint_mask=mask, max_k=4,
            )
            assert out.min() >= 0.0, f"Below 0 for {mode_name}"
            assert out.max() <= 1.0, f"Above 1 for {mode_name}"

    def test_a0_equals_constrained(self, small_actuators, small_cfg, small_eng_cfg, small_matrices):
        """A0 (all constraints ON) must equal project_schedule('constrained')."""
        matrix, anchor = small_matrices
        mask = ABLATION_MODES["A0_full_constraints"]
        ablation_out = project_schedule_ablation(
            matrix, anchor=anchor, actuators=small_actuators,
            cfg=small_cfg, engineering_cfg=small_eng_cfg,
            constraint_mask=mask, max_k=4,
        )
        constrained_out = _o206.project_schedule(
            matrix, anchor=anchor, actuators=small_actuators,
            cfg=small_cfg, engineering_cfg=small_eng_cfg,
            constraint_mode="constrained", max_k=4,
        )
        np.testing.assert_array_almost_equal(ablation_out, constrained_out, decimal=10)

    def test_a8_equals_relaxed(self, small_actuators, small_cfg, small_eng_cfg, small_matrices):
        """A8 (all constraints OFF) must equal project_schedule('relaxed')."""
        matrix, anchor = small_matrices
        mask = ABLATION_MODES["A8_operational_relaxed"]
        ablation_out = project_schedule_ablation(
            matrix, anchor=anchor, actuators=small_actuators,
            cfg=small_cfg, engineering_cfg=small_eng_cfg,
            constraint_mask=mask, max_k=None,
        )
        relaxed_out = _o206.project_schedule(
            matrix, anchor=anchor, actuators=small_actuators,
            cfg=small_cfg, engineering_cfg=small_eng_cfg,
            constraint_mode="relaxed", max_k=None,
        )
        np.testing.assert_array_almost_equal(ablation_out, relaxed_out, decimal=10)

    def test_relaxing_more_constraints_gives_more_freedom(self, small_actuators, small_cfg, small_eng_cfg, small_matrices):
        """Relaxing constraints should produce equal or more deviation from anchor."""
        matrix, anchor = small_matrices
        a0_mask = ABLATION_MODES["A0_full_constraints"]
        a8_mask = ABLATION_MODES["A8_operational_relaxed"]
        out_a0 = project_schedule_ablation(
            matrix, anchor=anchor, actuators=small_actuators,
            cfg=small_cfg, engineering_cfg=small_eng_cfg,
            constraint_mask=a0_mask, max_k=4,
        )
        out_a8 = project_schedule_ablation(
            matrix, anchor=anchor, actuators=small_actuators,
            cfg=small_cfg, engineering_cfg=small_eng_cfg,
            constraint_mask=a8_mask, max_k=None,
        )
        dev_a0 = np.abs(out_a0 - anchor).sum()
        dev_a8 = np.abs(out_a8 - anchor).sum()
        assert dev_a8 >= dev_a0 - 1e-9, "Relaxing all constraints should allow >= deviation"


# ---------------------------------------------------------------------------
# Tests: mapping functions
# ---------------------------------------------------------------------------

class TestMappingFunctions:
    def test_ablation_to_constraint_mode(self):
        assert ablation_mode_for_constraint_mode("constrained") == "A0_full_constraints"
        assert ablation_mode_for_constraint_mode("relaxed") == "A8_operational_relaxed"

    def test_constraint_mode_roundtrip(self):
        assert constraint_mode_for_ablation("A0_full_constraints") == "constrained"
        assert constraint_mode_for_ablation("A8_operational_relaxed") == "relaxed"
        assert constraint_mode_for_ablation("A1_relax_rate").startswith("ablation_")

    def test_unknown_constraint_mode_raises(self):
        with pytest.raises(ValueError):
            ablation_mode_for_constraint_mode("unknown_mode")


# ---------------------------------------------------------------------------
# Tests: event classification (Gate 1 logic)
# ---------------------------------------------------------------------------

class TestEventClassification:
    def _make_frame(self, constraint_modes, strict_feasible_flags):
        n = len(constraint_modes)
        return pd.DataFrame({
            "constraint_mode": constraint_modes,
            "strict_feasible": strict_feasible_flags,
            "pfv_feasible": strict_feasible_flags,
            "tfv_feasible": strict_feasible_flags,
            "peak_feasible": strict_feasible_flags,
            "nonpriority_feasible": [True] * n,
        })

    def test_constrained_feasible_found(self):
        frame = self._make_frame(["constrained", "constrained", "relaxed"], [True, False, False])
        result = _o206.classify_event(frame)
        assert result == "feasible_found"

    def test_relaxed_only_feasible(self):
        # classify_event returns feasible_found because at least one row is feasible
        # The constrained vs relaxed distinction is in Gate 1 reclassification
        frame = self._make_frame(["constrained", "constrained", "relaxed"], [False, False, True])
        result = _o206.classify_event(frame)
        assert result == "feasible_found"

    def test_no_feasible_at_all(self):
        frame = self._make_frame(["constrained", "relaxed"], [False, False])
        result = _o206.classify_event(frame)
        assert result in (
            "pfv_safe_but_internal_performance_unreachable",
            "no_feasible_neighbourhood_solution",
            "internal_performance_reachable_but_pfv_unsafe",
            "objectives_reachable_separately_not_jointly",
        )
