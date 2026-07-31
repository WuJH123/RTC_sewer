"""Tests for Gate 3 Golden Case Planner V4."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]

from sewerrtc.prompt3.golden_case_planner_v4 import GoldenCasePlannerV4


@pytest.fixture(scope="module")
def planner():
    """Run the planner once for all tests."""
    p = GoldenCasePlannerV4(PROJECT_ROOT)
    p.run()
    return p


@pytest.fixture(scope="module")
def out_dir():
    return PROJECT_ROOT / "outputs" / "project6_dual_reference_v4" / "golden_v4" / "planning"


# ---------------------------------------------------------------------------
# Output file existence
# ---------------------------------------------------------------------------

REQUIRED_FILES = [
    "v4_golden_event_inventory.csv",
    "v4_golden_event_selection.csv",
    "v4_golden_recovery_eligibility.csv",
    "v4_golden_checkpoint_catalog.csv",
    "v4_golden_case_plan.csv",
    "v4_golden_reference_plan.csv",
    "v4_golden_candidate_coverage.csv",
    "v4_golden_batch_schedule.csv",
    "v4_golden_formal_blacklist.json",
    "v4_golden_plan_audit.json",
    "v4_golden_plan_provenance.json",
    "completion.json",
]


@pytest.mark.parametrize("fname", REQUIRED_FILES)
def test_output_file_exists(out_dir, fname):
    assert (out_dir / fname).exists(), f"Missing output: {fname}"


# ---------------------------------------------------------------------------
# Event selection
# ---------------------------------------------------------------------------

def test_selected_event_count(planner):
    assert len(planner.selected_events) == 8


def test_censored_stress_count(planner):
    assert len(planner.censored_stress) <= 2


def test_recovery_qualified_count(planner):
    # All events in formal blind -> 0 recovery-qualified
    assert len(planner.recovery_qualified) == 0


def test_censored_event_is_108(planner):
    assert "V31_RP10_D5H_P35_v31_independent_gamma_108" in planner.censored_stress


def test_censored_event_label_scope(planner):
    ev = planner.events["V31_RP10_D5H_P35_v31_independent_gamma_108"]
    assert ev.label_scope == "h120_only"
    assert ev.recovery_class == "censored_stress"


# ---------------------------------------------------------------------------
# Duration diversity
# ---------------------------------------------------------------------------

def test_duration_diversity(planner):
    durations = set(planner.events[e].duration_min for e in planner.selected_events)
    assert len(durations) >= 2, f"Need >= 2 duration classes, got {durations}"


def test_pattern_diversity(planner):
    patterns = set(planner.events[e].pattern for e in planner.selected_events)
    assert len(patterns) >= 2, f"Need >= 2 patterns, got {patterns}"


# ---------------------------------------------------------------------------
# Checkpoint planning
# ---------------------------------------------------------------------------

def test_checkpoint_count(planner):
    assert len(planner.checkpoint_plans) == 8 * 5  # 8 events x 5 phases


def test_checkpoint_phases(planner):
    for eid in planner.selected_events:
        phases = [cp.rainfall_phase for cp in planner.checkpoint_plans if cp.event_id == eid]
        assert set(phases) == {"rising", "pre_peak", "peak", "late_rain", "recession"}


def test_checkpoint_monotonic(planner):
    """Checkpoints must be monotonically increasing within each event."""
    for eid in planner.selected_events:
        cps = [cp for cp in planner.checkpoint_plans if cp.event_id == eid]
        cps.sort(key=lambda c: c.elapsed_min)
        for i in range(1, len(cps)):
            assert cps[i].elapsed_min > cps[i - 1].elapsed_min


def test_checkpoint_future_info_flag(planner):
    for cp in planner.checkpoint_plans:
        assert cp.planning_only_future_information is True


# ---------------------------------------------------------------------------
# Candidate families
# ---------------------------------------------------------------------------

def test_candidate_family_count(planner):
    assert len(planner.CANDIDATE_FAMILIES) >= 22


def test_candidate_family_categories(planner):
    cats = set(f.category for f in planner.CANDIDATE_FAMILIES)
    assert cats >= {"reference", "diagnostic", "safety", "perturbation"}


# ---------------------------------------------------------------------------
# Formal blacklist
# ---------------------------------------------------------------------------

def test_formal_blacklist_nonempty(planner):
    assert len(planner.formal_blacklist) > 0


def test_all_selected_in_blacklist(planner):
    """All golden events should be in formal blind."""
    bl = set(planner.formal_blacklist)
    for eid in planner.selected_events:
        assert eid in bl or planner.events[eid].in_formal_blind


# ---------------------------------------------------------------------------
# Gate 3 metadata
# ---------------------------------------------------------------------------

def test_gate3_metadata(out_dir):
    completion = json.loads((out_dir / "completion.json").read_text(encoding="utf-8"))
    meta = completion["gate3_metadata"]
    assert meta["h120_execution_valid"] is True
    assert meta["same_state_counterfactual_valid"] is True
    assert meta["action_hydraulic_causality_valid"] is True
    assert meta["full_event_valid_for_current_stress_event"] is False
    assert meta["current_event_recovery_censored"] is True
    assert meta["gate4_authorized"] is False


def test_gate3_verdict(out_dir):
    verdict = json.loads((out_dir / "gate3_verdict.json").read_text(encoding="utf-8"))
    assert verdict["gate4_authorized"] is False
    assert verdict["authorization_type"] == "PLAN_ONLY_CONDITIONAL"


# ---------------------------------------------------------------------------
# Plan accounting
# ---------------------------------------------------------------------------

def test_case_plan_row_count(out_dir):
    import pandas as pd
    df = pd.read_csv(out_dir / "v4_golden_case_plan.csv")
    # 8 events * 5 checkpoints * 22 families = 880
    assert len(df) == 880


def test_reference_plan_row_count(out_dir):
    import pandas as pd
    df = pd.read_csv(out_dir / "v4_golden_reference_plan.csv")
    # 8 events * 5 checkpoints * 3 reference families = 120
    assert len(df) == 120


def test_batch_schedule_row_count(out_dir):
    import pandas as pd
    df = pd.read_csv(out_dir / "v4_golden_batch_schedule.csv")
    # 8 events * 5 checkpoints = 40
    assert len(df) == 40


# ---------------------------------------------------------------------------
# No SWMM execution
# ---------------------------------------------------------------------------

def test_no_new_swmm(out_dir):
    audit = json.loads((out_dir / "v4_golden_plan_audit.json").read_text(encoding="utf-8"))
    assert audit["no_new_swmm_run"] is True
