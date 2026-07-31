"""Tests for V4.2 reserved evaluation isolation (v42_sample_classifier).

Verifies:
- Reserved samples not in any development dataset
- Calibration/Locked/Formal/Challenge excluded
"""

from __future__ import annotations

import pandas as pd
import pytest

from sewerrtc.v4.v42_sample_classifier import (
    AdmissionGrade,
    _is_reserved_split,
    _classify_single_sample,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestReservedSplitDetection:
    """Reserved samples correctly identified."""

    def test_calibration_is_reserved(self):
        assert _is_reserved_split("calibration") is True

    def test_locked_validation_is_reserved(self):
        assert _is_reserved_split("locked_validation") is True

    def test_formal_is_reserved(self):
        assert _is_reserved_split("formal") is True

    def test_challenge_is_reserved(self):
        assert _is_reserved_split("challenge") is True

    def test_round0_not_reserved(self):
        assert _is_reserved_split("round0") is False

    def test_empty_not_reserved(self):
        assert _is_reserved_split("") is False

    def test_case_insensitive(self):
        assert _is_reserved_split("Calibration") is True
        assert _is_reserved_split("FORMAL") is True


class TestReservedClassification:
    """Reserved samples classified as RESERVED_EVALUATION."""

    def test_reserved_split_gets_reserved_grade(self):
        lineage_row = pd.Series({
            "sample_idx": 0,
            "event_id": "E1",
            "checkpoint_id": "C1",
            "state_key": "S1",
            "split": "calibration",
            "source_round": "calibration",
        })
        grade, reasons, details = _classify_single_sample(
            sample_idx=0,
            lineage_row=lineage_row,
            semantic_row=None,
            dwf_row=None,
            pfv_pass=None,
            tfv_peak_pass=None,
            history_info=None,
            is_physical_duplicate=False,
            has_actual_action=True,
            has_facility_response=True,
            has_real_kpi_difference=True,
            n_actual_unique_candidates=1,
        )
        assert grade == AdmissionGrade.RESERVED_EVALUATION

    def test_all_four_reserved_splits_excluded(self):
        """All 4 reserved splits produce RESERVED_EVALUATION."""
        for split in ["calibration", "locked_validation", "formal", "challenge"]:
            lineage_row = pd.Series({
                "sample_idx": 0,
                "event_id": "E1",
                "checkpoint_id": "C1",
                "split": split,
                "source_round": split,
            })
            grade, _, _ = _classify_single_sample(
                sample_idx=0,
                lineage_row=lineage_row,
                semantic_row=None,
                dwf_row=None,
                pfv_pass=None,
                tfv_peak_pass=None,
                history_info=None,
                is_physical_duplicate=False,
                has_actual_action=True,
                has_facility_response=True,
                has_real_kpi_difference=True,
                n_actual_unique_candidates=1,
            )
            assert grade == AdmissionGrade.RESERVED_EVALUATION


class TestReservedNotInDevelopment:
    """Reserved samples not in any development dataset."""

    def test_reserved_grade_not_in_development_set(self):
        development_grades = {
            AdmissionGrade.TARGET_FULL_SUPERVISION,
            AdmissionGrade.TARGET_RECOMPUTABLE,
            AdmissionGrade.SOURCE_DWF_FULL_SUPERVISION,
            AdmissionGrade.DYNAMICS_PRETRAIN_ONLY,
            AdmissionGrade.ACTUATOR_EFFECT_ONLY,
            AdmissionGrade.RANKING_ONLY,
            AdmissionGrade.DIAGNOSTIC_ONLY,
            AdmissionGrade.CONSUMED_DEVELOPMENT,
        }
        assert AdmissionGrade.RESERVED_EVALUATION not in development_grades
