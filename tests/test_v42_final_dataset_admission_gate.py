"""Tests for V4.2 final dataset admission gate (v42_admission_gate).

Verifies:
- Gate checks return correct structure
- All 13 checks are evaluated
- Verdict is one of PASS/DATA_CONTRACT_FAIL/UNDERPOWERED/PARTIAL_POOL_ONLY
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from sewerrtc.v4.v42_admission_gate import (
    run_admission_gate,
    GateVerdict,
    AdmissionGateResult,
    _CRITICAL_CHECKS,
    _EXPECTED_DATASETS,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGateVerdictValues:
    """Verdict is one of PASS/DATA_CONTRACT_FAIL/UNDERPOWERED/PARTIAL_POOL_ONLY."""

    def test_pass_exists(self):
        assert GateVerdict.PASS == "PASS"

    def test_data_contract_fail_exists(self):
        assert GateVerdict.DATA_CONTRACT_FAIL == "DATA_CONTRACT_FAIL"

    def test_underpowered_exists(self):
        assert GateVerdict.UNDERPOWERED == "UNDERPOWERED"

    def test_partial_pool_only_exists(self):
        assert GateVerdict.PARTIAL_POOL_ONLY == "PARTIAL_POOL_ONLY"

    def test_exactly_four_verdicts(self):
        assert len(GateVerdict) == 4


class TestGateCheckStructure:
    """Gate checks return correct structure."""

    def test_result_has_verdict(self):
        result = AdmissionGateResult(
            verdict=GateVerdict.PASS,
            checks={},
            summary="test",
        )
        assert result.verdict == GateVerdict.PASS

    def test_result_has_checks_dict(self):
        result = AdmissionGateResult(
            verdict=GateVerdict.PASS,
            checks={"01_test": {"pass": True, "detail": "ok"}},
            summary="test",
        )
        assert "01_test" in result.checks
        assert result.checks["01_test"]["pass"] is True

    def test_result_has_summary_string(self):
        result = AdmissionGateResult(
            verdict=GateVerdict.PASS,
            checks={},
            summary="All checks passed",
        )
        assert isinstance(result.summary, str)


class TestThirteenChecksEvaluated:
    """All 13 checks are evaluated."""

    def test_critical_checks_defined(self):
        assert _CRITICAL_CHECKS == {1, 2, 3, 7, 8, 10}

    def test_expected_datasets_count(self):
        assert len(_EXPECTED_DATASETS) == 12

    def test_run_admission_gate_returns_13_checks(self, tmp_path: Path):
        """run_admission_gate should evaluate exactly 13 checks."""
        # Set up minimal directory structure
        audit_dir = tmp_path / "audits" / "v42_final_pool"
        audit_dir.mkdir(parents=True)
        data_dir = tmp_path / "data" / "v42_final_unified"
        data_dir.mkdir(parents=True)

        # Create minimal audit files so checks can run
        # history_rebuild_audit.json
        (audit_dir / "history_rebuild_audit.json").write_text(json.dumps({
            "samples_with_full_13_frames": 10,
            "total_samples_attempted": 10,
        }))

        # semantic_source_summary.csv
        (audit_dir / "semantic_source_summary.csv").write_text(
            "source_round,n_samples,n_events,branch_contract_failures,"
            "action_contract_failures,time_contract_failures\n"
            "round0,10,2,0,0,0\n"
        )

        # pfv_oracle_audit.json
        (audit_dir / "pfv_oracle_audit.json").write_text(json.dumps({
            "pass": True, "n_samples": 10, "n_mismatches": 0,
        }))

        # tfv_peak_oracle_audit.json
        (audit_dir / "tfv_peak_oracle_audit.json").write_text(json.dumps({
            "pass": True, "n_samples": 10,
            "tfv_max_abs_error_m3": 0.0,
            "peak_max_abs_error_m3s": 0.0,
        }))

        # deduplication_audit.json
        (audit_dir / "deduplication_audit.json").write_text(json.dumps({
            "total_samples": 10,
            "duplicate_group_count": 0,
            "duplicate_sample_count": 0,
        }))

        # dwf_audit_summary.json
        (audit_dir / "dwf_audit_summary.json").write_text(json.dumps({
            "total_samples": 10,
            "classification_counts": {"SOURCE_DWF_FULL_SUPERVISION": 5},
        }))

        # dataset_manifest.json
        (data_dir / "dataset_manifest.json").write_text(json.dumps({
            "per_dataset_counts": {
                "target_no_dwf_full_supervision": 200,
                "source_dwf_full_supervision": 50,
            },
            "datasets": [{"dataset": d, "sample_count": 10} for d in _EXPECTED_DATASETS],
        }))

        # Create minimal parquet files for schema check
        import pandas as pd
        for fname in [
            "target_no_dwf_full_supervision.parquet",
            "sample_lineage.parquet",
        ]:
            pd.DataFrame({"sample_id": ["s1"]}).to_parquet(data_dir / fname)

        result = run_admission_gate(tmp_path, tmp_path)

        assert isinstance(result, AdmissionGateResult)
        assert len(result.checks) == 13
        assert result.verdict in {v.value for v in GateVerdict}
