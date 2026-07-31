"""V4.2 train-grouped gate — 12-item pass/fail, exit codes, single-failure."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from sewerrtc.v4.v42_gate import (
    GateVerdict,
    _check_01_oof_r2_positive,
    _check_05_pfv_false_safe,
    _check_07_top5_feasible_recall,
    _check_10_fold_level_r2,
    _safe_float,
    evaluate_v42_train_grouped_gate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _good_fold_metrics(n_folds: int = 5) -> list[dict[str, float]]:
    """Generate fold metrics that pass most checks."""
    return [
        {
            "pfv_delta_r2": 0.5, "tfv_delta_r2": 0.4, "peak_delta_r2": 0.3,
            "pfv_delta_mae": 0.5, "tfv_delta_mae": 0.6, "peak_delta_mae": 0.7,
            "pfv_safe_false_safe_rate": 0.10,
            "peak_noninferior_false_safe_rate": 0.10,
            "top5_feasible_recall": 0.90,
            "decision_regret": 0.1, "baseline_regret_hgb": 0.5,
            "uncertainty_error_spearman": 0.3,
        }
        for _ in range(n_folds)
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestSafeFloat:
    def test_valid_float(self):
        assert _safe_float(0.5) == 0.5
        assert _safe_float("0.5") == 0.5

    def test_nan_returns_none(self):
        assert _safe_float(float("nan")) is None

    def test_none_returns_none(self):
        assert _safe_float(None) is None


class TestCheck01OofR2:
    def test_all_positive_pass(self):
        result = _check_01_oof_r2_positive(_good_fold_metrics())
        assert result["passed"] is True

    def test_negative_r2_fails(self):
        metrics = _good_fold_metrics()
        for m in metrics:
            m["pfv_delta_r2"] = -0.1
        result = _check_01_oof_r2_positive(metrics)
        assert result["passed"] is False


class TestCheck05FalseSafe:
    def test_low_rate_passes(self):
        result = _check_05_pfv_false_safe(_good_fold_metrics())
        assert result["passed"] is True

    def test_high_rate_fails(self):
        metrics = _good_fold_metrics()
        for m in metrics:
            m["pfv_safe_false_safe_rate"] = 0.50
        result = _check_05_pfv_false_safe(metrics)
        assert result["passed"] is False


class TestCheck07Recall:
    def test_high_recall_passes(self):
        result = _check_07_top5_feasible_recall(_good_fold_metrics())
        assert result["passed"] is True

    def test_low_recall_fails(self):
        metrics = _good_fold_metrics()
        for m in metrics:
            m["top5_feasible_recall"] = 0.50
        result = _check_07_top5_feasible_recall(metrics)
        assert result["passed"] is False


class TestCheck10FoldR2:
    def test_all_folds_pass(self):
        result = _check_10_fold_level_r2(_good_fold_metrics(5))
        assert result["passed"] is True
        assert result["n_passing_folds"] == 5

    def test_one_fold_fails_still_passes(self):
        metrics = _good_fold_metrics(5)
        metrics[0]["pfv_delta_r2"] = -0.1
        result = _check_10_fold_level_r2(metrics)
        assert result["passed"] is True  # 4/5 still passes

    def test_two_folds_fail(self):
        metrics = _good_fold_metrics(5)
        metrics[0]["pfv_delta_r2"] = -0.1
        metrics[1]["tfv_delta_r2"] = -0.1
        result = _check_10_fold_level_r2(metrics)
        assert result["passed"] is False  # only 3/5


class TestGateVerdict:
    def test_pass_verdict(self):
        v = GateVerdict(verdict="PASS", checks_passed=12)
        assert v.to_dict()["verdict"] == "PASS"

    def test_scientific_fail(self):
        v = GateVerdict(verdict="SCIENTIFIC_FAIL", checks_passed=8)
        d = v.to_dict()
        assert d["checks_passed"] == 8


class TestEvaluateGate:
    def test_missing_dir_returns_data_contract_fail(self, tmp_path: Path):
        result = evaluate_v42_train_grouped_gate(tmp_path / "nonexistent")
        assert result.verdict == "DATA_CONTRACT_FAIL"

    def test_valid_cv_results(self, tmp_path: Path):
        cv_data = {
            "per_seed": [{
                "folds": [
                    {"final_metrics": fm} for fm in _good_fold_metrics(5)
                ]
            }],
            "per_event_metrics": {
                f"evt_{i}": {"mean_r2": 0.4} for i in range(5)
            },
        }
        (tmp_path / "training_history.json").write_text(
            json.dumps(cv_data), encoding="utf-8"
        )
        result = evaluate_v42_train_grouped_gate(tmp_path)
        assert result.verdict in ("PASS", "SCIENTIFIC_FAIL")
        assert result.checks_total == 12
