from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROBUSTNESS = ROOT / "sewerrtc" / "state" / "gat_robustness.py"


def test_sensor_failure_contract_contains_all_required_scenarios() -> None:
    text = ROBUSTNESS.read_text(encoding="utf-8")
    for scenario in [
        "baseline",
        "drop_single_sensor",
        "drop_random_5pct",
        "drop_random_10pct",
        "drop_priority_related_sensor",
        "drop_sentinel_related_sensor",
        "regional_sensor_outage",
        "stale_10min",
        "stale_20min",
        "positive_bias",
        "negative_bias",
    ]:
        assert scenario in text


def test_sensor_failure_completion_matrix_has_required_columns() -> None:
    text = ROBUSTNESS.read_text(encoding="utf-8")
    for column in [
        "scenario_id",
        "seed",
        "requested_samples",
        "completed_samples",
        "unique_samples",
        "duplicate_samples",
        "affected_sensor_count",
        "metrics_complete",
    ]:
        assert column in text


def test_sensor_failure_performance_gate_stays_uncalibrated() -> None:
    text = ROBUSTNESS.read_text(encoding="utf-8")
    assert "sensor_failure_execution_complete" in text
    assert "sensor_failure_performance_gate" in text
    assert "uncalibrated" in text
