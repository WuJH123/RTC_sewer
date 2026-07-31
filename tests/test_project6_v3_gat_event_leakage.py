from __future__ import annotations

from pathlib import Path

from sewerrtc.state.gat_event_leakage import compare_training_and_validation_events, near_duplicate_rows


ROOT = Path(__file__).resolve().parents[1]


def test_incomplete_training_manifest_keeps_leakage_incomplete() -> None:
    decision = compare_training_and_validation_events(
        [],
        [{"event_id": "E1", "split": "validation"}],
        training_complete=False,
        validation_complete=True,
    )
    assert decision.status == "incomplete"


def test_exact_event_overlap_is_fail() -> None:
    decision = compare_training_and_validation_events(
        [{"event_id": "E1", "split": "train"}],
        [{"event_id": "E1", "split": "validation"}],
        training_complete=True,
        validation_complete=True,
    )
    assert decision.status == "fail"
    assert any(row["match_type"] == "exact_event_id" for row in decision.rows)


def test_rainfall_hash_overlap_is_fail_even_with_different_event_name() -> None:
    decision = compare_training_and_validation_events(
        [{"event_id": "train_A", "split": "train", "rainfall_series_sha256": "abc"}],
        [{"event_id": "val_B", "split": "validation", "rainfall_series_sha256": "abc"}],
        training_complete=True,
        validation_complete=True,
    )
    assert decision.status == "fail"
    assert any(row["match_type"] == "rainfall_series_or_file_hash" for row in decision.rows)


def test_storm_family_near_duplicate_is_reported() -> None:
    rows = near_duplicate_rows(
        [{"event_id": "train_A", "split": "train", "storm_family_id": "family_1"}],
        [{"event_id": "val_B", "split": "validation", "storm_family_id": "family_1"}],
    )
    assert rows[0]["match_type"] == "storm_family_candidate"


def test_audit_stage_is_registered_without_reusing_gat_inference() -> None:
    runner = (ROOT / "scripts" / "project6_runs" / "RUN_PROJECT6_PFVFIRST_DUALFALLBACK_10MIN_V3.ps1").read_text(encoding="utf-8")
    script = (ROOT / "scripts" / "149_audit_gat_validation_provenance.py").read_text(encoding="utf-8")
    assert "AuditGATValidationProvenance" in runner
    assert "torch" not in script
    assert "run_sr0p15_robustness_audit" not in script

