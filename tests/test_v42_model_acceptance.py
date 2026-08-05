from __future__ import annotations

import json
from pathlib import Path

from sewerrtc.v4.v42_model_acceptance import audit_model_acceptance


REQUIRED = [
    "step1_unobserved_depth",
    "step1_priority_depth",
    "step1_wet_or_high_depth",
    "step2_branch_depth",
    "step2_flooding_rate",
    "step2_pfv_budget_metric",
    "step2_tfv_delta",
    "step2_storage_volume",
    "step2_managed_facility_flow",
]


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _base(paper: Path) -> dict:
    _write(paper / "step1_gat/evidence.json", {"gat_model_sha256": "gat-123"})
    _write(
        paper / "step2_surrogate/evidence.json",
        {"surrogate_model_sha256": "step2-456"},
    )
    return {
        "status": "pass",
        "model_accuracy_acceptance_pass": True,
        "quantitative_swmm_comparison_performed": True,
        "event_balanced_metrics_reported": True,
        "thresholds_frozen_before_policy_lock": True,
        "uses_locked_or_final_for_threshold_tuning": False,
        "evaluation_role": "calibration",
        "rainfall_group_isolated": True,
        "training_rainfall_overlap_count": 0,
        "acceptance_contract_sha256": "contract-789",
        "gat_model_sha256": "gat-123",
        "surrogate_model_sha256": "step2-456",
        "accepted_metric_families": REQUIRED,
    }


def test_model_acceptance_passes_only_for_frozen_calibration_evidence(tmp_path: Path) -> None:
    paper = tmp_path / "v42_paper"
    payload = _base(paper)
    _write(paper / "model_acceptance/evidence.json", payload)
    result = audit_model_acceptance(paper)
    assert result["status"] == "pass"
    assert result["model_accuracy_acceptance_pass"] is True
    assert result["evidence_sha256"]


def test_model_acceptance_rejects_hash_mismatch_or_heldout_tuning(tmp_path: Path) -> None:
    paper = tmp_path / "v42_paper"
    payload = _base(paper)
    payload["surrogate_model_sha256"] = "wrong"
    payload["uses_locked_or_final_for_threshold_tuning"] = True
    _write(paper / "model_acceptance/evidence.json", payload)
    result = audit_model_acceptance(paper)
    assert result["status"] == "fail"
    assert "model_acceptance_surrogate_hash_mismatch" in result["reasons"]
    assert "locked_or_final_used_for_accuracy_threshold_tuning" in result["reasons"]


def test_model_acceptance_rejects_missing_metric_family(tmp_path: Path) -> None:
    paper = tmp_path / "v42_paper"
    payload = _base(paper)
    payload["accepted_metric_families"] = REQUIRED[:-1]
    _write(paper / "model_acceptance/evidence.json", payload)
    result = audit_model_acceptance(paper)
    assert result["status"] == "fail"
    assert any(
        reason.startswith("model_acceptance_metric_families_missing:")
        for reason in result["reasons"]
    )
