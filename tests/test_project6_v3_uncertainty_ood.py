from __future__ import annotations

from sewerrtc.prompt3 import action_effect_mpc as p3


def test_uncertainty_ood_gates_require_source_reports(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p3, "MODEL_DIR", tmp_path)

    code, _ = p3.evaluate_uncertainty_gate("configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml", smoke=True)
    assert code == 3

    p3.write_json(tmp_path / "uncertainty_smoke_calibration_report.json", {"status": "pass"})
    code, _ = p3.evaluate_uncertainty_gate("configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml", smoke=True)
    assert code == 0


def test_ood_report_contains_support_fields(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p3, "ACTION_DATASET_DIR", tmp_path / "dataset")
    monkeypatch.setattr(p3, "MODEL_DIR", tmp_path / "models")
    p3.write_csv(tmp_path / "dataset" / "action_effect_dataset_manifest.csv", [{"sample_id": "s1"}])

    code, outputs = p3.train_ood_model("configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml", smoke=True)

    assert code == 0
    report = p3.read_json(outputs["report"])
    assert "support_count" in report["features"]
    assert report["high_ood_candidate_eligible"] is False

