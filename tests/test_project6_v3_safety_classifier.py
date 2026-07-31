from __future__ import annotations

from sewerrtc.prompt3 import action_effect_mpc as p3


def test_safety_classifier_reports_more_than_accuracy(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p3, "ACTION_DATASET_DIR", tmp_path / "dataset")
    monkeypatch.setattr(p3, "MODEL_DIR", tmp_path / "models")
    p3.write_csv(
        tmp_path / "dataset" / "action_effect_dataset_manifest.csv",
        [{"sample_id": "s1", "pfv_improved_vs_internal": "true", "tfv_noninferior_vs_fallback": "true", "peak_noninferior_vs_fallback": "true"}],
    )

    code, outputs = p3.train_safety_classifier("configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml", smoke=True)

    assert code == 0
    report = p3.read_json(outputs["report"])
    assert report["reports_more_than_accuracy"] is True
    assert "severe_false_safe_count" in report

