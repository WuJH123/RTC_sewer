from __future__ import annotations

from sewerrtc.prompt3 import action_effect_mpc as p3


def test_fallback_selector_does_not_use_true_future(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p3, "ACTION_DATASET_DIR", tmp_path / "dataset")
    monkeypatch.setattr(p3, "MODEL_DIR", tmp_path / "models")
    p3.write_csv(tmp_path / "dataset" / "action_effect_dataset_manifest.csv", [{"sample_id": "s1"}])

    code, outputs = p3.train_fallback_selector("configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml", smoke=True)

    assert code == 0
    report = p3.read_json(outputs["report"])
    assert report["fallback_frozen_before_candidate"] is True
    assert report["uses_true_future"] is False

