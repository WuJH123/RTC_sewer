from __future__ import annotations

from sewerrtc.prompt3 import action_effect_mpc as p3


def test_dataset_gate_blocks_under_1500_formal_samples(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p3, "ACTION_DATASET_DIR", tmp_path)
    p3.write_csv(tmp_path / "action_effect_dataset_manifest.csv", [{"sample_id": "s1", "actual_action_present": "true"}])
    p3.write_json(tmp_path / "action_effect_dataset_audit_report.json", {"status": "pass"})

    code, outputs = p3.evaluate_action_effect_dataset_gate("configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml")

    assert code == 3
    gate = p3.read_json(outputs["gate"])
    assert gate["status"] == "blocked"
    assert gate["minimum_sample_count"] == 1500


def test_dataset_audit_rejects_true_future_and_binary_intermediate(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p3, "ACTION_DATASET_DIR", tmp_path)
    p3.write_csv(
        tmp_path / "action_effect_dataset_manifest.csv",
        [{"sample_id": "s1", "true_future_in_model_input": "true", "actual_action_present": "true", "binary_intermediate_values": "1"}],
    )

    code, outputs = p3.audit_action_effect_dataset("configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml")

    assert code == 5
    report = p3.read_json(outputs["audit"])
    assert "true_future_in_model_input" in report["failures"]
    assert "binary_intermediate_values" in report["failures"]


def test_build_action_effect_dataset_consumes_round0_label_manifest(tmp_path, monkeypatch) -> None:
    round0_dir = tmp_path / "round0_dataset"
    action_dir = tmp_path / "action_effect_dataset"
    monkeypatch.setattr(p3, "ROUND0_DATASET_DIR", round0_dir)
    monkeypatch.setattr(p3, "ACTION_DATASET_DIR", action_dir)
    round0_dir.mkdir()
    action_dir.mkdir()
    p3.write_json(round0_dir / "round0_data_gate.json", {"status": "pass"})
    p3.write_csv(
        round0_dir / "round0_label_manifest.csv",
        [
            {
                "sample_id": "s1",
                "candidate_id": "s1",
                "event_id": "e1",
                "checkpoint_id": "cp1",
                "delta_PFV_vs_internal": "-1.0",
                "delta_TFV_vs_fallback": "2.0",
                "delta_peak_vs_fallback": "0.1",
                "actual_action_present": "true",
                "true_future_in_model_input": "false",
                "binary_intermediate_values": "0",
                "add350_residual_override": "false",
            }
        ],
    )

    code, outputs = p3.build_action_effect_dataset("configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml", "round0")

    assert code == 0
    report = p3.read_json(outputs["report"])
    rows = p3.read_csv(outputs["manifest"])
    assert report["sample_count"] == 1
    assert rows[0]["delta_PFV_vs_internal"] == "-1.0"
