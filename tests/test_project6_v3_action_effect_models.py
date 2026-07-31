from __future__ import annotations

import numpy as np

from sewerrtc.prompt3 import action_effect_mpc as p3


def test_formal_ensemble_requires_at_least_five_members(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p3, "ACTION_DATASET_DIR", tmp_path / "dataset")
    monkeypatch.setattr(p3, "MODEL_DIR", tmp_path / "models")
    rows = []
    for idx in range(1500):
        rows.append(
            {
                "sample_id": f"s{idx}",
                "k_value": "1",
                "concurrency": "1",
                "delta_PFV_vs_internal": "0",
                "delta_TFV_vs_fallback": "0",
                "delta_peak_vs_fallback": "0",
            }
        )
    p3.write_csv(tmp_path / "dataset" / "action_effect_dataset_manifest.csv", rows)

    code, outputs = p3.train_action_effect_ensemble("configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml", smoke=False, ensemble_size=2)

    assert code == 3
    assert p3.read_json(outputs["report"])["blocking_reasons"] == ["ensemble_size_below_5_for_formal"]


def test_smoke_training_writes_smoke_model_not_formal_lock(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p3, "ACTION_DATASET_DIR", tmp_path / "dataset")
    monkeypatch.setattr(p3, "MODEL_DIR", tmp_path / "models")
    rows = []
    for idx in range(12):
        rows.append(
            {
                "sample_id": f"s{idx}",
                "k_value": "1",
                "concurrency": "1",
                "action_direction": "increase",
                "action_magnitude": "small",
                "delta_PFV_vs_internal": str(idx),
                "delta_TFV_vs_fallback": "0",
                "delta_peak_vs_fallback": "0",
            }
        )
    p3.write_csv(tmp_path / "dataset" / "action_effect_dataset_manifest.csv", rows)

    code, outputs = p3.train_action_effect_ensemble("configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml", smoke=True, max_samples=128, ensemble_size=2)

    assert code == 0
    assert outputs["model"].name == "action_effect_ensemble_smoke.npz"
    assert not (tmp_path / "models" / "action_effect_model_lock.json").exists()
    assert np.load(outputs["model"])["weights"].shape[0] == 2


def test_build_action_effect_dataset_combines_requested_rounds(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p3, "ROUND0_DATASET_DIR", tmp_path / "round0_dataset")
    monkeypatch.setattr(p3, "ACTION_DATASET_DIR", tmp_path / "action_effect_dataset")
    p3.ROUND0_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    p3.ACTION_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    for round_name in ["round0", "round1", "round2"]:
        p3.write_json(p3.ROUND0_DATASET_DIR / f"{round_name}_data_gate.json", {"status": "pass", "formal_target_met": round_name == "round0"})
        p3.write_csv(
            p3.ROUND0_DATASET_DIR / f"{round_name}_label_manifest.csv",
            [
                {
                    "sample_id": f"{round_name}_s1",
                    "candidate_id": f"{round_name}_c1",
                    "event_id": f"{round_name}_event",
                    "checkpoint_id": f"{round_name}_checkpoint",
                    "delta_PFV_vs_internal": "1.0",
                    "delta_TFV_vs_fallback": "2.0",
                    "delta_peak_vs_fallback": "3.0",
                    "actual_action_present": "true",
                    "true_future_in_model_input": "false",
                    "binary_intermediate_values": "0",
                    "add350_residual_override": "false",
                }
            ],
        )

    code, outputs = p3.build_action_effect_dataset(
        "configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml",
        include_rounds="round0,round1,round2",
    )

    assert code == 0
    rows = p3.read_csv(outputs["manifest"])
    assert len(rows) == 3
    assert {row["source_round"] for row in rows} == {"round0", "round1", "round2"}
    report = p3.read_json(outputs["report"])
    assert report["source_round_counts"] == {"round0": 1, "round1": 1, "round2": 1}
    assert len(report["round_lineage"]) == 6


def test_action_effect_model_gate_rejects_stale_dataset_hash(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p3, "ACTION_DATASET_DIR", tmp_path / "dataset")
    monkeypatch.setattr(p3, "MODEL_DIR", tmp_path / "models")
    p3.ACTION_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    p3.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    p3.write_csv(p3.ACTION_DATASET_DIR / "action_effect_dataset_manifest.csv", [{"sample_id": "current"}])
    p3.write_json(
        p3.MODEL_DIR / "action_effect_ensemble_report.json",
        {"status": "pass", "ensemble_size": 5, "dataset_sha256": "old_dataset_hash"},
    )

    code, outputs = p3.evaluate_action_effect_model_gate("configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml")

    assert code == 6
    gate = p3.read_json(outputs["gate"])
    assert gate["status"] == "contract_mismatch"
    assert gate["dataset_hash_matches"] is False


def test_mpc_contract_rejects_stale_model_gate(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p3, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(p3, "MPC_DIR", tmp_path / "mpc")
    p3.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    p3.MPC_DIR.mkdir(parents=True, exist_ok=True)
    p3.write_json(p3.MODEL_DIR / "action_effect_ensemble_report.json", {"status": "pass", "ensemble_size": 5})
    p3.write_json(p3.MODEL_DIR / "action_effect_model_gate.json", {"status": "contract_mismatch"})

    code, outputs = p3.build_pfvfirst_dualfallback_mpc("configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml")

    assert code == 6
    assert p3.read_json(outputs["contract"])["status"] == "contract_mismatch"


def test_prompt3_model_gate_requires_auxiliary_reports_for_current_model(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p3, "MODEL_DIR", tmp_path / "models")
    p3.MODEL_DIR.mkdir(parents=True, exist_ok=True)
    p3.write_json(p3.MODEL_DIR / "action_effect_ensemble_report.json", {"status": "pass", "model_sha256": "new_model", "dataset_sha256": "new_dataset"})
    p3.write_json(p3.MODEL_DIR / "action_effect_model_gate.json", {"status": "pass", "dataset_hash_matches": True})
    for name in ["uncertainty_gate.json", "ood_gate.json", "safety_classifier_gate.json"]:
        p3.write_json(
            p3.MODEL_DIR / name,
            {
                "status": "pass",
                "source_binding": {"model_sha256": "old_model", "dataset_sha256": "old_dataset"},
            },
        )
    p3.write_json(p3.MODEL_DIR / "fallback_selector_report.json", {"status": "pass", "model_sha256": "old_model", "dataset_sha256": "old_dataset"})

    code, outputs = p3.evaluate_prompt3_model_gate("configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml")

    assert code == 3
    gate = p3.read_json(outputs["gate"])
    assert gate["status"] == "blocked"
    assert gate["checks"]["action_effect_model"] is True
    assert gate["checks"]["uncertainty"] is False
    assert gate["checks"]["fallback_selector"] is False
