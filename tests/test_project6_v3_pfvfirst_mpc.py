from __future__ import annotations

from sewerrtc.prompt3 import action_effect_mpc as p3


def test_mpc_contract_keeps_pfv_first_k8_and_hotstart_disabled(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p3, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(p3, "MPC_DIR", tmp_path / "mpc")
    p3.write_json(tmp_path / "models" / "action_effect_ensemble_smoke_report.json", {"status": "pass"})
    p3.write_json(tmp_path / "models" / "action_effect_model_smoke_gate.json", {"status": "pass"})

    code, outputs = p3.build_pfvfirst_dualfallback_mpc("configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml", smoke=True)

    assert code == 0
    contract = p3.read_json(outputs["contract"])
    assert contract["k_max"] == 8
    assert contract["hotstart_acceleration_allowed"] is False
    assert contract["variable_speed_pump"] == "add350.1"
    assert set(contract["binary_pumps"]) == {"ADD301.2", "ADD301.3"}


def test_mpc_unit_smoke_rejects_binary_intermediate_or_add350_binary(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p3, "MPC_DIR", tmp_path / "mpc")
    p3.write_csv(tmp_path / "mpc" / "mpc_unit_smoke_audit.csv", [{"status": "pass", "binary_intermediate_values": "0", "add350_binary_logic_used": "false"}])

    code, _ = p3.evaluate_mpc_unit_gate("configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml")
    assert code == 0

    p3.write_csv(tmp_path / "mpc" / "mpc_unit_smoke_audit.csv", [{"status": "pass", "binary_intermediate_values": "1", "add350_binary_logic_used": "false"}])
    code, _ = p3.evaluate_mpc_unit_gate("configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml")
    assert code == 5
