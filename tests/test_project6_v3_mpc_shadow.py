from __future__ import annotations

from sewerrtc.prompt3 import action_effect_mpc as p3


def test_shadow_gate_fails_if_candidate_was_executed(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p3, "SHADOW_DIR", tmp_path)
    p3.write_csv(tmp_path / "mpc_shadow_smoke_audit.csv", [{"candidate_executed": "true", "truth_leakage": "0", "action_legality": "pass"}])

    code, outputs = p3.evaluate_mpc_shadow_gate("configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml")

    assert code == 5
    assert p3.read_json(outputs["gate"])["status"] == "failed_gate"


def test_shadow_gate_requires_two_events(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p3, "SHADOW_DIR", tmp_path)
    p3.write_csv(
        tmp_path / "mpc_shadow_smoke_audit.csv",
        [
            {
                "event_id": "E1",
                "candidate_executed": "false",
                "truth_leakage": "0",
                "action_legality": "pass",
            }
        ],
    )
    p3.write_json(tmp_path / "mpc_shadow_smoke_report.json", {"event_count": 1})

    code, outputs = p3.evaluate_mpc_shadow_gate("configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml")

    gate = p3.read_json(outputs["gate"])
    assert code == 5
    assert gate["status"] == "failed_gate"
    assert gate["minimum_event_count"] == 2
