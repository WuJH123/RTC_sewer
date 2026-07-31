from __future__ import annotations

from pathlib import Path

from sewerrtc.data import round0_prompt2 as p2


def test_baseline_expansion_audit_requires_generation_manifest(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p2, "PROMPT2_BASELINE_DIR", tmp_path / "prompt2_baseline")
    p2.write_csv(tmp_path / "prompt2_baseline" / "baseline_trajectory_plan.csv", [{"event_id": "e1", "policy_id": "internal_rules"}])

    code, outputs = p2.audit_prompt2_baseline_expansion(Path("configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml"))
    report = p2.read_json(outputs["audit"])

    assert code == 3
    assert "generation_manifest_missing" in report["blocking_reasons"]


def test_baseline_expansion_audit_requires_30_events_and_90_trajectories(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p2, "PROMPT2_BASELINE_DIR", tmp_path / "prompt2_baseline")
    rows = [{"trajectory_id": f"e{i}_internal_rules", "event_id": f"e{i}", "policy_id": "internal_rules", "status": "completed"} for i in range(2)]
    p2.write_csv(tmp_path / "prompt2_baseline" / "baseline_trajectory_manifest.csv", rows)

    code, outputs = p2.audit_prompt2_baseline_expansion(Path("configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml"))
    report = p2.read_json(outputs["audit"])

    assert code == 3
    assert "completed_events_below_30" in report["blocking_reasons"]
    assert "completed_trajectories_below_90" in report["blocking_reasons"]
