from __future__ import annotations

from sewerrtc.data import round0_prompt2 as p2


def test_structural_dryrun_without_runtime_execution_cannot_pass(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p2, "ROUND0_DIR", tmp_path)
    p2.write_csv(tmp_path / "round0_hydraulic_dryrun_manifest.csv", [{"candidate_id": "c1", "runtime_executed": "false"}])
    p2.write_json(tmp_path / "round0_hydraulic_dryrun_report.json", {"status": "blocked", "runtime_executed": False})

    code, outputs = p2.evaluate_round0_hydraulic_dryrun_gate()

    assert code == 3
    assert p2.read_json(outputs["gate"])["status"] == "blocked"


def test_dryrun_gate_requires_at_least_12_real_swmm_candidates(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p2, "ROUND0_DIR", tmp_path)
    p2.write_csv(
        tmp_path / "round0_hydraulic_dryrun_manifest.csv",
        [
            {"candidate_id": f"c{i}", "runtime_executed": "true", "same_state_prefix_status": "pass", "swmm_status": "completed", "binary_intermediate_values": 0, "truth_leakage": 0, "recovery_label_status": "complete"}
            for i in range(11)
        ],
    )
    p2.write_json(tmp_path / "round0_hydraulic_dryrun_report.json", {"status": "completed", "runtime_executed": True})

    code, _ = p2.evaluate_round0_hydraulic_dryrun_gate()

    assert code == 5
