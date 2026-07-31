from __future__ import annotations

from sewerrtc.prompt3 import action_effect_mpc as p3


def test_closed_loop_smoke_requires_formal_model_gate(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p3, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(p3, "MPC_DIR", tmp_path / "mpc")

    code, outputs = p3.run_mpc_closed_loop_smoke("configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml", max_events=2)

    assert code == 3
    report = p3.read_json(outputs["report"])
    assert report["runtime_executed"] is False
    assert report["blocking_reasons"] == ["formal_model_gate_not_pass"]


def test_closed_loop_smoke_generates_auditable_first_step_decisions(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p3, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(p3, "MPC_DIR", tmp_path / "mpc")
    monkeypatch.setattr(p3, "ACTION_DATASET_DIR", tmp_path / "dataset")
    p3.MODEL_DIR.mkdir(parents=True)
    p3.MPC_DIR.mkdir(parents=True)
    p3.ACTION_DATASET_DIR.mkdir(parents=True)
    p3.write_json(p3.MODEL_DIR / "prompt3_model_gate.json", {"status": "pass"})
    p3.write_csv(
        p3.ACTION_DATASET_DIR / "action_effect_dataset_manifest.csv",
        [
            {
                "sample_id": "s1",
                "candidate_id": "c1",
                "event_id": "e1",
                "checkpoint_id": "cp1",
                "selected_fallback": "executable_passive",
                "pfv_improved_vs_internal": "true",
                "tfv_noninferior_vs_fallback": "true",
                "peak_noninferior_vs_fallback": "true",
                "k_value": "3",
                "binary_intermediate_values": "0",
                "delta_PFV_vs_internal": "-10",
                "delta_TFV_vs_fallback": "0",
                "delta_peak_vs_fallback": "0",
                "full_recovery_label_status": "complete",
            }
        ],
    )

    code, outputs = p3.run_mpc_closed_loop_smoke("configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml", max_events=1)
    assert code == 0
    report = p3.read_json(outputs["report"])
    assert report["runtime_executed"] is True
    assert report["candidate_execution_semantics"] == "first_10min_then_reoptimize"
    assert report["closed_loop_mode"] == "closed_loop_replay"
    assert report["hydraulic_evidence_source"] == "existing_same_state_candidate_branches"
    decisions = p3.read_csv(p3.MPC_DIR / "mpc_closed_loop_smoke_decisions.csv")
    assert decisions[0]["executed_action_steps"] == "1"
    assert decisions[0]["reoptimized_after_first_step"] == "true"

    gate_code, gate_outputs = p3.evaluate_mpc_closed_loop_smoke_gate("configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml")
    assert gate_code == 3
    gate = p3.read_json(gate_outputs["gate"])
    assert gate["status"] == "blocked"
    assert gate["checks"]["authoritative_swmm_evidence"] is False


def test_authoritative_dev_gate_requires_authoritative_swmm_evidence(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p3, "MODEL_DIR", tmp_path / "models")
    monkeypatch.setattr(p3, "MPC_DIR", tmp_path / "mpc")
    monkeypatch.setattr(p3, "ACTION_DATASET_DIR", tmp_path / "dataset")
    monkeypatch.setattr(p3, "AUTHORITATIVE_DIR", tmp_path / "authoritative")
    monkeypatch.setattr(p3, "EVALUATION_DIR", tmp_path / "evaluation")
    for path in [p3.MODEL_DIR, p3.MPC_DIR, p3.ACTION_DATASET_DIR, p3.AUTHORITATIVE_DIR, p3.EVALUATION_DIR]:
        path.mkdir(parents=True)
    p3.write_json(p3.MODEL_DIR / "prompt3_model_gate.json", {"status": "pass"})
    p3.write_json(p3.MPC_DIR / "mpc_contract_lock.json", {"status": "pass", "hotstart_acceleration_allowed": False, "k_max": 8})
    p3.write_csv(
        p3.ACTION_DATASET_DIR / "action_effect_dataset_manifest.csv",
        [
            {
                "sample_id": "s1",
                "candidate_id": "c1",
                "event_id": "e1",
                "checkpoint_id": "cp1",
                "selected_fallback": "internal_rules",
                "pfv_improved_vs_internal": "true",
                "tfv_noninferior_vs_fallback": "true",
                "peak_noninferior_vs_fallback": "true",
                "k_value": "3",
                "binary_intermediate_values": "0",
                "active_facility_ids": "OR1;ADD301.2",
                "delta_PFV_vs_internal": "-10",
                "delta_TFV_vs_fallback": "0",
                "delta_peak_vs_fallback": "0",
            }
        ],
    )

    code, outputs = p3.run_authoritative_closed_loop_dev("configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml", max_events=1)
    assert code == 0
    report = p3.read_json(outputs["report"])
    assert report["hydraulic_evidence_source"] == "authoritative_swmm"
    assert report["closed_loop_mode"] == "closed_loop_authoritative_swmm"
    assert report["actual_swmm_advance_per_10min"] is True
    assert report["uses_lookup_table_substitute"] is False

    gate_code, gate_outputs = p3.evaluate_authoritative_closed_loop_dev_gate("configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml")
    assert gate_code == 0
    gate = p3.read_json(gate_outputs["gate"])
    assert gate["status"] == "pass"
    assert gate["checks"]["authoritative_swmm_evidence"] is True


def test_paired_closed_loop_dev_requires_same_initial_state_per_event(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p3, "AUTHORITATIVE_DIR", tmp_path / "authoritative")
    p3.AUTHORITATIVE_DIR.mkdir(parents=True)
    p3.write_csv(
        p3.AUTHORITATIVE_DIR / "paired_closed_loop_dev_event_policy_results.csv",
        [
            {"event_id": "e1", "policy_id": policy, "initial_state_sha256": "same", "status": "pass", "engineering_violations": "0"}
            for policy in p3.EVALUATION_POLICIES
        ],
    )
    code, outputs = p3.evaluate_paired_closed_loop_dev_gate("configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml")
    assert code == 0
    assert p3.read_json(outputs["gate"])["status"] == "pass"


def test_evaluation_event_splits_have_36_formal_blind_events(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p3, "EVALUATION_DIR", tmp_path / "evaluation")
    p3.EVALUATION_DIR.mkdir(parents=True)
    code, outputs = p3.build_evaluation_event_splits("configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml")
    assert code == 0
    rows = p3.read_csv(outputs["splits"])
    formal = [row for row in rows if row["split"] == "formal_blind"]
    assert len(formal) == 36
    assert len({row["event_id"] for row in formal}) == 36

    audit_code, audit_outputs = p3.audit_evaluation_event_splits("configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml")
    assert audit_code in {0, 5}
    audit = p3.read_json(audit_outputs["audit"])
    if audit_code == 0:
        assert audit["status"] == "pass"
    else:
        assert audit["status"] == "failed_gate"
        assert any("formal_required_asset_missing" in failure or "missing_rainfall_path" in failure for failure in audit["failures"])
