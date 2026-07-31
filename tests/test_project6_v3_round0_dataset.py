from __future__ import annotations

from sewerrtc.data import round0_prompt2 as p2


def _write_detail(path, *, offset: float = 0.0) -> None:
    p2.write_csv(
        path,
        [
            {"elapsed_min": "0", "flood:P1": str(1.0 + offset), "flood:N1": str(2.0 + offset), "a:ADD301.2": "0"},
            {"elapsed_min": "30", "flood:P1": str(2.0 + offset), "flood:N1": str(3.0 + offset), "a:ADD301.2": "0"},
            {"elapsed_min": "60", "flood:P1": str(3.0 + offset), "flood:N1": str(4.0 + offset), "a:ADD301.2": "1"},
            {"elapsed_min": "90", "flood:P1": str(4.0 + offset), "flood:N1": str(5.0 + offset), "a:ADD301.2": "1"},
            {"elapsed_min": "120", "flood:P1": str(5.0 + offset), "flood:N1": str(6.0 + offset), "a:ADD301.2": "1"},
        ],
    )


def _round0_fixture(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p2, "ROUND0_DIR", tmp_path / "round0")
    monkeypatch.setattr(p2, "DATASET_DIR", tmp_path / "round0_dataset")
    monkeypatch.setattr(p2, "PRIORITY_NODES", tmp_path / "priority.txt")
    p2.ROUND0_DIR.mkdir(parents=True, exist_ok=True)
    p2.DATASET_DIR.mkdir(parents=True, exist_ok=True)
    p2.PRIORITY_NODES.write_text("P1\n", encoding="utf-8")
    details = tmp_path / "details"
    details.mkdir()
    for branch, offset in {
        "no_control": 0.0,
        "internal_rules": 1.0,
        "executable_passive": 2.0,
        "candidate": 3.0,
        "candidate_then_passive": 4.0,
        "candidate_then_internal": 5.0,
    }.items():
        _write_detail(details / f"{branch}.csv", offset=offset)

    candidates = [
        {
            "candidate_id": "c_main",
            "event_id": "e1",
            "checkpoint_id": "e1_internal_rules_t0000",
            "candidate_pool": "main",
            "anchor_type": "internal",
            "selected_fallback": "executable_passive",
            "phase": "rising",
            "runtime_executed": "true",
            "same_state_prefix_status": "pass",
            "swmm_status": "completed",
            "truth_leakage": "0",
            "binary_intermediate_values": "0",
            "engineering_violations": "",
            "recovery_label_status": "complete",
            "k_value": "2",
            "concurrency": "2",
            "action_directions": "increase",
            "action_magnitude": "medium",
            "binary_legality": "pass",
            "add350_residual_override": "False",
        },
        {
            "candidate_id": "c_reserve",
            "event_id": "e1",
            "checkpoint_id": "e1_internal_rules_t0000",
            "candidate_pool": "reserve",
            "anchor_type": "internal",
            "selected_fallback": "executable_passive",
            "runtime_executed": "true",
            "same_state_prefix_status": "pass",
            "swmm_status": "completed",
            "truth_leakage": "0",
            "binary_intermediate_values": "0",
            "engineering_violations": "",
            "recovery_label_status": "complete",
        },
    ]
    p2.write_csv(p2.ROUND0_DIR / "round0_generation_manifest.csv", candidates)
    branch_rows = []
    kpi_rows = []
    for cid in ["c_main", "c_reserve"]:
        for branch in ["no_control", "internal_rules", "executable_passive", "candidate", "candidate_then_passive", "candidate_then_internal"]:
            detail = details / f"{branch}.csv"
            branch_rows.append(
                {
                    "candidate_id": cid,
                    "branch_id": branch,
                    "branch_type": "reference_existing" if branch in {"no_control", "internal_rules", "executable_passive"} else "candidate_runtime",
                    "runtime_executed": "true",
                    "swmm_status": "completed",
                    "detail_file": str(detail),
                    "detail_sha256": p2.existing_hash(detail),
                    "same_state_prefix_status": "" if branch in {"no_control", "internal_rules", "executable_passive"} else "pass",
                    "recovery_label_status": "" if branch in {"no_control", "internal_rules", "executable_passive"} else "complete",
                    "recovery_status": "" if branch in {"no_control", "internal_rules", "executable_passive"} else "recovered",
                }
            )
            kpi_rows.append(
                {
                    "candidate_id": cid,
                    "branch_id": branch,
                    "TFV": "100",
                    "PFV": "10",
                    "peak_TFV_rate": "4",
                    "flood_duration_min": "30",
                    "priority_flood_duration_min": "30",
                    "detail_file": str(detail),
                }
            )
    p2.write_csv(p2.ROUND0_DIR / "round0_branch_audit.csv", branch_rows)
    p2.write_csv(p2.ROUND0_DIR / "round0_kpi_audit.csv", kpi_rows)


def test_generate_round0_batch_refresh_existing_only_reuses_completed_outputs(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p2, "ROUND0_DIR", tmp_path)
    p2.write_json(tmp_path / "round0_manifest_approval_lock.json", {"status": "pass"})
    p2.write_csv(
        tmp_path / "round0_generation_manifest.csv",
        [
            {
                "candidate_id": "c1",
                "runtime_executed": "true",
                "same_state_prefix_status": "pass",
                "swmm_status": "completed",
                "truth_leakage": "0",
            }
        ],
    )
    for name in ["round0_branch_audit.csv", "round0_action_audit.csv", "round0_kpi_audit.csv", "round0_fallback_audit.csv", "round0_failures.csv"]:
        p2.write_csv(tmp_path / name, [])
    p2.write_json(tmp_path / "round0_batch_report.json", {"status": "completed", "runtime_executed": True})

    code, outputs = p2.generate_round0_batch("configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml", refresh_existing_only=True)
    report = p2.read_json(outputs["report"])

    assert code == 0
    assert report["refresh_existing_only"] is True
    assert report["valid_candidate_count"] == 1


def test_failed_samples_do_not_make_dataset_gate_pass(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p2, "DATASET_DIR", tmp_path)
    p2.write_csv(tmp_path / "round0_dataset_manifest.csv", [{"sample_id": "s1", "status": "failed_runtime"}])

    code, outputs = p2.evaluate_round0_data_gate()

    assert code == 3
    assert p2.read_json(outputs["gate"])["status"] == "blocked"


def test_action_effect_readiness_requires_round0_data_gate(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p2, "DATASET_DIR", tmp_path)
    p2.write_json(tmp_path / "round0_data_gate.json", {"status": "blocked"})

    code, _ = p2.evaluate_action_effect_training_readiness()

    assert code == 3


def test_build_round0_dataset_materializes_labels_and_excludes_non_main(tmp_path, monkeypatch) -> None:
    _round0_fixture(tmp_path, monkeypatch)

    code, outputs = p2.build_round0_dataset("configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml", "round0")

    assert code == 0
    report = p2.read_json(outputs["report"])
    assert report["valid_sample_count"] == 1
    assert report["label_rows"] == 1
    rows = p2.read_csv(p2.DATASET_DIR / "round0_label_manifest.csv")
    assert len(rows) == 1
    row = rows[0]
    assert row["sample_id"] == "c_main"
    assert row["anchor_branch_id"] == "internal_rules"
    assert row["selected_fallback_branch_id"] == "executable_passive"
    assert row["h30_label_status"] == "available"
    assert row["full_recovery_label_status"] == "complete"
    assert "delta_PFV_Candidate-anchor_H30" in row


def test_build_round0_dataset_resume_reuses_current_completed_dataset(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p2, "ROUND0_DIR", tmp_path / "round0")
    monkeypatch.setattr(p2, "DATASET_DIR", tmp_path / "dataset")
    p2.ROUND0_DIR.mkdir(parents=True, exist_ok=True)
    p2.DATASET_DIR.mkdir(parents=True, exist_ok=True)
    source = p2.write_csv(
        p2.ROUND0_DIR / "round0_generation_manifest.csv",
        [{"candidate_id": "c1", "runtime_executed": "true", "same_state_prefix_status": "pass", "swmm_status": "completed"}],
    )
    p2.write_csv(p2.DATASET_DIR / "round0_dataset_manifest.csv", [{"candidate_id": "c1", "sample_id": "c1", "status": "completed"}])
    p2.write_csv(p2.DATASET_DIR / "round0_label_manifest.csv", [{"candidate_id": "c1", "sample_id": "c1"}])
    p2.write_csv(p2.DATASET_DIR / "round0_absolute_labels.csv", [{"candidate_id": "c1", "sample_id": "c1"}])
    p2.write_csv(p2.DATASET_DIR / "round0_delta_labels.csv", [{"candidate_id": "c1", "sample_id": "c1"}])
    p2.write_csv(p2.DATASET_DIR / "round0_label_failures.csv", [{"candidate_id": "c2", "failure_reason": "non_main_pool"}])
    p2.write_json(
        p2.DATASET_DIR / "round0_dataset_report.json",
        {
            "status": "completed",
            "valid_sample_count": 1,
            "source_generation_manifest": str(source),
            "source_generation_manifest_sha256": p2.existing_hash(source),
            "config_hash": p2.config_hash("configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml"),
            "label_rows": 1,
        },
    )

    def fail_if_rebuilt(_runtime_valid):
        raise AssertionError("resume should reuse the completed dataset")

    monkeypatch.setattr(p2, "_build_round0_labels", fail_if_rebuilt)

    code, outputs = p2.build_round0_dataset("configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml", "round0", resume=True)
    report = p2.read_json(outputs["report"])

    assert code == 0
    assert report["resume_reused_existing_dataset"] is True
    assert report["completion_marker_allowed"] is True


def test_round0_dataset_audit_uses_label_rows_not_reference_same_state_na(tmp_path, monkeypatch) -> None:
    _round0_fixture(tmp_path, monkeypatch)
    p2.build_round0_dataset("configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml", "round0")

    code, outputs = p2.audit_round0_dataset("round0")

    assert code == 3
    report = p2.read_json(outputs["audit"])
    assert report["failure_counts"]["below_minimum_valid_effective_candidate_count"] == 1
    assert report["failure_counts"].get("same_state_failure", 0) == 0
    assert report["label_rows"] == 1


def test_round1_dataset_gate_allows_real_smoke_batch_without_formal_target(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p2, "DATASET_DIR", tmp_path)
    rows = [{"candidate_id": f"r1_{idx}", "sample_id": f"r1_{idx}", "status": "completed"} for idx in range(12)]
    p2.write_csv(tmp_path / "round1_dataset_manifest.csv", rows)
    p2.write_csv(tmp_path / "round1_label_manifest.csv", rows)
    p2.write_csv(tmp_path / "round1_label_failures.csv", [])

    code, outputs = p2.audit_round0_dataset("round1")
    assert code == 0
    audit = p2.read_json(outputs["audit"])
    assert audit["minimum_valid_effective_candidate_count"] == 12
    assert audit["formal_target_met"] is False

    code, outputs = p2.evaluate_round_data_gate("round1")
    assert code == 0
    gate = p2.read_json(outputs["gate"])
    assert gate["status"] == "pass"
    assert gate["smoke_target_met"] is True
    assert gate["formal_target_met"] is False


def test_round_batch_resume_carries_existing_completed_rows(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p2, "ROUND0_DIR", tmp_path)
    p2.write_json(tmp_path / "round1_manifest_approval_lock.json", {"status": "pass"})
    p2.write_csv(
        tmp_path / "paired_manifest_round1.csv",
        [
            {"candidate_id": "already_done", "candidate_pool": "main"},
            {"candidate_id": "new_one", "candidate_pool": "main"},
        ],
    )
    p2.write_csv(
        tmp_path / "round1_generation_manifest.csv",
        [{"candidate_id": "already_done", "runtime_executed": "true", "same_state_prefix_status": "pass", "swmm_status": "completed"}],
    )

    captured = {}

    def fake_subset(_config, *, selected_candidates, prefix, workers, resume, minimum_pass_count):
        captured["ids"] = [row["candidate_id"] for row in selected_candidates]
        captured["minimum_pass_count"] = minimum_pass_count
        captured["prefix"] = prefix
        return 0, {"report": tmp_path / "round1_report.json"}

    monkeypatch.setattr(p2, "_run_round0_generation_subset", fake_subset)

    code, _ = p2.generate_round_batch("configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml", "round1", batch_size=1, workers=1, resume=True)

    assert code == 0
    assert captured["ids"] == ["already_done", "new_one"]
    assert captured["prefix"] == "round1"
    assert captured["minimum_pass_count"] == 2
