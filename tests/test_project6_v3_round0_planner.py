from __future__ import annotations

from pathlib import Path

from sewerrtc.data import round0_prompt2 as p2


def _valid_manifest_row(i: int, *, stratum: str = "1-2", pool: str = "main", anchor: str = "selected_safe_fallback") -> dict[str, str]:
    k = {"1-2": 2, "3-4": 4, "5-8": 8}.get(stratum, 2)
    active = ",".join(f"F{j}" for j in range(k))
    row = {field: "x" for field in p2.ROUND0_MANIFEST_FIELDS}
    row.update(
        {
            "case_id": f"c{i}",
            "candidate_id": f"c{i}",
            "event_id": f"e{i // 60}",
            "checkpoint_id": f"cp{i // 15}",
            "split": "development_fit",
            "anchor_type": anchor,
            "concurrency_stratum": stratum,
            "candidate_k": str(k),
            "k_value": str(k),
            "active_facility_ids": active,
            "active_facility_count_by_step": "[2,2,2,0,0,0,0,0,0,0,0,0]",
            "candidate_pool": pool,
            "binary_legality": "pass",
            "add350_residual_override": "False",
            "noop": "False",
            "duplicate": "False",
            "feasibility": "planned",
            "interaction_group_id": "verified_hydraulic_coupling",
            "interaction_type": "verified hydraulic coupling",
            "interaction_facility_ids": active,
            "transition_type": "",
            "binary_pump_id": "",
        }
    )
    return row


def test_round0_manifest_audit_rejects_less_than_1500_effective_candidates(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p2, "ROUND0_DIR", tmp_path)
    p2.write_csv(tmp_path / "paired_manifest_round0.csv", [{"candidate_id": "c1", "feasibility": "planned", "candidate_pool": "main"}])

    code, _ = p2.audit_round0_manifest(Path("configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml"))

    assert code == 5


def test_plan_round0_reports_insufficient_control_checkpoint_support(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p2, "CONTROL_DIR", tmp_path / "control_checkpoints")
    monkeypatch.setattr(p2, "ROUND0_DIR", tmp_path / "round0")
    p2.write_csv(
        tmp_path / "control_checkpoints" / "control_checkpoint_catalog.csv",
        [
            {
                "checkpoint_id": "cp60",
                "event_id": "e1",
                "storm_family_id": "s1",
                "phase": "rising",
                "split": "action_effect_fit",
                "round0_candidate_eligible": "true",
            }
        ],
    )
    p2.write_json(
        tmp_path / "control_checkpoints" / "control_checkpoint_catalog_report.json",
        {
            "control_aligned_checkpoint_count": 1,
            "unique_fit_events": 1,
            "support_status": "insufficient_support",
        },
    )

    code, outputs = p2.plan_round0_manifest(Path("configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml"), target=1800)
    report = p2.read_json(outputs["report"])

    assert code == 3
    assert report["control_aligned_checkpoint_count"] == 1
    assert "control_aligned_checkpoint_support_insufficient" in report["blocking_reasons"]


def test_plan_round0_records_deterministic_replay_not_hotstart(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p2, "CONTROL_DIR", tmp_path / "control")
    monkeypatch.setattr(p2, "ROUND0_DIR", tmp_path / "round0")
    (tmp_path / "control").mkdir()
    p2.write_csv(
        tmp_path / "control" / "control_checkpoint_catalog.csv",
        [{"checkpoint_id": f"cp{i}", "event_id": f"e{i}", "storm_family_id": "s", "phase": "rising", "split": "action_effect_fit", "round0_candidate_eligible": "true"} for i in range(200)],
    )

    _, outputs = p2.plan_round0_manifest(Path("configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml"), 1500, 0, 0)
    rows = p2.read_csv(outputs["manifest"])

    assert rows
    assert all(r["same_state_method"] == "deterministic_prefix_replay" for r in rows[:10])
    assert all(r["add350_residual_override"] in {"False", "false", False} for r in rows[:10])


def test_round0_manifest_audit_rejects_all_1_2_concurrency(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p2, "ROUND0_DIR", tmp_path)
    rows = [_valid_manifest_row(i, stratum="1-2", anchor="selected_safe_fallback" if i % 2 else "internal") for i in range(1800)]
    p2.write_csv(tmp_path / "paired_manifest_round0.csv", rows)

    code, outputs = p2.audit_round0_manifest(Path("configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml"))
    report = p2.read_json(outputs["report"])

    assert code == 5
    assert any("main_3-4_support" in reason for reason in report["support_failures"])
    assert any("main_5-8_support" in reason for reason in report["support_failures"])


def test_round0_manifest_audit_rejects_blank_candidate_k_and_pool(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p2, "ROUND0_DIR", tmp_path)
    rows = [_valid_manifest_row(i) for i in range(1800)]
    rows[0]["candidate_k"] = ""
    rows[1]["candidate_pool"] = ""
    p2.write_csv(tmp_path / "paired_manifest_round0.csv", rows)

    code, outputs = p2.audit_round0_manifest(Path("configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml"))
    audit = p2.read_csv(outputs["audit"])

    assert code == 5
    assert any(row["check"] == "candidate_k_missing" and int(float(row["count"])) == 1 for row in audit)
    assert any(row["check"] == "candidate_pool_missing" and int(float(row["count"])) == 1 for row in audit)


def test_plan_round0_manifest_includes_all_main_concurrency_strata(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p2, "CONTROL_DIR", tmp_path / "control")
    monkeypatch.setattr(p2, "ROUND0_DIR", tmp_path / "round0")
    (tmp_path / "control").mkdir()
    rows = [
        {
            "checkpoint_id": f"cp{i}",
            "event_id": f"e{i // 5}",
            "storm_family_id": f"s{i // 5}",
            "phase": "rising",
            "policy_id": "internal_rules" if i % 3 == 0 else "executable_passive" if i % 3 == 1 else "no_control",
            "split": "development_fit",
            "round0_candidate_eligible": "true",
        }
        for i in range(160)
    ]
    p2.write_csv(tmp_path / "control" / "control_checkpoint_catalog.csv", rows)
    p2.write_json(tmp_path / "control" / "control_checkpoint_catalog_report.json", {"support_status": "sufficient", "control_aligned_checkpoint_count": 160, "unique_fit_events": 32})

    code, outputs = p2.plan_round0_manifest(Path("configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml"), 1800, 400, 90)
    report = p2.read_json(outputs["report"])

    assert code == 0
    assert report["main_concurrency_counts"]["1-2"] >= 600
    assert report["main_concurrency_counts"]["3-4"] >= 500
    assert report["main_concurrency_counts"]["5-8"] >= 350


def test_audit_round0_manifest_invalidates_stale_approval_lock(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(p2, "ROUND0_DIR", tmp_path)
    p2.write_csv(tmp_path / "paired_manifest_round0.csv", [_valid_manifest_row(0)])
    p2.write_json(
        tmp_path / "round0_manifest_approval_lock.json",
        {
            "status": "pass",
            "round0_manifest_sha256": "old_hash",
            "allowed_for_generation": True,
        },
    )

    p2.audit_round0_manifest(Path("configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml"))
    lock = p2.read_json(tmp_path / "round0_manifest_approval_lock.json")

    assert lock["status"] == "stale"
    assert lock["allowed_for_generation"] is False
    assert lock["failure_reason"] == "manifest_hash_changed"
