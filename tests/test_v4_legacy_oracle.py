"""Gate P3 legacy oracle compatibility audit: pure functions and wiring.

Guards the 13-dimension fail-closed evaluation, the seed-only replay plan,
and the pipeline wiring of ``AuditLegacyOracleCompatibility``.  No SWMM run
is involved anywhere here.
"""

import json
from pathlib import Path

import pandas as pd

from sewerrtc.v4.legacy_oracle import (
    COMPATIBILITY_DIMENSIONS,
    KIND_GATE0_PROOF,
    KIND_ORACLE_CASE,
    KIND_V4_MANIFEST,
    SEED_FAMILY,
    audit_legacy_oracle_compatibility,
    build_replay_seed_plan,
    evaluate_legacy_frame,
)
from sewerrtc.v4.pipeline import (
    ALL_STAGES,
    OUTPUT_DIRECTORIES,
    PREREQUISITES,
    STAGE_ARTIFACTS,
    build_registry,
)
from sewerrtc.v4.runtime import RuntimeOptions

NETWORK_SHA = "n" * 64
RAIN_SHA = "r" * 64

CONTEXT = {
    "network_sha256": NETWORK_SHA,
    "rainfall_by_event": {"ev1": RAIN_SHA},
    "pilot_state_keys": {("ev1", "ck1")},
    "max_k": 8,
}


def _oracle_case_row(tmp_path: Path, **overrides) -> pd.DataFrame:
    schedule = tmp_path / "schedules" / "case1.csv"
    schedule.parent.mkdir(parents=True, exist_ok=True)
    schedule.write_text("step,facility,setting\n", encoding="utf-8")
    row = {
        "case_id": "case1",
        "event_id": "ev1",
        "constraint_mode": "constrained",
        "schedule_csv": "schedules/case1.csv",
        "schedule_sha256": "s" * 64,
        "status": "success",
        "inp_sha256": "x" * 64,
        "rainfall_sha256": RAIN_SHA,
        "max_simultaneous_deviations": 4,
        "runtime_executed": True,
        "authoritative_swmm": True,
    }
    row.update(overrides)
    return pd.DataFrame([row])


def test_oracle_case_row_reports_all_13_dimensions(tmp_path: Path) -> None:
    frame = _oracle_case_row(tmp_path)
    evaluated = evaluate_legacy_frame(
        frame,
        source="oracle_pareto_20ev",
        kind=KIND_ORACLE_CASE,
        context=CONTEXT,
        source_dir=tmp_path,
        legacy_root=tmp_path,
    )
    assert len(evaluated) == 1
    row = evaluated.iloc[0]
    for name in COMPATIBILITY_DIMENSIONS:
        assert f"dim_{name}" in evaluated.columns
        assert str(row[f"dim_{name}_evidence"])
    # Episode-level legacy evidence can never be fully compatible: no
    # checkpoint identity, no H120 window, no frozen-margin labels.
    assert not row["fully_compatible"]
    assert "checkpoint_identity" in row["failed_dimensions"]
    assert "network_sha256" in row["failed_dimensions"]
    # But a constrained, successful, authoritative run is seed-usable.
    assert row["seed_usable"]
    assert row["allowed_use"] == "search_seed_only"


def test_matching_network_and_rainfall_dimensions_pass(tmp_path: Path) -> None:
    frame = _oracle_case_row(tmp_path, inp_sha256=NETWORK_SHA)
    evaluated = evaluate_legacy_frame(
        frame,
        source="oracle_pareto_20ev",
        kind=KIND_ORACLE_CASE,
        context=CONTEXT,
        source_dir=tmp_path,
        legacy_root=tmp_path,
    )
    row = evaluated.iloc[0]
    assert row["dim_network_sha256"]
    assert row["dim_event_rainfall_sha256"]
    assert row["dim_k_at_most_8"]
    assert row["dim_provenance"]
    # Still not fully compatible: checkpoint/H120/margins remain unproven.
    assert not row["fully_compatible"]


def test_relaxed_mode_fails_provenance_but_stays_seed_usable(
    tmp_path: Path,
) -> None:
    frame = _oracle_case_row(tmp_path, constraint_mode="relaxed")
    evaluated = evaluate_legacy_frame(
        frame,
        source="constraint_ablation",
        kind=KIND_ORACLE_CASE,
        context=CONTEXT,
        source_dir=tmp_path,
        legacy_root=tmp_path,
    )
    row = evaluated.iloc[0]
    assert not row["dim_provenance"]
    # Seeds are re-projected under the frozen constraints, so a real relaxed
    # run may still seed the search; it must never become a label.
    assert row["seed_usable"]


def test_failed_or_missing_schedule_is_not_seed_usable(tmp_path: Path) -> None:
    failed = _oracle_case_row(tmp_path, status="failed")
    missing = _oracle_case_row(tmp_path, schedule_csv="does/not/exist.csv")
    for frame in (failed, missing):
        evaluated = evaluate_legacy_frame(
            frame,
            source="oracle_pareto_20ev",
            kind=KIND_ORACLE_CASE,
            context=CONTEXT,
            source_dir=tmp_path,
            legacy_root=tmp_path,
        )
        assert not evaluated.iloc[0]["seed_usable"]


def test_gate0_and_v4_rows_are_never_compatible_or_seed_usable() -> None:
    gate0 = pd.DataFrame(
        [
            {
                "event_id": "ev1",
                "constrained_strict_feasible": True,
                "readback_ok": True,
                "authoritative_swmm": True,
            }
        ]
    )
    v4 = pd.DataFrame(
        [
            {
                "sample_id": "s1",
                "event_id": "ev1",
                "checkpoint_id": "ck1",
                "k_value": 8,
                "hotstart_used_for_label": False,
                "runtime_executed": True,
                "actual_action_present": True,
                "h120_label_status": "ok",
                "v4_label_contract": "legacy_contract",
            }
        ]
    )
    for kind, frame in ((KIND_GATE0_PROOF, gate0), (KIND_V4_MANIFEST, v4)):
        evaluated = evaluate_legacy_frame(
            frame,
            source="src",
            kind=kind,
            context=CONTEXT,
            source_dir=Path("."),
            legacy_root=Path("."),
        )
        row = evaluated.iloc[0]
        assert not row["fully_compatible"]
        assert not row["seed_usable"]
        assert row["dim_reference_semantics"] is not True or kind is None
    # v4 manifest positives that ARE recorded must be honored as such.
    v4_row = evaluate_legacy_frame(
        v4,
        source="src",
        kind=KIND_V4_MANIFEST,
        context=CONTEXT,
        source_dir=Path("."),
        legacy_root=Path("."),
    ).iloc[0]
    assert v4_row["dim_no_hotstart"]
    assert v4_row["dim_k_at_most_8"]
    assert v4_row["dim_h120_window"]
    assert not v4_row["dim_reference_semantics"]


def test_replay_seed_plan_is_seed_only_and_projection_required(
    tmp_path: Path,
) -> None:
    frame = _oracle_case_row(tmp_path)
    evaluated = evaluate_legacy_frame(
        frame,
        source="oracle_pareto_20ev",
        kind=KIND_ORACLE_CASE,
        context=CONTEXT,
        source_dir=tmp_path,
        legacy_root=tmp_path,
    )
    plan = build_replay_seed_plan(evaluated)
    assert len(plan) == 1
    row = plan.iloc[0]
    assert row["candidate_family"] == SEED_FAMILY
    assert row["use"] == "search_seed_only"
    assert bool(row["requires_projection"])
    assert bool(row["label_use_forbidden"])
    assert build_replay_seed_plan(evaluated.iloc[0:0]).empty


def test_full_audit_counts_and_policy(tmp_path: Path) -> None:
    frame = _oracle_case_row(tmp_path)
    result = audit_legacy_oracle_compatibility(
        {"oracle_pareto_20ev": (KIND_ORACLE_CASE, frame, tmp_path)},
        network_sha256=NETWORK_SHA,
        rainfall_by_event={"ev1": RAIN_SHA},
        pilot_state_keys={("ev1", "ck1")},
        legacy_root=tmp_path,
        missing_sources=["missing/source.csv"],
    )
    audit = result["audit"]
    assert audit["status"] == "pass"
    assert audit["evidence_rows"] == 1
    assert audit["fully_compatible_rows"] == 0
    assert audit["incompatible_rows"] == 1
    assert audit["seed_usable_rows"] == 1
    assert audit["missing_sources"] == ["missing/source.csv"]
    assert audit["policy"]["labels_from_incompatible"] == "forbidden"
    assert set(audit["dimension_fail_counts"]) == set(
        COMPATIBILITY_DIMENSIONS
    )
    assert json.dumps(audit)  # payload must be JSON-serializable


def test_stage_wiring_constants() -> None:
    index = ALL_STAGES.index("AuditLegacyOracleCompatibility")
    assert ALL_STAGES[index - 1] == "EvaluatePilotGateV2"
    assert ALL_STAGES[index + 1] == "PlanPilotFeasibilityMap"
    assert STAGE_ARTIFACTS["AuditLegacyOracleCompatibility"] == (
        "pilot_feasibility_p3/legacy_oracle/"
        "legacy_oracle_compatibility_audit.json"
    )
    # Anchors on AuditContracts only: EvaluatePilotGateV2 holds a frozen
    # scientific_fail (exit 5) and must not deadlock the read-only scan.
    assert PREREQUISITES["AuditLegacyOracleCompatibility"] == (
        "AuditContracts",
    )
    assert "pilot_feasibility_p3" in OUTPUT_DIRECTORIES
    # Train1600 entry is untouched by the P3 chain.
    assert PREREQUISITES["PlanTrain1600"] == ("EvaluatePilotGate",)


def test_registry_registers_stage_and_reports_missing_inputs(
    tmp_path: Path,
) -> None:
    registry = build_registry(
        project_root=tmp_path,
        output_root=tmp_path / "out",
        config={},
    )
    assert "AuditLegacyOracleCompatibility" in registry.names
    result = registry.run(
        "AuditLegacyOracleCompatibility",
        RuntimeOptions(stage="AuditLegacyOracleCompatibility", dry_run=True),
    )
    # Gated behind AuditContracts, which has not passed in the tmp tree.
    assert result.exit_code != 0
    assert not result.scope_complete


def test_handler_end_to_end_on_synthetic_tree(tmp_path: Path) -> None:
    from sewerrtc.v4.pipeline_p3 import build_p3_handlers

    project = tmp_path / "project"
    legacy_root = tmp_path / "outputs" / "v4"
    output = legacy_root / "final_v4"
    (project / "data").mkdir(parents=True)
    network = project / "data" / "wuhan_v8_storage_retrofit.inp"
    network.write_text("[TITLE]\nsynthetic\n", encoding="utf-8")
    inventory_dir = output / "inventory"
    inventory_dir.mkdir(parents=True)
    pd.DataFrame(
        [{"event_id": "ev1", "rainfall_sha256": RAIN_SHA}]
    ).to_csv(inventory_dir / "event_inventory.csv", index=False)
    dataset_dir = output / "pilot" / "dataset"
    dataset_dir.mkdir(parents=True)
    pd.DataFrame(
        [{"event_id": "ev1", "checkpoint_id": "ck1"}]
    ).to_csv(dataset_dir / "pilot_sample_manifest.csv", index=False)
    source_dir = legacy_root / "oracle_pareto_20ev"
    source_dir.mkdir(parents=True)
    schedule = source_dir / "schedules" / "case1.csv"
    schedule.parent.mkdir(parents=True)
    schedule.write_text("step,facility,setting\n", encoding="utf-8")
    pd.DataFrame(
        [
            {
                "case_id": "case1",
                "event_id": "ev1",
                "constraint_mode": "constrained",
                "schedule_csv": "schedules/case1.csv",
                "schedule_sha256": "s" * 64,
                "status": "success",
                "inp_sha256": "x" * 64,
                "rainfall_sha256": RAIN_SHA,
                "max_simultaneous_deviations": 4,
                "runtime_executed": True,
                "authoritative_swmm": True,
            }
        ]
    ).to_csv(source_dir / "oracle_case_results.csv", index=False)
    handlers = build_p3_handlers(
        project_root=project, output_root=output, config={}
    )
    result = handlers["AuditLegacyOracleCompatibility"](
        RuntimeOptions(stage="AuditLegacyOracleCompatibility")
    )
    assert result.exit_code == 0, result.evidence
    target = output / "pilot_feasibility_p3" / "legacy_oracle"
    for name in (
        "legacy_oracle_compatible.csv",
        "legacy_oracle_incompatible.csv",
        "legacy_oracle_replay_plan.csv",
        "legacy_oracle_compatibility_audit.json",
        "completion.json",
    ):
        assert (target / name).exists(), name
    audit = json.loads(
        (target / "legacy_oracle_compatibility_audit.json").read_text(
            encoding="utf-8"
        )
    )
    assert audit["fully_compatible_rows"] == 0
    assert audit["seed_usable_rows"] == 1
    incompatible = pd.read_csv(target / "legacy_oracle_incompatible.csv")
    assert len(incompatible) == 1
    plan = pd.read_csv(target / "legacy_oracle_replay_plan.csv")
    assert plan.iloc[0]["candidate_family"] == SEED_FAMILY
    assert bool(plan.iloc[0]["label_use_forbidden"])
