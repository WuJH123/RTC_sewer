"""Wiring regression for the Pilot coverage-extension / Gate v2 stages.

These tests guard the pipeline constant tables, the parameterization of the
shared Pilot run/partial machinery, and the registry override order, without
running any SWMM case.
"""

import inspect
from pathlib import Path

from sewerrtc.v4.pipeline import (
    ALL_STAGES,
    LONG_RUN_STAGES,
    OUTPUT_DIRECTORIES,
    PARTIAL_STAGE_RUN,
    PREFLIGHT_STAGE_RUN,
    PREREQUISITES,
    RUN_STAGE_GROUP_KEYS,
    RUN_STAGE_PLANS,
    STAGE_ARTIFACTS,
    _audit_pilot_partial_handler,
    _build_pilot_partial_handler,
    _pilot_partial_bundle,
    _run_pilot400_handler,
    build_registry,
)
from sewerrtc.v4.runtime import RuntimeOptions


EXTENSION_STAGES = (
    "AuditPilotCoverageGaps",
    "PlanPilotCoverageExtension",
    "AuditPilotCoverageExtensionPlan",
    "AuditPilotCoverageExtensionPreflight",
    "RunPilotCoverageExtension",
    "BuildPilotCoverageExtensionPartial",
    "AuditPilotCoverageExtensionPartial",
    "BuildPilotCoverageExtensionDataset",
    "AuditPilotCoverageExtensionDataset",
    "PlanPilotFlatAuxiliary",
    "AuditPilotFlatAuxiliaryPreflight",
    "RunPilotFlatAuxiliary",
    "BuildPilotFlatAuxiliaryDataset",
    "AuditPilotFlatAuxiliaryDataset",
    "BuildPilotDatasetV2",
    "AuditPilotDatasetV2",
    "TrainPilotBaselinesV2",
    "EvaluatePilotGateV2",
)


def test_extension_stages_sit_between_pilot_gate_and_train1600() -> None:
    start = ALL_STAGES.index("EvaluatePilotGate")
    # The Train1600 V3 chain (FreezeP3Evidence...) now sits between the P3
    # gate and the legacy PlanTrain1600 chain, so the extension window ends
    # at FreezeP3Evidence.
    end = ALL_STAGES.index("FreezeP3Evidence")
    # The Gate P3 chain sits after the v2 gate, before the V3 freeze chain.
    assert ALL_STAGES[start + 1 : end] == EXTENSION_STAGES + (
        "AuditLegacyOracleCompatibility",
        "PlanPilotFeasibilityMap",
        "AuditPilotFeasibilityPreflight",
        "RunPilotFeasibilityMap",
        "BuildPilotFeasibilityPartial",
        "AuditPilotFeasibilityPartial",
        "BuildPilotFeasibilityMap",
        "AuditPilotFeasibilityMap",
        "BuildPilotDatasetV3",
        "AuditPilotDatasetV3",
        "TrainPilotBaselinesV3",
        "EvaluatePilotGateV3",
    )


def test_extension_constant_tables_are_complete() -> None:
    for stage in EXTENSION_STAGES:
        assert stage in STAGE_ARTIFACTS, stage
        assert stage in PREREQUISITES, stage
    assert "pilot_extension_v1" in OUTPUT_DIRECTORIES
    assert "RunPilotCoverageExtension" in LONG_RUN_STAGES
    assert "RunPilotFlatAuxiliary" in LONG_RUN_STAGES
    assert (
        PARTIAL_STAGE_RUN["BuildPilotCoverageExtensionPartial"]
        == "RunPilotCoverageExtension"
    )
    assert (
        PARTIAL_STAGE_RUN["AuditPilotCoverageExtensionPartial"]
        == "RunPilotCoverageExtension"
    )
    assert (
        PREFLIGHT_STAGE_RUN["AuditPilotCoverageExtensionPreflight"]
        == "RunPilotCoverageExtension"
    )
    assert (
        PREFLIGHT_STAGE_RUN["AuditPilotFlatAuxiliaryPreflight"]
        == "RunPilotFlatAuxiliary"
    )
    assert RUN_STAGE_PLANS["RunPilotCoverageExtension"] == (
        "pilot_extension_v1/planning/extension_candidate_plan.csv"
    )
    assert RUN_STAGE_PLANS["RunPilotFlatAuxiliary"] == (
        "pilot_extension_v1/flat_auxiliary/planning/"
        "flat_auxiliary_candidate_plan.csv"
    )
    assert RUN_STAGE_GROUP_KEYS["RunPilotCoverageExtension"] == "sample_id"
    assert RUN_STAGE_GROUP_KEYS["RunPilotFlatAuxiliary"] == "sample_id"


def test_gap_audit_depends_only_on_contracts_to_avoid_deadlock() -> None:
    # AuditPilotDataset holds a frozen scientific_fail under Gate v1, so the
    # read-only diagnosis must not gate on any pilot v1 stage status.
    assert PREREQUISITES["AuditPilotCoverageGaps"] == ("AuditContracts",)
    # Training gates on the dataset build; the audit verdict (expected
    # scientific_fail) is absorbed fail-closed by EvaluatePilotGateV2.
    assert PREREQUISITES["TrainPilotBaselinesV2"] == ("BuildPilotDatasetV2",)
    assert PREREQUISITES["EvaluatePilotGateV2"] == ("TrainPilotBaselinesV2",)
    # Train1600 entry is untouched: it still gates on the v1 pilot gate and
    # never on the v2 verdict, so passing Gate v2 cannot auto-open Train1600.
    assert PREREQUISITES["PlanTrain1600"] == ("EvaluatePilotGate",)


def test_pilot_machinery_defaults_preserve_v1_behavior() -> None:
    for func in (
        _run_pilot400_handler,
        _build_pilot_partial_handler,
        _audit_pilot_partial_handler,
        _pilot_partial_bundle,
    ):
        parameters = inspect.signature(func).parameters
        branch = parameters["branch_plan_rel"]
        assert branch.kind is inspect.Parameter.KEYWORD_ONLY
        assert branch.default == "pilot/planning/pilot_branch_plan.csv"
        run_key = "stage" if func is _run_pilot400_handler else "run_stage"
        run_param = parameters[run_key]
        assert run_param.kind is inspect.Parameter.KEYWORD_ONLY
        assert run_param.default == "RunPilot400"


def test_registry_registers_all_extension_stages(tmp_path: Path) -> None:
    registry = build_registry(
        project_root=tmp_path,
        output_root=tmp_path / "out",
        config={},
    )
    assert set(EXTENSION_STAGES).issubset(registry.names)


def test_extension_run_stage_is_gated_until_preflight_passes(
    tmp_path: Path,
) -> None:
    registry = build_registry(
        project_root=tmp_path,
        output_root=tmp_path / "out",
        config={},
    )
    result = registry.run(
        "RunPilotCoverageExtension",
        RuntimeOptions(stage="RunPilotCoverageExtension", dry_run=True),
    )
    assert result.exit_code != 0
    assert not result.scope_complete
    assert result.evidence.get("reason") == "prerequisite_not_passed"


def test_run_handler_reads_stage_specific_plan_paths(tmp_path: Path) -> None:
    handler = _run_pilot400_handler(
        tmp_path,
        {},
        stage="RunPilotCoverageExtension",
        branch_plan_rel="pilot_extension_v1/planning/extension_branch_plan.csv",
    )
    result = handler(RuntimeOptions(stage="RunPilotCoverageExtension"))
    assert result.exit_code != 0
    missing = result.evidence.get("missing_inputs", [])
    assert any("extension_candidate_plan.csv" in item for item in missing)
    assert any("extension_branch_plan.csv" in item for item in missing)
