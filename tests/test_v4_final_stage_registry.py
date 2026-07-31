from pathlib import Path

from sewerrtc.v4.pipeline import ALL_STAGES, build_registry
from sewerrtc.v4.runtime import RuntimeOptions, StageResult, stage_record


def test_final_registry_contains_complete_research_pipeline() -> None:
    required = {
        "AuditContracts",
        "BuildEventInventory",
        "ScanOpportunityPool",
        "AuditOpportunityCoverage",
        "PlanPeakBoundary",
        "RunPeakBoundary",
        "AuditPeakBoundary",
        "PlanPilot400",
        "RunPilot400",
        "EvaluatePilotGate",
        "PlanTrain1600",
        "RunTrainRound0",
        "RunTrainRound3",
        "TrainV4",
        "EvaluateV4Locked",
        "RunExactClosedLoop",
        "RunSurrogateClosedLoop",
        "LockPolicy",
        "RunChallenge",
        "BuildFormalBlindInventory",
        "RunFormalBlind",
        "BuildPaperResults",
        "BuildReproducibilityBundle",
        # V4.2 data pipeline stages
        "BuildV42EventUsageLedger",
        "AuditV42EventUsageLedger",
        "BuildV42UnifiedDevelopmentPool",
        "AuditV42UnifiedDevelopmentPool",
        "BuildV42DerivedSupervision",
        "AuditV42DerivedSupervision",
        "PlanV42NestedGroupedCV",
        "AuditV42NestedGroupedCVPlan",
        "RunV42NestedGroupedCV",
        "BuildV42NestedGroupedCVResults",
        "AuditV42NestedGroupedCVResults",
        # V4.2 validation stages
        "AuditV42HeadActivation",
        "AuditV42TargetMetricSemantics",
        "AuditV42RankingPhysics",
        "RunV42TinyOverfit",
        # V4.2 fresh evaluation stages
        "PlanV42FreshEvaluationSplit",
        "AuditV42FreshEvaluationAvailability",
    }
    assert required.issubset(ALL_STAGES)


def test_dry_run_long_stage_writes_no_scientific_pass(tmp_path: Path) -> None:
    registry = build_registry(
        project_root=tmp_path,
        output_root=tmp_path / "out",
        config={},
    )
    result = registry.run(
        "RunPilot400",
        RuntimeOptions(stage="RunPilot400", dry_run=True),
    )

    assert result.exit_code != 0
    assert not result.scope_complete


def test_every_stage_record_has_required_provenance_and_accounting_fields() -> None:
    record = stage_record(
        StageResult(
            "X",
            "incomplete",
            3,
            completed=2,
            remaining=4,
            batch_complete=True,
            scope_complete=False,
        ),
        config_sha="c",
        code_sha="g",
        input_sha="i",
        started_at=1.0,
        finished_at=2.0,
        run_uuid="u",
    )
    assert {
        "run_uuid",
        "config_sha",
        "code_git_sha",
        "input_sha",
        "started_at",
        "finished_at",
        "exit_code",
        "batch_complete",
        "scope_complete",
        "completed",
        "remaining",
        "completion_marker",
    }.issubset(record)
