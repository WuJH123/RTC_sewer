from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path
from typing import Callable

import pandas as pd

from .contracts import audit_final_contract
from .dataset import (
    build_branch_manifest,
    build_sample_manifest,
    dataset_accounting,
    audit_train1600_dataset,
    feature_schema,
    label_schema,
)
from .evaluation import audit_formal_blind_inventory, build_policy_lock
from .event_splits import (
    LEDGER_COLUMNS,
    USAGE_FLAG_COLUMNS,
    EventShortfallError,
    assign_split,
    build_event_usage_ledger,
    select_formal_blind_candidates,
    select_train1600_events,
)
from .inventory import build_inventory_from_catalog
from .opportunity import (
    audit_opportunity_coverage,
    build_canonical_catalogs,
    build_opportunity_pool,
    plan_opportunity_scans,
)
from .partial_audit import (
    applicable_release_level,
    audit_partial_quality,
    build_partial_bundle,
    partial_accounting,
    progressive_release_gate,
)
from .preflight import preflight_checks
from .peak_boundary import (
    audit_peak_boundary,
    build_peak_boundary_dataset,
    build_peak_candidate_catalog,
    build_peak_boundary_plan,
    build_peak_partial_bundle,
    peak_constraint_binding_audit,
)
from .peak_restamp import restamp_peak_boundary_evidence
from .pilot import (
    audit_pilot_dataset,
    build_pilot_planning_bundle,
    evaluate_pilot_gate,
)
from .pilot_candidates import (
    audit_pilot_materialized_plan,
    build_pilot_branch_plan,
    build_pilot_role_plan,
    materialize_pilot_candidates,
)
from .pilot_reducers import (
    build_pilot_dataset,
    build_pilot_partial_bundle,
)
from .pilot_run import (
    attach_branch_context,
    expand_pilot_completions,
    run_pilot_sample,
)
from .reporting import paper_artifact_paths
from .runtime import (
    EXIT_BLOCKED,
    EXIT_INCOMPLETE,
    EXIT_PASS,
    EXIT_SCIENTIFIC_FAIL,
    StageRegistry,
    StageResult,
    RuntimeOptions,
    atomic_write_json,
    completion_manifest,
    run_parallel_cases,
    working_code_sha,
)
from .simulation import run_prepared_case
from .training_plan import (
    audit_train1600_plan,
    build_round0_plan,
    build_train1600_target_plan,
    build_train_checkpoint_catalog,
)


ALL_STAGES = (
    "AuditContracts",
    "BuildEventInventory",
    "PlanOpportunityPool",
    "ScanOpportunityPool",
    "BuildOpportunityPool",
    "AuditOpportunityCoverage",
    "BuildPeakCandidateCatalog",
    "PlanPeakBoundary",
    "AuditPeakBoundaryPreflight",
    "RunPeakBoundary",
    "BuildPeakBoundaryPartial",
    "AuditPeakBoundaryPartial",
    "BuildPeakBoundaryDataset",
    "AuditPeakBoundary",
    "RestampPeakBoundaryEvidence",
    "ClassifyExistingGate5R",
    "PlanPilot400",
    "AuditPilotPlan",
    "AuditPilotPreflight",
    "RunPilot400",
    "BuildPilotPartial",
    "AuditPilotPartial",
    "BuildPilotDataset",
    "AuditPilotDataset",
    "TrainPilotBaselines",
    "EvaluatePilotGate",
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
    "FreezeP3Evidence",
    "AuditDataGenerationAuthorizationV3",
    "PlanTrain1600V3",
    "AuditTrain1600PlanV3",
    "AuditTrainRound0PreflightV3",
    "RunTrainRound0V3",
    "BuildTrainRound0PartialV3",
    "AuditTrainRound0PartialV3",
    "BuildTrainRound0V3",
    "AuditTrainRound0V3",
    "TrainActiveLearner0V3",
    "SelectTrainRound1V3",
    "RunTrainRound1V3",
    "BuildTrainRound1PartialV3",
    "AuditTrainRound1PartialV3",
    "BuildTrainRound1V3",
    "AuditTrainRound1V3",
    "TrainActiveLearner1V3",
    "SelectTrainRound2V3",
    "RunTrainRound2V3",
    "BuildTrainRound2PartialV3",
    "AuditTrainRound2PartialV3",
    "BuildTrainRound2V3",
    "AuditTrainRound2V3",
    "PlanCalibration200V3",
    "RunCalibration200V3",
    "BuildCalibration200V3",
    "AuditCalibration200V3",
    "PlanLockedValidation200V3",
    "RunLockedValidation200V3",
    "BuildLockedValidation200V3",
    "AuditLockedValidation200V3",
    "BuildTrain1600DatasetV3",
    "AuditTrain1600DatasetV3",
    "FreezeTrain1600V3Evidence",
    "AuditTrain1600LearnabilityV4",
    "AuditModelTrainingAuthorizationV4",
    "TrainV4Baselines",
    "EvaluateV4Baselines",
    "TrainV4TrueState",
    "CalibrateV4TrueState",
    "EvaluateV4TrueStateLocked",
    "AuditV4OfflineSafetyGate",
    # V4.1 Compact Head-Specific Surrogate Rescue -- Phase-1 (Train-only).
    "FreezeV4OfflineV0Evidence",
    "AuditV4LockedMetricComparabilityV0",
    "AuditV4GeneralizationFailureV0",
    "BuildV4FeatureBlockCatalogV1",
    "BuildV4LearningCurvesV1",
    "RunV4FeatureBlockAblationV1",
    "RunV4HeadArchitectureAblationV1",
    "AuditV4MultitaskGradientConflictV1",
    "SelectV4CompactModelV1",
    "TrainV4CompactTrueStateV1",
    # V4.1 Compact rescue -- Phase-2 (fresh independent Calibration / Locked).
    "PlanV4CompactCalibrationLockedV1",
    "AuditV4CompactEvaluationPlanV1",
    "RunV4CompactCalibrationV1",
    "BuildV4CompactCalibrationV1",
    "AuditV4CompactCalibrationV1",
    "RunV4CompactLockedV1",
    "BuildV4CompactLockedV1",
    "AuditV4CompactLockedV1",
    "CalibrateV4CompactV1",
    "EvaluateV4CompactLockedV1",
    "AuditV4PredictiveGeneralizationGateV1",
    "AuditGATClosedLoopReadiness",
    "PlanTrain1600",
    "AuditTrain1600Plan",
    "AuditTrainRound0Preflight",
    "RunTrainRound0",
    "BuildTrainRound0Partial",
    "AuditTrainRound0Partial",
    "AuditTrainRound0",
    "TrainActiveLearner0",
    "SelectTrainRound1",
    "AuditTrainRound1Preflight",
    "RunTrainRound1",
    "BuildTrainRound1Partial",
    "AuditTrainRound1Partial",
    "AuditTrainRound1",
    "TrainActiveLearner1",
    "SelectTrainRound2",
    "AuditTrainRound2Preflight",
    "RunTrainRound2",
    "BuildTrainRound2Partial",
    "AuditTrainRound2Partial",
    "AuditTrainRound2",
    "TrainActiveLearner2",
    "SelectTrainRound3",
    "AuditTrainRound3Preflight",
    "RunTrainRound3",
    "BuildTrainRound3Partial",
    "AuditTrainRound3Partial",
    "AuditTrainRound3",
    "BuildTrain1600Dataset",
    "AuditTrain1600Dataset",
    "TrainV4",
    "CalibrateV4",
    "EvaluateV4Locked",
    "PlanExactClosedLoop",
    "RunExactClosedLoop",
    "AuditExactClosedLoop",
    "PlanSurrogateClosedLoop",
    "RunSurrogateClosedLoop",
    "AuditSurrogateClosedLoop",
    "LockPolicy",
    "RunChallenge",
    "AuditChallenge",
    "BuildFormalBlindInventory",
    "RunFormalBlind",
    "AuditFormalBlind",
    "BuildPaperResults",
    "BuildPaperFigures",
    "BuildPaperTables",
    "BuildReproducibilityBundle",
    # V4.2 stages
    "FreezeV41ScientificFailure",
    "AuditV41ClassificationMetricSemantics",
    "BuildV42TrajectoryDataset",
    "AuditV42TrajectoryDataset",
    "TrainV42WaterBalanceBaseline",
    "EvaluateV42WaterBalanceBaseline",
    "TrainV42TwinGraphDynamics",
    "RunV42ArchitectureAblation",
    "RunV42StateScopeAblation",
    "RunV42TargetAblation",
    "BuildV42EventLearningCurve",
    # V4.2 data pipeline stages (event ledger → unified pool → supervision → CV)
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
    # V4.2 fresh evaluation
    "PlanV42FreshEvaluationSplit",
    "AuditV42FreshEvaluationAvailability",
    "AuditV42TrainGroupedGeneralizationGate",
    # V4.2 final data pool stages (priority contract → admission gate)
    "AuditV42PrioritySentinelContract",
    "FreezeV42PriorityContract",
    "BuildV42IndependentPfvOracle",
    "BuildV42SampleLineage",
    "AuditV42PhysicalDeduplication",
    "BuildV42HistoricalSemanticInventory",
    "AuditV42HistoricalSemanticInventory",
    "AuditV42DwfSources",
    "BuildV42Canonical13FrameTrajectories",
    "AuditV42Canonical13FrameTrajectories",
    "BuildV42TfvPeakOracle",
    "BuildV42SampleClassifier",
    "BuildV42FinalUnifiedDatasets",
    "AuditV42FinalUnifiedDatasets",
    "BuildV42GroupedSplits",
    "BuildV42PoolStatistics",
    "AuditV42FinalDatasetAdmissionGate",
)


LONG_RUN_STAGES = {
    "ScanOpportunityPool",
    "RunPeakBoundary",
    "RunPilot400",
    "RunPilotCoverageExtension",
    "RunPilotFlatAuxiliary",
    "RunPilotFeasibilityMap",
    "RunTrainRound0",
    "RunTrainRound1",
    "RunTrainRound2",
    "RunTrainRound3",
    "RunTrainRound0V3",
    "RunTrainRound1V3",
    "RunTrainRound2V3",
    "RunCalibration200V3",
    "RunLockedValidation200V3",
    "TrainActiveLearner0",
    "TrainActiveLearner1",
    "TrainActiveLearner2",
    "TrainV4",
    "CalibrateV4",
    "TrainV4TrueState",
    "CalibrateV4TrueState",
    "RunExactClosedLoop",
    "RunSurrogateClosedLoop",
    "RunChallenge",
    "RunFormalBlind",
}


OUTPUT_DIRECTORIES = (
    "inventory",
    "opportunities",
    "peak_boundary",
    "pilot",
    "pilot_extension_v1",
    "pilot_feasibility_p3",
    "train1600",
    "train1600_v3",
    "models",
    "exact_closed_loop",
    "surrogate_closed_loop",
    "challenge",
    "formal_blind",
    "paper",
    "audits",
    "logs",
    "heartbeats",
)


STAGE_ARTIFACTS = {
    "BuildEventInventory": "inventory/event_inventory.csv",
    "PlanOpportunityPool": "opportunities/opportunity_scan_plan.csv",
    "ScanOpportunityPool": "opportunities/opportunity_scan_run_manifest.csv",
    "BuildOpportunityPool": "opportunities/opportunity_pool.csv",
    "AuditOpportunityCoverage": "opportunities/opportunity_coverage_audit.json",
    "BuildPeakCandidateCatalog": "opportunities/peak_candidate_catalog.csv",
    "PlanPeakBoundary": "peak_boundary/peak_boundary_plan.csv",
    "RunPeakBoundary": "peak_boundary/run_manifest.csv",
    "BuildPeakBoundaryDataset": "peak_boundary/sample_manifest.csv",
    "AuditPeakBoundary": "peak_boundary/peak_boundary_audit.json",
    "RestampPeakBoundaryEvidence": "peak_boundary/restamp/restamp_stamp.json",
    "ClassifyExistingGate5R": "pilot/existing_gate5r_classification.csv",
    "PlanPilot400": "pilot/planning/pilot_candidate_plan.csv",
    "AuditPilotPlan": "pilot/planning/pilot_plan_audit.json",
    "RunPilot400": "pilot/run_manifest.csv",
    "BuildPilotDataset": "pilot/dataset/pilot_sample_manifest.csv",
    "AuditPilotDataset": "pilot/dataset/pilot_dataset_audit.json",
    "TrainPilotBaselines": "pilot/baseline_models_report.json",
    "EvaluatePilotGate": "pilot/evaluation/pilot_gate_verdict.json",
    "AuditPilotCoverageGaps": (
        "pilot_extension_v1/gaps/pilot_v1_gap_audit.json"
    ),
    "PlanPilotCoverageExtension": (
        "pilot_extension_v1/planning/extension_candidate_plan.csv"
    ),
    "AuditPilotCoverageExtensionPlan": (
        "pilot_extension_v1/planning/extension_plan_audit.json"
    ),
    "RunPilotCoverageExtension": "pilot_extension_v1/run_manifest.csv",
    "BuildPilotCoverageExtensionDataset": (
        "pilot_extension_v1/dataset/extension_sample_manifest.csv"
    ),
    "AuditPilotCoverageExtensionDataset": (
        "pilot_extension_v1/dataset/extension_dataset_audit.json"
    ),
    "PlanPilotFlatAuxiliary": (
        "pilot_extension_v1/flat_auxiliary/planning/"
        "flat_auxiliary_candidate_plan.csv"
    ),
    "RunPilotFlatAuxiliary": (
        "pilot_extension_v1/flat_auxiliary/run_manifest.csv"
    ),
    "BuildPilotFlatAuxiliaryDataset": (
        "pilot_extension_v1/flat_auxiliary/dataset/"
        "flat_auxiliary_sample_manifest.csv"
    ),
    "AuditPilotFlatAuxiliaryDataset": (
        "pilot_extension_v1/flat_auxiliary/dataset/"
        "flat_auxiliary_dataset_audit.json"
    ),
    "BuildPilotDatasetV2": "pilot/dataset_v2/pilot_v2_sample_manifest.csv",
    "AuditPilotDatasetV2": "pilot/dataset_v2/pilot_v2_dataset_audit.json",
    "TrainPilotBaselinesV2": (
        "pilot/dataset_v2/baseline_models_report_v2.json"
    ),
    "EvaluatePilotGateV2": "pilot/evaluation/pilot_gate_v2_verdict.json",
    "AuditLegacyOracleCompatibility": (
        "pilot_feasibility_p3/legacy_oracle/"
        "legacy_oracle_compatibility_audit.json"
    ),
    "PlanPilotFeasibilityMap": (
        "pilot_feasibility_p3/planning/feasibility_candidate_plan.csv"
    ),
    "RunPilotFeasibilityMap": "pilot_feasibility_p3/run_manifest.csv",
    "BuildPilotFeasibilityMap": (
        "pilot_feasibility_p3/map/pilot_state_feasibility_map.csv"
    ),
    "AuditPilotFeasibilityMap": (
        "pilot_feasibility_p3/map/pilot_feasibility_audit.json"
    ),
    "BuildPilotDatasetV3": (
        "pilot_feasibility_p3/dataset_v3/pilot_v3_sample_manifest.csv"
    ),
    "AuditPilotDatasetV3": (
        "pilot_feasibility_p3/dataset_v3/pilot_v3_dataset_audit.json"
    ),
    "TrainPilotBaselinesV3": (
        "pilot_feasibility_p3/dataset_v3/baseline_models_report_v3.json"
    ),
    "EvaluatePilotGateV3": (
        "pilot_feasibility_p3/evaluation/pilot_gate_v3_verdict.json"
    ),
    "FreezeP3Evidence": (
        "audits/frozen_evidence/pilot_feasibility_p3/freeze_pointer.json"
    ),
    "AuditDataGenerationAuthorizationV3": (
        "train1600_v3/authorization/data_generation_authorization_v3.json"
    ),
    "PlanTrain1600V3": (
        "train1600_v3/planning/train_checkpoint_catalog_v3.csv"
    ),
    "AuditTrain1600PlanV3": (
        "train1600_v3/planning/train1600_plan_audit_v3.json"
    ),
    "RunTrainRound0V3": "train1600_v3/round0/run_manifest.csv",
    "BuildTrainRound0V3": (
        "train1600_v3/round0/dataset/round_sample_manifest.csv"
    ),
    "AuditTrainRound0V3": "train1600_v3/round0/audit.json",
    "TrainActiveLearner0V3": (
        "train1600_v3/round0/active_learner_v3.json"
    ),
    "SelectTrainRound1V3": "train1600_v3/round1/plan.csv",
    "RunTrainRound1V3": "train1600_v3/round1/run_manifest.csv",
    "BuildTrainRound1V3": (
        "train1600_v3/round1/dataset/round_sample_manifest.csv"
    ),
    "AuditTrainRound1V3": "train1600_v3/round1/audit.json",
    "TrainActiveLearner1V3": (
        "train1600_v3/round1/active_learner_v3.json"
    ),
    "SelectTrainRound2V3": "train1600_v3/round2/plan.csv",
    "RunTrainRound2V3": "train1600_v3/round2/run_manifest.csv",
    "BuildTrainRound2V3": (
        "train1600_v3/round2/dataset/round_sample_manifest.csv"
    ),
    "AuditTrainRound2V3": "train1600_v3/round2/audit.json",
    "PlanCalibration200V3": "train1600_v3/calibration/plan.csv",
    "RunCalibration200V3": "train1600_v3/calibration/run_manifest.csv",
    "BuildCalibration200V3": (
        "train1600_v3/calibration/dataset/round_sample_manifest.csv"
    ),
    "AuditCalibration200V3": "train1600_v3/calibration/audit.json",
    "PlanLockedValidation200V3": (
        "train1600_v3/locked_validation/plan.csv"
    ),
    "RunLockedValidation200V3": (
        "train1600_v3/locked_validation/run_manifest.csv"
    ),
    "BuildLockedValidation200V3": (
        "train1600_v3/locked_validation/dataset/round_sample_manifest.csv"
    ),
    "AuditLockedValidation200V3": (
        "train1600_v3/locked_validation/audit.json"
    ),
    "BuildTrain1600DatasetV3": (
        "train1600_v3/dataset/train1600_v3_sample_manifest.csv"
    ),
    "AuditTrain1600DatasetV3": (
        "train1600_v3/dataset/train1600_v3_dataset_audit.json"
    ),
    "FreezeTrain1600V3Evidence": (
        "audits/frozen_evidence/train1600_v3/freeze_pointer.json"
    ),
    "AuditTrain1600LearnabilityV4": (
        "train1600_v3/training_readiness_v4/training_readiness_verdict.json"
    ),
    "AuditModelTrainingAuthorizationV4": (
        "train1600_v3/authorization/model_training_authorization_v4.json"
    ),
    "TrainV4Baselines": (
        "models/v4_true_state/baseline_models.json"
    ),
    "EvaluateV4Baselines": (
        "models/v4_true_state/baseline_evaluation.json"
    ),
    "TrainV4TrueState": (
        "models/v4_true_state/true_state_training_summary.json"
    ),
    "CalibrateV4TrueState": (
        "models/v4_true_state/true_state_calibration.json"
    ),
    "EvaluateV4TrueStateLocked": (
        "models/v4_true_state/locked_evaluation.json"
    ),
    "AuditV4OfflineSafetyGate": (
        "models/v4_true_state/offline_safety_gate.json"
    ),
    "FreezeV4OfflineV0Evidence": (
        "audits/frozen_evidence/v4_offline_v0/freeze_pointer.json"
    ),
    "AuditV4LockedMetricComparabilityV0": (
        "audits/v4_diagnostics/locked_v0_metric_comparability.json"
    ),
    "AuditV4GeneralizationFailureV0": (
        "audits/v4_diagnostics/generalization_failure_v0.json"
    ),
    "BuildV4FeatureBlockCatalogV1": (
        "models/v4_compact_v1/feature_block_catalog_summary.json"
    ),
    "BuildV4LearningCurvesV1": (
        "models/v4_compact_v1/learning_curves_summary.json"
    ),
    "RunV4FeatureBlockAblationV1": (
        "models/v4_compact_v1/feature_block_ablation.json"
    ),
    "RunV4HeadArchitectureAblationV1": (
        "models/v4_compact_v1/head_architecture_ablation.json"
    ),
    "AuditV4MultitaskGradientConflictV1": (
        "models/v4_compact_v1/gradient_conflict.json"
    ),
    "SelectV4CompactModelV1": (
        "models/v4_compact_v1/v4_compact_v1_selection.json"
    ),
    "TrainV4CompactTrueStateV1": (
        "models/v4_compact_v1/completion.json"
    ),
    "PlanV4CompactCalibrationLockedV1": (
        "v4_compact_eval/planning/evaluation_plan_freeze.json"
    ),
    "AuditV4CompactEvaluationPlanV1": (
        "v4_compact_eval/planning/evaluation_plan_audit.json"
    ),
    "RunV4CompactCalibrationV1": "v4_compact_eval/calibration/run_manifest.csv",
    "BuildV4CompactCalibrationV1": (
        "v4_compact_eval/calibration/dataset/round_sample_manifest.csv"
    ),
    "AuditV4CompactCalibrationV1": "v4_compact_eval/calibration/round_audit.json",
    "RunV4CompactLockedV1": "v4_compact_eval/locked/run_manifest.csv",
    "BuildV4CompactLockedV1": (
        "v4_compact_eval/locked/dataset/round_sample_manifest.csv"
    ),
    "AuditV4CompactLockedV1": "v4_compact_eval/locked/round_audit.json",
    "CalibrateV4CompactV1": (
        "models/v4_compact_v1/v4_compact_v1_calibration.json"
    ),
    "EvaluateV4CompactLockedV1": (
        "models/v4_compact_v1/v4_compact_v1_locked_evaluation.json"
    ),
    "AuditV4PredictiveGeneralizationGateV1": (
        "models/v4_compact_v1/v4_predictive_generalization_gate.json"
    ),
    "AuditGATClosedLoopReadiness": "audits/gat_closed_loop_readiness.json",
    "PlanTrain1600": "train1600/planning/train_checkpoint_catalog.csv",
    "AuditTrain1600Plan": "train1600/planning/train1600_plan_audit.json",
    "RunTrainRound0": "train1600/round0/run_manifest.csv",
    "AuditTrainRound0": "train1600/round0/audit.json",
    "TrainActiveLearner0": "train1600/round0/active_learner.json",
    "SelectTrainRound1": "train1600/round1/plan.csv",
    "RunTrainRound1": "train1600/round1/run_manifest.csv",
    "AuditTrainRound1": "train1600/round1/audit.json",
    "TrainActiveLearner1": "train1600/round1/active_learner.json",
    "SelectTrainRound2": "train1600/round2/plan.csv",
    "RunTrainRound2": "train1600/round2/run_manifest.csv",
    "AuditTrainRound2": "train1600/round2/audit.json",
    "TrainActiveLearner2": "train1600/round2/active_learner.json",
    "SelectTrainRound3": "train1600/round3/plan.csv",
    "RunTrainRound3": "train1600/round3/run_manifest.csv",
    "AuditTrainRound3": "train1600/round3/audit.json",
    "BuildTrain1600Dataset": "train1600/dataset/train1600_sample_manifest.csv",
    "AuditTrain1600Dataset": "train1600/dataset/train1600_dataset_audit.json",
    "TrainV4": "models/v4_model.json",
    "CalibrateV4": "models/v4_calibration.json",
    "EvaluateV4Locked": "models/v4_locked_evaluation.json",
    "PlanExactClosedLoop": "exact_closed_loop/plan.csv",
    "RunExactClosedLoop": "exact_closed_loop/run_manifest.csv",
    "AuditExactClosedLoop": "exact_closed_loop/audit.json",
    "PlanSurrogateClosedLoop": "surrogate_closed_loop/plan.csv",
    "RunSurrogateClosedLoop": "surrogate_closed_loop/run_manifest.csv",
    "AuditSurrogateClosedLoop": "surrogate_closed_loop/audit.json",
    "LockPolicy": "models/policy_lock.json",
    "RunChallenge": "challenge/run_manifest.csv",
    "AuditChallenge": "challenge/audit.json",
    "BuildFormalBlindInventory": "formal_blind/inventory.csv",
    "RunFormalBlind": "formal_blind/run_manifest.csv",
    "AuditFormalBlind": "formal_blind/audit.json",
    "BuildPaperResults": "paper/results/final_event_metrics.csv",
    "BuildPaperFigures": "paper/figures/figure_manifest.csv",
    "BuildPaperTables": "paper/tables/table_manifest.csv",
    "BuildReproducibilityBundle": "paper/reproducibility/sha_manifest.csv",
    # V4.2 stages
    "FreezeV41ScientificFailure": (
        "audits/frozen_evidence/v41_scientific_failure/v41_freeze_manifest.json"
    ),
    "AuditV41ClassificationMetricSemantics": (
        "audits/v42_metric_semantics/v41_metric_semantics_audit.json"
    ),
    "BuildV42TrajectoryDataset": (
        "train1600_v3/trajectory_manifest_v42.parquet"
    ),
    "AuditV42TrajectoryDataset": (
        "train1600_v3/trajectory_dataset_audit.json"
    ),
    "TrainV42WaterBalanceBaseline": (
        "models/v42_water_balance/water_balance_baseline_cv.json"
    ),
    "EvaluateV42WaterBalanceBaseline": (
        "models/v42_water_balance/water_balance_evaluation.json"
    ),
    "TrainV42TwinGraphDynamics": (
        "models/v42_twin/training_history.json"
    ),
    "RunV42ArchitectureAblation": (
        "models/v42_ablation/architecture_ablation.json"
    ),
    "RunV42StateScopeAblation": (
        "models/v42_ablation/state_scope_ablation.json"
    ),
    "RunV42TargetAblation": (
        "models/v42_ablation/target_ablation.json"
    ),
    "BuildV42EventLearningCurve": (
        "models/v42_twin/learning_curve_v42.json"
    ),
    "AuditV42TrainGroupedGeneralizationGate": (
        "audits/v42_gate/v42_train_gate_verdict.json"
    ),
    # V4.2 data pipeline artifacts
    "BuildV42EventUsageLedger": "v42/event_ledger/event_usage_ledger_v42.csv",
    "AuditV42EventUsageLedger": "v42/event_ledger/event_usage_audit.json",
    "BuildV42UnifiedDevelopmentPool": "v42/development_pool/unified_pool_manifest.parquet",
    "AuditV42UnifiedDevelopmentPool": "v42/development_pool/unified_pool_audit.json",
    "BuildV42DerivedSupervision": "v42/derived_supervision/supervision_signals.parquet",
    "AuditV42DerivedSupervision": "v42/derived_supervision/supervision_audit.json",
    "PlanV42NestedGroupedCV": "v42/cv/nested_cv_plan.json",
    "AuditV42NestedGroupedCVPlan": "v42/cv/nested_cv_plan_audit.json",
    "RunV42NestedGroupedCV": "v42/cv/nested_cv_results.json",
    "BuildV42NestedGroupedCVResults": "v42/cv/nested_cv_aggregated.json",
    "AuditV42NestedGroupedCVResults": "v42/cv/nested_cv_results_audit.json",
    "AuditV42HeadActivation": "v42/validation/head_activation_audit.json",
    "AuditV42TargetMetricSemantics": "v42/validation/metric_semantics_audit.json",
    "AuditV42RankingPhysics": "v42/validation/ranking_physics_audit.json",
    "RunV42TinyOverfit": "v42/validation/tiny_overfit_results.json",
    "PlanV42FreshEvaluationSplit": "v42/fresh_eval/fresh_evaluation_plan.json",
    "AuditV42FreshEvaluationAvailability": "v42/fresh_eval/fresh_eval_availability_audit.json",
    # V4.2 final data pool artifacts
    "AuditV42PrioritySentinelContract": "audits/v42_final_pool/priority_contract_audit.json",
    "FreezeV42PriorityContract": "docs/contracts/PROJECT6_V42_PRIORITY_PFV_CONTRACT.json",
    "BuildV42IndependentPfvOracle": "audits/v42_final_pool/pfv_oracle_audit.json",
    "BuildV42SampleLineage": "audits/v42_final_pool/sample_lineage.parquet",
    "AuditV42PhysicalDeduplication": "audits/v42_final_pool/deduplication_audit.json",
    "BuildV42HistoricalSemanticInventory": "audits/v42_final_pool/semantic_sample_inventory.parquet",
    "AuditV42HistoricalSemanticInventory": "audits/v42_final_pool/semantic_source_summary.csv",
    "AuditV42DwfSources": "audits/v42_final_pool/dwf_audit_summary.json",
    "BuildV42Canonical13FrameTrajectories": "audits/v42_final_pool/history_rebuild_audit.json",
    "AuditV42Canonical13FrameTrajectories": "audits/v42_final_pool/history_rebuild_audit.json",
    "BuildV42TfvPeakOracle": "audits/v42_final_pool/tfv_peak_oracle_audit.json",
    "BuildV42SampleClassifier": "audits/v42_final_pool/sample_classification_summary.json",
    "BuildV42FinalUnifiedDatasets": "data/v42_final_unified/dataset_manifest.json",
    "AuditV42FinalUnifiedDatasets": "audits/v42_final_pool/final_dataset_audit.json",
    "BuildV42GroupedSplits": "audits/v42_final_pool/grouped_splits.json",
    "BuildV42PoolStatistics": "audits/v42_final_pool/pool_statistics.json",
    "AuditV42FinalDatasetAdmissionGate": "audits/v42_final_pool/admission_gate_result.json",
}

# Partial stages read the run stage's completion markers without requiring
# scope completion; preflight stages gate the run stage before any case.
PARTIAL_STAGE_RUN = {
    "BuildPeakBoundaryPartial": "RunPeakBoundary",
    "AuditPeakBoundaryPartial": "RunPeakBoundary",
    "BuildPilotPartial": "RunPilot400",
    "AuditPilotPartial": "RunPilot400",
    "BuildPilotCoverageExtensionPartial": "RunPilotCoverageExtension",
    "AuditPilotCoverageExtensionPartial": "RunPilotCoverageExtension",
    "BuildPilotFeasibilityPartial": "RunPilotFeasibilityMap",
    "AuditPilotFeasibilityPartial": "RunPilotFeasibilityMap",
    "BuildTrainRound0Partial": "RunTrainRound0",
    "AuditTrainRound0Partial": "RunTrainRound0",
    "BuildTrainRound1Partial": "RunTrainRound1",
    "AuditTrainRound1Partial": "RunTrainRound1",
    "BuildTrainRound2Partial": "RunTrainRound2",
    "AuditTrainRound2Partial": "RunTrainRound2",
    "BuildTrainRound3Partial": "RunTrainRound3",
    "AuditTrainRound3Partial": "RunTrainRound3",
    "BuildTrainRound0PartialV3": "RunTrainRound0V3",
    "AuditTrainRound0PartialV3": "RunTrainRound0V3",
    "BuildTrainRound1PartialV3": "RunTrainRound1V3",
    "AuditTrainRound1PartialV3": "RunTrainRound1V3",
    "BuildTrainRound2PartialV3": "RunTrainRound2V3",
    "AuditTrainRound2PartialV3": "RunTrainRound2V3",
}

PREFLIGHT_STAGE_RUN = {
    "AuditPeakBoundaryPreflight": "RunPeakBoundary",
    "AuditPilotPreflight": "RunPilot400",
    "AuditPilotCoverageExtensionPreflight": "RunPilotCoverageExtension",
    "AuditPilotFlatAuxiliaryPreflight": "RunPilotFlatAuxiliary",
    "AuditPilotFeasibilityPreflight": "RunPilotFeasibilityMap",
    "AuditTrainRound0Preflight": "RunTrainRound0",
    "AuditTrainRound1Preflight": "RunTrainRound1",
    "AuditTrainRound2Preflight": "RunTrainRound2",
    "AuditTrainRound3Preflight": "RunTrainRound3",
    "AuditTrainRound0PreflightV3": "RunTrainRound0V3",
}

STAGE_ARTIFACTS.update(
    {
        stage: f"audits/partial/{stage}/latest.json"
        for stage in PARTIAL_STAGE_RUN
    }
)
STAGE_ARTIFACTS.update(
    {
        stage: f"audits/preflight/{stage}.json"
        for stage in PREFLIGHT_STAGE_RUN
    }
)

RUN_STAGE_PLANS = {
    "ScanOpportunityPool": "opportunities/opportunity_scan_plan.csv",
    "RunPeakBoundary": "peak_boundary/peak_boundary_plan.csv",
    "RunPilot400": "pilot/planning/pilot_candidate_plan.csv",
    "RunPilotCoverageExtension": (
        "pilot_extension_v1/planning/extension_candidate_plan.csv"
    ),
    "RunPilotFlatAuxiliary": (
        "pilot_extension_v1/flat_auxiliary/planning/"
        "flat_auxiliary_candidate_plan.csv"
    ),
    "RunPilotFeasibilityMap": (
        "pilot_feasibility_p3/planning/feasibility_candidate_plan.csv"
    ),
    "RunTrainRound0": "train1600/round0/plan.csv",
    "RunTrainRound1": "train1600/round1/plan.csv",
    "RunTrainRound2": "train1600/round2/plan.csv",
    "RunTrainRound3": "train1600/round3/plan.csv",
    "RunTrainRound0V3": "train1600_v3/round0/plan.csv",
    "RunTrainRound1V3": "train1600_v3/round1/plan.csv",
    "RunTrainRound2V3": "train1600_v3/round2/plan.csv",
    "RunCalibration200V3": "train1600_v3/calibration/plan.csv",
    "RunLockedValidation200V3": (
        "train1600_v3/locked_validation/plan.csv"
    ),
    "RunV4CompactCalibrationV1": "v4_compact_eval/calibration/plan.csv",
    "RunV4CompactLockedV1": "v4_compact_eval/locked/plan.csv",
    "RunExactClosedLoop": "exact_closed_loop/plan.csv",
    "RunSurrogateClosedLoop": "surrogate_closed_loop/plan.csv",
    "RunChallenge": "challenge/plan.csv",
    "RunFormalBlind": "formal_blind/plan.csv",
}

# Run stages whose plan rows are branches of an atomic multi-branch sample:
# ``Limit`` must count whole samples so a partial batch always runs every
# branch of each selected sample (never a stray branch).
RUN_STAGE_GROUP_KEYS = {
    "RunPeakBoundary": "sample_id",
    "RunPilot400": "sample_id",
    "RunPilotCoverageExtension": "sample_id",
    "RunPilotFlatAuxiliary": "sample_id",
    "RunPilotFeasibilityMap": "sample_id",
    "RunTrainRound0V3": "sample_id",
    "RunTrainRound1V3": "sample_id",
    "RunTrainRound2V3": "sample_id",
    "RunCalibration200V3": "sample_id",
    "RunLockedValidation200V3": "sample_id",
    "RunV4CompactCalibrationV1": "sample_id",
    "RunV4CompactLockedV1": "sample_id",
}

PREREQUISITES = {
    "BuildEventInventory": ("AuditContracts",),
    "PlanOpportunityPool": ("BuildEventInventory",),
    "ScanOpportunityPool": ("PlanOpportunityPool",),
    "BuildOpportunityPool": ("ScanOpportunityPool",),
    "AuditOpportunityCoverage": ("BuildOpportunityPool",),
    "BuildPeakCandidateCatalog": ("AuditOpportunityCoverage",),
    "PlanPeakBoundary": ("BuildPeakCandidateCatalog",),
    "AuditPeakBoundaryPreflight": ("PlanPeakBoundary",),
    "RunPeakBoundary": ("AuditPeakBoundaryPreflight",),
    "BuildPeakBoundaryPartial": ("PlanPeakBoundary",),
    "AuditPeakBoundaryPartial": ("BuildPeakBoundaryPartial",),
    "BuildPeakBoundaryDataset": ("RunPeakBoundary",),
    "AuditPeakBoundary": ("BuildPeakBoundaryDataset",),
    # The restamp gate deliberately depends on AuditContracts only: after a
    # code change every old status is stale under the new code SHA, so gating
    # on AuditPeakBoundary itself would deadlock the zero-SWMM revalidation.
    "RestampPeakBoundaryEvidence": ("AuditContracts",),
    "ClassifyExistingGate5R": ("AuditPeakBoundary",),
    "PlanPilot400": ("AuditPeakBoundary",),
    "AuditPilotPlan": ("PlanPilot400",),
    "AuditPilotPreflight": ("AuditPilotPlan",),
    "RunPilot400": ("AuditPilotPreflight",),
    "BuildPilotPartial": ("AuditPilotPlan",),
    "AuditPilotPartial": ("BuildPilotPartial",),
    "BuildPilotDataset": ("RunPilot400",),
    "AuditPilotDataset": ("BuildPilotDataset",),
    "TrainPilotBaselines": ("AuditPilotDataset",),
    "EvaluatePilotGate": ("TrainPilotBaselines",),
    # The gap audit anchors on AuditContracts only: after the extension code
    # change every pilot v1 status is stale under the new code SHA, and
    # AuditPilotDataset holds a frozen scientific_fail under Gate v1, so
    # gating on it would deadlock the read-only diagnosis.
    "AuditPilotCoverageGaps": ("AuditContracts",),
    "PlanPilotCoverageExtension": ("AuditPilotCoverageGaps",),
    "AuditPilotCoverageExtensionPlan": ("PlanPilotCoverageExtension",),
    "AuditPilotCoverageExtensionPreflight": (
        "AuditPilotCoverageExtensionPlan",
    ),
    "RunPilotCoverageExtension": ("AuditPilotCoverageExtensionPreflight",),
    "BuildPilotCoverageExtensionPartial": (
        "AuditPilotCoverageExtensionPlan",
    ),
    "AuditPilotCoverageExtensionPartial": (
        "BuildPilotCoverageExtensionPartial",
    ),
    "BuildPilotCoverageExtensionDataset": ("RunPilotCoverageExtension",),
    "AuditPilotCoverageExtensionDataset": (
        "BuildPilotCoverageExtensionDataset",
    ),
    "PlanPilotFlatAuxiliary": ("AuditPilotCoverageGaps",),
    "AuditPilotFlatAuxiliaryPreflight": ("PlanPilotFlatAuxiliary",),
    "RunPilotFlatAuxiliary": ("AuditPilotFlatAuxiliaryPreflight",),
    "BuildPilotFlatAuxiliaryDataset": ("RunPilotFlatAuxiliary",),
    "AuditPilotFlatAuxiliaryDataset": ("BuildPilotFlatAuxiliaryDataset",),
    "BuildPilotDatasetV2": ("AuditPilotCoverageExtensionDataset",),
    "AuditPilotDatasetV2": ("BuildPilotDatasetV2",),
    # Training gates on the dataset build, not the v2 audit verdict: the
    # audit is expected to hold a scientific_fail (joint/flat coverage) and
    # EvaluatePilotGateV2 absorbs that verdict via dataset_audit_v2_pass,
    # so gating training on audit exit 0 would deadlock the v2 chain.
    "TrainPilotBaselinesV2": ("BuildPilotDatasetV2",),
    "EvaluatePilotGateV2": ("TrainPilotBaselinesV2",),
    # The Gate P3 legacy audit anchors on AuditContracts only:
    # EvaluatePilotGateV2 holds a frozen scientific_fail (exit 5), so gating
    # the read-only compatibility scan on it would deadlock the P3 chain.
    "AuditLegacyOracleCompatibility": ("AuditContracts",),
    "PlanPilotFeasibilityMap": ("AuditLegacyOracleCompatibility",),
    "AuditPilotFeasibilityPreflight": ("PlanPilotFeasibilityMap",),
    "RunPilotFeasibilityMap": ("AuditPilotFeasibilityPreflight",),
    "BuildPilotFeasibilityPartial": ("PlanPilotFeasibilityMap",),
    "AuditPilotFeasibilityPartial": ("BuildPilotFeasibilityPartial",),
    "BuildPilotFeasibilityMap": ("RunPilotFeasibilityMap",),
    "AuditPilotFeasibilityMap": ("BuildPilotFeasibilityMap",),
    "BuildPilotDatasetV3": ("AuditPilotFeasibilityMap",),
    "AuditPilotDatasetV3": ("BuildPilotDatasetV3",),
    # Training gates on the dataset build, not the v3 audit verdict, for
    # the same anti-deadlock reason as the v2 chain: EvaluatePilotGateV3
    # absorbs the audit verdict via dataset_audit_v3_pass.
    "TrainPilotBaselinesV3": ("BuildPilotDatasetV3",),
    "EvaluatePilotGateV3": ("TrainPilotBaselinesV3",),
    # FreezeP3Evidence anchors on AuditContracts only: after this code
    # change every P3 stage status is stale under the new code SHA, and
    # EvaluatePilotGateV3 holds a frozen underpowered_validation verdict,
    # so gating the read-only freeze on it would deadlock the V3 chain.
    "FreezeP3Evidence": ("AuditContracts",),
    "AuditDataGenerationAuthorizationV3": ("FreezeP3Evidence",),
    "PlanTrain1600V3": ("AuditDataGenerationAuthorizationV3",),
    "AuditTrain1600PlanV3": ("PlanTrain1600V3",),
    "AuditTrainRound0PreflightV3": ("AuditTrain1600PlanV3",),
    "RunTrainRound0V3": ("AuditTrainRound0PreflightV3",),
    "BuildTrainRound0PartialV3": ("AuditTrain1600PlanV3",),
    "AuditTrainRound0PartialV3": ("BuildTrainRound0PartialV3",),
    "BuildTrainRound0V3": ("RunTrainRound0V3",),
    "AuditTrainRound0V3": ("BuildTrainRound0V3",),
    "TrainActiveLearner0V3": ("AuditTrainRound0V3",),
    "SelectTrainRound1V3": ("TrainActiveLearner0V3",),
    "RunTrainRound1V3": ("SelectTrainRound1V3",),
    "BuildTrainRound1PartialV3": ("SelectTrainRound1V3",),
    "AuditTrainRound1PartialV3": ("BuildTrainRound1PartialV3",),
    "BuildTrainRound1V3": ("RunTrainRound1V3",),
    "AuditTrainRound1V3": ("BuildTrainRound1V3",),
    "TrainActiveLearner1V3": ("AuditTrainRound1V3",),
    "SelectTrainRound2V3": ("TrainActiveLearner1V3",),
    "RunTrainRound2V3": ("SelectTrainRound2V3",),
    "BuildTrainRound2PartialV3": ("SelectTrainRound2V3",),
    "AuditTrainRound2PartialV3": ("BuildTrainRound2PartialV3",),
    "BuildTrainRound2V3": ("RunTrainRound2V3",),
    "AuditTrainRound2V3": ("BuildTrainRound2V3",),
    # Round 3 plans are frozen by PlanTrain1600V3 before any SWMM run;
    # the Run stages additionally wait for the completed Train rounds so
    # Calibration/Locked never interleave with Active Learning.
    "PlanCalibration200V3": ("AuditTrain1600PlanV3",),
    "RunCalibration200V3": (
        "PlanCalibration200V3",
        "AuditTrainRound2V3",
    ),
    "BuildCalibration200V3": ("RunCalibration200V3",),
    "AuditCalibration200V3": ("BuildCalibration200V3",),
    "PlanLockedValidation200V3": ("AuditTrain1600PlanV3",),
    "RunLockedValidation200V3": (
        "PlanLockedValidation200V3",
        "AuditTrainRound2V3",
    ),
    "BuildLockedValidation200V3": ("RunLockedValidation200V3",),
    "AuditLockedValidation200V3": ("BuildLockedValidation200V3",),
    "BuildTrain1600DatasetV3": (
        "AuditTrainRound2V3",
        "AuditCalibration200V3",
        "AuditLockedValidation200V3",
    ),
    "AuditTrain1600DatasetV3": ("BuildTrain1600DatasetV3",),
    "AuditTrain1600LearnabilityV4": ("FreezeTrain1600V3Evidence",),
    "AuditModelTrainingAuthorizationV4": (
        "AuditTrain1600LearnabilityV4",
    ),
    "TrainV4Baselines": ("AuditModelTrainingAuthorizationV4",),
    "EvaluateV4Baselines": ("TrainV4Baselines",),
    "TrainV4TrueState": ("EvaluateV4Baselines",),
    "CalibrateV4TrueState": ("TrainV4TrueState",),
    "EvaluateV4TrueStateLocked": ("CalibrateV4TrueState",),
    "AuditV4OfflineSafetyGate": ("EvaluateV4TrueStateLocked",),
    # V4.1 Phase-1: FreezeV4OfflineV0Evidence has no prerequisite chain -- the
    # V4.0 evidence lives under a previous code SHA, so the freeze re-verifies
    # it directly and re-stamps under the current SHA (like the Train1600
    # freeze).  Every downstream Train-only stage gates on the fresh freeze.
    "AuditV4LockedMetricComparabilityV0": ("FreezeV4OfflineV0Evidence",),
    "AuditV4GeneralizationFailureV0": ("FreezeV4OfflineV0Evidence",),
    "BuildV4FeatureBlockCatalogV1": ("FreezeV4OfflineV0Evidence",),
    "BuildV4LearningCurvesV1": ("BuildV4FeatureBlockCatalogV1",),
    "RunV4FeatureBlockAblationV1": ("BuildV4FeatureBlockCatalogV1",),
    "RunV4HeadArchitectureAblationV1": ("BuildV4FeatureBlockCatalogV1",),
    "AuditV4MultitaskGradientConflictV1": ("BuildV4FeatureBlockCatalogV1",),
    "SelectV4CompactModelV1": (
        "BuildV4LearningCurvesV1",
        "RunV4FeatureBlockAblationV1",
        "RunV4HeadArchitectureAblationV1",
        "AuditV4MultitaskGradientConflictV1",
    ),
    "TrainV4CompactTrueStateV1": ("SelectV4CompactModelV1",),
    # V4.1 Phase-2: the fresh evaluation split is planned/frozen from the
    # Reserve events after the compact model is trained; section-13 SWMM Build
    # stages depend on their Run stage, section 14 on the frozen model + fresh
    # Calibration, section 16 on section 14, and the gate audit on section 16.
    "PlanV4CompactCalibrationLockedV1": ("TrainV4CompactTrueStateV1",),
    "AuditV4CompactEvaluationPlanV1": ("PlanV4CompactCalibrationLockedV1",),
    "RunV4CompactCalibrationV1": ("AuditV4CompactEvaluationPlanV1",),
    "BuildV4CompactCalibrationV1": ("RunV4CompactCalibrationV1",),
    "AuditV4CompactCalibrationV1": ("BuildV4CompactCalibrationV1",),
    "RunV4CompactLockedV1": ("AuditV4CompactEvaluationPlanV1",),
    "BuildV4CompactLockedV1": ("RunV4CompactLockedV1",),
    "AuditV4CompactLockedV1": ("BuildV4CompactLockedV1",),
    "CalibrateV4CompactV1": (
        "TrainV4CompactTrueStateV1",
        "AuditV4CompactCalibrationV1",
    ),
    "EvaluateV4CompactLockedV1": (
        "CalibrateV4CompactV1",
        "AuditV4CompactLockedV1",
    ),
    "AuditV4PredictiveGeneralizationGateV1": ("EvaluateV4CompactLockedV1",),
    "PlanTrain1600": ("EvaluatePilotGate",),
    "AuditTrain1600Plan": ("PlanTrain1600",),
    "AuditTrainRound0Preflight": ("AuditTrain1600Plan",),
    "RunTrainRound0": ("AuditTrainRound0Preflight",),
    "BuildTrainRound0Partial": ("AuditTrain1600Plan",),
    "AuditTrainRound0Partial": ("BuildTrainRound0Partial",),
    "AuditTrainRound0": ("RunTrainRound0",),
    "TrainActiveLearner0": ("AuditTrainRound0",),
    "SelectTrainRound1": ("TrainActiveLearner0",),
    "AuditTrainRound1Preflight": ("SelectTrainRound1",),
    "RunTrainRound1": ("AuditTrainRound1Preflight",),
    "BuildTrainRound1Partial": ("SelectTrainRound1",),
    "AuditTrainRound1Partial": ("BuildTrainRound1Partial",),
    "AuditTrainRound1": ("RunTrainRound1",),
    "TrainActiveLearner1": ("AuditTrainRound1",),
    "SelectTrainRound2": ("TrainActiveLearner1",),
    "AuditTrainRound2Preflight": ("SelectTrainRound2",),
    "RunTrainRound2": ("AuditTrainRound2Preflight",),
    "BuildTrainRound2Partial": ("SelectTrainRound2",),
    "AuditTrainRound2Partial": ("BuildTrainRound2Partial",),
    "AuditTrainRound2": ("RunTrainRound2",),
    "TrainActiveLearner2": ("AuditTrainRound2",),
    "SelectTrainRound3": ("TrainActiveLearner2",),
    "AuditTrainRound3Preflight": ("SelectTrainRound3",),
    "RunTrainRound3": ("AuditTrainRound3Preflight",),
    "BuildTrainRound3Partial": ("SelectTrainRound3",),
    "AuditTrainRound3Partial": ("BuildTrainRound3Partial",),
    "AuditTrainRound3": ("RunTrainRound3",),
    "BuildTrain1600Dataset": ("AuditTrainRound3",),
    "AuditTrain1600Dataset": ("BuildTrain1600Dataset",),
    "TrainV4": ("AuditTrain1600Dataset",),
    "CalibrateV4": ("TrainV4",),
    "EvaluateV4Locked": ("CalibrateV4",),
    "AuditGATClosedLoopReadiness": (
        "AuditV4PredictiveGeneralizationGateV1",
    ),
    "PlanExactClosedLoop": (
        "AuditV4PredictiveGeneralizationGateV1",
        "AuditGATClosedLoopReadiness",
    ),
    "RunExactClosedLoop": ("PlanExactClosedLoop",),
    "AuditExactClosedLoop": ("RunExactClosedLoop",),
    "PlanSurrogateClosedLoop": ("AuditExactClosedLoop",),
    "RunSurrogateClosedLoop": ("PlanSurrogateClosedLoop",),
    "AuditSurrogateClosedLoop": ("RunSurrogateClosedLoop",),
    "LockPolicy": ("AuditSurrogateClosedLoop",),
    "RunChallenge": ("LockPolicy",),
    "AuditChallenge": ("RunChallenge",),
    "BuildFormalBlindInventory": ("AuditChallenge",),
    "RunFormalBlind": ("BuildFormalBlindInventory",),
    "AuditFormalBlind": ("RunFormalBlind",),
    "BuildPaperResults": ("AuditFormalBlind",),
    "BuildPaperFigures": ("BuildPaperResults",),
    "BuildPaperTables": ("BuildPaperResults",),
    "BuildReproducibilityBundle": (
        "BuildPaperResults",
        "BuildPaperFigures",
        "BuildPaperTables",
    ),
    # V4.2 stages
    "FreezeV41ScientificFailure": ("AuditContracts",),
    "AuditV41ClassificationMetricSemantics": ("FreezeV41ScientificFailure",),
    "BuildV42TrajectoryDataset": ("AuditV41ClassificationMetricSemantics",),
    "AuditV42TrajectoryDataset": ("BuildV42TrajectoryDataset",),
    "TrainV42WaterBalanceBaseline": ("AuditV42TrajectoryDataset",),
    "EvaluateV42WaterBalanceBaseline": ("TrainV42WaterBalanceBaseline",),
    "TrainV42TwinGraphDynamics": ("AuditV42TrajectoryDataset",),
    "RunV42ArchitectureAblation": ("TrainV42TwinGraphDynamics",),
    "RunV42StateScopeAblation": ("TrainV42TwinGraphDynamics",),
    "RunV42TargetAblation": ("TrainV42TwinGraphDynamics",),
    "BuildV42EventLearningCurve": ("TrainV42TwinGraphDynamics",),
    # V4.2 data pipeline prerequisites
    "BuildV42EventUsageLedger": ("FreezeV41ScientificFailure",),
    "AuditV42EventUsageLedger": ("BuildV42EventUsageLedger",),
    "BuildV42UnifiedDevelopmentPool": ("AuditV42EventUsageLedger",),
    "AuditV42UnifiedDevelopmentPool": ("BuildV42UnifiedDevelopmentPool",),
    "BuildV42DerivedSupervision": ("AuditV42UnifiedDevelopmentPool",),
    "AuditV42DerivedSupervision": ("BuildV42DerivedSupervision",),
    "PlanV42NestedGroupedCV": ("AuditV42DerivedSupervision",),
    "AuditV42NestedGroupedCVPlan": ("PlanV42NestedGroupedCV",),
    "RunV42NestedGroupedCV": ("AuditV42NestedGroupedCVPlan",),
    "BuildV42NestedGroupedCVResults": ("RunV42NestedGroupedCV",),
    "AuditV42NestedGroupedCVResults": ("BuildV42NestedGroupedCVResults",),
    # V4.2 validation prerequisites
    "AuditV42HeadActivation": ("TrainV42TwinGraphDynamics",),
    "AuditV42TargetMetricSemantics": ("TrainV42TwinGraphDynamics",),
    "AuditV42RankingPhysics": ("TrainV42TwinGraphDynamics",),
    "RunV42TinyOverfit": (
        "AuditV42HeadActivation",
        "AuditV42TargetMetricSemantics",
        "AuditV42RankingPhysics",
    ),
    # V4.2 fresh evaluation prerequisites
    "PlanV42FreshEvaluationSplit": ("AuditV42EventUsageLedger",),
    "AuditV42FreshEvaluationAvailability": ("PlanV42FreshEvaluationSplit",),
    "AuditV42TrainGroupedGeneralizationGate": (
        "AuditV42NestedGroupedCVResults",
        "RunV42TinyOverfit",
        "RunV42ArchitectureAblation",
        "RunV42StateScopeAblation",
        "RunV42TargetAblation",
        "BuildV42EventLearningCurve",
    ),
    # V4.2 final data pool prerequisites
    "AuditV42PrioritySentinelContract": [],
    "FreezeV42PriorityContract": ["AuditV42PrioritySentinelContract"],
    "BuildV42IndependentPfvOracle": ["FreezeV42PriorityContract"],
    "BuildV42SampleLineage": [],
    "AuditV42PhysicalDeduplication": ["BuildV42SampleLineage"],
    "BuildV42HistoricalSemanticInventory": [],
    "AuditV42HistoricalSemanticInventory": ["BuildV42HistoricalSemanticInventory"],
    "AuditV42DwfSources": [],
    "BuildV42Canonical13FrameTrajectories": [],
    "AuditV42Canonical13FrameTrajectories": ["BuildV42Canonical13FrameTrajectories"],
    "BuildV42TfvPeakOracle": ["FreezeV42PriorityContract"],
    "BuildV42SampleClassifier": [
        "AuditV42PhysicalDeduplication",
        "AuditV42HistoricalSemanticInventory",
        "AuditV42DwfSources",
        "AuditV42Canonical13FrameTrajectories",
        "BuildV42IndependentPfvOracle",
        "BuildV42TfvPeakOracle",
    ],
    "BuildV42FinalUnifiedDatasets": ["BuildV42SampleClassifier"],
    "AuditV42FinalUnifiedDatasets": ["BuildV42FinalUnifiedDatasets"],
    "BuildV42GroupedSplits": ["BuildV42FinalUnifiedDatasets"],
    "BuildV42PoolStatistics": ["BuildV42FinalUnifiedDatasets"],
    "AuditV42FinalDatasetAdmissionGate": [
        "BuildV42GroupedSplits",
        "BuildV42PoolStatistics",
        "AuditV42FinalUnifiedDatasets",
    ],
}


def prepare_output_tree(output_root: str | Path) -> None:
    root = Path(output_root)
    for directory in OUTPUT_DIRECTORIES:
        (root / directory).mkdir(parents=True, exist_ok=True)
    (root / "audits" / "stage_status").mkdir(parents=True, exist_ok=True)


def _contract_handler(
    project_root: Path, output_root: Path, config: dict
) -> Callable[[RuntimeOptions], StageResult]:
    def handler(_options: RuntimeOptions) -> StageResult:
        contract_rel = config.get("project", {}).get(
            "contract",
            "docs/contracts/PROJECT6_V4_FINAL_PIPELINE_CONTRACT.json",
        )
        contract_path = project_root / contract_rel
        if not contract_path.exists():
            return StageResult("AuditContracts", "blocked", EXIT_BLOCKED)
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        audit = audit_final_contract(contract, project_root)
        atomic_write_json(output_root / "audits" / "final_contract_audit.json", audit)
        passed = audit["status"] == "pass"
        return StageResult(
            "AuditContracts",
            audit["status"],
            EXIT_PASS if passed else EXIT_BLOCKED,
            completed=1 if passed else 0,
            remaining=0 if passed else 1,
            batch_complete=True,
            scope_complete=passed,
            evidence=audit,
        )

    return handler


def _artifact_handler(
    stage: str, output_root: Path
) -> Callable[[RuntimeOptions], StageResult]:
    def handler(options: RuntimeOptions) -> StageResult:
        relative = STAGE_ARTIFACTS.get(stage)
        artifact = output_root / relative if relative else None
        if stage in LONG_RUN_STAGES and options.dry_run:
            return StageResult(
                stage,
                "incomplete",
                EXIT_INCOMPLETE,
                remaining=1,
                batch_complete=False,
                scope_complete=False,
                evidence={
                    "dry_run": True,
                    "long_task_not_started": True,
                    "expected_artifact": str(artifact) if artifact else "",
                },
            )
        if artifact is not None and artifact.exists() and artifact.stat().st_size > 0:
            return StageResult(
                stage,
                "pass",
                EXIT_PASS,
                completed=1,
                remaining=0,
                batch_complete=True,
                scope_complete=True,
                evidence={"artifact": str(artifact)},
            )
        return StageResult(
            stage,
            "incomplete",
            EXIT_INCOMPLETE,
            completed=0,
            remaining=1,
            batch_complete=False,
            scope_complete=False,
            evidence={
                "reason": "required_artifact_missing",
                "expected_artifact": str(artifact) if artifact else "",
            },
        )

    return handler


def _run_case_stage_handler(
    stage: str, output_root: Path, config: dict
) -> Callable[[RuntimeOptions], StageResult]:
    def handler(options: RuntimeOptions) -> StageResult:
        plan_path = output_root / RUN_STAGE_PLANS[stage]
        if not plan_path.exists():
            return StageResult(
                stage,
                "incomplete",
                EXIT_INCOMPLETE,
                remaining=1,
                evidence={"reason": "executable plan missing", "plan": str(plan_path)},
            )
        plan = pd.read_csv(plan_path)
        required = {"case_id", "runner_function", "runner_kwargs"}
        missing = required - set(plan)
        if missing:
            return StageResult(
                stage,
                "blocked",
                EXIT_BLOCKED,
                remaining=len(plan),
                evidence={"reason": "plan is not executable", "missing": sorted(missing)},
            )
        input_sha = sha256_json(
            {
                "stage": stage,
                "plan_sha": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                "network_variant": config.get("runtime", {}).get("network_variant"),
            }
        )
        # Runs live beside the stage's run manifest artifact so that the
        # planning directory holds only the frozen plan files.
        run_root = (output_root / STAGE_ARTIFACTS[stage]).parent / "runs"
        result = run_parallel_cases(
            plan,
            run_root=run_root,
            worker=run_prepared_case,
            options=options,
            input_sha=input_sha,
            group_key=RUN_STAGE_GROUP_KEYS.get(stage),
            minimum_free_bytes=int(
                config.get("runtime", {}).get(
                    "minimum_free_disk_bytes", 1_000_000_000
                )
            ),
            minimum_free_memory_bytes=int(
                config.get("runtime", {}).get(
                    "minimum_free_memory_bytes", 1_000_000_000
                )
            ),
        )
        manifest = completion_manifest(run_root)
        artifact = output_root / STAGE_ARTIFACTS[stage]
        artifact.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(artifact, index=False)
        return result

    return handler


def _run_pilot400_handler(
    output_root: Path,
    config: dict,
    *,
    stage: str = "RunPilot400",
    branch_plan_rel: str = "pilot/planning/pilot_branch_plan.csv",
) -> Callable[[RuntimeOptions], StageResult]:
    """RunPilot400 sample tasks: one plan row is one four-branch sample.

    The picklable worker guarantees the three reference branches through the
    single-writer cache (at most 120 physical reference runs across the 40
    states) and then runs exactly one candidate, so ``Limit`` counts whole
    samples and the 1600 logical branches cost at most 520 SWMM runs.

    Extension run stages reuse this handler with their own ``stage`` and
    ``branch_plan_rel``; the reference root stays ``pilot`` so the v1
    reference cache is shared and never re-run.
    """

    def handler(options: RuntimeOptions) -> StageResult:
        plan_path = output_root / RUN_STAGE_PLANS[stage]
        branch_path = output_root / branch_plan_rel
        missing_inputs = [
            str(path)
            for path in (plan_path, branch_path)
            if not path.exists()
        ]
        if missing_inputs:
            return StageResult(
                stage,
                "incomplete",
                EXIT_INCOMPLETE,
                remaining=1,
                evidence={
                    "reason": "executable plan missing",
                    "missing_inputs": missing_inputs,
                },
            )
        plan = pd.read_csv(plan_path)
        required = {"case_id", "sample_id", "runner_function", "runner_kwargs"}
        missing_columns = required - set(plan)
        if missing_columns:
            return StageResult(
                stage,
                "blocked",
                EXIT_BLOCKED,
                remaining=len(plan),
                evidence={
                    "reason": "plan is not executable",
                    "missing": sorted(missing_columns),
                },
            )
        try:
            executable = attach_branch_context(
                plan,
                pd.read_csv(branch_path),
                reference_root=output_root / "pilot",
            )
        except ValueError as exc:
            return StageResult(
                stage, "blocked", EXIT_BLOCKED, evidence={"reason": str(exc)}
            )
        input_sha = sha256_json(
            {
                "stage": stage,
                "plan_sha": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                "branch_plan_sha": hashlib.sha256(
                    branch_path.read_bytes()
                ).hexdigest(),
                "network_variant": config.get("runtime", {}).get(
                    "network_variant"
                ),
            }
        )
        run_root = (output_root / STAGE_ARTIFACTS[stage]).parent / "runs"
        result = run_parallel_cases(
            executable,
            run_root=run_root,
            worker=run_pilot_sample,
            options=options,
            input_sha=input_sha,
            group_key=RUN_STAGE_GROUP_KEYS.get(stage),
            minimum_free_bytes=int(
                config.get("runtime", {}).get(
                    "minimum_free_disk_bytes", 1_000_000_000
                )
            ),
            minimum_free_memory_bytes=int(
                config.get("runtime", {}).get(
                    "minimum_free_memory_bytes", 1_000_000_000
                )
            ),
        )
        manifest = completion_manifest(run_root)
        artifact = output_root / STAGE_ARTIFACTS[stage]
        artifact.parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(artifact, index=False)
        return result

    return handler


def _partial_sources(stage: str, output_root: Path) -> tuple[Path, Path]:
    run_stage = PARTIAL_STAGE_RUN[stage]
    plan_path = output_root / RUN_STAGE_PLANS[run_stage]
    run_root = (output_root / STAGE_ARTIFACTS[run_stage]).parent / "runs"
    return plan_path, run_root


def _run_stage_sources(run_stage: str, output_root: Path) -> tuple[Path, Path]:
    """Resolve a run stage's frozen plan and its ``runs`` root directly.

    Unlike ``_partial_sources`` this takes the run stage name itself, so
    stage-specific partial builders can reach the RunPeakBoundary plan without
    routing through the ``PARTIAL_STAGE_RUN`` partial-name mapping.
    """
    plan_path = output_root / RUN_STAGE_PLANS[run_stage]
    run_root = (output_root / STAGE_ARTIFACTS[run_stage]).parent / "runs"
    return plan_path, run_root



def _build_partial_handler(
    stage: str, project_root: Path, output_root: Path, config: dict
) -> Callable[[RuntimeOptions], StageResult]:
    """Snapshot completed cases only; the formal manifests are untouched.

    Writes the six partial files under ``audits/partial/<stage>/<run_uuid>/``
    plus a ``latest.json`` pointer. Pending plan rows stay pending; the
    partial completion record always carries ``scope_complete=False``.
    """

    def handler(options: RuntimeOptions) -> StageResult:
        plan_path, run_root = _partial_sources(stage, output_root)
        if not plan_path.exists():
            return StageResult(
                stage,
                "blocked",
                EXIT_BLOCKED,
                evidence={"reason": "plan missing", "plan": str(plan_path)},
            )
        plan = pd.read_csv(plan_path)
        completions = (
            completion_manifest(run_root)
            if run_root.exists()
            else pd.DataFrame()
        )
        bundle = build_partial_bundle(plan, completions)
        run_uuid = uuid.uuid4().hex
        config_path = Path(options.config) if options.config else None
        config_sha = (
            hashlib.sha256(config_path.read_bytes()).hexdigest()
            if config_path is not None and config_path.exists()
            else ""
        )
        input_sha = sha256_json(
            {
                "stage": stage,
                "plan_sha": hashlib.sha256(
                    plan_path.read_bytes()
                ).hexdigest(),
                "run_root": str(run_root),
            }
        )
        accounting = partial_accounting(
            bundle,
            planned_scope_total=len(plan),
            run_uuid=run_uuid,
            input_sha=input_sha,
            config_sha=config_sha,
            code_sha=working_code_sha(project_root),
        )
        quality = audit_partial_quality(bundle)
        out_dir = output_root / "audits" / "partial" / stage / run_uuid
        out_dir.mkdir(parents=True, exist_ok=True)
        bundle["sample_manifest"].to_csv(
            out_dir / "partial_sample_manifest.csv", index=False
        )
        bundle["branch_manifest"].to_csv(
            out_dir / "partial_branch_manifest.csv", index=False
        )
        bundle["rejected"].to_csv(
            out_dir / "partial_rejected.csv", index=False
        )
        bundle["actual_duplicates"].to_csv(
            out_dir / "partial_actual_duplicates.csv", index=False
        )
        atomic_write_json(out_dir / "partial_quality_audit.json", quality)
        atomic_write_json(out_dir / "partial_completion.json", accounting)
        pointer = output_root / STAGE_ARTIFACTS[stage]
        pointer.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            pointer,
            {
                "stage": stage,
                "run_uuid": run_uuid,
                "directory": str(out_dir),
                "partial_only": True,
            },
        )
        return StageResult(
            stage,
            "pass",
            EXIT_PASS,
            completed=int(bundle["completed_total"]),
            remaining=int(len(bundle["pending"])),
            batch_complete=True,
            scope_complete=True,
            evidence={
                "partial_only": True,
                "full_gate_pass": False,
                "run_uuid": run_uuid,
                "accounting": accounting,
            },
        )

    return handler


def _audit_partial_handler(
    stage: str, output_root: Path, config: dict
) -> Callable[[RuntimeOptions], StageResult]:
    """Partial gate: recompute from source and apply the release level.

    A hard authenticity violation or stop condition among completed cases
    returns a non-zero exit and blocks further scale-up. Zero completed
    cases is ``incomplete``, never a failure. A pass here is explicitly
    ``partial_only`` and never a full-scope gate pass.
    """

    def handler(options: RuntimeOptions) -> StageResult:
        plan_path, run_root = _partial_sources(stage, output_root)
        if not plan_path.exists():
            return StageResult(
                stage,
                "blocked",
                EXIT_BLOCKED,
                evidence={"reason": "plan missing", "plan": str(plan_path)},
            )
        plan = pd.read_csv(plan_path)
        completions = (
            completion_manifest(run_root)
            if run_root.exists()
            else pd.DataFrame()
        )
        bundle = build_partial_bundle(plan, completions)
        completed = int(bundle["completed_total"])
        if completed == 0:
            return StageResult(
                stage,
                "incomplete",
                EXIT_INCOMPLETE,
                remaining=int(len(bundle["pending"])),
                batch_complete=False,
                scope_complete=False,
                evidence={
                    "reason": "no completed cases yet",
                    "partial_only": True,
                },
            )
        broken_pool = False
        if "error" in completions:
            broken_pool = bool(
                completions["error"]
                .astype(str)
                .str.contains("BrokenProcessPool", na=False)
                .any()
            )
        lock_conflicts = (
            len(list(run_root.glob("**/.reference.lock")))
            if run_root.exists()
            else 0
        )
        dead_zone = config.get("thresholds", {}).get("dead_zone", {})
        gate_config = {
            "noise_floor": {
                "delta_pfv_h120_vs_no_control": float(
                    dead_zone.get("pfv_m3", 0.0)
                ),
                "delta_tfv_h120_vs_dynamic_internal": float(
                    dead_zone.get("tfv_m3", 0.0)
                ),
                "delta_peak_h120_vs_dynamic_internal": float(
                    dead_zone.get("peak_m3s", 0.0)
                ),
            },
            "caps": config.get("partial", {}).get("caps", {}),
        }
        level = applicable_release_level(completed)
        gate = progressive_release_gate(
            bundle,
            level=level,
            evidence={
                "broken_process_pool": broken_pool,
                "reference_lock_conflicts": lock_conflicts,
            },
            config=gate_config,
        )
        artifact = output_root / STAGE_ARTIFACTS[stage]
        artifact.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(artifact, gate)
        passed = gate["status"] == "pass"
        return StageResult(
            stage,
            "pass" if passed else "scientific_fail",
            EXIT_PASS if passed else EXIT_SCIENTIFIC_FAIL,
            completed=completed,
            remaining=int(len(bundle["pending"])),
            batch_complete=True,
            scope_complete=passed,
            evidence={**gate, "stop_scale_up": not passed},
        )

    return handler


def _peak_partial_bundle(
    project_root: Path, output_root: Path, config: dict
) -> tuple[pd.DataFrame, dict]:
    """Read the Peak run plan/completions and reduce to a gate-ready bundle.

    Peak cases are four-branch samples, so the partial bundle is built by the
    same same-state four-branch reduction as the formal dataset instead of the
    generic single-row builder; a sample missing any branch stays pending.
    """
    plan_path, run_root = _run_stage_sources("RunPeakBoundary", output_root)
    plan = pd.read_csv(plan_path)
    completions = (
        completion_manifest(run_root) if run_root.exists() else pd.DataFrame()
    )
    project = config.get("project", {})
    bundle = build_peak_partial_bundle(
        plan,
        completions,
        priority_nodes=_read_facility_ids(
            project_root / str(project.get("priority_nodes", ""))
        ),
        facility_ids=_read_facility_ids(
            project_root / str(project.get("canonical_ids", ""))
        ),
        scientific_margin=config["thresholds"]["scientific_margin"],
        dead_zone=config["thresholds"]["dead_zone"],
    )
    return plan, bundle


def _build_peak_partial_handler(
    stage: str, project_root: Path, output_root: Path, config: dict
) -> Callable[[RuntimeOptions], StageResult]:
    """Snapshot completed four-branch Peak samples; formal manifests untouched."""

    def handler(options: RuntimeOptions) -> StageResult:
        plan_path, run_root = _run_stage_sources("RunPeakBoundary", output_root)
        if not plan_path.exists():
            return StageResult(
                stage,
                "blocked",
                EXIT_BLOCKED,
                evidence={"reason": "plan missing", "plan": str(plan_path)},
            )
        try:
            plan, bundle = _peak_partial_bundle(project_root, output_root, config)
        except (KeyError, OSError, ValueError) as exc:
            return StageResult(
                stage, "blocked", EXIT_BLOCKED, evidence={"reason": str(exc)}
            )
        run_uuid = uuid.uuid4().hex
        config_path = Path(options.config) if options.config else None
        config_sha = (
            hashlib.sha256(config_path.read_bytes()).hexdigest()
            if config_path is not None and config_path.exists()
            else ""
        )
        input_sha = sha256_json(
            {
                "stage": stage,
                "plan_sha": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                "run_root": str(run_root),
            }
        )
        accounting = partial_accounting(
            bundle,
            planned_scope_total=int(plan["sample_id"].nunique()),
            run_uuid=run_uuid,
            input_sha=input_sha,
            config_sha=config_sha,
            code_sha=working_code_sha(project_root),
        )
        quality = audit_partial_quality(bundle)
        out_dir = output_root / "audits" / "partial" / stage / run_uuid
        out_dir.mkdir(parents=True, exist_ok=True)
        bundle["sample_manifest"].to_csv(
            out_dir / "partial_sample_manifest.csv", index=False
        )
        bundle["branch_manifest"].to_csv(
            out_dir / "partial_branch_manifest.csv", index=False
        )
        bundle["rejected"].to_csv(
            out_dir / "partial_rejected.csv", index=False
        )
        bundle["actual_duplicates"].to_csv(
            out_dir / "partial_actual_duplicates.csv", index=False
        )
        atomic_write_json(out_dir / "partial_quality_audit.json", quality)
        atomic_write_json(out_dir / "partial_completion.json", accounting)
        pointer = output_root / STAGE_ARTIFACTS[stage]
        pointer.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            pointer,
            {
                "stage": stage,
                "run_uuid": run_uuid,
                "directory": str(out_dir),
                "partial_only": True,
            },
        )
        return StageResult(
            stage,
            "pass",
            EXIT_PASS,
            completed=int(bundle["completed_total"]),
            remaining=int(len(bundle["pending"])),
            batch_complete=True,
            scope_complete=True,
            evidence={
                "partial_only": True,
                "full_gate_pass": False,
                "run_uuid": run_uuid,
                "accounting": accounting,
            },
        )

    return handler


def _audit_peak_partial_handler(
    stage: str, project_root: Path, output_root: Path, config: dict
) -> Callable[[RuntimeOptions], StageResult]:
    """Peak partial gate over completed four-branch samples only."""

    def handler(_options: RuntimeOptions) -> StageResult:
        plan_path, run_root = _run_stage_sources("RunPeakBoundary", output_root)
        if not plan_path.exists():
            return StageResult(
                stage,
                "blocked",
                EXIT_BLOCKED,
                evidence={"reason": "plan missing", "plan": str(plan_path)},
            )
        try:
            _plan, bundle = _peak_partial_bundle(project_root, output_root, config)
        except (KeyError, OSError, ValueError) as exc:
            return StageResult(
                stage, "blocked", EXIT_BLOCKED, evidence={"reason": str(exc)}
            )
        completed = int(bundle["completed_total"])
        if completed == 0:
            return StageResult(
                stage,
                "incomplete",
                EXIT_INCOMPLETE,
                remaining=int(len(bundle["pending"])),
                batch_complete=False,
                scope_complete=False,
                evidence={
                    "reason": "no completed peak samples yet",
                    "partial_only": True,
                },
            )
        completions = (
            completion_manifest(run_root) if run_root.exists() else pd.DataFrame()
        )
        broken_pool = False
        if "error" in completions:
            broken_pool = bool(
                completions["error"]
                .astype(str)
                .str.contains("BrokenProcessPool", na=False)
                .any()
            )
        lock_conflicts = (
            len(list(run_root.glob("**/.reference.lock")))
            if run_root.exists()
            else 0
        )
        dead_zone = config.get("thresholds", {}).get("dead_zone", {})
        gate_config = {
            "noise_floor": {
                "delta_pfv_h120_vs_no_control": float(
                    dead_zone.get("pfv_m3", 0.0)
                ),
                "delta_tfv_h120_vs_dynamic_internal": float(
                    dead_zone.get("tfv_m3", 0.0)
                ),
                "delta_peak_h120_vs_dynamic_internal": float(
                    dead_zone.get("peak_m3s", 0.0)
                ),
            },
            "caps": config.get("partial", {}).get("caps", {}),
        }
        level = applicable_release_level(completed)
        gate = progressive_release_gate(
            bundle,
            level=level,
            evidence={
                "broken_process_pool": broken_pool,
                "reference_lock_conflicts": lock_conflicts,
            },
            config=gate_config,
        )
        artifact = output_root / STAGE_ARTIFACTS[stage]
        artifact.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(artifact, gate)
        passed = gate["status"] == "pass"
        return StageResult(
            stage,
            "pass" if passed else "scientific_fail",
            EXIT_PASS if passed else EXIT_SCIENTIFIC_FAIL,
            completed=completed,
            remaining=int(len(bundle["pending"])),
            batch_complete=True,
            scope_complete=passed,
            evidence={**gate, "stop_scale_up": not passed},
        )

    return handler


def _pilot_partial_bundle(
    project_root: Path,
    output_root: Path,
    config: dict,
    *,
    run_stage: str = "RunPilot400",
    branch_plan_rel: str = "pilot/planning/pilot_branch_plan.csv",
) -> tuple[pd.DataFrame, dict]:
    """Read the Pilot candidate plan/completions into a gate-ready bundle.

    Pilot samples are one completion per sample carrying four embedded
    branches; the sample-level completions are expanded to per-branch rows
    before the four-branch reduction so a sample missing any branch stays
    pending and reference branches are never counted as samples.
    """
    plan_path, run_root = _run_stage_sources(run_stage, output_root)
    plan = pd.read_csv(plan_path)
    branch_plan = pd.read_csv(output_root / branch_plan_rel)
    completions = (
        completion_manifest(run_root) if run_root.exists() else pd.DataFrame()
    )
    project = config.get("project", {})
    bundle = build_pilot_partial_bundle(
        plan,
        branch_plan,
        expand_pilot_completions(completions),
        priority_nodes=_read_facility_ids(
            project_root / str(project.get("priority_nodes", ""))
        ),
        facility_ids=_read_facility_ids(
            project_root / str(project.get("canonical_ids", ""))
        ),
        scientific_margin=config["thresholds"]["scientific_margin"],
        dead_zone=config["thresholds"]["dead_zone"],
    )
    return plan, bundle


def _build_pilot_partial_handler(
    stage: str,
    project_root: Path,
    output_root: Path,
    config: dict,
    *,
    run_stage: str = "RunPilot400",
    branch_plan_rel: str = "pilot/planning/pilot_branch_plan.csv",
) -> Callable[[RuntimeOptions], StageResult]:
    """Snapshot completed four-branch Pilot samples; formal manifests untouched."""

    def handler(options: RuntimeOptions) -> StageResult:
        plan_path, run_root = _run_stage_sources(run_stage, output_root)
        branch_path = output_root / branch_plan_rel
        if not plan_path.exists() or not branch_path.exists():
            return StageResult(
                stage,
                "blocked",
                EXIT_BLOCKED,
                evidence={
                    "reason": "pilot plan missing",
                    "plan": str(plan_path),
                    "branch_plan": str(branch_path),
                },
            )
        try:
            plan, bundle = _pilot_partial_bundle(
                project_root,
                output_root,
                config,
                run_stage=run_stage,
                branch_plan_rel=branch_plan_rel,
            )
        except (KeyError, OSError, ValueError) as exc:
            return StageResult(
                stage, "blocked", EXIT_BLOCKED, evidence={"reason": str(exc)}
            )
        run_uuid = uuid.uuid4().hex
        config_path = Path(options.config) if options.config else None
        config_sha = (
            hashlib.sha256(config_path.read_bytes()).hexdigest()
            if config_path is not None and config_path.exists()
            else ""
        )
        input_sha = sha256_json(
            {
                "stage": stage,
                "plan_sha": hashlib.sha256(plan_path.read_bytes()).hexdigest(),
                "run_root": str(run_root),
            }
        )
        accounting = partial_accounting(
            bundle,
            planned_scope_total=int(plan["sample_id"].nunique()),
            run_uuid=run_uuid,
            input_sha=input_sha,
            config_sha=config_sha,
            code_sha=working_code_sha(project_root),
        )
        quality = audit_partial_quality(bundle)
        out_dir = output_root / "audits" / "partial" / stage / run_uuid
        out_dir.mkdir(parents=True, exist_ok=True)
        bundle["sample_manifest"].to_csv(
            out_dir / "partial_sample_manifest.csv", index=False
        )
        bundle["branch_manifest"].to_csv(
            out_dir / "partial_branch_manifest.csv", index=False
        )
        bundle["rejected"].to_csv(
            out_dir / "partial_rejected.csv", index=False
        )
        bundle["actual_duplicates"].to_csv(
            out_dir / "partial_actual_duplicates.csv", index=False
        )
        atomic_write_json(out_dir / "partial_quality_audit.json", quality)
        atomic_write_json(out_dir / "partial_completion.json", accounting)
        pointer = output_root / STAGE_ARTIFACTS[stage]
        pointer.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            pointer,
            {
                "stage": stage,
                "run_uuid": run_uuid,
                "directory": str(out_dir),
                "partial_only": True,
            },
        )
        return StageResult(
            stage,
            "pass",
            EXIT_PASS,
            completed=int(bundle["completed_total"]),
            remaining=int(len(bundle["pending"])),
            batch_complete=True,
            scope_complete=True,
            evidence={
                "partial_only": True,
                "full_gate_pass": False,
                "run_uuid": run_uuid,
                "accounting": accounting,
            },
        )

    return handler


def _audit_pilot_partial_handler(
    stage: str,
    project_root: Path,
    output_root: Path,
    config: dict,
    *,
    run_stage: str = "RunPilot400",
    branch_plan_rel: str = "pilot/planning/pilot_branch_plan.csv",
) -> Callable[[RuntimeOptions], StageResult]:
    """Pilot partial gate over completed four-branch samples only."""

    def handler(_options: RuntimeOptions) -> StageResult:
        plan_path, run_root = _run_stage_sources(run_stage, output_root)
        if not plan_path.exists():
            return StageResult(
                stage,
                "blocked",
                EXIT_BLOCKED,
                evidence={"reason": "plan missing", "plan": str(plan_path)},
            )
        try:
            _plan, bundle = _pilot_partial_bundle(
                project_root,
                output_root,
                config,
                run_stage=run_stage,
                branch_plan_rel=branch_plan_rel,
            )
        except (KeyError, OSError, ValueError) as exc:
            return StageResult(
                stage, "blocked", EXIT_BLOCKED, evidence={"reason": str(exc)}
            )
        completed = int(bundle["completed_total"])
        if completed == 0:
            return StageResult(
                stage,
                "incomplete",
                EXIT_INCOMPLETE,
                remaining=int(len(bundle["pending"])),
                batch_complete=False,
                scope_complete=False,
                evidence={
                    "reason": "no completed pilot samples yet",
                    "partial_only": True,
                },
            )
        completions = (
            completion_manifest(run_root) if run_root.exists() else pd.DataFrame()
        )
        broken_pool = False
        if "error" in completions:
            broken_pool = bool(
                completions["error"]
                .astype(str)
                .str.contains("BrokenProcessPool", na=False)
                .any()
            )
        reference_root = output_root / "pilot" / "references"
        lock_conflicts = (
            len(list(reference_root.glob("**/.reference.lock")))
            + len(list(reference_root.glob("**/.writer.lock")))
            if reference_root.exists()
            else 0
        )
        dead_zone = config.get("thresholds", {}).get("dead_zone", {})
        gate_config = {
            "noise_floor": {
                "delta_pfv_h120_vs_no_control": float(
                    dead_zone.get("pfv_m3", 0.0)
                ),
                "delta_tfv_h120_vs_dynamic_internal": float(
                    dead_zone.get("tfv_m3", 0.0)
                ),
                "delta_peak_h120_vs_dynamic_internal": float(
                    dead_zone.get("peak_m3s", 0.0)
                ),
            },
            "caps": config.get("partial", {}).get("caps", {}),
        }
        level = applicable_release_level(completed)
        gate = progressive_release_gate(
            bundle,
            level=level,
            evidence={
                "broken_process_pool": broken_pool,
                "reference_lock_conflicts": lock_conflicts,
            },
            config=gate_config,
        )
        artifact = output_root / STAGE_ARTIFACTS[stage]
        artifact.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(artifact, gate)
        passed = gate["status"] == "pass"
        return StageResult(
            stage,
            "pass" if passed else "scientific_fail",
            EXIT_PASS if passed else EXIT_SCIENTIFIC_FAIL,
            completed=completed,
            remaining=int(len(bundle["pending"])),
            batch_complete=True,
            scope_complete=passed,
            evidence={**gate, "stop_scale_up": not passed},
        )

    return handler


def _preflight_handler(
    stage: str, project_root: Path, output_root: Path, config: dict
) -> Callable[[RuntimeOptions], StageResult]:
    """Fail-closed preflight for a real-SWMM run stage; blocked runs no case."""

    def handler(options: RuntimeOptions) -> StageResult:
        run_stage = PREFLIGHT_STAGE_RUN[stage]
        plan_path = output_root / RUN_STAGE_PLANS[run_stage]
        if not plan_path.exists():
            return StageResult(
                stage,
                "blocked",
                EXIT_BLOCKED,
                evidence={"reason": "plan missing", "plan": str(plan_path)},
            )
        plan = pd.read_csv(plan_path)
        run_root = (output_root / STAGE_ARTIFACTS[run_stage]).parent / "runs"
        input_sha = sha256_json(
            {
                "stage": run_stage,
                "plan_sha": hashlib.sha256(
                    plan_path.read_bytes()
                ).hexdigest(),
                "network_variant": config.get("runtime", {}).get(
                    "network_variant"
                ),
            }
        )
        writer_locks = (
            list(run_root.glob("**/*.lock")) if run_root.exists() else []
        )
        reference_locks = list(
            output_root.glob("**/.reference.lock")
        )
        heartbeat_root = output_root / "heartbeats"
        stale_after = time.time() - 600
        active_pids = [
            entry.name
            for entry in (
                heartbeat_root.iterdir() if heartbeat_root.exists() else []
            )
            if entry.is_file() and entry.stat().st_mtime >= stale_after
        ]
        report = preflight_checks(
            plan,
            workers=options.workers,
            output_root=output_root,
            input_sha=input_sha,
            evidence={
                "writer_lock_free": not writer_locks,
                "reference_cache_clean": not reference_locks,
                "active_conflicting_pids": active_pids,
            },
            minimum_free_bytes=int(
                config.get("runtime", {}).get(
                    "minimum_free_disk_bytes", 1_000_000_000
                )
            ),
            probe_torch=True,
        )
        artifact = output_root / STAGE_ARTIFACTS[stage]
        artifact.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(artifact, report)
        passed = report["status"] == "pass"
        return StageResult(
            stage,
            "pass" if passed else "blocked",
            EXIT_PASS if passed else EXIT_BLOCKED,
            completed=1 if passed else 0,
            remaining=0 if passed else 1,
            batch_complete=True,
            scope_complete=passed,
            evidence=report,
        )

    return handler


def _build_inventory_handler(
    project_root: Path, output_root: Path, config: dict
) -> Callable[[RuntimeOptions], StageResult]:
    def handler(options: RuntimeOptions) -> StageResult:
        catalog_path = project_root / config.get("project", {}).get(
            "event_catalog", ""
        )
        if not catalog_path.exists():
            return StageResult(
                "BuildEventInventory",
                "incomplete",
                EXIT_INCOMPLETE,
                remaining=1,
                evidence={"reason": "event_catalog_missing"},
            )
        revealed_path = project_root / config.get("project", {}).get(
            "revealed_event_registry", ""
        )
        revealed: set[str] = set()
        if revealed_path.exists():
            revealed_frame = pd.read_csv(revealed_path)
            if "event_id" in revealed_frame:
                revealed = set(revealed_frame["event_id"].astype(str))
        try:
            inventory = build_inventory_from_catalog(
                pd.read_csv(catalog_path),
                project_root=str(project_root),
                revealed_event_ids=revealed,
            )
        except (OSError, ValueError) as exc:
            return StageResult(
                "BuildEventInventory",
                "blocked",
                EXIT_BLOCKED,
                evidence={"reason": str(exc)},
            )
        output = output_root / STAGE_ARTIFACTS["BuildEventInventory"]
        output.parent.mkdir(parents=True, exist_ok=True)
        inventory.to_csv(output, index=False)
        return StageResult(
            "BuildEventInventory",
            "pass",
            EXIT_PASS,
            completed=len(inventory),
            remaining=0,
            batch_complete=True,
            scope_complete=True,
            evidence={
                "events": len(inventory),
                "eligible": int(inventory["eligible"].sum()),
            },
        )

    return handler


def _read_facility_ids(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith(("#", ";"))
    ]


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_ledger(path: Path) -> pd.DataFrame:
    ledger = pd.read_csv(path)
    missing = set(LEDGER_COLUMNS) - set(ledger)
    if missing:
        raise ValueError(f"ledger missing columns: {sorted(missing)}")
    for column in (*USAGE_FLAG_COLUMNS, "formal_eligible"):
        ledger[column] = ledger[column].fillna(False).astype(bool)
    for column in ("exclusion_reason", "assigned_split", "assignment_run_uuid"):
        ledger[column] = ledger[column].fillna("").astype(str)
    return ledger


def _merge_ledger_preserving_assignments(
    existing: pd.DataFrame, fresh: pd.DataFrame
) -> pd.DataFrame:
    """Rebuilds never wipe frozen splits or accumulated usage flags."""
    preserved = existing.set_index("event_id")
    merged = fresh.copy().set_index("event_id")
    carry_columns = [
        *USAGE_FLAG_COLUMNS,
        "formal_eligible",
        "exclusion_reason",
        "assigned_split",
        "assignment_run_uuid",
    ]
    shared = merged.index.intersection(preserved.index)
    for column in carry_columns:
        if column in ("opportunity_scanned",):
            # Scanned status only ever widens; never un-scan an event.
            merged.loc[shared, column] = (
                merged.loc[shared, column].astype(bool)
                | preserved.loc[shared, column].astype(bool)
            )
        elif column == "formal_eligible":
            merged.loc[shared, column] = (
                merged.loc[shared, column].astype(bool)
                & preserved.loc[shared, column].astype(bool)
            )
        else:
            merged.loc[shared, column] = preserved.loc[shared, column]
    return merged.reset_index()[list(LEDGER_COLUMNS)]


def _plan_opportunity_handler(
    project_root: Path, output_root: Path, config: dict
) -> Callable[[RuntimeOptions], StageResult]:
    def handler(_options: RuntimeOptions) -> StageResult:
        inventory_path = output_root / STAGE_ARTIFACTS["BuildEventInventory"]
        if not inventory_path.exists():
            return StageResult(
                "PlanOpportunityPool",
                "incomplete",
                EXIT_INCOMPLETE,
                remaining=1,
                evidence={"reason": "event_inventory_missing"},
            )
        try:
            plan = plan_opportunity_scans(
                pd.read_csv(inventory_path), config, project_root
            )
        except (KeyError, OSError, ValueError) as exc:
            return StageResult(
                "PlanOpportunityPool",
                "blocked",
                EXIT_BLOCKED,
                evidence={"reason": str(exc)},
            )
        target = output_root / STAGE_ARTIFACTS["PlanOpportunityPool"]
        target.parent.mkdir(parents=True, exist_ok=True)
        plan.to_csv(target, index=False)
        return StageResult(
            "PlanOpportunityPool",
            "pass",
            EXIT_PASS,
            completed=len(plan),
            remaining=0,
            batch_complete=True,
            scope_complete=True,
            evidence={
                "planned_events": int(plan["event_id"].nunique()),
                "allowed_runner_functions": sorted(
                    plan["runner_function"].unique().tolist()
                ),
                "plan": str(target),
            },
        )

    return handler


def _build_opportunity_handler(
    project_root: Path, output_root: Path, config: dict
) -> Callable[[RuntimeOptions], StageResult]:
    def handler(options: RuntimeOptions) -> StageResult:
        run_path = output_root / STAGE_ARTIFACTS["ScanOpportunityPool"]
        inventory_path = output_root / STAGE_ARTIFACTS["BuildEventInventory"]
        if not run_path.exists() or not inventory_path.exists():
            return StageResult(
                "BuildOpportunityPool",
                "incomplete",
                EXIT_INCOMPLETE,
                remaining=1,
                evidence={"reason": "run_manifest_or_inventory_missing"},
            )
        project = config.get("project", {})
        ids_path = project_root / str(project.get("canonical_ids", ""))
        semantics_path = project_root / str(
            project.get("facility_semantics", "")
        )
        network_path = project_root / str(project.get("network", ""))
        try:
            facility_ids = _read_facility_ids(ids_path)
            semantics = pd.read_csv(semantics_path)
            if not {"from_node", "to_node"}.issubset(semantics):
                from sewerrtc.simulation.action_policies import (
                    attach_reference_nodes,
                )

                semantics = attach_reference_nodes(
                    semantics.assign(
                        actuator_id=semantics.get(
                            "actuator_id", semantics["facility_id"]
                        )
                    ),
                    network_path,
                )
            pool, diagnostics = build_opportunity_pool(
                pd.read_csv(run_path),
                pd.read_csv(inventory_path),
                facility_ids=facility_ids,
                facility_semantics=semantics,
                responsive_threshold=float(
                    config.get("opportunity", {}).get(
                        "responsive_threshold", 0.25
                    )
                ),
                checkpoint_spacing_min=float(
                    config.get("opportunity", {}).get(
                        "minimum_checkpoint_spacing_min", 30
                    )
                ),
            )
            scan_status_path = (
                output_root
                / "audits"
                / "stage_status"
                / "ScanOpportunityPool.json"
            )
            try:
                source_run_uuid = str(
                    json.loads(
                        scan_status_path.read_text(encoding="utf-8")
                    ).get("run_uuid", "")
                )
            except (OSError, ValueError, TypeError):
                source_run_uuid = ""
            config_path = Path(options.config) if options.config else None
            config_sha = (
                _file_sha256(config_path)
                if config_path is not None and config_path.exists()
                else ""
            )
            catalogs = build_canonical_catalogs(
                pool,
                network_sha256=(
                    _file_sha256(network_path)
                    if network_path.is_file()
                    else ""
                ),
                config_sha256=config_sha,
                source_run_uuid=source_run_uuid,
            )
            ledger = build_event_usage_ledger(
                catalogs["event_tier_catalog"],
                scanned_event_ids=set(pool["event_id"].astype(str)),
            )
        except (KeyError, OSError, ValueError) as exc:
            return StageResult(
                "BuildOpportunityPool",
                "blocked",
                EXIT_BLOCKED,
                evidence={"reason": str(exc)},
            )
        target = output_root / STAGE_ARTIFACTS["BuildOpportunityPool"]
        opportunities = output_root / "opportunities"
        diagnostics_path = (
            opportunities / "opportunity_component_diagnostics.csv"
        )
        pool.to_csv(target, index=False)
        diagnostics.to_csv(diagnostics_path, index=False)
        # Canonical catalogs: the only permitted Pilot400/Train1600 sources.
        catalogs["event_tier_catalog"].to_csv(
            opportunities / "event_tier_catalog.csv", index=False
        )
        catalogs["standard_checkpoint_catalog"].to_csv(
            opportunities / "standard_checkpoint_catalog.csv", index=False
        )
        catalogs["short_event_checkpoint_catalog"].to_csv(
            opportunities / "short_event_checkpoint_catalog.csv", index=False
        )
        # Legacy non-canonical files must not survive as phantom sources.
        for legacy in (
            "pilot_checkpoint_catalog.csv",
            "train_checkpoint_catalog.csv",
        ):
            (opportunities / legacy).unlink(missing_ok=True)
        ledger_path = output_root / "inventory" / "event_usage_ledger.csv"
        if ledger_path.exists():
            try:
                ledger = _merge_ledger_preserving_assignments(
                    _load_ledger(ledger_path), ledger
                )
            except ValueError:
                pass  # A malformed old ledger never blocks the rebuild.
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ledger.to_csv(ledger_path, index=False)
        tier_events = pool.drop_duplicates("event_id")["event_tier"]
        return StageResult(
            "BuildOpportunityPool",
            "pass",
            EXIT_PASS,
            completed=len(pool),
            remaining=0,
            batch_complete=True,
            scope_complete=True,
            evidence={
                "checkpoints": int(len(pool)),
                "events": int(pool["event_id"].nunique()),
                "responsive": int(
                    pool["opportunity_class"].eq("responsive").sum()
                ),
                "low_opportunity": int(
                    pool["opportunity_class"].eq(
                        "low_opportunity"
                    ).sum()
                ),
                "event_tiers": {
                    tier: int(tier_events.eq(tier).sum())
                    for tier in (
                        "standard_4plus",
                        "short_3",
                        "short_2",
                        "ineligible",
                    )
                },
                "standard_checkpoint_rows": int(
                    len(catalogs["standard_checkpoint_catalog"])
                ),
                "standard_events": int(
                    catalogs["standard_checkpoint_catalog"][
                        "event_id"
                    ].nunique()
                ),
                "event_usage_ledger": str(ledger_path),
                "component_diagnostics": str(diagnostics_path),
            },
        )

    return handler


def _audit_opportunity_handler(
    output_root: Path, config: dict
) -> Callable[[RuntimeOptions], StageResult]:
    def handler(_options: RuntimeOptions) -> StageResult:
        source = output_root / STAGE_ARTIFACTS["BuildOpportunityPool"]
        if not source.exists():
            return StageResult(
                "AuditOpportunityCoverage",
                "incomplete",
                EXIT_INCOMPLETE,
                remaining=1,
            )
        pool = pd.read_csv(source)
        audit = audit_opportunity_coverage(pool, config)
        atomic_write_json(
            output_root / STAGE_ARTIFACTS["AuditOpportunityCoverage"],
            audit,
        )
        passed = audit["status"] == "pass"
        return StageResult(
            "AuditOpportunityCoverage",
            audit["status"],
            EXIT_PASS if passed else 5,
            completed=len(pool),
            remaining=0,
            batch_complete=True,
            scope_complete=passed,
            evidence=audit,
        )

    return handler


def _build_peak_catalog_handler(
    project_root: Path, output_root: Path, config: dict
) -> Callable[[RuntimeOptions], StageResult]:
    def handler(_options: RuntimeOptions) -> StageResult:
        source = output_root / STAGE_ARTIFACTS["BuildOpportunityPool"]
        if not source.exists():
            return StageResult(
                "BuildPeakCandidateCatalog",
                "incomplete",
                EXIT_INCOMPLETE,
                remaining=1,
            )
        project = config.get("project", {})
        try:
            facility_ids = _read_facility_ids(
                project_root / str(project.get("canonical_ids", ""))
            )
            semantics = pd.read_csv(
                project_root / str(project.get("facility_semantics", ""))
            )
            catalog = build_peak_candidate_catalog(
                pd.read_csv(source),
                facility_ids=facility_ids,
                facility_semantics=semantics,
                target_count=int(
                    config.get("peak_boundary", {}).get(
                        "degraded_samples_max", 60
                    )
                ),
            )
        except (KeyError, OSError, ValueError) as exc:
            return StageResult(
                "BuildPeakCandidateCatalog",
                "blocked",
                EXIT_BLOCKED,
                evidence={"reason": str(exc)},
            )
        target = output_root / STAGE_ARTIFACTS[
            "BuildPeakCandidateCatalog"
        ]
        catalog.to_csv(target, index=False)
        return StageResult(
            "BuildPeakCandidateCatalog",
            "pass",
            EXIT_PASS,
            completed=len(catalog),
            remaining=0,
            batch_complete=True,
            scope_complete=True,
            evidence={
                "candidate_count": int(len(catalog)),
                "events": int(catalog["event_id"].nunique()),
                "checkpoints": int(
                    catalog.groupby(["event_id", "checkpoint_id"]).ngroups
                ),
                "families": sorted(catalog["family"].unique().tolist()),
                "actual_peak_failures_claimed": 0,
                "requires_authoritative_peak_run": True,
            },
        )

    return handler


def _plan_peak_handler(output_root: Path) -> Callable[[RuntimeOptions], StageResult]:
    def handler(_options: RuntimeOptions) -> StageResult:
        source = output_root / "opportunities" / "peak_candidate_catalog.csv"
        if not source.exists():
            return StageResult(
                "PlanPeakBoundary",
                "incomplete",
                EXIT_INCOMPLETE,
                remaining=1,
                evidence={"reason": "peak_candidate_catalog_missing"},
            )
        try:
            plan = build_peak_boundary_plan(pd.read_csv(source))
        except ValueError as exc:
            return StageResult(
                "PlanPeakBoundary",
                "blocked",
                EXIT_BLOCKED,
                evidence={"reason": str(exc)},
            )
        target = output_root / STAGE_ARTIFACTS["PlanPeakBoundary"]
        target.parent.mkdir(parents=True, exist_ok=True)
        plan.to_csv(target, index=False)
        return StageResult(
            "PlanPeakBoundary",
            "pass",
            EXIT_PASS,
            completed=len(plan),
            batch_complete=True,
            scope_complete=True,
        )

    return handler


def _build_peak_dataset_handler(
    project_root: Path, output_root: Path, config: dict
) -> Callable[[RuntimeOptions], StageResult]:
    def handler(_options: RuntimeOptions) -> StageResult:
        source = output_root / STAGE_ARTIFACTS["RunPeakBoundary"]
        if not source.exists():
            return StageResult(
                "BuildPeakBoundaryDataset",
                "incomplete",
                EXIT_INCOMPLETE,
                remaining=1,
            )
        project = config.get("project", {})
        priority_path = project_root / str(
            project.get("priority_nodes", "")
        )
        try:
            samples, rejected = build_peak_boundary_dataset(
                pd.read_csv(source),
                priority_nodes=_read_facility_ids(priority_path),
                facility_ids=_read_facility_ids(
                    project_root / str(project.get("canonical_ids", ""))
                ),
                scientific_margin=config["thresholds"][
                    "scientific_margin"
                ],
                dead_zone=config["thresholds"]["dead_zone"],
            )
        except (KeyError, OSError, ValueError) as exc:
            return StageResult(
                "BuildPeakBoundaryDataset",
                "blocked",
                EXIT_BLOCKED,
                evidence={"reason": str(exc)},
            )
        target = output_root / STAGE_ARTIFACTS[
            "BuildPeakBoundaryDataset"
        ]
        rejected_path = (
            output_root / "peak_boundary" / "rejected_manifest.csv"
        )
        samples.to_csv(target, index=False)
        rejected.to_csv(rejected_path, index=False)
        # The accepted sample frame is the Peak-boundary anchor library that
        # PlanPilot400 consumes; the Opportunity stage never writes it.
        samples.to_csv(
            output_root / "peak_boundary" / "peak_boundary_anchor_library.csv",
            index=False,
        )
        if samples.empty:
            return StageResult(
                "BuildPeakBoundaryDataset",
                "incomplete",
                EXIT_INCOMPLETE,
                completed=0,
                remaining=int(
                    pd.read_csv(source)["sample_id"].nunique()
                ),
                batch_complete=True,
                scope_complete=False,
                evidence={
                    "reason": "no_same_state_complete_peak_samples",
                    "rejected": int(len(rejected)),
                },
            )
        return StageResult(
            "BuildPeakBoundaryDataset",
            "pass",
            EXIT_PASS,
            completed=len(samples),
            remaining=0,
            batch_complete=True,
            scope_complete=True,
            evidence={
                "accepted": int(len(samples)),
                "rejected": int(len(rejected)),
                "peak_degraded": int(
                    (~samples["peak_noninferior"].astype(bool)).sum()
                ),
                "pfv_safe_peak_hard_negative": int(
                    (
                        samples["pfv_safe"].astype(bool)
                        & ~samples["peak_noninferior"].astype(bool)
                    ).sum()
                ),
            },
        )

    return handler


def _audit_peak_handler(output_root: Path) -> Callable[[RuntimeOptions], StageResult]:
    def handler(_options: RuntimeOptions) -> StageResult:
        source = output_root / STAGE_ARTIFACTS[
            "BuildPeakBoundaryDataset"
        ]
        if not source.exists():
            return StageResult(
                "AuditPeakBoundary", "incomplete", EXIT_INCOMPLETE, remaining=1
            )
        audit = audit_peak_boundary(pd.read_csv(source))
        target = output_root / STAGE_ARTIFACTS["AuditPeakBoundary"]
        atomic_write_json(target, audit)
        passed = audit["status"] == "pass"
        if not passed:
            atomic_write_json(
                output_root
                / "peak_boundary"
                / "peak_constraint_binding_audit.json",
                peak_constraint_binding_audit(pd.read_csv(source)),
            )
        return StageResult(
            "AuditPeakBoundary",
            audit["status"],
            EXIT_PASS if passed else 5,
            completed=len(pd.read_csv(source)),
            batch_complete=True,
            scope_complete=passed,
            evidence=audit,
        )

    return handler


def _restamp_peak_handler(
    project_root: Path, output_root: Path, config: dict
) -> Callable[[RuntimeOptions], StageResult]:
    def handler(options: RuntimeOptions) -> StageResult:
        payload = restamp_peak_boundary_evidence(
            project_root,
            output_root,
            config,
            config_path=options.config or None,
        )
        status = str(payload.get("status", "blocked"))
        if status == "pass":
            return StageResult(
                "RestampPeakBoundaryEvidence",
                "pass",
                EXIT_PASS,
                completed=1,
                remaining=0,
                batch_complete=True,
                scope_complete=True,
                evidence=payload,
            )
        if status == "scientific_fail":
            return StageResult(
                "RestampPeakBoundaryEvidence",
                "scientific_fail",
                EXIT_SCIENTIFIC_FAIL,
                completed=0,
                remaining=1,
                batch_complete=True,
                scope_complete=False,
                evidence=payload,
            )
        return StageResult(
            "RestampPeakBoundaryEvidence",
            "blocked",
            EXIT_BLOCKED,
            completed=0,
            remaining=1,
            evidence=payload,
        )

    return handler


def _classify_gate5r_handler(
    output_root: Path,
) -> Callable[[RuntimeOptions], StageResult]:
    def handler(_options: RuntimeOptions) -> StageResult:
        samples_path = output_root / STAGE_ARTIFACTS["BuildPeakBoundaryDataset"]
        ledger_path = output_root / "inventory" / "event_usage_ledger.csv"
        if not samples_path.exists() or not ledger_path.exists():
            return StageResult(
                "ClassifyExistingGate5R",
                "incomplete",
                EXIT_INCOMPLETE,
                remaining=1,
                evidence={"reason": "peak_samples_or_ledger_missing"},
            )
        try:
            samples = pd.read_csv(samples_path)
            ledger = _load_ledger(ledger_path)
        except (OSError, ValueError) as exc:
            return StageResult(
                "ClassifyExistingGate5R",
                "blocked",
                EXIT_BLOCKED,
                evidence={"reason": str(exc)},
            )
        tuned_events = sorted(samples["event_id"].astype(str).unique())
        classification = (
            samples.drop_duplicates("event_id")[
                [
                    column
                    for column in ("event_id", "rainfall_sha256")
                    if column in samples
                ]
            ]
            .copy()
            .reset_index(drop=True)
        )
        classification["used_gate5r"] = True
        classification["used_peak_boundary"] = True
        classification["classification"] = "gate_tuning_development_only"
        target = output_root / STAGE_ARTIFACTS["ClassifyExistingGate5R"]
        target.parent.mkdir(parents=True, exist_ok=True)
        classification.to_csv(target, index=False)
        # Gate-tuning events are frozen out of Train/Calibration/Validation.
        tuned_mask = ledger["event_id"].astype(str).isin(tuned_events)
        ledger.loc[tuned_mask, "used_gate5r"] = True
        ledger.loc[tuned_mask, "used_peak_boundary"] = True
        ledger.loc[tuned_mask, "policy_tuned_on_event"] = True
        ledger.to_csv(ledger_path, index=False)
        return StageResult(
            "ClassifyExistingGate5R",
            "pass",
            EXIT_PASS,
            completed=len(classification),
            batch_complete=True,
            scope_complete=True,
            evidence={
                "gate_tuning_events": tuned_events,
                "classification": str(target),
            },
        )

    return handler


def _plan_pilot_handler(
    project_root: Path, output_root: Path, config: dict
) -> Callable[[RuntimeOptions], StageResult]:
    def handler(options: RuntimeOptions) -> StageResult:
        peak_audit = output_root / STAGE_ARTIFACTS["AuditPeakBoundary"]
        if not peak_audit.exists() or json.loads(
            peak_audit.read_text(encoding="utf-8")
        ).get("status") != "pass":
            return StageResult(
                "PlanPilot400",
                "blocked",
                EXIT_BLOCKED,
                evidence={"reason": "Peak Boundary gate must pass"},
            )
        inputs = {
            "standard_checkpoint_catalog": output_root
            / "opportunities"
            / "standard_checkpoint_catalog.csv",
            "event_usage_ledger": output_root
            / "inventory"
            / "event_usage_ledger.csv",
            "peak_boundary_anchor_library": output_root
            / "peak_boundary"
            / "peak_boundary_anchor_library.csv",
            "existing_gate5r_classification": output_root
            / STAGE_ARTIFACTS["ClassifyExistingGate5R"],
        }
        missing = sorted(
            name for name, path in inputs.items() if not path.exists()
        )
        if missing:
            return StageResult(
                "PlanPilot400",
                "incomplete",
                EXIT_INCOMPLETE,
                evidence={
                    "reason": "canonical_inputs_missing",
                    "missing_inputs": missing,
                },
            )
        run_uuid = str(uuid.uuid4())
        try:
            ledger = _load_ledger(inputs["event_usage_ledger"])
            bundle = build_pilot_planning_bundle(
                pd.read_csv(inputs["standard_checkpoint_catalog"]),
                ledger,
                peak_anchor_library=pd.read_csv(
                    inputs["peak_boundary_anchor_library"]
                ),
                gate5r_classification=pd.read_csv(
                    inputs["existing_gate5r_classification"]
                ),
                count=int(config.get("pilot400", {}).get("events", 8)),
            )
            ledger = assign_split(
                ledger,
                bundle["selected_events"],
                "pilot",
                assignment_run_uuid=run_uuid,
            )
        except ValueError as exc:
            return StageResult(
                "PlanPilot400",
                "blocked",
                EXIT_BLOCKED,
                evidence={"reason": str(exc)},
            )
        planning = output_root / "pilot" / "planning"
        planning.mkdir(parents=True, exist_ok=True)
        for name in (
            "pilot_event_selection",
            "pilot_checkpoint_catalog",
            "pilot_candidate_plan",
            "pilot_reference_plan",
            "pilot_candidate_coverage",
            "pilot_split_manifest",
        ):
            bundle[name].to_csv(planning / f"{name}.csv", index=False)
        atomic_write_json(
            planning / "pilot_plan_audit.json", bundle["pilot_plan_audit"]
        )
        project = config.get("project", {})
        contract_path = project_root / str(project.get("contract", ""))
        config_path = Path(options.config) if options.config else None
        contract_sha = (
            _file_sha256(contract_path) if contract_path.exists() else ""
        )
        config_sha = (
            _file_sha256(config_path)
            if config_path is not None and config_path.exists()
            else ""
        )
        peak_anchor_library = pd.read_csv(
            inputs["peak_boundary_anchor_library"]
        )
        try:
            role_plan = build_pilot_role_plan(
                bundle["pilot_checkpoint_catalog"]
            )
            candidate_plan, coverage_missing = materialize_pilot_candidates(
                role_plan,
                bundle["pilot_checkpoint_catalog"],
                facility_ids=_read_facility_ids(
                    project_root / str(project.get("canonical_ids", ""))
                ),
                facility_semantics=pd.read_csv(
                    project_root / str(project.get("facility_semantics", ""))
                ),
                peak_boundary_anchor_library=peak_anchor_library,
                contract_sha256=contract_sha,
                config_sha256=config_sha,
                code_sha256=working_code_sha(project_root),
                schedule_dir=planning / "schedules",
                schedule_dir_relative_to=output_root,
            )
            branch_plan = build_pilot_branch_plan(
                candidate_plan, contract_sha256=contract_sha
            )
        except (KeyError, OSError, ValueError) as exc:
            return StageResult(
                "PlanPilot400",
                "blocked",
                EXIT_BLOCKED,
                evidence={"reason": str(exc)},
            )
        if coverage_missing.empty:
            coverage_missing = pd.DataFrame(
                columns=[
                    "event_id",
                    "checkpoint_id",
                    "candidate_role",
                    "split",
                    "case_id",
                    "reason",
                ]
            )
        materialized_audit = audit_pilot_materialized_plan(
            role_plan,
            candidate_plan,
            branch_plan,
            coverage_missing,
            peak_tuned_event_ids=set(
                peak_anchor_library.get(
                    "event_id", pd.Series(dtype=str)
                ).astype(str)
            ),
        )
        role_plan.to_csv(planning / "pilot_role_plan.csv", index=False)
        candidate_plan.to_csv(
            planning / "pilot_candidate_plan.csv", index=False
        )
        branch_plan.to_csv(planning / "pilot_branch_plan.csv", index=False)
        coverage_missing.to_csv(
            planning / "pilot_coverage_missing.csv", index=False
        )
        atomic_write_json(
            planning / "pilot_materialized_plan_audit.json",
            materialized_audit,
        )
        ledger.to_csv(inputs["event_usage_ledger"], index=False)
        passed = (
            bundle["pilot_plan_audit"]["status"] == "pass"
            and materialized_audit["status"] == "pass"
        )
        atomic_write_json(
            planning / "completion.json",
            {
                "stage": "PlanPilot400",
                "run_uuid": run_uuid,
                "input_sha256": {
                    name: _file_sha256(path)
                    for name, path in inputs.items()
                },
                "selected_events": list(bundle["selected_events"]),
                "materialized_status": materialized_audit["status"],
                "candidate_rows": int(len(candidate_plan)),
                "branch_rows": int(len(branch_plan)),
                "coverage_missing_rows": int(len(coverage_missing)),
                "status": "pass" if passed else "blocked",
            },
        )
        return StageResult(
            "PlanPilot400",
            "pass" if passed else "blocked",
            EXIT_PASS if passed else EXIT_BLOCKED,
            completed=len(candidate_plan),
            batch_complete=True,
            scope_complete=passed,
            evidence={
                "selected_events": list(bundle["selected_events"]),
                "audit": bundle["pilot_plan_audit"],
                "materialized_audit_status": materialized_audit["status"],
                "candidate_rows": int(len(candidate_plan)),
                "branch_rows": int(len(branch_plan)),
            },
        )

    return handler


def _audit_pilot_plan_handler(
    output_root: Path,
) -> Callable[[RuntimeOptions], StageResult]:
    """Audit the materialized three-layer pilot plan (spec section IX)."""

    def handler(_options: RuntimeOptions) -> StageResult:
        planning = output_root / "pilot" / "planning"
        names = (
            "pilot_role_plan",
            "pilot_candidate_plan",
            "pilot_branch_plan",
            "pilot_coverage_missing",
        )
        missing = sorted(
            name
            for name in names
            if not (planning / f"{name}.csv").exists()
        )
        if missing:
            return StageResult(
                "AuditPilotPlan",
                "incomplete",
                EXIT_INCOMPLETE,
                remaining=1,
                evidence={"missing_inputs": missing},
            )
        role_plan = pd.read_csv(planning / "pilot_role_plan.csv")
        candidate_plan = pd.read_csv(planning / "pilot_candidate_plan.csv")
        branch_plan = pd.read_csv(planning / "pilot_branch_plan.csv")
        try:
            coverage_missing = pd.read_csv(
                planning / "pilot_coverage_missing.csv"
            )
        except pd.errors.EmptyDataError:
            coverage_missing = pd.DataFrame()
        anchor_path = (
            output_root / "peak_boundary" / "peak_boundary_anchor_library.csv"
        )
        peak_events: set[str] = set()
        if anchor_path.exists():
            anchors = pd.read_csv(anchor_path)
            peak_events = {
                str(item)
                for item in anchors.get(
                    "event_id", pd.Series(dtype=str)
                ).astype(str)
            }
        audit = audit_pilot_materialized_plan(
            role_plan,
            candidate_plan,
            branch_plan,
            coverage_missing,
            peak_tuned_event_ids=peak_events,
        )
        atomic_write_json(output_root / STAGE_ARTIFACTS["AuditPilotPlan"], audit)
        passed = audit["status"] == "pass"
        exit_code = EXIT_PASS
        if not passed:
            exit_code = (
                EXIT_SCIENTIFIC_FAIL
                if audit["status"] == "scientific_fail"
                else EXIT_BLOCKED
            )
        return StageResult(
            "AuditPilotPlan",
            audit["status"],
            exit_code,
            completed=len(candidate_plan),
            batch_complete=True,
            scope_complete=passed,
            evidence=audit,
        )

    return handler


def _plan_train_handler(
    project_root: Path, output_root: Path, config: dict
) -> Callable[[RuntimeOptions], StageResult]:
    def handler(options: RuntimeOptions) -> StageResult:
        train_root = output_root / "train1600"
        planning = train_root / "planning"
        catalog_path = planning / "train_checkpoint_catalog.csv"
        completion_path = planning / "completion.json"
        # 1. Pilot gate fail-closed: no verdict, no Train plan.
        verdict_path = output_root / STAGE_ARTIFACTS["EvaluatePilotGate"]
        if not verdict_path.exists():
            return StageResult(
                "PlanTrain1600",
                "blocked",
                EXIT_BLOCKED,
                evidence={"reason": "pilot_gate_verdict_missing"},
            )
        try:
            verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
        except ValueError:
            verdict = {}
        if (
            verdict.get("scientific_pass") is not True
            or int(verdict.get("exit_code", -1)) != 0
        ):
            return StageResult(
                "PlanTrain1600",
                "blocked",
                EXIT_BLOCKED,
                evidence={
                    "reason": "pilot_gate_not_passed",
                    "scientific_pass": bool(
                        verdict.get("scientific_pass", False)
                    ),
                    "exit_code": int(verdict.get("exit_code", -1)),
                },
            )
        # 2. Canonical inputs; a missing file is never papered over.
        standard_path = (
            output_root / "opportunities" / "standard_checkpoint_catalog.csv"
        )
        ledger_path = output_root / "inventory" / "event_usage_ledger.csv"
        missing = [
            str(path)
            for path in (standard_path, ledger_path)
            if not path.exists()
        ]
        if missing:
            return StageResult(
                "PlanTrain1600",
                "incomplete",
                EXIT_INCOMPLETE,
                evidence={
                    "reason": "canonical_inputs_missing",
                    "missing_inputs": missing,
                },
            )
        config_path = Path(options.config) if options.config else None
        input_identity = {
            "standard_catalog_sha256": _file_sha256(standard_path),
            "ledger_sha256": _file_sha256(ledger_path),
            "pilot_gate_sha256": _file_sha256(verdict_path),
            "config_sha256": (
                _file_sha256(config_path)
                if config_path is not None and config_path.exists()
                else ""
            ),
            "code_sha256": working_code_sha(project_root),
        }
        identity_sha = sha256_json(input_identity)
        # 3. Resume only on identical SHAs; stale plans fail closed.
        if catalog_path.exists():
            try:
                completion = json.loads(
                    completion_path.read_text(encoding="utf-8")
                )
            except (OSError, ValueError, TypeError):
                completion = {}
            if completion.get("input_identity_sha256") == identity_sha:
                catalog = pd.read_csv(catalog_path)
                return StageResult(
                    "PlanTrain1600",
                    "pass",
                    EXIT_PASS,
                    completed=len(catalog),
                    batch_complete=True,
                    scope_complete=len(catalog) == 320,
                    evidence={
                        "resumed": True,
                        "input_identity_sha256": identity_sha,
                    },
                )
            return StageResult(
                "PlanTrain1600",
                "blocked",
                EXIT_BLOCKED,
                evidence={
                    "reason": "stale_train_plan_detected",
                    "expected_input_identity_sha256": identity_sha,
                    "found_input_identity_sha256": completion.get(
                        "input_identity_sha256", ""
                    ),
                    "action_required": (
                        "archive train1600/planning before replanning; "
                        "stale plans are never silently reused"
                    ),
                },
            )
        run_uuid = str(uuid.uuid4())
        counts = {
            key: int(value)
            for key, value in config.get("train1600", {})
            .get(
                "split",
                {
                    "train": 48,
                    "calibration": 8,
                    "locked_validation": 8,
                    "reserve": 16,
                },
            )
            .items()
        }
        try:
            standard = pd.read_csv(standard_path)
            ledger = _load_ledger(ledger_path)
            # 4. Select and freeze the event split, then derive catalogs.
            selection = select_train1600_events(
                standard, ledger, counts=counts
            )
            train_catalog, reserve_catalog = build_train_checkpoint_catalog(
                standard, selection
            )
            target_plan = build_train1600_target_plan(train_catalog)
            round0_plan = build_round0_plan(train_catalog)
            for split in ("train", "calibration", "locked_validation", "reserve"):
                ledger = assign_split(
                    ledger,
                    selection[split],
                    split,
                    assignment_run_uuid=run_uuid,
                )
        except EventShortfallError as exc:
            planning.mkdir(parents=True, exist_ok=True)
            atomic_write_json(
                planning / "event_shortfall_report.json", exc.report
            )
            return StageResult(
                "PlanTrain1600",
                "blocked",
                EXIT_BLOCKED,
                evidence={
                    "reason": "standard_event_shortfall",
                    "report": str(planning / "event_shortfall_report.json"),
                    "shortfall": exc.report["shortfall"],
                },
            )
        except ValueError as exc:
            return StageResult(
                "PlanTrain1600", "blocked", EXIT_BLOCKED, evidence={"reason": str(exc)}
            )
        # 5. Persist the frozen plan, then the atomic completion marker.
        planning.mkdir(parents=True, exist_ok=True)
        train_catalog.to_csv(catalog_path, index=False)
        reserve_catalog.to_csv(
            planning / "reserve_checkpoint_catalog.csv", index=False
        )
        target_plan.to_csv(
            planning / "train1600_target_plan.csv", index=False
        )
        round0_dir = train_root / "round0"
        round0_dir.mkdir(parents=True, exist_ok=True)
        round0_plan.to_csv(round0_dir / "plan.csv", index=False)
        ledger.to_csv(ledger_path, index=False)
        # This stage itself freezes split assignments into the ledger, so the
        # resume identity must reflect the post-assignment ledger bytes.
        input_identity["ledger_sha256"] = _file_sha256(ledger_path)
        identity_sha = sha256_json(input_identity)
        atomic_write_json(
            planning / "event_selection.json",
            {split: list(events) for split, events in selection.items()},
        )
        atomic_write_json(
            completion_path,
            {
                "stage": "PlanTrain1600",
                "run_uuid": run_uuid,
                "input_identity": input_identity,
                "input_identity_sha256": identity_sha,
                "rows": {
                    "train_checkpoint_catalog": int(len(train_catalog)),
                    "reserve_checkpoint_catalog": int(len(reserve_catalog)),
                    "train1600_target_plan": int(len(target_plan)),
                    "round0_plan": int(len(round0_plan)),
                },
            },
        )
        return StageResult(
            "PlanTrain1600",
            "pass",
            EXIT_PASS,
            completed=len(train_catalog),
            batch_complete=True,
            scope_complete=len(train_catalog) == 320,
            evidence={
                "train_checkpoint_catalog": str(catalog_path),
                "train_rows": int(len(train_catalog)),
                "reserve_rows": int(len(reserve_catalog)),
                "target_plan_rows": int(len(target_plan)),
                "round0_rows": int(len(round0_plan)),
                "input_identity_sha256": identity_sha,
            },
        )

    return handler


def _audit_train_plan_handler(
    output_root: Path,
) -> Callable[[RuntimeOptions], StageResult]:
    def handler(_options: RuntimeOptions) -> StageResult:
        planning = output_root / "train1600" / "planning"
        catalog_path = planning / "train_checkpoint_catalog.csv"
        reserve_path = planning / "reserve_checkpoint_catalog.csv"
        selection_path = planning / "event_selection.json"
        missing = [
            str(path)
            for path in (catalog_path, reserve_path, selection_path)
            if not path.exists()
        ]
        if missing:
            return StageResult(
                "AuditTrain1600Plan",
                "incomplete",
                EXIT_INCOMPLETE,
                remaining=1,
                evidence={"missing_inputs": missing},
            )
        audit = audit_train1600_plan(
            pd.read_csv(catalog_path),
            pd.read_csv(reserve_path),
            json.loads(selection_path.read_text(encoding="utf-8")),
        )
        atomic_write_json(
            output_root / STAGE_ARTIFACTS["AuditTrain1600Plan"], audit
        )
        passed = audit["status"] == "pass"
        return StageResult(
            "AuditTrain1600Plan",
            audit["status"],
            EXIT_PASS if passed else EXIT_BLOCKED,
            completed=int(len(pd.read_csv(catalog_path))),
            batch_complete=True,
            scope_complete=passed,
            evidence=audit,
        )

    return handler


def _evaluate_pilot_gate_handler(
    output_root: Path,
) -> Callable[[RuntimeOptions], StageResult]:
    def handler(_options: RuntimeOptions) -> StageResult:
        dataset_audit_path = output_root / STAGE_ARTIFACTS["AuditPilotDataset"]
        baseline_path = output_root / STAGE_ARTIFACTS["TrainPilotBaselines"]
        missing = [
            str(path)
            for path in (dataset_audit_path, baseline_path)
            if not path.exists()
        ]
        if missing:
            return StageResult(
                "EvaluatePilotGate",
                "incomplete",
                EXIT_INCOMPLETE,
                remaining=1,
                evidence={"missing_inputs": missing},
            )
        verdict = evaluate_pilot_gate(
            json.loads(dataset_audit_path.read_text(encoding="utf-8")),
            json.loads(baseline_path.read_text(encoding="utf-8")),
        )
        atomic_write_json(
            output_root / STAGE_ARTIFACTS["EvaluatePilotGate"], verdict
        )
        passed = bool(verdict["scientific_pass"])
        return StageResult(
            "EvaluatePilotGate",
            verdict["status"],
            EXIT_PASS if passed else EXIT_SCIENTIFIC_FAIL,
            completed=1 if passed else 0,
            batch_complete=True,
            scope_complete=passed,
            evidence=verdict,
        )

    return handler


def _build_dataset_handler(
    stage: str, output_root: Path, scope: str
) -> Callable[[RuntimeOptions], StageResult]:
    """Shared Pilot/Train1600 dataset builder over branch run manifests."""

    def handler(_options: RuntimeOptions) -> StageResult:
        prefix = "pilot" if scope == "pilot" else "train1600"
        if scope == "pilot":
            manifest_paths = [output_root / STAGE_ARTIFACTS["RunPilot400"]]
            planned_path = (
                output_root / "pilot" / "planning" / "pilot_candidate_plan.csv"
            )
        else:
            manifest_paths = [
                output_root / f"train1600/round{index}/run_manifest.csv"
                for index in range(4)
            ]
            planned_path = (
                output_root
                / "train1600"
                / "planning"
                / "train1600_target_plan.csv"
            )
        missing = [
            str(path)
            for path in (*manifest_paths, planned_path)
            if not path.exists()
        ]
        if missing:
            return StageResult(
                stage,
                "incomplete",
                EXIT_INCOMPLETE,
                remaining=1,
                evidence={"missing_inputs": missing},
            )
        planned = pd.read_csv(planned_path)
        records = pd.concat(
            [pd.read_csv(path) for path in manifest_paths],
            ignore_index=True,
        )
        required = {
            "sample_id",
            "event_id",
            "checkpoint_id",
            "branch_role",
            "actual_schedule_sha256",
            "status",
        }
        missing_columns = required - set(records)
        if missing_columns:
            return StageResult(
                stage,
                "blocked",
                EXIT_BLOCKED,
                evidence={
                    "reason": "run_manifest_not_dataset_grade",
                    "missing_columns": sorted(missing_columns),
                },
            )
        passed_records = records[records["status"] == "pass"]
        try:
            branch_manifest = build_branch_manifest(passed_records)
            samples, duplicates, incomplete = build_sample_manifest(
                branch_manifest
            )
        except ValueError as exc:
            return StageResult(
                stage, "blocked", EXIT_BLOCKED, evidence={"reason": str(exc)}
            )
        failed = records[records["status"] != "pass"].copy()
        if len(failed):
            failed["rejection_reason"] = "branch_run_failed"
        rejected = pd.concat([duplicates, failed], ignore_index=True)
        known_cases = set(records.get("case_id", pd.Series(dtype=str)).astype(str))
        pending = planned[
            ~planned["case_id"].astype(str).isin(known_cases)
        ].copy()
        accounting = dataset_accounting(
            len(planned), samples, rejected, pending, incomplete
        )
        dataset_dir = output_root / scope / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        samples.to_csv(
            dataset_dir / f"{prefix}_sample_manifest.csv", index=False
        )
        branch_manifest.to_csv(
            dataset_dir / f"{prefix}_branch_manifest.csv", index=False
        )
        rejected.to_csv(dataset_dir / f"{prefix}_rejected.csv", index=False)
        pending.to_csv(dataset_dir / f"{prefix}_pending.csv", index=False)
        incomplete.to_csv(dataset_dir / f"{prefix}_missing.csv", index=False)
        duplicates.to_csv(
            dataset_dir / f"{prefix}_actual_duplicates.csv", index=False
        )
        atomic_write_json(
            dataset_dir / f"{prefix}_feature_schema.json", feature_schema()
        )
        atomic_write_json(
            dataset_dir / f"{prefix}_label_schema.json", label_schema()
        )
        if scope == "train1600" and "split" in samples:
            samples.drop_duplicates("event_id")[
                [
                    column
                    for column in ("event_id", "rainfall_sha256", "split")
                    if column in samples
                ]
            ].to_csv(
                dataset_dir / "train1600_split_manifest.csv", index=False
            )
        atomic_write_json(
            dataset_dir / f"{prefix}_provenance.json",
            {
                "planned_source": str(planned_path),
                "run_manifests": [str(path) for path in manifest_paths],
                "accounting": accounting,
            },
        )
        atomic_write_json(
            dataset_dir / "completion.json",
            {
                "stage": stage,
                "accounting": accounting,
                "accepted": int(len(samples)),
            },
        )
        complete = bool(accounting["accounting_closed"]) and len(pending) == 0
        return StageResult(
            stage,
            "pass" if complete else "incomplete",
            EXIT_PASS if complete else EXIT_INCOMPLETE,
            completed=len(samples),
            remaining=int(len(pending) + len(incomplete)),
            batch_complete=True,
            scope_complete=complete,
            evidence={"accounting": accounting},
        )

    return handler


LOCAL_RESPONSE_FLOOR = 1e-3


def _add_pilot_audit_columns(
    samples: pd.DataFrame, config: dict
) -> pd.DataFrame:
    """Derive the dataset-audit boolean columns from reduced sample rows."""
    result = samples.copy()
    if result.empty:
        for column in (
            "locally_responsive",
            "confirmed_flat",
            "tfv_noninferior",
        ):
            result[column] = pd.Series(dtype=bool)
        return result
    margin = float(
        config["thresholds"]["scientific_margin"].get("tfv_m3", 0.0)
    )
    result["locally_responsive"] = (
        pd.to_numeric(
            result.get("local_response_magnitude"), errors="coerce"
        )
        > LOCAL_RESPONSE_FLOOR
    )
    result["confirmed_flat"] = result.get(
        "flat_state", pd.Series(False, index=result.index)
    ).astype(bool)
    result["tfv_noninferior"] = (
        pd.to_numeric(
            result.get("delta_tfv_h120_vs_dynamic_internal"),
            errors="coerce",
        )
        <= margin
    )
    return result


def _build_pilot_dataset_handler(
    project_root: Path, output_root: Path, config: dict
) -> Callable[[RuntimeOptions], StageResult]:
    """Full-scope Pilot dataset from the sample-level run manifest."""

    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "BuildPilotDataset"
        plan_path, run_root = _run_stage_sources("RunPilot400", output_root)
        branch_path = (
            output_root / "pilot" / "planning" / "pilot_branch_plan.csv"
        )
        missing = [
            str(path)
            for path in (plan_path, branch_path)
            if not path.exists()
        ]
        if missing:
            return StageResult(
                stage,
                "incomplete",
                EXIT_INCOMPLETE,
                remaining=1,
                evidence={"missing_inputs": missing},
            )
        candidate_plan = pd.read_csv(plan_path)
        branch_plan = pd.read_csv(branch_path)
        completions = (
            completion_manifest(run_root)
            if run_root.exists()
            else pd.DataFrame()
        )
        project = config.get("project", {})
        try:
            result = build_pilot_dataset(
                candidate_plan,
                branch_plan,
                expand_pilot_completions(completions),
                priority_nodes=_read_facility_ids(
                    project_root / str(project.get("priority_nodes", ""))
                ),
                facility_ids=_read_facility_ids(
                    project_root / str(project.get("canonical_ids", ""))
                ),
                scientific_margin=config["thresholds"]["scientific_margin"],
                dead_zone=config["thresholds"]["dead_zone"],
            )
        except (KeyError, OSError, ValueError) as exc:
            return StageResult(
                stage, "blocked", EXIT_BLOCKED, evidence={"reason": str(exc)}
            )
        samples = _add_pilot_audit_columns(result["sample_manifest"], config)
        accounting = result["accounting"]
        dataset_dir = output_root / "pilot" / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        samples.to_csv(
            dataset_dir / "pilot_sample_manifest.csv", index=False
        )
        result["branch_manifest"].to_csv(
            dataset_dir / "pilot_branch_manifest.csv", index=False
        )
        result["rejected"].to_csv(
            dataset_dir / "pilot_rejected.csv", index=False
        )
        result["actual_duplicates"].to_csv(
            dataset_dir / "pilot_actual_duplicates.csv", index=False
        )
        result["pending"].to_csv(
            dataset_dir / "pilot_pending.csv", index=False
        )
        result["missing_confirmed"].to_csv(
            dataset_dir / "pilot_missing.csv", index=False
        )
        atomic_write_json(
            dataset_dir / "pilot_feature_schema.json", feature_schema()
        )
        atomic_write_json(
            dataset_dir / "pilot_label_schema.json", label_schema()
        )
        atomic_write_json(
            dataset_dir / "pilot_provenance.json",
            {
                "planned_source": str(plan_path),
                "branch_plan": str(branch_path),
                "run_root": str(run_root),
                "accounting": accounting,
            },
        )
        atomic_write_json(
            dataset_dir / "completion.json",
            {
                "stage": stage,
                "accounting": accounting,
                "accepted": int(accounting["accepted"]),
            },
        )
        complete = (
            bool(accounting["accounting_closed"])
            and int(accounting["missing"]) == 0
        )
        return StageResult(
            stage,
            "pass" if complete else "incomplete",
            EXIT_PASS if complete else EXIT_INCOMPLETE,
            completed=int(accounting["accepted"]),
            remaining=int(accounting["missing"]),
            batch_complete=True,
            scope_complete=complete,
            evidence={"accounting": accounting},
        )

    return handler


def _audit_pilot_dataset_handler(
    output_root: Path,
) -> Callable[[RuntimeOptions], StageResult]:
    def handler(_options: RuntimeOptions) -> StageResult:
        source = output_root / STAGE_ARTIFACTS["BuildPilotDataset"]
        if not source.exists():
            return StageResult(
                "AuditPilotDataset", "incomplete", EXIT_INCOMPLETE, remaining=1
            )
        audit = audit_pilot_dataset(pd.read_csv(source))
        atomic_write_json(
            output_root / STAGE_ARTIFACTS["AuditPilotDataset"], audit
        )
        passed = audit["status"] == "pass"
        return StageResult(
            "AuditPilotDataset",
            audit["status"],
            EXIT_PASS if passed else EXIT_SCIENTIFIC_FAIL,
            completed=int(len(pd.read_csv(source))),
            batch_complete=True,
            scope_complete=passed,
            evidence=audit,
        )

    return handler


def _audit_train_dataset_handler(
    output_root: Path,
) -> Callable[[RuntimeOptions], StageResult]:
    def handler(_options: RuntimeOptions) -> StageResult:
        dataset_dir = output_root / "train1600" / "dataset"
        sample_path = dataset_dir / "train1600_sample_manifest.csv"
        split_path = dataset_dir / "train1600_split_manifest.csv"
        completion_path = dataset_dir / "completion.json"
        missing = [
            str(path)
            for path in (sample_path, split_path, completion_path)
            if not path.exists()
        ]
        if missing:
            return StageResult(
                "AuditTrain1600Dataset",
                "incomplete",
                EXIT_INCOMPLETE,
                remaining=1,
                evidence={"missing_inputs": missing},
            )
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
        audit = audit_train1600_dataset(
            pd.read_csv(sample_path),
            pd.read_csv(split_path),
            accounting=completion.get("accounting", {}),
        )
        atomic_write_json(
            output_root / STAGE_ARTIFACTS["AuditTrain1600Dataset"], audit
        )
        passed = audit["status"] == "pass"
        return StageResult(
            "AuditTrain1600Dataset",
            audit["status"],
            EXIT_PASS if passed else EXIT_SCIENTIFIC_FAIL,
            completed=int(len(pd.read_csv(sample_path))),
            batch_complete=True,
            scope_complete=passed,
            evidence=audit,
        )

    return handler


def _build_formal_blind_handler(
    output_root: Path, config: dict
) -> Callable[[RuntimeOptions], StageResult]:
    def handler(_options: RuntimeOptions) -> StageResult:
        ledger_path = output_root / "inventory" / "event_usage_ledger.csv"
        if not ledger_path.exists():
            return StageResult(
                "BuildFormalBlindInventory",
                "incomplete",
                EXIT_INCOMPLETE,
                remaining=1,
                evidence={"reason": "event_usage_ledger_missing"},
            )
        try:
            ledger = _load_ledger(ledger_path)
            candidates = select_formal_blind_candidates(ledger)
        except ValueError as exc:
            return StageResult(
                "BuildFormalBlindInventory",
                "blocked",
                EXIT_BLOCKED,
                evidence={"reason": str(exc)},
            )
        minimum = int(
            config.get("formal_blind", {}).get("minimum_events", 24)
        )
        if len(candidates) < minimum:
            return StageResult(
                "BuildFormalBlindInventory",
                "blocked",
                EXIT_BLOCKED,
                evidence={
                    "reason": "insufficient_never_scanned_events",
                    "eligible_events": int(len(candidates)),
                    "required_events": minimum,
                    "action_required": (
                        "generate new independent rainfall events and "
                        "freeze them once after policy lock; never reuse "
                        "opportunity-scanned development events"
                    ),
                },
            )
        inventory = candidates[["event_id", "rainfall_sha256"]].copy()
        inventory["historically_used"] = False
        inventory["revealed"] = False
        audit = audit_formal_blind_inventory(inventory)
        if audit["status"] != "pass":
            return StageResult(
                "BuildFormalBlindInventory",
                "blocked",
                EXIT_BLOCKED,
                evidence=audit,
            )
        target = output_root / STAGE_ARTIFACTS["BuildFormalBlindInventory"]
        target.parent.mkdir(parents=True, exist_ok=True)
        inventory.to_csv(target, index=False)
        return StageResult(
            "BuildFormalBlindInventory",
            "pass",
            EXIT_PASS,
            completed=len(inventory),
            batch_complete=True,
            scope_complete=True,
            evidence=audit,
        )

    return handler


def build_registry(
    *,
    project_root: str | Path,
    output_root: str | Path,
    config: dict,
) -> StageRegistry:
    project = Path(project_root)
    output = Path(output_root)
    prepare_output_tree(output)
    registry = StageRegistry()
    # Imported lazily so pipeline_ext can import pipeline internals without a
    # circular module-level dependency.
    from .pipeline_ext import build_extension_handlers
    from .pipeline_p3 import build_p3_handlers
    from .pipeline_train_v3 import build_train_v3_handlers
    from .pipeline_train_v4 import build_train_v4_handlers
    from .pipeline_train_v4_model import build_train_v4_model_handlers
    from .pipeline_v4_compact import build_v4_compact_phase1_handlers
    from .pipeline_v4_compact_eval import build_v4_compact_phase2_handlers
    from .pipeline_v4_closed_loop import build_v4_closed_loop_handlers
    from .pipeline_v42 import build_v42_handlers

    extension_handlers = build_extension_handlers(
        project_root=project, output_root=output, config=config
    )
    p3_handlers = build_p3_handlers(
        project_root=project, output_root=output, config=config
    )
    train_v3_handlers = build_train_v3_handlers(
        project_root=project, output_root=output, config=config
    )
    train_v4_handlers = build_train_v4_handlers(
        project_root=project, output_root=output, config=config
    )
    train_v4_model_handlers = build_train_v4_model_handlers(
        project_root=project, output_root=output, config=config
    )
    v4_compact_phase1_handlers = build_v4_compact_phase1_handlers(
        project_root=project, output_root=output, config=config
    )
    v4_compact_phase2_handlers = build_v4_compact_phase2_handlers(
        project_root=project, output_root=output, config=config
    )
    v4_closed_loop_handlers = build_v4_closed_loop_handlers(
        project_root=project, output_root=output, config=config
    )
    v42_handlers = build_v42_handlers(
        project_root=project, output_root=output, config=config
    )
    registry.register("AuditContracts", _contract_handler(project, output, config))
    for stage in ALL_STAGES:
        if stage == "AuditContracts":
            continue
        handler = {
            "BuildEventInventory": _build_inventory_handler(
                project, output, config
            ),
            "PlanOpportunityPool": _plan_opportunity_handler(
                project, output, config
            ),
            "BuildOpportunityPool": _build_opportunity_handler(
                project, output, config
            ),
            "AuditOpportunityCoverage": _audit_opportunity_handler(
                output, config
            ),
            "BuildPeakCandidateCatalog": _build_peak_catalog_handler(
                project, output, config
            ),
            "PlanPeakBoundary": _plan_peak_handler(output),
            "BuildPeakBoundaryDataset": _build_peak_dataset_handler(
                project, output, config
            ),
            "AuditPeakBoundary": _audit_peak_handler(output),
            "RestampPeakBoundaryEvidence": _restamp_peak_handler(
                project, output, config
            ),
            "ClassifyExistingGate5R": _classify_gate5r_handler(output),
            "PlanPilot400": _plan_pilot_handler(project, output, config),
            "AuditPilotPlan": _audit_pilot_plan_handler(output),
            "BuildPilotDataset": _build_pilot_dataset_handler(
                project, output, config
            ),
            "AuditPilotDataset": _audit_pilot_dataset_handler(output),
            "EvaluatePilotGate": _evaluate_pilot_gate_handler(output),
            "PlanTrain1600": _plan_train_handler(project, output, config),
            "AuditTrain1600Plan": _audit_train_plan_handler(output),
            "BuildTrain1600Dataset": _build_dataset_handler(
                "BuildTrain1600Dataset", output, "train1600"
            ),
            "AuditTrain1600Dataset": _audit_train_dataset_handler(output),
            "BuildFormalBlindInventory": _build_formal_blind_handler(
                output, config
            ),
            **{
                run_stage: _run_case_stage_handler(
                    run_stage, output, config
                )
                for run_stage in RUN_STAGE_PLANS
            },
            **{
                partial_stage: _build_partial_handler(
                    partial_stage, project, output, config
                )
                for partial_stage in PARTIAL_STAGE_RUN
                if partial_stage.startswith("Build")
            },
            **{
                partial_stage: _audit_partial_handler(
                    partial_stage, output, config
                )
                for partial_stage in PARTIAL_STAGE_RUN
                if partial_stage.startswith("Audit")
            },
            **{
                preflight_stage: _preflight_handler(
                    preflight_stage, project, output, config
                )
                for preflight_stage in PREFLIGHT_STAGE_RUN
            },
            # Peak partial stages need the four-branch same-state reduction,
            # so they override the generic single-row partial handlers above.
            "BuildPeakBoundaryPartial": _build_peak_partial_handler(
                "BuildPeakBoundaryPartial", project, output, config
            ),
            "AuditPeakBoundaryPartial": _audit_peak_partial_handler(
                "AuditPeakBoundaryPartial", project, output, config
            ),
            # Pilot stages run one completion per sample with embedded
            # branches, so run and partial handlers override the generic
            # per-case handlers above.
            "RunPilot400": _run_pilot400_handler(output, config),
            "BuildPilotPartial": _build_pilot_partial_handler(
                "BuildPilotPartial", project, output, config
            ),
            "AuditPilotPartial": _audit_pilot_partial_handler(
                "AuditPilotPartial", project, output, config
            ),
            # Extension/v2 stages override the generic run/partial handlers
            # generated by the comprehensions above; preflight stages keep
            # the generic _preflight_handler via PREFLIGHT_STAGE_RUN.
            **extension_handlers,
            **p3_handlers,
            **train_v3_handlers,
            **train_v4_handlers,
            **train_v4_model_handlers,
            **v4_compact_phase1_handlers,
            **v4_compact_phase2_handlers,
            **v4_closed_loop_handlers,
            **v42_handlers,
        }.get(stage, _artifact_handler(stage, output))
        prerequisites = PREREQUISITES.get(stage, ())
        if prerequisites:
            inner_handler = handler

            def gated_handler(
                options: RuntimeOptions,
                *,
                _inner=inner_handler,
                _stage=stage,
                _prerequisites=prerequisites,
            ) -> StageResult:
                missing_or_failed = []
                config_path = Path(options.config) if options.config else None
                current_config_sha = (
                    hashlib.sha256(config_path.read_bytes()).hexdigest()
                    if config_path is not None and config_path.exists()
                    else ""
                )
                current_code_sha = working_code_sha(project)
                for prerequisite in _prerequisites:
                    status_path = (
                        output
                        / "audits"
                        / "stage_status"
                        / f"{prerequisite}.json"
                    )
                    try:
                        status = json.loads(
                            status_path.read_text(encoding="utf-8")
                        )
                    except (OSError, ValueError, TypeError):
                        status = {}
                    completion_path = status_path.with_name(
                        f"{prerequisite}.completion.json"
                    )
                    try:
                        completion = json.loads(
                            completion_path.read_text(encoding="utf-8")
                        )
                    except (OSError, ValueError, TypeError):
                        completion = {}
                    completion_valid = bool(completion) and (
                        completion.get("run_uuid") == status.get("run_uuid")
                        and completion.get("status_sha256")
                        == hashlib.sha256(status_path.read_bytes()).hexdigest()
                    )
                    config_matches = bool(current_config_sha) and (
                        status.get("config_sha") == current_config_sha
                    )
                    code_matches = (
                        status.get("code_git_sha") == current_code_sha
                    )
                    if (
                        int(status.get("exit_code", -1)) != 0
                        or not bool(status.get("scope_complete", False))
                        or not completion_valid
                        or not config_matches
                        or not code_matches
                    ):
                        missing_or_failed.append(prerequisite)
                if missing_or_failed:
                    return StageResult(
                        _stage,
                        "blocked",
                        EXIT_BLOCKED,
                        evidence={
                            "reason": "prerequisite_not_passed",
                            "prerequisites": missing_or_failed,
                            "long_task_not_started": bool(options.dry_run),
                        },
                    )
                return _inner(options)

            handler = gated_handler
        registry.register(stage, handler)
    return registry


def sha256_json(value: dict) -> str:
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(canonical).hexdigest()
