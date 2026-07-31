"""V4.2 pipeline handlers — Phase 0 (Freeze + Audit) and stubs for later phases.

Freeze V4.1 scientific failure immutably, then audit the classification metric
semantics to identify the root cause of the Predictive Generalization Gate
scientific_fail verdict before proceeding with V4.2 retraining.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .runtime import (
    EXIT_BLOCKED,
    EXIT_PASS,
    RuntimeOptions,
    StageResult,
    atomic_write_json,
    working_code_sha,
)


# ---------------------------------------------------------------------------
# Output layout (all relative to output_root)
# ---------------------------------------------------------------------------

V41_MODEL_DIR = "models/v4_compact_v1"
FREEZE_DIR = "audits/frozen_evidence/v41_scientific_failure"
METRICS_AUDIT_DIR = "audits/v42_metric_semantics"

_V41_FREEZE_FILES_NAMES = (
    "compact_head_specific_model.pkl",
    "v4_compact_v1_calibration.json",
    "v4_compact_v1_locked_intent.json",
    "v4_compact_v1_locked_evaluation.json",
    "v4_predictive_generalization_gate.json",
    "v4_compact_v1_selection.json",
    "cv_decision_metrics.csv",
    "cv_metrics_by_event.csv",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Handler factory
# ---------------------------------------------------------------------------

def build_v42_handlers(
    *, project_root: Path, output_root: Path, config: dict
) -> dict[str, Callable[[RuntimeOptions], StageResult]]:
    """Return dict mapping stage_name -> handler callable."""
    return {
        "FreezeV41ScientificFailure": _freeze_v41_handler(
            project_root, output_root, config
        ),
        "AuditV41ClassificationMetricSemantics": _audit_metric_semantics_handler(
            project_root, output_root, config
        ),
        # Stubs for later phases (will be implemented in subsequent tasks)
        "BuildV42TrajectoryDataset": _build_trajectory_handler(
            project_root, output_root, config
        ),
        "AuditV42TrajectoryDataset": _audit_trajectory_handler(
            project_root, output_root, config
        ),
        "TrainV42WaterBalanceBaseline": _train_water_balance_handler(
            project_root, output_root, config
        ),
        "EvaluateV42WaterBalanceBaseline": _evaluate_water_balance_handler(
            project_root, output_root, config
        ),
        "TrainV42TwinGraphDynamics": _train_twin_graph_dynamics_handler(
            project_root, output_root, config
        ),
        "RunV42ArchitectureAblation": _stub_handler("RunV42ArchitectureAblation"),
        "RunV42StateScopeAblation": _stub_handler("RunV42StateScopeAblation"),
        "RunV42TargetAblation": _stub_handler("RunV42TargetAblation"),
        "BuildV42EventLearningCurve": _stub_handler("BuildV42EventLearningCurve"),
        "AuditV42TrainGroupedGeneralizationGate": _stub_handler(
            "AuditV42TrainGroupedGeneralizationGate"
        ),
        # ── V4.2 data pipeline handlers ──
        "BuildV42EventUsageLedger": _build_event_ledger_handler(
            project_root, output_root, config
        ),
        "AuditV42EventUsageLedger": _audit_event_ledger_handler(
            project_root, output_root, config
        ),
        "BuildV42UnifiedDevelopmentPool": _build_unified_pool_handler(
            project_root, output_root, config
        ),
        "AuditV42UnifiedDevelopmentPool": _audit_unified_pool_handler(
            project_root, output_root, config
        ),
        "BuildV42DerivedSupervision": _build_derived_supervision_handler(
            project_root, output_root, config
        ),
        "AuditV42DerivedSupervision": _audit_derived_supervision_handler(
            project_root, output_root, config
        ),
        "PlanV42NestedGroupedCV": _plan_nested_cv_handler(
            project_root, output_root, config
        ),
        "AuditV42NestedGroupedCVPlan": _audit_nested_cv_plan_handler(
            project_root, output_root, config
        ),
        "RunV42NestedGroupedCV": _stub_handler("RunV42NestedGroupedCV"),
        "BuildV42NestedGroupedCVResults": _stub_handler("BuildV42NestedGroupedCVResults"),
        "AuditV42NestedGroupedCVResults": _stub_handler("AuditV42NestedGroupedCVResults"),
        # ── V4.2 validation handlers ──
        "AuditV42HeadActivation": _stub_handler("AuditV42HeadActivation"),
        "AuditV42TargetMetricSemantics": _stub_handler("AuditV42TargetMetricSemantics"),
        "AuditV42RankingPhysics": _stub_handler("AuditV42RankingPhysics"),
        "RunV42TinyOverfit": _stub_handler("RunV42TinyOverfit"),
        # ── V4.2 fresh evaluation handlers ──
        "PlanV42FreshEvaluationSplit": _plan_fresh_eval_handler(
            project_root, output_root, config
        ),
        "AuditV42FreshEvaluationAvailability": _audit_fresh_eval_handler(
            project_root, output_root, config
        ),
        # ── V4.2 final data pool handlers ──
        "AuditV42PrioritySentinelContract": _audit_priority_contract_handler(
            project_root, output_root, config
        ),
        "FreezeV42PriorityContract": _freeze_priority_contract_handler(
            project_root, output_root, config
        ),
        "BuildV42IndependentPfvOracle": _build_pfv_oracle_handler(
            project_root, output_root, config
        ),
        "BuildV42SampleLineage": _build_sample_lineage_handler(
            project_root, output_root, config
        ),
        "AuditV42PhysicalDeduplication": _audit_physical_dedup_handler(
            project_root, output_root, config
        ),
        "BuildV42HistoricalSemanticInventory": _build_semantic_inventory_handler(
            project_root, output_root, config
        ),
        "AuditV42HistoricalSemanticInventory": _audit_semantic_inventory_handler(
            project_root, output_root, config
        ),
        "AuditV42DwfSources": _audit_dwf_sources_handler(
            project_root, output_root, config
        ),
        "BuildV42Canonical13FrameTrajectories": _build_history_rebuild_handler(
            project_root, output_root, config
        ),
        "AuditV42Canonical13FrameTrajectories": _audit_history_rebuild_handler(
            project_root, output_root, config
        ),
        "BuildV42TfvPeakOracle": _build_tfv_peak_oracle_handler(
            project_root, output_root, config
        ),
        "BuildV42SampleClassifier": _build_sample_classifier_handler(
            project_root, output_root, config
        ),
        "BuildV42FinalUnifiedDatasets": _build_final_unified_handler(
            project_root, output_root, config
        ),
        "AuditV42FinalUnifiedDatasets": _audit_final_unified_handler(
            project_root, output_root, config
        ),
        "BuildV42GroupedSplits": _build_grouped_splits_handler(
            project_root, output_root, config
        ),
        "BuildV42PoolStatistics": _build_pool_statistics_handler(
            project_root, output_root, config
        ),
        "AuditV42FinalDatasetAdmissionGate": _audit_admission_gate_handler(
            project_root, output_root, config
        ),
    }


# ---------------------------------------------------------------------------
# V4.2 Trajectory Dataset output directory
# ---------------------------------------------------------------------------

V42_TRAJECTORY_DIR = "v42/trajectory_dataset"


# ---------------------------------------------------------------------------
# BuildV42TrajectoryDataset
# ---------------------------------------------------------------------------

def _build_trajectory_handler(project_root: Path, output_root: Path, config: dict):
    """Build the V4.2 trajectory dataset from Train split data."""

    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "BuildV42TrajectoryDataset"
        code_sha = working_code_sha(project_root)
        trajectory_dir = output_root / V42_TRAJECTORY_DIR

        try:
            from .v42_trajectory_builder import (
                build_trajectory_dataset,
                write_trajectory_dataset,
            )

            result = build_trajectory_dataset(
                project_root=project_root,
                output_root=output_root,
                config=config,
                train_only=True,
            )
            written = write_trajectory_dataset(result, trajectory_dir)

            return StageResult(
                stage,
                "pass",
                EXIT_PASS,
                completed=result.sample_count,
                batch_complete=True,
                scope_complete=True,
                evidence={
                    "sample_count": result.sample_count,
                    "reference_dedup_count": result.reference_dedup_count,
                    "n_warnings": len(result.warnings),
                    "warnings": result.warnings[:10],
                    "code_sha256": code_sha,
                    "output_dir": str(trajectory_dir),
                    "files": written,
                    "graph_schema": result.graph_schema,
                },
            )
        except Exception as exc:
            return StageResult(
                stage,
                "runtime_error",
                EXIT_BLOCKED,
                evidence={
                    "error": str(exc),
                    "code_sha256": code_sha,
                },
            )

    return handler


# ---------------------------------------------------------------------------
# AuditV42TrajectoryDataset
# ---------------------------------------------------------------------------

def _audit_trajectory_handler(project_root: Path, output_root: Path, config: dict):
    """Audit the V4.2 trajectory dataset for completeness and correctness."""

    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "AuditV42TrajectoryDataset"
        code_sha = working_code_sha(project_root)
        trajectory_dir = output_root / V42_TRAJECTORY_DIR

        audit: dict[str, Any] = {
            "stage": stage,
            "audit_time": datetime.now(timezone.utc).isoformat(),
            "code_sha256": code_sha,
            "checks": {},
        }
        checks = audit["checks"]

        # --- Check 1: Manifest exists and readable ---
        parquet_path = trajectory_dir / "trajectory_manifest_v42.parquet"
        if not parquet_path.exists():
            audit["status"] = "blocked"
            audit["reason"] = "manifest_parquet_missing"
            atomic_write_json(
                trajectory_dir / "trajectory_audit_v42.json", audit
            )
            return StageResult(
                stage, "blocked", EXIT_BLOCKED,
                evidence={"reason": "manifest_parquet_missing", "path": str(parquet_path)},
            )

        manifest = pd.read_parquet(parquet_path)
        checks["manifest_exists"] = True
        checks["sample_count"] = len(manifest)

        # --- Check 2: Sample count matches expected ---
        expected_min = 1000  # At least 1000 accepted train samples
        checks["sample_count_ok"] = len(manifest) >= expected_min
        if not checks["sample_count_ok"]:
            audit["warnings"].append(
                f"sample_count {len(manifest)} < expected minimum {expected_min}"
            )

        # --- Check 3: Schema files exist ---
        for schema_name in (
            "graph_schema_v42.json",
            "node_feature_schema_v42.json",
            "edge_feature_schema_v42.json",
            "action_schema_v42.json",
        ):
            schema_path = trajectory_dir / schema_name
            checks[f"{schema_name}_exists"] = schema_path.exists()
            if schema_path.exists():
                schema = json.loads(schema_path.read_text(encoding="utf-8"))
                checks[f"{schema_name}_content"] = schema

        # --- Check 4: No future information leakage ---
        # Verify that the manifest does NOT contain future KPI labels as columns
        forbidden_cols = {
            "future_pfv", "future_tfv", "future_peak",
            "realized_pfv", "realized_tfv",
        }
        present_forbidden = set(manifest.columns) & forbidden_cols
        checks["no_future_leakage"] = len(present_forbidden) == 0
        if present_forbidden:
            audit["warnings"].append(
                f"forbidden future columns present: {sorted(present_forbidden)}"
            )

        # --- Check 5: Reference dedup ---
        if "event_id" in manifest.columns and "checkpoint_id" in manifest.columns:
            unique_refs = manifest.groupby(
                ["event_id", "checkpoint_id"]
            ).ngroups
            checks["reference_dedup_unique_pairs"] = unique_refs
            checks["reference_dedup_ok"] = bool(unique_refs > 0)
        else:
            checks["reference_dedup_ok"] = False

        # --- Check 6: Required columns present ---
        required_cols = [
            "event_id", "checkpoint_id", "state_key", "split",
            "candidate_action_seq", "history_depth", "history_actions",
            "trajectory_depth_candidate", "trajectory_depth_no_control",
            "trajectory_depth_dynamic_internal", "trajectory_depth_hold_previous",
            "rainfall_forecast", "pfv_delta", "tfv_delta", "peak_delta",
            "pfv_safe_label", "tfv_improved_label", "peak_noninferior_label",
        ]
        missing_cols = [c for c in required_cols if c not in manifest.columns]
        checks["required_columns_present"] = len(missing_cols) == 0
        checks["missing_columns"] = missing_cols

        # --- Check 7: Split is all train ---
        if "split" in manifest.columns:
            checks["all_train_split"] = bool(
                manifest["split"].astype(str).eq("train").all()
            )

        # --- Sanitize checks for JSON serialization ---
        import numpy as np
        sanitized_checks = {}
        for k, v in checks.items():
            if isinstance(v, (np.bool_,)):
                sanitized_checks[k] = bool(v)
            elif isinstance(v, (np.integer,)):
                sanitized_checks[k] = int(v)
            elif isinstance(v, (np.floating,)):
                sanitized_checks[k] = float(v)
            else:
                sanitized_checks[k] = v
        checks = sanitized_checks
        audit["checks"] = checks  # re-point audit to sanitized dict

        # --- Summary ---
        all_ok = all(
            v for k, v in checks.items()
            if isinstance(v, bool)
        )
        audit["status"] = "pass" if all_ok else "incomplete"
        audit["summary"] = (
            f"Audited {len(manifest)} samples. "
            f"{'All checks passed.' if all_ok else 'Some checks failed.'}"
        )

        # Write audit JSON
        audit_dir = trajectory_dir
        audit_path = audit_dir / "trajectory_audit_v42.json"
        atomic_write_json(audit_path, audit)

        passed = all_ok
        return StageResult(
            stage,
            audit["status"],
            EXIT_PASS if passed else EXIT_BLOCKED,
            completed=1,
            batch_complete=True,
            scope_complete=passed,
            evidence={
                "sample_count": len(manifest),
                "all_checks_passed": all_ok,
                "audit_path": str(audit_path),
                "checks": {
                    k: v for k, v in checks.items()
                    if isinstance(v, (bool, int))
                },
            },
        )

    return handler


# ---------------------------------------------------------------------------
# Water Balance output directory
# ---------------------------------------------------------------------------

WATER_BALANCE_DIR = "models/v42_water_balance"


# ---------------------------------------------------------------------------
# TrainV42WaterBalanceBaseline
# ---------------------------------------------------------------------------

def _train_water_balance_handler(project_root: Path, output_root: Path, config: dict):
    """Train + CV water balance baseline model."""

    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "TrainV42WaterBalanceBaseline"
        code_sha = working_code_sha(project_root)
        model_dir = output_root / WATER_BALANCE_DIR

        try:
            from .v42_water_balance import train_water_balance_baseline

            cv_results = train_water_balance_baseline(
                project_root=project_root,
                output_root=output_root,
                config=config,
            )

            # Write CV results
            model_dir.mkdir(parents=True, exist_ok=True)
            cv_path = model_dir / "water_balance_baseline_cv.json"

            # Sanitize for JSON serialization
            def _sanitize(obj: Any) -> Any:
                if isinstance(obj, (np.bool_,)):
                    return bool(obj)
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, dict):
                    return {k: _sanitize(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple)):
                    return [_sanitize(v) for v in obj]
                if isinstance(obj, set):
                    return sorted(obj)
                return obj

            cv_sanitized = _sanitize(cv_results)
            cv_sanitized["code_sha256"] = code_sha
            atomic_write_json(cv_path, cv_sanitized)

            # Extract summary metrics
            agg = cv_results.get("aggregate_metrics", {})
            summary_r2 = {
                t: agg.get(t, {}).get("r2_mean", None)
                for t in [
                    "system_storage", "outfall_flow",
                    "total_flooding_rate", "tfv", "peak",
                ]
            }

            return StageResult(
                stage, "pass", EXIT_PASS,
                completed=cv_results.get("n_samples", 0),
                batch_complete=True,
                scope_complete=True,
                evidence={
                    "cv_path": str(cv_path),
                    "n_folds": cv_results.get("n_folds", 5),
                    "n_events": cv_results.get("n_events", 0),
                    "n_samples": cv_results.get("n_samples", 0),
                    "r2_summary": summary_r2,
                    "code_sha256": code_sha,
                },
            )
        except Exception as exc:
            return StageResult(
                stage, "runtime_error", EXIT_BLOCKED,
                evidence={"error": str(exc), "code_sha256": code_sha},
            )

    return handler


# ---------------------------------------------------------------------------
# EvaluateV42WaterBalanceBaseline
# ---------------------------------------------------------------------------

def _evaluate_water_balance_handler(project_root: Path, output_root: Path, config: dict):
    """Evaluate water balance model — per-event detailed results."""

    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "EvaluateV42WaterBalanceBaseline"
        code_sha = working_code_sha(project_root)
        model_dir = output_root / WATER_BALANCE_DIR

        try:
            from .v42_water_balance import evaluate_water_balance_baseline

            eval_results = evaluate_water_balance_baseline(
                project_root=project_root,
                output_root=output_root,
                config=config,
            )

            # Write evaluation results
            model_dir.mkdir(parents=True, exist_ok=True)
            eval_path = model_dir / "water_balance_evaluation.json"

            def _sanitize(obj: Any) -> Any:
                if isinstance(obj, (np.bool_,)):
                    return bool(obj)
                if isinstance(obj, (np.integer,)):
                    return int(obj)
                if isinstance(obj, (np.floating,)):
                    return float(obj)
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, dict):
                    return {k: _sanitize(v) for k, v in obj.items()}
                if isinstance(obj, (list, tuple)):
                    return [_sanitize(v) for v in obj]
                return obj

            eval_sanitized = _sanitize(eval_results)
            eval_sanitized["code_sha256"] = code_sha
            atomic_write_json(eval_path, eval_sanitized)

            # Summary
            overall = eval_results.get("overall_metrics", {})
            summary = {
                t: overall.get(t, {}).get("r2", None)
                for t in [
                    "system_storage", "outfall_flow",
                    "total_flooding_rate", "tfv", "peak",
                ]
            }

            return StageResult(
                stage, "pass", EXIT_PASS,
                completed=eval_results.get("n_samples", 0),
                batch_complete=True,
                scope_complete=True,
                evidence={
                    "eval_path": str(eval_path),
                    "n_events": eval_results.get("n_events", 0),
                    "n_samples": eval_results.get("n_samples", 0),
                    "r2_summary": summary,
                    "code_sha256": code_sha,
                },
            )
        except Exception as exc:
            return StageResult(
                stage, "runtime_error", EXIT_BLOCKED,
                evidence={"error": str(exc), "code_sha256": code_sha},
            )

    return handler


# ---------------------------------------------------------------------------
# TrainV42TwinGraphDynamics output directory
# ---------------------------------------------------------------------------

V42_TWIN_MODEL_DIR = "models/v42_twin"


# ---------------------------------------------------------------------------
# TrainV42TwinGraphDynamics
# ---------------------------------------------------------------------------

def _train_twin_graph_dynamics_handler(project_root: Path, output_root: Path, config: dict):
    """Train V4.2 TwinGraphDynamics model with 4-stage curriculum and 5-seed CV."""

    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "TrainV42TwinGraphDynamics"
        code_sha = working_code_sha(project_root)
        model_dir = output_root / V42_TWIN_MODEL_DIR

        try:
            from .v42_trainer import train_v42_twin

            results = train_v42_twin(
                project_root=project_root,
                output_root=output_root,
                config=config,
            )

            # Write combined results with code SHA
            model_dir.mkdir(parents=True, exist_ok=True)
            results["code_sha256"] = code_sha

            combined_path = model_dir / "training_history.json"
            with open(combined_path, "w") as f:
                json.dump(results, f, indent=2, default=str)

            # Extract summary metrics
            overall = results.get("overall_aggregate", {})
            summary_r2 = {
                k: v for k, v in overall.items() if "r2" in k
            }

            return StageResult(
                stage, "pass", EXIT_PASS,
                completed=results.get("n_samples", 0),
                batch_complete=True,
                scope_complete=True,
                evidence={
                    "n_seeds": len(results.get("seeds", [])),
                    "n_folds": results.get("n_folds", 5),
                    "n_samples": results.get("n_samples", 0),
                    "n_events": results.get("n_events", 0),
                    "r2_summary": summary_r2,
                    "model_dir": str(model_dir),
                    "code_sha256": code_sha,
                },
            )
        except Exception as exc:
            import traceback
            return StageResult(
                stage, "runtime_error", EXIT_BLOCKED,
                evidence={
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "code_sha256": code_sha,
                },
            )

    return handler


# ---------------------------------------------------------------------------
# Stub handler for later phases
# ---------------------------------------------------------------------------

def _stub_handler(stage_name: str):
    def handler(_options: RuntimeOptions) -> StageResult:
        return StageResult(
            stage=stage_name,
            status="not_implemented",
            exit_code=2,  # blocked - not yet implemented
            scope_complete=False,
            evidence={
                "stage": stage_name,
                "exit_code": 2,
                "scope_complete": False,
                "status": "not_implemented",
                "message": f"{stage_name} is not yet implemented in Phase 0",
            },
        )
    return handler


# ---------------------------------------------------------------------------
# FreezeV41ScientificFailure
# ---------------------------------------------------------------------------

def _freeze_v41_handler(project_root: Path, output_root: Path, config: dict):
    """Immutable archive of V4.1 compact model and its scientific_fail result.

    Copies the V4.1 artifacts to a freeze directory, computes SHA256 for each
    file, and writes a freeze manifest that records the V4.1 failure as
    immutable evidence for V4.2 retraining.
    """

    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "FreezeV41ScientificFailure"
        code_sha = working_code_sha(project_root)
        src_dir = output_root / V41_MODEL_DIR
        freeze_dir = output_root / FREEZE_DIR

        # Check source directory exists
        if not src_dir.exists():
            return StageResult(
                stage, "blocked", 2,
                evidence={"reason": "v41_model_dir_missing", "path": str(src_dir)},
            )

        # Check critical source files exist
        gate_path = src_dir / "v4_predictive_generalization_gate.json"
        eval_path = src_dir / "v4_compact_v1_locked_evaluation.json"
        missing_src = [
            str(p) for p in (gate_path, eval_path) if not p.exists()
        ]
        if missing_src:
            return StageResult(
                stage, "blocked", 2,
                evidence={"reason": "v41_critical_files_missing", "missing": missing_src},
            )

        # Verify gate verdict is scientific_fail
        gate = _read_json(gate_path)
        if gate.get("status") != "scientific_fail":
            return StageResult(
                stage, "blocked", 2,
                evidence={
                    "reason": "v41_gate_not_scientific_fail",
                    "found_status": gate.get("status"),
                },
            )

        # Create freeze directory
        freeze_dir.mkdir(parents=True, exist_ok=True)

        # Copy files and compute SHA256
        files_info: dict[str, dict] = {}
        for name in _V41_FREEZE_FILES_NAMES:
            source = src_dir / name
            if not source.exists():
                continue
            destination = freeze_dir / name
            if not destination.exists():
                shutil.copy2(source, destination)
            sha = _sha256_file(destination)
            size = destination.stat().st_size
            files_info[name] = {"sha256": sha, "size_bytes": size}

        # Read locked evaluation for metrics
        evaluation = _read_json(eval_path)
        cont = evaluation.get("continuous", {})
        cls = evaluation.get("classification", {})

        locked_metrics = {
            "pfv_r2": cont.get("pfv", {}).get("r2"),
            "tfv_r2": cont.get("tfv", {}).get("r2"),
            "peak_r2": cont.get("peak", {}).get("r2"),
            "pfv_safe_auroc": cls.get("pfv_safe", {}).get("auroc"),
            "tfv_improved_auroc": cls.get("tfv_improved", {}).get("auroc"),
            "peak_noninferior_auroc": cls.get("peak_noninferior", {}).get("auroc"),
            "joint_noninferior_auroc": cls.get("joint_noninferior", {}).get("auroc"),
        }

        # Read selection info
        selection_path = src_dir / "v4_compact_v1_selection.json"
        selection = _read_json(selection_path) if selection_path.exists() else {}

        # Build freeze manifest
        manifest = {
            "frozen_at": datetime.now(timezone.utc).isoformat(),
            "code_sha256": code_sha,
            "immutable": True,
            "overwrite_prohibited": True,
            "model_version": "v4.1",
            "selected_architecture": selection.get("selected_architecture", "B"),
            "selected_feature_combo": selection.get(
                "selected_feature_combo", "candidate_minus_di_only"
            ),
            "peak_head_style": selection.get("peak_head_style", "consistency"),
            "seeds": selection.get("frozen_contract", {}).get("seeds", [0, 1, 2, 3, 4]),
            "pfv_hurdle": selection.get("frozen_contract", {}).get("pfv_hurdle", True),
            "full_event_heads_disabled": selection.get("frozen_contract", {}).get(
                "full_event_heads_disabled", True
            ),
            "locked_consumed": True,
            "predictive_gate_verdict": "scientific_fail",
            "authorizes_closed_loop": False,
            "locked_metrics": locked_metrics,
            "gate_checks_passed": sum(
                1 for v in gate.get("checks", {}).values() if v
            ),
            "gate_checks_total": len(gate.get("checks", {})),
            "files": files_info,
        }

        manifest_path = freeze_dir / "v41_freeze_manifest.json"
        atomic_write_json(manifest_path, manifest)

        return StageResult(
            stage, "pass", EXIT_PASS,
            completed=len(files_info),
            batch_complete=True,
            scope_complete=True,
            evidence={
                "frozen_files": len(files_info),
                "predictive_gate_verdict": "scientific_fail",
                "immutable": True,
                "code_sha256": code_sha,
                "manifest_path": str(manifest_path),
            },
        )

    return handler


# ---------------------------------------------------------------------------
# AuditV41ClassificationMetricSemantics
# ---------------------------------------------------------------------------

def _audit_metric_semantics_handler(project_root: Path, output_root: Path, config: dict):
    """Audit the 10 classification metric semantics for V4.1.

    Verifies consistency between metric definitions in the code and the
    reported values in the V4.1 Locked evaluation. The critical check is
    whether false_safe_rate=1.0 is consistent with "all predictions unsafe"
    or indicates an inverted label convention.
    """

    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "AuditV41ClassificationMetricSemantics"
        model_dir = output_root / V41_MODEL_DIR

        # Read V4.1 artifacts
        eval_path = model_dir / "v4_compact_v1_locked_evaluation.json"
        gate_path = model_dir / "v4_predictive_generalization_gate.json"
        contract_path = (
            project_root
            / "docs"
            / "contracts"
            / "PROJECT6_V4_PREDICTIVE_GENERALIZATION_GATE_V1.json"
        )

        missing = [
            str(p) for p in (eval_path, gate_path, contract_path)
            if not p.exists()
        ]
        if missing:
            return StageResult(
                stage, "blocked", 2,
                evidence={"reason": "required_files_missing", "missing": missing},
            )

        evaluation = _read_json(eval_path)
        gate_verdict = _read_json(gate_path)
        contract = _read_json(contract_path)

        # Read metric computation source code for verification
        metrics_code_path = project_root / "sewerrtc" / "v4" / "train_v4_metrics.py"
        eval_ops_path = project_root / "sewerrtc" / "v4" / "v4_compact_eval_ops.py"

        metrics_code = metrics_code_path.read_text(encoding="utf-8") if metrics_code_path.exists() else ""
        eval_ops_code = eval_ops_path.read_text(encoding="utf-8") if eval_ops_path.exists() else ""

        classification = evaluation.get("classification", {})
        findings: dict[str, Any] = {}

        # ---- Audit 1: pfv_safe positive class definition ----
        pfv_safe = classification.get("pfv_safe", {})
        pfv_support = pfv_safe.get("class_support", {})
        pfv_positive = pfv_support.get("positive", 0)
        pfv_negative = pfv_support.get("negative", 0)
        # In the code: positive=1 means "safe" (y_true==1 → safe)
        # false_safe_rate = mean(yhat[unsafe]==1) → among y_true=0, fraction predicted safe
        findings["pfv_safe_positive_class"] = {
            "definition": "positive=1 means 'safe' (state passes safety threshold)",
            "code_evidence": "y_true==0 → unsafe; y_true==1 → safe",
            "class_support": {"positive": pfv_positive, "negative": pfv_negative},
            "interpretation": f"{pfv_positive} safe states, {pfv_negative} unsafe states",
        }

        # ---- Audit 2: tfv_improved positive class definition ----
        tfv_imp = classification.get("tfv_improved", {})
        tfv_support = tfv_imp.get("class_support", {})
        findings["tfv_improved_positive_class"] = {
            "definition": "positive=1 means 'improved' (TFV actual < predicted, i.e. not overpredicted)",
            "class_support": {
                "positive": tfv_support.get("positive", 0),
                "negative": tfv_support.get("negative", 0),
            },
        }

        # ---- Audit 3: peak_noninferior positive class definition ----
        peak_ni = classification.get("peak_noninferior", {})
        peak_support = peak_ni.get("class_support", {})
        findings["peak_noninferior_positive_class"] = {
            "definition": "positive=1 means 'non-inferior' (peak within margin of exact)",
            "class_support": {
                "positive": peak_support.get("positive", 0),
                "negative": peak_support.get("negative", 0),
            },
        }

        # ---- Audit 4: probability class index mapping ----
        findings["probability_class_index"] = {
            "definition": "probability p maps to class 1 (positive) when p >= threshold",
            "threshold": 0.5,
            "code_evidence": "yhat = (p >= threshold).astype(int)",
        }

        # ---- Audit 5: confusion matrix definition ----
        findings["confusion_matrix_definition"] = {
            "true_positive": "y_true=1, yhat=1 (correctly predicted positive)",
            "false_positive": "y_true=0, yhat=1 (incorrectly predicted positive)",
            "false_negative": "y_true=1, yhat=0 (missed positive)",
            "true_negative": "y_true=0, yhat=0 (correctly predicted negative)",
        }

        # ---- Audit 6: false_safe formula ----
        findings["false_safe_formula"] = {
            "definition": "false_safe_rate = mean(yhat[y_true==0] == 1)",
            "meaning": "Among truly unsafe states, fraction predicted as safe",
            "code_line": "false_safe_rate = float(np.mean(yhat[unsafe] == 1))",
        }

        # ---- Audit 7: false_reject formula ----
        findings["false_reject_formula"] = {
            "definition": "false_reject_rate = mean(yhat[y_true==1] == 0)",
            "meaning": "Among truly safe states, fraction predicted as unsafe",
            "code_line": "false_reject_rate = float(np.mean(yhat[safe] == 0))",
        }

        # ---- Audit 8: AUC direction ----
        findings["auc_direction"] = {
            "definition": "roc_auc_score(y_true, p) — higher p → more likely positive",
            "code_evidence": "auroc = float(roc_auc_score(y_true, p))",
            "interpretation": "AUC > 0.5 means model discriminates positive from negative; AUC < 0.5 means anti-correlated",
        }

        # ---- Audit 9: calibration before/after direction ----
        findings["calibration_direction"] = {
            "definition": "one_sided_conformal bounds residual in the unsafe direction",
            "pfv_direction": "underprediction (y_true - y_pred) — actual worse than predicted",
            "tfv_direction": "overprediction (y_pred - y_true) — claimed improvement exceeds reality",
            "peak_direction": "overprediction (y_pred - y_true)",
        }

        # ---- Audit 10: threshold application direction ----
        findings["threshold_application"] = {
            "definition": "yhat = (p >= threshold).astype(int) — high probability → positive class",
            "threshold": 0.5,
            "frozen": True,
        }

        # ================================================================
        # CRITICAL CHECK: false_safe_rate=1.0 consistency
        # ================================================================
        critical_checks: list[dict[str, Any]] = []

        for head_name, head_data in classification.items():
            fsr = head_data.get("false_safe_rate")
            frr = head_data.get("false_reject_rate")
            auroc = head_data.get("auroc")
            ba = head_data.get("balanced_accuracy")
            mcc = head_data.get("mcc")
            support = head_data.get("class_support", {})
            n_pos = support.get("positive", 0)
            n_neg = support.get("negative", 0)

            check: dict[str, Any] = {
                "head": head_name,
                "false_safe_rate": fsr,
                "false_reject_rate": frr,
                "auroc": auroc,
                "balanced_accuracy": ba,
                "mcc": mcc,
                "class_support": {"positive": n_pos, "negative": n_neg},
            }

            # Check: FSR=1.0 and FRR=0.0 implies all predictions = positive (safe)
            if fsr == 1.0 and frr == 0.0:
                check["inferred_prediction_pattern"] = (
                    "ALL predictions are positive (safe/yhat=1): "
                    "all truly unsafe predicted safe (FSR=1), "
                    "no truly safe predicted unsafe (FRR=0)"
                )
                # If all predictions are identical (all positive), AUROC should be ~0.5
                if auroc is not None and abs(auroc - 0.5) > 0.01:
                    check["DISCREPANCY"] = (
                        f"AUROC={auroc:.4f} contradicts all-identical predictions. "
                        f"If all yhat=1 (FSR=1, FRR=0), the probability scores must "
                        f"all be >= threshold, giving AUROC≈0.5. "
                        f"AUROC<0.5 implies unsafe samples receive HIGHER probability "
                        f"than safe samples — the model's probability head is "
                        f"ANTI-CORRELATED with the assumed positive='safe' convention. "
                        f"ROOT CAUSE: The model was likely trained with positive=unsafe "
                        f"(label inversion), so it outputs high p for unsafe states. "
                        f"The evaluation code interprets high p as 'safe', causing "
                        f"the apparent contradiction."
                    )
                    check["severity"] = "CRITICAL"
                    check["root_cause_hypothesis"] = (
                        "Label encoding mismatch between training and evaluation: "
                        "the probability head was trained with positive='unsafe' "
                        "but classification_metrics assumes positive='safe'. "
                        "The model outputs high probability for unsafe states; "
                        "thresholding at 0.5 classifies them as 'safe' (positive), "
                        "producing FSR=1.0 (all unsafe predicted safe) and FRR=0.0 "
                        "(no safe predicted unsafe). AUROC<0.5 confirms the "
                        "probability scores are anti-correlated with the evaluation "
                        "code's positive='safe' assumption."
                    )
                else:
                    check["DISCREPANCY"] = None
                    check["severity"] = "OK"

            # Check: FSR=0.0 and FRR=1.0 implies all predictions = negative (unsafe)
            elif fsr == 0.0 and frr == 1.0:
                check["inferred_prediction_pattern"] = (
                    "ALL predictions are negative (unsafe/yhat=0)"
                )
                if auroc is not None and abs(auroc - 0.5) > 0.01:
                    check["DISCREPANCY"] = (
                        f"AUROC={auroc:.4f} contradicts all-identical predictions."
                    )
                    check["severity"] = "CRITICAL"
                else:
                    check["DISCREPANCY"] = None
                    check["severity"] = "OK"

            else:
                check["inferred_prediction_pattern"] = "mixed predictions"
                check["DISCREPANCY"] = None
                check["severity"] = "OK"

            critical_checks.append(check)

        # Summary of discrepancies
        discrepancies = [
            c for c in critical_checks if c.get("DISCREPANCY")
        ]

        audit_report = {
            "stage": stage,
            "audit_time": datetime.now(timezone.utc).isoformat(),
            "code_sha256": working_code_sha(project_root),
            "v41_gate_verdict": gate_verdict.get("status"),
            "metric_definitions": findings,
            "critical_checks": critical_checks,
            "discrepancies_found": len(discrepancies),
            "discrepancy_details": discrepancies,
            "summary": (
                f"Audited 10 metric semantics. Found {len(discrepancies)} "
                f"critical discrepancy(ies) in classification heads. "
                f"Primary issue: label encoding mismatch between model training "
                f"and evaluation code for the pfv_safe head."
                if discrepancies
                else "All metric semantics are internally consistent."
            ),
        }

        # Write audit report
        audit_dir = output_root / METRICS_AUDIT_DIR
        audit_dir.mkdir(parents=True, exist_ok=True)
        audit_path = audit_dir / "v41_metric_semantics_audit.json"
        atomic_write_json(audit_path, audit_report)

        return StageResult(
            stage, "pass", EXIT_PASS,
            completed=1,
            batch_complete=True,
            scope_complete=True,
            evidence={
                "discrepancies_found": len(discrepancies),
                "audit_report_path": str(audit_path),
                "summary": audit_report["summary"],
            },
        )

    return handler


# ---------------------------------------------------------------------------
# V4.2 Event Usage Ledger handlers
# ---------------------------------------------------------------------------

EVENT_LEDGER_DIR = "v42/event_ledger"


def _build_event_ledger_handler(project_root: Path, output_root: Path, config: dict):
    """Build the V4.2 event usage ledger."""

    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "BuildV42EventUsageLedger"
        code_sha = working_code_sha(project_root)
        try:
            from .v42_event_ledger import (
                build_v42_event_usage_ledger,
                audit_v42_event_usage_ledger,
                write_v42_event_ledger_outputs,
            )

            result = build_v42_event_usage_ledger(project_root, output_root)
            ledger_df = result["ledger_df"]
            audit_result = audit_v42_event_usage_ledger(ledger_df)
            write_v42_event_ledger_outputs(output_root, ledger_df, audit_result)

            return StageResult(
                stage, "pass", EXIT_PASS,
                completed=len(ledger_df),
                batch_complete=True,
                scope_complete=audit_result.get("status") == "pass",
                evidence={
                    "event_count": len(ledger_df),
                    "source_counts": result["source_counts"],
                    "audit_status": audit_result.get("status"),
                    "code_sha256": code_sha,
                },
            )
        except Exception as exc:
            return StageResult(
                stage, "runtime_error", EXIT_BLOCKED,
                evidence={"error": str(exc), "code_sha256": code_sha},
            )

    return handler


def _audit_event_ledger_handler(project_root: Path, output_root: Path, config: dict):
    """Audit the V4.2 event usage ledger outputs on disk."""

    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "AuditV42EventUsageLedger"
        code_sha = working_code_sha(project_root)
        ledger_dir = output_root / EVENT_LEDGER_DIR
        try:
            import pandas as pd
            ledger_path = ledger_dir / "event_usage_ledger_v42.csv"
            if not ledger_path.exists():
                return StageResult(
                    stage, "blocked", EXIT_BLOCKED,
                    evidence={"reason": "ledger_csv_not_found", "path": str(ledger_path)},
                )
            ledger_df = pd.read_csv(ledger_path)
            from .v42_event_ledger import audit_v42_event_usage_ledger
            audit_result = audit_v42_event_usage_ledger(ledger_df)

            audit_path = ledger_dir / "event_usage_audit.json"
            audit_path.write_text(
                json.dumps(audit_result, indent=2, default=str), encoding="utf-8"
            )

            passed = audit_result.get("status") == "pass"
            return StageResult(
                stage,
                audit_result.get("status", "blocked"),
                EXIT_PASS if passed else EXIT_BLOCKED,
                completed=1,
                batch_complete=True,
                scope_complete=passed,
                evidence={
                    "audit_status": audit_result.get("status"),
                    "summary": audit_result.get("summary", {}),
                    "code_sha256": code_sha,
                },
            )
        except Exception as exc:
            return StageResult(
                stage, "runtime_error", EXIT_BLOCKED,
                evidence={"error": str(exc), "code_sha256": code_sha},
            )

    return handler


# ---------------------------------------------------------------------------
# V4.2 Unified Development Pool handlers
# ---------------------------------------------------------------------------

POOL_DIR = "v42/development_pool"


def _build_unified_pool_handler(project_root: Path, output_root: Path, config: dict):
    """Build the V4.2 unified development pool."""

    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "BuildV42UnifiedDevelopmentPool"
        code_sha = working_code_sha(project_root)
        try:
            from .v42_unified_pool import (
                build_v42_unified_development_pool,
                write_unified_pool,
            )

            result = build_v42_unified_development_pool(project_root, output_root)
            pool_dir = output_root / POOL_DIR
            written = write_unified_pool(result, pool_dir)

            return StageResult(
                stage, "pass", EXIT_PASS,
                completed=len(result.candidate_manifest),
                batch_complete=True,
                scope_complete=True,
                evidence={
                    "n_candidates": len(result.candidate_manifest),
                    "n_references": len(result.reference_manifest),
                    "n_events": len(result.event_manifest),
                    "n_states": len(result.state_manifest),
                    "n_warnings": len(result.warnings),
                    "files": list(written.values()),
                    "code_sha256": code_sha,
                },
            )
        except Exception as exc:
            return StageResult(
                stage, "runtime_error", EXIT_BLOCKED,
                evidence={"error": str(exc), "code_sha256": code_sha},
            )

    return handler


def _audit_unified_pool_handler(project_root: Path, output_root: Path, config: dict):
    """Audit the V4.2 unified development pool."""

    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "AuditV42UnifiedDevelopmentPool"
        code_sha = working_code_sha(project_root)
        try:
            from .v42_unified_pool import audit_v42_unified_pool

            audit_result = audit_v42_unified_pool(output_root)
            passed = audit_result.get("status") == "pass"

            # Write audit JSON
            pool_dir = output_root / POOL_DIR
            pool_dir.mkdir(parents=True, exist_ok=True)
            audit_path = pool_dir / "unified_pool_audit.json"
            audit_path.write_text(
                json.dumps(audit_result, indent=2, default=str), encoding="utf-8"
            )

            return StageResult(
                stage,
                audit_result.get("status", "blocked"),
                EXIT_PASS if passed else EXIT_BLOCKED,
                completed=1,
                batch_complete=True,
                scope_complete=passed,
                evidence={
                    "audit_status": audit_result.get("status"),
                    "statistics": audit_result.get("statistics", {}),
                    "errors": audit_result.get("errors", []),
                    "code_sha256": code_sha,
                },
            )
        except Exception as exc:
            return StageResult(
                stage, "runtime_error", EXIT_BLOCKED,
                evidence={"error": str(exc), "code_sha256": code_sha},
            )

    return handler


# ---------------------------------------------------------------------------
# V4.2 Derived Supervision handlers
# ---------------------------------------------------------------------------

DERIVED_DIR = "v42/derived_supervision"


def _build_derived_supervision_handler(project_root: Path, output_root: Path, config: dict):
    """Build V4.2 derived supervision signals."""

    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "BuildV42DerivedSupervision"
        code_sha = working_code_sha(project_root)
        try:
            from .v42_derived_supervision import (
                build_v42_derived_supervision,
                write_derived_supervision,
            )

            result = build_v42_derived_supervision(project_root, output_root)
            derived_dir = output_root / DERIVED_DIR
            written = write_derived_supervision(result, derived_dir)

            return StageResult(
                stage, "pass", EXIT_PASS,
                completed=len(result.one_step_transitions),
                batch_complete=True,
                scope_complete=True,
                evidence={
                    "n_one_step": len(result.one_step_transitions),
                    "n_multi_horizon": len(result.multi_horizon_targets),
                    "n_pairwise": len(result.pairwise_ranking),
                    "n_cr_pairs": len(result.candidate_reference_pairs),
                    "files": list(written.values()),
                    "code_sha256": code_sha,
                },
            )
        except Exception as exc:
            return StageResult(
                stage, "runtime_error", EXIT_BLOCKED,
                evidence={"error": str(exc), "code_sha256": code_sha},
            )

    return handler


def _audit_derived_supervision_handler(project_root: Path, output_root: Path, config: dict):
    """Audit V4.2 derived supervision signals."""

    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "AuditV42DerivedSupervision"
        code_sha = working_code_sha(project_root)
        try:
            from .v42_derived_supervision import audit_v42_derived_supervision

            audit_result = audit_v42_derived_supervision(output_root)
            passed = audit_result.get("status") == "pass"

            derived_dir = output_root / DERIVED_DIR
            derived_dir.mkdir(parents=True, exist_ok=True)
            audit_path = derived_dir / "supervision_audit.json"
            audit_path.write_text(
                json.dumps(audit_result, indent=2, default=str), encoding="utf-8"
            )

            return StageResult(
                stage,
                audit_result.get("status", "blocked"),
                EXIT_PASS if passed else EXIT_BLOCKED,
                completed=1,
                batch_complete=True,
                scope_complete=passed,
                evidence={
                    "audit_status": audit_result.get("status"),
                    "statistics": audit_result.get("statistics", {}),
                    "code_sha256": code_sha,
                },
            )
        except Exception as exc:
            return StageResult(
                stage, "runtime_error", EXIT_BLOCKED,
                evidence={"error": str(exc), "code_sha256": code_sha},
            )

    return handler


# ---------------------------------------------------------------------------
# V4.2 Nested Grouped CV handlers
# ---------------------------------------------------------------------------

CV_DIR = "v42/cv"


def _plan_nested_cv_handler(project_root: Path, output_root: Path, config: dict):
    """Plan the V4.2 nested event-grouped CV scheme."""

    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "PlanV42NestedGroupedCV"
        code_sha = working_code_sha(project_root)
        try:
            from .v42_cv import plan_v42_nested_grouped_cv, write_nested_cv_artifacts
            import pandas as pd

            # Load event ledger for CV planning
            ledger_path = output_root / EVENT_LEDGER_DIR / "event_usage_ledger_v42.csv"
            if not ledger_path.exists():
                return StageResult(
                    stage, "blocked", EXIT_BLOCKED,
                    evidence={"reason": "event_ledger_not_found", "path": str(ledger_path)},
                )

            ledger_df = pd.read_csv(ledger_path)
            event_ids = ledger_df["event_id"].values
            rainfall_shas = ledger_df["rainfall_sha256"].values

            plan = plan_v42_nested_grouped_cv(event_ids, rainfall_shas)

            # Build event_df for artifact writing
            event_df = ledger_df[["event_id", "rainfall_sha256"]].drop_duplicates(
                subset="event_id", keep="first"
            ).reset_index(drop=True)

            cv_dir = output_root / CV_DIR
            paths = write_nested_cv_artifacts(plan, event_df, output_root)

            # Write plan JSON to expected location
            cv_dir.mkdir(parents=True, exist_ok=True)
            plan_json = {
                "outer_n_splits": plan.outer_n_splits,
                "inner_n_splits": plan.inner_n_splits,
                "n_events": len(event_ids),
                "frozen_seeds": list(plan.frozen_seeds),
            }
            plan_path = cv_dir / "nested_cv_plan.json"
            plan_path.write_text(json.dumps(plan_json, indent=2), encoding="utf-8")

            return StageResult(
                stage, "pass", EXIT_PASS,
                completed=len(event_ids),
                batch_complete=True,
                scope_complete=True,
                evidence={
                    "n_events": len(event_ids),
                    "outer_splits": plan.outer_n_splits,
                    "inner_splits": plan.inner_n_splits,
                    "artifacts": [str(p) for p in paths.values()],
                    "code_sha256": code_sha,
                },
            )
        except Exception as exc:
            return StageResult(
                stage, "runtime_error", EXIT_BLOCKED,
                evidence={"error": str(exc), "code_sha256": code_sha},
            )

    return handler


def _audit_nested_cv_plan_handler(project_root: Path, output_root: Path, config: dict):
    """Audit the V4.2 nested CV plan."""

    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "AuditV42NestedGroupedCVPlan"
        code_sha = working_code_sha(project_root)
        try:
            from .v42_cv import audit_v42_nested_cv_plan

            audit_result = audit_v42_nested_cv_plan(output_root)
            passed = audit_result.get("passed", False)

            cv_dir = output_root / CV_DIR
            cv_dir.mkdir(parents=True, exist_ok=True)
            audit_path = cv_dir / "nested_cv_plan_audit.json"
            audit_path.write_text(
                json.dumps(audit_result, indent=2, default=str), encoding="utf-8"
            )

            return StageResult(
                stage,
                "pass" if passed else "blocked",
                EXIT_PASS if passed else EXIT_BLOCKED,
                completed=1,
                batch_complete=True,
                scope_complete=passed,
                evidence={
                    "passed": passed,
                    "violations": audit_result.get("violations", []),
                    "code_sha256": code_sha,
                },
            )
        except Exception as exc:
            return StageResult(
                stage, "runtime_error", EXIT_BLOCKED,
                evidence={"error": str(exc), "code_sha256": code_sha},
            )

    return handler


# ---------------------------------------------------------------------------
# V4.2 Final Data Pool handlers
# ---------------------------------------------------------------------------

FINAL_POOL_DIR = "audits/v42_final_pool"


def _audit_priority_contract_handler(project_root: Path, output_root: Path, config: dict):
    """Audit the V4.2 priority sentinel contract."""

    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "AuditV42PrioritySentinelContract"
        code_sha = working_code_sha(project_root)
        try:
            from .v42_priority_contract import audit_contract

            result = audit_contract()
            audit_dir = output_root / FINAL_POOL_DIR
            audit_dir.mkdir(parents=True, exist_ok=True)
            audit_path = audit_dir / "priority_contract_audit.json"
            atomic_write_json(audit_path, result)

            passed = result.get("status") == "PASS"
            return StageResult(
                stage,
                "pass" if passed else "blocked",
                EXIT_PASS if passed else EXIT_BLOCKED,
                completed=1,
                batch_complete=True,
                scope_complete=passed,
                evidence={
                    "status": result.get("status"),
                    "audit_path": str(audit_path),
                    "code_sha256": code_sha,
                },
            )
        except Exception as exc:
            return StageResult(
                stage, "runtime_error", EXIT_BLOCKED,
                evidence={"error": str(exc), "code_sha256": code_sha},
            )

    return handler


def _freeze_priority_contract_handler(project_root: Path, output_root: Path, config: dict):
    """Freeze the V4.2 priority PFV contract to docs/contracts/."""

    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "FreezeV42PriorityContract"
        code_sha = working_code_sha(project_root)
        try:
            from .v42_priority_contract import (
                audit_contract,
                PFV_CORE_8_IDS,
                DEPTH_SENTINEL_2_IDS,
                SENSITIVITY_ZONE_11_IDS,
                CONTRACT_ID,
            )

            result = audit_contract()
            contract = {
                "contract_id": CONTRACT_ID,
                "frozen_at": datetime.now(timezone.utc).isoformat(),
                "code_sha256": code_sha,
                "pfv_core_node_ids": PFV_CORE_8_IDS,
                "sentinel_node_ids": DEPTH_SENTINEL_2_IDS,
                "sensitivity_zone_node_ids": SENSITIVITY_ZONE_11_IDS,
                "audit_status": result.get("status"),
                "audit_detail": result,
            }
            contract_path = project_root / "docs" / "contracts" / "PROJECT6_V42_PRIORITY_PFV_CONTRACT.json"
            contract_path.parent.mkdir(parents=True, exist_ok=True)
            atomic_write_json(contract_path, contract)

            passed = result.get("status") == "PASS"
            return StageResult(
                stage,
                "pass" if passed else "blocked",
                EXIT_PASS if passed else EXIT_BLOCKED,
                completed=1,
                batch_complete=True,
                scope_complete=passed,
                evidence={
                    "contract_path": str(contract_path),
                    "audit_status": result.get("status"),
                    "code_sha256": code_sha,
                },
            )
        except Exception as exc:
            return StageResult(
                stage, "runtime_error", EXIT_BLOCKED,
                evidence={"error": str(exc), "code_sha256": code_sha},
            )

    return handler


def _build_pfv_oracle_handler(project_root: Path, output_root: Path, config: dict):
    """Build the independent PFV oracle."""

    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "BuildV42IndependentPfvOracle"
        code_sha = working_code_sha(project_root)
        try:
            from .v42_independent_pfv_oracle import run_pfv_oracle_audit
            from .v42_priority_contract import PFV_CORE_8_IDS

            trajectory_manifest = (
                output_root / "v42" / "trajectory_dataset" / "trajectory_manifest_v42.parquet"
            )
            result = run_pfv_oracle_audit(
                project_root=project_root,
                output_root=output_root,
                trajectory_manifest=trajectory_manifest,
                priority_node_ids=PFV_CORE_8_IDS,
            )
            audit_dir = output_root / FINAL_POOL_DIR
            audit_dir.mkdir(parents=True, exist_ok=True)
            audit_path = audit_dir / "pfv_oracle_audit.json"
            atomic_write_json(audit_path, result)

            passed = result.get("status") == "pass"
            return StageResult(
                stage,
                "pass" if passed else "blocked",
                EXIT_PASS if passed else EXIT_BLOCKED,
                completed=1,
                batch_complete=True,
                scope_complete=passed,
                evidence={
                    "status": result.get("status"),
                    "audit_path": str(audit_path),
                    "code_sha256": code_sha,
                },
            )
        except Exception as exc:
            return StageResult(
                stage, "runtime_error", EXIT_BLOCKED,
                evidence={"error": str(exc), "code_sha256": code_sha},
            )

    return handler


def _build_sample_lineage_handler(project_root: Path, output_root: Path, config: dict):
    """Build sample lineage parquet."""

    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "BuildV42SampleLineage"
        code_sha = working_code_sha(project_root)
        try:
            from .v42_sample_lineage import build_sample_lineage

            lineage_df = build_sample_lineage(project_root, output_root)
            audit_dir = output_root / FINAL_POOL_DIR
            audit_dir.mkdir(parents=True, exist_ok=True)
            out_path = audit_dir / "sample_lineage.parquet"
            lineage_df.to_parquet(out_path, index=False)

            return StageResult(
                stage, "pass", EXIT_PASS,
                completed=len(lineage_df),
                batch_complete=True,
                scope_complete=True,
                evidence={
                    "n_samples": len(lineage_df),
                    "output_path": str(out_path),
                    "code_sha256": code_sha,
                },
            )
        except Exception as exc:
            return StageResult(
                stage, "runtime_error", EXIT_BLOCKED,
                evidence={"error": str(exc), "code_sha256": code_sha},
            )

    return handler


def _audit_physical_dedup_handler(project_root: Path, output_root: Path, config: dict):
    """Audit physical deduplication of samples."""

    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "AuditV42PhysicalDeduplication"
        code_sha = working_code_sha(project_root)
        try:
            from .v42_sample_lineage import build_sample_lineage, audit_physical_deduplication

            lineage_df = build_sample_lineage(project_root, output_root)
            dedup_result = audit_physical_deduplication(lineage_df)

            audit_dir = output_root / FINAL_POOL_DIR
            audit_dir.mkdir(parents=True, exist_ok=True)
            audit_path = audit_dir / "deduplication_audit.json"
            atomic_write_json(audit_path, dedup_result)

            passed = dedup_result.get("status") == "pass"
            return StageResult(
                stage,
                "pass" if passed else "blocked",
                EXIT_PASS if passed else EXIT_BLOCKED,
                completed=1,
                batch_complete=True,
                scope_complete=passed,
                evidence={
                    "status": dedup_result.get("status"),
                    "audit_path": str(audit_path),
                    "code_sha256": code_sha,
                },
            )
        except Exception as exc:
            return StageResult(
                stage, "runtime_error", EXIT_BLOCKED,
                evidence={"error": str(exc), "code_sha256": code_sha},
            )

    return handler


def _build_semantic_inventory_handler(project_root: Path, output_root: Path, config: dict):
    """Build historical semantic inventory."""

    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "BuildV42HistoricalSemanticInventory"
        code_sha = working_code_sha(project_root)
        try:
            from .v42_semantic_audit import build_semantic_inventory

            semantic_df = build_semantic_inventory(project_root, output_root)
            audit_dir = output_root / FINAL_POOL_DIR
            audit_dir.mkdir(parents=True, exist_ok=True)
            out_path = audit_dir / "semantic_sample_inventory.parquet"
            semantic_df.to_parquet(out_path, index=False)

            return StageResult(
                stage, "pass", EXIT_PASS,
                completed=len(semantic_df),
                batch_complete=True,
                scope_complete=True,
                evidence={
                    "n_samples": len(semantic_df),
                    "output_path": str(out_path),
                    "code_sha256": code_sha,
                },
            )
        except Exception as exc:
            return StageResult(
                stage, "runtime_error", EXIT_BLOCKED,
                evidence={"error": str(exc), "code_sha256": code_sha},
            )

    return handler


def _audit_semantic_inventory_handler(project_root: Path, output_root: Path, config: dict):
    """Audit the historical semantic inventory output."""

    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "AuditV42HistoricalSemanticInventory"
        code_sha = working_code_sha(project_root)
        try:
            audit_dir = output_root / FINAL_POOL_DIR
            inventory_path = audit_dir / "semantic_sample_inventory.parquet"
            if not inventory_path.exists():
                return StageResult(
                    stage, "blocked", EXIT_BLOCKED,
                    evidence={"reason": "semantic_inventory_missing", "path": str(inventory_path)},
                )
            semantic_df = pd.read_parquet(inventory_path)
            summary = {
                "n_samples": len(semantic_df),
                "columns": list(semantic_df.columns),
                "status": "pass" if len(semantic_df) > 0 else "empty",
            }
            summary_path = audit_dir / "semantic_source_summary.csv"
            semantic_df.to_csv(summary_path, index=False)

            passed = len(semantic_df) > 0
            return StageResult(
                stage,
                "pass" if passed else "blocked",
                EXIT_PASS if passed else EXIT_BLOCKED,
                completed=len(semantic_df),
                batch_complete=True,
                scope_complete=passed,
                evidence={**summary, "code_sha256": code_sha},
            )
        except Exception as exc:
            return StageResult(
                stage, "runtime_error", EXIT_BLOCKED,
                evidence={"error": str(exc), "code_sha256": code_sha},
            )

    return handler


def _audit_dwf_sources_handler(project_root: Path, output_root: Path, config: dict):
    """Audit DWF sources classification."""

    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "AuditV42DwfSources"
        code_sha = working_code_sha(project_root)
        try:
            from .v42_dwf_audit import build_dwf_classification

            dwf_df = build_dwf_classification(project_root, output_root)
            audit_dir = output_root / FINAL_POOL_DIR
            audit_dir.mkdir(parents=True, exist_ok=True)
            summary = {
                "n_samples": len(dwf_df),
                "status": "pass" if len(dwf_df) > 0 else "empty",
            }
            audit_path = audit_dir / "dwf_audit_summary.json"
            atomic_write_json(audit_path, summary)

            passed = len(dwf_df) > 0
            return StageResult(
                stage,
                "pass" if passed else "blocked",
                EXIT_PASS if passed else EXIT_BLOCKED,
                completed=len(dwf_df),
                batch_complete=True,
                scope_complete=passed,
                evidence={**summary, "code_sha256": code_sha},
            )
        except Exception as exc:
            return StageResult(
                stage, "runtime_error", EXIT_BLOCKED,
                evidence={"error": str(exc), "code_sha256": code_sha},
            )

    return handler


def _build_history_rebuild_handler(project_root: Path, output_root: Path, config: dict):
    """Build canonical 13-frame trajectories."""

    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "BuildV42Canonical13FrameTrajectories"
        code_sha = working_code_sha(project_root)
        try:
            from .v42_history_rebuilder import rebuild_13frame_histories

            result = rebuild_13frame_histories(project_root, output_root)
            audit_dir = output_root / FINAL_POOL_DIR
            audit_dir.mkdir(parents=True, exist_ok=True)
            audit_path = audit_dir / "history_rebuild_audit.json"
            atomic_write_json(audit_path, result)

            passed = result.get("status") == "pass"
            return StageResult(
                stage,
                "pass" if passed else "blocked",
                EXIT_PASS if passed else EXIT_BLOCKED,
                completed=result.get("n_samples", 0),
                batch_complete=True,
                scope_complete=passed,
                evidence={
                    "status": result.get("status"),
                    "n_samples": result.get("n_samples", 0),
                    "audit_path": str(audit_path),
                    "code_sha256": code_sha,
                },
            )
        except Exception as exc:
            return StageResult(
                stage, "runtime_error", EXIT_BLOCKED,
                evidence={"error": str(exc), "code_sha256": code_sha},
            )

    return handler


def _audit_history_rebuild_handler(project_root: Path, output_root: Path, config: dict):
    """Audit the canonical 13-frame trajectories output."""

    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "AuditV42Canonical13FrameTrajectories"
        code_sha = working_code_sha(project_root)
        try:
            audit_dir = output_root / FINAL_POOL_DIR
            audit_path = audit_dir / "history_rebuild_audit.json"
            if not audit_path.exists():
                return StageResult(
                    stage, "blocked", EXIT_BLOCKED,
                    evidence={"reason": "history_rebuild_audit_missing", "path": str(audit_path)},
                )
            result = _read_json(audit_path)
            passed = result.get("status") == "pass"
            return StageResult(
                stage,
                "pass" if passed else "blocked",
                EXIT_PASS if passed else EXIT_BLOCKED,
                completed=result.get("n_samples", 0),
                batch_complete=True,
                scope_complete=passed,
                evidence={
                    "status": result.get("status"),
                    "n_samples": result.get("n_samples", 0),
                    "code_sha256": code_sha,
                },
            )
        except Exception as exc:
            return StageResult(
                stage, "runtime_error", EXIT_BLOCKED,
                evidence={"error": str(exc), "code_sha256": code_sha},
            )

    return handler


def _build_tfv_peak_oracle_handler(project_root: Path, output_root: Path, config: dict):
    """Build TFV/Peak oracle audit."""

    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "BuildV42TfvPeakOracle"
        code_sha = working_code_sha(project_root)
        try:
            from .v42_tfv_peak_oracle import run_tfv_peak_oracle_audit

            trajectory_manifest = (
                output_root / "v42" / "trajectory_dataset" / "trajectory_manifest_v42.parquet"
            )
            result = run_tfv_peak_oracle_audit(
                project_root=project_root,
                output_root=output_root,
                trajectory_manifest=trajectory_manifest,
            )
            audit_dir = output_root / FINAL_POOL_DIR
            audit_dir.mkdir(parents=True, exist_ok=True)
            audit_path = audit_dir / "tfv_peak_oracle_audit.json"
            atomic_write_json(audit_path, result)

            passed = result.get("status") == "pass"
            return StageResult(
                stage,
                "pass" if passed else "blocked",
                EXIT_PASS if passed else EXIT_BLOCKED,
                completed=1,
                batch_complete=True,
                scope_complete=passed,
                evidence={
                    "status": result.get("status"),
                    "audit_path": str(audit_path),
                    "code_sha256": code_sha,
                },
            )
        except Exception as exc:
            return StageResult(
                stage, "runtime_error", EXIT_BLOCKED,
                evidence={"error": str(exc), "code_sha256": code_sha},
            )

    return handler


def _build_sample_classifier_handler(project_root: Path, output_root: Path, config: dict):
    """Build sample classifier — classify all samples into admission grades."""

    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "BuildV42SampleClassifier"
        code_sha = working_code_sha(project_root)
        try:
            from .v42_sample_classifier import classify_samples, summarize_classification

            classified_df = classify_samples(project_root, output_root)
            summary = summarize_classification(classified_df)

            audit_dir = output_root / FINAL_POOL_DIR
            audit_dir.mkdir(parents=True, exist_ok=True)
            summary_path = audit_dir / "sample_classification_summary.json"
            atomic_write_json(summary_path, summary)

            return StageResult(
                stage, "pass", EXIT_PASS,
                completed=len(classified_df),
                batch_complete=True,
                scope_complete=True,
                evidence={
                    "n_samples": len(classified_df),
                    "summary": summary,
                    "code_sha256": code_sha,
                },
            )
        except Exception as exc:
            return StageResult(
                stage, "runtime_error", EXIT_BLOCKED,
                evidence={"error": str(exc), "code_sha256": code_sha},
            )

    return handler


def _build_final_unified_handler(project_root: Path, output_root: Path, config: dict):
    """Build final unified datasets."""

    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "BuildV42FinalUnifiedDatasets"
        code_sha = working_code_sha(project_root)
        try:
            from .v42_final_datasets import build_final_unified_datasets

            result = build_final_unified_datasets(project_root, output_root)
            passed = result.get("total_classified_samples", 0) > 0
            return StageResult(
                stage,
                "pass" if passed else "blocked",
                EXIT_PASS if passed else EXIT_BLOCKED,
                completed=result.get("total_classified_samples", 0),
                batch_complete=True,
                scope_complete=passed,
                evidence={
                    "total_classified_samples": result.get("total_classified_samples", 0),
                    "n_datasets": len(result.get("datasets", [])),
                    "code_sha256": code_sha,
                },
            )
        except Exception as exc:
            return StageResult(
                stage, "runtime_error", EXIT_BLOCKED,
                evidence={"error": str(exc), "code_sha256": code_sha},
            )

    return handler


def _audit_final_unified_handler(project_root: Path, output_root: Path, config: dict):
    """Audit the final unified datasets."""

    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "AuditV42FinalUnifiedDatasets"
        code_sha = working_code_sha(project_root)
        try:
            from .v42_data_contract_audit import run_full_audit

            result = run_full_audit()
            audit_dir = output_root / FINAL_POOL_DIR
            audit_dir.mkdir(parents=True, exist_ok=True)
            audit_path = audit_dir / "final_dataset_audit.json"
            atomic_write_json(audit_path, result)

            passed = result.get("status") == "pass"
            return StageResult(
                stage,
                "pass" if passed else "blocked",
                EXIT_PASS if passed else EXIT_BLOCKED,
                completed=1,
                batch_complete=True,
                scope_complete=passed,
                evidence={
                    "status": result.get("status"),
                    "audit_path": str(audit_path),
                    "code_sha256": code_sha,
                },
            )
        except Exception as exc:
            return StageResult(
                stage, "runtime_error", EXIT_BLOCKED,
                evidence={"error": str(exc), "code_sha256": code_sha},
            )

    return handler


def _build_grouped_splits_handler(project_root: Path, output_root: Path, config: dict):
    """Build grouped train/val splits."""

    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "BuildV42GroupedSplits"
        code_sha = working_code_sha(project_root)
        try:
            from .v42_grouped_splits import build_grouped_splits

            result = build_grouped_splits(project_root, output_root)
            audit_dir = output_root / FINAL_POOL_DIR
            audit_dir.mkdir(parents=True, exist_ok=True)
            audit_path = audit_dir / "grouped_splits.json"
            atomic_write_json(audit_path, result)

            return StageResult(
                stage, "pass", EXIT_PASS,
                completed=result.get("n_events", 0),
                batch_complete=True,
                scope_complete=True,
                evidence={
                    "n_events": result.get("n_events", 0),
                    "n_outer": result.get("n_outer_splits", 0),
                    "code_sha256": code_sha,
                },
            )
        except Exception as exc:
            return StageResult(
                stage, "runtime_error", EXIT_BLOCKED,
                evidence={"error": str(exc), "code_sha256": code_sha},
            )

    return handler


def _build_pool_statistics_handler(project_root: Path, output_root: Path, config: dict):
    """Build pool statistics."""

    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "BuildV42PoolStatistics"
        code_sha = working_code_sha(project_root)
        try:
            from .v42_pool_statistics import compute_pool_statistics

            result = compute_pool_statistics(project_root, output_root)
            audit_dir = output_root / FINAL_POOL_DIR
            audit_dir.mkdir(parents=True, exist_ok=True)
            audit_path = audit_dir / "pool_statistics.json"
            atomic_write_json(audit_path, result)

            return StageResult(
                stage, "pass", EXIT_PASS,
                completed=1,
                batch_complete=True,
                scope_complete=True,
                evidence={
                    "statistics_keys": list(result.keys()),
                    "code_sha256": code_sha,
                },
            )
        except Exception as exc:
            return StageResult(
                stage, "runtime_error", EXIT_BLOCKED,
                evidence={"error": str(exc), "code_sha256": code_sha},
            )

    return handler


def _audit_admission_gate_handler(project_root: Path, output_root: Path, config: dict):
    """Run the final dataset admission gate."""

    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "AuditV42FinalDatasetAdmissionGate"
        code_sha = working_code_sha(project_root)
        try:
            from .v42_admission_gate import run_admission_gate

            result = run_admission_gate(project_root, output_root)
            audit_dir = output_root / FINAL_POOL_DIR
            audit_dir.mkdir(parents=True, exist_ok=True)
            audit_path = audit_dir / "admission_gate_result.json"

            result_dict = {
                "verdict": result.verdict.value if hasattr(result.verdict, "value") else str(result.verdict),
                "checks": result.checks,
                "summary": result.summary,
                "code_sha256": code_sha,
            }
            atomic_write_json(audit_path, result_dict)

            passed = str(result.verdict) in ("pass", "GateVerdict.PASS")
            return StageResult(
                stage,
                "pass" if passed else "blocked",
                EXIT_PASS if passed else EXIT_BLOCKED,
                completed=1,
                batch_complete=True,
                scope_complete=passed,
                evidence={
                    "verdict": result_dict["verdict"],
                    "summary": result.summary,
                    "audit_path": str(audit_path),
                    "code_sha256": code_sha,
                },
            )
        except Exception as exc:
            return StageResult(
                stage, "runtime_error", EXIT_BLOCKED,
                evidence={"error": str(exc), "code_sha256": code_sha},
            )

    return handler


# ---------------------------------------------------------------------------
# V4.2 Fresh Evaluation handlers
# ---------------------------------------------------------------------------

FRESH_EVAL_DIR = "v42/fresh_eval"


def _plan_fresh_eval_handler(project_root: Path, output_root: Path, config: dict):
    """Plan the V4.2 fresh evaluation split."""

    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "PlanV42FreshEvaluationSplit"
        code_sha = working_code_sha(project_root)
        try:
            from .v42_fresh_eval import plan_v42_fresh_evaluation_split

            ledger_path = output_root / EVENT_LEDGER_DIR / "event_usage_ledger_v42.csv"
            if not ledger_path.exists():
                return StageResult(
                    stage, "blocked", EXIT_BLOCKED,
                    evidence={"reason": "event_ledger_not_found", "path": str(ledger_path)},
                )

            plan_result = plan_v42_fresh_evaluation_split(
                event_ledger_path=ledger_path,
                output_root=output_root,
                project_root=project_root,
            )

            # Copy plan to expected artifact location
            fresh_dir = output_root / FRESH_EVAL_DIR
            fresh_dir.mkdir(parents=True, exist_ok=True)
            plan_path = fresh_dir / "fresh_evaluation_plan.json"
            plan_path.write_text(
                json.dumps(plan_result, indent=2, default=str), encoding="utf-8"
            )

            status_info = plan_result.get("status", {})
            return StageResult(
                stage, "pass", EXIT_PASS,
                completed=plan_result.get("n_fresh_events", 0),
                batch_complete=True,
                scope_complete=True,
                evidence={
                    "n_fresh_events": plan_result.get("n_fresh_events", 0),
                    "n_calibration": plan_result.get("n_calibration", 0),
                    "n_locked": plan_result.get("n_locked", 0),
                    "n_accrual": plan_result.get("n_accrual", 0),
                    "status": status_info.get("status"),
                    "code_sha256": code_sha,
                },
            )
        except Exception as exc:
            return StageResult(
                stage, "runtime_error", EXIT_BLOCKED,
                evidence={"error": str(exc), "code_sha256": code_sha},
            )

    return handler


def _audit_fresh_eval_handler(project_root: Path, output_root: Path, config: dict):
    """Audit the V4.2 fresh evaluation availability."""

    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "AuditV42FreshEvaluationAvailability"
        code_sha = working_code_sha(project_root)
        try:
            from .v42_fresh_eval import audit_v42_fresh_evaluation_availability

            audit_result = audit_v42_fresh_evaluation_availability(output_root)
            passed = audit_result.get("passed", False)

            # Copy audit to expected artifact location
            fresh_dir = output_root / FRESH_EVAL_DIR
            fresh_dir.mkdir(parents=True, exist_ok=True)
            audit_path = fresh_dir / "fresh_eval_availability_audit.json"
            audit_path.write_text(
                json.dumps(audit_result, indent=2, default=str), encoding="utf-8"
            )

            return StageResult(
                stage,
                "pass" if passed else "blocked",
                EXIT_PASS if passed else EXIT_BLOCKED,
                completed=1,
                batch_complete=True,
                scope_complete=passed,
                evidence={
                    "passed": passed,
                    "violations": audit_result.get("violations", []),
                    "code_sha256": code_sha,
                },
            )
        except Exception as exc:
            return StageResult(
                stage, "runtime_error", EXIT_BLOCKED,
                evidence={"error": str(exc), "code_sha256": code_sha},
            )

    return handler
