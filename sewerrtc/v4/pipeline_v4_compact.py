"""V4.1 Compact Head-Specific Surrogate Rescue -- Phase-1 stages (spec 1-11).

Train-only and fully offline: freeze the V4.0 negative result, diagnose the old
Locked failure (read-only), build the physical feature-block catalog, run the
Train-grouped learning-curve / feature-block / head-architecture ablations, the
multitask gradient audit, the Train-grouped compact-model selection and finally
train the compact head-specific V4.1 model on the original Train 1200.

No SWMM, no closed loop, no new Calibration / Locked.  The old Calibration and
old Locked are never used to pick features, models or hyper-parameters; the old
Locked 200 is read only to *explain* the V4.0 failure (sections 3-4).
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .runtime import (
    EXIT_PASS,
    RuntimeOptions,
    StageResult,
    atomic_write_json,
    working_code_sha,
)
from .pipeline_train_v4 import (
    _blocked,
    _missing_result,
    _read_json,
    _sha_file,
    _thresholds,
)
from .pipeline_train_v4_model import (
    CALIBRATION_REL,
    LOCKED_RESULT_REL,
    MODEL_BLOB_REL,
    MODEL_DIR_REL,
    OFFLINE_GATE_REL,
    TRAIN_SUMMARY_REL,
    _config_sha,
    _model_config,
    _prepare_data,
)
from .train_v4_models import TrueStateEnsemble
from .v4_compact_diag_ops import (
    build_feature_block_catalog,
    generalization_failure_split,
    locked_metric_comparability,
)
from .v4_compact_curve_ops import (
    build_learning_curves,
    diagnose_learning_curves,
    run_feature_block_ablation,
)
from .v4_compact_model_ops import (
    CompactHeadSpecificModel,
    audit_gradient_conflict,
    compact_cv_report,
    run_head_architecture_ablation,
    select_compact_model,
)

# ---------------------------------------------------------------------------
# Output layout (all relative to output_root)
# ---------------------------------------------------------------------------

V0_FREEZE_ROOT_REL = "audits/frozen_evidence/v4_offline_v0"
V0_FREEZE_POINTER_REL = f"{V0_FREEZE_ROOT_REL}/freeze_pointer.json"
V0_FREEZE_NAME = "v4_offline_v0_freeze.json"

DIAG_DIR_REL = "audits/v4_diagnostics"
METRIC_COMPARABILITY_REL = f"{DIAG_DIR_REL}/locked_v0_metric_comparability.json"
GENERALIZATION_FAILURE_REL = f"{DIAG_DIR_REL}/generalization_failure_v0.json"

COMPACT_DIR_REL = "models/v4_compact_v1"
FEATURE_BLOCK_SUMMARY_REL = f"{COMPACT_DIR_REL}/feature_block_catalog_summary.json"
LEARNING_CURVE_SUMMARY_REL = f"{COMPACT_DIR_REL}/learning_curves_summary.json"
FEATURE_ABLATION_REL = f"{COMPACT_DIR_REL}/feature_block_ablation.json"
HEAD_ARCH_ABLATION_REL = f"{COMPACT_DIR_REL}/head_architecture_ablation.json"
GRADIENT_CONFLICT_REL = f"{COMPACT_DIR_REL}/gradient_conflict.json"
SELECTION_REL = f"{COMPACT_DIR_REL}/v4_compact_v1_selection.json"
COMPACT_MODEL_BLOB_REL = f"{COMPACT_DIR_REL}/compact_head_specific_model.pkl"
COMPACT_COMPLETION_REL = f"{COMPACT_DIR_REL}/completion.json"

# Evidence segments of the V4.0 True-state model archived by the freeze.
_V0_FREEZE_FILES = (
    "true_state_model.pkl",
    "true_state_training_summary.json",
    "true_state_calibration.json",
    "locked_evaluation.json",
    "locked_evaluation_intent.json",
    "baseline_models.json",
    "baseline_evaluation.json",
    "offline_safety_gate.json",
    "runtime_identity.json",
    "action_domain_lock.json",
    "training_input_audit.json",
    "feature_leakage_final_audit.json",
    "baseline_full_train_models.pkl",
)


def _round(value, ndigits: int = 6):
    try:
        return round(float(value), ndigits)
    except (TypeError, ValueError):
        return value


# ---------------------------------------------------------------------------
# Section 1: FreezeV4OfflineV0Evidence
# ---------------------------------------------------------------------------

def _freeze_v0_handler(project_root: Path, output_root: Path, config: dict):
    """Immutable archive of the V4.0 True-state model and its negative result.

    No prerequisite chain: the V4.0 evidence lives on disk under a *previous*
    code sha, so this stage re-verifies it directly and re-stamps the freeze
    under the current code sha (mirrors ``FreezeTrain1600V3Evidence``).
    """

    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "FreezeV4OfflineV0Evidence"
        code_sha = working_code_sha(project_root)
        src_dir = output_root / MODEL_DIR_REL
        gate_path = output_root / OFFLINE_GATE_REL
        locked_path = output_root / LOCKED_RESULT_REL
        missing = [
            str(p) for p in (gate_path, locked_path) if not p.exists()
        ]
        if missing:
            return _missing_result(stage, missing)
        gate = _read_json(gate_path)
        if str(gate.get("offline_status")) != "pass":
            return _blocked(
                stage,
                "v4_offline_integrity_gate_not_pass",
                found=str(gate.get("offline_status")),
            )
        locked = _read_json(locked_path)
        cont = locked.get("continuous", {})
        if int(locked.get("n", 0)) <= 0:
            return _blocked(stage, "v4_locked_evaluation_empty")

        pointer_path = output_root / V0_FREEZE_POINTER_REL
        if pointer_path.exists():
            pointer = _read_json(pointer_path)
            prev_dir = output_root / str(pointer.get("frozen_dir_rel", ""))
            prev_freeze = prev_dir / V0_FREEZE_NAME
            if (
                str(pointer.get("code_sha256", "")) == code_sha
                and prev_freeze.exists()
            ):
                prev = _read_json(prev_freeze)
                intact = all(
                    (prev_dir / rel).exists()
                    and _sha_file(prev_dir / rel) == sha
                    for rel, sha in prev.get("file_sha256", {}).items()
                )
                if intact:
                    return StageResult(
                        stage, "pass", EXIT_PASS, completed=1,
                        batch_complete=True, scope_complete=True,
                        evidence={
                            "already_frozen": True, "immutable": True,
                            "frozen_dir": str(prev_dir), "code_sha256": code_sha,
                        },
                    )

        target = output_root / V0_FREEZE_ROOT_REL / code_sha
        target.mkdir(parents=True, exist_ok=True)
        copied: dict[str, str] = {}
        for name in _V0_FREEZE_FILES:
            source = src_dir / name
            if not source.exists():
                continue
            destination = target / name
            if not destination.exists():
                shutil.copy2(source, destination)
            copied[name] = _sha_file(source)

        payload = {
            "stage": stage,
            "model_version": "v4.0",
            "integrity_gate_pass": True,
            "predictive_generalization_pass": False,
            "locked_consumed": True,
            "locked_reusable_for_model_selection": False,
            "pfv_locked_r2": _round(cont.get("pfv", {}).get("r2")),
            "tfv_locked_r2": _round(cont.get("tfv", {}).get("r2")),
            "peak_locked_r2": _round(cont.get("peak", {}).get("r2")),
            "predictive_gate_v0": "scientific_fail",
            "immutable": True,
            "file_sha256": copied,
            "code_sha256": code_sha,
        }
        atomic_write_json(target / V0_FREEZE_NAME, payload)
        pointer_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            pointer_path,
            {
                "stage": stage,
                "frozen_dir_rel": f"{V0_FREEZE_ROOT_REL}/{code_sha}",
                "code_sha256": code_sha,
                "freeze_sha256": _sha_file(target / V0_FREEZE_NAME),
                "predictive_gate_v0": "scientific_fail",
                "immutable": True,
            },
        )
        return StageResult(
            stage, "pass", EXIT_PASS, completed=len(copied),
            batch_complete=True, scope_complete=True,
            evidence={
                "frozen_files": len(copied),
                "pfv_locked_r2": payload["pfv_locked_r2"],
                "tfv_locked_r2": payload["tfv_locked_r2"],
                "peak_locked_r2": payload["peak_locked_r2"],
                "predictive_gate_v0": "scientific_fail",
                "immutable": True,
                "code_sha256": code_sha,
            },
        )

    return handler


# ---------------------------------------------------------------------------
# Section 3: AuditV4LockedMetricComparabilityV0 (old Locked, read-only)
# ---------------------------------------------------------------------------

def _metric_comparability_handler(project_root: Path, output_root: Path, config: dict):
    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "AuditV4LockedMetricComparabilityV0"
        model_path = output_root / MODEL_BLOB_REL
        calib_path = output_root / CALIBRATION_REL
        if not model_path.exists():
            return _missing_result(stage, [str(model_path)])
        data, missing = _prepare_data(output_root, config)
        if data is None:
            return _missing_result(stage, missing)
        _, dead_zones = _thresholds(config)
        cfg = _model_config(config)
        model = TrueStateEnsemble.from_bytes(model_path.read_bytes())
        calibration = _read_json(calib_path) if calib_path.exists() else None
        result = locked_metric_comparability(
            model, data, cfg=cfg, dead_zones=dead_zones, calibration=calibration
        )
        out_dir = output_root / DIAG_DIR_REL
        out_dir.mkdir(parents=True, exist_ok=True)
        result["head_metrics"].to_csv(out_dir / "locked_v0_head_metrics.csv", index=False)
        result["baseline_predictions"].to_parquet(
            out_dir / "locked_v0_baseline_predictions.parquet", index=False
        )
        result["class_support"].to_csv(out_dir / "locked_v0_class_support.csv", index=False)
        atomic_write_json(
            out_dir / "locked_v0_confusion_matrices.json", result["confusion_matrices"]
        )
        report = dict(result["report"])
        report["code_sha256"] = working_code_sha(project_root)
        report["config_sha256"] = _config_sha(_options)
        atomic_write_json(output_root / METRIC_COMPARABILITY_REL, report)
        return StageResult(
            stage, "pass", EXIT_PASS, completed=1,
            batch_complete=True, scope_complete=True,
            evidence={
                "role": "explain_v4_0_failure_only",
                "usable_for_v4_1_selection": False,
                "locked_n": report["locked_n"],
                "any_direction_suspect": report["any_direction_suspect"],
            },
        )

    return handler


# ---------------------------------------------------------------------------
# Section 4: AuditV4GeneralizationFailureV0 (old Locked, diagnostic)
# ---------------------------------------------------------------------------

def _generalization_failure_handler(project_root: Path, output_root: Path, config: dict):
    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "AuditV4GeneralizationFailureV0"
        model_path = output_root / MODEL_BLOB_REL
        if not model_path.exists():
            return _missing_result(stage, [str(model_path)])
        data, missing = _prepare_data(output_root, config)
        if data is None:
            return _missing_result(stage, missing)
        model = TrueStateEnsemble.from_bytes(model_path.read_bytes())
        tables = generalization_failure_split(model, data)
        out_dir = output_root / DIAG_DIR_REL
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, df in tables.items():
            df.to_csv(out_dir / f"{name}.csv", index=False)
        by_event = tables["locked_error_by_event"]
        drift = tables["train_locked_shift_report"]
        top_event = (
            by_event.sort_values(
                [c for c in by_event.columns if c.endswith("_mae")][:1] or ["n"],
                ascending=False,
            ).head(2)
            if not by_event.empty else by_event
        )
        report = {
            "stage": stage,
            "role": "diagnostic_only_not_for_selection",
            "n_locked_events": int(by_event["event_id"].nunique()) if "event_id" in by_event else 0,
            "top_error_events": top_event.to_dict(orient="records"),
            "max_standardized_mean_shift": float(drift["standardized_mean_shift"].max())
            if not drift.empty else None,
            "features_out_of_train_range": int(
                (drift["out_of_train_range_rate"] > 0).sum()
            ) if not drift.empty else 0,
            "tables": sorted(tables),
            "code_sha256": working_code_sha(project_root),
        }
        atomic_write_json(output_root / GENERALIZATION_FAILURE_REL, report)
        return StageResult(
            stage, "pass", EXIT_PASS, completed=1,
            batch_complete=True, scope_complete=True,
            evidence={
                "role": "diagnostic_only_not_for_selection",
                "features_out_of_train_range": report["features_out_of_train_range"],
                "max_standardized_mean_shift": report["max_standardized_mean_shift"],
            },
        )

    return handler


# ---------------------------------------------------------------------------
# Section 5: BuildV4FeatureBlockCatalogV1
# ---------------------------------------------------------------------------

def _feature_block_handler(project_root: Path, output_root: Path, config: dict):
    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "BuildV4FeatureBlockCatalogV1"
        data, missing = _prepare_data(output_root, config)
        if data is None:
            return _missing_result(stage, missing)
        catalog = build_feature_block_catalog(data)
        out_dir = output_root / COMPACT_DIR_REL
        out_dir.mkdir(parents=True, exist_ok=True)
        catalog.to_csv(out_dir / "feature_block_catalog.csv", index=False)
        block_counts = (
            catalog.groupby("physical_block")["feature_name"].count().to_dict()
        )
        summary = {
            "stage": stage,
            "n_features": int(len(catalog)),
            "block_counts": {str(k): int(v) for k, v in block_counts.items()},
            "remove_candidates": int(catalog["remove_candidate"].sum()),
            "remove_reasons": (
                catalog[catalog["remove_candidate"]]["remove_reason"]
                .value_counts().to_dict()
            ),
            "removal_uses_old_locked_error": False,
            "code_sha256": working_code_sha(project_root),
            "config_sha256": _config_sha(_options),
        }
        atomic_write_json(output_root / FEATURE_BLOCK_SUMMARY_REL, summary)
        return StageResult(
            stage, "pass", EXIT_PASS, completed=int(len(catalog)),
            batch_complete=True, scope_complete=True,
            evidence={
                "n_features": summary["n_features"],
                "remove_candidates": summary["remove_candidates"],
                "block_counts": summary["block_counts"],
            },
        )

    return handler


# ---------------------------------------------------------------------------
# Section 6: BuildV4LearningCurvesV1 (Train-only, event-grouped)
# ---------------------------------------------------------------------------

def _learning_curves_handler(project_root: Path, output_root: Path, config: dict):
    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "BuildV4LearningCurvesV1"
        data, missing = _prepare_data(output_root, config)
        if data is None:
            return _missing_result(stage, missing)
        _, dead_zones = _thresholds(config)
        cfg = _model_config(config)
        curve = build_learning_curves(data, cfg=cfg, dead_zones=dead_zones)
        verdicts = diagnose_learning_curves(curve)
        out_dir = output_root / COMPACT_DIR_REL
        out_dir.mkdir(parents=True, exist_ok=True)
        curve.to_csv(out_dir / "learning_curves.csv", index=False)
        summary = {
            "stage": stage,
            "reads_old_calibration": False,
            "reads_old_locked": False,
            "event_grouped": True,
            "verdicts": verdicts,
            "code_sha256": working_code_sha(project_root),
            "config_sha256": _config_sha(_options),
        }
        atomic_write_json(output_root / LEARNING_CURVE_SUMMARY_REL, summary)
        return StageResult(
            stage, "pass", EXIT_PASS, completed=int(len(curve)),
            batch_complete=True, scope_complete=True,
            evidence={"heads": sorted(verdicts), "reads_old_locked": False},
        )

    return handler


# ---------------------------------------------------------------------------
# Section 7: RunV4FeatureBlockAblationV1 (Train event-grouped CV, fold-local)
# ---------------------------------------------------------------------------

def _feature_ablation_handler(project_root: Path, output_root: Path, config: dict):
    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "RunV4FeatureBlockAblationV1"
        data, missing = _prepare_data(output_root, config)
        if data is None:
            return _missing_result(stage, missing)
        _, dead_zones = _thresholds(config)
        cfg = _model_config(config)
        ablation = run_feature_block_ablation(data, cfg=cfg, dead_zones=dead_zones)
        out_dir = output_root / COMPACT_DIR_REL
        out_dir.mkdir(parents=True, exist_ok=True)
        ablation.to_csv(out_dir / "feature_block_ablation.csv", index=False)
        summary = {
            "stage": stage,
            "feature_selection_fold_local": True,
            "select_on_full_train_then_cv": False,
            "reads_old_locked": False,
            "n_combos": int(ablation["combo"].nunique()) if "combo" in ablation else 0,
            "combos": sorted(ablation["combo"].unique().tolist())
            if "combo" in ablation else [],
            "code_sha256": working_code_sha(project_root),
            "config_sha256": _config_sha(_options),
        }
        atomic_write_json(output_root / FEATURE_ABLATION_REL, summary)
        return StageResult(
            stage, "pass", EXIT_PASS, completed=int(len(ablation)),
            batch_complete=True, scope_complete=True,
            evidence={
                "n_combos": summary["n_combos"],
                "feature_selection_fold_local": True,
            },
        )

    return handler


# ---------------------------------------------------------------------------
# Section 8: RunV4HeadArchitectureAblationV1
# ---------------------------------------------------------------------------

def _head_architecture_handler(project_root: Path, output_root: Path, config: dict):
    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "RunV4HeadArchitectureAblationV1"
        data, missing = _prepare_data(output_root, config)
        if data is None:
            return _missing_result(stage, missing)
        _, dead_zones = _thresholds(config)
        cfg = _model_config(config)
        arch = run_head_architecture_ablation(data, cfg=cfg, dead_zones=dead_zones)
        out_dir = output_root / COMPACT_DIR_REL
        out_dir.mkdir(parents=True, exist_ok=True)
        arch.to_csv(out_dir / "head_architecture_ablation.csv", index=False)
        summary = {
            "stage": stage,
            "architectures": ["A", "B", "C", "D"],
            "peak_variants": ["direct", "sequence", "consistency"],
            "reads_old_locked": False,
            "rows": int(len(arch)),
            "code_sha256": working_code_sha(project_root),
            "config_sha256": _config_sha(_options),
        }
        atomic_write_json(output_root / HEAD_ARCH_ABLATION_REL, summary)
        return StageResult(
            stage, "pass", EXIT_PASS, completed=int(len(arch)),
            batch_complete=True, scope_complete=True,
            evidence={"rows": summary["rows"], "reads_old_locked": False},
        )

    return handler


# ---------------------------------------------------------------------------
# Section 9: AuditV4MultitaskGradientConflictV1
# ---------------------------------------------------------------------------

def _gradient_conflict_handler(project_root: Path, output_root: Path, config: dict):
    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "AuditV4MultitaskGradientConflictV1"
        data, missing = _prepare_data(output_root, config)
        if data is None:
            return _missing_result(stage, missing)
        report = audit_gradient_conflict(data)
        report["code_sha256"] = working_code_sha(project_root)
        report["config_sha256"] = _config_sha(_options)
        out_dir = output_root / COMPACT_DIR_REL
        out_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(output_root / GRADIENT_CONFLICT_REL, report)
        return StageResult(
            stage, "pass", EXIT_PASS, completed=1,
            batch_complete=True, scope_complete=True,
            evidence={
                "conflict_fraction": report["conflict_fraction"],
                "persistent_conflict": report["persistent_conflict"],
                "recommend_multitask_mitigation": report["recommend_multitask_mitigation"],
                "shared_encoder_dominant_head": report["shared_encoder_dominant_head"],
            },
        )

    return handler


# ---------------------------------------------------------------------------
# Section 10: SelectV4CompactModelV1 (Train-grouped evidence only)
# ---------------------------------------------------------------------------

def _select_compact_handler(project_root: Path, output_root: Path, config: dict):
    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "SelectV4CompactModelV1"
        out_dir = output_root / COMPACT_DIR_REL
        curve_summary = output_root / LEARNING_CURVE_SUMMARY_REL
        ablation_csv = out_dir / "feature_block_ablation.csv"
        arch_csv = out_dir / "head_architecture_ablation.csv"
        gradient_path = output_root / GRADIENT_CONFLICT_REL
        missing = [
            str(p) for p in (curve_summary, ablation_csv, arch_csv, gradient_path)
            if not p.exists()
        ]
        if missing:
            return _missing_result(stage, missing)
        learning_diag = _read_json(curve_summary).get("verdicts", {})
        ablation = pd.read_csv(ablation_csv)
        architecture = pd.read_csv(arch_csv)
        gradient = _read_json(gradient_path)
        selection = select_compact_model(
            learning_diag=learning_diag,
            ablation=ablation,
            architecture=architecture,
            gradient=gradient,
        )
        selection["code_sha256"] = working_code_sha(project_root)
        selection["config_sha256"] = _config_sha(_options)
        atomic_write_json(output_root / SELECTION_REL, selection)
        return StageResult(
            stage, "pass", EXIT_PASS, completed=1,
            batch_complete=True, scope_complete=True,
            evidence={
                "selected_architecture": selection["selected_architecture"],
                "peak_head_style": selection["peak_head_style"],
                "selected_feature_combo": selection["selected_feature_combo"],
                "multitask_mitigation": selection["multitask_mitigation"],
                "reads_old_locked": selection["reads_old_locked"],
            },
        )

    return handler


# ---------------------------------------------------------------------------
# Section 11: TrainV4CompactTrueStateV1 (original Train 1200, 5 frozen seeds)
# ---------------------------------------------------------------------------

def _train_compact_handler(project_root: Path, output_root: Path, config: dict):
    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "TrainV4CompactTrueStateV1"
        selection_path = output_root / SELECTION_REL
        if not selection_path.exists():
            return _missing_result(stage, [str(selection_path)])
        data, missing = _prepare_data(output_root, config)
        if data is None:
            return _missing_result(stage, missing)
        _, dead_zones = _thresholds(config)
        cfg = _model_config(config)
        selection = _read_json(selection_path)
        seeds = tuple(selection.get("frozen_contract", {}).get("seeds", [0, 1, 2, 3, 4]))
        peak_style = str(selection.get("peak_head_style", "direct"))
        pfv_dz = float(dead_zones.get("pfv", 1.0))

        def factory() -> CompactHeadSpecificModel:
            return CompactHeadSpecificModel(
                cfg=cfg, seeds=seeds, pfv_dead_zone=pfv_dz, peak_style=peak_style
            )

        # Train-grouped CV predictions/metrics (early stopping evidence only
        # from Train-internal folds; old Calibration / Locked never read).
        cv = compact_cv_report(factory, data, cfg=cfg, dead_zones=dead_zones)
        model = factory().fit(data)
        blob = model.to_bytes()
        out_dir = output_root / COMPACT_DIR_REL
        out_dir.mkdir(parents=True, exist_ok=True)
        (output_root / COMPACT_MODEL_BLOB_REL).write_bytes(blob)
        model_sha = hashlib.sha256(blob).hexdigest()

        cv["predictions"].to_parquet(out_dir / "train_cv_predictions.parquet", index=False)
        cv["by_event"].to_csv(out_dir / "cv_metrics_by_event.csv", index=False)
        pd.DataFrame(cv["metrics"]).to_csv(out_dir / "cv_decision_metrics.csv", index=False)
        # The compact output directory already owns this artifact.  Copy only
        # when a future layout supplies it from a different directory.
        curves_src = output_root / "models/v4_compact_v1/learning_curves.csv"
        curves_dst = out_dir / "learning_curves.csv"
        if curves_src.exists() and curves_src.resolve() != curves_dst.resolve():
            shutil.copy2(curves_src, curves_dst)
        tr = data.split_index("train")
        feature_contract = {
            "n_features": int(data.features.shape[1]),
            "head_feature_blocks": selection.get("head_feature_blocks", {}),
            "selected_feature_combo": selection.get("selected_feature_combo"),
            "reads_old_calibration": False,
            "reads_old_locked": False,
        }
        atomic_write_json(out_dir / "feature_contract.json", feature_contract)
        model_contract = {
            "stage": stage,
            "model_version": "v4.1_compact",
            "model_sha256": model_sha,
            "model_blob_rel": COMPACT_MODEL_BLOB_REL,
            "n_train": int(tr.size),
            "seeds": list(seeds),
            "deterministic": True,
            "event_balanced_weights": True,
            "pfv_hurdle": True,
            "pfv_dead_zone": pfv_dz,
            "peak_head_style": peak_style,
            "full_event_heads_disabled": True,
            "online_disabled_k": [1, 2],
            "selected_architecture": selection.get("selected_architecture"),
            "reads_old_calibration": False,
            "reads_old_locked": False,
            "code_sha256": working_code_sha(project_root),
            "config_sha256": _config_sha(_options),
        }
        atomic_write_json(out_dir / "model_contract.json", model_contract)
        pd.DataFrame(
            {"seed": list(seeds), "model_version": ["v4.1_compact"] * len(seeds)}
        ).to_csv(out_dir / "seed_manifest.csv", index=False)
        completion = {
            "stage": stage,
            "model_sha256": model_sha,
            "cv_metrics": cv["metrics"],
            "cv_decision": cv["decision"],
            "n_train": int(tr.size),
            "artifacts": [
                "feature_contract.json", "model_contract.json", "seed_manifest.csv",
                "train_cv_predictions.parquet", "cv_metrics_by_event.csv",
                "cv_decision_metrics.csv", "compact_head_specific_model.pkl",
            ],
            "code_sha256": working_code_sha(project_root),
        }
        atomic_write_json(output_root / COMPACT_COMPLETION_REL, completion)
        return StageResult(
            stage, "pass", EXIT_PASS, completed=1,
            batch_complete=True, scope_complete=True,
            evidence={
                "model_sha256": model_sha,
                "n_train": int(tr.size),
                "seeds": list(seeds),
                "peak_head_style": peak_style,
                "reads_old_locked": False,
            },
        )

    return handler


# ---------------------------------------------------------------------------
# Registry factory (Phase-1 stages)
# ---------------------------------------------------------------------------

def build_v4_compact_phase1_handlers(
    *, project_root: Path, output_root: Path, config: dict
) -> dict[str, Callable[[RuntimeOptions], StageResult]]:
    return {
        "FreezeV4OfflineV0Evidence": _freeze_v0_handler(project_root, output_root, config),
        "AuditV4LockedMetricComparabilityV0": _metric_comparability_handler(
            project_root, output_root, config
        ),
        "AuditV4GeneralizationFailureV0": _generalization_failure_handler(
            project_root, output_root, config
        ),
        "BuildV4FeatureBlockCatalogV1": _feature_block_handler(
            project_root, output_root, config
        ),
        "BuildV4LearningCurvesV1": _learning_curves_handler(
            project_root, output_root, config
        ),
        "RunV4FeatureBlockAblationV1": _feature_ablation_handler(
            project_root, output_root, config
        ),
        "RunV4HeadArchitectureAblationV1": _head_architecture_handler(
            project_root, output_root, config
        ),
        "AuditV4MultitaskGradientConflictV1": _gradient_conflict_handler(
            project_root, output_root, config
        ),
        "SelectV4CompactModelV1": _select_compact_handler(
            project_root, output_root, config
        ),
        "TrainV4CompactTrueStateV1": _train_compact_handler(
            project_root, output_root, config
        ),
    }
