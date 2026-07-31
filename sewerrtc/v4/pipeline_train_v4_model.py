"""V4 True-state model stages (spec sections 4-9), offline only.

Six stages, chained strictly after ``AuditModelTrainingAuthorizationV4`` (never
the legacy ``TrainV4 -> AuditTrain1600Dataset`` path):

    TrainV4Baselines -> EvaluateV4Baselines -> TrainV4TrueState
    -> CalibrateV4TrueState -> EvaluateV4TrueStateLocked
    -> AuditV4OfflineSafetyGate

All data is read from the frozen evidence directory; the loader enforces the
gate-based 1600-sample acceptance and the anti-leakage contract.  No SWMM, no
closed loop.  ``EvaluateV4TrueStateLocked`` is one-shot and protected by an
immutable intent record.  ``AuditV4OfflineSafetyGate`` only reports offline
pass/fail and never lifts the deferred Model Safety Gate.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .runtime import (
    EXIT_BLOCKED,
    EXIT_PASS,
    EXIT_SCIENTIFIC_FAIL,
    RuntimeOptions,
    StageResult,
    atomic_write_json,
    working_code_sha,
)
from .pipeline_train_v4 import (
    FREEZE_NAME,
    _blocked,
    _frozen_dir,
    _missing_result,
    _read_json,
    _sha_file,
    _thresholds,
)
from .train_v4_loader import build_training_data, load_accepted_frame
from .train_v4_models import (
    ModelConfig,
    TrueStateEnsemble,
    calibrate,
    evaluate_split,
    fit_baselines,
)
from .train_v4_preflight import (
    build_action_domain_lock,
    build_feature_lineage,
    build_leakage_final_audit,
    build_runtime_identity,
    build_training_input_audit,
)
from .train_v4_baselines_cv import (
    BaselineModels,
    baseline_stop_verdict,
    run_baseline_cv,
)

MODEL_DIR_REL = "models/v4_true_state"
BASELINE_MODELS_REL = f"{MODEL_DIR_REL}/baseline_models.json"
BASELINE_EVAL_REL = f"{MODEL_DIR_REL}/baseline_evaluation.json"
MODEL_BLOB_REL = f"{MODEL_DIR_REL}/true_state_model.pkl"
TRAIN_SUMMARY_REL = f"{MODEL_DIR_REL}/true_state_training_summary.json"
CALIBRATION_REL = f"{MODEL_DIR_REL}/true_state_calibration.json"
LOCKED_INTENT_REL = f"{MODEL_DIR_REL}/locked_evaluation_intent.json"
LOCKED_RESULT_REL = f"{MODEL_DIR_REL}/locked_evaluation.json"
OFFLINE_GATE_REL = f"{MODEL_DIR_REL}/offline_safety_gate.json"

# Spec sections 1-5 preflight artifacts (written by TrainV4Baselines).
RUNTIME_IDENTITY_REL = f"{MODEL_DIR_REL}/runtime_identity.json"
ACTION_DOMAIN_LOCK_REL = f"{MODEL_DIR_REL}/action_domain_lock.json"
TRAINING_INPUT_AUDIT_REL = f"{MODEL_DIR_REL}/training_input_audit.json"
FEATURE_LINEAGE_REL = f"{MODEL_DIR_REL}/feature_lineage.csv"
FEATURE_LEAKAGE_AUDIT_REL = f"{MODEL_DIR_REL}/feature_leakage_final_audit.json"
BASELINE_FULL_TRAIN_MODELS_REL = f"{MODEL_DIR_REL}/baseline_full_train_models.pkl"

# Spec section 6 baseline six-file-set (separate dir; the stage-completion
# artifact stays under models/v4_true_state/ so STAGE_ARTIFACTS is stable).
BASELINES_DIR_REL = "models/v4_baselines"

MANIFEST_REL = "dataset/train1600_v3_sample_manifest.csv"
CATALOG_REL = "planning/train_checkpoint_catalog_v3.csv"

# Online candidate K policy (spec 4/6): temporarily restrict the *online*
# generator to the trained support K in {4,6,8}; K=1/2 are disabled online.
# K<=8 itself is never modified.
ONLINE_ALLOWED_K = [4, 6, 8]
ONLINE_DISABLED_K = [1, 2]


def _model_config(config: dict) -> ModelConfig:
    section = (config.get("v4_true_state") or {}) if isinstance(config, dict) else {}
    base = ModelConfig()
    if section.get("light"):
        return base.light()
    return ModelConfig(
        seeds=tuple(section.get("seeds", base.seeds)),
        hgb_max_iter=int(section.get("hgb_max_iter", base.hgb_max_iter)),
        hgb_max_depth=section.get("hgb_max_depth", base.hgb_max_depth),
        hgb_learning_rate=float(
            section.get("hgb_learning_rate", base.hgb_learning_rate)
        ),
        hard_negative_weight=float(
            section.get("hard_negative_weight", base.hard_negative_weight)
        ),
    )


def _load_frozen(output_root: Path):
    frozen = _frozen_dir(output_root)
    if frozen is None:
        return None, None, None
    manifest_path = frozen / MANIFEST_REL
    catalog_path = frozen / CATALOG_REL
    return frozen, manifest_path, catalog_path


def _relaxed(config: dict) -> bool:
    """True when running on a relaxed fixture (light or explicit sample-count
    override); the formal config sets neither and gets full enforcement."""
    section = (config.get("v4_true_state") or {}) if isinstance(config, dict) else {}
    return bool(section.get("light")) or ("require_accepted_count" in section)


def _prepare(output_root: Path, config: dict, *, require_count=1600):
    """Return (frame, data, missing, manifest_path, frozen_dir).

    ``frame`` is the accepted DataFrame (needed by the preflight audits);
    ``data`` is the assembled :class:`TrainingData`.
    """
    frozen, manifest_path, catalog_path = _load_frozen(output_root)
    if frozen is None:
        return None, None, ["freeze_pointer.json"], None, None
    missing = [
        str(p) for p in (manifest_path, catalog_path) if not p.exists()
    ]
    if missing:
        return None, None, missing, None, None
    manifest = pd.read_csv(manifest_path)
    catalog = pd.read_csv(catalog_path)
    # The formal pipeline enforces exactly 1600 accepted samples; tests may
    # relax this via config to exercise the stages on small fixtures.
    section = (config.get("v4_true_state") or {}) if isinstance(config, dict) else {}
    if "require_accepted_count" in section:
        require_count = section["require_accepted_count"]
    frame = load_accepted_frame(manifest, require_count=require_count)
    data = build_training_data(manifest, catalog, require_count=require_count)
    return frame, data, None, manifest_path, frozen


def _prepare_data(output_root: Path, config: dict, *, require_count=1600):
    frame, data, missing, _, _ = _prepare(
        output_root, config, require_count=require_count
    )
    if data is None:
        return None, missing
    return data, None


def _config_sha(options: RuntimeOptions) -> str:
    path = Path(options.config) if options.config else None
    if path and path.exists():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    return ""


def _write_preflight(
    project_root, output_root, options, config, frame, data, manifest_path, frozen
):
    """Write spec sections 1-5 artifacts; return (ok, evidence, stop_reason).

    Leakage is always a hard stop (spec section 5: count must be 0).  The
    input-audit split-size expectation (1200/200/200) is a formal-run hard stop
    only; relaxed fixtures record the audit without stopping.
    """
    code_sha = working_code_sha(project_root)
    config_sha = _config_sha(options)
    manifest_sha = _sha_file(manifest_path) if manifest_path else ""
    freeze_file = (frozen / FREEZE_NAME) if frozen else None
    freeze_sha = (
        _sha_file(freeze_file)
        if freeze_file is not None and freeze_file.exists()
        else ""
    )
    cfg = _model_config(config)

    identity = build_runtime_identity(
        code_sha256=code_sha,
        config_sha256=config_sha,
        freeze_sha256=freeze_sha,
        manifest_sha256=manifest_sha,
        frozen_seeds=list(cfg.seeds),
    )
    (output_root / RUNTIME_IDENTITY_REL).parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(output_root / RUNTIME_IDENTITY_REL, identity)

    lock = build_action_domain_lock(frame)
    atomic_write_json(output_root / ACTION_DOMAIN_LOCK_REL, lock)

    audit = build_training_input_audit(frame, data)
    atomic_write_json(output_root / TRAINING_INPUT_AUDIT_REL, audit)

    lineage = build_feature_lineage(data.feature_names)
    lineage.to_csv(output_root / FEATURE_LINEAGE_REL, index=False)

    leakage = build_leakage_final_audit(data.feature_names)
    atomic_write_json(output_root / FEATURE_LEAKAGE_AUDIT_REL, leakage)

    stop_reason = None
    if leakage["status"] != "pass":
        stop_reason = "feature_leakage_detected"
    elif audit["status"] != "pass" and not _relaxed(config):
        stop_reason = "training_input_audit_failed"
    evidence = {
        "runtime_identity_written": True,
        "observed_actual_k": lock["observed_actual_k"],
        "training_input_audit_status": audit["status"],
        "feature_leakage_status": leakage["status"],
        "leakage_count": leakage["leakage_count"],
        "n_features": leakage["n_features"],
    }
    return stop_reason is None, evidence, stop_reason


# ---------------------------------------------------------------------------
# Stage 4a: TrainV4Baselines
# ---------------------------------------------------------------------------

def _train_v4_baselines_handler(project_root, output_root, config):
    def handler(options: RuntimeOptions) -> StageResult:
        stage = "TrainV4Baselines"
        frame, data, missing, manifest_path, frozen = _prepare(output_root, config)
        if data is None:
            return _missing_result(stage, missing)
        _, dead_zones = _thresholds(config)
        cfg = _model_config(config)

        # Spec sections 1-5: runtime identity, action-domain lock, input audit,
        # feature lineage and final leakage audit (hard-stop on leakage).
        ok, pf_evidence, stop_reason = _write_preflight(
            project_root, output_root, options, config,
            frame, data, manifest_path, frozen,
        )
        if not ok:
            return _blocked(stage, stop_reason, **pf_evidence)

        # Spec section 6: event-grouped CV baselines (candidate rows never
        # split randomly).
        cv = run_baseline_cv(data, cfg=cfg, dead_zones=dead_zones)
        verdict = baseline_stop_verdict(cv["summary"])

        # Six-file-set under models/v4_baselines/.
        bdir = output_root / BASELINES_DIR_REL
        bdir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(cv["by_fold"]).to_csv(
            bdir / "baseline_metrics_by_fold.csv", index=False
        )
        pd.DataFrame(cv["by_event"]).to_csv(
            bdir / "baseline_metrics_by_event.csv", index=False
        )
        pd.DataFrame(cv["predictions"]).to_csv(
            bdir / "baseline_predictions.csv", index=False
        )
        pd.DataFrame(cv["ranking"]).to_csv(
            bdir / "baseline_ranking_metrics.csv", index=False
        )
        atomic_write_json(
            bdir / "baseline_summary.json",
            {
                "summary": cv["summary"],
                "n_folds": cv["n_folds"],
                "grouping": cv["grouping"],
                "candidate_rows_randomly_split": cv[
                    "candidate_rows_randomly_split"
                ],
                "stop_verdict": verdict,
                "code_sha256": working_code_sha(project_root),
            },
        )
        atomic_write_json(
            bdir / "completion.json",
            {
                "stage": stage,
                "artifacts": [
                    "baseline_metrics_by_fold.csv",
                    "baseline_metrics_by_event.csv",
                    "baseline_predictions.csv",
                    "baseline_ranking_metrics.csv",
                    "baseline_summary.json",
                ],
            },
        )

        # Full-train baseline models blob for the Locked relative comparison.
        bm = BaselineModels(cfg).fit(data)
        (output_root / BASELINE_FULL_TRAIN_MODELS_REL).write_bytes(bm.to_bytes())

        # Stage-completion artifact stays under models/v4_true_state/.
        report = {
            "code_sha256": working_code_sha(project_root),
            "config_sha256": _config_sha(options),
            "accepted_count": data.n_samples,
            "n_features": int(data.features.shape[1]),
            "full_event_enabled": data.full_event_enabled,
            "baselines_dir": BASELINES_DIR_REL,
            "cv": {
                "n_folds": cv["n_folds"],
                "grouping": cv["grouping"],
                "candidate_rows_randomly_split": cv[
                    "candidate_rows_randomly_split"
                ],
            },
            "summary": cv["summary"],
            "stop_verdict": verdict,
            "preflight": pf_evidence,
        }
        out = output_root / BASELINE_MODELS_REL
        out.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(out, report)
        return StageResult(
            stage, "pass", EXIT_PASS, completed=1,
            batch_complete=True, scope_complete=True,
            evidence={
                "accepted_count": data.n_samples,
                "n_features": int(data.features.shape[1]),
                "baseline_heads": list(cv["summary"]["continuous"].keys()),
                "stop_training": verdict["stop_training"],
                **pf_evidence,
            },
        )

    return handler


# ---------------------------------------------------------------------------
# Stage 4b: EvaluateV4Baselines
# ---------------------------------------------------------------------------

def _evaluate_v4_baselines_handler(project_root, output_root, config):
    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "EvaluateV4Baselines"
        report_path = output_root / BASELINE_MODELS_REL
        if not report_path.exists():
            return _missing_result(stage, [str(report_path)])
        report = _read_json(report_path)
        summary = report.get("summary", {})
        verdict = baseline_stop_verdict(summary)
        # Spec section 6 stop rule: training stops only if NO continuous head
        # beats zero AND no classification head shows signal.  Relaxed fixtures
        # never trigger a scientific stop.
        stop = bool(verdict.get("stop_training")) and not _relaxed(config)
        out = {
            "stop_verdict": verdict,
            "continuous_any_beats_zero": verdict.get(
                "continuous_any_beats_zero"
            ),
            "classification_any_signal": verdict.get(
                "classification_any_signal"
            ),
            "code_sha256": working_code_sha(project_root),
            "note": (
                "Spec section 6: training stops only if NO continuous head "
                "beats zero AND no classification head shows signal."
            ),
        }
        atomic_write_json(output_root / BASELINE_EVAL_REL, out)
        if stop:
            return StageResult(
                stage, "scientific_fail", EXIT_SCIENTIFIC_FAIL,
                completed=1, batch_complete=True, scope_complete=False,
                evidence=out,
            )
        return StageResult(
            stage, "pass", EXIT_PASS, completed=1,
            batch_complete=True, scope_complete=True,
            evidence=out,
        )

    return handler


# ---------------------------------------------------------------------------
# Stage 5-6: TrainV4TrueState
# ---------------------------------------------------------------------------

def _train_v4_true_state_handler(project_root, output_root, config):
    def handler(options: RuntimeOptions) -> StageResult:
        stage = "TrainV4TrueState"
        data, missing = _prepare_data(output_root, config)
        if data is None:
            return _missing_result(stage, missing)
        _, dead_zones = _thresholds(config)
        cfg = _model_config(config)
        model = TrueStateEnsemble(cfg=cfg, pfv_dead_zone=float(dead_zones["pfv"]))
        model.fit(data)
        blob = model.to_bytes()
        model_path = output_root / MODEL_BLOB_REL
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_path.write_bytes(blob)
        model_sha = hashlib.sha256(blob).hexdigest()

        tr = data.split_index("train")
        train_metrics = evaluate_split(model, data, "train")
        peak_hard = int(
            (data.hard_negative_type[tr] == "Peak_hard_negative").sum()
        )
        summary = {
            "model_sha256": model_sha,
            "model_blob_rel": MODEL_BLOB_REL,
            "code_sha256": working_code_sha(project_root),
            "config_sha256": _config_sha(options),
            "n_train": int(tr.size),
            "n_features": int(data.features.shape[1]),
            "seeds": list(cfg.seeds),
            "pfv_hurdle": True,
            "pfv_dead_zone": float(dead_zones["pfv"]),
            "full_event_heads_enabled": data.full_event_enabled,
            "full_event_note": (
                "full_event_eligible is false for all rows; full-event head "
                "disabled and never imputed"
            ),
            "peak_hard_negatives_in_train": peak_hard,
            "peak_hard_negatives_downsampled": False,
            "online_candidate_k_policy": {
                "allowed_online": ONLINE_ALLOWED_K,
                "disabled_online": ONLINE_DISABLED_K,
                "k_le_8_modified": False,
            },
            "train_metrics": train_metrics,
        }
        atomic_write_json(output_root / TRAIN_SUMMARY_REL, summary)
        return StageResult(
            stage, "pass", EXIT_PASS, completed=1,
            batch_complete=True, scope_complete=True,
            evidence={
                "model_sha256": model_sha,
                "n_train": int(tr.size),
                "peak_hard_negatives_in_train": peak_hard,
                "full_event_heads_enabled": data.full_event_enabled,
            },
        )

    return handler


# ---------------------------------------------------------------------------
# Stage 7: CalibrateV4TrueState
# ---------------------------------------------------------------------------

def _calibrate_v4_true_state_handler(project_root, output_root, config):
    def handler(options: RuntimeOptions) -> StageResult:
        stage = "CalibrateV4TrueState"
        model_path = output_root / MODEL_BLOB_REL
        summary_path = output_root / TRAIN_SUMMARY_REL
        if not model_path.exists() or not summary_path.exists():
            return _missing_result(
                stage,
                [str(p) for p in (model_path, summary_path) if not p.exists()],
            )
        data, missing = _prepare_data(output_root, config)
        if data is None:
            return _missing_result(stage, missing)
        model = TrueStateEnsemble.from_bytes(model_path.read_bytes())
        cfg = _model_config(config)
        # Calibration uses the Calibration split only; Locked is never touched.
        calibration = calibrate(model, data, cfg=cfg)
        calibration["reads_locked"] = False
        calibration["model_sha256"] = hashlib.sha256(
            model_path.read_bytes()
        ).hexdigest()
        calibration["code_sha256"] = working_code_sha(project_root)
        out = output_root / CALIBRATION_REL
        atomic_write_json(out, calibration)
        calibration_sha = _sha_file(out)
        return StageResult(
            stage, "pass", EXIT_PASS, completed=1,
            batch_complete=True, scope_complete=True,
            evidence={
                "calibration_n": calibration["calibration_n"],
                "calibration_sha256": calibration_sha,
                "reads_locked": False,
            },
        )

    return handler


# ---------------------------------------------------------------------------
# Stage 8: EvaluateV4TrueStateLocked (ONE-SHOT, intent-protected)
# ---------------------------------------------------------------------------

def _locked_event_rainfall_sha(data, output_root: Path) -> str:
    lk = data.split_index("locked_validation")
    keys = sorted(
        f"{data.event_id[i]}::{data.state_key[i]}" for i in lk
    )
    return hashlib.sha256("\n".join(keys).encode("utf-8")).hexdigest()


def _evaluate_v4_locked_handler(project_root, output_root, config):
    def handler(options: RuntimeOptions) -> StageResult:
        stage = "EvaluateV4TrueStateLocked"
        intent_path = output_root / LOCKED_INTENT_REL
        result_path = output_root / LOCKED_RESULT_REL
        # One-shot protection: refuse if intent OR result already exists.
        if intent_path.exists() or result_path.exists():
            return _blocked(
                stage,
                "locked_evaluation_already_executed",
                intent_exists=intent_path.exists(),
                result_exists=result_path.exists(),
            )
        model_path = output_root / MODEL_BLOB_REL
        calib_path = output_root / CALIBRATION_REL
        if not model_path.exists() or not calib_path.exists():
            return _missing_result(
                stage,
                [str(p) for p in (model_path, calib_path) if not p.exists()],
            )
        data, missing = _prepare_data(output_root, config)
        if data is None:
            return _missing_result(stage, missing)
        model = TrueStateEnsemble.from_bytes(model_path.read_bytes())
        calibration = _read_json(calib_path)

        intent = {
            "stage": stage,
            "one_shot": True,
            "immutable": True,
            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "config_sha256": _config_sha(options),
            "calibration_sha256": _sha_file(calib_path),
            "locked_event_rainfall_sha256": _locked_event_rainfall_sha(
                data, output_root
            ),
            "code_sha256": working_code_sha(project_root),
        }
        intent_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(intent_path, intent)

        locked_report = evaluate_split(
            model, data, "locked_validation", calibration=calibration
        )
        locked_report.update(
            {
                "intent_sha256": _sha_file(intent_path),
                "used_for_tuning": False,
                "note": (
                    "Locked results are read-only evidence; never fed back "
                    "into model structure, loss, hyper-parameters, thresholds "
                    "or candidate rules."
                ),
            }
        )
        atomic_write_json(result_path, locked_report)
        return StageResult(
            stage, "pass", EXIT_PASS, completed=1,
            batch_complete=True, scope_complete=True,
            evidence={
                "locked_n": locked_report.get("n", 0),
                "one_shot_recorded": True,
                "intent_sha256": intent["model_sha256"],
            },
        )

    return handler


# ---------------------------------------------------------------------------
# Stage 9: AuditV4OfflineSafetyGate (offline pass/fail; keeps deferred)
# ---------------------------------------------------------------------------

def _audit_v4_offline_safety_gate_handler(project_root, output_root, config):
    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "AuditV4OfflineSafetyGate"
        required = {
            "baseline_models": output_root / BASELINE_MODELS_REL,
            "baseline_evaluation": output_root / BASELINE_EVAL_REL,
            "training_summary": output_root / TRAIN_SUMMARY_REL,
            "calibration": output_root / CALIBRATION_REL,
            "locked_evaluation": output_root / LOCKED_RESULT_REL,
        }
        missing = [str(p) for p in required.values() if not p.exists()]
        if missing:
            return _missing_result(stage, missing)
        reports = {name: _read_json(path) for name, path in required.items()}
        checks = {
            "calibration_did_not_read_locked": (
                reports["calibration"].get("reads_locked") is False
            ),
            "locked_not_used_for_tuning": (
                reports["locked_evaluation"].get("used_for_tuning") is False
            ),
            "peak_hard_negatives_preserved": (
                reports["training_summary"].get(
                    "peak_hard_negatives_downsampled"
                )
                is False
                and int(
                    reports["training_summary"].get(
                        "peak_hard_negatives_in_train", 0
                    )
                )
                > 0
            ),
            "full_event_heads_disabled": (
                reports["training_summary"].get("full_event_heads_enabled")
                is False
            ),
            "online_k1_k2_disabled": (
                reports["training_summary"]
                .get("online_candidate_k_policy", {})
                .get("disabled_online")
                == ONLINE_DISABLED_K
            ),
            "locked_evaluation_present": (
                int(reports["locked_evaluation"].get("n", 0)) > 0
            ),
        }
        offline_pass = all(checks.values())
        gate = {
            "gate": "PROJECT6_V4_OFFLINE_SAFETY_GATE",
            "gate_role": "offline_integrity_only",
            "does_not_authorize_closed_loop": True,
            "offline_status": "pass" if offline_pass else "fail",
            "checks": checks,
            "model_safety_gate_status": "deferred",
            "model_safety_gate_note": (
                "Offline gate cannot lift the deferred Model Safety Gate; "
                "Policy Lock / Challenge / Formal Blind remain unauthorized."
            ),
            "locked_metrics_summary": {
                "continuous": reports["locked_evaluation"].get("continuous", {}),
                "classification": reports["locked_evaluation"].get(
                    "classification", {}
                ),
            },
            "code_sha256": working_code_sha(project_root),
        }
        atomic_write_json(output_root / OFFLINE_GATE_REL, gate)
        return StageResult(
            stage,
            "pass" if offline_pass else "scientific_fail",
            EXIT_PASS if offline_pass else EXIT_SCIENTIFIC_FAIL,
            completed=1,
            batch_complete=True,
            scope_complete=offline_pass,
            evidence={
                "offline_status": gate["offline_status"],
                "checks": checks,
                "model_safety_gate_status": "deferred",
            },
        )

    return handler


# ---------------------------------------------------------------------------
# Registry factory
# ---------------------------------------------------------------------------

def build_train_v4_model_handlers(
    *, project_root: Path, output_root: Path, config: dict
) -> dict[str, Callable[[RuntimeOptions], StageResult]]:
    return {
        "TrainV4Baselines": _train_v4_baselines_handler(
            project_root, output_root, config
        ),
        "EvaluateV4Baselines": _evaluate_v4_baselines_handler(
            project_root, output_root, config
        ),
        "TrainV4TrueState": _train_v4_true_state_handler(
            project_root, output_root, config
        ),
        "CalibrateV4TrueState": _calibrate_v4_true_state_handler(
            project_root, output_root, config
        ),
        "EvaluateV4TrueStateLocked": _evaluate_v4_locked_handler(
            project_root, output_root, config
        ),
        "AuditV4OfflineSafetyGate": _audit_v4_offline_safety_gate_handler(
            project_root, output_root, config
        ),
    }
