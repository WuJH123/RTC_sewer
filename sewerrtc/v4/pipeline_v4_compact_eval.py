"""V4.1 Compact rescue -- Phase-2 stages (spec sections 12-16).

The *brand-new independent* Calibration / Locked evaluation of the compact V4.1
model.  Nothing here reads the old Calibration or old Locked to pick anything;
the 16 unused Reserve events are the only evaluation source.

Stage map (spec 12-16):

* section 12 -- ``PlanV4CompactCalibrationLockedV1`` freezes the fresh
  evaluation split (4 calibration / 8 locked / 4 accrual reserve) from the
  Reserve events, marks the old Cal / Locked consumed-by-v4.0 and ineligible,
  and ``AuditV4CompactEvaluationPlanV1`` verifies it is frozen, Reserve-only,
  fresh and disjoint;
* section 13 -- six SWMM Run / Build / Audit stages that mirror the V3 pilot
  four-branch runner + round reducer + hard authenticity audit but write under
  ``v4_compact_eval/`` (never under the V3 ``train1600_v3/`` tree);
* section 14 -- ``CalibrateV4CompactV1`` calibrates intervals / probabilities on
  the *new* Calibration only (never Locked, never updates weights);
* section 15/16 -- ``EvaluateV4CompactLockedV1`` runs the one-shot Locked
  evaluation of the compact model and ``AuditV4PredictiveGeneralizationGateV1``
  scores the frozen Predictive Generalization Gate.

No stage here starts a closed loop; ``PlanExactClosedLoopV4`` is authorized only
when the gate verdict is ``pass`` (spec section 17), by the runner, never here.
"""
from __future__ import annotations

import hashlib
import pickle
from pathlib import Path
from typing import Callable

import pandas as pd

from .runtime import (
    EXIT_INCOMPLETE,
    EXIT_PASS,
    EXIT_SCIENTIFIC_FAIL,
    RuntimeOptions,
    StageResult,
    atomic_write_json,
    completion_manifest,
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
    _config_sha,
    _load_frozen,
    _model_config,
)
from .pipeline_ext import _facility_inputs
from .pipeline_train_v3 import DATASET_CONTRACT_REL
from .pilot_candidates import build_pilot_branch_plan, materialize_pilot_candidates
from .train_v4_loader import build_training_data
from .train1600_v3 import build_v3_role_plan
from .v4_compact_eval_ops import (
    audit_evaluation_plan,
    calibrate_compact,
    evaluate_compact_locked,
    evaluate_predictive_gate,
    plan_fresh_evaluation_split,
)

# ---------------------------------------------------------------------------
# Output layout (all relative to output_root unless noted)
# ---------------------------------------------------------------------------

V4C_EVAL_ROOT = "v4_compact_eval"
V4C_PLANNING_DIR_REL = f"{V4C_EVAL_ROOT}/planning"
V4C_CAL_PLAN_REL = f"{V4C_PLANNING_DIR_REL}/v4_compact_v1_calibration_plan.csv"
V4C_LOCKED_PLAN_REL = f"{V4C_PLANNING_DIR_REL}/v4_compact_v1_locked_plan.csv"
V4C_ACCRUAL_PLAN_REL = f"{V4C_PLANNING_DIR_REL}/v4_compact_v1_accrual_plan.csv"
V4C_PLAN_FREEZE_REL = f"{V4C_PLANNING_DIR_REL}/evaluation_plan_freeze.json"
V4C_EVAL_PLAN_AUDIT_REL = f"{V4C_PLANNING_DIR_REL}/evaluation_plan_audit.json"
V4C_OLD_CONSUMPTION_REL = (
    f"{V4C_PLANNING_DIR_REL}/old_calibration_locked_consumption.json"
)
V4C_BRANCH_PLAN_REL = f"{V4C_PLANNING_DIR_REL}/v4_compact_branch_plan.csv"

# section-13 SWMM segments (mirror the V3 four-branch pilot runner)
V4C_CAL_RUN_MANIFEST_REL = f"{V4C_EVAL_ROOT}/calibration/run_manifest.csv"
V4C_LOCKED_RUN_MANIFEST_REL = f"{V4C_EVAL_ROOT}/locked/run_manifest.csv"
V4C_CAL_RUN_PLAN_REL = f"{V4C_EVAL_ROOT}/calibration/plan.csv"
V4C_LOCKED_RUN_PLAN_REL = f"{V4C_EVAL_ROOT}/locked/plan.csv"
V4C_CAL_AUDIT_REL = f"{V4C_EVAL_ROOT}/calibration/round_audit.json"
V4C_LOCKED_AUDIT_REL = f"{V4C_EVAL_ROOT}/locked/round_audit.json"

# section 14/16 model artifacts (share the compact model directory)
COMPACT_DIR_REL = "models/v4_compact_v1"
COMPACT_MODEL_BLOB_REL = f"{COMPACT_DIR_REL}/compact_head_specific_model.pkl"
V4C_CALIBRATION_REL = f"{COMPACT_DIR_REL}/v4_compact_v1_calibration.json"
V4C_LOCKED_INTENT_REL = f"{COMPACT_DIR_REL}/v4_compact_v1_locked_intent.json"
V4C_LOCKED_RESULT_REL = f"{COMPACT_DIR_REL}/v4_compact_v1_locked_evaluation.json"
V4C_GATE_VERDICT_REL = f"{COMPACT_DIR_REL}/v4_predictive_generalization_gate.json"

# The Predictive Generalization Gate contract lives in the repo (project_root).
GATE_CONTRACT_REL = (
    "docs/contracts/PROJECT6_V4_PREDICTIVE_GENERALIZATION_GATE_V1.json"
)

EVENT_LEDGER_REL = "inventory/event_usage_ledger.csv"

# Run stage -> (segment dir, split label, accepted target).
SEGMENTS_V4C = {
    "RunV4CompactCalibrationV1": ("calibration", "v4.1_calibration", 100),
    "RunV4CompactLockedV1": ("locked", "v4.1_locked", 200),
}
BUILD_STAGE_RUN = {
    "BuildV4CompactCalibrationV1": "RunV4CompactCalibrationV1",
    "BuildV4CompactLockedV1": "RunV4CompactLockedV1",
}
AUDIT_STAGE_RUN = {
    "AuditV4CompactCalibrationV1": "RunV4CompactCalibrationV1",
    "AuditV4CompactLockedV1": "RunV4CompactLockedV1",
}


def _v4c_segment_dataset_dir(output_root: Path, seg: str) -> Path:
    return output_root / V4C_EVAL_ROOT / seg / "dataset"


# ---------------------------------------------------------------------------
# Fresh evaluation data assembly (Train + fresh split; old Cal/Locked unread)
# ---------------------------------------------------------------------------

def _prepare_compact_eval_data(
    output_root: Path,
    config: dict,
    *,
    segments: list[tuple[str, str]],
):
    """Assemble ``TrainingData`` from the frozen Train manifest plus one or more
    freshly built section-13 segments (split relabelled).

    Returns ``(data, missing)``.  The old Calibration / Locked splits present in
    the frozen manifest are never indexed by the Phase-2 ops, so they cannot
    leak into the new evaluation.
    """
    frozen, manifest_path, catalog_path = _load_frozen(output_root)
    if frozen is None:
        return None, ["freeze_pointer.json"]
    missing = [str(p) for p in (manifest_path, catalog_path) if not p.exists()]
    fresh: list[pd.DataFrame] = []
    for seg, split_name in segments:
        seg_manifest = (
            _v4c_segment_dataset_dir(output_root, seg) / "round_sample_manifest.csv"
        )
        if not seg_manifest.exists():
            missing.append(str(seg_manifest))
            continue
        frame = pd.read_csv(seg_manifest).copy()
        frame["split"] = split_name
        fresh.append(frame)
    if missing:
        return None, missing
    combined = pd.concat(
        [pd.read_csv(manifest_path), *fresh], ignore_index=True
    )
    data = build_training_data(
        combined, pd.read_csv(catalog_path), require_count=None
    )
    return data, None


# ---------------------------------------------------------------------------
# Section 12: PlanV4CompactCalibrationLockedV1 / AuditV4CompactEvaluationPlanV1
# ---------------------------------------------------------------------------

def _plan_evaluation_handler(project_root: Path, output_root: Path, config: dict):
    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "PlanV4CompactCalibrationLockedV1"
        ledger_path = output_root / EVENT_LEDGER_REL
        if not ledger_path.exists():
            return _missing_result(stage, [str(ledger_path)])
        ledger = pd.read_csv(ledger_path)
        try:
            plan = plan_fresh_evaluation_split(ledger)
        except ValueError as exc:
            return _blocked(stage, str(exc))
        planning = output_root / V4C_PLANNING_DIR_REL
        planning.mkdir(parents=True, exist_ok=True)
        plan_rels = {
            "v4.1_calibration": V4C_CAL_PLAN_REL,
            "v4.1_locked": V4C_LOCKED_PLAN_REL,
            "locked_accrual_reserve": V4C_ACCRUAL_PLAN_REL,
        }
        # The split freeze alone is not executable.  Materialize new legal
        # candidates from the reserve checkpoint catalog before any SWMM run.
        # Reuse the V3 contract-projected candidate builder rather than copying
        # old action labels or schedules into the newly held-out events.
        reserve_catalog_path = (
            output_root / "train1600_v3/planning/train_reserve_catalog_v3.csv"
        )
        peak_anchor_path = output_root / "peak_boundary/peak_boundary_anchor_library.csv"
        contract_path = project_root / DATASET_CONTRACT_REL
        missing = [
            str(path) for path in (reserve_catalog_path, peak_anchor_path, contract_path)
            if not path.exists()
        ]
        if missing:
            return _missing_result(stage, missing)
        reserve_catalog = pd.read_csv(reserve_catalog_path)
        facility = _facility_inputs(project_root, config)
        branch_frames: list[pd.DataFrame] = []
        for split, rel in plan_rels.items():
            event_rows = pd.DataFrame(plan["plans"].get(split, [])).copy()
            if split == "locked_accrual_reserve":
                event_rows.to_csv(output_root / rel, index=False)
                continue
            target = int(plan["per_split_samples"][split])
            subset = reserve_catalog.merge(
                event_rows[["event_id", "rainfall_sha256"]],
                on=["event_id", "rainfall_sha256"], how="inner",
            ).copy()
            subset["split"] = split
            if len(subset) * 5 != target:
                return _blocked(
                    stage,
                    f"reserve checkpoint count cannot produce {target} candidates for {split}",
                    checkpoint_rows=int(len(subset)), target=target,
                )
            roles = build_v3_role_plan(subset)
            primary = roles[roles["plan_tier"] == "primary"].copy()
            schedule_dir = planning / "schedules" / split
            candidates, coverage_missing = materialize_pilot_candidates(
                primary,
                subset,
                facility_ids=facility["facility_ids"],
                facility_semantics=facility["facility_semantics"],
                peak_boundary_anchor_library=pd.read_csv(peak_anchor_path),
                contract_sha256=_sha_file(contract_path),
                config_sha256=_config_sha(_options),
                code_sha256=working_code_sha(project_root),
                schedule_dir=schedule_dir,
                schedule_dir_relative_to=output_root,
            )
            if len(candidates) != target or not coverage_missing.empty:
                return _blocked(
                    stage,
                    f"candidate materialization incomplete for {split}",
                    candidates=int(len(candidates)), target=target,
                    coverage_missing=int(len(coverage_missing)),
                )
            candidates.to_csv(output_root / rel, index=False)
            segment = "calibration" if split == "v4.1_calibration" else "locked"
            segment_dir = output_root / V4C_EVAL_ROOT / segment
            segment_dir.mkdir(parents=True, exist_ok=True)
            candidates.to_csv(segment_dir / "plan.csv", index=False)
            branch_frames.append(
                build_pilot_branch_plan(
                    candidates, contract_sha256=_sha_file(contract_path)
                )
            )
        if not branch_frames:
            return _blocked(stage, "no executable evaluation candidate plan")
        branch_plan = pd.concat(branch_frames, ignore_index=True)
        if branch_plan.duplicated(["sample_id", "branch_role"]).any():
            return _blocked(stage, "duplicate sample/branch rows across evaluation splits")
        branch_plan.to_csv(output_root / V4C_BRANCH_PLAN_REL, index=False)
        for split, rel in plan_rels.items():
            if not (output_root / rel).exists():
                pd.DataFrame(plan["plans"].get(split, [])).to_csv(output_root / rel, index=False)
        freeze = dict(plan)
        freeze["code_sha256"] = working_code_sha(project_root)
        freeze["config_sha256"] = _config_sha(_options)
        freeze["executable_plan_sha256"] = {
            split: _sha_file(output_root / rel) for split, rel in plan_rels.items()
        }
        freeze["branch_plan_sha256"] = _sha_file(output_root / V4C_BRANCH_PLAN_REL)
        atomic_write_json(output_root / V4C_PLAN_FREEZE_REL, freeze)
        # Mark the *old* V4.0 Calibration / Locked consumed and ineligible for
        # the V1 official evaluation (spec section 12); never delete or relabel.
        atomic_write_json(
            output_root / V4C_OLD_CONSUMPTION_REL,
            {
                "stage": stage,
                "old_calibration": {
                    "consumed_by_model_version": "v4.0",
                    "eligible_for_v1_official_evaluation": False,
                },
                "old_locked": {
                    "consumed_by_model_version": "v4.0",
                    "eligible_for_v1_official_evaluation": False,
                    "reusable_for_model_selection": False,
                },
                "code_sha256": working_code_sha(project_root),
            },
        )
        return StageResult(
            stage, "pass", EXIT_PASS, completed=1,
            batch_complete=True, scope_complete=True,
            evidence={
                "counts": plan["counts"],
                "per_split_samples": plan["per_split_samples"],
                "frozen_order_sha256": plan["frozen_order_sha256"],
                "reads_old_locked_for_selection": False,
            },
        )

    return handler


def _audit_evaluation_plan_handler(project_root: Path, output_root: Path, config: dict):
    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "AuditV4CompactEvaluationPlanV1"
        freeze_path = output_root / V4C_PLAN_FREEZE_REL
        ledger_path = output_root / EVENT_LEDGER_REL
        missing = [
            str(p) for p in (freeze_path, ledger_path) if not p.exists()
        ]
        if missing:
            return _missing_result(stage, missing)
        audit = audit_evaluation_plan(
            _read_json(freeze_path), pd.read_csv(ledger_path)
        )
        audit["code_sha256"] = working_code_sha(project_root)
        atomic_write_json(output_root / V4C_EVAL_PLAN_AUDIT_REL, audit)
        passed = audit["status"] == "pass"
        return StageResult(
            stage, audit["status"], EXIT_PASS if passed else EXIT_INCOMPLETE,
            completed=1, batch_complete=True, scope_complete=passed,
            evidence={
                "status": audit["status"],
                "checks": audit["checks"],
                "n_calibration_events": audit["n_calibration_events"],
                "n_locked_events": audit["n_locked_events"],
                "n_accrual_events": audit["n_accrual_events"],
            },
        )

    return handler


# ---------------------------------------------------------------------------
# Section 13: SWMM Build / Audit stages (mirror V3 four-branch reducer + audit)
# ---------------------------------------------------------------------------

def _build_v4c_round_handler(
    project_root: Path, output_root: Path, config: dict, *, stage: str, run_stage: str
) -> Callable[[RuntimeOptions], StageResult]:
    """Reduce one V4.1 segment's completed four-branch runs into its dataset."""
    seg, split_label, _target = SEGMENTS_V4C[run_stage]

    def handler(_options: RuntimeOptions) -> StageResult:
        from .pipeline import _add_pilot_audit_columns, _run_stage_sources
        from .pipeline_ext import _facility_inputs
        from .pilot_reducers import build_pilot_dataset
        from .pilot_run import expand_pilot_completions
        from .pipeline_train_v3 import (
            _carry_plan_columns,
            _stamp_sampled_only_labels,
        )

        plan_path, run_root = _run_stage_sources(run_stage, output_root)
        branch_path = output_root / V4C_BRANCH_PLAN_REL
        missing = [
            str(path) for path in (plan_path, branch_path) if not path.exists()
        ]
        if missing:
            return _missing_result(stage, missing)
        candidate_plan = pd.read_csv(plan_path)
        completions = (
            completion_manifest(run_root) if run_root.exists() else pd.DataFrame()
        )
        try:
            inputs = _facility_inputs(project_root, config)
            result = build_pilot_dataset(
                candidate_plan,
                pd.read_csv(branch_path),
                expand_pilot_completions(completions),
                priority_nodes=inputs["priority_nodes"],
                facility_ids=inputs["facility_ids"],
                scientific_margin=config["thresholds"]["scientific_margin"],
                dead_zone=config["thresholds"]["dead_zone"],
            )
            samples = _stamp_sampled_only_labels(
                _carry_plan_columns(
                    _add_pilot_audit_columns(result["sample_manifest"], config),
                    candidate_plan,
                )
            )
        except (KeyError, OSError, ValueError) as exc:
            return _blocked(stage, str(exc))
        # Freshly built rows carry the frozen V4.1 split label from section 12.
        samples["split"] = split_label
        dataset_dir = _v4c_segment_dataset_dir(output_root, seg)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        samples.to_csv(dataset_dir / "round_sample_manifest.csv", index=False)
        for name in ("branch_manifest", "rejected", "actual_duplicates", "pending"):
            result[name].to_csv(dataset_dir / f"round_{name}.csv", index=False)
        result["missing_confirmed"].to_csv(
            dataset_dir / "round_missing.csv", index=False
        )
        accounting = result["accounting"]
        atomic_write_json(
            dataset_dir / "completion.json",
            {
                "stage": stage,
                "run_stage": run_stage,
                "segment": seg,
                "split": split_label,
                "accounting": accounting,
                "accepted": int(accounting["accepted"]),
                "planned_rows": int(len(candidate_plan)),
            },
        )
        closed = bool(accounting.get("accounting_closed"))
        pending = int(len(result["pending"]))
        return StageResult(
            stage, "pass" if closed else "incomplete",
            EXIT_PASS if closed else EXIT_INCOMPLETE,
            completed=int(accounting["accepted"]), remaining=pending,
            batch_complete=True, scope_complete=closed,
            evidence={
                "segment": seg, "split": split_label,
                "accepted": int(accounting["accepted"]),
                "pending": pending, "accounting_closed": closed,
            },
        )

    return handler


def _audit_v4c_round_handler(
    project_root: Path, output_root: Path, config: dict, *, stage: str, run_stage: str
) -> Callable[[RuntimeOptions], StageResult]:
    """Full-round hard gate: authenticity, accounting, frozen-plan-before-run."""
    seg, split_label, accepted_target = SEGMENTS_V4C[run_stage]
    audit_rel = (
        V4C_CAL_AUDIT_REL if seg == "calibration" else V4C_LOCKED_AUDIT_REL
    )

    def handler(_options: RuntimeOptions) -> StageResult:
        from .train1600_v3 import audit_round_dataset_v3
        from .partial_audit import HARD_AUTHENTICITY_COLUMNS
        from .pipeline_train_v3 import _reference_cache_check

        dataset_dir = _v4c_segment_dataset_dir(output_root, seg)
        manifest_path = dataset_dir / "round_sample_manifest.csv"
        completion_path = dataset_dir / "completion.json"
        freeze_path = output_root / V4C_PLAN_FREEZE_REL
        missing = [
            str(path)
            for path in (manifest_path, completion_path, freeze_path)
            if not path.exists()
        ]
        if missing:
            return _missing_result(stage, missing)
        try:
            samples = pd.read_csv(manifest_path)
            accounting = _read_json(completion_path).get("accounting", {})
            expected_sha, current_sha, cache_detail = _reference_cache_check(
                output_root
            )
            audit = audit_round_dataset_v3(
                samples,
                accounting,
                stage=stage,
                accepted_target=accepted_target,
                hard_columns=HARD_AUTHENTICITY_COLUMNS,
                reference_cache_sha256=current_sha,
                expected_reference_cache_sha256=expected_sha,
            )
        except (KeyError, OSError, ValueError) as exc:
            return _blocked(stage, str(exc))
        audit["reference_cache_detail"] = cache_detail
        # The evaluation split must be frozen (section 12) before any new label.
        freeze = _read_json(freeze_path)
        plan_frozen = bool(freeze.get("frozen_before_any_new_label")) and (
            split_label in freeze.get("splits", {})
        )
        audit["hard_checks"]["evaluation_split_frozen_before_run"] = plan_frozen
        if not plan_frozen:
            audit["status"] = "blocked"
        audit["code_sha256"] = working_code_sha(project_root)
        artifact = output_root / audit_rel
        artifact.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(artifact, audit)
        passed = audit["status"] == "pass"
        return StageResult(
            stage, audit["status"], EXIT_PASS if passed else EXIT_INCOMPLETE,
            completed=int(audit.get("accepted", 0)),
            batch_complete=True, scope_complete=passed, evidence=audit,
        )

    return handler


# ---------------------------------------------------------------------------
# Section 14: CalibrateV4CompactV1 (new Calibration only; weights untouched)
# ---------------------------------------------------------------------------

def _calibrate_compact_handler(project_root: Path, output_root: Path, config: dict):
    def handler(options: RuntimeOptions) -> StageResult:
        stage = "CalibrateV4CompactV1"
        model_path = output_root / COMPACT_MODEL_BLOB_REL
        if not model_path.exists():
            return _missing_result(stage, [str(model_path)])
        data, missing = _prepare_compact_eval_data(
            output_root, config, segments=[("calibration", "v4.1_calibration")]
        )
        if data is None:
            return _missing_result(stage, missing)
        _, dead_zones = _thresholds(config)
        cfg = _model_config(config)
        model = pickle.loads(model_path.read_bytes())
        try:
            report = calibrate_compact(
                model, data, cfg=cfg, dead_zones=dead_zones,
                calibration_split="v4.1_calibration",
            )
        except (KeyError, ValueError) as exc:
            return _blocked(stage, str(exc))
        report["model_sha256"] = hashlib.sha256(model_path.read_bytes()).hexdigest()
        report["code_sha256"] = working_code_sha(project_root)
        report["config_sha256"] = _config_sha(options)
        atomic_write_json(output_root / V4C_CALIBRATION_REL, report)
        return StageResult(
            stage, "pass", EXIT_PASS, completed=int(report["calibration_n"]),
            batch_complete=True, scope_complete=True,
            evidence={
                "calibration_n": report["calibration_n"],
                "reads_locked": report["reads_locked"],
                "updates_model_weights": report["updates_model_weights"],
                "disabled_probability_heads": report["disabled_probability_heads"],
            },
        )

    return handler


# ---------------------------------------------------------------------------
# Section 15/16: EvaluateV4CompactLockedV1 (one-shot) + gate audit
# ---------------------------------------------------------------------------

def _evaluate_compact_locked_handler(project_root: Path, output_root: Path, config: dict):
    def handler(options: RuntimeOptions) -> StageResult:
        stage = "EvaluateV4CompactLockedV1"
        intent_path = output_root / V4C_LOCKED_INTENT_REL
        result_path = output_root / V4C_LOCKED_RESULT_REL
        # One-shot protection: refuse if intent OR result already exists.
        if intent_path.exists() or result_path.exists():
            return _blocked(
                stage, "locked_evaluation_already_executed",
                intent_exists=intent_path.exists(),
                result_exists=result_path.exists(),
            )
        model_path = output_root / COMPACT_MODEL_BLOB_REL
        calib_path = output_root / V4C_CALIBRATION_REL
        contract_path = project_root / GATE_CONTRACT_REL
        freeze_path = output_root / V4C_PLAN_FREEZE_REL
        locked_manifest = (
            _v4c_segment_dataset_dir(output_root, "locked")
            / "round_sample_manifest.csv"
        )
        # All inputs must exist *before* the intent is written so the one-shot
        # is never burned on a missing-input run (spec section 16 checks 1-6).
        required = (model_path, calib_path, contract_path, freeze_path, locked_manifest)
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            return _missing_result(stage, missing)

        intent = {
            "stage": stage,
            "one_shot": True,
            "immutable": True,
            "gate_contract": "PROJECT6_V4_PREDICTIVE_GENERALIZATION_GATE_V1",
            "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
            "calibration_sha256": _sha_file(calib_path),
            "gate_contract_sha256": _sha_file(contract_path),
            "evaluation_plan_freeze_sha256": _sha_file(freeze_path),
            "config_sha256": _config_sha(options),
            "code_sha256": working_code_sha(project_root),
            "locked_split": "v4.1_locked",
            "pre_execution_checks": {
                "locked_intent_absent_before_write": True,
                "locked_result_absent": not result_path.exists(),
                "compact_model_sha_frozen": True,
                "calibration_sha_frozen": True,
                "gate_contract_sha_frozen": True,
                "evaluation_data_not_read_before_intent": True,
            },
        }
        intent_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(intent_path, intent)

        # Only now is the Locked evaluation data read.
        data, data_missing = _prepare_compact_eval_data(
            output_root, config, segments=[("locked", "v4.1_locked")]
        )
        if data is None:
            return _missing_result(stage, data_missing)
        _, dead_zones = _thresholds(config)
        cfg = _model_config(config)
        model = pickle.loads(model_path.read_bytes())
        calibration = _read_json(calib_path)
        try:
            locked_report = evaluate_compact_locked(
                model, data, cfg=cfg, dead_zones=dead_zones,
                calibration=calibration, locked_split="v4.1_locked",
            )
        except (KeyError, ValueError) as exc:
            return _blocked(stage, str(exc))
        locked_report.update(
            {
                "intent_sha256": _sha_file(intent_path),
                "used_for_tuning": False,
                "note": (
                    "Locked results are read-only evidence; never fed back into "
                    "model structure, loss, hyper-parameters, thresholds, "
                    "candidate rules or the evaluation split."
                ),
                "code_sha256": working_code_sha(project_root),
            }
        )
        atomic_write_json(result_path, locked_report)
        return StageResult(
            stage, "pass", EXIT_PASS, completed=int(locked_report.get("n", 0)),
            batch_complete=True, scope_complete=True,
            evidence={
                "locked_n": locked_report.get("n", 0),
                "one_shot_recorded": True,
                "used_for_tuning": False,
                "intent_sha256": intent["model_sha256"],
            },
        )

    return handler


def _audit_predictive_gate_handler(project_root: Path, output_root: Path, config: dict):
    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "AuditV4PredictiveGeneralizationGateV1"
        result_path = output_root / V4C_LOCKED_RESULT_REL
        contract_path = project_root / GATE_CONTRACT_REL
        missing = [
            str(p) for p in (result_path, contract_path) if not p.exists()
        ]
        if missing:
            return _missing_result(stage, missing)
        contract = _read_json(contract_path)
        verdict = evaluate_predictive_gate(_read_json(result_path), contract)
        verdict["code_sha256"] = working_code_sha(project_root)
        verdict["contract_sha256"] = _sha_file(contract_path)
        atomic_write_json(output_root / V4C_GATE_VERDICT_REL, verdict)
        status = verdict["status"]
        exit_code = {
            "pass": EXIT_PASS,
            "underpowered": EXIT_INCOMPLETE,
            "scientific_fail": EXIT_SCIENTIFIC_FAIL,
        }.get(status, EXIT_SCIENTIFIC_FAIL)
        return StageResult(
            stage, status, exit_code, completed=1,
            batch_complete=True, scope_complete=(status == "pass"),
            evidence={
                "status": status,
                "authorizes_closed_loop": verdict["authorizes_closed_loop"],
                "checks": verdict["checks"],
                "reasons": verdict["reasons"],
            },
        )

    return handler


# ---------------------------------------------------------------------------
# Registry factory (Phase-2 stages)
# ---------------------------------------------------------------------------

def build_v4_compact_phase2_handlers(
    *, project_root: Path, output_root: Path, config: dict
) -> dict[str, Callable[[RuntimeOptions], StageResult]]:
    from .pipeline import _run_pilot400_handler

    handlers: dict[str, Callable[[RuntimeOptions], StageResult]] = {
        "PlanV4CompactCalibrationLockedV1": _plan_evaluation_handler(
            project_root, output_root, config
        ),
        "AuditV4CompactEvaluationPlanV1": _audit_evaluation_plan_handler(
            project_root, output_root, config
        ),
        "CalibrateV4CompactV1": _calibrate_compact_handler(
            project_root, output_root, config
        ),
        "EvaluateV4CompactLockedV1": _evaluate_compact_locked_handler(
            project_root, output_root, config
        ),
        "AuditV4PredictiveGeneralizationGateV1": _audit_predictive_gate_handler(
            project_root, output_root, config
        ),
    }
    # Section 13 Run stages reuse the shared four-branch pilot runner; the
    # reference cache root stays ``pilot`` so no reference run is repeated.
    for run_stage in SEGMENTS_V4C:
        handlers[run_stage] = _run_pilot400_handler(
            output_root, config, stage=run_stage, branch_plan_rel=V4C_BRANCH_PLAN_REL
        )
    for build_stage, run_stage in BUILD_STAGE_RUN.items():
        handlers[build_stage] = _build_v4c_round_handler(
            project_root, output_root, config, stage=build_stage, run_stage=run_stage
        )
    for audit_stage, run_stage in AUDIT_STAGE_RUN.items():
        handlers[audit_stage] = _audit_v4c_round_handler(
            project_root, output_root, config, stage=audit_stage, run_stage=run_stage
        )
    return handlers
