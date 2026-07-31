"""Train1600 V3 stage handlers (gate split + 1200/200/200 production chain).

Wired lazily from ``pipeline.build_registry`` like ``pipeline_ext`` and
``pipeline_p3``.  All V3 stages write only under ``train1600_v3/`` plus the
immutable freeze archive under ``audits/frozen_evidence/``; frozen Gate P3
evidence is a read-only input and the P3 verdict (underpowered_validation)
is never overwritten.  Formal 1600-sample SWMM runs never start from here:
Run stages only execute when the runner invokes them explicitly.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from pathlib import Path
from typing import Callable

import pandas as pd

from .event_splits import (
    EventShortfallError,
    assign_split,
    select_train1600_events,
)
from .partial_audit import HARD_AUTHENTICITY_COLUMNS
from .pilot_candidates import (
    build_pilot_branch_plan,
    materialize_pilot_candidates,
)
from .pilot_reducers import build_pilot_dataset
from .pilot_run import expand_pilot_completions
from .runtime import (
    EXIT_BLOCKED,
    EXIT_INCOMPLETE,
    EXIT_PASS,
    RuntimeOptions,
    StageResult,
    atomic_write_json,
    completion_manifest,
    working_code_sha,
)
from .train1600_v3 import (
    ROUND_TARGETS_V3,
    assert_train_split_only,
    assign_primary_candidates_to_rounds_v3,
    audit_round_dataset_v3,
    audit_train1600_dataset_v3,
    audit_train1600_plan_v3,
    apply_state_feasibility_scorer,
    build_p3_freeze_payload,
    build_per_state_progress_v3,
    build_train_round_rotation_v3,
    build_v3_role_plan,
    evaluate_data_generation_authorization_v3,
    fit_state_feasibility_scorer,
    model_safety_gate_v3_status,
    rank_remaining_candidates_v3,
    select_round_candidates_v3,
)
from .training_plan import build_train_checkpoint_catalog

T16_ROOT = "train1600_v3"
PLANNING_DIR_REL = f"{T16_ROOT}/planning"
MASTER_PLAN_REL = f"{PLANNING_DIR_REL}/train_candidate_plan_v3.csv"
T16_BRANCH_PLAN_REL = f"{PLANNING_DIR_REL}/train_branch_plan_v3.csv"
CATALOG_REL = f"{PLANNING_DIR_REL}/train_checkpoint_catalog_v3.csv"
RESERVE_CATALOG_REL = f"{PLANNING_DIR_REL}/train_reserve_catalog_v3.csv"
ROLE_PLAN_REL = f"{PLANNING_DIR_REL}/train_role_plan_v3.csv"
ROTATION_REL = f"{PLANNING_DIR_REL}/train_extra_rotation_v3.csv"
STATE_TARGETS_REL = f"{PLANNING_DIR_REL}/train_round_state_targets_v3.csv"
PROGRESS_REL = f"{PLANNING_DIR_REL}/per_state_progress_v3.csv"
SCORER_REL = f"{PLANNING_DIR_REL}/stratification_scorer_v3.json"
PLAN_FREEZE_REL = f"{PLANNING_DIR_REL}/plan_freeze_v3.json"
SELECTION_REL = f"{PLANNING_DIR_REL}/train_event_selection_v3.json"
COVERAGE_MISSING_REL = f"{PLANNING_DIR_REL}/train_coverage_missing_v3.csv"
CAL_FROZEN_REL = f"{PLANNING_DIR_REL}/calibration_plan_frozen_v3.csv"
LOCKED_FROZEN_REL = (
    f"{PLANNING_DIR_REL}/locked_validation_plan_frozen_v3.csv"
)
AUTH_REL = f"{T16_ROOT}/authorization/data_generation_authorization_v3.json"
MSG_REL = f"{T16_ROOT}/authorization/model_safety_gate_v3_status.json"
FREEZE_POINTER_REL = (
    "audits/frozen_evidence/pilot_feasibility_p3/freeze_pointer.json"
)
FREEZE_ROOT_REL = "audits/frozen_evidence/pilot_feasibility_p3"

P3_EVIDENCE_DIRS = (
    "map",
    "dataset",
    "dataset_v3",
    "evaluation",
    "planning",
    "legacy_oracle",
)

DGA_CONTRACT_REL = (
    "docs/contracts/PROJECT6_V4_DATA_GENERATION_AUTHORIZATION_V3.json"
)
MSG_CONTRACT_REL = "docs/contracts/PROJECT6_V4_MODEL_SAFETY_GATE_V3.json"
DATASET_CONTRACT_REL = "docs/contracts/PROJECT6_V4_TRAIN1600_DATASET_V3.json"
ACCRUAL_CONTRACT_REL = (
    "docs/contracts/PROJECT6_V4_LOCKED_VALIDATION_ACCRUAL_V3.json"
)

SEGMENTS_V3 = {
    "RunTrainRound0V3": ("round0", 400),
    "RunTrainRound1V3": ("round1", 400),
    "RunTrainRound2V3": ("round2", 400),
    "RunCalibration200V3": ("calibration", 200),
    "RunLockedValidation200V3": ("locked_validation", 200),
}


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _missing_result(stage: str, missing: list[str]) -> StageResult:
    return StageResult(
        stage,
        "incomplete",
        EXIT_INCOMPLETE,
        remaining=1,
        evidence={"reason": "inputs_missing", "missing_inputs": missing},
    )


def _blocked(stage: str, reason: str, **extra) -> StageResult:
    return StageResult(
        stage,
        "blocked",
        EXIT_BLOCKED,
        evidence={"reason": reason, **extra},
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _aggregate_sha(manifest: dict[str, str]) -> str:
    lines = "\n".join(
        f"{name}:{sha}" for name, sha in sorted(manifest.items())
    )
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# FreezeP3Evidence
# ---------------------------------------------------------------------------

def _freeze_p3_handler(
    project_root: Path, output_root: Path, config: dict
) -> Callable[[RuntimeOptions], StageResult]:
    """Archive Gate P3 evidence immutably; never overwrite an existing freeze."""

    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "FreezeP3Evidence"
        pointer_path = output_root / FREEZE_POINTER_REL
        if pointer_path.exists():
            pointer = _read_json(pointer_path)
            frozen_dir = output_root / str(pointer.get("frozen_dir_rel", ""))
            if (frozen_dir / "pilot_feasibility_p3_freeze.json").exists():
                return StageResult(
                    stage,
                    "pass",
                    EXIT_PASS,
                    completed=1,
                    batch_complete=True,
                    scope_complete=True,
                    evidence={
                        "already_frozen": True,
                        "immutable": True,
                        "frozen_dir": str(frozen_dir),
                    },
                )
        p3_root = output_root / "pilot_feasibility_p3"
        verdict_path = p3_root / "evaluation" / "pilot_gate_v3_verdict.json"
        map_audit_path = p3_root / "map" / "pilot_feasibility_audit.json"
        dataset_audit_path = (
            p3_root / "dataset_v3" / "pilot_v3_dataset_audit.json"
        )
        reference_root = output_root / "pilot" / "references"
        missing = [
            str(path)
            for path in (
                verdict_path,
                map_audit_path,
                dataset_audit_path,
                reference_root,
            )
            if not path.exists()
        ]
        missing.extend(
            str(p3_root / name)
            for name in P3_EVIDENCE_DIRS
            if not (p3_root / name).exists()
        )
        if missing:
            return _missing_result(stage, missing)
        verdict = _read_json(verdict_path)
        if str(verdict.get("status")) != "underpowered_validation":
            return _blocked(
                stage,
                "p3_verdict_must_remain_underpowered_validation",
                found=str(verdict.get("status")),
            )
        code_sha = working_code_sha(project_root)
        target = output_root / FREEZE_ROOT_REL / code_sha
        target.mkdir(parents=True, exist_ok=True)
        copied: dict[str, str] = {}
        for name in P3_EVIDENCE_DIRS:
            source_dir = p3_root / name
            for source in sorted(source_dir.rglob("*")):
                if not source.is_file():
                    continue
                rel = source.relative_to(p3_root).as_posix()
                destination = target / rel
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    shutil.copy2(source, destination)
                copied[rel] = _sha_file(source)
        runs_manifest: dict[str, str] = {}
        runs_root = p3_root / "runs"
        if runs_root.exists():
            for source in sorted(runs_root.rglob("*")):
                if source.is_file():
                    runs_manifest[
                        source.relative_to(p3_root).as_posix()
                    ] = _sha_file(source)
        reference_manifest = {
            source.relative_to(reference_root).as_posix(): _sha_file(source)
            for source in sorted(reference_root.rglob("*"))
            if source.is_file()
        }
        reference_cache_sha = _aggregate_sha(reference_manifest)
        try:
            payload = build_p3_freeze_payload(
                verdict=verdict,
                map_audit=_read_json(map_audit_path),
                dataset_audit=_read_json(dataset_audit_path),
                file_manifest=copied,
                reference_cache_sha256=reference_cache_sha,
                code_sha256=code_sha,
            )
        except ValueError as exc:
            return _blocked(stage, str(exc))
        payload["runs_aggregate_sha256"] = _aggregate_sha(runs_manifest)
        payload["runs_file_count"] = len(runs_manifest)
        pd.DataFrame(
            sorted(runs_manifest.items()), columns=["path", "sha256"]
        ).to_csv(target / "runs_sha_manifest.csv", index=False)
        pd.DataFrame(
            sorted(reference_manifest.items()), columns=["path", "sha256"]
        ).to_csv(target / "reference_cache_sha_manifest.csv", index=False)
        atomic_write_json(
            target / "pilot_feasibility_p3_freeze.json", payload
        )
        pointer_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            pointer_path,
            {
                "stage": stage,
                "frozen_dir_rel": f"{FREEZE_ROOT_REL}/{code_sha}",
                "code_sha256": code_sha,
                "freeze_sha256": _sha_file(
                    target / "pilot_feasibility_p3_freeze.json"
                ),
                "immutable": True,
            },
        )
        return StageResult(
            stage,
            "pass",
            EXIT_PASS,
            completed=len(copied),
            batch_complete=True,
            scope_complete=True,
            evidence={
                "frozen_files": len(copied),
                "runs_files_hashed": len(runs_manifest),
                "reference_cache_files": len(reference_manifest),
                "reference_cache_sha256": reference_cache_sha,
                "verdict_preserved": "underpowered_validation",
            },
        )

    return handler


def _frozen_evidence_dir(output_root: Path) -> Path | None:
    pointer_path = output_root / FREEZE_POINTER_REL
    if not pointer_path.exists():
        return None
    pointer = _read_json(pointer_path)
    frozen = output_root / str(pointer.get("frozen_dir_rel", ""))
    return frozen if frozen.exists() else None


# ---------------------------------------------------------------------------
# AuditDataGenerationAuthorizationV3
# ---------------------------------------------------------------------------

def _audit_dga_handler(
    project_root: Path, output_root: Path, config: dict
) -> Callable[[RuntimeOptions], StageResult]:
    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "AuditDataGenerationAuthorizationV3"
        frozen = _frozen_evidence_dir(output_root)
        if frozen is None:
            return _missing_result(
                stage, [str(output_root / FREEZE_POINTER_REL)]
            )
        contract_path = project_root / DGA_CONTRACT_REL
        map_path = frozen / "map" / "pilot_feasibility_audit.json"
        verdict_path = frozen / "evaluation" / "pilot_gate_v3_verdict.json"
        dataset_path = frozen / "dataset_v3" / "pilot_v3_dataset_audit.json"
        missing = [
            str(path)
            for path in (contract_path, map_path, verdict_path, dataset_path)
            if not path.exists()
        ]
        if missing:
            return _missing_result(stage, missing)
        verdict = _read_json(verdict_path)
        result = evaluate_data_generation_authorization_v3(
            _read_json(map_path), verdict, _read_json(dataset_path)
        )
        result["contract"] = DGA_CONTRACT_REL
        result["contract_sha256"] = _sha_file(contract_path)
        result["frozen_evidence_dir"] = str(frozen)
        result["code_sha256"] = working_code_sha(project_root)
        preserved = (
            result["p3_verdict_preserved"] == "underpowered_validation"
        )
        if not preserved:
            result["status"] = "blocked"
            result["scientific_pass"] = False
            result["train1600_planning_authorized"] = False
        auth_path = output_root / AUTH_REL
        auth_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(auth_path, result)
        msg = model_safety_gate_v3_status()
        msg["contract"] = MSG_CONTRACT_REL
        msg["evaluated_with_code_sha256"] = result["code_sha256"]
        atomic_write_json(output_root / MSG_REL, msg)
        passed = bool(result["scientific_pass"]) and preserved
        return StageResult(
            stage,
            "pass" if passed else "blocked",
            EXIT_PASS if passed else EXIT_BLOCKED,
            completed=1,
            batch_complete=True,
            scope_complete=passed,
            evidence={
                "scientific_pass": result["scientific_pass"],
                "train1600_planning_authorized": result[
                    "train1600_planning_authorized"
                ],
                "p3_verdict_preserved": result["p3_verdict_preserved"],
                "model_safety_gate_status": msg["status"],
                "conditions": result["conditions"],
            },
        )

    return handler


# ---------------------------------------------------------------------------
# PlanTrain1600V3
# ---------------------------------------------------------------------------

def _plan_train1600_v3_handler(
    project_root: Path, output_root: Path, config: dict
) -> Callable[[RuntimeOptions], StageResult]:
    """Freeze the whole Train1600 V3 candidate universe in one pass.

    320 states x 10 candidates (5 primary + 5 state-reserve) = 3200
    materialized candidates and a 12800-row branch plan; the Calibration and
    Locked Validation subsets are SHA-frozen here, before any corresponding
    SWMM run can exist.
    """

    def handler(options: RuntimeOptions) -> StageResult:
        from .pipeline import _file_sha256, _load_ledger
        from .pipeline_ext import _facility_inputs

        stage = "PlanTrain1600V3"
        auth_path = output_root / AUTH_REL
        if not auth_path.exists():
            return _missing_result(stage, [str(auth_path)])
        authorization = _read_json(auth_path)
        if not bool(authorization.get("train1600_planning_authorized")):
            return _blocked(
                stage, "data_generation_authorization_not_granted"
            )
        freeze_path = output_root / PLAN_FREEZE_REL
        if freeze_path.exists():
            freeze = _read_json(freeze_path)
            stale = {
                name: rel
                for name, rel in freeze.get("frozen_files", {}).items()
                if not (output_root / rel).exists()
                or _sha_file(output_root / rel)
                != freeze.get("sha256", {}).get(name)
            }
            if stale:
                return _blocked(
                    stage,
                    "frozen_plan_files_modified",
                    stale=sorted(stale),
                )
            return StageResult(
                stage,
                "pass",
                EXIT_PASS,
                completed=int(freeze.get("candidate_rows", 0)),
                batch_complete=True,
                scope_complete=True,
                evidence={"already_frozen": True, "plan_freeze": freeze},
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
            "dataset_contract": project_root / DATASET_CONTRACT_REL,
        }
        frozen = _frozen_evidence_dir(output_root)
        p3_map_path = (
            (frozen / "map" / "pilot_state_feasibility_map.csv")
            if frozen is not None
            else output_root
            / "pilot_feasibility_p3"
            / "map"
            / "pilot_state_feasibility_map.csv"
        )
        inputs["p3_state_feasibility_map"] = p3_map_path
        missing = sorted(
            name for name, path in inputs.items() if not path.exists()
        )
        if missing:
            return _missing_result(stage, missing)
        run_uuid = str(uuid.uuid4())
        planning = output_root / PLANNING_DIR_REL
        planning.mkdir(parents=True, exist_ok=True)
        try:
            facility = _facility_inputs(project_root, config)
            standard_catalog = pd.read_csv(
                inputs["standard_checkpoint_catalog"]
            )
            ledger = _load_ledger(inputs["event_usage_ledger"])
            selection_path = output_root / SELECTION_REL
            if selection_path.exists():
                selection = _read_json(selection_path)["selection"]
            else:
                selection = select_train1600_events(
                    standard_catalog, ledger
                )
            train_catalog, reserve_catalog = build_train_checkpoint_catalog(
                standard_catalog, selection
            )
            for split in ("train", "calibration", "locked_validation"):
                ledger = assign_split(
                    ledger,
                    selection[split],
                    split,
                    assignment_run_uuid=run_uuid,
                )
            ledger = assign_split(
                ledger,
                selection["reserve"],
                "reserve",
                assignment_run_uuid=run_uuid,
            )
            scorer = fit_state_feasibility_scorer(
                pd.read_csv(inputs["p3_state_feasibility_map"]),
                standard_catalog,
            )
            stratified = apply_state_feasibility_scorer(
                scorer, train_catalog
            )
            stratified_reserve = apply_state_feasibility_scorer(
                scorer, reserve_catalog
            )
            role_plan = build_v3_role_plan(stratified)
            candidate_plan, coverage_missing = (
                materialize_pilot_candidates(
                    role_plan,
                    train_catalog,
                    facility_ids=facility["facility_ids"],
                    facility_semantics=facility["facility_semantics"],
                    peak_boundary_anchor_library=pd.read_csv(
                        inputs["peak_boundary_anchor_library"]
                    ),
                    contract_sha256=_file_sha256(
                        inputs["dataset_contract"]
                    ),
                    config_sha256=(
                        _file_sha256(Path(options.config))
                        if options.config
                        and Path(options.config).exists()
                        else ""
                    ),
                    code_sha256=working_code_sha(project_root),
                    schedule_dir=planning / "schedules",
                    schedule_dir_relative_to=output_root,
                )
            )
            v3_columns = role_plan[
                [
                    "case_id",
                    "candidate_role_v3",
                    "predicted_stratum",
                    "plan_tier",
                    "replenish_source",
                ]
            ]
            candidate_plan = candidate_plan.merge(
                v3_columns, on="case_id", how="left"
            )
            branch_plan = build_pilot_branch_plan(
                candidate_plan,
                contract_sha256=_file_sha256(inputs["dataset_contract"]),
            )
            rotation = build_train_round_rotation_v3(train_catalog)
            round_assignment = assign_primary_candidates_to_rounds_v3(
                candidate_plan[
                    candidate_plan["plan_tier"].astype(str) == "primary"
                ],
                rotation,
            )
            round0_plan = round_assignment[
                round_assignment["round"] == 0
            ].drop(columns=["round"])
            calibration_plan = candidate_plan[
                (candidate_plan["split"].astype(str) == "calibration")
                & (candidate_plan["plan_tier"].astype(str) == "primary")
            ].reset_index(drop=True)
            locked_plan = candidate_plan[
                (
                    candidate_plan["split"].astype(str)
                    == "locked_validation"
                )
                & (candidate_plan["plan_tier"].astype(str) == "primary")
            ].reset_index(drop=True)
            progress = build_per_state_progress_v3(role_plan)
        except (EventShortfallError, KeyError, OSError, ValueError) as exc:
            evidence: dict = {"reason": str(exc)}
            if isinstance(exc, EventShortfallError):
                evidence["shortfall_report"] = exc.report
            return StageResult(
                stage, "blocked", EXIT_BLOCKED, evidence=evidence
            )
        train_catalog.to_csv(output_root / CATALOG_REL, index=False)
        reserve_catalog_out = stratified_reserve
        reserve_catalog_out.to_csv(
            output_root / RESERVE_CATALOG_REL, index=False
        )
        role_plan.to_csv(output_root / ROLE_PLAN_REL, index=False)
        candidate_plan.to_csv(output_root / MASTER_PLAN_REL, index=False)
        branch_plan.to_csv(
            output_root / T16_BRANCH_PLAN_REL, index=False
        )
        coverage_missing.to_csv(
            output_root / COVERAGE_MISSING_REL, index=False
        )
        rotation.to_csv(output_root / ROTATION_REL, index=False)
        rotation[
            [
                "event_id",
                "checkpoint_id",
                "state_id",
                "rest_round",
                "round0_target",
                "round1_target",
                "round2_target",
                "total_target",
            ]
        ].to_csv(output_root / STATE_TARGETS_REL, index=False)
        progress.to_csv(output_root / PROGRESS_REL, index=False)
        atomic_write_json(output_root / SCORER_REL, scorer)
        atomic_write_json(
            output_root / SELECTION_REL,
            {"selection": selection, "run_uuid": run_uuid},
        )
        round0_dir = output_root / T16_ROOT / "round0"
        round0_dir.mkdir(parents=True, exist_ok=True)
        round0_plan.to_csv(round0_dir / "plan.csv", index=False)
        calibration_plan.to_csv(
            output_root / CAL_FROZEN_REL, index=False
        )
        locked_plan.to_csv(output_root / LOCKED_FROZEN_REL, index=False)
        ledger.to_csv(inputs["event_usage_ledger"], index=False)
        frozen_files = {
            "master_candidate_plan": MASTER_PLAN_REL,
            "branch_plan": T16_BRANCH_PLAN_REL,
            "role_plan": ROLE_PLAN_REL,
            "train_checkpoint_catalog": CATALOG_REL,
            "rotation": ROTATION_REL,
            "round0_plan": f"{T16_ROOT}/round0/plan.csv",
            "calibration_plan_frozen": CAL_FROZEN_REL,
            "locked_validation_plan_frozen": LOCKED_FROZEN_REL,
        }
        freeze = {
            "stage": stage,
            "run_uuid": run_uuid,
            "frozen_files": frozen_files,
            "sha256": {
                name: _sha_file(output_root / rel)
                for name, rel in frozen_files.items()
            },
            "calibration_plan_sha256": _sha_file(
                output_root / CAL_FROZEN_REL
            ),
            "locked_validation_plan_sha256": _sha_file(
                output_root / LOCKED_FROZEN_REL
            ),
            "candidate_rows": int(len(candidate_plan)),
            "branch_rows": int(len(branch_plan)),
            "coverage_missing_rows": int(len(coverage_missing)),
            "plans_frozen_before_any_corresponding_swmm": True,
        }
        atomic_write_json(freeze_path, freeze)
        audit = audit_train1600_plan_v3(
            train_catalog,
            reserve_catalog,
            selection,
            role_plan,
            rotation,
            plan_freeze=freeze,
        )
        atomic_write_json(
            planning / "train1600_plan_audit_preview_v3.json", audit
        )
        atomic_write_json(
            planning / "completion.json",
            {
                "stage": stage,
                "run_uuid": run_uuid,
                "input_sha256": {
                    name: _file_sha256(path)
                    for name, path in inputs.items()
                },
                "candidate_rows": int(len(candidate_plan)),
                "branch_rows": int(len(branch_plan)),
                "round0_rows": int(len(round0_plan)),
                "calibration_rows": int(len(calibration_plan)),
                "locked_validation_rows": int(len(locked_plan)),
                "status": audit["status"],
            },
        )
        passed = audit["status"] == "pass"
        return StageResult(
            stage,
            "pass" if passed else "blocked",
            EXIT_PASS if passed else EXIT_BLOCKED,
            completed=int(len(candidate_plan)),
            batch_complete=True,
            scope_complete=passed,
            evidence={
                "selection": {
                    split: len(events)
                    for split, events in selection.items()
                },
                "candidate_rows": int(len(candidate_plan)),
                "branch_rows": int(len(branch_plan)),
                "coverage_missing_rows": int(len(coverage_missing)),
                "round0_rows": int(len(round0_plan)),
                "plan_audit_status": audit["status"],
                "scorer_method": scorer["method"],
            },
        )

    return handler


def _audit_train1600_plan_v3_handler(
    project_root: Path, output_root: Path, config: dict
) -> Callable[[RuntimeOptions], StageResult]:
    def handler(_options: RuntimeOptions) -> StageResult:
        from .pipeline import STAGE_ARTIFACTS

        stage = "AuditTrain1600PlanV3"
        paths = {
            "train_catalog": output_root / CATALOG_REL,
            "reserve_catalog": output_root / RESERVE_CATALOG_REL,
            "role_plan": output_root / ROLE_PLAN_REL,
            "rotation": output_root / ROTATION_REL,
            "selection": output_root / SELECTION_REL,
            "plan_freeze": output_root / PLAN_FREEZE_REL,
        }
        missing = [
            str(path) for path in paths.values() if not path.exists()
        ]
        if missing:
            return _missing_result(stage, missing)
        try:
            freeze = _read_json(paths["plan_freeze"])
            sha_ok = {
                name: _sha_file(output_root / rel)
                == freeze.get("sha256", {}).get(name)
                for name, rel in freeze.get("frozen_files", {}).items()
            }
            audit = audit_train1600_plan_v3(
                pd.read_csv(paths["train_catalog"]),
                pd.read_csv(paths["reserve_catalog"]),
                _read_json(paths["selection"])["selection"],
                pd.read_csv(paths["role_plan"]),
                pd.read_csv(paths["rotation"]),
                plan_freeze=freeze,
            )
        except (KeyError, OSError, ValueError) as exc:
            return _blocked(stage, str(exc))
        audit["frozen_file_sha_verified"] = sha_ok
        if not all(sha_ok.values()):
            audit["status"] = "blocked"
        artifact = output_root / STAGE_ARTIFACTS[stage]
        artifact.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(artifact, audit)
        passed = audit["status"] == "pass"
        return StageResult(
            stage,
            audit["status"],
            EXIT_PASS if passed else EXIT_BLOCKED,
            completed=1,
            batch_complete=True,
            scope_complete=passed,
            evidence=audit,
        )

    return handler


# ---------------------------------------------------------------------------
# Round build / audit (rounds 0-2, calibration, locked validation)
# ---------------------------------------------------------------------------

def _segment_dataset_dir(output_root: Path, seg: str) -> Path:
    return output_root / T16_ROOT / seg / "dataset"


def _load_segment_samples(
    output_root: Path, segments: list[str]
) -> pd.DataFrame:
    frames = [
        pd.read_csv(path)
        for seg in segments
        for path in [
            _segment_dataset_dir(output_root, seg)
            / "round_sample_manifest.csv"
        ]
        if path.exists()
    ]
    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()


_PLAN_CARRY_COLUMNS = (
    "split",
    "candidate_role_v3",
    "predicted_stratum",
    "plan_tier",
)


def _carry_plan_columns(
    samples: pd.DataFrame, candidate_plan: pd.DataFrame
) -> pd.DataFrame:
    """Merge plan-only V3 columns into the sample manifest by case id."""
    if not len(samples):
        return samples
    key = "case_id" if "case_id" in samples else "sample_id"
    if key not in samples or "case_id" not in candidate_plan:
        return samples
    extras = [
        column
        for column in _PLAN_CARRY_COLUMNS
        if column in candidate_plan and column not in samples
    ]
    if not extras:
        return samples
    right = candidate_plan[["case_id", *extras]].rename(
        columns={"case_id": key}
    )
    return samples.merge(right.drop_duplicates(key), on=key, how="left")


_SAMPLED_ONLY_LABEL_TOKEN = "sampled_only"
_SAMPLED_ONLY_STATE_KEYS = ["event_id", "checkpoint_id"]


def _stamp_sampled_only_labels(samples: pd.DataFrame) -> pd.DataFrame:
    """Stamp the sampled-only state-feasibility label contract (spec 1).

    Train1600 states are never subjected to a P3 Exact intervention search,
    so every accepted row is recorded ``sampled_only`` with
    ``exact_search_performed=False``.  The absence of a joint solution among
    the sampled candidates is reported literally through
    ``joint_found_in_sampled_set``; it is never rewritten into a
    ``fallback_only_under_budget`` verdict or a physical-infeasible flag, and
    the per-state ``candidate_search_budget`` only counts the candidates that
    were actually reduced for that state.
    """
    result = samples.copy()
    contract_columns = (
        "state_feasibility_label_source",
        "state_feasibility_label_validity",
        "candidate_search_budget",
        "exact_search_performed",
        "joint_found_in_sampled_set",
    )
    if not len(result):
        for column in contract_columns:
            result[column] = pd.Series(dtype=object)
        return result
    result["state_feasibility_label_source"] = _SAMPLED_ONLY_LABEL_TOKEN
    result["state_feasibility_label_validity"] = _SAMPLED_ONLY_LABEL_TOKEN
    result["exact_search_performed"] = False
    if set(_SAMPLED_ONLY_STATE_KEYS) <= set(result.columns):
        count_column = "sample_id" if "sample_id" in result else "event_id"
        result["candidate_search_budget"] = (
            result.groupby(_SAMPLED_ONLY_STATE_KEYS)[count_column]
            .transform("size")
            .astype(int)
        )
        joint = (
            result["joint_noninferior"].astype(bool)
            if "joint_noninferior" in result
            else pd.Series(False, index=result.index)
        )
        result["joint_found_in_sampled_set"] = (
            joint.groupby(
                [result[key] for key in _SAMPLED_ONLY_STATE_KEYS]
            )
            .transform("any")
            .astype(bool)
        )
    else:
        result["candidate_search_budget"] = int(len(result))
        result["joint_found_in_sampled_set"] = bool(
            result["joint_noninferior"].astype(bool).any()
            if "joint_noninferior" in result
            else False
        )
    return result


def _refresh_progress(output_root: Path) -> pd.DataFrame | None:
    """Recompute the per-state accepted ledger from built segments."""
    role_plan_path = output_root / ROLE_PLAN_REL
    if not role_plan_path.exists():
        return None
    accepted = _load_segment_samples(
        output_root, [seg for seg, _target in SEGMENTS_V3.values()]
    )
    progress = build_per_state_progress_v3(
        pd.read_csv(role_plan_path),
        accepted if len(accepted) else None,
    )
    progress.to_csv(output_root / PROGRESS_REL, index=False)
    return progress


def _reference_cache_check(output_root: Path) -> tuple[str, str, dict]:
    """Re-verify the frozen P3 reference cache files are untouched.

    Consistency means: every file recorded in the freeze-time manifest
    still exists with the same SHA.  New cache entries created for new V3
    states do not break consistency.
    """
    frozen = _frozen_evidence_dir(output_root)
    if frozen is None:
        return "", "", {"frozen_manifest_available": False}
    manifest_path = frozen / "reference_cache_sha_manifest.csv"
    payload_path = frozen / "pilot_feasibility_p3_freeze.json"
    if not manifest_path.exists() or not payload_path.exists():
        return "", "", {"frozen_manifest_available": False}
    expected = str(
        _read_json(payload_path).get("reference_cache_sha256", "")
    )
    manifest = pd.read_csv(manifest_path)
    reference_root = output_root / "pilot" / "references"
    current: dict[str, str] = {}
    missing = 0
    for _, row in manifest.iterrows():
        rel = str(row["path"])
        path = reference_root / rel
        if path.exists():
            current[rel] = _sha_file(path)
        else:
            current[rel] = ""
            missing += 1
    return (
        expected,
        _aggregate_sha(current),
        {
            "frozen_manifest_available": True,
            "frozen_files_checked": len(current),
            "frozen_files_missing": missing,
        },
    )


def _build_round_v3_handler(
    project_root: Path,
    output_root: Path,
    config: dict,
    *,
    stage: str,
    run_stage: str,
) -> Callable[[RuntimeOptions], StageResult]:
    """Reduce one segment's completed runs into its formal round dataset."""
    seg, _accepted_target = SEGMENTS_V3[run_stage]

    def handler(_options: RuntimeOptions) -> StageResult:
        from .pipeline import _add_pilot_audit_columns, _run_stage_sources
        from .pipeline_ext import _facility_inputs

        plan_path, run_root = _run_stage_sources(run_stage, output_root)
        branch_path = output_root / T16_BRANCH_PLAN_REL
        missing = [
            str(path)
            for path in (plan_path, branch_path)
            if not path.exists()
        ]
        if missing:
            return _missing_result(stage, missing)
        candidate_plan = pd.read_csv(plan_path)
        completions = (
            completion_manifest(run_root)
            if run_root.exists()
            else pd.DataFrame()
        )
        try:
            inputs = _facility_inputs(project_root, config)
            result = build_pilot_dataset(
                candidate_plan,
                pd.read_csv(branch_path),
                expand_pilot_completions(completions),
                priority_nodes=inputs["priority_nodes"],
                facility_ids=inputs["facility_ids"],
                scientific_margin=config["thresholds"][
                    "scientific_margin"
                ],
                dead_zone=config["thresholds"]["dead_zone"],
            )
            samples = _stamp_sampled_only_labels(
                _carry_plan_columns(
                    _add_pilot_audit_columns(
                        result["sample_manifest"], config
                    ),
                    candidate_plan,
                )
            )
        except (KeyError, OSError, ValueError) as exc:
            return _blocked(stage, str(exc))
        dataset_dir = _segment_dataset_dir(output_root, seg)
        dataset_dir.mkdir(parents=True, exist_ok=True)
        samples.to_csv(
            dataset_dir / "round_sample_manifest.csv", index=False
        )
        for name in (
            "branch_manifest",
            "rejected",
            "actual_duplicates",
            "pending",
        ):
            result[name].to_csv(
                dataset_dir / f"round_{name}.csv", index=False
            )
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
                "accounting": accounting,
                "accepted": int(accounting["accepted"]),
                "planned_rows": int(len(candidate_plan)),
            },
        )
        _refresh_progress(output_root)
        closed = bool(accounting.get("accounting_closed"))
        pending = int(len(result["pending"]))
        return StageResult(
            stage,
            "pass" if closed else "incomplete",
            EXIT_PASS if closed else EXIT_INCOMPLETE,
            completed=int(accounting["accepted"]),
            remaining=pending,
            batch_complete=True,
            scope_complete=closed,
            evidence={
                "segment": seg,
                "accepted": int(accounting["accepted"]),
                "pending": pending,
                "accounting_closed": closed,
            },
        )

    return handler


def _audit_round_v3_handler(
    project_root: Path,
    output_root: Path,
    config: dict,
    *,
    stage: str,
    run_stage: str,
) -> Callable[[RuntimeOptions], StageResult]:
    """Full-round hard gate: authenticity, accounting, reference cache,
    and (for Calibration/Locked) the pre-run plan freeze."""
    seg, accepted_target = SEGMENTS_V3[run_stage]

    def handler(_options: RuntimeOptions) -> StageResult:
        from .pipeline import STAGE_ARTIFACTS

        dataset_dir = _segment_dataset_dir(output_root, seg)
        manifest_path = dataset_dir / "round_sample_manifest.csv"
        completion_path = dataset_dir / "completion.json"
        missing = [
            str(path)
            for path in (manifest_path, completion_path)
            if not path.exists()
        ]
        if missing:
            return _missing_result(stage, missing)
        try:
            samples = pd.read_csv(manifest_path)
            accounting = _read_json(completion_path).get("accounting", {})
            expected_sha, current_sha, cache_detail = (
                _reference_cache_check(output_root)
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
        if seg in ("calibration", "locked_validation"):
            frozen_rel = (
                CAL_FROZEN_REL
                if seg == "calibration"
                else LOCKED_FROZEN_REL
            )
            sha_key = (
                "calibration_plan_sha256"
                if seg == "calibration"
                else "locked_validation_plan_sha256"
            )
            freeze_path = output_root / PLAN_FREEZE_REL
            plan_path = output_root / T16_ROOT / seg / "plan.csv"
            frozen_ok = False
            if freeze_path.exists() and plan_path.exists():
                expected_plan = str(
                    _read_json(freeze_path).get(sha_key, "")
                )
                frozen_ok = bool(
                    expected_plan
                    and _sha_file(plan_path) == expected_plan
                    and _sha_file(output_root / frozen_rel)
                    == expected_plan
                )
            audit["hard_checks"]["plan_frozen_before_run"] = frozen_ok
            if not frozen_ok:
                audit["status"] = "blocked"
        artifact = output_root / STAGE_ARTIFACTS[stage]
        artifact.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(artifact, audit)
        passed = audit["status"] == "pass"
        return StageResult(
            stage,
            audit["status"],
            EXIT_PASS if passed else EXIT_BLOCKED,
            completed=int(audit.get("accepted", 0)),
            batch_complete=True,
            scope_complete=passed,
            evidence=audit,
        )

    return handler


# ---------------------------------------------------------------------------
# Active learning (train split only) and round selection
# ---------------------------------------------------------------------------

def _active_learner_v3_handler(
    project_root: Path,
    output_root: Path,
    config: dict,
    *,
    stage: str,
    round_index: int,
) -> Callable[[RuntimeOptions], StageResult]:
    """Rank remaining Train candidates from completed Train rounds only.

    Calibration and Locked Validation datasets are never read here; a
    non-train row in the accepted input fails closed.
    """

    def handler(_options: RuntimeOptions) -> StageResult:
        master_path = output_root / MASTER_PLAN_REL
        role_plan_path = output_root / ROLE_PLAN_REL
        missing = [
            str(path)
            for path in (master_path, role_plan_path)
            if not path.exists()
        ]
        if missing:
            return _missing_result(stage, missing)
        train_segments = [
            f"round{index}" for index in range(round_index + 1)
        ]
        accepted = _load_segment_samples(output_root, train_segments)
        if not len(accepted):
            return _blocked(stage, "no_completed_train_round_data")
        try:
            assert_train_split_only(accepted)
            master = pd.read_csv(master_path)
            used: set[str] = set()
            for index in range(round_index + 1):
                plan_path = (
                    output_root / T16_ROOT / f"round{index}" / "plan.csv"
                )
                if plan_path.exists():
                    used.update(
                        pd.read_csv(plan_path)["case_id"].astype(str)
                    )
            progress = build_per_state_progress_v3(
                pd.read_csv(role_plan_path), accepted
            )
            remaining = master[
                ~master["case_id"].astype(str).isin(used)
            ]
            ranking = rank_remaining_candidates_v3(
                remaining, accepted, progress
            )
        except (KeyError, OSError, ValueError) as exc:
            return _blocked(stage, str(exc))
        progress.to_csv(output_root / PROGRESS_REL, index=False)
        out_dir = output_root / T16_ROOT / f"round{round_index}"
        out_dir.mkdir(parents=True, exist_ok=True)
        ranking.to_csv(out_dir / "al_ranking.csv", index=False)
        payload = {
            "stage": stage,
            "round_index": round_index,
            "reads_train_split_only": True,
            "calibration_locked_never_read": True,
            "sources": train_segments,
            "accepted_train_samples": int(len(accepted)),
            "ranked_candidates": int(len(ranking)),
            "full_states_skipped": int(
                (progress["accepted"] >= progress["target_accepted"])
                .sum()
            ),
        }
        atomic_write_json(out_dir / "active_learner_v3.json", payload)
        return StageResult(
            stage,
            "pass",
            EXIT_PASS,
            completed=int(len(ranking)),
            batch_complete=True,
            scope_complete=True,
            evidence=payload,
        )

    return handler


def _select_round_v3_handler(
    project_root: Path,
    output_root: Path,
    config: dict,
    *,
    stage: str,
    round_index: int,
) -> Callable[[RuntimeOptions], StageResult]:
    """Materialize this round's executable plan from the frozen pool."""

    def handler(_options: RuntimeOptions) -> StageResult:
        master_path = output_root / MASTER_PLAN_REL
        rotation_path = output_root / ROTATION_REL
        progress_path = output_root / PROGRESS_REL
        ranking_path = (
            output_root
            / T16_ROOT
            / f"round{round_index - 1}"
            / "al_ranking.csv"
        )
        missing = [
            str(path)
            for path in (
                master_path,
                rotation_path,
                progress_path,
                ranking_path,
            )
            if not path.exists()
        ]
        if missing:
            return _missing_result(stage, missing)
        try:
            master = pd.read_csv(master_path)
            rotation = pd.read_csv(rotation_path)
            progress = pd.read_csv(progress_path)
            ranking = pd.read_csv(ranking_path)
            used: set[str] = set()
            for index in range(round_index):
                plan_path = (
                    output_root / T16_ROOT / f"round{index}" / "plan.csv"
                )
                if plan_path.exists():
                    used.update(
                        pd.read_csv(plan_path)["case_id"].astype(str)
                    )
            selected = select_round_candidates_v3(
                master,
                rotation,
                round_index,
                progress,
                used,
                ranking=ranking,
            )
        except (KeyError, OSError, ValueError) as exc:
            return _blocked(stage, str(exc))
        if not len(selected):
            return _blocked(stage, "no_candidates_selected")
        non_train = set(selected["split"].astype(str)) - {"train"}
        if non_train:
            return _blocked(
                stage,
                "selected_non_train_rows",
                splits=sorted(non_train),
            )
        out_dir = output_root / T16_ROOT / f"round{round_index}"
        out_dir.mkdir(parents=True, exist_ok=True)
        selected.to_csv(out_dir / "plan.csv", index=False)
        target_total = int(
            rotation[f"round{round_index}_target"].sum()
        )
        atomic_write_json(
            out_dir / "selection_completion.json",
            {
                "stage": stage,
                "round_index": round_index,
                "selected_rows": int(len(selected)),
                "round_target_total": target_total,
                "reused_case_ids": 0,
            },
        )
        return StageResult(
            stage,
            "pass",
            EXIT_PASS,
            completed=int(len(selected)),
            batch_complete=True,
            scope_complete=True,
            evidence={
                "round_index": round_index,
                "selected_rows": int(len(selected)),
                "round_target_total": target_total,
            },
        )

    return handler


# ---------------------------------------------------------------------------
# Calibration / Locked Validation frozen plans (Round 3)
# ---------------------------------------------------------------------------

def _plan_frozen_segment_handler(
    project_root: Path,
    output_root: Path,
    config: dict,
    *,
    stage: str,
    seg: str,
    frozen_rel: str,
    sha_key: str,
) -> Callable[[RuntimeOptions], StageResult]:
    """Publish a pre-frozen Round 3 plan after re-verifying its SHA.

    The plan content was frozen by PlanTrain1600V3 before any
    corresponding SWMM run; this stage only verifies and copies it, so
    Calibration results can never alter the Locked Validation plan.
    """

    def handler(_options: RuntimeOptions) -> StageResult:
        from .pipeline import STAGE_ARTIFACTS

        freeze_path = output_root / PLAN_FREEZE_REL
        frozen_plan = output_root / frozen_rel
        missing = [
            str(path)
            for path in (freeze_path, frozen_plan)
            if not path.exists()
        ]
        if missing:
            return _missing_result(stage, missing)
        expected = str(_read_json(freeze_path).get(sha_key, ""))
        actual = _sha_file(frozen_plan)
        if not expected or actual != expected:
            return _blocked(
                stage,
                "frozen_plan_sha_mismatch",
                expected_sha256=expected,
                actual_sha256=actual,
            )
        plan = pd.read_csv(frozen_plan)
        _seg, target = SEGMENTS_V3[
            "RunCalibration200V3"
            if seg == "calibration"
            else "RunLockedValidation200V3"
        ]
        artifact = output_root / STAGE_ARTIFACTS[stage]
        artifact.parent.mkdir(parents=True, exist_ok=True)
        plan.to_csv(artifact, index=False)
        atomic_write_json(
            artifact.parent / "plan_freeze_check.json",
            {
                "stage": stage,
                "segment": seg,
                "frozen_plan": frozen_rel,
                "frozen_sha256": expected,
                "published_rows": int(len(plan)),
                "expected_rows": int(target),
                "frozen_before_any_corresponding_swmm": True,
            },
        )
        rows_ok = len(plan) == target
        return StageResult(
            stage,
            "pass" if rows_ok else "blocked",
            EXIT_PASS if rows_ok else EXIT_BLOCKED,
            completed=int(len(plan)),
            batch_complete=True,
            scope_complete=rows_ok,
            evidence={
                "segment": seg,
                "rows": int(len(plan)),
                "expected_rows": int(target),
                "frozen_sha_verified": True,
            },
        )

    return handler


# ---------------------------------------------------------------------------
# Final 1600 dataset build / audit
# ---------------------------------------------------------------------------

_FINAL_SEGMENTS = (
    "round0",
    "round1",
    "round2",
    "calibration",
    "locked_validation",
)


def _build_final_dataset_handler(
    project_root: Path, output_root: Path, config: dict
) -> Callable[[RuntimeOptions], StageResult]:
    def handler(_options: RuntimeOptions) -> StageResult:
        from .pipeline import STAGE_ARTIFACTS

        stage = "BuildTrain1600DatasetV3"
        missing = [
            str(_segment_dataset_dir(output_root, seg))
            for seg in _FINAL_SEGMENTS
            if not (
                _segment_dataset_dir(output_root, seg)
                / "round_sample_manifest.csv"
            ).exists()
        ]
        if missing:
            return _missing_result(stage, missing)
        samples = _load_segment_samples(
            output_root, list(_FINAL_SEGMENTS)
        )
        accounting = {
            seg: _read_json(
                _segment_dataset_dir(output_root, seg)
                / "completion.json"
            ).get("accounting", {})
            for seg in _FINAL_SEGMENTS
        }
        artifact = output_root / STAGE_ARTIFACTS[stage]
        artifact.parent.mkdir(parents=True, exist_ok=True)
        samples.to_csv(artifact, index=False)
        atomic_write_json(
            artifact.parent / "completion.json",
            {
                "stage": stage,
                "total_samples": int(len(samples)),
                "segment_accounting": accounting,
                "segments": list(_FINAL_SEGMENTS),
            },
        )
        return StageResult(
            stage,
            "pass",
            EXIT_PASS,
            completed=int(len(samples)),
            batch_complete=True,
            scope_complete=True,
            evidence={
                "total_samples": int(len(samples)),
                "per_segment": {
                    seg: int(item.get("accepted", 0))
                    for seg, item in accounting.items()
                },
            },
        )

    return handler


def _audit_final_dataset_handler(
    project_root: Path, output_root: Path, config: dict
) -> Callable[[RuntimeOptions], StageResult]:
    def handler(_options: RuntimeOptions) -> StageResult:
        from .pipeline import STAGE_ARTIFACTS

        stage = "AuditTrain1600DatasetV3"
        manifest_path = (
            output_root / STAGE_ARTIFACTS["BuildTrain1600DatasetV3"]
        )
        catalog_path = output_root / CATALOG_REL
        selection_path = output_root / SELECTION_REL
        missing = [
            str(path)
            for path in (manifest_path, catalog_path, selection_path)
            if not path.exists()
        ]
        if missing:
            return _missing_result(stage, missing)
        try:
            audit = audit_train1600_dataset_v3(
                pd.read_csv(manifest_path),
                pd.read_csv(catalog_path),
                _read_json(selection_path)["selection"],
                hard_columns=HARD_AUTHENTICITY_COLUMNS,
            )
        except (KeyError, OSError, ValueError) as exc:
            return _blocked(stage, str(exc))
        artifact = output_root / STAGE_ARTIFACTS[stage]
        artifact.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(artifact, audit)
        passed = audit["status"] == "pass"
        return StageResult(
            stage,
            audit["status"],
            EXIT_PASS if passed else EXIT_BLOCKED,
            completed=1,
            batch_complete=True,
            scope_complete=passed,
            evidence=audit,
        )

    return handler


# ---------------------------------------------------------------------------
# Registry factory
# ---------------------------------------------------------------------------

_BUILD_AUDIT_ROUND_STAGES = (
    ("BuildTrainRound0V3", "AuditTrainRound0V3", "RunTrainRound0V3"),
    ("BuildTrainRound1V3", "AuditTrainRound1V3", "RunTrainRound1V3"),
    ("BuildTrainRound2V3", "AuditTrainRound2V3", "RunTrainRound2V3"),
    (
        "BuildCalibration200V3",
        "AuditCalibration200V3",
        "RunCalibration200V3",
    ),
    (
        "BuildLockedValidation200V3",
        "AuditLockedValidation200V3",
        "RunLockedValidation200V3",
    ),
)


def build_train_v3_handlers(
    *, project_root: Path, output_root: Path, config: dict
) -> dict[str, Callable[[RuntimeOptions], StageResult]]:
    """Handlers for the Train1600 V3 chain, merged by ``build_registry``.

    Run stages reuse the pilot four-branch runner (shared v1 reference
    cache under ``pilot/``); partial stages reuse the pilot partial
    snapshot/gate pair against the V3 branch plan.  Nothing here starts
    a formal run by itself.
    """
    from .pipeline import (
        _audit_pilot_partial_handler,
        _build_pilot_partial_handler,
        _run_pilot400_handler,
    )

    handlers: dict[str, Callable[[RuntimeOptions], StageResult]] = {
        "FreezeP3Evidence": _freeze_p3_handler(
            project_root, output_root, config
        ),
        "AuditDataGenerationAuthorizationV3": _audit_dga_handler(
            project_root, output_root, config
        ),
        "PlanTrain1600V3": _plan_train1600_v3_handler(
            project_root, output_root, config
        ),
        "AuditTrain1600PlanV3": _audit_train1600_plan_v3_handler(
            project_root, output_root, config
        ),
        "TrainActiveLearner0V3": _active_learner_v3_handler(
            project_root,
            output_root,
            config,
            stage="TrainActiveLearner0V3",
            round_index=0,
        ),
        "TrainActiveLearner1V3": _active_learner_v3_handler(
            project_root,
            output_root,
            config,
            stage="TrainActiveLearner1V3",
            round_index=1,
        ),
        "SelectTrainRound1V3": _select_round_v3_handler(
            project_root,
            output_root,
            config,
            stage="SelectTrainRound1V3",
            round_index=1,
        ),
        "SelectTrainRound2V3": _select_round_v3_handler(
            project_root,
            output_root,
            config,
            stage="SelectTrainRound2V3",
            round_index=2,
        ),
        "PlanCalibration200V3": _plan_frozen_segment_handler(
            project_root,
            output_root,
            config,
            stage="PlanCalibration200V3",
            seg="calibration",
            frozen_rel=CAL_FROZEN_REL,
            sha_key="calibration_plan_sha256",
        ),
        "PlanLockedValidation200V3": _plan_frozen_segment_handler(
            project_root,
            output_root,
            config,
            stage="PlanLockedValidation200V3",
            seg="locked_validation",
            frozen_rel=LOCKED_FROZEN_REL,
            sha_key="locked_validation_plan_sha256",
        ),
        "BuildTrain1600DatasetV3": _build_final_dataset_handler(
            project_root, output_root, config
        ),
        "AuditTrain1600DatasetV3": _audit_final_dataset_handler(
            project_root, output_root, config
        ),
    }
    for run_stage in SEGMENTS_V3:
        handlers[run_stage] = _run_pilot400_handler(
            output_root,
            config,
            stage=run_stage,
            branch_plan_rel=T16_BRANCH_PLAN_REL,
        )
    for build_stage, audit_stage, run_stage in _BUILD_AUDIT_ROUND_STAGES:
        handlers[build_stage] = _build_round_v3_handler(
            project_root,
            output_root,
            config,
            stage=build_stage,
            run_stage=run_stage,
        )
        handlers[audit_stage] = _audit_round_v3_handler(
            project_root,
            output_root,
            config,
            stage=audit_stage,
            run_stage=run_stage,
        )
    for index in range(3):
        run_stage = f"RunTrainRound{index}V3"
        handlers[f"BuildTrainRound{index}PartialV3"] = (
            _build_pilot_partial_handler(
                f"BuildTrainRound{index}PartialV3",
                project_root,
                output_root,
                config,
                run_stage=run_stage,
                branch_plan_rel=T16_BRANCH_PLAN_REL,
            )
        )
        handlers[f"AuditTrainRound{index}PartialV3"] = (
            _audit_pilot_partial_handler(
                f"AuditTrainRound{index}PartialV3",
                project_root,
                output_root,
                config,
                run_stage=run_stage,
                branch_plan_rel=T16_BRANCH_PLAN_REL,
            )
        )
    return handlers
