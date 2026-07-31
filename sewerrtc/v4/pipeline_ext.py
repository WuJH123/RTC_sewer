"""Stage handlers for the Pilot Coverage Extension and Dataset/Gate v2.

Wired lazily from ``pipeline.build_registry`` so this module may import
``pipeline`` helpers at module level without a circular import.  All stages
write only under ``pilot_extension_v1/`` and ``pilot/dataset_v2/`` (plus the
Gate v2 verdict under ``pilot/evaluation/``); the frozen Pilot400 v1 plan,
runs and dataset directories are never modified.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pandas as pd

from .partial_audit import HARD_AUTHENTICITY_COLUMNS
from .pilot_candidates import build_pilot_branch_plan
from .pilot_extension import (
    FLAT_AUXILIARY_PHASE,
    JOINT_EXTENSION_PHASE,
    audit_pilot_coverage_gaps,
    audit_pilot_extension_dataset,
    audit_pilot_extension_plan,
    plan_pilot_coverage_extension,
    plan_pilot_flat_auxiliary,
    select_joint_extension_states,
)
from .pilot_reducers import build_pilot_dataset
from .pilot_run import expand_pilot_completions
from .pilot_v2 import (
    audit_pilot_dataset_v2,
    build_pilot_dataset_v2,
    evaluate_pilot_gate_v2,
    train_pilot_baselines_v2,
)
from .runtime import (
    EXIT_BLOCKED,
    EXIT_INCOMPLETE,
    EXIT_PASS,
    EXIT_SCIENTIFIC_FAIL,
    RuntimeOptions,
    StageResult,
    atomic_write_json,
    completion_manifest,
)

EXTENSION_ROOT = "pilot_extension_v1"
EXTENSION_BRANCH_PLAN = f"{EXTENSION_ROOT}/planning/extension_branch_plan.csv"
FLAT_AUX_BRANCH_PLAN = (
    f"{EXTENSION_ROOT}/flat_auxiliary/planning/flat_auxiliary_branch_plan.csv"
)
GAP_AUDIT_REL = f"{EXTENSION_ROOT}/gaps/pilot_v1_gap_audit.json"
GAP_CATALOG_REL = f"{EXTENSION_ROOT}/gaps/pilot_v1_state_gap_catalog.csv"
V1_SAMPLES_REL = "pilot/dataset/pilot_sample_manifest.csv"
V1_BRANCHES_REL = "pilot/dataset/pilot_branch_manifest.csv"
V1_CANDIDATE_PLAN_REL = "pilot/planning/pilot_candidate_plan.csv"


def _facility_inputs(project_root: Path, config: dict) -> dict:
    from .pipeline import _file_sha256, _read_facility_ids

    project = config.get("project", {})
    return {
        "facility_ids": _read_facility_ids(
            project_root / str(project.get("canonical_ids", ""))
        ),
        "facility_semantics": pd.read_csv(
            project_root / str(project.get("facility_semantics", ""))
        ),
        "priority_nodes": _read_facility_ids(
            project_root / str(project.get("priority_nodes", ""))
        ),
        "contract_sha": _file_sha256(
            project_root / str(project.get("contract", ""))
        ),
    }


def _missing_inputs_result(stage: str, missing: list[str]) -> StageResult:
    return StageResult(
        stage,
        "incomplete",
        EXIT_INCOMPLETE,
        remaining=1,
        evidence={"missing_inputs": missing},
    )


def _audit_coverage_gaps_handler(
    project_root: Path, output_root: Path, config: dict
) -> Callable[[RuntimeOptions], StageResult]:
    """Read-only diagnosis of the frozen v1 dataset; never edits v1 files."""

    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "AuditPilotCoverageGaps"
        source = output_root / V1_SAMPLES_REL
        if not source.exists():
            return _missing_inputs_result(stage, [str(source)])
        try:
            inputs = _facility_inputs(project_root, config)
            result = audit_pilot_coverage_gaps(
                pd.read_csv(source),
                facility_ids=inputs["facility_ids"],
            )
        except (KeyError, OSError, ValueError) as exc:
            return StageResult(
                stage, "blocked", EXIT_BLOCKED, evidence={"reason": str(exc)}
            )
        gaps_dir = output_root / EXTENSION_ROOT / "gaps"
        gaps_dir.mkdir(parents=True, exist_ok=True)
        result["state_gap_catalog"].to_csv(
            gaps_dir / "pilot_v1_state_gap_catalog.csv", index=False
        )
        result["missing_joint_states"].to_csv(
            gaps_dir / "missing_joint_states.csv", index=False
        )
        result["missing_flat_event_support"].to_csv(
            gaps_dir / "missing_flat_event_support.csv", index=False
        )
        result["candidate_family_gap"].to_csv(
            gaps_dir / "candidate_family_gap.csv", index=False
        )
        result["actuator_coverage_gap"].to_csv(
            gaps_dir / "actuator_coverage_gap.csv", index=False
        )
        audit = result["gap_audit"]
        atomic_write_json(gaps_dir / "pilot_v1_gap_audit.json", audit)
        atomic_write_json(
            gaps_dir / "completion.json",
            {"stage": stage, "read_only": True, "audit": audit},
        )
        return StageResult(
            stage,
            "pass",
            EXIT_PASS,
            completed=int(audit["states_total"]),
            batch_complete=True,
            scope_complete=True,
            evidence=audit,
        )

    return handler


def _plan_coverage_extension_handler(
    project_root: Path, output_root: Path, config: dict
) -> Callable[[RuntimeOptions], StageResult]:
    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "PlanPilotCoverageExtension"
        gap_path = output_root / GAP_CATALOG_REL
        v1_plan_path = output_root / V1_CANDIDATE_PLAN_REL
        v1_samples_path = output_root / V1_SAMPLES_REL
        missing = [
            str(path)
            for path in (gap_path, v1_plan_path, v1_samples_path)
            if not path.exists()
        ]
        if missing:
            return _missing_inputs_result(stage, missing)
        planning_dir = output_root / EXTENSION_ROOT / "planning"
        planning_dir.mkdir(parents=True, exist_ok=True)
        try:
            inputs = _facility_inputs(project_root, config)
            selected = select_joint_extension_states(pd.read_csv(gap_path))
            result = plan_pilot_coverage_extension(
                selected,
                pd.read_csv(v1_plan_path),
                pd.read_csv(v1_samples_path),
                reference_root=output_root / "pilot",
                facility_ids=inputs["facility_ids"],
                facility_semantics=inputs["facility_semantics"],
                schedule_dir=planning_dir / "schedules",
                schedule_dir_relative_to=output_root,
            )
            candidate_plan = result["candidate_plan"]
            branch_plan = build_pilot_branch_plan(
                candidate_plan, contract_sha256=inputs["contract_sha"]
            )
        except (KeyError, OSError, ValueError) as exc:
            return StageResult(
                stage, "blocked", EXIT_BLOCKED, evidence={"reason": str(exc)}
            )
        selected.to_csv(
            planning_dir / "extension_state_selection.csv", index=False
        )
        candidate_plan.to_csv(
            planning_dir / "extension_candidate_plan.csv", index=False
        )
        branch_plan.to_csv(
            planning_dir / "extension_branch_plan.csv", index=False
        )
        result["coverage_missing"].to_csv(
            planning_dir / "extension_coverage_missing.csv", index=False
        )
        summary = {
            "stage": stage,
            "source_phase": JOINT_EXTENSION_PHASE,
            "candidates": int(len(candidate_plan)),
            "states": int(
                candidate_plan.drop_duplicates(
                    ["event_id", "checkpoint_id"]
                ).shape[0]
            ),
            "events": int(candidate_plan["event_id"].nunique()),
            "families": sorted(
                candidate_plan["candidate_family"].astype(str).unique()
            ),
            "reference_reused": True,
            "new_reference_runs_planned": 0,
        }
        atomic_write_json(planning_dir / "completion.json", summary)
        return StageResult(
            stage,
            "pass",
            EXIT_PASS,
            completed=int(len(candidate_plan)),
            batch_complete=True,
            scope_complete=True,
            evidence=summary,
        )

    return handler


def _audit_extension_plan_handler(
    output_root: Path,
) -> Callable[[RuntimeOptions], StageResult]:
    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "AuditPilotCoverageExtensionPlan"
        planning_dir = output_root / EXTENSION_ROOT / "planning"
        plan_path = planning_dir / "extension_candidate_plan.csv"
        branch_path = planning_dir / "extension_branch_plan.csv"
        v1_plan_path = output_root / V1_CANDIDATE_PLAN_REL
        v1_samples_path = output_root / V1_SAMPLES_REL
        missing = [
            str(path)
            for path in (
                plan_path,
                branch_path,
                v1_plan_path,
                v1_samples_path,
            )
            if not path.exists()
        ]
        if missing:
            return _missing_inputs_result(stage, missing)
        try:
            audit = audit_pilot_extension_plan(
                pd.read_csv(plan_path),
                pd.read_csv(branch_path),
                pd.read_csv(v1_plan_path),
                pd.read_csv(v1_samples_path),
                source_phase=JOINT_EXTENSION_PHASE,
                min_candidates=32,
                max_candidates=60,
                min_states=8,
                min_events=3,
                min_families=4,
            )
        except (KeyError, OSError, ValueError) as exc:
            return StageResult(
                stage, "blocked", EXIT_BLOCKED, evidence={"reason": str(exc)}
            )
        atomic_write_json(
            planning_dir / "extension_plan_audit.json", audit
        )
        passed = audit["status"] == "pass"
        return StageResult(
            stage,
            audit["status"],
            EXIT_PASS if passed else EXIT_BLOCKED,
            completed=int(len(pd.read_csv(plan_path))),
            batch_complete=True,
            scope_complete=passed,
            evidence=audit,
        )

    return handler


def _plan_flat_auxiliary_handler(
    project_root: Path, output_root: Path, config: dict
) -> Callable[[RuntimeOptions], StageResult]:
    """Conditional Flat Auxiliary plan; only armed when event support < 3."""

    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "PlanPilotFlatAuxiliary"
        gap_audit_path = output_root / GAP_AUDIT_REL
        gap_path = output_root / GAP_CATALOG_REL
        v1_plan_path = output_root / V1_CANDIDATE_PLAN_REL
        v1_samples_path = output_root / V1_SAMPLES_REL
        missing = [
            str(path)
            for path in (
                gap_audit_path,
                gap_path,
                v1_plan_path,
                v1_samples_path,
            )
            if not path.exists()
        ]
        if missing:
            return _missing_inputs_result(stage, missing)
        gap_audit = json.loads(gap_audit_path.read_text(encoding="utf-8"))
        planning_dir = (
            output_root / EXTENSION_ROOT / "flat_auxiliary" / "planning"
        )
        planning_dir.mkdir(parents=True, exist_ok=True)
        if not bool(gap_audit.get("must_run_flat_auxiliary", False)):
            summary = {
                "stage": stage,
                "not_required": True,
                "confirmed_flat_event_support": gap_audit.get(
                    "confirmed_flat_event_support"
                ),
            }
            pd.DataFrame(
                columns=["sample_id", "event_id", "checkpoint_id"]
            ).to_csv(
                planning_dir / "flat_auxiliary_candidate_plan.csv",
                index=False,
            )
            atomic_write_json(planning_dir / "completion.json", summary)
            return StageResult(
                stage,
                "pass",
                EXIT_PASS,
                completed=0,
                batch_complete=True,
                scope_complete=True,
                evidence=summary,
            )
        try:
            inputs = _facility_inputs(project_root, config)
            v1_plan = pd.read_csv(v1_plan_path)
            v1_samples = pd.read_csv(v1_samples_path)
            result = plan_pilot_flat_auxiliary(
                pd.read_csv(gap_path),
                v1_plan,
                v1_samples,
                facility_ids=inputs["facility_ids"],
                facility_semantics=inputs["facility_semantics"],
                schedule_dir=planning_dir / "schedules",
                schedule_dir_relative_to=output_root,
            )
            candidate_plan = result["candidate_plan"]
            branch_plan = build_pilot_branch_plan(
                candidate_plan, contract_sha256=inputs["contract_sha"]
            )
            # No separate plan-audit stage exists for the auxiliary branch,
            # so the plan gate runs fail-closed right here.
            audit = audit_pilot_extension_plan(
                candidate_plan,
                branch_plan,
                v1_plan,
                v1_samples,
                source_phase=FLAT_AUXILIARY_PHASE,
                min_candidates=12,
                max_candidates=30,
                min_states=4,
                min_events=3,
                min_families=2,
                per_state_min=3,
                per_state_max=5,
            )
        except (KeyError, OSError, ValueError) as exc:
            return StageResult(
                stage, "blocked", EXIT_BLOCKED, evidence={"reason": str(exc)}
            )
        atomic_write_json(
            planning_dir / "flat_auxiliary_plan_audit.json", audit
        )
        if audit["status"] != "pass":
            return StageResult(
                stage, "blocked", EXIT_BLOCKED, evidence=audit
            )
        candidate_plan.to_csv(
            planning_dir / "flat_auxiliary_candidate_plan.csv", index=False
        )
        branch_plan.to_csv(
            planning_dir / "flat_auxiliary_branch_plan.csv", index=False
        )
        summary = {
            "stage": stage,
            "source_phase": FLAT_AUXILIARY_PHASE,
            "candidates": int(len(candidate_plan)),
            "states": int(
                candidate_plan.drop_duplicates(
                    ["event_id", "checkpoint_id"]
                ).shape[0]
            ),
            "events": int(candidate_plan["event_id"].nunique()),
            "plan_audit": audit["status"],
        }
        atomic_write_json(planning_dir / "completion.json", summary)
        return StageResult(
            stage,
            "pass",
            EXIT_PASS,
            completed=int(len(candidate_plan)),
            batch_complete=True,
            scope_complete=True,
            evidence=summary,
        )

    return handler


def _build_extension_dataset_handler(
    stage: str,
    project_root: Path,
    output_root: Path,
    config: dict,
    *,
    run_stage: str,
    branch_plan_rel: str,
    dataset_dir_rel: str,
    prefix: str,
) -> Callable[[RuntimeOptions], StageResult]:
    """Full-scope extension dataset from the extension run manifest."""

    def handler(_options: RuntimeOptions) -> StageResult:
        from .pipeline import (
            RUN_STAGE_PLANS,
            _add_pilot_audit_columns,
            _run_stage_sources,
        )

        plan_path, run_root = _run_stage_sources(run_stage, output_root)
        branch_path = output_root / branch_plan_rel
        missing = [
            str(path)
            for path in (plan_path, branch_path)
            if not path.exists()
        ]
        if missing:
            return _missing_inputs_result(stage, missing)
        candidate_plan = pd.read_csv(plan_path)
        branch_plan = pd.read_csv(branch_path)
        completions = (
            completion_manifest(run_root)
            if run_root.exists()
            else pd.DataFrame()
        )
        try:
            inputs = _facility_inputs(project_root, config)
            result = build_pilot_dataset(
                candidate_plan,
                branch_plan,
                expand_pilot_completions(completions),
                priority_nodes=inputs["priority_nodes"],
                facility_ids=inputs["facility_ids"],
                scientific_margin=config["thresholds"][
                    "scientific_margin"
                ],
                dead_zone=config["thresholds"]["dead_zone"],
            )
        except (KeyError, OSError, ValueError) as exc:
            return StageResult(
                stage, "blocked", EXIT_BLOCKED, evidence={"reason": str(exc)}
            )
        samples = _add_pilot_audit_columns(result["sample_manifest"], config)
        accounting = result["accounting"]
        dataset_dir = output_root / dataset_dir_rel
        dataset_dir.mkdir(parents=True, exist_ok=True)
        samples.to_csv(
            dataset_dir / f"{prefix}_sample_manifest.csv", index=False
        )
        result["branch_manifest"].to_csv(
            dataset_dir / f"{prefix}_branch_manifest.csv", index=False
        )
        result["rejected"].to_csv(
            dataset_dir / f"{prefix}_rejected.csv", index=False
        )
        result["actual_duplicates"].to_csv(
            dataset_dir / f"{prefix}_actual_duplicates.csv", index=False
        )
        result["pending"].to_csv(
            dataset_dir / f"{prefix}_pending.csv", index=False
        )
        result["missing_confirmed"].to_csv(
            dataset_dir / f"{prefix}_missing.csv", index=False
        )
        atomic_write_json(
            dataset_dir / f"{prefix}_provenance.json",
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


def _audit_extension_dataset_handler(
    stage: str,
    output_root: Path,
    *,
    dataset_dir_rel: str,
    prefix: str,
    audit_name: str,
    expected_source_phase: str,
) -> Callable[[RuntimeOptions], StageResult]:
    def handler(_options: RuntimeOptions) -> StageResult:
        dataset_dir = output_root / dataset_dir_rel
        sample_path = dataset_dir / f"{prefix}_sample_manifest.csv"
        completion_path = dataset_dir / "completion.json"
        v1_samples_path = output_root / V1_SAMPLES_REL
        missing = [
            str(path)
            for path in (sample_path, completion_path, v1_samples_path)
            if not path.exists()
        ]
        if missing:
            return _missing_inputs_result(stage, missing)
        completion = json.loads(
            completion_path.read_text(encoding="utf-8")
        )
        samples = pd.read_csv(sample_path)
        v1_actual = set(
            pd.read_csv(v1_samples_path)["actual_schedule_sha256"].astype(
                str
            )
        )
        try:
            audit = audit_pilot_extension_dataset(
                samples,
                completion.get("accounting", {}),
                expected_source_phase=expected_source_phase,
                v1_actual_shas=v1_actual,
                hard_columns=HARD_AUTHENTICITY_COLUMNS,
            )
        except (KeyError, OSError, ValueError) as exc:
            return StageResult(
                stage, "blocked", EXIT_BLOCKED, evidence={"reason": str(exc)}
            )
        atomic_write_json(dataset_dir / audit_name, audit)
        passed = audit["status"] == "pass"
        return StageResult(
            stage,
            audit["status"],
            EXIT_PASS if passed else EXIT_SCIENTIFIC_FAIL,
            completed=int(len(samples)),
            batch_complete=True,
            scope_complete=passed,
            evidence=audit,
        )

    return handler


def _build_dataset_v2_handler(
    output_root: Path,
) -> Callable[[RuntimeOptions], StageResult]:
    """Merge primary400 + extension (+ conditional flat auxiliary) into v2."""

    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "BuildPilotDatasetV2"
        v1_samples_path = output_root / V1_SAMPLES_REL
        ext_dir = output_root / EXTENSION_ROOT / "dataset"
        ext_samples_path = ext_dir / "extension_sample_manifest.csv"
        ext_audit_path = ext_dir / "extension_dataset_audit.json"
        gap_audit_path = output_root / GAP_AUDIT_REL
        missing = [
            str(path)
            for path in (
                v1_samples_path,
                ext_samples_path,
                ext_audit_path,
                gap_audit_path,
            )
            if not path.exists()
        ]
        if missing:
            return _missing_inputs_result(stage, missing)
        ext_audit = json.loads(ext_audit_path.read_text(encoding="utf-8"))
        if ext_audit.get("status") != "pass":
            return StageResult(
                stage,
                "blocked",
                EXIT_BLOCKED,
                evidence={
                    "reason": "extension_dataset_audit_not_passed",
                    "extension_status": ext_audit.get("status"),
                },
            )
        gap_audit = json.loads(gap_audit_path.read_text(encoding="utf-8"))
        must_run_flat = bool(
            gap_audit.get("must_run_flat_auxiliary", False)
        )
        flat_dir = (
            output_root / EXTENSION_ROOT / "flat_auxiliary" / "dataset"
        )
        flat_samples_path = flat_dir / "flat_auxiliary_sample_manifest.csv"
        flat_audit_path = flat_dir / "flat_auxiliary_dataset_audit.json"
        flat_frame = None
        flat_branches = None
        if must_run_flat:
            if not flat_samples_path.exists() or not flat_audit_path.exists():
                return StageResult(
                    stage,
                    "blocked",
                    EXIT_BLOCKED,
                    evidence={
                        "reason": (
                            "flat auxiliary is mandatory (event support "
                            "below 3) but its dataset or audit is missing"
                        ),
                        "flat_samples": str(flat_samples_path),
                        "flat_audit": str(flat_audit_path),
                    },
                )
            flat_audit = json.loads(
                flat_audit_path.read_text(encoding="utf-8")
            )
            if flat_audit.get("status") != "pass":
                return StageResult(
                    stage,
                    "blocked",
                    EXIT_BLOCKED,
                    evidence={
                        "reason": "flat_auxiliary_dataset_audit_not_passed",
                        "flat_status": flat_audit.get("status"),
                    },
                )
            flat_frame = pd.read_csv(flat_samples_path)
            flat_branch_path = (
                flat_dir / "flat_auxiliary_branch_manifest.csv"
            )
            if flat_branch_path.exists():
                flat_branches = pd.read_csv(flat_branch_path)
        branch_frames = []
        for path in (
            output_root / V1_BRANCHES_REL,
            ext_dir / "extension_branch_manifest.csv",
        ):
            if path.exists():
                branch_frames.append(pd.read_csv(path))
        if flat_branches is not None:
            branch_frames.append(flat_branches)
        try:
            result = build_pilot_dataset_v2(
                pd.read_csv(v1_samples_path),
                pd.read_csv(ext_samples_path),
                flat_frame,
                branch_frames=branch_frames,
            )
        except (KeyError, OSError, ValueError) as exc:
            return StageResult(
                stage, "blocked", EXIT_BLOCKED, evidence={"reason": str(exc)}
            )
        dataset_dir = output_root / "pilot" / "dataset_v2"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        result["sample_manifest"].to_csv(
            dataset_dir / "pilot_v2_sample_manifest.csv", index=False
        )
        result["branch_manifest"].to_csv(
            dataset_dir / "pilot_v2_branch_manifest.csv", index=False
        )
        result["split_manifest"].to_csv(
            dataset_dir / "pilot_v2_split_manifest.csv", index=False
        )
        result["source_manifest"].to_csv(
            dataset_dir / "pilot_v2_source_manifest.csv", index=False
        )
        result["actual_duplicates"].to_csv(
            dataset_dir / "pilot_v2_actual_duplicates.csv", index=False
        )
        accounting = result["accounting"]
        atomic_write_json(
            dataset_dir / "completion.json",
            {
                "stage": stage,
                "accounting": accounting,
                "flat_auxiliary_included": bool(flat_frame is not None),
                "must_run_flat_auxiliary": must_run_flat,
            },
        )
        complete = bool(accounting["accounting_closed"])
        return StageResult(
            stage,
            "pass" if complete else "blocked",
            EXIT_PASS if complete else EXIT_BLOCKED,
            completed=int(accounting["total_samples"]),
            batch_complete=True,
            scope_complete=complete,
            evidence={"accounting": accounting},
        )

    return handler


def _audit_dataset_v2_handler(
    output_root: Path,
) -> Callable[[RuntimeOptions], StageResult]:
    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "AuditPilotDatasetV2"
        dataset_dir = output_root / "pilot" / "dataset_v2"
        sample_path = dataset_dir / "pilot_v2_sample_manifest.csv"
        if not sample_path.exists():
            return _missing_inputs_result(stage, [str(sample_path)])
        samples = pd.read_csv(sample_path)
        audit = audit_pilot_dataset_v2(samples)
        atomic_write_json(
            dataset_dir / "pilot_v2_dataset_audit.json", audit
        )
        passed = audit["status"] == "pass"
        return StageResult(
            stage,
            audit["status"],
            EXIT_PASS if passed else EXIT_SCIENTIFIC_FAIL,
            completed=int(len(samples)),
            batch_complete=True,
            scope_complete=passed,
            evidence=audit,
        )

    return handler


def _train_baselines_v2_handler(
    output_root: Path,
) -> Callable[[RuntimeOptions], StageResult]:
    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "TrainPilotBaselinesV2"
        dataset_dir = output_root / "pilot" / "dataset_v2"
        sample_path = dataset_dir / "pilot_v2_sample_manifest.csv"
        if not sample_path.exists():
            return _missing_inputs_result(stage, [str(sample_path)])
        try:
            report = train_pilot_baselines_v2(pd.read_csv(sample_path))
        except (ImportError, KeyError, OSError, ValueError) as exc:
            return StageResult(
                stage, "blocked", EXIT_BLOCKED, evidence={"reason": str(exc)}
            )
        atomic_write_json(
            dataset_dir / "baseline_models_report_v2.json", report
        )
        return StageResult(
            stage,
            "pass",
            EXIT_PASS,
            completed=int(report["split_policy"]["train_rows"]),
            batch_complete=True,
            scope_complete=True,
            evidence={
                "models": report["models"],
                "split_policy": report["split_policy"],
            },
        )

    return handler


def _evaluate_gate_v2_handler(
    output_root: Path,
) -> Callable[[RuntimeOptions], StageResult]:
    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "EvaluatePilotGateV2"
        dataset_dir = output_root / "pilot" / "dataset_v2"
        audit_path = dataset_dir / "pilot_v2_dataset_audit.json"
        report_path = dataset_dir / "baseline_models_report_v2.json"
        missing = [
            str(path)
            for path in (audit_path, report_path)
            if not path.exists()
        ]
        if missing:
            return _missing_inputs_result(stage, missing)
        verdict = evaluate_pilot_gate_v2(
            json.loads(audit_path.read_text(encoding="utf-8")),
            json.loads(report_path.read_text(encoding="utf-8")),
        )
        target = (
            output_root / "pilot" / "evaluation"
            / "pilot_gate_v2_verdict.json"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(target, verdict)
        passed = bool(verdict["scientific_pass"])
        return StageResult(
            stage,
            verdict["status"],
            EXIT_PASS if passed else EXIT_SCIENTIFIC_FAIL,
            completed=1 if passed else 0,
            batch_complete=True,
            scope_complete=passed,
            evidence=verdict,
        )

    return handler


def build_extension_handlers(
    *, project_root: Path, output_root: Path, config: dict
) -> dict[str, Callable[[RuntimeOptions], StageResult]]:
    """All extension/v2 stage handlers keyed by stage name.

    Run and partial handlers reuse the parameterized Pilot machinery so the
    v1 reference cache under ``pilot/references`` is hit unchanged and every
    extension candidate costs exactly one new physical SWMM run.
    """
    from .pipeline import (
        _audit_pilot_partial_handler,
        _build_pilot_partial_handler,
        _run_pilot400_handler,
    )

    project = Path(project_root)
    output = Path(output_root)
    return {
        "AuditPilotCoverageGaps": _audit_coverage_gaps_handler(
            project, output, config
        ),
        "PlanPilotCoverageExtension": _plan_coverage_extension_handler(
            project, output, config
        ),
        "AuditPilotCoverageExtensionPlan": _audit_extension_plan_handler(
            output
        ),
        "RunPilotCoverageExtension": _run_pilot400_handler(
            output,
            config,
            stage="RunPilotCoverageExtension",
            branch_plan_rel=EXTENSION_BRANCH_PLAN,
        ),
        "BuildPilotCoverageExtensionPartial": _build_pilot_partial_handler(
            "BuildPilotCoverageExtensionPartial",
            project,
            output,
            config,
            run_stage="RunPilotCoverageExtension",
            branch_plan_rel=EXTENSION_BRANCH_PLAN,
        ),
        "AuditPilotCoverageExtensionPartial": _audit_pilot_partial_handler(
            "AuditPilotCoverageExtensionPartial",
            project,
            output,
            config,
            run_stage="RunPilotCoverageExtension",
            branch_plan_rel=EXTENSION_BRANCH_PLAN,
        ),
        "BuildPilotCoverageExtensionDataset": (
            _build_extension_dataset_handler(
                "BuildPilotCoverageExtensionDataset",
                project,
                output,
                config,
                run_stage="RunPilotCoverageExtension",
                branch_plan_rel=EXTENSION_BRANCH_PLAN,
                dataset_dir_rel=f"{EXTENSION_ROOT}/dataset",
                prefix="extension",
            )
        ),
        "AuditPilotCoverageExtensionDataset": (
            _audit_extension_dataset_handler(
                "AuditPilotCoverageExtensionDataset",
                output,
                dataset_dir_rel=f"{EXTENSION_ROOT}/dataset",
                prefix="extension",
                audit_name="extension_dataset_audit.json",
                expected_source_phase=JOINT_EXTENSION_PHASE,
            )
        ),
        "PlanPilotFlatAuxiliary": _plan_flat_auxiliary_handler(
            project, output, config
        ),
        "RunPilotFlatAuxiliary": _run_pilot400_handler(
            output,
            config,
            stage="RunPilotFlatAuxiliary",
            branch_plan_rel=FLAT_AUX_BRANCH_PLAN,
        ),
        "BuildPilotFlatAuxiliaryDataset": (
            _build_extension_dataset_handler(
                "BuildPilotFlatAuxiliaryDataset",
                project,
                output,
                config,
                run_stage="RunPilotFlatAuxiliary",
                branch_plan_rel=FLAT_AUX_BRANCH_PLAN,
                dataset_dir_rel=f"{EXTENSION_ROOT}/flat_auxiliary/dataset",
                prefix="flat_auxiliary",
            )
        ),
        "AuditPilotFlatAuxiliaryDataset": (
            _audit_extension_dataset_handler(
                "AuditPilotFlatAuxiliaryDataset",
                output,
                dataset_dir_rel=f"{EXTENSION_ROOT}/flat_auxiliary/dataset",
                prefix="flat_auxiliary",
                audit_name="flat_auxiliary_dataset_audit.json",
                expected_source_phase=FLAT_AUXILIARY_PHASE,
            )
        ),
        "BuildPilotDatasetV2": _build_dataset_v2_handler(output),
        "AuditPilotDatasetV2": _audit_dataset_v2_handler(output),
        "TrainPilotBaselinesV2": _train_baselines_v2_handler(output),
        "EvaluatePilotGateV2": _evaluate_gate_v2_handler(output),
    }
