"""Gate P3 stage handlers (feasibility mapping chain).

Wired lazily from ``pipeline.build_registry`` in the same fashion as
``pipeline_ext``.  All P3 stages write only under
``pilot_feasibility_p3/``; frozen Pilot v1/v2 evidence and legacy oracle
outputs are read-only inputs.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import pandas as pd

from .legacy_oracle import (
    KIND_GATE0_PROOF,
    KIND_ORACLE_CASE,
    KIND_V4_MANIFEST,
    audit_legacy_oracle_compatibility,
)
from .partial_audit import HARD_AUTHENTICITY_COLUMNS
from .pilot_candidates import build_pilot_branch_plan
from .pilot_feasibility import (
    audit_feasibility_state_catalog,
    build_feasibility_state_catalog,
)
from .pilot_feasibility_map import (
    audit_feasibility_map,
    build_best_candidates,
    classify_feasibility_states,
    combine_state_samples,
    plan_feasibility_round_b_directives,
)
from .pilot_feasibility_search import (
    ROUND_B,
    audit_feasibility_plan,
    plan_pilot_feasibility_map,
)
from .pilot_reducers import build_pilot_dataset
from .pilot_run import expand_pilot_completions
from .pilot_v3 import (
    audit_pilot_dataset_v3,
    build_pilot_dataset_v3,
    evaluate_pilot_gate_v3,
    train_pilot_baselines_v3,
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

P3_ROOT = "pilot_feasibility_p3"
LEGACY_DIR_REL = f"{P3_ROOT}/legacy_oracle"
REPLAY_PLAN_REL = f"{LEGACY_DIR_REL}/legacy_oracle_replay_plan.csv"
CATALOG_REL = f"{P3_ROOT}/pilot_feasibility_state_catalog.csv"
PLANNING_DIR_REL = f"{P3_ROOT}/planning"
FEAS_PLAN_REL = f"{PLANNING_DIR_REL}/feasibility_candidate_plan.csv"
FEAS_BRANCH_PLAN_REL = f"{PLANNING_DIR_REL}/feasibility_branch_plan.csv"
MAP_DIR_REL = f"{P3_ROOT}/map"
DATASET_DIR_REL = f"{P3_ROOT}/dataset"
DATASET_V3_DIR_REL = f"{P3_ROOT}/dataset_v3"
EVALUATION_V3_DIR_REL = f"{P3_ROOT}/evaluation"
MAP_AUDIT_REL = f"{MAP_DIR_REL}/pilot_feasibility_audit.json"
P3_CONTRACT_REL = "docs/contracts/PROJECT6_V4_PILOT_FEASIBILITY_GATE_P3.json"
V2_SAMPLES_REL = "pilot/dataset_v2/pilot_v2_sample_manifest.csv"
V1_PLAN_REL = "pilot/planning/pilot_candidate_plan.csv"

# Legacy evidence sources relative to the shared v4 output parent
# (outputs/project6_dual_reference_v4), which holds final_v4 as a sibling.
LEGACY_SOURCES = (
    ("oracle_pareto_20ev", KIND_ORACLE_CASE, "oracle_pareto_20ev/oracle_case_results.csv"),
    (
        "oracle_pareto_smoke1_fix",
        KIND_ORACLE_CASE,
        "oracle_pareto_smoke1_fix/oracle_case_results.csv",
    ),
    (
        "constraint_ablation",
        KIND_ORACLE_CASE,
        "oracle_bottleneck_diagnosis/constraint_ablation_results.csv",
    ),
    (
        "gate0_feasible_candidate_proof",
        KIND_GATE0_PROOF,
        "oracle_bottleneck_diagnosis/gate0_feasible_candidate_proof.csv",
    ),
    (
        "action_effect_dataset_v4",
        KIND_V4_MANIFEST,
        "action_effect_dataset_v4/v4_dataset_manifest.csv",
    ),
)

V1_SAMPLES_REL = "pilot/dataset/pilot_sample_manifest.csv"
EVENT_INVENTORY_REL = "inventory/event_inventory.csv"


def _missing_inputs_result(stage: str, missing: list[str]) -> StageResult:
    return StageResult(
        stage,
        "incomplete",
        EXIT_INCOMPLETE,
        remaining=1,
        evidence={"missing_inputs": missing},
    )


def _audit_legacy_oracle_handler(
    project_root: Path, output_root: Path, config: dict
) -> Callable[[RuntimeOptions], StageResult]:
    """Read-only 13-dimension compatibility scan of all legacy evidence."""

    def handler(_options: RuntimeOptions) -> StageResult:
        from .pipeline import _file_sha256

        stage = "AuditLegacyOracleCompatibility"
        legacy_root = output_root.parent
        inventory_path = output_root / EVENT_INVENTORY_REL
        samples_path = output_root / V1_SAMPLES_REL
        network_rel = str(
            config.get("project", {}).get(
                "network", "data/wuhan_v8_storage_retrofit.inp"
            )
        )
        network_path = project_root / network_rel
        missing = [
            str(path)
            for path in (inventory_path, samples_path, network_path)
            if not path.exists()
        ]
        if missing:
            return _missing_inputs_result(stage, missing)
        try:
            inventory = pd.read_csv(inventory_path)
            samples = pd.read_csv(samples_path)
            rainfall_by_event = {
                str(row["event_id"]): str(row["rainfall_sha256"])
                for _, row in inventory.iterrows()
                if str(row.get("rainfall_sha256", "")).strip()
            }
            pilot_state_keys = {
                (str(event), str(checkpoint))
                for event, checkpoint in samples[
                    ["event_id", "checkpoint_id"]
                ].itertuples(index=False)
            }
            sources: dict[str, tuple[str, pd.DataFrame, Path]] = {}
            missing_sources: list[str] = []
            for name, kind, rel in LEGACY_SOURCES:
                path = legacy_root / rel
                if not path.exists():
                    missing_sources.append(rel)
                    continue
                sources[name] = (kind, pd.read_csv(path), path.parent)
            if not sources:
                return StageResult(
                    stage,
                    "blocked",
                    EXIT_BLOCKED,
                    evidence={"reason": "no legacy evidence sources found"},
                )
            result = audit_legacy_oracle_compatibility(
                sources,
                network_sha256=_file_sha256(network_path),
                rainfall_by_event=rainfall_by_event,
                pilot_state_keys=pilot_state_keys,
                legacy_root=legacy_root,
                missing_sources=missing_sources,
            )
        except (KeyError, OSError, ValueError) as exc:
            return StageResult(
                stage, "blocked", EXIT_BLOCKED, evidence={"reason": str(exc)}
            )
        target = output_root / LEGACY_DIR_REL
        target.mkdir(parents=True, exist_ok=True)
        result["compatible"].to_csv(
            target / "legacy_oracle_compatible.csv", index=False
        )
        result["incompatible"].to_csv(
            target / "legacy_oracle_incompatible.csv", index=False
        )
        result["replay_plan"].to_csv(
            target / "legacy_oracle_replay_plan.csv", index=False
        )
        audit = result["audit"]
        atomic_write_json(
            target / "legacy_oracle_compatibility_audit.json", audit
        )
        atomic_write_json(
            target / "completion.json",
            {"stage": stage, "read_only": True, "audit": audit},
        )
        passed = audit["status"] == "pass"
        return StageResult(
            stage,
            audit["status"],
            EXIT_PASS if passed else EXIT_BLOCKED,
            completed=int(audit["evidence_rows"]),
            batch_complete=True,
            scope_complete=passed,
            evidence=audit,
        )

    return handler


def build_p3_handlers(
    *, project_root: Path, output_root: Path, config: dict
) -> dict[str, Callable[[RuntimeOptions], StageResult]]:
    """All Gate P3 stage handlers keyed by stage name.

    Run and partial stages reuse the parameterized Pilot machinery so the
    frozen v1 reference cache under ``pilot/references`` is hit unchanged
    and every feasibility candidate costs exactly one new SWMM run.
    """
    from .pipeline import (
        _audit_pilot_partial_handler,
        _build_pilot_partial_handler,
        _run_pilot400_handler,
    )

    project = Path(project_root)
    output = Path(output_root)
    return {
        "AuditLegacyOracleCompatibility": _audit_legacy_oracle_handler(
            project, output, config
        ),
        "PlanPilotFeasibilityMap": _plan_feasibility_map_handler(
            project, output, config
        ),
        "RunPilotFeasibilityMap": _run_pilot400_handler(
            output,
            config,
            stage="RunPilotFeasibilityMap",
            branch_plan_rel=FEAS_BRANCH_PLAN_REL,
        ),
        "BuildPilotFeasibilityPartial": _build_pilot_partial_handler(
            "BuildPilotFeasibilityPartial",
            project,
            output,
            config,
            run_stage="RunPilotFeasibilityMap",
            branch_plan_rel=FEAS_BRANCH_PLAN_REL,
        ),
        "AuditPilotFeasibilityPartial": _audit_pilot_partial_handler(
            "AuditPilotFeasibilityPartial",
            project,
            output,
            config,
            run_stage="RunPilotFeasibilityMap",
            branch_plan_rel=FEAS_BRANCH_PLAN_REL,
        ),
        "BuildPilotFeasibilityMap": _build_feasibility_map_handler(
            project, output, config
        ),
        "AuditPilotFeasibilityMap": _audit_feasibility_map_handler(
            project, output, config
        ),
        "BuildPilotDatasetV3": _build_dataset_v3_handler(output),
        "AuditPilotDatasetV3": _audit_dataset_v3_handler(output),
        "TrainPilotBaselinesV3": _train_baselines_v3_handler(output, config),
        "EvaluatePilotGateV3": _evaluate_gate_v3_handler(output),
    }


def _boundary_band(project_root: Path) -> dict:
    """Frozen boundary band from the immutable P3 contract, fail-closed."""
    contract_path = project_root / P3_CONTRACT_REL
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    return contract["frozen_thresholds"]["boundary_band"]


def _plan_feasibility_map_handler(
    project_root: Path, output_root: Path, config: dict
) -> Callable[[RuntimeOptions], StageResult]:
    """Catalog + Round A (and Round B when directed) plan, fail-closed."""

    def handler(_options: RuntimeOptions) -> StageResult:
        from .pipeline_ext import _facility_inputs

        stage = "PlanPilotFeasibilityMap"
        v2_path = output_root / V2_SAMPLES_REL
        v1_plan_path = output_root / V1_PLAN_REL
        replay_path = output_root / REPLAY_PLAN_REL
        contract_path = project_root / P3_CONTRACT_REL
        missing = [
            str(path)
            for path in (v2_path, v1_plan_path, replay_path, contract_path)
            if not path.exists()
        ]
        if missing:
            return _missing_inputs_result(stage, missing)
        planning_dir = output_root / PLANNING_DIR_REL
        planning_dir.mkdir(parents=True, exist_ok=True)
        try:
            inputs = _facility_inputs(project_root, config)
            v2_samples = pd.read_csv(v2_path)
            v1_plan = pd.read_csv(v1_plan_path)
            catalog = build_feasibility_state_catalog(
                v2_samples, facility_ids=inputs["facility_ids"]
            )
            catalog_audit = audit_feasibility_state_catalog(catalog)
            if catalog_audit["status"] != "pass":
                return StageResult(
                    stage, "blocked", EXIT_BLOCKED, evidence=catalog_audit
                )
            directives_path = (
                output_root / MAP_DIR_REL / "round_b_directives.csv"
            )
            directives = (
                pd.read_csv(directives_path)
                if directives_path.exists()
                else None
            )
            result = plan_pilot_feasibility_map(
                catalog,
                v1_plan,
                v2_samples,
                pd.read_csv(replay_path),
                reference_root=output_root / "pilot",
                facility_ids=inputs["facility_ids"],
                facility_semantics=inputs["facility_semantics"],
                schedule_dir=planning_dir / "schedules",
                schedule_dir_relative_to=output_root,
                round_b_directives=directives,
            )
            candidate_plan = result["candidate_plan"]
            branch_plan = build_pilot_branch_plan(
                candidate_plan, contract_sha256=inputs["contract_sha"]
            )
            plan_audit = audit_feasibility_plan(
                candidate_plan, branch_plan, v1_plan, v2_samples, catalog
            )
        except (KeyError, OSError, ValueError) as exc:
            return StageResult(
                stage, "blocked", EXIT_BLOCKED, evidence={"reason": str(exc)}
            )
        catalog.to_csv(output_root / CATALOG_REL, index=False)
        atomic_write_json(
            output_root / P3_ROOT / "pilot_feasibility_catalog_audit.json",
            catalog_audit,
        )
        candidate_plan.to_csv(
            planning_dir / "feasibility_candidate_plan.csv", index=False
        )
        branch_plan.to_csv(
            planning_dir / "feasibility_branch_plan.csv", index=False
        )
        result["search_coverage"].to_csv(
            planning_dir / "feasibility_search_coverage_planned.csv",
            index=False,
        )
        atomic_write_json(
            planning_dir / "feasibility_plan_audit.json", plan_audit
        )
        if plan_audit["status"] != "pass":
            return StageResult(
                stage, "blocked", EXIT_BLOCKED, evidence=plan_audit
            )
        return StageResult(
            stage,
            "pass",
            EXIT_PASS,
            completed=int(len(candidate_plan)),
            batch_complete=True,
            scope_complete=True,
            evidence={
                "catalog_audit": catalog_audit,
                "plan_audit": plan_audit,
                "round_b_directives_applied": bool(
                    directives is not None and len(directives)
                ),
            },
        )

    return handler


def _state_keyed_counts(
    frames: list[pd.DataFrame], plan: pd.DataFrame
) -> dict[tuple[str, str], int]:
    """Per-state row counts for outcome frames keyed only by sample_id."""
    state_by_sample = {
        str(row["sample_id"]): (
            str(row["event_id"]),
            str(row["checkpoint_id"]),
        )
        for _, row in plan.iterrows()
    }
    counts: dict[tuple[str, str], int] = {}
    for frame in frames:
        if frame is None or frame.empty or "sample_id" not in frame:
            continue
        for sample_id in frame["sample_id"].astype(str):
            key = state_by_sample.get(sample_id)
            if key is not None:
                counts[key] = counts.get(key, 0) + 1
    return counts


def _round_b_directives_path(
    map_dir: Path, candidate_plan: pd.DataFrame
) -> Path:
    """Freeze ``round_b_directives.csv`` once Round B rows exist in the plan.

    The directives file is a planning input: re-building after Round B has
    been planned/executed must not overwrite the directives that produced
    the executed plan (Plan idempotency).  The recomputed residual is
    parked as a diagnostic file instead.
    """
    if "search_round" in candidate_plan and bool(
        (candidate_plan["search_round"].astype(str) == ROUND_B).any()
    ):
        return map_dir / "round_b_directives_residual_diagnostic.csv"
    return map_dir / "round_b_directives.csv"


def _build_feasibility_map_handler(
    project_root: Path, output_root: Path, config: dict
) -> Callable[[RuntimeOptions], StageResult]:
    """Dataset build + frozen-vocabulary state classification + Round B plan."""

    def handler(_options: RuntimeOptions) -> StageResult:
        from .pipeline import _add_pilot_audit_columns, _run_stage_sources
        from .pipeline_ext import _facility_inputs

        stage = "BuildPilotFeasibilityMap"
        plan_path, run_root = _run_stage_sources(
            "RunPilotFeasibilityMap", output_root
        )
        branch_path = output_root / FEAS_BRANCH_PLAN_REL
        catalog_path = output_root / CATALOG_REL
        v2_path = output_root / V2_SAMPLES_REL
        contract_path = project_root / P3_CONTRACT_REL
        missing = [
            str(path)
            for path in (
                plan_path,
                branch_path,
                catalog_path,
                v2_path,
                contract_path,
            )
            if not path.exists()
        ]
        if missing:
            return _missing_inputs_result(stage, missing)
        candidate_plan = pd.read_csv(plan_path)
        completions = (
            completion_manifest(run_root)
            if run_root.exists()
            else pd.DataFrame()
        )
        try:
            inputs = _facility_inputs(project_root, config)
            scientific_margin = config["thresholds"]["scientific_margin"]
            result = build_pilot_dataset(
                candidate_plan,
                pd.read_csv(branch_path),
                expand_pilot_completions(completions),
                priority_nodes=inputs["priority_nodes"],
                facility_ids=inputs["facility_ids"],
                scientific_margin=scientific_margin,
                dead_zone=config["thresholds"]["dead_zone"],
            )
            samples = _add_pilot_audit_columns(
                result["sample_manifest"], config
            )
            catalog = pd.read_csv(catalog_path)
            combined = combine_state_samples(pd.read_csv(v2_path), samples)
            boundary_band = _boundary_band(project_root)
            unaccounted = _state_keyed_counts(
                [result["pending"], result["missing_confirmed"]],
                candidate_plan,
            )
            map_frame = classify_feasibility_states(
                catalog,
                combined,
                scientific_margin=scientific_margin,
                boundary_band=boundary_band,
                unaccounted_by_state=unaccounted,
            )
            best = build_best_candidates(combined)
            directives = plan_feasibility_round_b_directives(
                catalog,
                combined,
                candidate_plan,
                scientific_margin=scientific_margin,
                boundary_band=boundary_band,
            )
        except (KeyError, OSError, ValueError) as exc:
            return StageResult(
                stage, "blocked", EXIT_BLOCKED, evidence={"reason": str(exc)}
            )
        dataset_dir = output_root / DATASET_DIR_REL
        dataset_dir.mkdir(parents=True, exist_ok=True)
        samples.to_csv(
            dataset_dir / "feasibility_sample_manifest.csv", index=False
        )
        for name in (
            "branch_manifest",
            "rejected",
            "actual_duplicates",
            "pending",
        ):
            result[name].to_csv(
                dataset_dir / f"feasibility_{name}.csv", index=False
            )
        result["missing_confirmed"].to_csv(
            dataset_dir / "feasibility_missing.csv", index=False
        )
        accounting = result["accounting"]
        atomic_write_json(
            dataset_dir / "completion.json",
            {
                "stage": stage,
                "accounting": accounting,
                "accepted": int(accounting["accepted"]),
            },
        )
        map_dir = output_root / MAP_DIR_REL
        map_dir.mkdir(parents=True, exist_ok=True)
        map_frame.to_csv(
            map_dir / "pilot_state_feasibility_map.csv", index=False
        )
        best.to_csv(
            map_dir / "pilot_state_best_candidates.csv", index=False
        )
        coverage_path = (
            output_root
            / PLANNING_DIR_REL
            / "feasibility_search_coverage_planned.csv"
        )
        accepted_counts = _state_keyed_counts([samples], candidate_plan)
        coverage = (
            pd.read_csv(coverage_path)
            if coverage_path.exists()
            else pd.DataFrame(columns=["event_id", "checkpoint_id"])
        )
        if len(coverage):
            coverage["accepted_samples"] = [
                accepted_counts.get(
                    (str(row["event_id"]), str(row["checkpoint_id"])), 0
                )
                for _, row in coverage.iterrows()
            ]
            coverage["unaccounted_rows"] = [
                unaccounted.get(
                    (str(row["event_id"]), str(row["checkpoint_id"])), 0
                )
                for _, row in coverage.iterrows()
            ]
        coverage.to_csv(
            map_dir / "pilot_state_search_coverage.csv", index=False
        )
        map_frame[
            map_frame["state_feasibility_class"] == "execution_unresolved"
        ].to_csv(map_dir / "pilot_state_unresolved.csv", index=False)
        directives_path = _round_b_directives_path(map_dir, candidate_plan)
        directives.to_csv(directives_path, index=False)
        complete = (
            bool(accounting["accounting_closed"])
            and int(accounting["missing"]) == 0
            and int(len(result["pending"])) == 0
        )
        return StageResult(
            stage,
            "pass" if complete else "incomplete",
            EXIT_PASS if complete else EXIT_INCOMPLETE,
            completed=int(accounting["accepted"]),
            remaining=int(accounting["missing"]) + int(len(result["pending"])),
            batch_complete=True,
            scope_complete=complete,
            evidence={
                "accounting": accounting,
                "class_counts": map_frame["state_feasibility_class"]
                .value_counts()
                .to_dict(),
                "round_b_directive_states": int(len(directives)),
                "round_b_directives_frozen": directives_path.name
                != "round_b_directives.csv",
            },
        )

    return handler


def _audit_feasibility_map_handler(
    project_root: Path, output_root: Path, config: dict
) -> Callable[[RuntimeOptions], StageResult]:
    """Mechanical hard gate over the built map (scientific gates stay in v3)."""

    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "AuditPilotFeasibilityMap"
        map_dir = output_root / MAP_DIR_REL
        dataset_dir = output_root / DATASET_DIR_REL
        map_path = map_dir / "pilot_state_feasibility_map.csv"
        samples_path = dataset_dir / "feasibility_sample_manifest.csv"
        completion_path = dataset_dir / "completion.json"
        duplicates_path = dataset_dir / "feasibility_actual_duplicates.csv"
        catalog_path = output_root / CATALOG_REL
        plan_path = output_root / FEAS_PLAN_REL
        v2_path = output_root / V2_SAMPLES_REL
        missing = [
            str(path)
            for path in (
                map_path,
                samples_path,
                completion_path,
                catalog_path,
                plan_path,
                v2_path,
            )
            if not path.exists()
        ]
        if missing:
            return _missing_inputs_result(stage, missing)
        try:
            completion = json.loads(
                completion_path.read_text(encoding="utf-8")
            )
            duplicates = (
                len(pd.read_csv(duplicates_path))
                if duplicates_path.exists()
                else 0
            )
            audit = audit_feasibility_map(
                pd.read_csv(map_path),
                combine_state_samples(
                    pd.read_csv(v2_path), pd.read_csv(samples_path)
                ),
                completion.get("accounting", {}),
                catalog=pd.read_csv(catalog_path),
                candidate_plan=pd.read_csv(plan_path),
                hard_columns=HARD_AUTHENTICITY_COLUMNS,
                actual_duplicates=duplicates,
            )
        except (KeyError, OSError, ValueError) as exc:
            return StageResult(
                stage, "blocked", EXIT_BLOCKED, evidence={"reason": str(exc)}
            )
        atomic_write_json(
            map_dir / "pilot_feasibility_audit.json", audit
        )
        passed = audit["status"] == "pass"
        return StageResult(
            stage,
            audit["status"],
            EXIT_PASS if passed else EXIT_BLOCKED,
            completed=int(audit["feasibility_samples"]),
            batch_complete=True,
            scope_complete=passed,
            evidence=audit,
        )

    return handler


def _build_dataset_v3_handler(
    output_root: Path,
) -> Callable[[RuntimeOptions], StageResult]:
    """Merge frozen v2 rows + feasibility rows into Dataset v3."""

    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "BuildPilotDatasetV3"
        v2_path = output_root / V2_SAMPLES_REL
        feas_path = (
            output_root / DATASET_DIR_REL / "feasibility_sample_manifest.csv"
        )
        map_path = (
            output_root / MAP_DIR_REL / "pilot_state_feasibility_map.csv"
        )
        map_audit_path = output_root / MAP_AUDIT_REL
        missing = [
            str(path)
            for path in (v2_path, feas_path, map_path, map_audit_path)
            if not path.exists()
        ]
        if missing:
            return _missing_inputs_result(stage, missing)
        map_audit = json.loads(map_audit_path.read_text(encoding="utf-8"))
        if map_audit.get("status") != "pass":
            return StageResult(
                stage,
                "blocked",
                EXIT_BLOCKED,
                evidence={
                    "reason": "feasibility_map_audit_not_passed",
                    "map_audit_status": map_audit.get("status"),
                },
            )
        try:
            result = build_pilot_dataset_v3(
                pd.read_csv(v2_path),
                pd.read_csv(feas_path),
                pd.read_csv(map_path),
            )
        except (KeyError, OSError, ValueError) as exc:
            return StageResult(
                stage, "blocked", EXIT_BLOCKED, evidence={"reason": str(exc)}
            )
        dataset_dir = output_root / DATASET_V3_DIR_REL
        dataset_dir.mkdir(parents=True, exist_ok=True)
        result["sample_manifest"].to_csv(
            dataset_dir / "pilot_v3_sample_manifest.csv", index=False
        )
        result["state_manifest"].to_csv(
            dataset_dir / "pilot_v3_state_manifest.csv", index=False
        )
        result["split_manifest"].to_csv(
            dataset_dir / "pilot_v3_split_manifest.csv", index=False
        )
        result["source_manifest"].to_csv(
            dataset_dir / "pilot_v3_source_manifest.csv", index=False
        )
        accounting = result["accounting"]
        atomic_write_json(
            dataset_dir / "completion.json",
            {"stage": stage, "accounting": accounting},
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


def _audit_dataset_v3_handler(
    output_root: Path,
) -> Callable[[RuntimeOptions], StageResult]:
    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "AuditPilotDatasetV3"
        dataset_dir = output_root / DATASET_V3_DIR_REL
        sample_path = dataset_dir / "pilot_v3_sample_manifest.csv"
        if not sample_path.exists():
            return _missing_inputs_result(stage, [str(sample_path)])
        samples = pd.read_csv(sample_path)
        audit = audit_pilot_dataset_v3(samples)
        atomic_write_json(
            dataset_dir / "pilot_v3_dataset_audit.json", audit
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


def _train_baselines_v3_handler(
    output_root: Path, config: dict
) -> Callable[[RuntimeOptions], StageResult]:
    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "TrainPilotBaselinesV3"
        dataset_dir = output_root / DATASET_V3_DIR_REL
        sample_path = dataset_dir / "pilot_v3_sample_manifest.csv"
        state_path = dataset_dir / "pilot_v3_state_manifest.csv"
        missing = [
            str(path)
            for path in (sample_path, state_path)
            if not path.exists()
        ]
        if missing:
            return _missing_inputs_result(stage, missing)
        tfv_margin = float(
            config.get("thresholds", {})
            .get("scientific_margin", {})
            .get("tfv_m3", 0.0)
        )
        try:
            report = train_pilot_baselines_v3(
                pd.read_csv(sample_path),
                pd.read_csv(state_path),
                tfv_margin=tfv_margin,
            )
        except (ImportError, KeyError, OSError, ValueError) as exc:
            return StageResult(
                stage, "blocked", EXIT_BLOCKED, evidence={"reason": str(exc)}
            )
        atomic_write_json(
            dataset_dir / "baseline_models_report_v3.json", report
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
                "state_level": {
                    key: report["state_level"].get(key)
                    for key in (
                        "labeled_states",
                        "train_states",
                        "train_label_distribution",
                        "unseen_state_evaluation",
                        "unseen_state_evaluation_reason",
                    )
                    if key in report["state_level"]
                },
            },
        )

    return handler


def _evaluate_gate_v3_handler(
    output_root: Path,
) -> Callable[[RuntimeOptions], StageResult]:
    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "EvaluatePilotGateV3"
        dataset_dir = output_root / DATASET_V3_DIR_REL
        audit_path = dataset_dir / "pilot_v3_dataset_audit.json"
        report_path = dataset_dir / "baseline_models_report_v3.json"
        map_audit_path = output_root / MAP_AUDIT_REL
        missing = [
            str(path)
            for path in (audit_path, report_path, map_audit_path)
            if not path.exists()
        ]
        if missing:
            return _missing_inputs_result(stage, missing)
        verdict = evaluate_pilot_gate_v3(
            json.loads(audit_path.read_text(encoding="utf-8")),
            json.loads(report_path.read_text(encoding="utf-8")),
            json.loads(map_audit_path.read_text(encoding="utf-8")),
        )
        target = (
            output_root
            / EVALUATION_V3_DIR_REL
            / "pilot_gate_v3_verdict.json"
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

