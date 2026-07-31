"""Project6 V4 Gate 5R fail-closed orchestration.

This entrypoint never treats file existence as scientific success.  Planning,
authoritative SWMM execution, dataset building, and science audits are separate
stages with stable exit codes:

0 pass, 2 contract/input blocked, 3 incomplete, 4 runtime error, 5 science fail.
"""
from __future__ import annotations

import argparse
import atexit
from datetime import datetime, timedelta
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sewerrtc.control.v4_action_authority import classify_action_authority
from sewerrtc.control.v4_candidate_generator import (
    CandidateContext,
    V4CandidateGenerator,
    project_candidate_schedule,
    project_frozen_anchor_schedule,
)
from sewerrtc.prompt3.gate5r_pipeline import (
    EXIT_BLOCKED,
    EXIT_INCOMPLETE,
    EXIT_PASS,
    EXIT_RUNTIME_ERROR,
    EXIT_SCIENTIFIC_FAIL,
    accounting_is_closed,
    audit_contract_values,
    audit_existing_gate5,
    action_authority_reference_name,
    build_event_inventory,
    build_formal_1600_plan,
    build_pilot_plan,
    branch_state_hashes,
    canary_gate_status,
    classify_candidate_result,
    confirmed_flat_fraction_is_in_range,
    gate_exit_code,
    hashes_match_across_branches,
    post_decision_readback_mask,
    rebuild_run_manifest_from_completions,
    reference_cache_is_ready,
    reference_cache_key,
    safe_repeat_noise_ranges,
    scan_existing_dynamic_internal,
    schedule_action_cost,
    select_pending_plan,
)
from sewerrtc.simulation.kpi_metrics import compute_window_kpis


STAGES = (
    "AuditContracts",
    "ReauditExistingGate5",
    "BuildEventInventory",
    "ScanOpportunities",
    "PlanExcitationCanary",
    "PlanExactPrefixTiny",
    "RunExactPrefixTiny",
    "AuditExactPrefixTiny",
    "RunExcitationCanary",
    "AuditExcitationCanary",
    "DiscoverExactAnchors",
    "AuditExactAnchors",
    "PlanPilot",
    "RunPilot",
    "BuildPilotDataset",
    "AuditPilotDataset",
    "TrainPilotBaselines",
    "EvaluatePilotGate",
    "PlanFormal1600",
    "RunFormal1600",
    "BuildFormal1600",
    "AuditFormal1600",
    "TrainV4Informative",
    "EvaluateV4InformativeGate",
)


class Gate5RLockError(RuntimeError):
    pass


def _read_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _acquire_writer_lock(output_root: Path, stage: str) -> Path:
    """Prevent two Gate 5R processes from writing the same output root."""
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".writer.lock"
    if lock_path.exists():
        live_pid: int | None = None
        try:
            existing = _read_json(lock_path)
            pid = int(existing.get("pid", -1))
            if pid > 0:
                os.kill(pid, 0)
                live_pid = pid
        except ProcessLookupError:
            pass
        except PermissionError:
            raise Gate5RLockError(
                "Gate 5R writer lock owner cannot be inspected"
            )
        except (ValueError, TypeError, json.JSONDecodeError):
            live_pid = None
        if live_pid is not None:
            raise Gate5RLockError(
                f"Gate 5R output is already locked by live PID {live_pid}"
            )
        if lock_path.exists():
            stale_dir = output_root / "stale_locks"
            stale_dir.mkdir(parents=True, exist_ok=True)
            os.replace(
                lock_path,
                stale_dir / f"writer_{int(time.time())}.lock",
            )
    descriptor = os.open(
        lock_path,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY,
    )
    try:
        os.write(
            descriptor,
            json.dumps(
                {
                    "pid": os.getpid(),
                    "stage": stage,
                    "created_at_epoch": time.time(),
                }
            ).encode("utf-8"),
        )
    finally:
        os.close(descriptor)

    def release() -> None:
        try:
            if lock_path.exists():
                current = _read_json(lock_path)
                if int(current.get("pid", -1)) == os.getpid():
                    lock_path.unlink()
        except (OSError, ValueError, TypeError):
            pass

    atexit.register(release)
    return lock_path


def _resolve(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _read_ids(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_context(config_path: Path) -> dict:
    config = _read_yaml(config_path)
    root = _resolve(PROJECT_ROOT, config.get("project", {}).get("root", PROJECT_ROOT))
    output_root = _resolve(root, config["gate5r"]["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    return {"config": config, "root": root, "output_root": output_root}


def _status(output_root: Path, stage: str, status: str, **evidence: Any) -> int:
    payload = {
        "stage": stage,
        "status": status,
        "exit_code": gate_exit_code(status),
        "created_at_epoch": time.time(),
        **evidence,
    }
    _atomic_json(output_root / "stage_status" / f"{stage}.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return int(payload["exit_code"])


def _collect_revealed_events(root: Path) -> tuple[set[str], list[str]]:
    revealed: set[str] = set()
    evidence: list[str] = []
    roots = [
        root / "outputs" / "project6_dual_reference_v4" / "oracle_pareto_20ev",
        root / "outputs" / "project6_dual_reference_v4" / "oracle_bottleneck_diagnosis",
        root
        / "outputs"
        / "project6_pfvfirst_dualfallback_10min_v3"
        / "formal_evaluation",
        root
        / "outputs"
        / "project6_pfvfirst_dualfallback_10min_v3_1"
        / "formal_evaluation",
        root
        / "outputs"
        / "project6_pfvfirst_dualfallback_10min_v3_3"
        / "formal_evaluation",
    ]
    for evidence_root in roots:
        if not evidence_root.exists():
            continue
        # Only top-level event/result manifests are provenance evidence.
        # Recursing into ``events`` and ``ablation_schedules`` would scan
        # thousands of large hydraulic trajectory tables.
        for csv_path in evidence_root.glob("*.csv"):
            if (
                evidence_root.name == "formal_evaluation"
                and not csv_path.name.endswith("event_policy_results.csv")
            ):
                continue
            try:
                header = pd.read_csv(csv_path, nrows=0).columns
                event_column = next(
                    (
                        column
                        for column in ("event_id", "event", "canonical_event_id")
                        if column in header
                    ),
                    None,
                )
                if event_column is None:
                    continue
                values = (
                    pd.read_csv(csv_path, usecols=[event_column])[event_column]
                    .dropna()
                    .astype(str)
                )
                revealed.update(values.tolist())
                evidence.append(str(csv_path))
            except Exception:
                continue
    return revealed, evidence


def stage_audit_contracts(context: dict) -> int:
    config, root, output_root = (
        context["config"],
        context["root"],
        context["output_root"],
    )
    project = config["project"]
    facility_ids = _read_ids(_resolve(root, project["canonical_ids"]))
    recovery = _read_json(_resolve(root, project["recovery_contract"]))
    dataset = _read_json(_resolve(root, project["dataset_contract"]))
    v4 = _read_yaml(_resolve(root, project["v4_config"]))
    audit = audit_contract_values(recovery, dataset, v4, facility_ids)
    from sewerrtc.contracts.swmm_control_parser import parse_swmm_controls

    native_actions = parse_swmm_controls(_resolve(root, project["network"]))[
        "actions"
    ]
    managed_ids = set(facility_ids)
    managed_native_targets = {
        str(action["actuator_id"])
        for action in native_actions
        if str(action["actuator_id"]) in managed_ids
    }
    background_native_targets = {
        str(action["actuator_id"])
        for action in native_actions
        if str(action["actuator_id"]) not in managed_ids
    }
    audit["checks"]["canonical_order_matches_recovery_contract"] = (
        facility_ids == recovery.get("engineering36_canonical_order", [])
    )
    audit["checks"]["managed_native_targets_present"] = bool(
        managed_native_targets
    )
    audit["checks"]["native_background_target_count_54"] = (
        len(background_native_targets) == 54
    )
    active_network = _resolve(root, project["network"])
    active_network_hash = _sha256(active_network)
    audit["checks"]["active_network_matches_contract_hash"] = (
        active_network_hash.lower()
        == str(recovery.get("network_sha256", "")).lower()
    )
    base_contract = recovery.get("base_inflow_contract", {})
    original_backup = Path(
        str(base_contract.get("original_network_backup", ""))
    )
    audit["checks"]["base_inflow_disabled"] = (
        base_contract.get("enabled") is False
        and base_contract.get("inp_section_present") is False
        and base_contract.get("pattern_section_present") is False
    )
    audit["checks"]["original_network_backup_preserved"] = (
        original_backup.exists()
        and _sha256(original_backup).lower()
        == str(base_contract.get("original_network_sha256", "")).lower()
    )
    audit["control_partition"] = {
        "native_action_count": int(len(native_actions)),
        "managed_native_target_count": int(len(managed_native_targets)),
        "native_background_target_count": int(len(background_native_targets)),
    }
    audit["failed_checks"] = [
        name for name, passed in audit["checks"].items() if not passed
    ]
    audit["status"] = "pass" if not audit["failed_checks"] else "blocked"
    immutable = {
        "network_sha256": active_network_hash,
        "canonical_ids_sha256": _sha256(_resolve(root, project["canonical_ids"])),
        "recovery_contract_sha256": _sha256(
            _resolve(root, project["recovery_contract"])
        ),
        "dataset_contract_sha256": _sha256(
            _resolve(root, project["dataset_contract"])
        ),
        "gate5r_config_sha256": _sha256(Path(context["config_path"])),
    }
    audit["immutable_hashes"] = immutable
    _atomic_json(output_root / "contract_audit.json", audit)
    return _status(
        output_root,
        "AuditContracts",
        audit["status"],
        failed_checks=audit["failed_checks"],
        evidence=str(output_root / "contract_audit.json"),
    )


def stage_reaudit_existing(context: dict) -> int:
    audit = audit_existing_gate5(context["root"], context["output_root"] / "reaudit")
    return _status(
        context["output_root"],
        "ReauditExistingGate5",
        audit["status"],
        legacy_evidence_status=audit.get("legacy_evidence_status"),
        candidates=audit.get("n_candidates", 0),
        evidence=str(context["output_root"] / "reaudit"),
    )


def stage_build_inventory(context: dict) -> int:
    config, root, output_root = (
        context["config"],
        context["root"],
        context["output_root"],
    )
    revealed, evidence = _collect_revealed_events(root)
    inventory = build_event_inventory(
        _resolve(root, config["project"]["event_catalog"]), revealed
    )
    descriptor_rows: list[dict] = []
    for index, row in inventory.iterrows():
        rainfall_path = Path(str(row.get("rainfall_path", "")))
        descriptor = {
            "duration_min": row.get("duration_min", np.nan),
            "total_depth": row.get("total_depth", np.nan),
            "peak_intensity": row.get("peak_intensity", np.nan),
            "peak_time": row.get("peak_time", np.nan),
            "peak_timing_ratio": np.nan,
        }
        if rainfall_path.exists():
            try:
                rain = pd.read_csv(rainfall_path)
                time_column = (
                    "elapsed_min"
                    if "elapsed_min" in rain.columns
                    else ("minute" if "minute" in rain.columns else "")
                )
                intensity_column = (
                    "intensity_mm_h"
                    if "intensity_mm_h" in rain.columns
                    else ("rainfall_mm_h" if "rainfall_mm_h" in rain.columns else "")
                )
                if time_column and intensity_column and len(rain):
                    elapsed = pd.to_numeric(
                        rain[time_column], errors="coerce"
                    ).fillna(0.0)
                    intensity = pd.to_numeric(
                        rain[intensity_column], errors="coerce"
                    ).fillna(0.0)
                    duration = float(elapsed.max())
                    diffs = np.diff(np.sort(elapsed.unique()))
                    step_min = float(np.median(diffs[diffs > 0])) if np.any(diffs > 0) else 5.0
                    peak_index = int(np.argmax(intensity.to_numpy(float)))
                    peak_time = float(elapsed.iloc[peak_index])
                    descriptor.update(
                        {
                            "duration_min": duration,
                            "total_depth": float(
                                intensity.sum() * step_min / 60.0
                            ),
                            "peak_intensity": float(intensity.max()),
                            "peak_time": peak_time,
                            "peak_timing_ratio": (
                                peak_time / duration if duration > 0 else 0.0
                            ),
                        }
                    )
            except (OSError, ValueError, KeyError):
                pass
        descriptor_rows.append(descriptor)
    descriptors = pd.DataFrame(descriptor_rows, index=inventory.index)
    for column in descriptors.columns:
        inventory[column] = descriptors[column]
    inventory_path = output_root / "inventory" / "event_inventory.csv"
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory.to_csv(inventory_path, index=False)
    pd.DataFrame({"event_id": sorted(revealed)}).to_csv(
        inventory_path.parent / "revealed_event_blacklist.csv", index=False
    )
    eligible = int(inventory["eligible"].sum())
    unique_series = int(
        inventory.loc[inventory["eligible"], "rainfall_series_sha256"].nunique()
    )
    audit = {
        "status": "pass" if eligible >= 80 and unique_series >= 80 else "blocked",
        "inventory_rows": int(len(inventory)),
        "eligible_events": eligible,
        "eligible_unique_rainfall_series": unique_series,
        "revealed_events": len(revealed),
        "revealed_evidence_files": evidence,
    }
    _atomic_json(inventory_path.parent / "event_inventory_audit.json", audit)
    return _status(
        output_root,
        "BuildEventInventory",
        audit["status"],
        **{key: value for key, value in audit.items() if key != "status"},
    )


def _maximin_event_subset(
    events: pd.DataFrame, count: int, seed: int
) -> pd.DataFrame:
    """Deterministically cover rainfall duration, depth, intensity and timing."""
    if len(events) <= int(count):
        return events.copy()
    columns = [
        "duration_min",
        "total_depth",
        "peak_intensity",
        "peak_timing_ratio",
    ]
    matrix = (
        events[columns]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(events[columns].apply(pd.to_numeric, errors="coerce").median())
        .fillna(0.0)
        .to_numpy(float)
    )
    scale = np.std(matrix, axis=0)
    scale[scale <= 1e-12] = 1.0
    matrix = (matrix - np.mean(matrix, axis=0)) / scale
    rng = np.random.default_rng(int(seed))
    first_candidates = np.flatnonzero(
        matrix[:, 1] == np.max(matrix[:, 1])
    )
    selected = [int(rng.choice(first_candidates))]
    while len(selected) < int(count):
        distances = np.min(
            np.linalg.norm(
                matrix[:, None, :] - matrix[np.asarray(selected)][None, :, :],
                axis=2,
            ),
            axis=1,
        )
        distances[np.asarray(selected)] = -np.inf
        selected.append(int(np.argmax(distances)))
    result = events.iloc[selected].copy()
    result["diversity_selection_rank"] = np.arange(1, len(result) + 1)
    return result


def _existing_internal_details(root: Path) -> list[Path]:
    trajectory_roots = [
        root
        / "outputs"
        / "project6_pfvfirst_dualfallback_10min_v3"
        / "gat"
        / "independent_holdout"
        / "generated_trajectories"
        / "trajectories",
        root
        / "outputs"
        / "project6_pfvfirst_dualfallback_10min_v3"
        / "baselines"
        / "trajectories",
        root
        / "outputs"
        / "project6_dual_reference_v4"
        / "gate5r_informative_v2_no_dwf"
        / "opportunity"
        / "baseline_runs",
    ]
    paths: list[Path] = []
    for trajectory_root in trajectory_roots:
        if trajectory_root.exists():
            paths.extend(sorted(trajectory_root.glob("*__internal_rules_detail.csv")))
            paths.extend(
                sorted(
                    trajectory_root.glob(
                        "*/dynamic_internal_opportunity_detail.csv"
                    )
                )
            )
    return paths


def stage_scan_opportunities(
    context: dict, workers: int = 4, resume: bool = False
) -> int:
    config, root, output_root = (
        context["config"],
        context["root"],
        context["output_root"],
    )
    inventory_path = output_root / "inventory" / "event_inventory.csv"
    if not inventory_path.exists():
        return _status(
            output_root,
            "ScanOpportunities",
            "incomplete",
            reason="BuildEventInventory must pass first",
        )
    inventory = pd.read_csv(inventory_path)
    eligible = set(
        inventory.loc[inventory["eligible"].astype(bool), "event_id"].astype(str)
    )
    facility_ids = _read_ids(_resolve(root, config["project"]["canonical_ids"]))
    semantics = pd.read_csv(
        _resolve(root, config["project"]["facility_semantics"])
    )
    from sewerrtc.simulation.action_policies import attach_reference_nodes

    semantics = attach_reference_nodes(
        semantics.assign(
            actuator_id=semantics.get("actuator_id", semantics["facility_id"])
        ),
        _resolve(root, config["project"]["network"]),
    )
    facility_nodes = {
        str(row["facility_id"]): (
            str(row.get("from_node", "")),
            str(row.get("to_node", "")),
        )
        for _, row in semantics.iterrows()
    }

    def source_event_id(path: Path) -> str:
        try:
            row = pd.read_csv(path, usecols=["event_id"], nrows=1)
            if len(row):
                return str(row["event_id"].iloc[0])
        except (OSError, ValueError):
            pass
        return path.name.split("__internal_rules_detail.csv")[0]

    paths = [
        path
        for path in _existing_internal_details(root)
        if source_event_id(path) in eligible
    ]
    event_limit = int(config["opportunity"]["scan_development_events"])
    paths = paths[:event_limit]
    generated_manifest_path = (
        output_root / "opportunity" / "baseline_generation_manifest.csv"
    )
    if len(paths) < event_limit:
        selected = inventory[
            inventory["eligible"].astype(bool)
            & inventory["rainfall_path"].astype(str).map(lambda value: Path(value).exists())
        ].drop_duplicates("event_id")
        selected = _maximin_event_subset(
            selected,
            event_limit,
            int(config["runtime"]["seed"]),
        )
        node_columns = [
            column
            for column in (
                "storage_id",
                "reference_node",
                "from_node",
                "to_node",
            )
            if column in semantics.columns
        ]
        record_nodes = set(
            _read_ids(_resolve(root, config["project"]["priority_nodes"]))
        )
        for column in node_columns:
            record_nodes.update(
                value
                for value in semantics[column].dropna().astype(str)
                if value and value.lower() != "nan"
            )
        payloads = []
        spinup = int(config["runtime"]["spinup_min"])
        baseline_root = output_root / "opportunity" / "baseline_runs"
        for _, event in selected.iterrows():
            rainfall = Path(str(event["rainfall_path"]))
            payloads.append(
                {
                    "event_id": str(event["event_id"]),
                    "network": str(_resolve(root, config["project"]["network"])),
                    "rainfall_path": str(rainfall),
                    "rain_duration_min": _rainfall_duration_min(rainfall),
                    "spinup_min": spinup,
                    "facility_semantics": str(
                        _resolve(root, config["project"]["facility_semantics"])
                    ),
                    "priority_nodes": _read_ids(
                        _resolve(root, config["project"]["priority_nodes"])
                    ),
                    "record_node_ids": sorted(record_nodes),
                    "out_dir": str(baseline_root / str(event["event_id"])),
                    "resume": bool(resume),
                }
            )
        generated: list[dict] = []
        if workers <= 1:
            generated = [_run_opportunity_worker(payload) for payload in payloads]
        else:
            with ProcessPoolExecutor(max_workers=min(int(workers), 16)) as executor:
                futures = [
                    executor.submit(_run_opportunity_worker, payload)
                    for payload in payloads
                ]
                for future in as_completed(futures):
                    generated.append(future.result())
        generated_frame = pd.DataFrame(generated)
        generated_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        generated_frame.to_csv(generated_manifest_path, index=False)
        paths = [
            Path(value)
            for value in generated_frame.loc[
                generated_frame["status"].eq("accepted"), "detail_path"
            ]
        ]
    opportunities = scan_existing_dynamic_internal(
        paths, facility_ids, facility_nodes=facility_nodes
    )
    if not opportunities.empty:
        # Only decision instants with complete 60-min history and H120 future.
        spinup_min = int(config["runtime"]["spinup_min"])
        max_by_event = opportunities.groupby("event_id")["elapsed_min"].transform(
            "max"
        )
        opportunities = opportunities[
            (opportunities["elapsed_min"] >= spinup_min + 60.0)
            & (opportunities["elapsed_min"] <= max_by_event - 120.0)
            & (opportunities["elapsed_min"] % 10.0 < 1e-6)
        ].copy()
        responsive = float(config["opportunity"]["responsive_threshold"])
        weak = float(config["opportunity"]["weak_threshold"])
        opportunities["opportunity_class"] = np.where(
            opportunities["opportunity_score"] >= responsive,
            "responsive",
            np.where(
                opportunities["opportunity_score"] >= weak,
                "weakly_responsive",
                "flat",
            ),
        )
        opportunities = opportunities.rename(
            columns={"elapsed_min": "checkpoint_min"}
        )
        opportunities["spinup_min"] = spinup_min
    out = output_root / "opportunity" / "control_opportunity_catalog.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    opportunities.to_csv(out, index=False)
    responsive_events = (
        int(
            opportunities.loc[
                opportunities["opportunity_class"] == "responsive", "event_id"
            ].nunique()
        )
        if not opportunities.empty
        else 0
    )
    pre_screen_flat_count = (
        int(opportunities["opportunity_class"].eq("flat").sum())
        if not opportunities.empty
        else 0
    )
    low_opportunity_controls = int(not opportunities.empty)
    status = (
        "pass"
        if responsive_events >= 3 and low_opportunity_controls >= 1
        else "incomplete"
    )
    return _status(
        output_root,
        "ScanOpportunities",
        status,
        source_details=len(paths),
        responsive_events=responsive_events,
        pre_screen_flat_checkpoints=pre_screen_flat_count,
        low_opportunity_controls=low_opportunity_controls,
        opportunity_scoring_version="v3_facility_local",
        evidence=str(out),
        generation_manifest=str(generated_manifest_path),
    )


def _checkpoint_context(
    detail_path: Path,
    event_id: str,
    checkpoint_min: float,
    facility_ids: list[str],
    facility_semantics: pd.DataFrame,
    priority_nodes: list[str],
) -> tuple[CandidateContext, dict]:
    header = pd.read_csv(detail_path, nrows=0).columns.tolist()
    usecols = [
        column
        for column in header
        if column == "elapsed_min"
        or column.startswith(("a:", "flow:", "storage_volume:", "flood:", "h:"))
    ]
    detail = pd.read_csv(detail_path, usecols=usecols)
    nearest = int(
        np.argmin(np.abs(detail["elapsed_min"].to_numpy(float) - checkpoint_min))
    )
    row = detail.iloc[nearest]
    current = np.asarray(
        [float(row.get(f"a:{facility_id}", 1.0)) for facility_id in facility_ids]
    )
    raw_fallback = np.repeat(current.reshape(1, -1), 12, axis=0)
    fallback, fallback_projection_audit = project_frozen_anchor_schedule(
        raw_fallback,
        facility_ids,
        facility_semantics,
        max_k=8,
    )
    flow = {
        facility_id: abs(float(row.get(f"flow:{facility_id}", 0.0)))
        for facility_id in facility_ids
    }
    semantics = facility_semantics.set_index("facility_id", drop=False)
    head_difference: dict[str, float] = {}
    priority_risk: dict[str, float] = {}
    downstream_capacity: dict[str, float] = {}
    storage_headroom: dict[str, float] = {}
    priority_set = set(priority_nodes)
    for facility_id in facility_ids:
        semantic = (
            semantics.loc[facility_id]
            if facility_id in semantics.index
            else pd.Series(dtype=object)
        )
        upstream = str(semantic.get("from_node", ""))
        downstream = str(semantic.get("to_node", ""))
        upstream_head = float(row.get(f"head:{upstream}", row.get(f"h:{upstream}", 0.0)))
        downstream_head = float(
            row.get(f"head:{downstream}", row.get(f"h:{downstream}", 0.0))
        )
        head_difference[facility_id] = abs(upstream_head - downstream_head)
        priority_risk[facility_id] = max(
            upstream_head if upstream in priority_set else 0.0,
            downstream_head if downstream in priority_set else 0.0,
        )
        downstream_capacity[facility_id] = 1.0 / (1.0 + max(downstream_head, 0.0))
        storage_node = (
            upstream
            if str(semantic.get("storage_role", "")).lower() == "storage_outlet"
            else downstream
        )
        storage_depth = float(
            row.get(f"h:{storage_node}", row.get(f"head:{storage_node}", 0.0))
        )
        storage_headroom[facility_id] = 1.0 / (1.0 + max(storage_depth, 0.0))
    opportunity_scores = {
        facility_id: flow[facility_id] + 0.1 * head_difference[facility_id]
        for facility_id in facility_ids
    }
    active = {
        facility_id
        for facility_id in facility_ids
        if flow[facility_id] > 1e-8 or head_difference[facility_id] > 1e-4
    }
    context = CandidateContext(
        event_id=event_id,
        checkpoint_id=f"{event_id}__{checkpoint_min:.1f}",
        facility_ids=facility_ids,
        frozen_fallback_schedule=fallback,
        dynamic_internal_schedule=fallback,
        no_control_schedule=np.ones_like(fallback),
        hold_previous_schedule=fallback,
        opportunity_scores=opportunity_scores,
        hydraulically_active=active,
        event_phase=str(row.get("phase", "unknown")),
        online_features={
            "facility_flow_by_facility": flow,
            "head_difference_by_facility": head_difference,
            "priority_risk_by_facility": priority_risk,
            "storage_headroom_by_facility": storage_headroom,
            "downstream_capacity_by_facility": downstream_capacity,
        },
    )
    return context, {
        "active_facilities": sorted(active),
        "current_action": current,
        "fallback_projection_audit": fallback_projection_audit,
        "head_difference": head_difference,
    }


def _write_candidate_plan(
    checkpoints: pd.DataFrame,
    context: dict,
    out_path: Path,
    max_candidates: int,
) -> pd.DataFrame:
    config, root = context["config"], context["root"]
    facility_ids = _read_ids(_resolve(root, config["project"]["canonical_ids"]))
    semantics = pd.read_csv(_resolve(root, config["project"]["facility_semantics"]))
    from sewerrtc.simulation.action_policies import attach_reference_nodes

    semantics = attach_reference_nodes(
        semantics.assign(
            actuator_id=semantics.get("actuator_id", semantics["facility_id"])
        ),
        _resolve(root, config["project"]["network"]),
    )
    priority_nodes = _read_ids(_resolve(root, config["project"]["priority_nodes"]))
    generator = V4CandidateGenerator(
        facility_ids=facility_ids,
        facility_semantics=semantics,
        priority_nodes=_read_ids(_resolve(root, config["project"]["priority_nodes"])),
        max_k=8,
    )
    inventory = pd.read_csv(
        context["output_root"] / "inventory" / "event_inventory.csv"
    ).set_index("event_id", drop=False)
    rows: list[dict] = []
    for _, checkpoint in checkpoints.iterrows():
        event_id = str(checkpoint["event_id"])
        detail_path = Path(str(checkpoint["source_detail"]))
        candidate_context, current = _checkpoint_context(
            detail_path,
            event_id,
            float(checkpoint["checkpoint_min"]),
            facility_ids,
            semantics,
            priority_nodes,
        )
        checkpoint_role = str(
            checkpoint.get("checkpoint_role", "responsive")
        )
        candidates = generator.generate(
            candidate_context,
            # Role filtering happens below. Generate a broad pool first so
            # flat probes cannot consume a responsive checkpoint's quota.
            max_total=max(200, max_candidates),
        )
        if checkpoint_role == "flat_action_probe":
            candidates = [
                candidate
                for candidate in candidates
                if candidate.family == "flat_action_probe"
            ][:max_candidates]
        else:
            candidates = [
                candidate
                for candidate in candidates
                if candidate.family != "flat_action_probe"
            ][:max_candidates]
        rainfall_path = (
            str(inventory.loc[event_id, "rainfall_path"])
            if event_id in inventory.index
            else ""
        )
        for candidate in candidates:
            rows.append(
                {
                    "case_id": (
                        f"{event_id}__{float(checkpoint['checkpoint_min']):.0f}"
                        f"__{candidate.candidate_id}"
                    ),
                    "event_id": event_id,
                    "checkpoint_min": float(checkpoint["checkpoint_min"]),
                    "spinup_min": int(checkpoint.get("spinup_min", 0)),
                    "checkpoint_role": checkpoint_role,
                    "source_detail": str(detail_path),
                    "rainfall_path": rainfall_path,
                    "candidate_id": candidate.candidate_id,
                    "family": candidate.family,
                    "anchor": candidate.anchor_name,
                    "requested_schedule_json": json.dumps(
                        candidate.requested_schedule.tolist(), separators=(",", ":")
                    ),
                    "projected_schedule_json": json.dumps(
                        candidate.projected_schedule.tolist(), separators=(",", ":")
                    ),
                    "projected_schedule_sha256": candidate.projected_schedule_hash,
                    "constraint_audit_json": json.dumps(
                        candidate.constraint_audit, separators=(",", ":")
                    ),
                    "frozen_anchor_schedule_json": json.dumps(
                        candidate_context.frozen_fallback_schedule.tolist(),
                        separators=(",", ":"),
                    ),
                    "active_facilities_json": json.dumps(
                        current["active_facilities"], separators=(",", ":")
                    ),
                    "status": "planned",
                }
            )
    plan = pd.DataFrame(rows).drop_duplicates(
        ["event_id", "checkpoint_min", "projected_schedule_sha256"]
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plan.to_csv(out_path, index=False)
    return plan


def _exact_local_search_plan(
    canary_samples: pd.DataFrame,
    base_plan: pd.DataFrame,
    context: dict,
    max_per_checkpoint: int = 100,
) -> pd.DataFrame:
    """Expand SWMM-ranked Canary schedules with explicit sparse search operators."""
    config, root = context["config"], context["root"]
    facility_ids = _read_ids(_resolve(root, config["project"]["canonical_ids"]))
    semantics = pd.read_csv(_resolve(root, config["project"]["facility_semantics"]))
    rows: list[dict] = []
    key_columns = ["event_id", "checkpoint_min"]
    for key, checkpoint_plan in base_plan.groupby(key_columns):
        event_id, checkpoint_min = key
        evidence = canary_samples[
            (canary_samples["event_id"].astype(str) == str(event_id))
            & np.isclose(
                pd.to_numeric(canary_samples["checkpoint_min"], errors="coerce"),
                float(checkpoint_min),
            )
        ].copy()
        if evidence.empty:
            continue
        evidence["_search_penalty"] = (
            evidence["delta_pfv_h120_vs_no_control_m3"].clip(lower=0.0)
            + evidence["delta_tfv_h120_vs_dynamic_internal_m3"].clip(lower=0.0)
            + 300.0
            * evidence["delta_peak_h120_vs_dynamic_internal_m3s"].clip(lower=0.0)
        )
        beams = evidence.sort_values(
            ["_search_penalty", "delta_tfv_h120_vs_dynamic_internal_m3"]
        ).head(5)
        template = checkpoint_plan.iloc[0]
        active = json.loads(template.get("active_facilities_json", "[]"))
        active_indices = [
            facility_ids.index(facility_id)
            for facility_id in active
            if facility_id in facility_ids
        ][:8]

        def append_schedule(
            requested: np.ndarray,
            family: str,
            candidate_id: str,
            anchor: np.ndarray,
            operator: str,
            parent_case_id: str,
        ) -> None:
            projected, audit = project_candidate_schedule(
                requested,
                anchor,
                facility_ids,
                semantics,
                max_k=8,
            )
            if max(audit["k_by_step"], default=0) == 0:
                return
            item = template.to_dict()
            item.update(
                {
                    "case_id": (
                        f"{event_id}__{float(checkpoint_min):.0f}"
                        f"__exact_{candidate_id}"
                    ),
                    "candidate_id": f"exact_{candidate_id}",
                    "family": family,
                    "anchor": "frozen_fallback",
                    "requested_schedule_json": json.dumps(
                        requested.tolist(), separators=(",", ":")
                    ),
                    "projected_schedule_json": json.dumps(
                        projected.tolist(), separators=(",", ":")
                    ),
                    "projected_schedule_sha256": audit[
                        "projected_schedule_hash"
                    ],
                    "constraint_audit_json": json.dumps(
                        audit, separators=(",", ":")
                    ),
                    "search_operator": operator,
                    "parent_case_id": parent_case_id,
                }
            )
            rows.append(item)

        for beam_rank, (_, beam) in enumerate(beams.iterrows()):
            schedule = np.asarray(
                json.loads(beam["projected_schedule_json"]), dtype=float
            )
            anchor = np.asarray(
                json.loads(beam["frozen_anchor_schedule_json"]), dtype=float
            )
            changed = np.flatnonzero(
                np.any(np.abs(schedule - anchor) > 1e-6, axis=0)
            ).tolist()
            parent = str(beam["case_id"])
            # Leave-one-out removes PFV-dangerous coordinates from an otherwise
            # promising exact-SWMM schedule.
            for index in changed:
                requested = schedule.copy()
                requested[:, index] = anchor[:, index]
                append_schedule(
                    requested,
                    "exact_leave_one_out",
                    f"b{beam_rank}_loo{index}",
                    anchor,
                    "leave_one_out",
                    parent,
                )
            # Coordinate search uses legal absolute levels, not small saturated
            # DI perturbations.
            for index in active_indices[:4]:
                for level in (0.0, 0.25, 0.50, 0.75, 1.0):
                    requested = schedule.copy()
                    requested[:6, index] = level
                    append_schedule(
                        requested,
                        "exact_coordinate",
                        f"b{beam_rank}_c{index}_{level:.2f}",
                        anchor,
                        "coordinate_search",
                        parent,
                    )
            # Stepwise add/remove and a small beam over active pairs.
            for k in (1, 2, 4, 6, 8):
                chosen = active_indices[: min(k, len(active_indices))]
                if not chosen:
                    continue
                requested = anchor.copy()
                requested[:6, chosen] = 1.0
                append_schedule(
                    requested,
                    "exact_stepwise",
                    f"b{beam_rank}_add{k}",
                    anchor,
                    "stepwise_add",
                    parent,
                )
            for first, second in zip(active_indices[:4], active_indices[1:5]):
                requested = anchor.copy()
                requested[:4, first] = 1.0 - np.round(anchor[:4, first])
                requested[2:6, second] = 1.0 - np.round(anchor[2:6, second])
                append_schedule(
                    requested,
                    "exact_beam_pair",
                    f"b{beam_rank}_pair{first}_{second}",
                    anchor,
                    "beam_pair",
                    parent,
                )
    if not rows:
        return pd.DataFrame(columns=base_plan.columns)
    local = pd.DataFrame(rows)
    combined = pd.concat([base_plan, local], ignore_index=True, sort=False)
    combined = combined.drop_duplicates(
        ["event_id", "checkpoint_min", "projected_schedule_sha256"]
    )
    return (
        combined.groupby(["event_id", "checkpoint_min"], group_keys=False)
        .head(int(max_per_checkpoint))
        .reset_index(drop=True)
    )


def stage_plan_canary(context: dict) -> int:
    tiny_gate = context["output_root"] / "stage_status" / "AuditExactPrefixTiny.json"
    if not tiny_gate.exists() or _read_json(tiny_gate).get("status") != "pass":
        return _status(
            context["output_root"],
            "PlanExcitationCanary",
            "incomplete",
            reason="AuditExactPrefixTiny must pass before the 80-case Canary",
        )
    catalog_path = (
        context["output_root"] / "opportunity" / "control_opportunity_catalog.csv"
    )
    if not catalog_path.exists():
        return _status(
            context["output_root"],
            "PlanExcitationCanary",
            "incomplete",
            reason="ScanOpportunities must pass first",
        )
    catalog = pd.read_csv(catalog_path)
    responsive = (
        catalog[catalog["opportunity_class"] == "responsive"]
        .sort_values("opportunity_score", ascending=False)
        .drop_duplicates("event_id")
        .head(3)
        .copy()
    )
    responsive_keys = set(
        zip(
            responsive["event_id"].astype(str),
            responsive["checkpoint_min"].astype(float),
        )
    )
    low_control_pool = catalog[
        ~catalog.apply(
            lambda row: (
                str(row["event_id"]),
                float(row["checkpoint_min"]),
            )
            in responsive_keys,
            axis=1,
        )
    ]
    low_control = low_control_pool.nsmallest(1, "opportunity_score").copy()
    if len(responsive) < 3 or low_control.empty:
        return _status(
            context["output_root"],
            "PlanExcitationCanary",
            "incomplete",
            responsive_events=int(len(responsive)),
            low_opportunity_controls=int(len(low_control)),
        )
    responsive["checkpoint_role"] = "responsive"
    low_control["checkpoint_role"] = "flat_action_probe"
    checkpoints = pd.concat([responsive, low_control], ignore_index=True)
    plan = _write_candidate_plan(
        checkpoints,
        context,
        context["output_root"] / "canary" / "canary_case_plan.csv",
        max_candidates=20,
    )
    plan["repeat_count"] = 1
    responsive_indices = plan.index[plan["checkpoint_role"].eq("responsive")]
    if len(responsive_indices):
        plan.loc[responsive_indices[0], "repeat_count"] = int(
            context["config"]["thresholds"]["canary"]["repeat_count"]
        )
        plan.to_csv(
            context["output_root"] / "canary" / "canary_case_plan.csv",
            index=False,
        )
    status = "pass" if not plan.empty else "incomplete"
    return _status(
        context["output_root"],
        "PlanExcitationCanary",
        status,
        planned_cases=int(len(plan)),
        events=int(plan["event_id"].nunique()) if len(plan) else 0,
    )


def stage_plan_exact_prefix_tiny(context: dict) -> int:
    catalog_path = (
        context["output_root"] / "opportunity" / "control_opportunity_catalog.csv"
    )
    if not catalog_path.exists():
        return _status(
            context["output_root"],
            "PlanExactPrefixTiny",
            "incomplete",
            reason="ScanOpportunities must pass first",
        )
    catalog = pd.read_csv(catalog_path)
    responsive = catalog[
        catalog["opportunity_class"].eq("responsive")
    ].sort_values(
        ["opportunity_score", "event_id", "checkpoint_min"],
        ascending=[False, True, True],
    )
    if responsive.empty:
        return _status(
            context["output_root"],
            "PlanExactPrefixTiny",
            "incomplete",
            reason="no responsive checkpoint available",
        )
    checkpoint = responsive.head(1).copy()
    checkpoint["checkpoint_role"] = "responsive"
    tiny_path = context["output_root"] / "tiny" / "tiny_case_plan.csv"
    generated = _write_candidate_plan(
        checkpoint,
        context,
        tiny_path,
        max_candidates=20,
    )
    if generated.empty:
        return _status(
            context["output_root"],
            "PlanExactPrefixTiny",
            "incomplete",
            reason="candidate generator returned no actual-unique candidate",
        )
    tiny = generated.head(1).copy()
    tiny["repeat_count"] = 1
    tiny.to_csv(tiny_path, index=False)
    return _status(
        context["output_root"],
        "PlanExactPrefixTiny",
        "pass",
        planned_cases=1,
        event_id=str(tiny.iloc[0]["event_id"]),
        checkpoint_min=float(tiny.iloc[0]["checkpoint_min"]),
        case_id=str(tiny.iloc[0]["case_id"]),
        evidence=str(tiny_path),
    )


def stage_audit_exact_prefix_tiny(context: dict) -> int:
    plan = context["output_root"] / "tiny" / "tiny_case_plan.csv"
    runs = context["output_root"] / "tiny" / "runs"
    samples, audit = _audit_run_dataset(
        context, plan, runs, "tiny/tiny_sample_manifest.csv"
    )
    pass_gate = bool(
        audit["accounting_closed"]
        and audit["accepted"] == 1
        and audit["readback_complete"]
        and audit["same_state_hash_complete"]
        and audit["max_actual_k"] <= 8
        and len(samples) == 1
        and bool(samples["prefix_history_hash_match"].all())
        and bool(samples["checkpoint_pre_action_hash_match"].all())
        and bool(
            samples["post_actual_schedule_sha256"].astype(str).str.len().gt(0).all()
        )
        and bool(
            samples["post_readback_schedule_sha256"].astype(str).str.len().gt(0).all()
        )
    )
    status = canary_gate_status(pass_gate, audit["accepted"])
    return _status(
        context["output_root"],
        "AuditExactPrefixTiny",
        status,
        **audit,
        exact_prefix_required=True,
        numerical_repeat_max_range=safe_repeat_noise_ranges(samples),
    )


def _rainfall_duration_min(path: Path) -> float:
    frame = pd.read_csv(path)
    if "elapsed_min" in frame.columns:
        return float(pd.to_numeric(frame["elapsed_min"], errors="coerce").max())
    if "minute" in frame.columns:
        return float(pd.to_numeric(frame["minute"], errors="coerce").max())
    return float(max(120, len(frame) * 5))


def _create_spinup_inp(
    network: Path,
    rainfall: Path,
    output: Path,
    rain_duration_min: float,
    spinup_min: int,
    tail_min: int = 180,
    disabled_control_targets: list[str] | None = None,
) -> int:
    from sewerrtc.io.swmm_mutation import mutate_inp_for_event

    total_duration = int(spinup_min + rain_duration_min + tail_min)
    mutate_inp_for_event(
        network,
        rainfall,
        output,
        total_duration,
        strip_controls=False,
        disabled_control_targets=disabled_control_targets,
    )
    if spinup_min <= 0:
        return total_duration
    lines = output.read_text(encoding="utf-8", errors="ignore").splitlines()
    shifted: list[str] = []
    offset = timedelta(minutes=int(spinup_min))
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("RTC_RAIN_TS"):
            parts = stripped.split()
            if len(parts) >= 4:
                try:
                    date = datetime.strptime(parts[1], "%m/%d/%Y")
                    clock = datetime.strptime(parts[2], "%H:%M")
                    original = date.replace(hour=clock.hour, minute=clock.minute)
                    updated = original + offset
                    shifted.append(
                        f"{'RTC_RAIN_TS':<16} {updated:%m/%d/%Y} "
                        f"{updated:%H:%M} {parts[3]}"
                    )
                    continue
                except ValueError:
                    pass
        shifted.append(line)
    shifted_tmp = output.with_name(f"{output.name}.shift.tmp")
    shifted_tmp.write_text("\n".join(shifted) + "\n", encoding="utf-8")
    os.replace(shifted_tmp, output)
    return total_duration


def _run_opportunity_worker(payload: dict) -> dict:
    try:
        from sewerrtc.simulation.pyswmm_runner import run_swmm_fixed_action

        out_dir = Path(payload["out_dir"])
        out_dir.mkdir(parents=True, exist_ok=True)
        detail = out_dir / "dynamic_internal_opportunity_detail.csv"
        if detail.exists() and payload["resume"]:
            return {
                "event_id": payload["event_id"],
                "status": "accepted",
                "detail_path": str(detail),
                "reused": True,
                "error": "",
            }
        inp = out_dir / "event_spinup.inp"
        total = _create_spinup_inp(
            Path(payload["network"]),
            Path(payload["rainfall_path"]),
            inp,
            float(payload["rain_duration_min"]),
            int(payload["spinup_min"]),
        )
        actuators = pd.read_csv(payload["facility_semantics"])
        if "actuator_id" not in actuators.columns:
            actuators["actuator_id"] = actuators["facility_id"]
        if "link_type" not in actuators.columns and "actuator_type" in actuators.columns:
            actuators["link_type"] = actuators["actuator_type"]
        run_swmm_fixed_action(
            inp_path=inp,
            actuators=actuators,
            priority_nodes=payload["priority_nodes"],
            out_detail_csv=detail,
            event_id=payload["event_id"],
            duration_min=int(payload["spinup_min"] + payload["rain_duration_min"]),
            prefix_schedule=None,
            override_start_min=float(total + 1),
            post_action=np.ones(len(actuators), dtype=float),
            control_step_sec=300,
            decision_interval_sec=600,
            simulation_duration_min=total,
            policy_id="dynamic_internal_opportunity",
            cleanup_swmm_artifacts=True,
            record_node_ids=payload["record_node_ids"],
            hydraulic_summary_start_min=float(payload["spinup_min"]),
        )
        return {
            "event_id": payload["event_id"],
            "status": "accepted",
            "detail_path": str(detail),
            "reused": False,
            "error": "",
        }
    except Exception as exc:
        return {
            "event_id": payload.get("event_id", ""),
            "status": "failed",
            "detail_path": "",
            "reused": False,
            "error": f"{type(exc).__name__}: {exc}",
        }


def _run_case_worker(payload: dict) -> dict:
    """Isolated authoritative SWMM worker for one candidate/reference branch set."""
    try:
        project_root = Path(payload["project_root"])
        case_dir = Path(payload["case_dir"])
        case_dir.mkdir(parents=True, exist_ok=True)
        from sewerrtc.simulation.pyswmm_runner import run_swmm_fixed_action

        actuators = pd.read_csv(payload["facility_semantics"])
        if "actuator_id" not in actuators.columns:
            actuators["actuator_id"] = actuators["facility_id"]
        if "link_type" not in actuators.columns and "actuator_type" in actuators.columns:
            actuators["link_type"] = actuators["actuator_type"]
        priority = payload["priority_nodes"]
        checkpoint = float(payload["checkpoint_min"])
        simulation_duration = int(payload["simulation_duration_min"])
        reference_dir = Path(payload["reference_dir"])
        reference_dir.mkdir(parents=True, exist_ok=True)
        reference_completion_path = reference_dir / "completion.json"
        reference_complete = False
        if reference_completion_path.exists():
            try:
                reference_completion = _read_json(reference_completion_path)
                expected_reference_paths = [
                    reference_dir / "no_control_detail.csv",
                    reference_dir / "dynamic_internal_rules_detail.csv",
                    reference_dir / "hold_previous_detail.csv",
                ]
                reference_complete = bool(
                    reference_completion.get("status") == "pass"
                    and reference_completion.get("reference_contract_hash")
                    == payload["reference_contract_hash"]
                    and all(path.exists() and path.stat().st_size > 0 for path in expected_reference_paths)
                )
            except (OSError, ValueError, TypeError):
                reference_complete = False
        native_inp_path = reference_dir / "event_with_controls.inp"
        if not native_inp_path.exists():
            required_tail_min = max(
                180,
                int(
                    math.ceil(
                        simulation_duration
                        - int(payload.get("spinup_min", 0))
                        - float(payload["rain_duration_min"])
                    )
                ),
            )
            _create_spinup_inp(
                Path(payload["network"]),
                Path(payload["rainfall_path"]),
                native_inp_path,
                float(payload["rain_duration_min"]),
                int(payload.get("spinup_min", 0)),
                tail_min=required_tail_min,
            )
        from sewerrtc.contracts.swmm_control_parser import parse_swmm_controls

        managed_ids = set(actuators["actuator_id"].astype(str))
        native_actions = parse_swmm_controls(native_inp_path)["actions"]
        native_background_targets = {
            str(action["actuator_id"])
            for action in native_actions
            if str(action["actuator_id"]) not in managed_ids
        }
        control_partition = {
            "native_action_count": int(len(native_actions)),
            "managed_native_target_count": int(
                len(
                    {
                        str(action["actuator_id"])
                        for action in native_actions
                        if str(action["actuator_id"]) in managed_ids
                    }
                )
            ),
            "native_background_target_count": int(
                len(native_background_targets)
            ),
            "prefix_mode": "common_native_rules",
            "post_mode": "time_gated_high_priority_rules",
            "override_priority": int(payload.get("override_priority", 100)),
        }
        candidate_path = case_dir / "candidate_detail.csv"
        case_completion_path = case_dir / "completion.json"
        if (
            payload.get("resume", False)
            and reference_complete
            and case_completion_path.exists()
        ):
            existing_case = _read_json(case_completion_path)
            repeat_paths = [
                Path(str(value))
                for value in existing_case.get("candidate_repeat_paths", [])
            ]
            if (
                existing_case.get("status") == "pass"
                and existing_case.get("reference_contract_hash")
                == payload["reference_contract_hash"]
                and candidate_path.exists()
                and candidate_path.stat().st_size > 0
                and len(repeat_paths) >= int(payload.get("repeat_count", 1))
                and all(path.exists() and path.stat().st_size > 0 for path in repeat_paths)
            ):
                return {
                    "case_id": payload["case_id"],
                    "status": "accepted",
                    "error": "",
                    "reused": True,
                }
        schedule = np.asarray(payload["candidate_schedule"], dtype=float)
        anchor = np.asarray(payload["anchor_schedule"], dtype=float)
        ones = np.ones_like(schedule)
        reference_branches = {
            "no_control": ones,
            "hold_previous": anchor,
        }
        results: dict[str, dict] = {}
        di_path = reference_dir / "dynamic_internal_rules_detail.csv"
        if not (reference_complete and payload.get("reuse_references", False)):
            results["dynamic_internal_rules"] = run_swmm_fixed_action(
                inp_path=native_inp_path,
                actuators=actuators,
                priority_nodes=priority,
                out_detail_csv=di_path,
                event_id=payload["event_id"],
                duration_min=int(payload["rain_duration_min"]),
                prefix_schedule=None,
                override_start_min=checkpoint + 120.0,
                post_action=np.ones(len(actuators), dtype=float),
                control_step_sec=300,
                decision_interval_sec=600,
                stop_after_override_min=0.0,
                prefix_history_min=180.0,
                simulation_duration_min=simulation_duration,
                policy_id="dynamic_internal_rules",
                cleanup_swmm_artifacts=True,
                hydraulic_summary_start_min=checkpoint - 60.0,
            )
        else:
            results["dynamic_internal_rules"] = {
                "detail_file": str(di_path),
                "reused": True,
            }
        from sewerrtc.io.swmm_mutation import inject_time_gated_control_schedule

        def run_rule_branch(
            branch: str,
            branch_schedule: np.ndarray,
            detail_path: Path,
            branch_inp: Path,
            policy_id: str,
        ) -> dict:
            inject_time_gated_control_schedule(
                native_inp_path,
                branch_inp,
                actuators,
                branch_schedule,
                checkpoint_min=checkpoint,
                decision_interval_sec=600,
                priority=int(payload.get("override_priority", 100)),
                rule_prefix=(
                    f"G5R_{branch[:4]}_"
                    f"{hashlib.sha256(str(payload['case_id']).encode('utf-8')).hexdigest()[:10]}"
                ),
            )
            return run_swmm_fixed_action(
                inp_path=branch_inp,
                actuators=actuators,
                priority_nodes=priority,
                out_detail_csv=detail_path,
                event_id=payload["event_id"],
                duration_min=int(payload["rain_duration_min"]),
                prefix_schedule=None,
                override_start_min=checkpoint,
                post_action=branch_schedule,
                control_step_sec=300,
                decision_interval_sec=600,
                stop_after_override_min=120,
                prefix_history_min=60,
                simulation_duration_min=simulation_duration,
                policy_id=policy_id,
                cleanup_swmm_artifacts=True,
                hydraulic_summary_start_min=checkpoint - 60.0,
                post_control_mode="native_rules",
            )

        candidate_inp = case_dir / "candidate_time_gated.inp"
        results["candidate"] = run_rule_branch(
            "candidate",
            schedule,
            candidate_path,
            candidate_inp,
            "candidate",
        )
        candidate_repeat_paths = [str(candidate_path)]
        for repeat_index in range(2, int(payload.get("repeat_count", 1)) + 1):
            repeat_path = case_dir / f"candidate_repeat_{repeat_index}_detail.csv"
            run_rule_branch(
                f"candidate_repeat_{repeat_index}",
                schedule,
                repeat_path,
                case_dir / f"candidate_repeat_{repeat_index}_time_gated.inp",
                f"candidate_repeat_{repeat_index}",
            )
            candidate_repeat_paths.append(str(repeat_path))
        for branch, branch_schedule in reference_branches.items():
            detail_path = reference_dir / f"{branch}_detail.csv"
            if reference_complete and payload.get("reuse_references", False):
                results[branch] = {"detail_file": str(detail_path), "reused": True}
                continue
            result = run_rule_branch(
                branch,
                branch_schedule,
                detail_path,
                reference_dir / f"{branch}_time_gated.inp",
                branch,
            )
            results[branch] = result
        if not reference_complete:
            _atomic_json(
                reference_completion_path,
                {
                    "status": "pass",
                    "event_id": payload["event_id"],
                    "checkpoint_min": checkpoint,
                    "branches": {
                        "no_control": str(reference_dir / "no_control_detail.csv"),
                        "dynamic_internal_rules": str(di_path),
                        "hold_previous": str(reference_dir / "hold_previous_detail.csv"),
                    },
                    "network_sha256": _sha256(Path(payload["network"])),
                    "rainfall_sha256": _sha256(Path(payload["rainfall_path"])),
                    "hotstart_used": False,
                    "control_partition": control_partition,
                    "reference_contract_hash": payload[
                        "reference_contract_hash"
                    ],
                },
            )
        branch_paths = {
            "candidate": candidate_path,
            "no_control": reference_dir / "no_control_detail.csv",
            "dynamic_internal_rules": di_path,
            "hold_previous": reference_dir / "hold_previous_detail.csv",
        }
        branch_hashes = {
            branch: branch_state_hashes(
                pd.read_csv(path),
                checkpoint_min=checkpoint,
                facility_ids=actuators["actuator_id"].astype(str).tolist(),
            )
            for branch, path in branch_paths.items()
        }
        completion = {
            "status": "pass",
            "case_id": payload["case_id"],
            "branches": {name: str(path) for name, path in branch_paths.items()},
            "network_sha256": _sha256(Path(payload["network"])),
            "rainfall_sha256": _sha256(Path(payload["rainfall_path"])),
            "hotstart_used": False,
            "candidate_repeat_paths": candidate_repeat_paths,
            "control_partition": control_partition,
            "reference_contract_hash": payload["reference_contract_hash"],
            "branch_hashes": branch_hashes,
            "prefix_history_hash_match": hashes_match_across_branches(
                branch_hashes, ("prefix_history_sha256",)
            ),
            "checkpoint_pre_action_hash_match": hashes_match_across_branches(
                branch_hashes, ("checkpoint_pre_action_sha256",)
            ),
        }
        _atomic_json(case_dir / "completion.json", completion)
        return {"case_id": payload["case_id"], "status": "accepted", "error": ""}
    except Exception as exc:
        return {
            "case_id": payload.get("case_id", ""),
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
        }


def _run_checkpoint_worker(payloads: list[dict]) -> list[dict]:
    """Run one checkpoint serially so its three references are cached once."""
    results = []
    progress_root = (
        Path(payloads[0]["case_dir"]).parent / "_checkpoint_progress"
        if payloads
        else None
    )
    for index, payload in enumerate(payloads):
        item = dict(payload)
        item["reuse_references"] = bool(payload.get("resume", False) or index > 0)
        result = _run_case_worker(item)
        results.append(result)
        if progress_root is not None:
            progress_root.mkdir(parents=True, exist_ok=True)
            progress_id = hashlib.sha256(
                (
                    str(payload["event_id"])
                    + "|"
                    + str(payload["checkpoint_min"])
                    + (
                        "|" + str(payload["case_id"])
                        if len(payloads) == 1
                        else ""
                    )
                ).encode("utf-8")
            ).hexdigest()[:16]
            _atomic_json(
                progress_root / f"{progress_id}.json",
                {
                    "event_id": payload["event_id"],
                    "checkpoint_min": payload["checkpoint_min"],
                    "completed_cases": len(results),
                    "planned_cases": len(payloads),
                    "last_case_id": payload["case_id"],
                    "last_status": result.get("status"),
                    "heartbeat_epoch": time.time(),
                },
            )
        if result.get("status") == "failed":
            break
    return results


def _run_plan(
    context: dict,
    plan_path: Path,
    run_root: Path,
    workers: int,
    limit: int,
    resume: bool,
) -> tuple[pd.DataFrame, int]:
    if not plan_path.exists():
        return pd.DataFrame(), EXIT_INCOMPLETE
    full_plan = pd.read_csv(plan_path)
    planned_total = int(len(full_plan))
    config, root = context["config"], context["root"]
    priority = _read_ids(_resolve(root, config["project"]["priority_nodes"]))
    reference_contract_hash = hashlib.sha256(
        (
            _sha256(_resolve(root, config["project"]["dataset_contract"]))
            + _sha256(_resolve(root, config["project"]["recovery_contract"]))
            + _sha256(Path(context["config_path"]))
        ).encode("ascii")
    ).hexdigest()
    completed_before_manifest = (
        rebuild_run_manifest_from_completions(
            full_plan,
            run_root,
            reference_contract_hash=reference_contract_hash,
        )
        if resume
        else pd.DataFrame()
    )
    completed_before = (
        set(completed_before_manifest["case_id"].astype(str))
        if "case_id" in completed_before_manifest.columns
        else set()
    )
    plan = select_pending_plan(
        full_plan,
        completed_case_ids=completed_before,
        limit=limit,
    )
    payloads: list[dict] = []
    for _, row in plan.iterrows():
        rainfall = Path(str(row["rainfall_path"]))
        if not rainfall.exists():
            continue
        rain_duration = _rainfall_duration_min(rainfall)
        checkpoint = float(row["checkpoint_min"])
        spinup = int(row.get("spinup_min", config["runtime"].get("spinup_min", 0)))
        simulation_duration = int(
            max(spinup + rain_duration + 180, checkpoint + 125)
        )
        anchor_schedule = np.asarray(
            json.loads(row["frozen_anchor_schedule_json"]), dtype=np.float64
        )
        anchor_schedule_hash = hashlib.sha256(
            np.round(anchor_schedule, 8).tobytes()
        ).hexdigest()
        payloads.append(
            {
                "project_root": str(root),
                "case_id": str(row["case_id"]),
                "event_id": str(row["event_id"]),
                "checkpoint_min": checkpoint,
                "network": str(_resolve(root, config["project"]["network"])),
                "rainfall_path": str(rainfall),
                "rain_duration_min": rain_duration,
                "spinup_min": spinup,
                "simulation_duration_min": simulation_duration,
                "facility_semantics": str(
                    _resolve(root, config["project"]["facility_semantics"])
                ),
                "priority_nodes": priority,
                "candidate_schedule": json.loads(row["projected_schedule_json"]),
                "anchor_schedule": anchor_schedule.tolist(),
                "case_dir": str(run_root / str(row["case_id"])),
                "reference_dir": str(
                    run_root
                    / "_reference_cache"
                    / reference_cache_key(
                        str(row["event_id"]),
                        checkpoint,
                        f"{reference_contract_hash}|{anchor_schedule_hash}",
                    )
                ),
                "resume": bool(resume),
                "repeat_count": int(row.get("repeat_count", 1)),
                "reference_contract_hash": reference_contract_hash,
                "override_priority": int(
                    config.get("runtime", {}).get("override_rule_priority", 100)
                ),
            }
        )
    results: list[dict] = []
    manifest_path = run_root / "run_manifest.csv"
    recovered_manifest = (
        completed_before_manifest
        if resume
        else pd.DataFrame()
    )
    previous_manifest = (
        pd.concat(
            [
                pd.read_csv(manifest_path)
                if manifest_path.exists()
                else pd.DataFrame(),
                recovered_manifest,
            ],
            ignore_index=True,
            sort=False,
        )
        if resume
        else pd.DataFrame()
    )
    if "case_id" in previous_manifest.columns:
        previous_manifest = previous_manifest.drop_duplicates(
            "case_id", keep="last"
        )
    grouped: dict[tuple[str, float], list[dict]] = {}
    for payload in payloads:
        grouped.setdefault(
            (str(payload["event_id"]), float(payload["checkpoint_min"])), []
        ).append(payload)
    groups = list(grouped.values())
    run_root.mkdir(parents=True, exist_ok=True)

    def flush_progress() -> None:
        frame = pd.concat(
            [previous_manifest, pd.DataFrame(results)],
            ignore_index=True,
        )
        if "case_id" in frame.columns:
            frame = frame.drop_duplicates("case_id", keep="last")
        temporary = run_root / "run_manifest.csv.tmp"
        frame.to_csv(temporary, index=False)
        os.replace(temporary, run_root / "run_manifest.csv")
        _atomic_json(
            run_root / "heartbeat.json",
            {
                "completed_cases": int(len(frame)),
                "planned_cases": planned_total,
                "attempted_this_batch": int(len(results)),
                "heartbeat_epoch": time.time(),
            },
        )

    if workers <= 1:
        for group in groups:
            results.extend(_run_checkpoint_worker(group))
            flush_progress()
    else:
        worker_count = min(int(workers), 16)

        def execute_units(units: list[list[dict]]) -> None:
            if not units:
                return
            with ProcessPoolExecutor(
                max_workers=min(worker_count, len(units))
            ) as executor:
                futures = [
                    executor.submit(_run_checkpoint_worker, unit)
                    for unit in units
                ]
                for future in as_completed(futures):
                    results.extend(future.result())
                    flush_progress()

        # Phase 1: one writer per missing event/checkpoint reference cache.
        # Phase 2: all remaining candidates can safely read those immutable
        # caches and fan out to the full worker budget.
        prewarm_units: list[list[dict]] = []
        remaining_payloads: list[dict] = []
        for group in groups:
            if reference_cache_is_ready(
                Path(group[0]["reference_dir"]),
                str(group[0]["reference_contract_hash"]),
            ):
                remaining_payloads.extend(group)
            else:
                prewarm_units.append([group[0]])
                remaining_payloads.extend(group[1:])
        execute_units(prewarm_units)
        if not any(item.get("status") == "failed" for item in results):
            execute_units([[payload] for payload in remaining_payloads])
    result = pd.DataFrame(results)
    if not previous_manifest.empty:
        accumulated = pd.concat([previous_manifest, result], ignore_index=True)
        if "case_id" in accumulated.columns:
            accumulated = accumulated.drop_duplicates("case_id", keep="last")
    else:
        accumulated = result.copy()
    accumulated.to_csv(manifest_path, index=False)
    completed_after_manifest = rebuild_run_manifest_from_completions(
        full_plan,
        run_root,
        reference_contract_hash=reference_contract_hash,
    )
    completed_after = (
        set(completed_after_manifest["case_id"].astype(str))
        if "case_id" in completed_after_manifest.columns
        else set()
    )
    remaining = max(0, planned_total - len(completed_after))
    _atomic_json(
        run_root / "run_progress.json",
        {
            "planned_total": planned_total,
            "completed_before": int(len(completed_before)),
            "attempted_this_batch": int(len(result)),
            "completed_total": int(len(completed_after)),
            "remaining": int(remaining),
            "batch_complete": bool(
                len(result) == len(plan)
                and (result.empty or not result["status"].eq("failed").any())
            ),
            "scope_complete": remaining == 0,
        },
    )
    batch_failed = bool(
        len(result) and result["status"].eq("failed").any()
    )
    code = (
        EXIT_PASS
        if len(result) == len(plan) and not batch_failed
        else (
            EXIT_RUNTIME_ERROR
            if batch_failed
            else EXIT_INCOMPLETE
        )
    )
    return result, code


def stage_run_named_plan(
    context: dict,
    stage: str,
    plan_path: Path,
    run_root: Path,
    workers: int,
    limit: int,
    resume: bool,
) -> int:
    result, code = _run_plan(
        context, plan_path, run_root, workers=workers, limit=limit, resume=resume
    )
    progress_path = run_root / "run_progress.json"
    progress = _read_json(progress_path) if progress_path.exists() else {}
    status = {
        EXIT_PASS: "pass",
        EXIT_INCOMPLETE: "incomplete",
        EXIT_RUNTIME_ERROR: "runtime_error",
    }.get(code, "runtime_error")
    return _status(
        context["output_root"],
        stage,
        status,
        attempted=int(len(result)),
        accepted=int(result["status"].eq("accepted").sum()) if len(result) else 0,
        failed=int(result["status"].eq("failed").sum()) if len(result) else 0,
        manifest=str(run_root / "run_manifest.csv"),
        **progress,
    )


def _window(path: Path, checkpoint: float) -> pd.DataFrame:
    frame = pd.read_csv(path)
    return frame[
        (frame["elapsed_min"] >= checkpoint)
        & (frame["elapsed_min"] < checkpoint + 120.0)
    ].reset_index(drop=True)


def _audit_run_dataset(
    context: dict, plan_path: Path, run_root: Path, output_name: str
) -> tuple[pd.DataFrame, dict]:
    config, root = context["config"], context["root"]
    from sewerrtc.prompt3 import action_effect_v4 as v4_features
    plan = pd.read_csv(plan_path)
    priority = _read_ids(_resolve(root, config["project"]["priority_nodes"]))
    facilities = _read_ids(_resolve(root, config["project"]["canonical_ids"]))
    rows: list[dict] = []
    missing = 0
    for _, item in plan.iterrows():
        case_dir = run_root / str(item["case_id"])
        completion_path = case_dir / "completion.json"
        if completion_path.exists():
            completion = _read_json(completion_path)
            paths = {
                branch: Path(path)
                for branch, path in completion.get("branches", {}).items()
            }
        else:
            paths = {}
        required_branches = {
            "candidate",
            "no_control",
            "dynamic_internal_rules",
            "hold_previous",
        }
        if set(paths) != required_branches or not all(
            path.exists() for path in paths.values()
        ):
            missing += 1
            continue
        checkpoint = float(item["checkpoint_min"])
        full_detail = {
            name: pd.read_csv(path) for name, path in paths.items()
        }
        detail = {
            name: frame[
                (pd.to_numeric(frame["elapsed_min"], errors="coerce") >= checkpoint)
                & (
                    pd.to_numeric(frame["elapsed_min"], errors="coerce")
                    < checkpoint + 120.0
                )
            ].reset_index(drop=True)
            for name, frame in full_detail.items()
        }
        kpi = {
            name: compute_window_kpis(frame, priority, checkpoint, 120, 300)
            for name, frame in detail.items()
        }
        repeat_kpis = [kpi["candidate"]]
        for repeat_path_value in completion.get("candidate_repeat_paths", [])[1:]:
            repeat_path = Path(str(repeat_path_value))
            if repeat_path.exists():
                repeat_frame = _window(repeat_path, checkpoint)
                repeat_kpis.append(
                    compute_window_kpis(
                        repeat_frame, priority, checkpoint, 120, 300
                    )
                )
        repeat_noise = {
            metric: float(
                max(value[metric] for value in repeat_kpis)
                - min(value[metric] for value in repeat_kpis)
            )
            for metric in ("PFV", "TFV", "peak_TFV_rate")
        }
        authority = classify_action_authority(
            detail[action_authority_reference_name()],
            detail["candidate"],
            facilities,
        )
        schedule = np.asarray(json.loads(item["projected_schedule_json"]), dtype=float)
        anchor = np.asarray(
            json.loads(item["frozen_anchor_schedule_json"]), dtype=float
        )
        cost = schedule_action_cost(schedule, anchor)
        action_columns = [
            f"actual_setting:{facility_id}"
            for facility_id in facilities
            if f"actual_setting:{facility_id}" in detail["candidate"].columns
        ]
        requested_columns = [
            f"requested_setting:{facility_id}"
            for facility_id in facilities
            if f"requested_setting:{facility_id}" in detail["candidate"].columns
        ]
        candidate_full = full_detail["candidate"]
        action_frame = candidate_full[
            post_decision_readback_mask(
                candidate_full["elapsed_min"],
                checkpoint_min=checkpoint,
                decision_interval_min=10.0,
                sample_interval_min=5.0,
                horizon_min=120.0,
            )
        ].reset_index(drop=True)
        actual_values = action_frame[action_columns].to_numpy(float)
        requested_values = (
            action_frame[requested_columns].to_numpy(float)
            if len(requested_columns) == len(action_columns)
            else np.empty((0, 0))
        )
        target_columns = [
            f"target_setting:{facility_id}"
            for facility_id in facilities
            if f"target_setting:{facility_id}" in detail["candidate"].columns
        ]
        target_values = (
            action_frame[target_columns].to_numpy(float)
            if len(target_columns) == len(action_columns)
            else np.empty((0, 0))
        )
        readback_columns = [
            f"readback_setting:{facility_id}"
            for facility_id in facilities
            if f"readback_setting:{facility_id}" in detail["candidate"].columns
        ]
        readback_values = (
            action_frame[readback_columns].to_numpy(float)
            if len(readback_columns) == len(action_columns)
            else np.empty((0, 0))
        )
        readback_worst = (
            float(
                max(
                    np.nanmax(np.abs(actual_values - requested_values)),
                    np.nanmax(np.abs(readback_values - requested_values)),
                    np.nanmax(np.abs(target_values - requested_values)),
                )
            )
            if (
                actual_values.size
                and requested_values.shape == actual_values.shape
                and readback_values.shape == actual_values.shape
                and target_values.shape == actual_values.shape
            )
            else float("inf")
        )
        readback_ok = bool(readback_worst <= 1e-4)
        sampled_anchor = np.repeat(anchor, 2, axis=0)[: len(actual_values)]
        actual_k_by_sample = (
            (np.abs(actual_values - sampled_anchor) > 1e-6).sum(axis=1)
            if actual_values.size
            else np.asarray([], dtype=int)
        )
        actual_max_k = int(actual_k_by_sample.max()) if len(actual_k_by_sample) else 0

        state_hashes = {
            name: branch_state_hashes(
                frame,
                checkpoint_min=checkpoint,
                facility_ids=facilities,
            )
            for name, frame in full_detail.items()
        }
        prefix_history_hash_match = hashes_match_across_branches(
            state_hashes, ("prefix_history_sha256",)
        )
        checkpoint_pre_action_hash_match = hashes_match_across_branches(
            state_hashes, ("checkpoint_pre_action_sha256",)
        )
        state_hash_match = (
            prefix_history_hash_match and checkpoint_pre_action_hash_match
        )

        def residual_sequence(
            candidate_frame: pd.DataFrame,
            reference_frame: pd.DataFrame,
            columns: list[str],
        ) -> list[list[float]]:
            shared = [
                column
                for column in columns
                if column in candidate_frame.columns and column in reference_frame.columns
            ]
            if not shared:
                return []
            rows_count = min(len(candidate_frame), len(reference_frame))
            candidate_values = (
                candidate_frame.iloc[:rows_count][shared]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0.0)
                .to_numpy(float)
            )
            reference_values = (
                reference_frame.iloc[:rows_count][shared]
                .apply(pd.to_numeric, errors="coerce")
                .fillna(0.0)
                .to_numpy(float)
            )
            return (candidate_values - reference_values)[::2][:12].tolist()

        priority_columns = [
            f"h:{node_id}"
            for node_id in priority
            if f"h:{node_id}" in detail["candidate"].columns
        ]
        active_facilities = json.loads(item.get("active_facilities_json", "[]"))
        active_flow_columns = [
            f"flow:{facility_id}"
            for facility_id in active_facilities
            if f"flow:{facility_id}" in detail["candidate"].columns
        ]
        storage_columns = [
            column
            for column in detail["candidate"].columns
            if column.startswith("storage_volume:")
        ]
        summary_columns = [
            column
            for column in (
                "tfv_rate_m3s",
                "system_stored_volume_m3",
                "excess_fullness_mean",
                "excess_fullness_p95",
                "excess_fullness_fraction",
            )
            if column in detail["candidate"].columns
        ]
        rainfall_forecast = (
            pd.to_numeric(
                detail["candidate"].get("rainfall_mm_h", pd.Series(dtype=float)),
                errors="coerce",
            )
            .fillna(0.0)
            .to_numpy(float)[::2][:12]
            .tolist()
        )
        current_row = detail["candidate"].iloc[0]
        reconstructed_state = [
            float(value)
            for value in pd.to_numeric(
                current_row[
                    [
                        column
                        for column in detail["candidate"].columns
                        if column.startswith("h:")
                    ]
                ],
                errors="coerce",
            )
            .dropna()
            .to_numpy(float)
        ]
        runtime_context = v4_features.runtime_context_features(
            {
                "elapsed_min": max(
                    0.0, checkpoint - float(item.get("spinup_min", 0.0))
                ),
                "event_duration_min": max(
                    120.0,
                    _rainfall_duration_min(Path(str(item["rainfall_path"]))),
                ),
                "phase": str(current_row.get("phase", "")),
                "reconstructed_state": reconstructed_state,
                "rainfall_window": rainfall_forecast,
                "current_action": actual_values[0].tolist()
                if len(actual_values)
                else anchor[0].tolist(),
            }
        )
        schedule_delta = schedule - anchor
        changed_coordinates = np.any(np.abs(schedule_delta) > 1e-6, axis=0)
        active_count = int(changed_coordinates.sum())
        first_delta = schedule[0] - anchor[0]
        binary_ids = {"ADD301.2", "ADD301.3"}
        binary_indices = [
            facilities.index(facility_id)
            for facility_id in binary_ids
            if facility_id in facilities
        ]
        action_features = np.asarray(
            [
                active_count,
                active_count / max(1, len(facilities)),
                float(np.mean(first_delta > 1e-6)),
                float(np.mean(first_delta < -1e-6)),
                float(np.mean(np.abs(schedule_delta))),
                float(np.max(np.abs(schedule_delta))),
                float(
                    sum(
                        anchor[0, index] < 0.5 and schedule[0, index] >= 0.5
                        for index in binary_indices
                    )
                ),
                float(
                    sum(
                        anchor[0, index] >= 0.5 and schedule[0, index] < 0.5
                        for index in binary_indices
                    )
                ),
                float(str(item.get("anchor", "")).lower() == "dynamic_internal"),
            ],
            dtype=float,
        )
        deltas = {
            "delta_pfv_h120_vs_no_control_m3": (
                kpi["candidate"]["PFV"] - kpi["no_control"]["PFV"]
            ),
            "delta_tfv_h120_vs_dynamic_internal_m3": (
                kpi["candidate"]["TFV"] - kpi["dynamic_internal_rules"]["TFV"]
            ),
            "delta_peak_h120_vs_dynamic_internal_m3s": (
                kpi["candidate"]["peak_TFV_rate"]
                - kpi["dynamic_internal_rules"]["peak_TFV_rate"]
            ),
        }
        science = classify_candidate_result(
            deltas["delta_pfv_h120_vs_no_control_m3"],
            deltas["delta_tfv_h120_vs_dynamic_internal_m3"],
            deltas["delta_peak_h120_vs_dynamic_internal_m3s"],
            cost,
            minimum_tfv_improvement_m3=float(
                config["thresholds"]["materially_beneficial"][
                    "minimum_tfv_improvement_m3"
                ]
            ),
            minimum_benefit_cost_ratio=float(
                config["thresholds"]["materially_beneficial"][
                    "minimum_benefit_cost_ratio"
                ]
            ),
        )
        rows.append(
            {
                **item.to_dict(),
                **deltas,
                **science,
                **authority.to_dict(),
                "confirmed_flat": authority.authority_class
                in {
                    "B_realized_no_hydraulic_opportunity",
                    "D_realized_hydraulically_flat",
                },
                "action_cost": cost,
                "actual_schedule_sha256": hashlib.sha256(
                    np.round(actual_values, 8).tobytes()
                ).hexdigest(),
                "readback_ok": readback_ok,
                "readback_worst_abs": readback_worst,
                "actual_max_k": actual_max_k,
                "state_hash_match": state_hash_match,
                "prefix_history_hash_match": prefix_history_hash_match,
                "checkpoint_pre_action_hash_match": (
                    checkpoint_pre_action_hash_match
                ),
                "prefix_history_sha256": state_hashes["candidate"][
                    "prefix_history_sha256"
                ],
                "checkpoint_pre_action_state_sha256": state_hashes[
                    "candidate"
                ]["checkpoint_pre_action_sha256"],
                "post_actual_schedule_sha256": state_hashes["candidate"][
                    "post_actual_schedule_sha256"
                ],
                "post_readback_schedule_sha256": state_hashes["candidate"][
                    "post_readback_schedule_sha256"
                ],
                "initial_state_hash": state_hashes["candidate"][
                    "checkpoint_pre_action_sha256"
                ],
                "candidate_detail_path": str(paths["candidate"]),
                "no_control_detail_path": str(paths["no_control"]),
                "dynamic_internal_detail_path": str(
                    paths["dynamic_internal_rules"]
                ),
                "hold_previous_detail_path": str(paths["hold_previous"]),
                "rainfall_forecast_12step_json": json.dumps(
                    rainfall_forecast, separators=(",", ":")
                ),
                "priority_depth_residual_12step_json": json.dumps(
                    residual_sequence(
                        detail["candidate"],
                        detail["no_control"],
                        priority_columns,
                    ),
                    separators=(",", ":"),
                ),
                "sentinel_depth_residual_12step_json": json.dumps(
                    residual_sequence(
                        detail["candidate"],
                        detail["no_control"],
                        priority_columns,
                    ),
                    separators=(",", ":"),
                ),
                "active_link_flow_residual_12step_json": json.dumps(
                    residual_sequence(
                        detail["candidate"],
                        detail["dynamic_internal_rules"],
                        active_flow_columns,
                    ),
                    separators=(",", ":"),
                ),
                "storage_volume_residual_12step_json": json.dumps(
                    residual_sequence(
                        detail["candidate"],
                        detail["dynamic_internal_rules"],
                        storage_columns,
                    ),
                    separators=(",", ":"),
                ),
                "hydraulic_summary_residual_12step_json": json.dumps(
                    residual_sequence(
                        detail["candidate"],
                        detail["dynamic_internal_rules"],
                        summary_columns,
                    ),
                    separators=(",", ":"),
                ),
                "tfv_rate_residual_12step_json": json.dumps(
                    residual_sequence(
                        detail["candidate"],
                        detail["dynamic_internal_rules"],
                        ["tfv_rate_m3s"],
                    ),
                    separators=(",", ":"),
                ),
                "system_stored_volume_residual_12step_json": json.dumps(
                    residual_sequence(
                        detail["candidate"],
                        detail["dynamic_internal_rules"],
                        ["system_stored_volume_m3"],
                    ),
                    separators=(",", ":"),
                ),
                "excess_fullness_residual_12step_json": json.dumps(
                    residual_sequence(
                        detail["candidate"],
                        detail["dynamic_internal_rules"],
                        [
                            "excess_fullness_mean",
                            "excess_fullness_p95",
                            "excess_fullness_fraction",
                        ],
                    ),
                    separators=(",", ":"),
                ),
                "active_facility_count": int(len(active_facilities)),
                "repeat_count_completed": int(len(repeat_kpis)),
                "repeat_pfv_range_m3": repeat_noise["PFV"],
                "repeat_tfv_range_m3": repeat_noise["TFV"],
                "repeat_peak_range_m3s": repeat_noise["peak_TFV_rate"],
                "h120_complete": all(len(frame) >= 24 for frame in detail.values()),
                "label_validity_h120": True,
                "label_validity_full": False,
                "full_eligible": False,
                "delta_pfv_full_vs_no_control_m3": np.nan,
                "delta_tfv_full_vs_dynamic_internal_m3": np.nan,
                "delta_peak_full_vs_dynamic_internal_m3s": np.nan,
                **{
                    f"v4_ctx_{name}": float(value)
                    for name, value in zip(
                        v4_features.CONTEXT_FEATURE_NAMES, runtime_context
                    )
                },
                **{
                    f"v4_act_{name}": float(value)
                    for name, value in zip(
                        v4_features.ACTION_FEATURE_NAMES, action_features
                    )
                },
            }
        )
    all_samples = pd.DataFrame(rows)
    out = context["output_root"] / output_name
    out.parent.mkdir(parents=True, exist_ok=True)
    if len(all_samples):
        duplicate_mask = all_samples.duplicated(
            ["event_id", "checkpoint_min", "actual_schedule_sha256"],
            keep="first",
        )
        rejection_reasons: list[str] = []
        accepted_mask: list[bool] = []
        for index, sample in all_samples.iterrows():
            reasons = []
            if not bool(sample["h120_complete"]):
                reasons.append("h120_incomplete")
            if not bool(sample["readback_ok"]):
                reasons.append("readback_mismatch")
            if not bool(sample["state_hash_match"]):
                reasons.append("same_state_hash_mismatch")
            if int(sample["actual_max_k"]) > 8:
                reasons.append("actual_k_exceeds_8")
            if int(sample["actual_max_k"]) == 0:
                reasons.append("actual_no_op")
            if bool(duplicate_mask.loc[index]):
                reasons.append("duplicate_actual_schedule")
            rejection_reasons.append("|".join(reasons))
            accepted_mask.append(not reasons)
        all_samples["rejection_reason"] = rejection_reasons
        all_samples["quality_accepted"] = accepted_mask
        samples = all_samples[all_samples["quality_accepted"]].copy()
        rejected_samples = all_samples[~all_samples["quality_accepted"]].copy()
    else:
        samples = all_samples.copy()
        rejected_samples = all_samples.copy()
    samples.to_csv(out, index=False)
    rejected_path = out.with_name(f"{out.stem}_rejected.csv")
    rejected_samples.to_csv(rejected_path, index=False)
    accepted = len(samples)
    rejected = len(rejected_samples)
    duplicate_within_checkpoint = (
        int(
            all_samples.duplicated(
                ["event_id", "checkpoint_min", "actual_schedule_sha256"],
                keep=False,
            ).sum()
        )
        if len(all_samples)
        else 0
    )
    audit = {
        "planned": int(len(plan)),
        "accepted": int(accepted),
        "rejected": int(rejected),
        "pending": 0,
        "missing": int(missing),
        "accounting_closed": accounting_is_closed(
            len(plan), accepted, rejected, 0, missing
        ),
        "actual_unique_case_keys": int(accepted),
        "actual_duplicates_within_checkpoint": duplicate_within_checkpoint,
        "readback_complete": bool(samples["readback_ok"].all()) if accepted else False,
        "same_state_hash_complete": (
            bool(samples["state_hash_match"].all()) if accepted else False
        ),
        "max_actual_k": int(samples["actual_max_k"].max()) if accepted else 0,
        "informative_fraction": (
            float(samples["locally_responsive"].mean()) if accepted else 0.0
        ),
        "joint_noninferior": (
            int(samples["joint_noninferior"].sum()) if accepted else 0
        ),
        "materially_beneficial": (
            int(samples["materially_beneficial"].sum()) if accepted else 0
        ),
        "rejected_manifest": str(rejected_path),
    }
    _atomic_json(out.with_suffix(".audit.json"), audit)
    return samples, audit


def stage_audit_canary(context: dict) -> int:
    plan = context["output_root"] / "canary" / "canary_case_plan.csv"
    runs = context["output_root"] / "canary" / "runs"
    samples, audit = _audit_run_dataset(
        context, plan, runs, "canary/canary_sample_manifest.csv"
    )
    constraint_ok = True
    if plan.exists():
        for raw in pd.read_csv(plan)["constraint_audit_json"]:
            item = json.loads(raw)
            constraint_ok &= (
                item["max_k"] <= 8 and item["binary_ok"] and item["rate_ok"]
            )
    responsive_samples = samples[
        samples["checkpoint_role"].astype(str).str.startswith("responsive")
    ]
    low_control_samples = samples[
        samples["checkpoint_role"].eq("flat_action_probe")
    ]
    responsive_informative_fraction = (
        float(responsive_samples["locally_responsive"].mean())
        if len(responsive_samples)
        else 0.0
    )
    low_control_flat_fraction = (
        float(low_control_samples["confirmed_flat"].mean())
        if len(low_control_samples)
        else 0.0
    )
    confirmed_flat_fraction = float(samples["confirmed_flat"].mean())
    pilot_flat_min = float(
        context["config"]["thresholds"]["pilot"]["flat_fraction_min"]
    )
    pilot_flat_max = float(
        context["config"]["thresholds"]["pilot"]["flat_fraction_max"]
    )
    pass_gate = (
        audit["accounting_closed"]
        and audit["accepted"] > 0
        and audit["actual_duplicates_within_checkpoint"] == 0
        and audit["readback_complete"]
        and audit["same_state_hash_complete"]
        and audit["max_actual_k"] <= 8
        and constraint_ok
        and audit["informative_fraction"]
        >= float(
            context["config"]["thresholds"]["canary"][
                "minimum_informative_fraction"
            ]
        )
        and responsive_informative_fraction
        >= float(
            context["config"]["thresholds"]["canary"][
                "minimum_informative_fraction"
            ]
        )
        and confirmed_flat_fraction_is_in_range(
            samples, pilot_flat_min, pilot_flat_max
        )
        and bool(samples["pfv_noninferior"].any())
        and int(samples["repeat_count_completed"].max())
        >= int(context["config"]["thresholds"]["canary"]["repeat_count"])
    )
    status = canary_gate_status(pass_gate, audit["accepted"])
    return _status(
        context["output_root"],
        "AuditExcitationCanary",
        status,
        **audit,
        constraint_ok=bool(constraint_ok),
        anchor_discovery_required=not bool(
            samples["materially_beneficial"].any()
        ),
        responsive_informative_fraction=responsive_informative_fraction,
        low_control_confirmed_flat_fraction=low_control_flat_fraction,
        confirmed_flat_fraction=confirmed_flat_fraction,
        confirmed_flat_fraction_required=[
            pilot_flat_min,
            pilot_flat_max,
        ],
        numerical_repeat_max_range={
            **safe_repeat_noise_ranges(samples),
        },
    )


def _anchor_science_requirements(samples: pd.DataFrame) -> dict[str, bool]:
    return {
        "pfv_safe": bool(samples["pfv_noninferior"].astype(bool).any()),
        "tfv_improved": bool(
            (samples["delta_tfv_h120_vs_dynamic_internal_m3"] < 0).any()
        ),
        "peak_noninferior": bool(
            samples["peak_noninferior"].astype(bool).any()
        ),
        "joint_noninferior": bool(
            samples["joint_noninferior"].astype(bool).any()
        ),
        "materially_beneficial": bool(
            samples["materially_beneficial"].astype(bool).any()
        ),
        "hard_negative_pfv": bool(
            (
                (samples["delta_tfv_h120_vs_dynamic_internal_m3"] < 0)
                & ~samples["pfv_noninferior"].astype(bool)
            ).any()
        ),
        "hard_negative_peak": bool(
            (
                samples["pfv_noninferior"].astype(bool)
                & ~samples["peak_noninferior"].astype(bool)
            ).any()
        ),
        "hard_negative_tfv": bool(
            (
                (samples["delta_peak_h120_vs_dynamic_internal_m3s"] < 0)
                & ~samples["tfv_noninferior"].astype(bool)
            ).any()
        ),
    }


def stage_discover_anchors(
    context: dict, workers: int, limit: int, resume: bool
) -> int:
    canary_samples = (
        context["output_root"] / "canary" / "canary_sample_manifest.csv"
    )
    canary_plan = context["output_root"] / "canary" / "canary_case_plan.csv"
    if not canary_samples.exists() or not canary_plan.exists():
        return _status(
            context["output_root"],
            "DiscoverExactAnchors",
            "incomplete",
            reason="Canary evidence is missing",
        )
    samples = pd.read_csv(canary_samples)
    anchor_dir = context["output_root"] / "anchors"
    anchor_dir.mkdir(parents=True, exist_ok=True)
    material = samples[samples["materially_beneficial"].astype(bool)].copy()
    material.to_csv(anchor_dir / "seed_material_anchors.csv", index=False)
    seed_requirements = _anchor_science_requirements(samples)
    if all(seed_requirements.values()):
        samples.to_csv(anchor_dir / "exact_anchor_manifest.csv", index=False)
        return _status(
            context["output_root"],
            "DiscoverExactAnchors",
            "pass",
            reused_canary_evidence=int(len(samples)),
            seed_requirements=seed_requirements,
        )
    # Expand around the most responsive checkpoints with a larger family budget.
    plan = pd.read_csv(canary_plan)
    checkpoints = (
        plan[
            ["event_id", "checkpoint_min", "checkpoint_role", "source_detail"]
        ]
        .drop_duplicates()
        .query("checkpoint_role == 'responsive'")
    )
    anchor_plan = _write_candidate_plan(
        checkpoints,
        context,
        context["output_root"] / "anchors" / "anchor_case_plan.csv",
        max_candidates=100,
    )
    prior_hashes = set(plan["projected_schedule_sha256"].astype(str))
    anchor_plan = anchor_plan[
        ~anchor_plan["projected_schedule_sha256"].astype(str).isin(prior_hashes)
    ].copy()
    anchor_plan = _exact_local_search_plan(
        samples,
        anchor_plan,
        context,
        max_per_checkpoint=100,
    )
    anchor_plan = anchor_plan[
        ~anchor_plan["projected_schedule_sha256"].astype(str).isin(prior_hashes)
    ].copy()
    anchor_plan.to_csv(
        context["output_root"] / "anchors" / "anchor_case_plan.csv", index=False
    )
    result, code = _run_plan(
        context,
        context["output_root"] / "anchors" / "anchor_case_plan.csv",
        context["output_root"] / "anchors" / "runs",
        workers=workers,
        limit=limit,
        resume=resume,
    )
    progress_path = context["output_root"] / "anchors" / "runs" / "run_progress.json"
    progress = _read_json(progress_path) if progress_path.exists() else {}
    status = {
        EXIT_PASS: "pass",
        EXIT_INCOMPLETE: "incomplete",
        EXIT_RUNTIME_ERROR: "runtime_error",
    }.get(code, "runtime_error")
    return _status(
        context["output_root"],
        "DiscoverExactAnchors",
        status,
        planned=int(len(anchor_plan)),
        completed=int(len(result)),
        **progress,
        search_operators=[
            "coordinate_search",
            "stepwise_add",
            "leave_one_out",
            "beam_pair",
        ],
        seed_requirements=seed_requirements,
    )


def stage_audit_anchors(context: dict) -> int:
    anchor_dir = context["output_root"] / "anchors"
    canary_path = (
        context["output_root"] / "canary" / "canary_sample_manifest.csv"
    )
    if not canary_path.exists():
        return _status(
            context["output_root"],
            "AuditExactAnchors",
            "incomplete",
            reason="Canary sample manifest is missing",
        )
    evidence = [pd.read_csv(canary_path)]
    audit: dict = {"canary_evidence": int(len(evidence[0]))}
    plan = anchor_dir / "anchor_case_plan.csv"
    runs = anchor_dir / "runs"
    if plan.exists() and len(pd.read_csv(plan)):
        search_samples, search_audit = _audit_run_dataset(
            context,
            plan,
            runs,
            "anchors/exact_search_sample_manifest.csv",
        )
        evidence.append(search_samples)
        audit["exact_search"] = search_audit
    samples = pd.concat(evidence, ignore_index=True, sort=False).drop_duplicates(
        ["event_id", "checkpoint_min", "actual_schedule_sha256"]
    )
    samples.to_csv(anchor_dir / "exact_anchor_manifest.csv", index=False)
    required = _anchor_science_requirements(samples)
    status = "pass" if all(required.values()) else "scientific_fail"
    _atomic_json(
        context["output_root"] / "anchors" / "anchor_science_audit.json",
        {"status": status, "checks": required, **audit},
    )
    if status != "pass":
        by_checkpoint = (
            samples.groupby(["event_id", "checkpoint_min"])
            .agg(
                joint_noninferior=("joint_noninferior", "any"),
                materially_beneficial=("materially_beneficial", "any"),
                locally_responsive=("locally_responsive", "mean"),
                max_actual_k=("actual_max_k", "max"),
            )
            .reset_index()
            .to_dict(orient="records")
        )
        _atomic_json(
            context["output_root"]
            / "anchors"
            / "candidate_coverage_failure.json",
            {
                "status": "scientific_fail",
                "checks": required,
                "checkpoint_diagnosis": by_checkpoint,
                "possible_causes": [
                    "no_safe_intervention_opportunity_at_tested_state",
                    "candidate_family_coverage_insufficient",
                    "engineering_projection_removed_candidate",
                    "legacy_oracle_contract_not_equivalent",
                ],
                "k_was_not_expanded": True,
                "scientific_margins_were_not_relaxed": True,
            },
        )
    return _status(
        context["output_root"], "AuditExactAnchors", status, checks=required
    )


def stage_plan_pilot(context: dict) -> int:
    anchor_gate = (
        context["output_root"] / "stage_status" / "AuditExactAnchors.json"
    )
    if not anchor_gate.exists() or _read_json(anchor_gate)["exit_code"] != 0:
        return _status(
            context["output_root"],
            "PlanPilot",
            "blocked",
            reason="AuditExactAnchors must pass",
        )
    opportunities = pd.read_csv(
        context["output_root"] / "opportunity" / "control_opportunity_catalog.csv"
    )
    try:
        checkpoints = build_pilot_plan(opportunities)
    except ValueError as exc:
        return _status(
            context["output_root"], "PlanPilot", "incomplete", reason=str(exc)
        )
    plan = _write_candidate_plan(
        checkpoints,
        context,
        context["output_root"] / "pilot" / "pilot_case_plan.csv",
        max_candidates=15,
    )
    # Fifteen candidates per checkpoint gives 600 planned cases (inside the
    # frozen 300-700 Pilot budget) while preserving all family quotas.
    plan = plan.groupby(["event_id", "checkpoint_min"], group_keys=False).head(15)
    plan.to_csv(context["output_root"] / "pilot" / "pilot_case_plan.csv", index=False)
    status = "pass" if 300 <= len(plan) <= 700 else "incomplete"
    return _status(
        context["output_root"],
        "PlanPilot",
        status,
        planned=int(len(plan)),
        events=int(plan["event_id"].nunique()),
        checkpoints=int(plan.groupby(["event_id", "checkpoint_min"]).ngroups),
    )


def stage_build_dataset(
    context: dict, scope: str, plan_name: str, run_name: str
) -> int:
    plan = context["output_root"] / scope / plan_name
    runs = context["output_root"] / scope / run_name
    if not plan.exists():
        return _status(
            context["output_root"],
            f"Build{scope.title()}Dataset",
            "incomplete",
            reason="plan missing",
        )
    _, audit = _audit_run_dataset(
        context, plan, runs, f"{scope}/{scope}_sample_manifest.csv"
    )
    status = (
        "pass"
        if audit["accounting_closed"] and audit["missing"] == 0
        else "incomplete"
    )
    stage = "BuildPilotDataset" if scope == "pilot" else "BuildFormal1600"
    return _status(context["output_root"], stage, status, **audit)


def stage_audit_pilot(context: dict) -> int:
    path = context["output_root"] / "pilot" / "pilot_sample_manifest.csv"
    if not path.exists():
        return _status(
            context["output_root"], "AuditPilotDataset", "incomplete", reason="dataset missing"
        )
    data = pd.read_csv(path)
    responsive = data[data["checkpoint_role"] == "responsive"]
    by_checkpoint = responsive.groupby(["event_id", "checkpoint_min"])[
        "joint_noninferior"
    ].any()
    informative = float(data["locally_responsive"].mean())
    flat_fraction = float(data["confirmed_flat"].mean())
    low_control = data[
        data["checkpoint_role"].eq("flat_action_probe")
    ]
    responsive_data = data[
        data["checkpoint_role"].astype(str).str.startswith("responsive")
    ]
    responsive_informative = (
        float(responsive_data["locally_responsive"].mean())
        if len(responsive_data)
        else 0.0
    )
    low_control_flat = (
        float(low_control["confirmed_flat"].mean())
        if len(low_control)
        else 0.0
    )
    dead_zone = context["config"]["thresholds"]["dead_zone"]

    def three_class_signal(column: str, tolerance: float) -> bool:
        values = pd.to_numeric(data[column], errors="coerce")
        classes = np.where(
            values < -float(tolerance),
            "improved",
            np.where(values > float(tolerance), "degraded", "neutral"),
        )
        return len(set(classes.tolist())) == 3

    checkpoint_roles = (
        data[["event_id", "checkpoint_min", "checkpoint_role"]]
        .drop_duplicates()
        .groupby("event_id")["checkpoint_role"]
    )
    checks = {
        "accepted_300_to_700": 300 <= len(data) <= 700,
        "events_exactly_8": data["event_id"].nunique() == 8,
        "five_checkpoints_per_event": data.groupby("event_id")[
            "checkpoint_min"
        ].nunique().eq(5).all(),
        "four_responsive_one_low_control_per_event": bool(
            checkpoint_roles.apply(
                lambda roles: (
                    sum(str(role).startswith("responsive") for role in roles) == 4
                    and sum(
                        str(role) == "flat_action_probe"
                        for role in roles
                    )
                    == 1
                )
            ).all()
        ),
        "actual_unique_within_checkpoint": not data.duplicated(
            ["event_id", "checkpoint_min", "actual_schedule_sha256"]
        ).any(),
        "readback_complete": bool(data["readback_ok"].all()),
        "same_state_hash_complete": bool(data["state_hash_match"].all()),
        "actual_k_at_most_8": int(data["actual_max_k"].max()) <= 8,
        "responsive_informative_at_least_70pct": (
            responsive_informative >= 0.70
        ),
        "low_control_confirmed_flat_at_least_70pct": (
            low_control_flat >= 0.70
        ),
        "flat_fraction_10_to_20pct": 0.10 <= flat_fraction <= 0.20,
        "joint_checkpoint_fraction_at_least_30pct": (
            float(by_checkpoint.mean()) >= 0.30 if len(by_checkpoint) else False
        ),
        "pfv_improved_neutral_degraded_signal": three_class_signal(
            "delta_pfv_h120_vs_no_control_m3", dead_zone["pfv_m3"]
        ),
        "tfv_improved_neutral_degraded_signal": three_class_signal(
            "delta_tfv_h120_vs_dynamic_internal_m3", dead_zone["tfv_m3"]
        ),
        "peak_improved_neutral_degraded_signal": three_class_signal(
            "delta_peak_h120_vs_dynamic_internal_m3s", dead_zone["peak_m3s"]
        ),
    }
    status = "pass" if all(checks.values()) else "scientific_fail"
    _atomic_json(
        context["output_root"] / "pilot" / "pilot_dataset_audit.json",
        {"status": status, "checks": checks},
    )
    return _status(
        context["output_root"], "AuditPilotDataset", status, checks=checks
    )


def stage_train_baselines(context: dict, scope: str, stage: str) -> int:
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, r2_score
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    required_gate = (
        "AuditPilotDataset" if scope == "pilot" else "AuditFormal1600"
    )
    required_path = context["output_root"] / "stage_status" / f"{required_gate}.json"
    if not required_path.exists() or _read_json(required_path)["exit_code"] != 0:
        return _status(
            context["output_root"],
            stage,
            "blocked",
            reason=f"{required_gate} must pass",
        )
    data_path = context["output_root"] / scope / f"{scope}_sample_manifest.csv"
    if not data_path.exists():
        return _status(
            context["output_root"], stage, "incomplete", reason="dataset missing"
        )
    data = pd.read_csv(data_path)
    base_features = [
        "checkpoint_min",
        "action_cost",
        "active_facility_count",
    ]
    learned_features = sorted(
        column
        for column in data.columns
        if column.startswith(("v4_ctx_", "v4_act_"))
    )
    features = [
        *base_features,
        *learned_features,
        "family",
        "checkpoint_role",
    ]
    train_events = sorted(data["event_id"].astype(str).unique())
    validation_events = set(train_events[::5])
    train = data[~data["event_id"].astype(str).isin(validation_events)]
    validation = data[data["event_id"].astype(str).isin(validation_events)]
    if train.empty or validation.empty:
        return _status(
            context["output_root"], stage, "incomplete", reason="event-held-out split empty"
        )
    numerical = [
        column
        for column in features
        if column not in {"family", "checkpoint_role"}
    ]
    categorical = ["family", "checkpoint_role"]
    preprocess = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numerical,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "onehot",
                            OneHotEncoder(
                                handle_unknown="ignore", sparse_output=False
                            ),
                        ),
                    ]
                ),
                categorical,
            ),
        ]
    )
    reports: dict[str, dict] = {}
    for target in (
        "delta_pfv_h120_vs_no_control_m3",
        "delta_tfv_h120_vs_dynamic_internal_m3",
        "delta_peak_h120_vs_dynamic_internal_m3s",
    ):
        model = Pipeline([("preprocess", preprocess), ("model", Ridge(alpha=1.0))])
        model.fit(train[features], train[target])
        prediction = model.predict(validation[features])
        reports[f"ridge_{target}"] = {
            "r2": float(r2_score(validation[target], prediction)),
            "zero_r2": float(
                r2_score(validation[target], np.zeros(len(validation)))
            ),
        }
    for target in ("pfv_noninferior", "joint_noninferior"):
        if train[target].nunique() < 2 or validation[target].nunique() < 2:
            reports[f"logistic_{target}"] = {"blocked": "single_class"}
            continue
        for name, estimator in (
            ("logistic", LogisticRegression(max_iter=1000, class_weight="balanced")),
            ("gradient_boosting", HistGradientBoostingClassifier(max_iter=100)),
        ):
            model = Pipeline([("preprocess", preprocess), ("model", estimator)])
            model.fit(train[features], train[target].astype(int))
            prediction = model.predict(validation[features])
            reports[f"{name}_{target}"] = {
                "accuracy": float(accuracy_score(validation[target], prediction)),
                "balanced_accuracy": float(
                    balanced_accuracy_score(validation[target], prediction)
                ),
                "majority_accuracy": float(
                    validation[target].value_counts(normalize=True).max()
                ),
            }
    report_path = context["output_root"] / scope / f"{scope}_baseline_report.json"
    _atomic_json(report_path, {"reports": reports, "validation_events": sorted(validation_events)})
    return _status(
        context["output_root"], stage, "pass", report=str(report_path)
    )


def stage_evaluate_baselines(context: dict, scope: str, stage: str) -> int:
    required_stage = (
        "TrainPilotBaselines" if scope == "pilot" else "TrainV4Informative"
    )
    required_path = context["output_root"] / "stage_status" / f"{required_stage}.json"
    if not required_path.exists() or _read_json(required_path)["exit_code"] != 0:
        return _status(
            context["output_root"],
            stage,
            "blocked",
            reason=f"{required_stage} must pass",
        )
    report_path = context["output_root"] / scope / f"{scope}_baseline_report.json"
    if not report_path.exists():
        return _status(
            context["output_root"], stage, "incomplete", reason="baseline report missing"
        )
    reports = _read_json(report_path)["reports"]
    classification = [
        value
        for key, value in reports.items()
        if key.startswith(("logistic_", "gradient_boosting_"))
        and "balanced_accuracy" in value
    ]
    regression = [
        value
        for key, value in reports.items()
        if key.startswith("ridge_") and "r2" in value
    ]
    checks = {
        "classification_above_majority": bool(classification)
        and all(
            item["balanced_accuracy"] > 0.5
            and item["accuracy"] > item["majority_accuracy"]
            for item in classification
        ),
        "regression_above_zero": bool(regression)
        and all(item["r2"] > item["zero_r2"] for item in regression),
    }
    status = "pass" if all(checks.values()) else "scientific_fail"
    return _status(context["output_root"], stage, status, checks=checks)


def stage_plan_formal(context: dict) -> int:
    inventory_path = context["output_root"] / "inventory" / "event_inventory.csv"
    pilot_gate = context["output_root"] / "stage_status" / "EvaluatePilotGate.json"
    if not pilot_gate.exists() or _read_json(pilot_gate)["exit_code"] != 0:
        return _status(
            context["output_root"],
            "PlanFormal1600",
            "blocked",
            reason="EvaluatePilotGate must pass",
        )
    try:
        plan, partition = build_formal_1600_plan(pd.read_csv(inventory_path))
    except ValueError as exc:
        return _status(
            context["output_root"], "PlanFormal1600", "blocked", reason=str(exc)
        )
    formal = context["output_root"] / "formal1600"
    formal.mkdir(parents=True, exist_ok=True)
    partition.to_csv(formal / "event_partition.csv", index=False)
    # Case roles require an opportunity/active-learning result.  Do not invent
    # schedules: the role plan is intentionally separate and blocks RunFormal.
    plan.to_csv(formal / "formal1600_role_plan.csv", index=False)
    return _status(
        context["output_root"],
        "PlanFormal1600",
        "pass",
        role_rows=int(len(plan)),
        primary_events=64,
        reserve_events=16,
    )


def _formal_role_selection(
    candidate_plan: pd.DataFrame, pilot_samples: pd.DataFrame
) -> pd.DataFrame:
    """Select five planned roles using Pilot family evidence only."""
    family_stats = (
        pilot_samples.groupby("family")
        .agg(
            material_rate=("materially_beneficial", "mean"),
            joint_rate=("joint_noninferior", "mean"),
            mean_abs_pfv=("delta_pfv_h120_vs_no_control_m3", lambda x: np.mean(np.abs(x))),
            pfv_hard_negative_rate=(
                "pfv_noninferior",
                lambda x: 1.0 - np.mean(np.asarray(x, dtype=bool)),
            ),
            peak_failure_rate=(
                "peak_noninferior",
                lambda x: 1.0 - np.mean(np.asarray(x, dtype=bool)),
            ),
            support=("case_id", "count"),
        )
        .reset_index()
    )
    preferences = {
        "best_safe": family_stats.sort_values(
            ["material_rate", "joint_rate"], ascending=False
        )["family"].tolist(),
        "pfv_boundary": family_stats.sort_values("mean_abs_pfv")["family"].tolist(),
        "tfv_improved_pfv_unsafe": family_stats.sort_values(
            "pfv_hard_negative_rate", ascending=False
        )["family"].tolist(),
        "peak_degraded": family_stats.sort_values(
            "peak_failure_rate", ascending=False
        )["family"].tolist(),
        "uncertainty_or_coverage": family_stats.sort_values("support")[
            "family"
        ].tolist(),
    }
    rows: list[pd.Series] = []
    for _, group in candidate_plan.groupby(["event_id", "checkpoint_min"]):
        used: set[str] = set()
        for role, preferred_families in preferences.items():
            available = group[~group["projected_schedule_sha256"].isin(used)]
            chosen = None
            for family in preferred_families:
                matching = available[available["family"] == family]
                if not matching.empty:
                    chosen = matching.iloc[0]
                    break
            if chosen is None and not available.empty:
                chosen = available.iloc[0]
            if chosen is None:
                continue
            item = chosen.copy()
            item["candidate_role"] = role
            rows.append(item)
            used.add(str(chosen["projected_schedule_sha256"]))
    return pd.DataFrame(rows)


def stage_run_formal(
    context: dict, workers: int, limit: int, resume: bool
) -> int:
    formal = context["output_root"] / "formal1600"
    partition_path = formal / "event_partition.csv"
    pilot_path = context["output_root"] / "pilot" / "pilot_sample_manifest.csv"
    if not partition_path.exists() or not pilot_path.exists():
        return _status(
            context["output_root"],
            "RunFormal1600",
            "incomplete",
            reason="formal partition or passed Pilot dataset is missing",
        )
    partition = pd.read_csv(partition_path)
    inventory = pd.read_csv(
        context["output_root"] / "inventory" / "event_inventory.csv"
    ).set_index("event_id", drop=False)
    config, root = context["config"], context["root"]
    semantics = pd.read_csv(_resolve(root, config["project"]["facility_semantics"]))
    from sewerrtc.simulation.action_policies import attach_reference_nodes

    semantics = attach_reference_nodes(
        semantics.assign(
            actuator_id=semantics.get("actuator_id", semantics["facility_id"])
        ),
        _resolve(root, config["project"]["network"]),
    )
    record_nodes = set(_read_ids(_resolve(root, config["project"]["priority_nodes"])))
    for column in ("storage_id", "reference_node", "from_node", "to_node"):
        if column in semantics.columns:
            record_nodes.update(
                value
                for value in semantics[column].dropna().astype(str)
                if value and value.lower() != "nan"
            )
    baseline_root = formal / "opportunity_runs"
    payloads: list[dict] = []
    for _, event in partition.iterrows():
        event_id = str(event["event_id"])
        if event_id not in inventory.index:
            continue
        rainfall = Path(str(inventory.loc[event_id, "rainfall_path"]))
        if not rainfall.exists():
            continue
        payloads.append(
            {
                "event_id": event_id,
                "network": str(_resolve(root, config["project"]["network"])),
                "rainfall_path": str(rainfall),
                "rain_duration_min": _rainfall_duration_min(rainfall),
                "spinup_min": int(config["runtime"]["spinup_min"]),
                "facility_semantics": str(
                    _resolve(root, config["project"]["facility_semantics"])
                ),
                "priority_nodes": _read_ids(
                    _resolve(root, config["project"]["priority_nodes"])
                ),
                "record_node_ids": sorted(record_nodes),
                "out_dir": str(baseline_root / event_id),
                "resume": bool(resume),
            }
        )
    baseline_results: list[dict] = []
    with ProcessPoolExecutor(max_workers=min(max(1, int(workers)), 16)) as executor:
        futures = [
            executor.submit(_run_opportunity_worker, payload) for payload in payloads
        ]
        for future in as_completed(futures):
            baseline_results.append(future.result())
    baseline_manifest = pd.DataFrame(baseline_results)
    baseline_manifest.to_csv(formal / "opportunity_run_manifest.csv", index=False)
    good_paths = [
        Path(value)
        for value in baseline_manifest.loc[
            baseline_manifest["status"].eq("accepted"), "detail_path"
        ]
    ]
    facilities = _read_ids(_resolve(root, config["project"]["canonical_ids"]))
    opportunities = scan_existing_dynamic_internal(good_paths, facilities)
    if opportunities.empty:
        return _status(
            context["output_root"],
            "RunFormal1600",
            "runtime_error",
            reason="no opportunity baselines completed",
        )
    formal_max_by_event = opportunities.groupby("event_id")[
        "elapsed_min"
    ].transform("max")
    opportunities = opportunities[
        (opportunities["elapsed_min"] >= int(config["runtime"]["spinup_min"]) + 60)
        & (opportunities["elapsed_min"] <= formal_max_by_event - 120.0)
        & (opportunities["elapsed_min"] % 10.0 < 1e-6)
    ].rename(columns={"elapsed_min": "checkpoint_min"})
    responsive_threshold = float(config["opportunity"]["responsive_threshold"])
    weak_threshold = float(config["opportunity"]["weak_threshold"])
    opportunities["opportunity_class"] = np.where(
        opportunities["opportunity_score"] >= responsive_threshold,
        "responsive",
        np.where(
            opportunities["opportunity_score"] >= weak_threshold,
            "weakly_responsive",
            "flat",
        ),
    )
    opportunities["spinup_min"] = int(config["runtime"]["spinup_min"])
    opportunities.to_csv(formal / "formal_opportunity_catalog.csv", index=False)

    primary = partition[partition["split"] != "reserve"].copy()
    reserve = partition[partition["split"] == "reserve"].copy()
    valid: dict[str, pd.DataFrame] = {}
    for event_id, group in opportunities.groupby("event_id"):
        if group["opportunity_class"].eq("responsive").sum() >= 4:
            valid[str(event_id)] = group
    replacements: list[dict] = []
    reserve_ids = [
        event_id for event_id in reserve["event_id"].astype(str) if event_id in valid
    ]
    final_events: list[dict] = []
    for _, event in primary.iterrows():
        event_id = str(event["event_id"])
        if event_id in valid:
            final_events.append({"event_id": event_id, "split": event["split"]})
        elif reserve_ids:
            replacement = reserve_ids.pop(0)
            replacements.append(
                {
                    "replaced_event_id": event_id,
                    "replacement_event_id": replacement,
                    "inherited_split": event["split"],
                    "reason": "checkpoint_opportunity_quota_failed",
                }
            )
            final_events.append(
                {"event_id": replacement, "split": event["split"]}
            )
    pd.DataFrame(replacements).to_csv(
        formal / "reserve_replacements.csv", index=False
    )
    if len(final_events) != 64:
        return _status(
            context["output_root"],
            "RunFormal1600",
            "scientific_fail",
            valid_primary_or_replaced=len(final_events),
            reason="fewer than 64 events satisfy responsive/flat checkpoint quotas",
        )
    checkpoints: list[pd.DataFrame] = []
    split_map = {item["event_id"]: item["split"] for item in final_events}
    for item in final_events:
        group = valid[item["event_id"]]
        responsive = group[group["opportunity_class"] == "responsive"].nlargest(
            4, "opportunity_score"
        )
        responsive_minutes = set(responsive["elapsed_min"].astype(float))
        low_control = group[
            ~group["elapsed_min"].astype(float).isin(responsive_minutes)
        ].nsmallest(
            1,
            "opportunity_score",
        )
        chosen = pd.concat([responsive, low_control], ignore_index=True)
        chosen["checkpoint_role"] = [
            "responsive_1",
            "responsive_2",
            "responsive_3",
            "responsive_4",
            "flat_action_probe",
        ]
        checkpoints.append(chosen)
    checkpoint_plan = pd.concat(checkpoints, ignore_index=True)
    all_candidates = _write_candidate_plan(
        checkpoint_plan,
        context,
        formal / "formal1600_candidate_pool.csv",
        max_candidates=60,
    )
    selected = _formal_role_selection(all_candidates, pd.read_csv(pilot_path))
    selected["split"] = selected["event_id"].map(split_map)
    selected.to_csv(formal / "formal1600_case_plan.csv", index=False)
    if (
        len(selected) != 1600
        or selected["event_id"].nunique() != 64
        or not selected.groupby(["event_id", "checkpoint_min"]).size().eq(5).all()
    ):
        return _status(
            context["output_root"],
            "RunFormal1600",
            "incomplete",
            selected_cases=int(len(selected)),
            reason="could not materialize five unique candidates per checkpoint",
        )
    result, code = _run_plan(
        context,
        formal / "formal1600_case_plan.csv",
        formal / "runs",
        workers=workers,
        limit=limit,
        resume=resume,
    )
    progress_path = formal / "runs" / "run_progress.json"
    progress = _read_json(progress_path) if progress_path.exists() else {}
    status = {
        EXIT_PASS: "pass",
        EXIT_INCOMPLETE: "incomplete",
        EXIT_RUNTIME_ERROR: "runtime_error",
    }.get(code, "runtime_error")
    return _status(
        context["output_root"],
        "RunFormal1600",
        status,
        planned=1600,
        completed=int(len(result)),
        **progress,
        reference_cache_scope="event+checkpoint+dataset_contract_sha256",
    )


def stage_audit_formal(context: dict) -> int:
    path = context["output_root"] / "formal1600" / "formal1600_sample_manifest.csv"
    if not path.exists():
        return _status(
            context["output_root"],
            "AuditFormal1600",
            "incomplete",
            reason="formal dataset missing",
        )
    data = pd.read_csv(path)
    split_counts = data[["event_id", "split"]].drop_duplicates()["split"].value_counts()
    full_columns = [
        "delta_pfv_full_vs_no_control_m3",
        "delta_tfv_full_vs_dynamic_internal_m3",
        "delta_peak_full_vs_dynamic_internal_m3s",
    ]
    low_control = data[
        data["checkpoint_role"].eq("flat_action_probe")
    ]
    confirmed_flat_fraction = float(data["confirmed_flat"].mean())
    low_control_flat_fraction = (
        float(low_control["confirmed_flat"].mean())
        if len(low_control)
        else 0.0
    )
    checks = {
        "accepted_1600": len(data) == 1600,
        "events_64": data["event_id"].nunique() == 64,
        "split_counts_48_8_8": split_counts.to_dict()
        == {"train": 48, "model_validation": 8, "challenge": 8},
        "five_checkpoints_per_event": data.groupby("event_id")[
            "checkpoint_min"
        ].nunique().eq(5).all(),
        "five_candidates_per_checkpoint": data.groupby(
            ["event_id", "checkpoint_min"]
        ).size().eq(5).all(),
        "event_split_no_leakage": not data.groupby("event_id")["split"]
        .nunique()
        .gt(1)
        .any(),
        "readback_complete": bool(data["readback_ok"].all()),
        "same_state_complete": bool(data["state_hash_match"].all()),
        "actual_k_at_most_8": int(data["actual_max_k"].max()) <= 8,
        "actual_unique_within_checkpoint": not data.duplicated(
            ["event_id", "checkpoint_min", "actual_schedule_sha256"]
        ).any(),
        "confirmed_flat_fraction_10_to_20pct": (
            0.10 <= confirmed_flat_fraction <= 0.20
        ),
        "low_control_confirmed_flat_at_least_70pct": (
            low_control_flat_fraction >= 0.70
        ),
        "full_heads_disabled": bool(
            (~data["full_eligible"].astype(bool)).all()
            and (~data["label_validity_full"].astype(bool)).all()
            and all(
                column in data.columns
                and pd.to_numeric(data[column], errors="coerce").isna().all()
                for column in full_columns
            )
        ),
        "process_labels_present": bool(
            data["priority_depth_residual_12step_json"].astype(str).str.len().gt(2).all()
            and data["hydraulic_summary_residual_12step_json"]
            .astype(str)
            .str.len()
            .gt(2)
            .all()
        ),
    }
    status = "pass" if all(checks.values()) else "scientific_fail"
    _atomic_json(
        context["output_root"] / "formal1600" / "formal1600_dataset_audit.json",
        {"status": status, "checks": checks},
    )
    return _status(context["output_root"], "AuditFormal1600", status, checks=checks)


def _ridge_ensemble(
    x: np.ndarray,
    y: np.ndarray,
    event_ids: np.ndarray,
    seeds: list[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    from sewerrtc.prompt3.action_effect_mpc import _fit_ridge

    unique_events = np.unique(event_ids)
    members = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        sampled_events = rng.choice(
            unique_events, size=len(unique_events), replace=True
        )
        indices = np.concatenate(
            [np.flatnonzero(event_ids == event_id) for event_id in sampled_events]
        )
        weights, mean, scale, _ = _fit_ridge(x[indices], y[indices])
        members.append((weights, mean, scale))
    return (
        np.asarray([member[0] for member in members]),
        np.asarray([member[1] for member in members]),
        np.asarray([member[2] for member in members]),
    )


def _ensemble_predict(
    weights: np.ndarray, means: np.ndarray, scales: np.ndarray, x: np.ndarray
) -> np.ndarray:
    predictions = []
    design_ones = np.ones((len(x), 1), dtype=float)
    for weight, mean, scale in zip(weights, means, scales):
        normalised = (x - mean) / scale
        design = np.concatenate([design_ones, normalised], axis=1)
        predictions.append(design @ weight)
    return np.asarray(predictions)


def stage_train_informative(context: dict) -> int:
    audit_path = context["output_root"] / "stage_status" / "AuditFormal1600.json"
    data_path = (
        context["output_root"] / "formal1600" / "formal1600_sample_manifest.csv"
    )
    if not audit_path.exists() or _read_json(audit_path)["exit_code"] != 0:
        return _status(
            context["output_root"],
            "TrainV4Informative",
            "blocked",
            reason="AuditFormal1600 must pass",
        )
    data = pd.read_csv(data_path)
    from sewerrtc.prompt3 import action_effect_v4 as v4

    feature_names = [
        *(f"v4_ctx_{name}" for name in v4.CONTEXT_FEATURE_NAMES),
        *(f"v4_act_{name}" for name in v4.ACTION_FEATURE_NAMES),
    ]
    label_names = [
        "delta_pfv_h120_vs_no_control_m3",
        "delta_tfv_h120_vs_dynamic_internal_m3",
        "delta_peak_h120_vs_dynamic_internal_m3s",
    ]
    train = data[data["split"] == "train"].copy()
    validation = data[data["split"] == "model_validation"].copy()
    if train.empty or validation.empty:
        return _status(
            context["output_root"],
            "TrainV4Informative",
            "blocked",
            reason="frozen train/model-validation split is empty",
        )
    x_train = train[feature_names].fillna(0.0).to_numpy(float)
    y_train = train[label_names].to_numpy(float)
    seeds = [20260723, 20260724, 20260725, 20260726, 20260727]
    weights, means, scales = _ridge_ensemble(
        x_train, y_train, train["event_id"].astype(str).to_numpy(), seeds
    )
    x_validation = validation[feature_names].fillna(0.0).to_numpy(float)
    validation_members = _ensemble_predict(weights, means, scales, x_validation)
    validation_prediction = validation_members.mean(axis=0)
    validation_error = np.abs(
        validation_prediction - validation[label_names].to_numpy(float)
    )
    conformal = np.quantile(validation_error, 0.95, axis=0)

    def trajectory_matrix(frame: pd.DataFrame) -> np.ndarray:
        rows = []
        for _, row in frame.iterrows():
            priority_values = np.asarray(
                json.loads(row["priority_depth_residual_12step_json"]), dtype=float
            ).ravel()
            summary_values = np.asarray(
                json.loads(row["hydraulic_summary_residual_12step_json"]), dtype=float
            ).ravel()
            rows.append(np.concatenate([priority_values, summary_values]))
        width = max(len(row) for row in rows)
        return np.asarray(
            [np.pad(row, (0, width - len(row))) for row in rows], dtype=float
        )

    trajectory_train = trajectory_matrix(train)
    trajectory_weights, trajectory_means, trajectory_scales = _ridge_ensemble(
        x_train,
        trajectory_train,
        train["event_id"].astype(str).to_numpy(),
        seeds,
    )
    base_path = (
        context["root"]
        / "outputs"
        / "project6_dual_reference_v4"
        / "action_effect_models_v4"
        / "action_effect_dual_reference_v4.npz"
    )
    base_payload: dict[str, np.ndarray] = {}
    if base_path.exists():
        with np.load(base_path, allow_pickle=False) as base:
            for name in (
                "reference_weights",
                "reference_feature_mean",
                "reference_feature_scale",
                "reference_labels",
            ):
                if name in base:
                    base_payload[f"base_{name}"] = base[name]
    model_dir = context["output_root"] / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "action_effect_v4_informative.npz"
    np.savez(
        model_path,
        residual_weights=weights,
        residual_feature_mean=means,
        residual_feature_scale=scales,
        residual_labels=np.asarray(label_names),
        feature_names=np.asarray(feature_names),
        residual_conformal=conformal,
        trajectory_weights=trajectory_weights,
        trajectory_feature_mean=trajectory_means,
        trajectory_feature_scale=trajectory_scales,
        seeds=np.asarray(seeds),
        full_event_heads_enabled=np.asarray([False]),
        **base_payload,
    )
    report = {
        "status": "pass",
        "model_path": str(model_path),
        "model_sha256": _sha256(model_path),
        "train_events": int(train["event_id"].nunique()),
        "validation_events": int(validation["event_id"].nunique()),
        "challenge_events_touched": 0,
        "base_reference_head_preserved": bool(base_payload),
        "full_event_heads_enabled": False,
        "future_hydraulic_truth_features": 0,
    }
    _atomic_json(model_dir / "training_report.json", report)
    return _status(
        context["output_root"],
        "TrainV4Informative",
        "pass",
        model=str(model_path),
        base_reference_head_preserved=bool(base_payload),
    )


def stage_evaluate_informative(context: dict) -> int:
    model_path = (
        context["output_root"] / "models" / "action_effect_v4_informative.npz"
    )
    data_path = (
        context["output_root"] / "formal1600" / "formal1600_sample_manifest.csv"
    )
    if not model_path.exists() or not data_path.exists():
        return _status(
            context["output_root"],
            "EvaluateV4InformativeGate",
            "incomplete",
            reason="model or formal dataset missing",
        )
    data = pd.read_csv(data_path)
    challenge = data[data["split"] == "challenge"].copy()
    if challenge.empty:
        return _status(
            context["output_root"],
            "EvaluateV4InformativeGate",
            "incomplete",
            reason="frozen challenge split is empty",
        )
    with np.load(model_path, allow_pickle=False) as model:
        feature_names = model["feature_names"].astype(str).tolist()
        labels = model["residual_labels"].astype(str).tolist()
        members = _ensemble_predict(
            model["residual_weights"],
            model["residual_feature_mean"],
            model["residual_feature_scale"],
            challenge[feature_names].fillna(0.0).to_numpy(float),
        )
    prediction = members.mean(axis=0)
    truth = challenge[labels].to_numpy(float)
    thresholds = [0.70, 0.70, 0.80]
    dead_zones = [
        float(context["config"]["thresholds"]["dead_zone"]["pfv_m3"]),
        float(context["config"]["thresholds"]["dead_zone"]["tfv_m3"]),
        float(context["config"]["thresholds"]["dead_zone"]["peak_m3s"]),
    ]
    metrics: dict[str, dict] = {}
    checks: dict[str, bool] = {}
    for index, (label, threshold, dead_zone) in enumerate(
        zip(labels, thresholds, dead_zones)
    ):
        predicted_class = np.where(
            prediction[:, index] < -dead_zone,
            -1,
            np.where(prediction[:, index] > dead_zone, 1, 0),
        )
        truth_class = np.where(
            truth[:, index] < -dead_zone,
            -1,
            np.where(truth[:, index] > dead_zone, 1, 0),
        )
        correct = predicted_class == truth_class
        per_event = pd.DataFrame(
            {
                "event_id": challenge["event_id"].astype(str).to_numpy(),
                "correct": correct.astype(float),
            }
        ).groupby("event_id")["correct"].mean()
        overall = float(np.mean(correct))
        balanced = float(per_event.mean())
        class_counts = pd.Series(truth_class).value_counts(normalize=True)
        majority_baseline = float(class_counts.max())
        metrics[label] = {
            "three_class_direction_accuracy": overall,
            "event_balanced_direction_accuracy": balanced,
            "worst_event_direction_accuracy": float(per_event.min()),
            "majority_class_baseline": majority_baseline,
            "dead_zone": dead_zone,
            "ensemble_uncertainty_mean": float(
                members[:, :, index].std(axis=0).mean()
            ),
        }
        checks[label] = (
            overall >= threshold
            and balanced >= threshold
            and overall > majority_baseline
        )
    status = "pass" if all(checks.values()) else "scientific_fail"
    report = {
        "status": status,
        "checks": checks,
        "metrics": metrics,
        "challenge_events": int(challenge["event_id"].nunique()),
        "challenge_evaluated_once": True,
        "full_event_heads_enabled": False,
    }
    _atomic_json(
        context["output_root"] / "models" / "informative_model_gate.json",
        report,
    )
    return _status(
        context["output_root"],
        "EvaluateV4InformativeGate",
        status,
        checks=checks,
        report=str(
            context["output_root"] / "models" / "informative_model_gate.json"
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config_path = Path(args.config).resolve()
    try:
        context = _load_context(config_path)
        context["config_path"] = config_path
        stage = args.stage
        _acquire_writer_lock(context["output_root"], stage)
        if stage == "AuditContracts":
            return stage_audit_contracts(context)
        if stage == "ReauditExistingGate5":
            return stage_reaudit_existing(context)
        if stage == "BuildEventInventory":
            return stage_build_inventory(context)
        if stage == "ScanOpportunities":
            return stage_scan_opportunities(context, args.workers, args.resume)
        if stage == "PlanExcitationCanary":
            return stage_plan_canary(context)
        if stage == "PlanExactPrefixTiny":
            return stage_plan_exact_prefix_tiny(context)
        if stage == "RunExactPrefixTiny":
            return stage_run_named_plan(
                context,
                stage,
                context["output_root"] / "tiny" / "tiny_case_plan.csv",
                context["output_root"] / "tiny" / "runs",
                1,
                1,
                False,
            )
        if stage == "AuditExactPrefixTiny":
            return stage_audit_exact_prefix_tiny(context)
        if stage == "RunExcitationCanary":
            return stage_run_named_plan(
                context,
                stage,
                context["output_root"] / "canary" / "canary_case_plan.csv",
                context["output_root"] / "canary" / "runs",
                args.workers,
                args.limit,
                args.resume,
            )
        if stage == "AuditExcitationCanary":
            return stage_audit_canary(context)
        if stage == "DiscoverExactAnchors":
            return stage_discover_anchors(
                context, args.workers, args.limit, args.resume
            )
        if stage == "AuditExactAnchors":
            return stage_audit_anchors(context)
        if stage == "PlanPilot":
            return stage_plan_pilot(context)
        if stage == "RunPilot":
            return stage_run_named_plan(
                context,
                stage,
                context["output_root"] / "pilot" / "pilot_case_plan.csv",
                context["output_root"] / "pilot" / "runs",
                args.workers,
                args.limit,
                args.resume,
            )
        if stage == "BuildPilotDataset":
            return stage_build_dataset(
                context, "pilot", "pilot_case_plan.csv", "runs"
            )
        if stage == "AuditPilotDataset":
            return stage_audit_pilot(context)
        if stage == "TrainPilotBaselines":
            return stage_train_baselines(context, "pilot", stage)
        if stage == "EvaluatePilotGate":
            return stage_evaluate_baselines(context, "pilot", stage)
        if stage == "PlanFormal1600":
            return stage_plan_formal(context)
        if stage == "RunFormal1600":
            return stage_run_formal(
                context, args.workers, args.limit, args.resume
            )
        if stage == "BuildFormal1600":
            return stage_build_dataset(
                context, "formal1600", "formal1600_case_plan.csv", "runs"
            )
        if stage == "AuditFormal1600":
            return stage_audit_formal(context)
        if stage == "TrainV4Informative":
            return stage_train_informative(context)
        if stage == "EvaluateV4InformativeGate":
            return stage_evaluate_informative(context)
        return EXIT_BLOCKED
    except Gate5RLockError as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        return EXIT_BLOCKED
    except (FileNotFoundError, KeyError, ValueError) as exc:
        if "context" in locals():
            return _status(
                context["output_root"],
                args.stage,
                "blocked",
                reason=f"{type(exc).__name__}: {exc}",
            )
        print(f"BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_BLOCKED
    except Exception as exc:
        if "context" in locals():
            return _status(
                context["output_root"],
                args.stage,
                "runtime_error",
                reason=f"{type(exc).__name__}: {exc}",
            )
        print(f"RUNTIME ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return EXIT_RUNTIME_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
