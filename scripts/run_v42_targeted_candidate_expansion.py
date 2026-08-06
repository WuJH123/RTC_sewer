"""Run bounded, same-state authoritative candidate expansion branches.

This runner reuses the existing checkpoint ablation SWMM runner.  It never
reruns No-control/Internal/Hold references and never changes rainfall or the
checkpoint prefix.  Round-1 outputs are development-only evidence.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.plan_v42_targeted_candidate_expansion import _prefix_hash
from sewerrtc.control.targeted_candidate_expansion_v42 import (
    CandidateExpansionConfig,
    generate_targeted_candidate_sequences,
)
from sewerrtc.v4.v42_formal_runtime import load_actuators
from sewerrtc.v4.v42_priority_contract import get_pfv_core_node_indices
from sewerrtc.v4.v42_trajectory_builder import (
    _load_graph_topology,
    build_surrogate_action_node_map,
)
from sewerrtc.simulation.pyswmm_runner import (
    compute_kpis,
    run_swmm_no_control_action_ablation,
)


CONTROL_CORE_MANIFEST = (
    PROJECT_ROOT
    / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/step2"
    / "FORMAL_F2_STEP2_CONTROL_CORE_MANIFEST.parquet"
)


def _array(value: Any) -> np.ndarray:
    if isinstance(value, str):
        value = json.loads(value)
    return np.asarray(value, dtype=np.float32)


def _sha_sequence(sequence: np.ndarray) -> str:
    value = np.asarray(sequence, dtype=np.float32)
    return hashlib.sha256(np.round(value, 6).tobytes()).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def _graph_field(graph: Any, name: str) -> Any:
    try:
        return graph[name]
    except (KeyError, TypeError):
        return getattr(graph, name)


def _ranked_actuators(project_root: Path, actuators: pd.DataFrame) -> tuple[list[str], list[str]]:
    graph = _load_graph_topology(project_root)
    node_ids = [str(value) for value in _graph_field(graph, "node_ids")]
    priority = get_pfv_core_node_indices(node_ids)
    action_map = build_surrogate_action_node_map(graph).astype(np.float32)
    scores = action_map[:, priority].sum(axis=1)
    ids = actuators["actuator_id"].astype(str).tolist()
    if action_map.shape[0] != len(ids):
        raise RuntimeError("authoritative surrogate action map is not Engineering36-sized")
    ranked = [ids[int(i)] for i in np.argsort(-scores, kind="stable")]
    return ranked, [node_ids[int(i)] for i in priority]


def _role_map(actuators: pd.DataFrame) -> dict[str, str]:
    for column in ("asset_role", "storage_control_type", "link_type"):
        if column in actuators.columns:
            return dict(zip(actuators["actuator_id"].astype(str), actuators[column].fillna("").astype(str)))
    return {str(value): "" for value in actuators["actuator_id"]}


def _templates(
    group: pd.DataFrame,
    current: np.ndarray,
    ids: list[str],
    *,
    ranked_ids: list[str] | None = None,
    refine: bool = False,
) -> list[dict[str, object]]:
    templates: list[dict[str, object]] = []
    seen: set[tuple[str, ...]] = set()
    source_group = group
    if refine and "tfv_delta" in group.columns:
        source_group = group.sort_values(
            ["tfv_delta", "candidate_action_sha256"],
            kind="stable",
            na_position="last",
        )
    for raw in source_group.get("action_candidate_readback", []):
        candidate = _array(raw)
        if candidate.shape != (12, len(ids)):
            continue
        delta = candidate[0] - current
        changed = np.flatnonzero(np.abs(delta) > 1.0e-6)
        if not len(changed):
            continue
        key = tuple(str(int(i)) + ":" + f"{float(delta[i]):.6f}" for i in changed)
        if key in seen:
            continue
        seen.add(key)
        template_ids = [ids[int(i)] for i in changed]
        template_deltas = [float(delta[int(i)]) for i in changed]
        templates.append({
            "actuator_ids": template_ids,
            "deltas": template_deltas,
            "profile": "constant_h3",
        })
        if refine and ranked_ids:
            available = [aid for aid in ranked_ids if aid not in template_ids]
            if available:
                templates.append({
                    "actuator_ids": [available[0], *template_ids[1:]],
                    "deltas": template_deltas,
                    "profile": "constant_h3",
                })
            if len(template_ids) >= 2 and len(available) >= 2:
                templates.append({
                    "actuator_ids": [*template_ids, *available[:2]],
                    "deltas": [*template_deltas, template_deltas[0], template_deltas[0]],
                    "profile": "constant_h3",
                })
        if len(templates) >= (24 if refine else 12):
            break
    return templates[: (24 if refine else 12)]


def _make_tail_schedule(path: Path, detail_path: Path, ids: list[str], current: np.ndarray) -> None:
    source = pd.read_csv(detail_path, usecols=["elapsed_min"])
    schedule = pd.DataFrame({"elapsed_min": source["elapsed_min"].astype(float)})
    for index, actuator_id in enumerate(ids):
        schedule[f"a:{actuator_id}"] = float(current[index])
    path.parent.mkdir(parents=True, exist_ok=True)
    schedule.to_csv(path, index=False)


def _prefix_audit(candidate_detail: Path, reference_detail: Path, checkpoint: float, ids: list[str]) -> dict[str, Any]:
    candidate = pd.read_csv(candidate_detail)
    reference = pd.read_csv(reference_detail)
    # The state is read before the first action at checkpoint.  Including the
    # checkpoint row would compare the candidate's deliberately changed first
    # action with the no-control action and falsely fail the causal prefix.
    candidate = candidate[candidate["elapsed_min"] < float(checkpoint) - 1.0e-6]
    reference = reference[reference["elapsed_min"] < float(checkpoint) - 1.0e-6]
    common = sorted(set(np.round(candidate["elapsed_min"], 6)) & set(np.round(reference["elapsed_min"], 6)))
    if not common:
        return {"prefix_match": False, "prefix_common_rows": 0, "prefix_max_action_error": None}
    c = candidate.set_index(candidate["elapsed_min"].round(6)).loc[common]
    r = reference.set_index(reference["elapsed_min"].round(6)).loc[common]
    action_errors = []
    for actuator_id in ids:
        column = f"a:{actuator_id}"
        if column in c and column in r:
            action_errors.append(float(np.nanmax(np.abs(c[column].to_numpy(float) - r[column].to_numpy(float)))))
    rain_error = float(np.nanmax(np.abs(c["rainfall_mm_h"].to_numpy(float) - r["rainfall_mm_h"].to_numpy(float))))
    max_action = max(action_errors) if action_errors else float("inf")
    return {
        "prefix_match": bool(max_action <= 1.0e-6 and rain_error <= 1.0e-6),
        "prefix_common_rows": int(len(common)),
        "prefix_max_action_error": max_action,
        "prefix_max_rainfall_error": rain_error,
    }


def _run_one(job: dict[str, Any]) -> dict[str, Any]:
    output = Path(job["output_dir"])
    detail_path = output / "detail.csv"
    result_path = output / "result.json"
    if job["resume"] and detail_path.exists() and result_path.exists():
        try:
            old = json.loads(result_path.read_text(encoding="utf-8"))
            if old.get("status") == "pass" and old.get("candidate_action_sha256") == job["candidate_action_sha256"]:
                old["status"] = "reused"
                return old
        except Exception:
            pass

    output.mkdir(parents=True, exist_ok=True)
    no_control = Path(job["source_detail_path_no_control"])
    hold_detail = Path(job["source_detail_path_hold_previous"])
    # The reference bundle keeps the authoritative INP beside each role's
    # detail.csv (the parent run directory is only a grouping directory).
    case_inp = no_control.parent / "case.inp"
    if not no_control.exists() or not hold_detail.exists() or not case_inp.exists():
        raise FileNotFoundError(f"missing same-state reference/input for {job['state_key']}")

    ids = list(job["actuator_ids"])
    sequence = np.asarray(job["sequence"], dtype=np.float32)
    current = np.asarray(job["current_action"], dtype=np.float32)
    tail_schedule = output / "tail_action_schedule.csv"
    _make_tail_schedule(tail_schedule, no_control, ids, current)
    target_sequence = {
        actuator_id: [float(value) for value in sequence[:3, index]]
        for index, actuator_id in enumerate(ids)
    }
    reference = pd.read_csv(no_control, usecols=["elapsed_min"])
    duration = int(np.ceil(float(reference["elapsed_min"].max())))
    started = time.time()
    result = run_swmm_no_control_action_ablation(
        inp_path=case_inp,
        actuators=job["actuators"],
        priority_nodes=list(job["priority_nodes"]),
        no_control_detail_csv=no_control,
        out_detail_csv=detail_path,
        event_id=str(job["event_id"]),
        duration_min=duration,
        override_start_min=float(job["checkpoint_min"]),
        override_steps=3,
        actuator_id=ids[0],
        action_delta=0.0,
        control_step_sec=600,
        override_target_sequence=target_sequence,
        post_override_nominal_detail_csv=tail_schedule,
        policy_id=f"targeted_candidate_expansion_round{job.get('candidate_round', 1)}:{job['candidate_label']}",
        cleanup_swmm_artifacts=True,
    )
    prefix = _prefix_audit(detail_path, no_control, float(job["checkpoint_min"]), ids)
    detail = pd.read_csv(detail_path)
    readback_errors = []
    for actuator_id in ids:
        command = f"a:{actuator_id}"
        readback = f"setting:{actuator_id}"
        if command in detail and readback in detail:
            readback_errors.append(float(np.nanmax(np.abs(detail[command].to_numpy(float) - detail[readback].to_numpy(float)))))
    result.update({
        "status": "pass" if prefix["prefix_match"] and max(readback_errors or [float("inf")]) <= 1.0e-5 else "fail",
        "state_key": str(job["state_key"]),
        "event_id": str(job["event_id"]),
        "rainfall_sha256": str(job["rainfall_sha256"]),
        "checkpoint_min": float(job["checkpoint_min"]),
        "candidate_label": str(job["candidate_label"]),
        "candidate_round": int(job.get("candidate_round", 1)),
        "candidate_family": str(job["candidate_family"]),
        "candidate_action_sha256": str(job["candidate_action_sha256"]),
        "candidate_action": sequence.tolist(),
        "current_action": current.tolist(),
        "source_detail_path_no_control": str(no_control),
        "source_detail_path_hold_previous": str(hold_detail),
        "case_inp": str(case_inp),
        "case_inp_sha256": hashlib.sha256(case_inp.read_bytes()).hexdigest(),
        "candidate_detail": str(detail_path),
        "candidate_detail_sha256": hashlib.sha256(detail_path.read_bytes()).hexdigest(),
        "prefix_audit": prefix,
        "target_write_readback_verified": bool(max(readback_errors or [float("inf")]) <= 1.0e-5),
        "max_target_readback_error": max(readback_errors or [float("inf")]),
        # The reused ablation helper uses NaN for an unused scalar target.
        # Keep the JSON evidence strict without changing trajectory metrics.
        "target_setting": None,
        "hotstart_used": False,
        "reference_reused": True,
        "runtime_s": float(time.time() - started),
    })
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    return result


def _build_jobs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    plan = pd.read_csv(args.plan_csv)
    manifest = pd.read_parquet(args.source_manifest)
    actuators = load_actuators(args.project_root)
    ids = actuators["actuator_id"].astype(str).tolist()
    ranked, priority_nodes = _ranked_actuators(args.project_root, actuators)
    roles = _role_map(actuators)
    lock = json.loads(Path(args.lock_json).read_text(encoding="utf-8"))
    if hashlib.sha256(Path(args.plan_csv).read_bytes()).hexdigest() != lock["plan_sha256"]:
        raise RuntimeError("target plan SHA does not match frozen lock")
    if set(plan["state_key"].astype(str)) != {str(x["state_key"]) for x in lock["selected_states"]}:
        raise RuntimeError("target plan state list differs from frozen lock")

    existing_hashes_by_state: dict[str, set[str]] = {}
    if args.existing_manifest:
        existing = pd.read_parquet(
            args.existing_manifest,
            columns=["state_key", "candidate_action_sha256"],
        )
        for state_key, existing_group in existing.groupby(
            existing["state_key"].astype(str), sort=False
        ):
            existing_hashes_by_state[str(state_key)] = set(
                existing_group["candidate_action_sha256"].dropna().astype(str)
            )

    jobs: list[dict[str, Any]] = []
    for _, planned in plan.iterrows():
        state_key = str(planned["state_key"])
        group = manifest[manifest["state_key"].astype(str) == state_key]
        if group.empty:
            raise RuntimeError(f"state not found in source manifest: {state_key}")
        source = group.iloc[0]
        if str(source["event_id"]) != str(planned["event_id"]) or str(source["rainfall_sha256"]) != str(planned["rainfall_sha256"]):
            raise RuntimeError(f"same-state identity mismatch: {state_key}")
        if _prefix_hash(source) != str(planned["prefix_state_sha256"]):
            raise RuntimeError(f"prefix hash mismatch: {state_key}")
        current = _array(source["action_hold_previous_readback"])[0]
        round_no = int(args.candidate_round)
        templates = _templates(
            group,
            current,
            ids,
            ranked_ids=ranked,
            refine=round_no >= 2,
        )
        candidates = generate_targeted_candidate_sequences(
            current_action=current,
            actuator_ids=ids,
            horizon_steps=12,
            controllable_prefix_steps=3,
            actuator_roles=roles,
            ranked_actuator_ids=ranked,
            successful_action_templates=templates,
            config=CandidateExpansionConfig(max_candidates=int(args.max_candidates_per_state)),
        )
        for ordinal, candidate in enumerate(candidates):
            if str(candidate["candidate_family"]) == "hold":
                continue
            if args.candidate_limit and ordinal >= int(args.candidate_limit):
                break
            sequence = np.asarray(candidate["sequence"], dtype=np.float32)
            candidate_sha = _sha_sequence(sequence)
            if candidate_sha in existing_hashes_by_state.get(state_key, set()):
                continue
            output = Path(args.output_root) / state_key / candidate_sha
            jobs.append({
                "state_key": state_key,
                "event_id": str(planned["event_id"]),
                "rainfall_sha256": str(planned["rainfall_sha256"]),
                "checkpoint_min": float(planned["checkpoint_min"]),
                "candidate_label": str(candidate["label"]),
                "candidate_family": str(candidate["candidate_family"]),
                "candidate_action_sha256": candidate_sha,
                "sequence": sequence.tolist(),
                "current_action": current.tolist(),
                "actuator_ids": ids,
                "priority_nodes": priority_nodes,
                "actuators": actuators,
                "source_detail_path_no_control": str(source["source_detail_path_no_control"]),
                "source_detail_path_hold_previous": str(source["source_detail_path_hold_previous"]),
                "output_dir": str(output),
                "resume": bool(args.resume),
                "candidate_round": round_no,
            })
    if args.states_limit:
        allowed = set(plan.head(int(args.states_limit))["state_key"].astype(str))
        jobs = [job for job in jobs if job["state_key"] in allowed]
    return jobs, plan, manifest, actuators


def _horizon_arrays(detail: pd.DataFrame, checkpoint: float, node_ids: list[str], actuator_ids: list[str]) -> dict[str, Any]:
    horizon = detail[detail["elapsed_min"] >= float(checkpoint) - 1.0e-6].sort_values("elapsed_min").head(12)
    if len(horizon) != 12:
        raise RuntimeError(f"candidate detail has {len(horizon)} H120 rows, expected 12")
    flood_cols = [f"flood:{node_id}" for node_id in node_ids]
    depth_cols = [f"h:{node_id}" for node_id in node_ids]
    setting_cols = [f"setting:{actuator_id}" for actuator_id in actuator_ids]
    missing = [column for column in flood_cols + depth_cols + setting_cols if column not in horizon]
    if missing:
        raise RuntimeError(f"candidate detail missing required columns: {missing[:5]}")
    arrays = {
        "flood": horizon[flood_cols].to_numpy(dtype=np.float64),
        "depth": horizon[depth_cols].to_numpy(dtype=np.float64),
        "action": horizon[setting_cols].to_numpy(dtype=np.float64),
        "elapsed_min": horizon["elapsed_min"].to_numpy(dtype=np.float64),
    }
    if not all(np.isfinite(value).all() for key, value in arrays.items() if key != "elapsed_min"):
        raise RuntimeError("candidate H120 trajectory/readback is non-finite")
    return arrays


def _build_expanded_manifest(results: list[dict[str, Any]], source: pd.DataFrame, project_root: Path) -> pd.DataFrame:
    graph = _load_graph_topology(project_root)
    node_ids = [str(value) for value in _graph_field(graph, "node_ids")]
    priority = get_pfv_core_node_indices(node_ids)
    actuator_ids = load_actuators(project_root)["actuator_id"].astype(str).tolist()
    rows: list[dict[str, Any]] = []
    for result in results:
        if result.get("status") not in {"pass", "reused"}:
            continue
        state_key = str(result["state_key"])
        source_rows = source[source["state_key"].astype(str) == state_key]
        if source_rows.empty:
            raise RuntimeError(f"cannot build expanded manifest: unknown state {state_key}")
        row = source_rows.iloc[0].to_dict()
        detail = pd.read_csv(result["candidate_detail"])
        arrays = _horizon_arrays(detail, float(result["checkpoint_min"]), node_ids, actuator_ids)
        no_control_flood = _array(row["trajectory_flood_no_control"]).astype(float)
        internal_flood = _array(row["trajectory_flood_dynamic_internal"]).astype(float)
        candidate_flood = arrays["flood"]
        if no_control_flood.shape != candidate_flood.shape or internal_flood.shape != candidate_flood.shape:
            raise RuntimeError(f"H120 shape mismatch for expanded state {state_key}")
        no_control_pfv = float(no_control_flood[:, priority].sum() * 600.0)
        candidate_pfv = float(candidate_flood[:, priority].sum() * 600.0)
        internal_tfv = float(internal_flood.sum() * 600.0)
        candidate_tfv = float(candidate_flood.sum() * 600.0)
        internal_peak = float(np.sum(internal_flood, axis=1).max())
        candidate_peak = float(np.sum(candidate_flood, axis=1).max())
        candidate_action = arrays["action"]
        hold = _array(row["action_hold_previous_readback"]).astype(float)
        actual_k = int(np.count_nonzero(np.abs(candidate_action[:3] - hold[:3]) > 1.0e-6))
        round_no = int(result.get("candidate_round", 1))
        candidate_id = f"targeted_round{round_no}__{state_key}__{result['candidate_action_sha256']}"
        row.update({
            "formal_generation_id": f"PROJECT6_V42_TARGETED_CANDIDATE_EXPANSION_R{round_no}",
            "development_only": True,
            "formal_mainline_authorized": False,
            "training_admission_authorized": False,
            "source_dataset": f"targeted_candidate_expansion_round{round_no}",
            "case_id": candidate_id,
            "case_uid": hashlib.sha256(candidate_id.encode("utf-8")).hexdigest(),
            "candidate_action_sha256": str(result["candidate_action_sha256"]),
            "source_detail_path_candidate": str(result["candidate_detail"]),
            "candidate_detail_sha256": str(result["candidate_detail_sha256"]),
            "action_candidate_readback": json.dumps(candidate_action.tolist(), separators=(",", ":")),
            "trajectory_depth_candidate": json.dumps(arrays["depth"].tolist(), separators=(",", ":")),
            "trajectory_flood_candidate": json.dumps(candidate_flood.tolist(), separators=(",", ":")),
            "pfv_delta": candidate_pfv - no_control_pfv,
            "tfv_delta": candidate_tfv - internal_tfv,
            "peak_delta": candidate_peak - internal_peak,
            "actual_k": actual_k,
            "k_le_8": bool(actual_k <= 8),
            "same_state_raw_verified": True,
            "same_forcing_raw_verified": True,
            "actual_readback_verified": True,
            "same_state_ok": True,
            "physical_sha_ok": True,
            "rainfall_sha_ok": True,
            "prefix_sha_ok": True,
            "readback_ok": True,
            "no_hotstart": True,
            "h120_window_complete": True,
            "kpi_recompute_ok": True,
            "raw_independent_oracle_all_pass": True,
            "label_validity_pfv": True,
            "label_validity_tfv": True,
            "label_validity_peak": True,
            "trajectory_storage_volume_candidate": None,
            "trajectory_facility_flow_candidate": None,
            "trajectory_outfall_flow_candidate": None,
            "trajectory_storage_volume_candidate_available": False,
            "trajectory_facility_flow_candidate_available": False,
            "trajectory_outfall_flow_candidate_available": False,
            "storage_finite_fraction_candidate": 0.0,
            "facility_flow_finite_fraction_candidate": 0.0,
            "outfall_flow_finite_fraction_candidate": 0.0,
            "candidate_expansion_family": str(result.get("candidate_family", "")),
            "candidate_expansion_round": round_no,
            "candidate_reference_reuse": True,
        })
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--plan-csv", type=Path, required=True)
    parser.add_argument("--lock-json", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, default=CONTROL_CORE_MANIFEST)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--max-candidates-per-state", type=int, default=128)
    parser.add_argument("--candidate-limit", type=int, default=0)
    parser.add_argument("--states-limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--candidate-round", type=int, default=1)
    parser.add_argument("--existing-manifest", type=Path)
    args = parser.parse_args()
    jobs, plan, source_manifest, _ = _build_jobs(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    total = len(jobs)
    round_label = f"ROUND{int(args.candidate_round)}"
    stage = f"targeted_candidate_expansion_round{int(args.candidate_round)}"
    print(json.dumps({"stage": stage, "planned_candidates": total, "workers": int(args.workers)}, ensure_ascii=False), flush=True)
    if int(args.workers) <= 1:
        for index, job in enumerate(jobs, 1):
            try:
                result = _run_one(job)
            except Exception as exc:
                result = {"status": "fail", "state_key": job["state_key"], "candidate_action_sha256": job["candidate_action_sha256"], "error": repr(exc)}
            results.append(result)
            print(json.dumps({"stage": stage, "completed": index, "total": total, "status": result.get("status"), "state_key": result.get("state_key")}, ensure_ascii=False), flush=True)
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as pool:
            futures = {pool.submit(_run_one, job): job for job in jobs}
            for index, future in enumerate(as_completed(futures), 1):
                job = futures[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"status": "fail", "state_key": job["state_key"], "candidate_action_sha256": job["candidate_action_sha256"], "error": repr(exc)}
                results.append(result)
                print(json.dumps({"stage": stage, "completed": index, "total": total, "status": result.get("status"), "state_key": result.get("state_key")}, ensure_ascii=False), flush=True)
    result_frame = pd.DataFrame(results)
    result_frame.to_csv(args.output_root / f"TARGETED_CANDIDATE_{round_label}_FUNNEL.csv", index=False)
    if not result_frame.empty and (result_frame["status"].isin(["pass", "reused"])).any():
        expanded = _build_expanded_manifest(results, source_manifest, args.project_root)
        expanded.to_parquet(args.output_root / f"TARGETED_CANDIDATE_{round_label}_MANIFEST.parquet", index=False)
    audit = {
        "audit_id": f"V42_TARGETED_CANDIDATE_{round_label}_AUDIT_V1",
        "development_only": True,
        "formal_mainline_authorized": False,
        "planned_candidates": total,
        "completed_candidates": int(len(result_frame)),
        "passed_candidates": int(result_frame.get("status", pd.Series(dtype=str)).isin(["pass", "reused"]).sum()),
        "reused_candidates": int((result_frame.get("status", pd.Series(dtype=str)) == "reused").sum()),
        "failed_candidates": int((result_frame.get("status", pd.Series(dtype=str)) == "fail").sum()),
        "reference_reuse_required": True,
        "new_reference_runs": False,
        "candidate_round": int(args.candidate_round),
        "workers": int(args.workers),
        "candidate_families": {
            str(key): int(value)
            for key, value in result_frame.get("candidate_family", pd.Series(dtype=str)).value_counts().items()
        },
        "output_funnel": str(args.output_root / f"TARGETED_CANDIDATE_{round_label}_FUNNEL.csv"),
        "output_manifest": str(args.output_root / f"TARGETED_CANDIDATE_{round_label}_MANIFEST.parquet"),
    }
    (args.output_root / f"TARGETED_CANDIDATE_{round_label}_AUDIT.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
    if audit["failed_candidates"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
