from __future__ import annotations

"""Canonicalize the frozen 36-action encoding and prepare, but never run, data plans."""

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch

from sewerrtc.control.actuator_scope import select_actuators_for_scope
from sewerrtc.control.canonical_action_order import CanonicalActionOrder
from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config


SCHEMA_VERSION = "project6_36_canonical_joint_plan_v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _npz_member(path: Path, member: str) -> np.ndarray:
    with zipfile.ZipFile(path) as archive:
        return np.load(io.BytesIO(archive.read(member)), allow_pickle=True)


def _ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")]


def _write_stage(stage_dir: Path, stage: str, hashes: dict, expected: list[Path], status: str, note: str) -> None:
    ensure_dir(stage_dir)
    (stage_dir / "input_hashes.json").write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    (stage_dir / "stage_manifest.json").write_text(json.dumps({"stage": stage, "schema_version": SCHEMA_VERSION, "expected_outputs": [str(x) for x in expected], "key": _json_hash(hashes)}, indent=2), encoding="utf-8")
    (stage_dir / "stage_status.json").write_text(json.dumps({"stage": stage, "status": status, "schema_version": SCHEMA_VERSION, "note": note}, indent=2), encoding="utf-8")
    pd.DataFrame(columns=["case_id", "status"]).to_csv(stage_dir / "completed_case_ids.csv", index=False)
    pd.DataFrame(columns=["case_id", "error"]).to_csv(stage_dir / "failures.csv", index=False)


def _semantics(cfg: dict, canonical: CanonicalActionOrder) -> pd.DataFrame:
    audit = pd.read_csv(cfg_path(cfg, "outputs.audit") / "actuator_table.csv")
    assets = pd.read_csv(cfg_path(cfg, "network.retrofit_asset_manifest"))
    asset_map = assets.set_index("actuator_id").to_dict("index")
    by_id = audit.set_index("actuator_id")
    rows = []
    for canonical_position, aid in enumerate(canonical.canonical_ids):
        row = by_id.loc[aid]
        extra = asset_map.get(aid, {})
        link_type = str(row["link_type"]).lower()
        if aid in {"ADD301.2", "ADD301.3"}:
            semantic, lo, hi, binary = "binary_pump", 0.0, 1.0, True
        elif aid == "add350.1":
            semantic, lo, hi, binary = "continuous_pump", 0.0, 1.0, False
        elif "storage" in str(extra.get("asset_class", "")):
            semantic, lo, hi, binary = str(extra["control_semantics"]), 0.0, 1.0, False
        else:
            semantic, lo, hi, binary = ("continuous_weir" if link_type == "weir" else "continuous_orifice"), 0.0, 1.0, False
        rows.append({
            "canonical_position": canonical_position,
            "actuator_id": aid,
            "global_109_index": canonical.canonical_global_indices[canonical_position],
            "old_mask_position": canonical.canonical_to_old_indices[canonical_position],
            "link_type": link_type,
            "control_semantics": semantic,
            "setting_min": lo,
            "setting_max": hi,
            "is_binary": binary,
            "is_continuous": not binary,
            "upstream_node": str(extra.get("upstream_node", row.get("from_node", ""))),
            "downstream_node": str(extra.get("downstream_node", row.get("to_node", ""))),
            "storage_association": str(extra.get("storage_node", "")),
        })
    return pd.DataFrame(rows)


def _event_fields(event_id: str) -> tuple[str, str, str]:
    parts = event_id.split("_", 2)
    return parts[0], parts[1] if len(parts) > 1 else "D?", parts[2] if len(parts) > 2 else "unknown"


def _anchors(eligible: list[str]) -> list[str]:
    """Three deterministic non-calibration/formal anchors per return period."""
    patterns = ["chicago_center", "chicago_late", "block"]
    durations = [75, 210, 300]
    chosen = []
    for rp in ["T5", "T10", "T20", "T30", "T50", "T75", "T100"]:
        candidates = [eid for eid in eligible if eid.startswith(rp + "_")]
        for duration, pattern in zip(durations, patterns):
            wanted = f"{rp}_D{duration}_{pattern}"
            chosen.append(wanted if wanted in candidates else sorted(candidates)[len(chosen) % len(candidates)])
    return chosen


def _case_id(inp_hash: str, event_id: str, split: str, reference_policy: str, candidate: dict, order_hash: str, semantics_hash: str, code_hash: str) -> str:
    payload = {
        "inp_hash": inp_hash, "event_id": event_id, "split_timestamp": split,
        "reference_policy": reference_policy, "candidate_action_sequence": candidate,
        "canonical_action_order_hash": order_hash, "actuator_semantics_hash": semantics_hash, "code_hash": code_hash,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _pair_candidates(semantics: pd.DataFrame) -> list[tuple[str, str, str]]:
    ids = semantics["actuator_id"].tolist()
    forced = [
        ("RTC_IN_01", "RTC_OUT_01", "storage_inlet_outlet"), ("RTC_IN_02", "RTC_OUT_02", "storage_inlet_outlet"), ("RTC_IN_03", "RTC_OUT_03", "storage_inlet_outlet"),
        ("add350.1", "jichangheTank.2", "pump_regulator"), ("ADD301.2", "Zhongyi-2.2", "pump_regulator"), ("ADD301.3", "Zhongyi-2.2", "pump_regulator"),
        ("RTC_OUT_01", "HS2512760.1", "storage_downstream"), ("RTC_OUT_02", "gbz1.8", "storage_downstream"), ("RTC_OUT_03", "HS1306663.1", "storage_priority"),
    ]
    out = [(a, b, reason) for a, b, reason in forced if a in ids and b in ids]
    # Fill only with hydraulically adjacent facilities; this is intentionally
    # bounded and never becomes an all-pairs enumeration.
    records = semantics.set_index("actuator_id")
    for i, left in enumerate(ids):
        for right in ids[i + 1:]:
            if len(out) >= 31:
                return out
            l, r = records.loc[left], records.loc[right]
            adjacent = l["upstream_node"] in {r["upstream_node"], r["downstream_node"]} or l["downstream_node"] in {r["upstream_node"], r["downstream_node"]}
            if adjacent and (left, right, "shared_hydraulic_node") not in out:
                out.append((left, right, "shared_hydraulic_node"))
    # The remaining pairs use the fixed canonical order as a deterministic,
    # bounded coverage fallback and remain labelled for manual review.
    for i, left in enumerate(ids):
        for right in ids[i + 1:]:
            if len(out) >= 31:
                return out
            candidate = (left, right, "priority_domain_review")
            if not any({left, right} == {a, b} for a, b, _ in out):
                out.append(candidate)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Canonicalize 36-action order and create a no-execution joint-data plan.")
    ap.add_argument("--config", default="configs/wuhan_project6_v8_storage_36.yaml")
    ap.add_argument("--stage", choices=["CanonicalizeActionOrder", "PrepareJointDataPlan"], required=True)
    ap.add_argument("--out-root", default="outputs/project6_36_fulltrain_v1")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config); root = Path(cfg["project_root"])
    out_root = ensure_dir(root / args.out_root)
    canonical_dir, plan_dir = out_root / "canonical_action_order", out_root / "joint_data_plan"
    mask_path = cfg_path(cfg, "network.control_enabled_actuator_ids_file")
    cache109 = root / "outputs" / "cache_all109" / "transition_cache.npz"
    cache36 = root / "outputs" / "cache_v8_storage_variablepump" / "transition_cache.npz"
    inp = cfg_path(cfg, "network.inp")
    formal = root / "outputs" / "storage_retrofit" / str(cfg["storage_retrofit"]["run_tag"]) / "formal35_events.csv"
    calibration = formal.parent / "calibration_events.csv"
    code_hash = _sha256(Path(__file__))
    hashes = {"config": _sha256(Path(cfg["_config_path"])), "inp": _sha256(inp), "old_mask": _sha256(mask_path), "cache109": _sha256(cache109), "cache36": _sha256(cache36), "code": code_hash}
    global_ids = [str(x).removeprefix("a:") for x in _npz_member(cache109, "action_cols.npy").tolist()]
    old_ids = _ids(mask_path)
    order = CanonicalActionOrder.from_global_registry(global_ids, old_ids)
    semantics = _semantics(cfg, order)
    semantics_hash = _json_hash(semantics.to_dict("records"))

    if args.stage == "CanonicalizeActionOrder":
        outputs = [canonical_dir / name for name in ["canonical_36_actuator_order.csv", "canonical_36_control_mask.txt", "canonical_to_global109_index.csv", "global109_to_canonical36_index.csv", "old_mask_to_canonical_index.csv", "canonical_to_old_mask_index.csv", "action_order_manifest.json", "reuse_plan_canonical_correction.csv"]]
        status_path = canonical_dir / "stage_status.json"; input_path = canonical_dir / "input_hashes.json"
        if not args.force and status_path.exists() and input_path.exists() and all(p.exists() for p in outputs):
            status = json.loads(status_path.read_text(encoding="utf-8"))
            if status.get("status") == "complete" and json.loads(input_path.read_text(encoding="utf-8")) == hashes:
                print(json.dumps({"stage": args.stage, "status": "skipped_hash_match", "out": str(canonical_dir)})); return
        ensure_dir(canonical_dir)
        semantics.to_csv(canonical_dir / "canonical_36_actuator_order.csv", index=False)
        (canonical_dir / "canonical_36_control_mask.txt").write_text("\n".join(order.canonical_ids) + "\n", encoding="utf-8")
        pd.DataFrame({"canonical_position": range(36), "actuator_id": order.canonical_ids, "global_109_index": order.canonical_global_indices}).to_csv(canonical_dir / "canonical_to_global109_index.csv", index=False)
        pd.DataFrame({"global_109_index": range(109), "actuator_id": order.global_ids, "canonical_position": [order.canonical_ids.index(aid) if aid in order.canonical_ids else "" for aid in order.global_ids]}).to_csv(canonical_dir / "global109_to_canonical36_index.csv", index=False)
        pd.DataFrame({"old_mask_position": range(36), "actuator_id": order.old_mask_ids, "canonical_position": order.old_to_canonical_indices}).to_csv(canonical_dir / "old_mask_to_canonical_index.csv", index=False)
        pd.DataFrame({"canonical_position": range(36), "actuator_id": order.canonical_ids, "old_mask_position": order.canonical_to_old_indices}).to_csv(canonical_dir / "canonical_to_old_mask_index.csv", index=False)
        old_plan = out_root / "reuse_plan.csv"
        if old_plan.exists():
            corrected_plan = pd.read_csv(old_plan)
            hit = corrected_plan["artifact"].eq("cache_v8_storage_variablepump")
            corrected_plan.loc[hit, "semantic_compatibility"] = "compatible_after_global109_canonical_remapping"
            corrected_plan.loc[hit, "reason"] = "Cache order equals the global-109 canonical order. The old text mask is display/membership order only; formal and calibration events remain excluded from effect labels."
        else:
            corrected_plan = pd.DataFrame(columns=["artifact", "path", "hash", "schema", "semantic_compatibility", "decision", "reason", "downstream_stage"])
        corrected_plan.to_csv(canonical_dir / "reuse_plan_canonical_correction.csv", index=False)
        cache36_ids = [str(x).removeprefix("a:") for x in _npz_member(cache36, "action_cols.npy").tolist()]
        online = select_actuators_for_scope(pd.read_csv(cfg_path(cfg, "outputs.audit") / "actuator_table.csv"), "control_enabled")["actuator_id"].astype(str).tolist()
        manifest = {"schema_version": SCHEMA_VERSION, "canonical_order_hash": order.manifest_hash, "semantics_hash": semantics_hash, "global_registry_count": 109, "canonical_count": 36, "canonical_ids": list(order.canonical_ids), "old_display_mask_ids": old_ids, "cache36_matches_canonical": cache36_ids == list(order.canonical_ids), "online_selected_order_matches_canonical": online == list(order.canonical_ids), "training_cache_action_order": cache36_ids, "online_action_order": online, "conclusion": "The old text mask is a membership/display order only. Cache, training action columns, and online selected actuator order use the same global-109 canonical order."}
        (canonical_dir / "action_order_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        all_ok = cache36_ids == list(order.canonical_ids) and online == list(order.canonical_ids) and semantics["actuator_id"].nunique() == 36 and semantics.loc[semantics["actuator_id"].isin(["ADD301.2", "ADD301.3"]), "is_binary"].all() and bool(semantics.loc[semantics["actuator_id"].eq("add350.1"), "is_continuous"].iloc[0])
        _write_stage(canonical_dir, args.stage, hashes, outputs, "complete" if all_ok else "incomplete", "No models trained and no SWMM cases started.")
        print(json.dumps(manifest, indent=2)); return

    if not (canonical_dir / "action_order_manifest.json").exists():
        raise RuntimeError("CanonicalizeActionOrder must complete before PrepareJointDataPlan.")
    outputs = [plan_dir / name for name in ["joint_action_case_manifest.csv", "existing_coverage_summary.csv", "coverage_gaps.csv", "reusable_case_ids.csv", "new_case_ids.csv", "excluded_event_ids.csv", "data_generation_plan.json", "estimated_runtime.json", "data_leakage_provenance.json"]]
    status_path = plan_dir / "stage_status.json"; input_path = plan_dir / "input_hashes.json"
    plan_hashes = {**hashes, "canonical_order": order.manifest_hash, "semantics": semantics_hash, "formal": _sha256(formal), "calibration": _sha256(calibration)}
    if not args.force and status_path.exists() and input_path.exists() and all(p.exists() for p in outputs):
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") == "complete" and json.loads(input_path.read_text(encoding="utf-8")) == plan_hashes:
            print(json.dumps({"stage": args.stage, "status": "skipped_hash_match", "out": str(plan_dir)})); return
    ensure_dir(plan_dir)
    formal_ids = set(pd.read_csv(formal)["event_id"].astype(str)); calibration_ids = set(pd.read_csv(calibration)["event_id"].astype(str)); excluded = formal_ids | calibration_ids
    cache_events = set(map(str, _npz_member(cache36, "event_ids.npy").tolist()))
    eligible = sorted(cache_events - excluded)
    summary = pd.read_csv(root / "outputs" / "data_bank_train_v8_storage_variablepump" / "summary.csv")
    reusable = summary[~summary["event_id"].astype(str).isin(excluded) & ~summary["policy_id"].astype(str).eq("no_control")].copy()
    reusable["reuse_role"] = "hydraulic_pretraining_only_not_formal_effect_label"
    reusable_case = reusable[["event_id", "policy_id", "detail_file", "reuse_role"]].copy()
    reusable_case["reusable_case_id"] = [hashlib.sha256(f"{e}|{p}|{d}|{order.manifest_hash}".encode()).hexdigest() for e, p, d in reusable_case[["event_id", "policy_id", "detail_file"]].itertuples(index=False, name=None)]
    reusable_case.to_csv(plan_dir / "reusable_case_ids.csv", index=False)
    coverage_rows = []
    for row in semantics.itertuples(index=False):
        coverage_rows.append({"actuator_id": row.actuator_id, "link_type": row.link_type, "control_semantics": row.control_semantics, "existing_pretraining_trajectory_count": len(reusable_case), "exact_same_state_effect_cases": 0, "formal_effect_eligible": False, "required_new_pair_experiments": 6 if row.is_binary else 6, "coverage_status": "raw_action_pretraining_present_but_exact_counterfactual_missing"})
    pd.DataFrame(coverage_rows).to_csv(plan_dir / "existing_coverage_summary.csv", index=False)
    gaps = pd.DataFrame(coverage_rows)[["actuator_id", "control_semantics", "coverage_status"]]
    gaps["gap"] = "same-state paired no-control counterfactual absent; must not use old trajectory difference as formal effect label"
    gaps.to_csv(plan_dir / "coverage_gaps.csv", index=False)
    pd.DataFrame({"event_id": sorted(excluded), "split": ["formal35" if x in formal_ids else "calibration14" for x in sorted(excluded)], "exclusion_reason": "excluded from action-effect training, fine-tuning, uncertainty, reliability and threshold calibration"}).to_csv(plan_dir / "excluded_event_ids.csv", index=False)

    gat = root / "outputs" / "models_v8_storage_variablepump" / "gat_sr0p10.pt"
    checkpoint = torch.load(gat, map_location="cpu", weights_only=False)
    gat_seen = set(map(str, checkpoint.get("train_events", []))) | set(map(str, checkpoint.get("val_events", [])))
    overlap = sorted(formal_ids.intersection(gat_seen))
    leakage = {"gat_checkpoint": str(gat), "gat_hash": _sha256(gat), "gat_split_strategy": checkpoint.get("split_strategy"), "formal35_count": len(formal_ids), "formal35_seen_by_gat": overlap, "formal35_seen_count": len(overlap), "development_comparison_not_fully_blind": bool(overlap), "decision": "Existing formal35 is development comparison only, not a fully blind final holdout.", "untouched_formal_holdout_plan": "No T5-T100 event in the current 210-event library is untouched by the reused GAT. Before publication, generate or acquire a distinct external/observed rainfall holdout and keep it out of all GAT/action/effect/uncertainty/calibration datasets."}
    (plan_dir / "data_leakage_provenance.json").write_text(json.dumps(leakage, indent=2), encoding="utf-8")

    anchors = _anchors(eligible)
    phases = [("rising", "0.25"), ("peak", "0.50"), ("recession", "0.80")]
    specs: list[dict] = []
    continuous = semantics[~semantics["is_binary"]]
    binary = semantics[semantics["is_binary"]]
    for i, row in enumerate(continuous.itertuples(index=False)):
        for phase, split in phases:
            for mode, delta in [("small_ramp", 0.10), ("medium_pulse", 0.20)]: specs.append({"kind": "single_continuous", "actuators": [row.actuator_id], "phase": phase, "split": split, "mode": mode, "delta": delta, "event_id": anchors[(i * 6 + len(specs)) % len(anchors)]})
    for i, row in enumerate(binary.itertuples(index=False)):
        for phase, split in phases:
            for mode in ["off_to_on_hold", "on_to_off_hold"]: specs.append({"kind": "single_binary", "actuators": [row.actuator_id], "phase": phase, "split": split, "mode": mode, "delta": None, "event_id": anchors[(i * 6 + len(specs)) % len(anchors)]})
    for phase, split in phases:
        for mode in ["ramp_up", "ramp_down", "hold_then_release"]: specs.append({"kind": "add350_continuous_profile", "actuators": ["add350.1"], "phase": phase, "split": split, "mode": mode, "delta": 0.20, "event_id": anchors[len(specs) % len(anchors)]})
    for aid in ["RTC_IN_01", "RTC_OUT_01", "RTC_IN_02", "RTC_OUT_02", "RTC_IN_03", "RTC_OUT_03"]:
        for phase, split in phases: specs.append({"kind": "storage_interlock", "actuators": [aid], "phase": phase, "split": split, "mode": "retain_release", "delta": 0.20, "event_id": anchors[len(specs) % len(anchors)]})
    for left, right, reason in _pair_candidates(semantics):
        for phase, split in phases: specs.append({"kind": "hydraulic_pair", "actuators": [left, right], "phase": phase, "split": split, "mode": "coordinated_pair", "delta": 0.10, "pair_reason": reason, "event_id": anchors[len(specs) % len(anchors)]})
    if len(specs) != 336:
        raise AssertionError(f"expected 336 paired experiments, got {len(specs)}")
    rows = []
    for spec in specs:
        candidate = {key: value for key, value in spec.items() if key not in {"event_id", "phase", "split"}}
        pair_id = _case_id(_sha256(inp), spec["event_id"], spec["split"], "no_control", candidate, order.manifest_hash, semantics_hash, code_hash)
        for branch, policy, execution_sequence in [("A", "no_control", {"mode": "default_no_control"}), ("B", "candidate", candidate)]:
            # Logical IDs retain the candidate specification for both branches,
            # so each paired experiment remains uniquely traceable.  Reference
            # branches with the same event/split may later share one physical
            # checkpointed execution, recorded separately below.
            case_id = _case_id(_sha256(inp), spec["event_id"], spec["split"], policy, candidate, order.manifest_hash, semantics_hash, code_hash)
            execution_key = _case_id(_sha256(inp), spec["event_id"], spec["split"], policy, execution_sequence, order.manifest_hash, semantics_hash, code_hash)
            rows.append({"case_id": case_id, "execution_case_id": execution_key, "pair_id": pair_id, "branch": branch, "event_id": spec["event_id"], "phase": spec["phase"], "split_timestamp_fraction": spec["split"], "reference_policy": "no_control", "candidate_action_sequence": json.dumps(candidate, sort_keys=True), "executed_action_sequence": json.dumps(execution_sequence, sort_keys=True), "canonical_action_order_hash": order.manifest_hash, "actuator_semantics_hash": semantics_hash, "code_hash": code_hash, "requires_same_state_branching": True, "status": "planned_not_started"})
    manifest = pd.DataFrame(rows)
    manifest.to_csv(plan_dir / "joint_action_case_manifest.csv", index=False)
    physical_cases = manifest.sort_values("branch").drop_duplicates("execution_case_id").copy()
    physical_cases.to_csv(plan_dir / "new_case_ids.csv", index=False)
    median_wall_sec = float(pd.to_numeric(summary.loc[~summary["event_id"].astype(str).isin(excluded), "wall_time_sec"], errors="coerce").dropna().median())
    detail_paths = [Path(value) for value in reusable_case["detail_file"].head(128)]
    detail_bytes = [path.stat().st_size for path in detail_paths if path.exists()]
    median_detail_bytes = float(np.median(detail_bytes)) if detail_bytes else 2.0 * 1024 * 1024
    runtime = {"paired_experiments": 336, "logical_branch_rows": len(manifest), "unique_physical_swmm_cases": len(physical_cases), "reference_checkpoint_reuse_required": True, "max_new_swmm_cases": 800, "within_hard_cap": len(physical_cases) <= 800, "estimated_case_wall_sec_from_existing_bank_median": round(median_wall_sec, 2), "estimated_wall_hours_at_16_workers": round(len(physical_cases) * median_wall_sec / 16 / 3600, 2), "sampled_detail_files_for_disk_estimate": len(detail_bytes), "median_detail_bytes": int(median_detail_bytes), "estimated_detail_disk_gb": round(len(physical_cases) * median_detail_bytes / 1024**3, 2), "execution": "not started"}
    (plan_dir / "estimated_runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")
    plan = {"schema_version": SCHEMA_VERSION, "execution": "plan_only_no_swmm_started", "canonical_action_order_hash": order.manifest_hash, "actuator_semantics_hash": semantics_hash, "eligible_pretraining_events": len(eligible), "excluded_events": len(excluded), "reusable_pretraining_trajectories": len(reusable_case), "exact_same_state_reusable_effect_cases": 0, "new_paired_experiments": 336, "logical_branch_rows": len(manifest), "new_unique_physical_swmm_cases": len(physical_cases), "hard_cap": 800, "coverage": {"continuous_single": 204, "binary_single": 12, "add350_profiles": 9, "storage_interlock": 18, "hydraulic_pair": 93}, "anchor_events": anchors, "formal_effect_rule": "Only same-state branch A no-control vs branch B candidate cases created from this manifest can become formal action-effect labels."}
    (plan_dir / "data_generation_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    _write_stage(plan_dir, args.stage, plan_hashes, outputs, "complete", "Plan only. No PySWMM, model training, smoke, calibration, formal, or baseline run occurred.")
    print(json.dumps(plan, indent=2))


if __name__ == "__main__":
    main()
