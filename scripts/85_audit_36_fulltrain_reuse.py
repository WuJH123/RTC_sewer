from __future__ import annotations

"""Read-only audit and reuse planning for the frozen Project6 36-asset line.

This script deliberately does not call PySWMM or any training entry point.  It
creates traceable, idempotent stage records for the two planning stages that
precede any new joint-action data generation.
"""

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

from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config
from sewerrtc.control.canonical_action_order import CanonicalActionOrder


SCHEMA_VERSION = "project6_36_fulltrain_audit_v1"
VALID_STATES = {"pending", "running", "complete", "failed", "incomplete"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def json_hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def rel_or_abs(root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def npz_small_array(path: Path, member: str) -> np.ndarray:
    """Read only a small npy member from a compressed cache archive."""
    with zipfile.ZipFile(path) as archive:
        return np.load(io.BytesIO(archive.read(member)), allow_pickle=True)


def cache_metadata(path: Path) -> dict:
    if not path.exists():
        return {"exists": False}
    with zipfile.ZipFile(path) as archive:
        members = {info.filename: {"raw_bytes": info.file_size, "zip_bytes": info.compress_size} for info in archive.infolist()}
    action_cols = [str(v).removeprefix("a:") for v in npz_small_array(path, "action_cols.npy").tolist()]
    event_ids = sorted(set(map(str, npz_small_array(path, "event_ids.npy").tolist())))
    policy_ids = sorted(set(map(str, npz_small_array(path, "policy_ids.npy").tolist())))
    return {
        "exists": True,
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
        "action_columns": action_cols,
        "action_dim": len(action_cols),
        "event_ids": event_ids,
        "event_count": len(event_ids),
        "policy_ids": policy_ids,
        "members": members,
    }


def write_stage_files(stage_dir: Path, stage: str, input_hashes: dict, outputs: list[Path], status: str, note: str = "") -> None:
    if status not in VALID_STATES:
        raise ValueError(f"invalid stage status: {status}")
    ensure_dir(stage_dir)
    manifest = {
        "stage": stage,
        "schema_version": SCHEMA_VERSION,
        "expected_outputs": [str(path) for path in outputs],
        "idempotence_key": json_hash({"stage": stage, "input_hashes": input_hashes, "schema_version": SCHEMA_VERSION}),
    }
    (stage_dir / "stage_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (stage_dir / "input_hashes.json").write_text(json.dumps(input_hashes, indent=2, ensure_ascii=False), encoding="utf-8")
    (stage_dir / "stage_status.json").write_text(
        json.dumps({"stage": stage, "status": status, "schema_version": SCHEMA_VERSION, "note": note}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    pd.DataFrame(columns=["case_id", "status", "artifact"]).to_csv(stage_dir / "completed_case_ids.csv", index=False)
    pd.DataFrame(columns=["case_id", "stage", "error"]).to_csv(stage_dir / "failures.csv", index=False)


def control_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")]


def main() -> None:
    ap = argparse.ArgumentParser(description="Read-only Audit and ReusePlan for Project6 frozen 36-asset full training.")
    ap.add_argument("--config", default="configs/wuhan_project6_v8_storage_36.yaml")
    ap.add_argument("--stage", choices=["Audit", "ReusePlan"], required=True)
    ap.add_argument("--out-root", default="outputs/project6_36_fulltrain_v1")
    ap.add_argument("--force", action="store_true", help="Rewrite planning artifacts even if the idempotence key matches.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    root = Path(cfg["project_root"])
    out_root = ensure_dir(rel_or_abs(root, args.out_root))
    audit_dir = out_root / "audit"
    reuse_dir = out_root / "reuse_plan"

    inp = cfg_path(cfg, "network.inp")
    control_mask = cfg_path(cfg, "network.control_enabled_actuator_ids_file")
    retrofit_assets = cfg_path(cfg, "network.retrofit_asset_manifest")
    audit_table = cfg_path(cfg, "outputs.audit") / "actuator_table.csv"
    rainfall_table = cfg_path(cfg, "outputs.rainfall") / "rainfall_event_table.csv"
    scenario_dir = root / "outputs" / "storage_retrofit" / str(cfg["storage_retrofit"]["run_tag"])
    calibration_path = scenario_dir / "calibration_events.csv"
    formal_path = scenario_dir / "formal35_events.csv"
    old_v8_table = rel_or_abs(root, cfg["storage_retrofit"]["original_v8_control_table"])
    cache36 = root / "outputs" / "cache_v8_storage_variablepump" / "transition_cache.npz"
    cache109 = root / "outputs" / "cache_all109" / "transition_cache.npz"
    gat = root / "outputs" / "models_v8_storage_variablepump" / "gat_sr0p10.pt"
    old_horizon = root / "outputs" / "models_v8_storage_variablepump" / "horizon_temporal_gnn.pt"
    old_uncertainty = root / "outputs" / "models_v8_storage_variablepump" / "horizon_residual_quantile_uncertainty.npz"

    ids = control_ids(control_mask)
    assets = pd.read_csv(retrofit_assets)
    table = pd.read_csv(audit_table)
    v8 = pd.read_csv(old_v8_table)
    selected = table[table["actuator_id"].astype(str).isin(ids)].copy()
    selected["actuator_id"] = selected["actuator_id"].astype(str)
    selected = selected.set_index("actuator_id").reindex(ids).reset_index()
    missing = selected[selected.get("link_type").isna()]["actuator_id"].tolist()
    if missing:
        raise RuntimeError(f"frozen control mask contains unaudited assets: {missing}")

    cache36_meta = cache_metadata(cache36)
    cache109_meta = cache_metadata(cache109)
    cache36_ids = cache36_meta.get("action_columns", [])
    cache36_order_match = cache36_ids == ids
    canonical_order = CanonicalActionOrder.from_global_registry(cache109_meta.get("action_columns", []), ids)
    cache36_matches_canonical = cache36_ids == list(canonical_order.canonical_ids)
    historical_indices = dict(zip(assets["actuator_id"].astype(str), assets["action_index"].astype(int)))
    selected["historical_action_index"] = selected["actuator_id"].map(historical_indices)

    old_v8_ids = v8["actuator_id"].astype(str).tolist()
    new_ids = [value for value in ids if value not in old_v8_ids]
    expected_new = assets["actuator_id"].astype(str).tolist()
    binary = {"ADD301.2", "ADD301.3"}
    selected["control_semantics"] = selected["actuator_id"].map(
        {**dict(zip(assets["actuator_id"].astype(str), assets["control_semantics"].astype(str))), "add350.1": "continuous_pump"}
    ).fillna(selected["link_type"].astype(str).map({"orifice": "continuous_orifice", "weir": "continuous_weir", "pump": "unknown_pump"}))
    selected.loc[selected["actuator_id"].isin(binary), "control_semantics"] = "binary_pump"
    selected.loc[selected["actuator_id"].eq("add350.1"), "control_semantics"] = "continuous_pump"
    selected["setting_range"] = selected["actuator_id"].map(dict(zip(assets["actuator_id"].astype(str), assets["setting_range"].astype(str)))).fillna("[0.0,1.0]")
    selected.loc[selected["actuator_id"].isin(binary), "setting_range"] = "{0,1}"
    selected["is_frozen_control_asset"] = True
    selected["sequence_position"] = range(len(selected))

    formal_events = set(pd.read_csv(formal_path)["event_id"].astype(str))
    calibration_events = set(pd.read_csv(calibration_path)["event_id"].astype(str))
    cache_event_overlap = sorted(formal_events.intersection(cache36_meta.get("event_ids", [])))
    calibration_overlap = sorted(calibration_events.intersection(cache36_meta.get("event_ids", [])))
    inputs = {
        "config": sha256(Path(cfg["_config_path"])),
        "inp": sha256(inp),
        "control_mask": sha256(control_mask),
        "retrofit_asset_manifest": sha256(retrofit_assets),
        "actuator_audit_table": sha256(audit_table),
        "rainfall_table": sha256(rainfall_table),
        "formal_events": sha256(formal_path),
        "calibration_events": sha256(calibration_path),
        "audit_script": sha256(Path(__file__)),
    }
    if cache36.exists():
        inputs["cache36"] = cache36_meta["sha256"]
    if cache109.exists():
        inputs["cache109"] = cache109_meta["sha256"]

    if args.stage == "Audit":
        outputs = [audit_dir / "audit_report.json", audit_dir / "reuse_inventory.csv", audit_dir / "actuator_semantics.csv", audit_dir / "incompatible_artifacts.csv"]
        old_status = audit_dir / "stage_status.json"
        old_inputs = audit_dir / "input_hashes.json"
        if not args.force and old_status.exists() and old_inputs.exists() and all(path.exists() for path in outputs):
            status = json.loads(old_status.read_text(encoding="utf-8"))
            if status.get("status") == "complete" and json.loads(old_inputs.read_text(encoding="utf-8")) == inputs:
                print(json.dumps({"stage": "Audit", "status": "skipped_hash_match", "out_dir": str(audit_dir)}, ensure_ascii=False))
                return
        incompatibilities = []
        if len(ids) != 36:
            incompatibilities.append({"artifact": "control_mask", "reason": f"expected 36 IDs, found {len(ids)}", "severity": "error"})
        if len(set(ids)) != 36:
            incompatibilities.append({"artifact": "control_mask", "reason": "duplicate actuator IDs", "severity": "error"})
        if set(new_ids) != set(expected_new):
            incompatibilities.append({"artifact": "36_asset_composition", "reason": "10 retrofit assets do not match the frozen manifest", "severity": "error"})
        if not cache36_order_match:
            incompatibilities.append({"artifact": "control_mask_display_order", "reason": "display order differs from cache; global-109 canonical order is authoritative", "severity": "warning"})
        if not cache36_matches_canonical:
            incompatibilities.append({"artifact": "cache_v8_storage_variablepump", "reason": "36-action cache order differs from global-109 canonical order", "severity": "error"})
        if cache_event_overlap:
            incompatibilities.append({"artifact": "cache_v8_storage_variablepump", "reason": f"formal event leakage: {len(cache_event_overlap)} formal events occur in cache", "severity": "warning"})
        if selected.loc[selected["actuator_id"].isin(binary), "control_semantics"].ne("binary_pump").any():
            incompatibilities.append({"artifact": "pump_semantics", "reason": "ADD301 pumps are not binary in audit table", "severity": "error"})
        if selected.loc[selected["actuator_id"].eq("add350.1"), "control_semantics"].ne("continuous_pump").any():
            incompatibilities.append({"artifact": "pump_semantics", "reason": "add350.1 is not continuous in audit table", "severity": "error"})
        inventory = pd.DataFrame(
            [
                {"artifact": "rainfall_library", "path": str(rainfall_table), "exists": rainfall_table.exists(), "schema": "event table", "note": "reuse directly"},
                {"artifact": "event_splits", "path": str(scenario_dir), "exists": formal_path.exists() and calibration_path.exists(), "schema": "14 calibration + 35 formal", "note": "frozen split"},
                {"artifact": "frozen_36_control_mask", "path": str(control_mask), "exists": control_mask.exists(), "schema": "36 ordered IDs", "note": "must remain immutable"},
                {"artifact": "gat_checkpoint", "path": str(gat), "exists": gat.exists(), "schema": "state encoder", "note": "freeze or low-LR fine-tune only"},
                {"artifact": "36_action_cache", "path": str(cache36), "exists": cache36.exists(), "schema": f"action_dim={cache36_meta.get('action_dim')}", "note": "formal-event leakage; pretraining only"},
                {"artifact": "109_action_cache", "path": str(cache109), "exists": cache109.exists(), "schema": f"action_dim={cache109_meta.get('action_dim')}", "note": "historical hydraulic/action pretraining only"},
                {"artifact": "old_horizon_model", "path": str(old_horizon), "exists": old_horizon.exists(), "schema": "aggregate-action temporal GNN", "note": "not valid as formal raw-action effect model"},
                {"artifact": "old_uncertainty", "path": str(old_uncertainty), "exists": old_uncertainty.exists(), "schema": "old quantile calibration", "note": "must be rebuilt"},
            ]
        )
        report = {
            "schema_version": SCHEMA_VERSION,
            "stage": "Audit",
            "frozen_control_count": len(ids),
            "frozen_control_ids": ids,
            "legacy_v8_count": len(old_v8_ids),
            "retrofit_count": len(new_ids),
            "retrofit_ids": new_ids,
            "historical_action_index_invariant": all(value in cache109_meta.get("action_columns", []) for value in ids),
            "cache36_order_matches_control_mask": cache36_order_match,
            "cache36_order_matches_global109_canonical": cache36_matches_canonical,
            "canonical_action_order_hash": canonical_order.manifest_hash,
            "cache36_event_count": cache36_meta.get("event_count"),
            "formal_events": len(formal_events),
            "calibration_events": len(calibration_events),
            "formal_events_present_in_cache36": cache_event_overlap,
            "calibration_events_present_in_cache36": calibration_overlap,
            "semantics": {"add350.1": "continuous [0,1]", "ADD301.2": "binary {0,1}", "ADD301.3": "binary {0,1}"},
            "audit_passed": not any(row["severity"] == "error" for row in incompatibilities),
            "warnings": [row["reason"] for row in incompatibilities if row["severity"] == "warning"],
        }
        ensure_dir(audit_dir)
        selected.to_csv(audit_dir / "actuator_semantics.csv", index=False)
        inventory.to_csv(audit_dir / "reuse_inventory.csv", index=False)
        pd.DataFrame(incompatibilities, columns=["artifact", "reason", "severity"]).to_csv(audit_dir / "incompatible_artifacts.csv", index=False)
        (audit_dir / "audit_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        write_stage_files(audit_dir, "Audit", inputs, outputs, "complete" if report["audit_passed"] else "incomplete", "Read-only audit; no PySWMM or training executed.")
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return

    audit_report = audit_dir / "audit_report.json"
    if not audit_report.exists():
        raise RuntimeError("Audit must complete before ReusePlan.")
    outputs = [out_root / "reuse_plan.csv"]
    decisions = [
        ("rainfall_library", rainfall_table, "reuse_directly", "same retrofit event library", "BuildMissingJointData"),
        ("frozen_event_splits", scenario_dir, "reuse_directly", "calibration/formal tables are fixed; formal IDs must be excluded from effect training", "TrainActionEffect"),
        ("priority_sensors_graph", cfg_path(cfg, "outputs.network"), "reuse_directly", "topology and sparse sensing definition are unchanged", "TrainActionEffect"),
        ("gat_sr0p10", gat, "reuse_directly", "hydraulic state encoder is action-dimension independent; validate locally before optional low-LR fine-tuning", "TrainActionEffect"),
        ("cache_v8_storage_variablepump", cache36, "reuse_for_pretraining", "cache uses global-109 canonical order (not the legacy display mask); all formal/calibration events must be excluded from formal-effect labels", "BuildMissingJointData"),
        ("cache_all109", cache109, "reuse_for_pretraining", "use state/rain/target hydraulic trajectories and aligned historical actions only", "TrainActionEffect"),
        ("old_26asset_trajectories", old_v8_table, "reuse_for_pretraining", "hydraulic/action pretraining only; not same 36-asset effect semantics", "TrainActionEffect"),
        ("old_36asset_failed_trajectories", root / "outputs" / "closed_loop_paired_no_controls", "reuse_as_negative_sample", "retain rejected/unsafe trajectories as negative examples after semantic filtering", "BuildMissingJointData"),
        ("old_aggregate_action_horizon_model", old_horizon, "ignore", "uses aggregate action representation; prohibited for the formal raw [B,H,36] effect controller", "TrainActionEffect"),
        ("old_uncertainty_quantiles", old_uncertainty, "rebuild", "calibration is tied to obsolete action semantics and target labels", "TrainUncertainty"),
        ("raw_joint_effect_dataset", out_root / "joint_action_case_manifest.csv", "rebuild", "requires same-state no-control references and raw candidate/reference sequences", "BuildMissingJointData"),
        ("internal_efd_autorbc_baselines", root / "outputs" / "closed_loop_paired_no_controls", "rerun", "reuse only if all six hash fields match; this audit cannot prove code/initial-condition/options equivalence", "Formal"),
    ]
    rows = []
    for artifact, path, decision, reason, downstream in decisions:
        path = Path(path)
        rows.append({
            "artifact": artifact,
            "path": str(path),
            "hash": sha256(path) if path.is_file() else "directory_or_missing",
            "schema": "see audit_report" if artifact != "cache_v8_storage_variablepump" else "36-action cache [state, rain, target_state, action_seq]",
            "semantic_compatibility": "manual_review" if decision in {"rerun", "rebuild"} else "compatible_with_restrictions",
            "decision": decision,
            "reason": reason,
            "downstream_stage": downstream,
        })
    plan = pd.DataFrame(rows)
    plan.to_csv(out_root / "reuse_plan.csv", index=False)
    reuse_inputs = {**inputs, "audit_report": sha256(audit_report)}
    write_stage_files(reuse_dir, "ReusePlan", reuse_inputs, outputs, "complete", "Planning only; no PySWMM or training executed.")
    print(json.dumps({"stage": "ReusePlan", "out": str(out_root / "reuse_plan.csv"), "decisions": plan["decision"].value_counts().to_dict()}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
