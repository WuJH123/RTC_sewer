from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from sewerrtc.control.actuator_scope import select_actuators_for_scope
from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _action_columns(actuators: pd.DataFrame) -> list[str]:
    return [f"a:{actuator_id}" for actuator_id in actuators["actuator_id"].astype(str)]


def _event_policy(path: Path) -> tuple[str, str]:
    stem = path.stem.removesuffix("_detail")
    event_id, sep, policy_id = stem.rpartition("__")
    if not sep:
        raise ValueError(f"Trajectory detail name has no event/policy separator: {path.name}")
    return event_id, policy_id


def _header_action_columns(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle), [])
    return [column for column in header if column.startswith("a:")]


def _validate_rainfall_library(source_root: Path, cfg: dict) -> tuple[list[str], dict]:
    target_table = pd.read_csv(cfg_path(cfg, "outputs.rainfall") / "rainfall_event_table.csv").set_index("event_id")
    source_table_path = source_root / "outputs" / "rainfall_library" / "rainfall_event_table.csv"
    if not source_table_path.exists():
        raise FileNotFoundError(f"Project5 rainfall event table is missing: {source_table_path}")
    source_table = pd.read_csv(source_table_path).set_index("event_id")
    target_events = sorted(target_table.index.astype(str).tolist())
    missing = sorted(set(target_events) - set(source_table.index.astype(str)))
    if missing:
        raise ValueError(f"Project5 is missing {len(missing)} current Project6 events, e.g. {missing[:5]}")

    mismatches = []
    for event_id in target_events:
        source_csv = Path(source_table.loc[event_id, "rainfall_csv"])
        target_csv = Path(target_table.loc[event_id, "rainfall_csv"])
        if not source_csv.exists() or not target_csv.exists() or _sha256(source_csv) != _sha256(target_csv):
            mismatches.append(event_id)
    if mismatches:
        raise ValueError(f"Rainfall inputs differ for {len(mismatches)} events, e.g. {mismatches[:5]}")
    return target_events, {
        "source_rainfall_event_table": str(source_table_path),
        "target_rainfall_event_table": str(cfg_path(cfg, "outputs.rainfall") / "rainfall_event_table.csv"),
        "events_verified": len(target_events),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Import Project5 raw trajectories only after INP, rainfall and 109-action-schema verification."
    )
    ap.add_argument("--config", default="configs/wuhan_project6.yaml")
    ap.add_argument("--source-root", required=True, help="Explicit Project5 root; it is never inferred automatically.")
    ap.add_argument("--overwrite", action="store_true", help="Replace same-named destination details after validation.")
    ap.add_argument("--resume", action="store_true", help="Resume a partially completed import; complete same-size files are skipped.")
    ap.add_argument(
        "--include-source-events-for-gat",
        action="store_true",
        help="Also import compatible Project5 events outside the Project6 formal rainfall table for GAT/cache training only.",
    )
    args = ap.parse_args()

    cfg = load_config(args.config)
    source_root = Path(args.source_root).resolve()
    target_root = cfg_path(cfg, "project_root")
    source_inp = source_root / "data" / "wuhan_with_controls.inp"
    target_inp = cfg_path(cfg, "network.inp")
    if not source_inp.exists() or not target_inp.exists():
        raise FileNotFoundError("Source or target wuhan_with_controls.inp is missing")
    if _sha256(source_inp) != _sha256(target_inp):
        raise ValueError("Project5 and Project6 INP fingerprints differ; raw trajectories cannot be reused")

    audit_path = cfg_path(cfg, "outputs.audit") / "actuator_table.csv"
    source_audit_path = source_root / "outputs" / "audit" / "actuator_table.csv"
    if not audit_path.exists() or not source_audit_path.exists():
        raise FileNotFoundError("Source or target actuator_table.csv is missing; run 00_audit_inp.py first")
    scope = str((cfg.get("controller", {}) or {}).get("actuator_scope", "existing_rtc"))
    target_actuators = select_actuators_for_scope(pd.read_csv(audit_path), scope)
    source_actuators = pd.read_csv(source_audit_path)
    expected_actions = _action_columns(target_actuators)
    source_actions = _action_columns(source_actuators)
    if source_actions != expected_actions:
        raise ValueError(
            "Actuator schema differs between Project5 source and Project6 target: "
            f"source={len(source_actions)} target={len(expected_actions)}"
        )

    event_ids, rainfall_meta = _validate_rainfall_library(source_root, cfg)
    formal_event_ids = list(event_ids)
    if args.include_source_events_for_gat:
        source_table = pd.read_csv(source_root / "outputs" / "rainfall_library" / "rainfall_event_table.csv")
        event_ids = sorted(source_table["event_id"].astype(str).unique().tolist())
    source_bank = source_root / "outputs" / "data_bank_train_paired_no_controls"
    source_details = source_bank / "trajectories"
    if not source_details.exists():
        raise FileNotFoundError(f"Project5 trajectory directory is missing: {source_details}")
    target_bank = ensure_dir(cfg_path(cfg, "outputs.data_bank_train"))
    target_details = ensure_dir(target_bank / "trajectories")
    existing = list(target_details.glob("*_detail.csv"))
    if existing and not args.overwrite and not args.resume:
        raise FileExistsError(
            f"Destination trajectory directory is not empty ({len(existing)} details): {target_details}. "
            "Use a new output path or --overwrite after reviewing its manifest."
        )

    imported, rejected = [], []
    for source_detail in sorted(source_details.glob("*_detail.csv")):
        try:
            event_id, policy_id = _event_policy(source_detail)
        except ValueError as exc:
            rejected.append({"detail_file": str(source_detail), "reason": str(exc)})
            continue
        if event_id not in event_ids:
            continue
        action_columns = _header_action_columns(source_detail)
        if action_columns != expected_actions:
            rejected.append(
                {
                    "detail_file": str(source_detail),
                    "event_id": event_id,
                    "policy_id": policy_id,
                    "reason": f"action_schema_mismatch source={len(action_columns)} target={len(expected_actions)}",
                }
            )
            continue
        target_detail = target_details / source_detail.name
        if target_detail.exists() and args.resume and target_detail.stat().st_size == source_detail.stat().st_size:
            pass
        else:
            if target_detail.exists() and not args.overwrite and not args.resume:
                raise FileExistsError(f"Destination detail already exists: {target_detail}")
            shutil.copy2(source_detail, target_detail)
        imported.append(
            {
                "event_id": event_id,
                "policy_id": policy_id,
                "detail_file": str(target_detail),
                "source_detail_file": str(source_detail),
            }
        )
    if not imported:
        raise RuntimeError("No compatible Project5 trajectories were imported")

    schedule = pd.DataFrame(imported)
    schedule[["event_id", "policy_id", "detail_file"]].to_csv(target_bank / "trajectory_schedule.csv", index=False)
    source_summary = source_bank / "summary.csv"
    if source_summary.exists():
        summary = pd.read_csv(source_summary)
        keys = set(zip(schedule["event_id"], schedule["policy_id"]))
        if {"event_id", "policy_id"}.issubset(summary.columns):
            summary = summary[
                summary.apply(lambda row: (str(row["event_id"]), str(row["policy_id"])) in keys, axis=1)
            ].copy()
            details = schedule.set_index(["event_id", "policy_id"])["detail_file"]
            summary["detail_file"] = [details.get((str(e), str(p)), "") for e, p in zip(summary["event_id"], summary["policy_id"])]
            summary.to_csv(target_bank / "summary.csv", index=False)

    pd.DataFrame(rejected).to_csv(target_bank / "project5_import_rejected.csv", index=False)
    report = {
        "source_root": str(source_root),
        "target_root": str(target_root),
        "source_inp_sha256": _sha256(source_inp),
        "target_inp_sha256": _sha256(target_inp),
        "actuator_scope": scope,
        "action_count": len(expected_actions),
        "imported_details": len(imported),
        "imported_events": int(schedule["event_id"].nunique()),
        "imported_policies": int(schedule["policy_id"].nunique()),
        "formal_events_verified": int(len(formal_event_ids)),
        "extra_events_for_gat": int(len(set(event_ids) - set(formal_event_ids))),
        "include_source_events_for_gat": bool(args.include_source_events_for_gat),
        "rejected_details": len(rejected),
        **rainfall_meta,
    }
    (target_bank / "project5_import_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
