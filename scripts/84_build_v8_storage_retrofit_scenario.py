from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from sewerrtc.control.retrofit_assets import validate_retrofit_asset_mix
from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config


RETURN_PERIODS = ["T5", "T10", "T20", "T30", "T50", "T75", "T100"]
PATTERNS = ["chicago_center", "chicago_early", "chicago_late", "block", "double_peak"]
DURATIONS = [75, 105, 150, 210, 240, 300]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _return_period_column(events: pd.DataFrame) -> pd.DataFrame:
    work = events.copy()
    if "return_period" not in work:
        work["return_period"] = work.get("rain_id", work["event_id"].astype(str).str.extract(r"^(T\d+)_", expand=False))
    work["return_period"] = work["return_period"].astype(str)
    return work


def build_event_splits(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create 14 calibration and 35 formal events with no overlap."""
    work = _return_period_column(events)
    work["duration_min"] = pd.to_numeric(work["duration_min"], errors="raise").astype(int)
    work["pattern"] = work["pattern"].astype(str)
    calibration_parts: list[pd.DataFrame] = []
    formal_parts: list[pd.DataFrame] = []
    for i, rp in enumerate(RETURN_PERIODS):
        group = work[work["return_period"].eq(rp)].copy()
        if group.empty:
            raise ValueError(f"rainfall library has no {rp} events")
        calibration_ids: list[str] = []
        for j in range(2):
            duration = DURATIONS[(i * 2 + j) % len(DURATIONS)]
            pattern = PATTERNS[(i * 2 + j) % len(PATTERNS)]
            match = group[group["duration_min"].eq(duration) & group["pattern"].eq(pattern)]
            if match.empty:
                raise ValueError(f"missing calibration event {rp}/D{duration}/{pattern}")
            calibration_ids.append(str(match.iloc[0]["event_id"]))
        calibration_parts.append(group[group["event_id"].astype(str).isin(calibration_ids)])
        for j, pattern in enumerate(PATTERNS):
            duration = DURATIONS[(i + j + 2) % len(DURATIONS)]
            match = group[
                group["duration_min"].eq(duration)
                & group["pattern"].eq(pattern)
                & ~group["event_id"].astype(str).isin(calibration_ids)
            ]
            if match.empty:
                alternatives = group[
                    group["pattern"].eq(pattern)
                    & ~group["event_id"].astype(str).isin(calibration_ids)
                ].sort_values("duration_min")
                if alternatives.empty:
                    raise ValueError(f"missing formal event {rp}/{pattern}")
                match = alternatives.head(1)
            formal_parts.append(match.head(1))
    calibration = pd.concat(calibration_parts, ignore_index=True).sort_values(["return_period", "duration_min", "pattern"])
    formal = pd.concat(formal_parts, ignore_index=True).sort_values(["return_period", "pattern", "duration_min"])
    if set(calibration["event_id"]).intersection(set(formal["event_id"])):
        raise AssertionError("calibration and formal event splits overlap")
    return calibration.reset_index(drop=True), formal.reset_index(drop=True)


def build_extended_formal_split(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create the optional 70-event formal split, disjoint from calibration."""
    calibration, _ = build_event_splits(events)
    work = _return_period_column(events)
    work["duration_min"] = pd.to_numeric(work["duration_min"], errors="raise").astype(int)
    blocked = set(calibration["event_id"].astype(str))
    selected: list[pd.DataFrame] = []
    for rp in RETURN_PERIODS:
        for pattern in PATTERNS:
            candidates = work[
                work["return_period"].eq(rp)
                & work["pattern"].astype(str).eq(pattern)
                & ~work["event_id"].astype(str).isin(blocked)
            ].sort_values("duration_min")
            if len(candidates) < 2:
                raise ValueError(f"need two disjoint formal events for {rp}/{pattern}")
            selected.append(candidates.iloc[[0, -1]])
    formal70 = pd.concat(selected, ignore_index=True).sort_values(["return_period", "pattern", "duration_min"])
    return calibration.reset_index(drop=True), formal70.reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build isolated Project6-v8-storage-retrofit metadata and event splits.")
    parser.add_argument("--config", default="configs/wuhan_project6_v8_storage.yaml")
    parser.add_argument("--run-tag", default="project6_v8_storage_T5_T100_v1")
    parser.add_argument("--extended", action="store_true", help="Also write the optional 70-event formal split.")
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = Path(cfg["project_root"])
    audit_dir = ensure_dir(cfg_path(cfg, "outputs.audit"))
    scenario_dir = ensure_dir(root / "outputs" / "storage_retrofit" / str(args.run_tag))
    manifest_path = cfg_path(cfg, "network.retrofit_asset_manifest")
    assets = pd.read_csv(manifest_path)
    composition = validate_retrofit_asset_mix(assets, action_dim=109)
    actuator_path = audit_dir / "actuator_table.csv"
    if not actuator_path.exists():
        raise FileNotFoundError(f"Run scripts/00_audit_inp.py first: {actuator_path}")
    actuators = pd.read_csv(actuator_path)
    selected = actuators[actuators["actuator_id"].astype(str).isin(assets["actuator_id"].astype(str))].copy()
    if len(selected) != len(assets):
        missing = sorted(set(assets["actuator_id"].astype(str)).difference(set(selected["actuator_id"].astype(str))))
        raise ValueError(f"retrofit assets absent from audited action table: {missing}")
    merged = assets.merge(
        selected[["actuator_id", "actuator_index", "control_enabled", "near_storage", "storage_control_type"]],
        on="actuator_id",
        how="left",
        suffixes=("_historical", "_audited"),
    )
    if not merged["control_enabled"].fillna(False).all():
        raise ValueError("all selected retrofit assets must be control_enabled in the scenario audit")
    if not (merged["action_index"].astype(int) == merged["actuator_index"].astype(int)).all():
        raise ValueError("historical 109-action indices do not match the audited action order")
    merged.to_csv(audit_dir / "storage_retrofit_asset_audit.csv", index=False)

    rain_table = pd.read_csv(cfg_path(cfg, "outputs.rainfall") / "rainfall_event_table.csv")
    calibration, formal = build_event_splits(rain_table)
    calibration.to_csv(scenario_dir / "calibration_events.csv", index=False)
    formal.to_csv(scenario_dir / "formal35_events.csv", index=False)
    formal70 = None
    if args.extended:
        _, formal70 = build_extended_formal_split(rain_table)
        formal70.to_csv(scenario_dir / "formal70_events.csv", index=False)
    control_file = cfg_path(cfg, "network.control_enabled_actuator_ids_file")
    inp = cfg_path(cfg, "network.inp")
    report = {
        "run_tag": str(args.run_tag),
        "scenario": "project6_v8_storage_retrofit",
        "composition": composition,
        "historical_action_dimension": 109,
        "control_enabled_count": int(actuators.get("control_enabled", pd.Series(dtype=bool)).sum()),
        "no_new_nodes": True,
        "input_hashes": {
            "retrofit_inp": _sha256(inp),
            "asset_manifest": _sha256(manifest_path),
            "control_mask": _sha256(control_file),
            "source_rainfall_table": _sha256(cfg_path(cfg, "outputs.rainfall") / "rainfall_event_table.csv"),
        },
        "event_split": {
            "calibration_events": int(len(calibration)),
            "formal_events": int(len(formal)),
            "extended_formal_events": int(len(formal70)) if formal70 is not None else 0,
        },
        "online_reference_mode": str((cfg.get("controller", {}) or {}).get("reference_policy_for_constraints", "")),
        "status": "metadata_built_no_swmm_started",
    }
    (scenario_dir / "scenario_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
