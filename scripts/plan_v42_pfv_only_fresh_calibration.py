"""Freeze a pre-control-only PFV calibration event plan.

This is planning only: it reads the existing event ledger, inventory, and
rainfall forcing files. It never runs SWMM and never writes Formal evidence.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _forcing(path: Path) -> dict[str, object]:
    header = pd.read_csv(path, nrows=0)
    cols = set(map(str, header.columns))
    intensity = next((c for c in ("rainfall_mm_h", "intensity_mm_h", "rainfall") if c in cols), None)
    if "elapsed_min" not in cols or intensity is None:
        raise ValueError(f"forcing lacks elapsed_min/intensity columns: {path}")
    frame = pd.read_csv(path, usecols=["elapsed_min", intensity])
    values = frame.apply(pd.to_numeric, errors="coerce").dropna()
    values = values.sort_values("elapsed_min", kind="mergesort")
    if values.empty:
        raise ValueError(f"empty forcing: {path}")
    elapsed = values["elapsed_min"].to_numpy(dtype=np.float64)
    rain = values[intensity].to_numpy(dtype=np.float64)
    if not np.isfinite(elapsed).all() or not np.isfinite(rain).all():
        raise ValueError(f"nonfinite forcing: {path}")
    if len(elapsed) > 1:
        dt = np.diff(elapsed)
        dt = dt[np.isfinite(dt) & (dt > 0)]
        step = float(np.median(dt)) if len(dt) else 5.0
    else:
        step = 5.0
    canonical = np.ascontiguousarray(np.column_stack((elapsed, rain)).astype("<f8", copy=False))
    return {
        "forcing_content_sha256": hashlib.sha256(canonical.tobytes()).hexdigest(),
        "forcing_rows": int(len(elapsed)),
        "duration_min": float(elapsed[-1] - elapsed[0]),
        "total_depth_mm": float(np.sum(rain) * step / 60.0),
        "peak_intensity_mm_h": float(np.max(rain)),
    }


def select_plan(ledger: pd.DataFrame, inventory: pd.DataFrame, count: int = 12) -> pd.DataFrame:
    untouched = ledger.loc[ledger["formal_f2_role"].astype(str).eq("unused_untouched")].copy()
    if untouched.empty:
        raise RuntimeError("no unused_untouched rainfall groups available")
    untouched = untouched[["inventory_event_id", "rainfall_sha256", "rainfall_group_key"]]
    if untouched["rainfall_sha256"].duplicated().any():
        raise RuntimeError("unused_untouched has duplicate rainfall SHA")
    inv = inventory.copy()
    inv["event_id"] = inv["event_id"].astype(str)
    inv = inv.drop(columns=["rainfall_sha256"], errors="ignore")
    merged = untouched.merge(inv, left_on="inventory_event_id", right_on="event_id", how="left", validate="one_to_one")
    if merged["rainfall_path"].isna().any():
        raise RuntimeError("untouched event missing rainfall_path")
    rows: list[dict[str, object]] = []
    for row in merged.to_dict("records"):
        path = Path(str(row["rainfall_path"]))
        if not path.exists():
            raise FileNotFoundError(path)
        rows.append(
            {
                "event_id": str(row["inventory_event_id"]),
                "rainfall_sha256": str(row["rainfall_sha256"]),
                "rainfall_group_key": str(row["rainfall_group_key"]),
                "storm_family_id": str(row.get("storm_family_id", "")),
                "rainfall_path": str(path.resolve()),
                **_forcing(path),
            }
        )
    frame = pd.DataFrame(rows).sort_values(["total_depth_mm", "peak_intensity_mm_h", "event_id"], kind="mergesort").reset_index(drop=True)
    frame["severity_stratum"] = pd.qcut(frame["total_depth_mm"].rank(method="first"), 3, labels=["low", "medium", "high"]).astype(str)
    per_stratum = count // 3
    if count % 3:
        raise ValueError("count must be divisible by three for low/medium/high selection")
    chosen: list[pd.Series] = []
    for stratum in ("low", "medium", "high"):
        part = frame.loc[frame["severity_stratum"].eq(stratum)].sort_values(["storm_family_id", "event_id"], kind="mergesort")
        picked: list[pd.Series] = []
        families: set[str] = set()
        for _, row in part.iterrows():
            if row["storm_family_id"] not in families:
                picked.append(row)
                families.add(str(row["storm_family_id"]))
            if len(picked) == per_stratum:
                break
        if len(picked) < per_stratum:
            for _, row in part.iterrows():
                if all(str(row["event_id"]) != str(x["event_id"]) for x in picked):
                    picked.append(row)
                if len(picked) == per_stratum:
                    break
        if len(picked) != per_stratum:
            raise RuntimeError(f"cannot select {per_stratum} events in {stratum}")
        chosen.extend(picked)
    result = pd.DataFrame(chosen).sort_values(["severity_stratum", "event_id"], key=lambda s: s.map({"low": 0, "medium": 1, "high": 2}) if s.name == "severity_stratum" else s, kind="mergesort").reset_index(drop=True)
    result.insert(0, "selection_order", np.arange(1, len(result) + 1))
    result["selection_role"] = "fresh_pfv_only_calibration"
    result["pre_control_descriptors_only"] = True
    result["selection_uses_control_outcome"] = False
    result["authoritative_swmm_required"] = True
    result["formal_mainline_authorized"] = False
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--project-root", type=Path, default=root)
    ap.add_argument("--count", type=int, default=12)
    ap.add_argument("--output-dir", type=Path, default=root / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/pfv_only_v2")
    args = ap.parse_args()
    formal = args.project_root / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2"
    ledger_path = formal / "prepare/FORMAL_F2_EVENT_LEDGER.csv"
    inventory_path = args.project_root / "outputs/project6_dual_reference_v4/final_v4/inventory/event_inventory.csv"
    plan = select_plan(pd.read_csv(ledger_path), pd.read_csv(inventory_path), args.count)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "FRESH_PFV_ONLY_CALIBRATION_PLAN.csv"
    json_path = args.output_dir / "FRESH_PFV_ONLY_CALIBRATION_PLAN.json"
    plan.to_csv(csv_path, index=False)
    payload = {
        "status": "planned_not_executed",
        "formal_mainline_authorized": False,
        "swmm_runs": 0,
        "selected_groups": int(plan["rainfall_sha256"].nunique()),
        "selection_rule": "12 untouched groups, four each from deterministic low/medium/high total-depth strata, family-diverse within stratum; forcing descriptors only",
        "selection_uses_control_outcome": False,
        "ledger_sha256": _sha256(ledger_path),
        "inventory_sha256": _sha256(inventory_path),
        "output_plan": str(csv_path.resolve()),
    }
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False), flush=True)
    print(plan[["selection_order", "event_id", "severity_stratum", "storm_family_id", "total_depth_mm", "peak_intensity_mm_h"]].to_string(index=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
