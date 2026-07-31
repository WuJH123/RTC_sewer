from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config


def _configured_representatives(cfg: dict) -> list[str]:
    rain = cfg.get("rainfall", {}) or {}
    ids = rain.get("representative_event_ids", []) or []
    return [str(x).strip() for x in ids if str(x).strip()]


def select_representative_events(cfg: dict, table: pd.DataFrame, max_events: int = 0) -> pd.DataFrame:
    if "event_id" not in table:
        raise ValueError("rainfall_event_table.csv must contain event_id")
    table = table.copy()
    table["event_id"] = table["event_id"].astype(str)
    configured = _configured_representatives(cfg)
    if configured:
        missing = [event_id for event_id in configured if event_id not in set(table["event_id"])]
        if missing:
            raise ValueError(f"Configured representative_event_ids are absent from rainfall_event_table.csv: {missing}")
        order = {event_id: i for i, event_id in enumerate(configured)}
        selected = table[table["event_id"].isin(configured)].copy()
        selected["_representative_order"] = selected["event_id"].map(order).astype(int)
        selected = selected.sort_values("_representative_order").drop(columns=["_representative_order"])
        if max_events and len(selected) < int(max_events):
            work = table[~table["event_id"].isin(selected["event_id"])].copy()
            for col in ["total_depth_mm", "duration_min", "peak_intensity_mm_h"]:
                work[col] = pd.to_numeric(work.get(col, 0.0), errors="coerce").fillna(0.0)
            ranked = work.sort_values(
                ["total_depth_mm", "duration_min", "peak_intensity_mm_h"],
                ascending=[False, False, False],
            )
            stratified_rows = []
            if {"rain_id", "pattern"}.issubset(work.columns):
                for _, sub in work.groupby(["rain_id", "pattern"], sort=False):
                    stratified_rows.append(
                        sub.sort_values(
                            ["duration_min", "total_depth_mm", "peak_intensity_mm_h"],
                            ascending=[False, False, False],
                        ).head(1)
                    )
            stratified = pd.concat(stratified_rows, ignore_index=True, sort=False) if stratified_rows else ranked.iloc[0:0]
            supplement = pd.concat([stratified, ranked], ignore_index=True, sort=False)
            supplement = supplement.drop_duplicates(subset=["event_id"], keep="first")
            selected = pd.concat(
                [selected, supplement.head(int(max_events) - len(selected))],
                ignore_index=True,
                sort=False,
            )
    else:
        work = table.copy()
        for col in ["total_depth_mm", "duration_min", "peak_intensity_mm_h"]:
            work[col] = pd.to_numeric(work.get(col, 0.0), errors="coerce").fillna(0.0)
        patterns = list(dict.fromkeys(work.get("pattern", pd.Series(["unknown"])).astype(str).tolist()))
        rows = []
        per_pattern = max(1, int(max_events or 8) // max(1, len(patterns)))
        for pattern in patterns:
            sub = work[work["pattern"].astype(str).eq(pattern)].copy()
            sub = sub.sort_values(["total_depth_mm", "duration_min", "peak_intensity_mm_h"], ascending=[False, False, False])
            rows.append(sub.head(per_pattern))
        selected = pd.concat(rows, ignore_index=True, sort=False) if rows else work.head(max_events or 8)
    if max_events and len(selected) > int(max_events):
        selected = selected.head(int(max_events))
    return selected.reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wuhan.yaml")
    ap.add_argument("--max-events", type=int, default=0)
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = cfg_path(cfg, "project_root")
    table_path = cfg_path(cfg, "outputs.rainfall") / "rainfall_event_table.csv"
    if not table_path.exists():
        raise FileNotFoundError(f"Missing rainfall event table: {table_path}")
    table = pd.read_csv(table_path)
    selected = select_representative_events(cfg, table, max_events=int(args.max_events or 0))
    out_dir = ensure_dir(Path(args.out_dir) if args.out_dir else cfg_path(cfg, "outputs.design"))
    selected_path = out_dir / "representative_events.csv"
    ids_path = out_dir / "representative_event_ids.txt"
    selected.to_csv(selected_path, index=False)
    ids_path.write_text("\n".join(selected["event_id"].astype(str).tolist()) + "\n", encoding="utf-8")
    report = {
        "rainfall_event_table": str(table_path),
        "events_available": int(len(table)),
        "representative_events": int(len(selected)),
        "event_ids": selected["event_id"].astype(str).tolist(),
        "outputs": {"events": str(selected_path), "event_ids": str(ids_path)},
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
