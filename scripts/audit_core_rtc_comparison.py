"""Summarize authoritative Core RTC run-result metadata without reading detail CSVs."""
from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def _mean(rows: list[dict[str, object]], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return statistics.mean(values) if values else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, required=True)
    ap.add_argument("--role", required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    base = args.run_root / "paper_execution" / args.role
    event_rows: list[dict[str, object]] = []
    for event_dir in sorted(base.iterdir()):
        if not event_dir.is_dir():
            continue
        for strategy_dir in sorted(event_dir.iterdir()):
            if not strategy_dir.is_dir():
                continue
            files = sorted(strategy_dir.glob("*/run_result.json"))
            if len(files) != 1:
                continue
            data = json.loads(files[0].read_text(encoding="utf-8"))
            kpi = data.get("kpis", {})
            event_rows.append(
                {
                    "event_id": event_dir.name,
                    "strategy": strategy_dir.name,
                    "authority": data.get("authority"),
                    "state_source": data.get("state_source"),
                    "PFV_m3": kpi.get("PFV"),
                    "TFV_m3": kpi.get("TFV"),
                    "peak_TFV_rate": kpi.get("peak_TFV_rate"),
                    "action_changes": kpi.get("action_changes"),
                    "fallback_rate": data.get("fallback_rate", 0.0),
                    "decision_count": data.get("decision_count", 0),
                    "runtime_sec": data.get("runtime_sec", 0.0),
                    "detail_path": data.get("detail_path"),
                }
            )
    by_event = {}
    for row in event_rows:
        by_event.setdefault(row["event_id"], {})[row["strategy"]] = row
    for event_id, strategies in by_event.items():
        nc = strategies.get("No-control")
        proposed = strategies.get("Proposed")
        internal = strategies.get("Internal")
        if nc and proposed:
            budget = 100.0 + 1.05 * float(nc["PFV_m3"])
            proposed["PFV_budget_m3"] = budget
            proposed["PFV_margin_m3"] = budget - float(proposed["PFV_m3"])
            proposed["PFV_pass"] = bool(float(proposed["PFV_m3"]) <= budget)
        if proposed and internal:
            proposed["TFV_delta_vs_Internal_m3"] = float(proposed["TFV_m3"]) - float(internal["TFV_m3"])
        if proposed and nc:
            proposed["TFV_delta_vs_No_control_m3"] = float(proposed["TFV_m3"]) - float(nc["TFV_m3"])

    strategies = sorted({str(row["strategy"]) for row in event_rows})
    aggregate = {}
    for strategy in strategies:
        rows = [row for row in event_rows if row["strategy"] == strategy]
        aggregate[strategy] = {
            "events": len(rows),
            "PFV_mean_m3": _mean(rows, "PFV_m3"),
            "TFV_mean_m3": _mean(rows, "TFV_m3"),
            "peak_mean_rate": _mean(rows, "peak_TFV_rate"),
            "action_changes_mean": _mean(rows, "action_changes"),
            "fallback_rate_mean": _mean(rows, "fallback_rate"),
            "runtime_mean_sec": _mean(rows, "runtime_sec"),
        }
    proposed_rows = [row for row in event_rows if row["strategy"] == "Proposed"]
    aggregate["Proposed"]["PFV_pass_events"] = sum(bool(row.get("PFV_pass")) for row in proposed_rows)
    aggregate["Proposed"]["PFV_violation_events"] = sum(not bool(row.get("PFV_pass")) for row in proposed_rows)
    aggregate["Proposed"]["TFV_improved_vs_Internal_events"] = sum(
        float(row["TFV_delta_vs_Internal_m3"]) < 0 for row in proposed_rows if row.get("TFV_delta_vs_Internal_m3") is not None
    )
    aggregate["Proposed"]["TFV_delta_vs_Internal_mean_m3"] = _mean(proposed_rows, "TFV_delta_vs_Internal_m3")
    aggregate["Proposed"]["TFV_delta_vs_No_control_mean_m3"] = _mean(proposed_rows, "TFV_delta_vs_No_control_m3")
    result = {
        "run_root": str(args.run_root),
        "role": args.role,
        "contract": "PROJECT6_V42_SIMPLE_RTC_CORE_V1",
        "hydraulic_rule": "PFV_Proposed <= 100 + 1.05 * PFV_No_control",
        "objective": "minimize_TFV_candidate_over_PFV_admitted_candidate_set",
        "event_rows": event_rows,
        "aggregate": aggregate,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path = args.output.with_suffix(".csv")
    fields = sorted({key for row in event_rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(event_rows)
    print(json.dumps(result["aggregate"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
