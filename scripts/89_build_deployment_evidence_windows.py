from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build event-time deployment evidence windows from accepted-action audit."
    )
    parser.add_argument("--audit-csv", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-report", default="")
    parser.add_argument("--tfv-delta-max", type=float, default=0.0)
    parser.add_argument("--peak-delta-max", type=float, default=0.0)
    parser.add_argument("--pfv-gain-min", type=float, default=-100.0)
    parser.add_argument("--min-events", type=int, default=1)
    args = parser.parse_args()

    audit = pd.read_csv(args.audit_csv)
    required = {"event_id", "elapsed_min", "true_TFV_delta", "true_peak_delta", "true_PFV_gain"}
    missing = sorted(required - set(audit.columns))
    if missing:
        raise KeyError(f"accepted-action audit is missing required columns: {missing}")
    work = audit.copy()
    for col in ["elapsed_min", "true_TFV_delta", "true_peak_delta", "true_PFV_gain"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    keep = (
        work["event_id"].astype(str).ne("")
        & work["elapsed_min"].notna()
        & work["true_TFV_delta"].le(float(args.tfv_delta_max))
        & work["true_peak_delta"].le(float(args.peak_delta_max))
        & work["true_PFV_gain"].ge(float(args.pfv_gain_min))
    )
    evidence = work.loc[keep].copy()
    evidence = evidence.sort_values(["event_id", "elapsed_min"]).drop_duplicates(
        subset=["event_id", "elapsed_min"], keep="first"
    )
    evidence["evidence_source"] = str(Path(args.audit_csv))
    evidence["deployment_rule"] = (
        f"true_TFV_delta<={float(args.tfv_delta_max)};"
        f"true_peak_delta<={float(args.peak_delta_max)};"
        f"true_PFV_gain>={float(args.pfv_gain_min)}"
    )
    out_cols = [
        "event_id",
        "elapsed_min",
        "target_actuators",
        "true_TFV_delta",
        "true_peak_delta",
        "true_PFV_gain",
        "evidence_source",
        "deployment_rule",
    ]
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    evidence[out_cols].to_csv(out_path, index=False)

    by_event = evidence.groupby("event_id").agg(
        windows=("elapsed_min", "count"),
        first_elapsed_min=("elapsed_min", "min"),
        last_elapsed_min=("elapsed_min", "max"),
        mean_true_TFV_delta=("true_TFV_delta", "mean"),
        mean_true_peak_delta=("true_peak_delta", "mean"),
        mean_true_PFV_gain=("true_PFV_gain", "mean"),
    )
    report = {
        "audit_csv": str(Path(args.audit_csv)),
        "out_csv": str(out_path),
        "rows": int(len(evidence)),
        "events": int(evidence["event_id"].nunique()),
        "min_events_required": int(args.min_events),
        "passed": bool(evidence["event_id"].nunique() >= int(args.min_events)),
        "rule": {
            "tfv_delta_max": float(args.tfv_delta_max),
            "peak_delta_max": float(args.peak_delta_max),
            "pfv_gain_min": float(args.pfv_gain_min),
        },
        "by_event": by_event.reset_index().to_dict(orient="records"),
    }
    report_path = Path(args.out_report) if args.out_report else out_path.with_suffix(".json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
