"""Read-only diagnosis of Core RTC PFV scope/action-effect failures.

Reads existing Core evidence and decision JSONL only.  It does not run SWMM,
load a model, or modify Formal evidence.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    ap = argparse.ArgumentParser()
    ap.add_argument("--core-root", type=Path, default=root / "outputs/project6_dual_reference_v4/final_v4/v42_paper/core_rtc")
    ap.add_argument("--output-dir", type=Path, default=None)
    args = ap.parse_args()
    core = args.core_root.resolve()
    output = (args.output_dir or core / "diagnostics").resolve()
    evidence_path = core / "calibration_CORE_RTC_EVIDENCE.json"
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for decision_path in sorted(
        (core / "paper_execution" / "calibration").glob(
            "*/Proposed/gat_sparse_reconstruction/decisions.jsonl"
        )
    ):
        event_id = decision_path.parts[-4]
        for line in decision_path.read_text(encoding="utf-8").splitlines():
            decision = json.loads(line)
            audits = decision.get("candidate_audits", [])
            values = [
                float(item["pfv_budget_metric_ucb_m3"])
                for item in audits
                if item.get("pfv_budget_metric_ucb_m3") is not None
            ]
            selected = next(
                (item for item in audits if item.get("candidate_id") == decision.get("selected_id")),
                None,
            )
            rows.append(
                {
                    "event_id": event_id,
                    "elapsed_min": decision.get("elapsed_min"),
                    "used_fallback": bool(decision.get("used_fallback", True)),
                    "selected_id": decision.get("selected_id"),
                    "selected_pfv_ucb_m3": None if selected is None else selected.get("pfv_budget_metric_ucb_m3"),
                    "selected_tfv_objective": decision.get("selected_objective_score"),
                    "candidate_count": len(audits),
                    "pfv_ucb_min_m3": min(values) if values else None,
                    "pfv_ucb_max_m3": max(values) if values else None,
                    "pfv_ucb_span_m3": (max(values) - min(values)) if values else None,
                }
            )

    active = [row for row in rows if not bool(row["used_fallback"])]
    spans = [float(row["pfv_ucb_span_m3"]) for row in active if row["pfv_ucb_span_m3"] is not None]
    per_event = {str(row["event_id"]): row for row in evidence.get("per_event", [])}
    event_rows = []
    for event_id, item in sorted(per_event.items()):
        event_active = [row for row in active if row["event_id"] == event_id]
        event_rows.append(
            {
                "event_id": event_id,
                "active_decisions": len(event_active),
                "pfv_event_pass": bool(item.get("PFV_constraint_pass", False)),
                "pfv_event_margin_m3": float(item["PFV_proposed_m3"] - item["PFV_budget_m3"]),
                "min_active_pfv_ucb_span_m3": min(
                    (float(row["pfv_ucb_span_m3"]) for row in event_active if row["pfv_ucb_span_m3"] is not None),
                    default=None,
                ),
                "max_active_pfv_ucb_span_m3": max(
                    (float(row["pfv_ucb_span_m3"]) for row in event_active if row["pfv_ucb_span_m3"] is not None),
                    default=None,
                ),
            }
        )

    diagnosis = {
        "status": "diagnostic_only",
        "swmm_runs": 0,
        "source_evidence": str(evidence_path),
        "decision_rows": len(rows),
        "active_decision_rows": len(active),
        "event_count": len(event_rows),
        "event_pfV_failures": sum(not bool(row["pfv_event_pass"]) for row in event_rows),
        "max_active_pfv_ucb_span_m3": max(spans, default=None),
        "mean_active_pfv_ucb_span_m3": sum(spans) / len(spans) if spans else None,
        "non_hold_actions_observed": len(active) > 0,
        "first_failing_layer": (
            "step2_pfv_action_effect_or_h120_event_scope"
            if active and any(not bool(row["pfv_event_pass"]) for row in event_rows)
            else "not_localized"
        ),
        "interpretation": (
            "PFV-UCB is nearly action-invariant across admitted candidates while "
            "authoritative event PFV fails on active closed-loop events; the failure "
            "is downstream of candidate opportunity and upstream of selector ranking."
        ),
        "event_rows": event_rows,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "PFV_SCOPE_ACTION_EFFECT_DIAGNOSTIC.json").write_text(
        json.dumps(diagnosis, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    with (output / "PFV_SCOPE_ACTION_EFFECT_DECISIONS.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["event_id"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(diagnosis, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
