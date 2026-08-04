"""Authoritative scientific outcome audit for the V4.2 final held-out test.

Pipeline completeness is not scientific success.  This audit recomputes KPIs
from authoritative SWMM ``detail.csv`` files for the frozen final holdout and
checks the only hydraulic hard outcome contract requested for V2:

    PFV_proposed <= 100 m3 + 1.05 * PFV_no_control

for every final event.  TFV is the performance objective and is reported
against all baselines (especially native Internal), but it is not converted into
a second hard safety constraint. Global Peak is reporting-only.

This event-level audit complements, rather than replaces, the online H120
surrogate admission rule. A passed online predictor cannot make an authoritative
SWMM outcome violation disappear.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sewerrtc.simulation.kpi_metrics import compute_kpis
from sewerrtc.v4.formal_f2 import sha256_file
from sewerrtc.v4.v42_priority_contract import PFV_CORE_8_IDS

EXPECTED_EVENT_COUNT = 24
EXPECTED_STRATEGIES = (
    "Proposed",
    "EFD",
    "Auto-RBC",
    "All-close",
    "No-control",
    "Internal",
    "Hold",
)
PFV_ABSOLUTE_ALLOWANCE_M3 = 100.0
PFV_RELATIVE_ALLOWANCE_FRACTION = 0.05
DT_SEC = 300


def _finite_float(value: Any, *, name: str) -> float:
    number = float(value)
    if not np.isfinite(number):
        raise ValueError(f"{name} is NaN/Inf")
    return number


def _strategy_summary(frame: pd.DataFrame, strategy: str) -> dict[str, float]:
    part = frame.loc[frame["strategy"].eq(strategy)]
    if part.empty:
        return {}
    return {
        "PFV_event_balanced_mean_m3": float(part["PFV"].mean()),
        "TFV_event_balanced_mean_m3": float(part["TFV"].mean()),
        "global_peak_event_balanced_mean_m3s": float(part["peak_TFV_rate"].mean()),
        "PFV_event_balanced_median_m3": float(part["PFV"].median()),
        "TFV_event_balanced_median_m3": float(part["TFV"].median()),
        "global_peak_event_balanced_median_m3s": float(part["peak_TFV_rate"].median()),
    }


def audit_final_scientific_outcomes(output_root: str | Path) -> dict[str, Any]:
    output_root = Path(output_root)
    paper_root = output_root / "v42_paper"
    formal_root = paper_root / "formal_f2"
    final_evidence = paper_root / "formal_blind/evidence.json"
    ledger_path = formal_root / "paper_execution/FORMAL_EXECUTION_LEDGER.csv"
    reasons: list[str] = []

    if not final_evidence.exists():
        reasons.append("final_heldout_evidence_missing")
    if not ledger_path.exists():
        reasons.append("formal_execution_ledger_missing")
        return {
            "status": "fail",
            "scientific_constraint_pass": False,
            "reasons": reasons,
        }

    try:
        ledger = pd.read_csv(ledger_path, low_memory=False)
    except Exception as exc:
        return {
            "status": "fail",
            "scientific_constraint_pass": False,
            "reasons": reasons + [f"formal_execution_ledger_unreadable:{type(exc).__name__}"],
        }

    required_cols = {
        "role",
        "event_id",
        "strategy",
        "status",
        "authority",
        "detail_path",
        "detail_sha256",
    }
    missing_cols = sorted(required_cols - set(map(str, ledger.columns)))
    if missing_cols:
        return {
            "status": "fail",
            "scientific_constraint_pass": False,
            "reasons": reasons + [f"formal_execution_ledger_missing_columns:{missing_cols}"],
        }

    final = ledger.loc[
        ledger["role"].astype(str).eq("formal_blind")
        & ledger["status"].astype(str).eq("pass")
        & ledger["authority"].astype(str).str.casefold().eq("authoritative_swmm")
    ].copy()
    # Failed/retried rows may remain in an append-only ledger. The most recent
    # passing authoritative row is the evidence authority for each pair.
    final = final.drop_duplicates(["event_id", "strategy"], keep="last")
    event_ids = sorted(set(final["event_id"].astype(str)))
    if len(event_ids) != EXPECTED_EVENT_COUNT:
        reasons.append(
            f"final_event_count_mismatch:expected_{EXPECTED_EVENT_COUNT}:got_{len(event_ids)}"
        )

    rows: list[dict[str, Any]] = []
    for event_id in event_ids:
        for strategy in EXPECTED_STRATEGIES:
            match = final.loc[
                final["event_id"].astype(str).eq(event_id)
                & final["strategy"].astype(str).eq(strategy)
            ]
            if len(match) != 1:
                reasons.append(f"missing_authoritative_final_pair:{event_id}:{strategy}")
                continue
            source = match.iloc[0]
            detail_path = Path(str(source["detail_path"]))
            expected_hash = str(source["detail_sha256"]).strip()
            if not detail_path.exists():
                reasons.append(f"final_detail_missing:{event_id}:{strategy}")
                continue
            if not expected_hash or sha256_file(detail_path) != expected_hash:
                reasons.append(f"final_detail_hash_mismatch:{event_id}:{strategy}")
                continue
            try:
                detail = pd.read_csv(detail_path, low_memory=False)
                kpis = compute_kpis(detail, PFV_CORE_8_IDS, dt_sec=DT_SEC)
                rows.append(
                    {
                        "event_id": event_id,
                        "strategy": strategy,
                        "PFV": _finite_float(kpis["PFV"], name="PFV"),
                        "TFV": _finite_float(kpis["TFV"], name="TFV"),
                        "peak_TFV_rate": _finite_float(
                            kpis["peak_TFV_rate"], name="peak_TFV_rate"
                        ),
                    }
                )
            except Exception as exc:
                reasons.append(
                    f"final_kpi_recompute_failed:{event_id}:{strategy}:{type(exc).__name__}"
                )

    kpi_frame = pd.DataFrame(rows)
    expected_rows = EXPECTED_EVENT_COUNT * len(EXPECTED_STRATEGIES)
    if len(kpi_frame) != expected_rows:
        reasons.append(
            f"final_authoritative_kpi_pair_count_mismatch:expected_{expected_rows}:got_{len(kpi_frame)}"
        )

    per_event: list[dict[str, Any]] = []
    pfv_violation_count = 0
    if not kpi_frame.empty:
        for event_id in event_ids:
            part = kpi_frame.loc[kpi_frame["event_id"].eq(event_id)].set_index("strategy")
            if not {"Proposed", "No-control"}.issubset(part.index):
                continue
            proposed_pfv = float(part.loc["Proposed", "PFV"])
            no_control_pfv = max(0.0, float(part.loc["No-control", "PFV"]))
            margin = proposed_pfv - (1.0 + PFV_RELATIVE_ALLOWANCE_FRACTION) * no_control_pfv
            pfv_pass = margin <= PFV_ABSOLUTE_ALLOWANCE_M3 + 1.0e-6
            if not pfv_pass:
                pfv_violation_count += 1
            proposed_tfv = float(part.loc["Proposed", "TFV"])
            internal_tfv = (
                float(part.loc["Internal", "TFV"])
                if "Internal" in part.index
                else float("nan")
            )
            proposed_peak = float(part.loc["Proposed", "peak_TFV_rate"])
            internal_peak = (
                float(part.loc["Internal", "peak_TFV_rate"])
                if "Internal" in part.index
                else float("nan")
            )
            per_event.append(
                {
                    "event_id": event_id,
                    "proposed_PFV_m3": proposed_pfv,
                    "no_control_PFV_m3": no_control_pfv,
                    "PFV_budget_limit_m3": PFV_ABSOLUTE_ALLOWANCE_M3
                    + (1.0 + PFV_RELATIVE_ALLOWANCE_FRACTION) * no_control_pfv,
                    "PFV_budget_metric_m3": margin,
                    "PFV_constraint_pass": bool(pfv_pass),
                    "proposed_TFV_m3": proposed_tfv,
                    "internal_TFV_m3": internal_tfv,
                    "delta_TFV_vs_internal_m3": proposed_tfv - internal_tfv,
                    "proposed_global_peak_m3s": proposed_peak,
                    "internal_global_peak_m3s": internal_peak,
                    "delta_global_peak_vs_internal_m3s": proposed_peak - internal_peak,
                }
            )

    complete_pairs = len(kpi_frame) == expected_rows and len(event_ids) == EXPECTED_EVENT_COUNT
    scientific_constraint_pass = complete_pairs and pfv_violation_count == 0
    if complete_pairs and pfv_violation_count:
        reasons.append(f"final_PFV_hard_constraint_violations:{pfv_violation_count}")

    summaries = {
        strategy: _strategy_summary(kpi_frame, strategy)
        for strategy in EXPECTED_STRATEGIES
        if not kpi_frame.empty
    }
    tfv_delta_vs_internal_mean = None
    peak_delta_vs_internal_mean = None
    if summaries.get("Proposed") and summaries.get("Internal"):
        tfv_delta_vs_internal_mean = (
            summaries["Proposed"]["TFV_event_balanced_mean_m3"]
            - summaries["Internal"]["TFV_event_balanced_mean_m3"]
        )
        peak_delta_vs_internal_mean = (
            summaries["Proposed"]["global_peak_event_balanced_mean_m3s"]
            - summaries["Internal"]["global_peak_event_balanced_mean_m3s"]
        )

    return {
        "status": "pass" if scientific_constraint_pass and not reasons else "fail",
        "scientific_constraint_pass": scientific_constraint_pass,
        "constraint": "PFV_proposed <= 100 m3 + 1.05 * PFV_no_control on every final event",
        "PFV_absolute_allowance_m3": PFV_ABSOLUTE_ALLOWANCE_M3,
        "PFV_relative_allowance_fraction": PFV_RELATIVE_ALLOWANCE_FRACTION,
        "event_count": len(event_ids),
        "authoritative_strategy_event_pairs": len(kpi_frame),
        "PFV_violation_count": int(pfv_violation_count),
        "PFV_event_pass_fraction": (
            float(1.0 - pfv_violation_count / EXPECTED_EVENT_COUNT)
            if complete_pairs
            else None
        ),
        "strategy_event_balanced_summary": summaries,
        "delta_TFV_vs_internal_event_balanced_mean_m3": tfv_delta_vs_internal_mean,
        "delta_global_peak_vs_internal_event_balanced_mean_m3s_reporting_only": peak_delta_vs_internal_mean,
        "TFV_role": "primary_performance_objective_reporting_not_second_hard_constraint",
        "global_peak_role": "reporting_only",
        "per_event": per_event,
        "reasons": reasons,
    }
