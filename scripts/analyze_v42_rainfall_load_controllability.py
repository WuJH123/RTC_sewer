"""Read-only load/controllability analysis for an existing Core RTC run.

This intentionally does not run SWMM or infer an authoritative candidate oracle.
It reads only the supplied comparison/evidence files and the referenced NC/Internal
detail CSVs.  The load index uses no Proposed outcome.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from statistics import median


def _float(value: object) -> float | None:
    if value in (None, "", "NA", "nan", "NaN"):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _rank_percentile(values: dict[str, float]) -> dict[str, float]:
    ordered = sorted(values.items(), key=lambda item: (item[1], item[0]))
    n = len(ordered)
    return {event: (index / (n - 1) if n > 1 else 0.0) for index, (event, _) in enumerate(ordered)}


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) != len(y) or len(x) < 3:
        return None
    rx = _rank_percentile({str(i): value for i, value in enumerate(x)})
    ry = _rank_percentile({str(i): value for i, value in enumerate(y)})
    xv = [rx[str(i)] for i in range(len(x))]
    yv = [ry[str(i)] for i in range(len(y))]
    mx, my = _mean(xv), _mean(yv)
    assert mx is not None and my is not None
    num = sum((a - mx) * (b - my) for a, b in zip(xv, yv))
    denx = math.sqrt(sum((a - mx) ** 2 for a in xv))
    deny = math.sqrt(sum((b - my) ** 2 for b in yv))
    return num / (denx * deny) if denx and deny else None


def _read_detail(path: Path) -> dict[str, float | int | str | None]:
    import pandas as pd

    header = list(pd.read_csv(path, nrows=0).columns)
    flood = [column for column in header if column.startswith("flood:")]
    storage = [column for column in header if column.startswith("storage_volume:")]
    usecols = [column for column in ("elapsed_min", "rainfall_mm_h", *flood, *storage) if column in header]
    frame = pd.read_csv(path, usecols=usecols)
    elapsed = pd.to_numeric(frame.get("elapsed_min"), errors="coerce")
    rainfall = pd.to_numeric(frame.get("rainfall_mm_h"), errors="coerce").fillna(0.0)
    finite_rain = rainfall.replace([float("inf"), -float("inf")], float("nan")).dropna()
    elapsed_values = elapsed.dropna().tolist()
    diffs = [float(b) - float(a) for a, b in zip(elapsed_values, elapsed_values[1:]) if float(b) > float(a)]
    dt_min = median(diffs) if diffs else 10.0
    wet = finite_rain[finite_rain > 0]
    total_depth = float(finite_rain.sum()) * dt_min / 60.0
    peak = float(finite_rain.max()) if len(finite_rain) else None
    wet_duration = float(len(wet) * dt_min)
    peak_position = None
    if peak is not None and elapsed.notna().any():
        peak_position = float(elapsed.iloc[int(finite_rain.idxmax())])
    flood_count = None
    flood_fraction = None
    if flood:
        flood_values = frame[flood].apply(pd.to_numeric, errors="coerce")
        flood_count_series = (flood_values.fillna(0.0) > 0.0).sum(axis=1)
        flood_count = int(flood_count_series.max())
        flood_fraction = float((flood_count_series / len(flood)).max()) if flood else None
    storage_max = None
    storage_mean = None
    if storage:
        storage_values = frame[storage].apply(pd.to_numeric, errors="coerce")
        total_storage = storage_values.sum(axis=1, skipna=True)
        storage_max = float(total_storage.max())
        storage_mean = float(total_storage.mean())
    return {
        "rainfall_total_depth_mm": total_depth,
        "rainfall_peak_mm_h": peak,
        "rainfall_wet_duration_min": wet_duration,
        "rainfall_mean_wet_intensity_mm_h": float(wet.mean()) if len(wet) else None,
        "rainfall_peak_position_min": peak_position,
        "detail_rows": int(len(frame)),
        "detail_dt_min": dt_min,
        "flood_node_count_max": flood_count,
        "flooded_node_fraction_max": flood_fraction,
        "storage_volume_sum_max_observed": storage_max,
        "storage_volume_sum_mean_observed": storage_mean,
        "storage_utilization_proxy_available": False,
        "conveyance_slack_available": False,
    }


def _parse_event(event_id: str) -> dict[str, object]:
    match = re.match(r"T(?P<return_period>\d+)_D(?P<duration>\d+)_(?P<family>.+)$", event_id)
    if not match:
        return {"rainfall_return_period": None, "rainfall_declared_duration_min": None, "rainfall_family": None}
    return {
        "rainfall_return_period": int(match.group("return_period")),
        "rainfall_declared_duration_min": int(match.group("duration")),
        "rainfall_family": match.group("family"),
    }


def _read_decisions(path: Path) -> dict[str, object]:
    result = {"decision_count": 0, "predicted_safe_candidate_count": 0, "predicted_nonhold_safe_candidate_count": 0}
    if not path.exists():
        return result
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        result["decision_count"] += 1
        audits = row.get("candidate_audits") or []
        result["predicted_safe_candidate_count"] += sum(bool(item.get("safe")) for item in audits)
        result["predicted_nonhold_safe_candidate_count"] += sum(
            bool(item.get("safe")) and item.get("candidate_id") not in {"hold_native", "frozen_hold_readback"}
            for item in audits
        )
    return result


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--comparison", type=Path, required=True)
    ap.add_argument("--evidence", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    args = ap.parse_args()

    with args.comparison.open(encoding="utf-8", newline="") as handle:
        comparison = list(csv.DictReader(handle))
    by_event: dict[str, dict[str, dict[str, object]]] = {}
    for row in comparison:
        by_event.setdefault(row["event_id"], {})[row["strategy"]] = row

    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    evidence_events = {row["event_id"]: row for row in evidence.get("per_event", [])}
    event_rows: list[dict[str, object]] = []
    for event_id in sorted(by_event):
        strategies = by_event[event_id]
        nc = strategies.get("No-control", {})
        internal = strategies.get("Internal", {})
        proposed = strategies.get("Proposed", {})
        nc_path = Path(str(nc.get("detail_path", "")))
        internal_path = Path(str(internal.get("detail_path", "")))
        forcing = _read_detail(nc_path)
        internal_detail = _read_detail(internal_path)
        event_meta = _parse_event(event_id)
        nc_pfv = _float(nc.get("PFV_m3"))
        int_tfv = _float(internal.get("TFV_m3"))
        prop_tfv = _float(proposed.get("TFV_m3"))
        prop_pfv = _float(proposed.get("PFV_m3"))
        budget = _float(proposed.get("PFV_budget_m3"))
        gain = int_tfv - prop_tfv if int_tfv is not None and prop_tfv is not None else None
        reduction = 100.0 * gain / int_tfv if gain is not None and int_tfv else None
        # detail.csv -> strategy backend -> strategy -> event directory
        decision_meta = _read_decisions(nc_path.parents[2] / "Proposed" / "gat_sparse_reconstruction" / "decisions.jsonl")
        event_evidence = evidence_events.get(event_id, {})
        row: dict[str, object] = {
            "event_id": event_id,
            **event_meta,
            **forcing,
            "internal_detail_rows": internal_detail.get("detail_rows"),
            "PFV_no_control_m3": nc_pfv,
            "TFV_internal_m3": int_tfv,
            "global_peak_internal_TFV_rate": _float(internal.get("peak_TFV_rate")),
            "flood_node_count_max_internal": internal_detail.get("flood_node_count_max"),
            "flooded_node_fraction_max_internal": internal_detail.get("flooded_node_fraction_max"),
            "storage_volume_sum_max_observed_internal": internal_detail.get("storage_volume_sum_max_observed"),
            "PFV_proposed_m3": prop_pfv,
            "PFV_budget_m3": budget,
            "PFV_margin_m3": _float(proposed.get("PFV_margin_m3")),
            "PFV_pass": proposed.get("PFV_pass"),
            "TFV_proposed_m3": prop_tfv,
            "TFV_gain_m3": gain,
            "TFV_reduction_pct": reduction,
            "fallback_rate": _float(proposed.get("fallback_rate")),
            "action_changes": _float(proposed.get("action_changes")),
            "decision_count": event_evidence.get("decision_count", decision_meta["decision_count"]),
            "active_nonfallback_decisions": event_evidence.get("active_nonfallback_decisions"),
            "active_decision_fraction": (
                float(event_evidence["active_nonfallback_decisions"]) / float(event_evidence["decision_count"])
                if event_evidence.get("decision_count") and event_evidence.get("active_nonfallback_decisions") is not None
                else None
            ),
            "predicted_safe_candidate_count_sum": decision_meta["predicted_safe_candidate_count"],
            "predicted_nonhold_safe_candidate_count_sum": decision_meta["predicted_nonhold_safe_candidate_count"],
            "physical_oracle_status": "UNAVAILABLE_NO_AUTHORITATIVE_CANDIDATE_OUTCOME_MAPPING",
            "oracle_TFV_gain_m3": None,
            "selection_regret_m3": None,
            "controllability_classification": "UNRESOLVED_WITHOUT_PHYSICAL_ORACLE",
            "internal_detail_path": str(internal_path),
            "no_control_detail_path": str(nc_path),
        }
        event_rows.append(row)

    load_variables = [
        "rainfall_total_depth_mm",
        "rainfall_peak_mm_h",
        "PFV_no_control_m3",
        "TFV_internal_m3",
        "global_peak_internal_TFV_rate",
        "flooded_node_fraction_max_internal",
        "storage_volume_sum_max_observed_internal",
    ]
    ranks: dict[str, dict[str, float]] = {}
    for variable in load_variables:
        values = {str(row["event_id"]): float(row[variable]) for row in event_rows if _float(row.get(variable)) is not None}
        ranks[variable] = _rank_percentile(values)
    for row in event_rows:
        event_id = str(row["event_id"])
        scores = [rank[event_id] for rank in ranks.values() if event_id in rank]
        row["hydraulic_load_index"] = _mean(scores)
        row["hydraulic_load_variable_count"] = len(scores)
    ordered = sorted(event_rows, key=lambda row: (float(row["hydraulic_load_index"]), str(row["event_id"])))
    labels = ["LOW_LOAD", "MODERATE_LOAD", "NEAR_CAPACITY", "SEVERE_OVERLOAD"]
    for index, row in enumerate(ordered):
        row["load_regime"] = labels[min(len(labels) - 1, index * len(labels) // len(ordered))]

    benefit_rows = [row for row in event_rows if _float(row.get("hydraulic_load_index")) is not None and _float(row.get("TFV_reduction_pct")) is not None]
    load_benefit = [{
        "event_id": row["event_id"],
        "load_regime": row["load_regime"],
        "hydraulic_load_index": row["hydraulic_load_index"],
        "TFV_gain_m3": row["TFV_gain_m3"],
        "TFV_reduction_pct": row["TFV_reduction_pct"],
        "PFV_no_control_m3": row["PFV_no_control_m3"],
        "TFV_internal_m3": row["TFV_internal_m3"],
    } for row in benefit_rows]
    correlations = {
        "hydraulic_load_index_vs_TFV_reduction_pct": _spearman(
            [float(row["hydraulic_load_index"]) for row in benefit_rows],
            [float(row["TFV_reduction_pct"]) for row in benefit_rows],
        )
    }
    for variable in ["rainfall_total_depth_mm", "rainfall_peak_mm_h", "PFV_no_control_m3", "TFV_internal_m3", "flooded_node_fraction_max_internal", "storage_volume_sum_max_observed_internal"]:
        pairs = [(float(row[variable]), float(row["TFV_reduction_pct"])) for row in benefit_rows if _float(row.get(variable)) is not None]
        correlations[f"{variable}_vs_TFV_reduction_pct"] = _spearman([a for a, _ in pairs], [b for _, b in pairs])

    regime_rows: list[dict[str, object]] = []
    for regime in labels:
        rows = [row for row in event_rows if row.get("load_regime") == regime]
        gains = [float(row["TFV_reduction_pct"]) for row in rows if _float(row.get("TFV_reduction_pct")) is not None]
        regime_rows.append({
            "load_regime": regime,
            "events": len(rows),
            "PFV_pass_events": sum(str(row.get("PFV_pass")).lower() == "true" for row in rows),
            "TFV_reduction_pct_mean": _mean(gains),
            "TFV_reduction_pct_median": median(gains) if gains else None,
            "active_decision_fraction_mean": _mean([float(row["active_decision_fraction"]) for row in rows if _float(row.get("active_decision_fraction")) is not None]),
            "fallback_rate_mean": _mean([float(row["fallback_rate"]) for row in rows if _float(row.get("fallback_rate")) is not None]),
            "physical_oracle_available": False,
        })

    decision_audit = [{
        "event_id": row["event_id"],
        "load_regime": row["load_regime"],
        "hydraulic_load_index": row["hydraulic_load_index"],
        "decision_count": row["decision_count"],
        "predicted_safe_candidate_count_sum": row["predicted_safe_candidate_count_sum"],
        "predicted_nonhold_safe_candidate_count_sum": row["predicted_nonhold_safe_candidate_count_sum"],
        "selected_TFV_gain_m3": row["TFV_gain_m3"],
        "oracle_TFV_gain_m3": None,
        "selection_regret_m3": None,
        "physical_oracle_status": row["physical_oracle_status"],
        "controllability_classification": row["controllability_classification"],
    } for row in event_rows]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "RAIN_LOAD_EVENT_TABLE.csv", event_rows)
    _write_csv(args.output_dir / "LOAD_VS_TFV_BENEFIT.csv", load_benefit)
    _write_csv(args.output_dir / "RTC_CONTROLLABILITY_DECISION_AUDIT.csv", decision_audit)
    (args.output_dir / "RAIN_LOAD_EVENT_TABLE.json").write_text(json.dumps(event_rows, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "LOAD_VS_TFV_BENEFIT.json").write_text(json.dumps({"correlations": correlations, "rows": load_benefit}, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "RTC_CONTROLLABILITY_DECISION_AUDIT.json").write_text(json.dumps({"status": "diagnostic", "oracle": "unavailable", "rows": decision_audit}, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "RAIN_LOAD_CONTROLLABILITY_ANALYSIS.json").write_text(json.dumps({
        "contract": "RAIN_LOAD_CONTROLLABILITY_ANALYSIS_V1",
        "status": "pass_diagnostic",
        "swmm_started": False,
        "uses_proposed_outcome_in_load_index": False,
        "event_count": len(event_rows),
        "correlations": correlations,
        "regimes": regime_rows,
        "physical_oracle_status": "UNAVAILABLE_NO_AUTHORITATIVE_CANDIDATE_OUTCOME_MAPPING",
        "oracle_requirement": "authoritative candidate outcome trajectories keyed to these 12 event/state/action decisions",
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    try:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(8, 5))
        for row in event_rows:
            plt.scatter(float(row["hydraulic_load_index"]), float(row["TFV_reduction_pct"]), s=35)
            plt.annotate(str(row["event_id"]), (float(row["hydraulic_load_index"]), float(row["TFV_reduction_pct"])), fontsize=6)
        plt.axhline(0.0, color="black", linewidth=0.7)
        plt.xlabel("Hydraulic load index (development percentile composite)")
        plt.ylabel("TFV reduction vs Internal (%)")
        plt.title("Core RTC: hydraulic load vs TFV benefit")
        plt.tight_layout()
        plt.savefig(args.output_dir / "LOAD_VS_TFV_BENEFIT.png", dpi=160)
        plt.close()
    except Exception as exc:  # plotting is diagnostic, never a scientific gate
        (args.output_dir / "LOAD_VS_TFV_BENEFIT_PLOT_WARNING.txt").write_text(str(exc), encoding="utf-8")
    print(json.dumps({"events": len(event_rows), "correlations": correlations, "oracle": "unavailable"}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
