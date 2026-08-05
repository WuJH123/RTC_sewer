"""Read-only rainfall-load and controllability analysis for the Core RTC run.

This script consumes existing authoritative detail/result files only.  It does
not import the controller or start SWMM.  Actual candidate-oracle fields stay
missing when no candidate trajectories for the same events are available.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


EVENT_RE = re.compile(r"^T(?P<rp>[^_]+)_D(?P<duration>[^_]+)_(?P<family>.+)$")


def _num(value: Any) -> float | None:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _parse_inp_limits(path: Path) -> dict[str, float]:
    """Read only junction/storage max-depth columns from an INP file."""
    limits: dict[str, float] = {}
    section = ""
    if not path.exists():
        return limits
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.split(";")[0].strip()
        if not line:
            continue
        if line.startswith("[") and "]" in line:
            section = line[1 : line.index("]")].upper()
            continue
        if section not in {"JUNCTIONS", "STORAGE"}:
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        depth = _num(parts[2])
        if depth is not None and depth > 0:
            limits[parts[0]] = depth
    return limits


def _detail_metrics(detail_path: Path, inp_limits: dict[str, float]) -> dict[str, Any]:
    header = pd.read_csv(detail_path, nrows=0).columns.tolist()
    base = [c for c in ("elapsed_min", "rainfall_mm_h") if c in header]
    h_cols = [c for c in header if c.startswith("h:")]
    flood_cols = [c for c in header if c.startswith("flood:")]
    storage_cols = [c for c in header if c.startswith("storage_volume:")]
    usecols = list(dict.fromkeys(base + h_cols + flood_cols + storage_cols))
    frame = pd.read_csv(detail_path, usecols=usecols, low_memory=False)
    for col in usecols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    elapsed = frame.get("elapsed_min", pd.Series(dtype=float)).to_numpy(dtype=float)
    rain = frame.get("rainfall_mm_h", pd.Series(dtype=float)).fillna(0).to_numpy(dtype=float)
    dt_hours = np.diff(elapsed) / 60.0 if len(elapsed) > 1 else np.array([])
    if len(dt_hours) and np.all(np.isfinite(dt_hours)) and np.nanmedian(dt_hours) > 0:
        total_rain = float(np.sum(np.maximum(rain[:-1], 0) * dt_hours))
        step_hours = float(np.nanmedian(dt_hours))
    else:
        step_hours = 5.0 / 60.0
        total_rain = float(np.sum(np.maximum(rain, 0) * step_hours))
    wet = rain > 1e-9
    wet_mean = float(np.mean(rain[wet])) if np.any(wet) else 0.0
    peak_idx = int(np.nanargmax(rain)) if len(rain) else 0
    peak_position = float(elapsed[peak_idx] / max(elapsed[-1], 1.0)) if len(elapsed) else None

    def matrix_stats(cols: list[str], threshold: float = 1e-9) -> dict[str, Any]:
        if not cols:
            return {"available": False}
        values = frame[cols].to_numpy(dtype=float)
        finite = np.isfinite(values)
        values = np.where(finite, values, 0.0)
        active = values > threshold
        counts = active.sum(axis=1)
        return {
            "available": True,
            "finite_fraction": float(np.mean(finite)),
            "max_active_count": int(np.max(counts)) if len(counts) else 0,
            "mean_active_fraction": float(np.mean(active)),
            "max_active_fraction": float(np.max(active.mean(axis=1))) if len(values) else 0.0,
            "column_count": len(cols),
        }

    flood = matrix_stats(flood_cols)
    storage = matrix_stats(storage_cols, threshold=-np.inf)
    storage_values = (
        frame[storage_cols].to_numpy(dtype=float) if storage_cols else np.empty((0, 0))
    )
    storage_values = storage_values[np.isfinite(storage_values)]
    h_ratio_values: list[float] = []
    for col in h_cols:
        node = col[2:]
        limit = inp_limits.get(node)
        if limit is None:
            continue
        values = pd.to_numeric(frame[col], errors="coerce").to_numpy(dtype=float)
        h_ratio_values.extend((values[np.isfinite(values)] / limit).tolist())
    ratio = np.asarray(h_ratio_values, dtype=float)
    return {
        "detail_path": str(detail_path),
        "elapsed_min": float(elapsed[-1] - elapsed[0]) if len(elapsed) > 1 else None,
        "total_rainfall_mm": total_rain,
        "peak_rainfall_mm_h": float(np.nanmax(rain)) if len(rain) else None,
        "mean_wet_rainfall_mm_h": wet_mean,
        "wet_duration_min": float(np.sum(wet) * step_hours * 60.0),
        "rainfall_peak_position_fraction": peak_position,
        "max_flooded_node_count": flood.get("max_active_count"),
        "max_flooded_node_fraction": flood.get("max_active_fraction"),
        "mean_flooded_node_fraction": flood.get("mean_active_fraction"),
        "flood_finite_fraction": flood.get("finite_fraction"),
        # A junction depth / max-depth ratio is not storage-volume utilization.
        # Keep it as a diagnostic proxy and leave true storage utilization
        # unavailable unless a storage-volume capacity contract is present.
        "max_storage_utilization": None,
        "mean_storage_utilization": None,
        "high_storage_fraction": None,
        "max_depth_fullness_proxy": float(np.max(ratio)) if len(ratio) else None,
        "mean_depth_fullness_proxy": float(np.mean(ratio)) if len(ratio) else None,
        "max_storage_volume_raw": float(np.max(storage_values)) if len(storage_values) else None,
        "storage_finite_fraction": storage.get("finite_fraction"),
        "conveyance_freeboard_proxy_available": False,
    }


def _decision_metrics(proposed_dir: Path, detail_path: Path) -> dict[str, Any]:
    logs = sorted(proposed_dir.glob("decisions*.jsonl"))
    records: list[dict[str, Any]] = []
    for path in logs:
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                records.append(value)
    settings = [c for c in pd.read_csv(detail_path, nrows=0).columns if c.startswith("setting:")]
    k_values: list[int] = []
    if settings:
        frame = pd.read_csv(detail_path, usecols=["elapsed_min"] + settings, low_memory=False)
        for col in settings:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        values = frame[settings].to_numpy(dtype=float)
        elapsed = frame["elapsed_min"].to_numpy(dtype=float)
        for index in range(1, len(values)):
            if elapsed[index] % 10.0 > 1e-6:
                continue
            previous = values[index - 1]
            current = values[index]
            finite = np.isfinite(previous) & np.isfinite(current)
            if np.any(finite):
                k_values.append(int(np.sum(np.abs(current[finite] - previous[finite]) > 1e-6)))
    predicted_safe = []
    predicted_best_tfv = []
    for record in records:
        audits = record.get("candidate_audits") or []
        predicted_safe.append(sum(bool(row.get("safe")) for row in audits if isinstance(row, dict)))
        tfvs = [
            _num(row.get("predicted_tfv_delta_m3"))
            for row in audits
            if isinstance(row, dict)
        ]
        tfvs = [v for v in tfvs if v is not None]
        predicted_best_tfv.append(min(tfvs) if tfvs else None)
    return {
        "decision_log_count": len(records),
        "predicted_safe_candidate_count_mean": float(np.mean(predicted_safe)) if predicted_safe else None,
        "predicted_safe_candidate_count_min": min(predicted_safe) if predicted_safe else None,
        "predicted_best_tfv_delta_mean": float(np.mean([v for v in predicted_best_tfv if v is not None]))
        if any(v is not None for v in predicted_best_tfv) else None,
        "mean_executed_K": float(np.mean(k_values)) if k_values else None,
        "max_executed_K": int(max(k_values)) if k_values else None,
        "executed_K_source": "setting trajectory 10-min deltas" if k_values else None,
        "selected_id_distribution": dict(
            pd.Series([str(r.get("selected_id")) for r in records]).value_counts()
        ) if records else {},
    }


def _rank_percentile(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().sum() <= 1 or numeric.nunique(dropna=True) <= 1:
        return pd.Series(np.where(numeric.notna(), 0.5, np.nan), index=series.index)
    rank = numeric.rank(method="average")
    return (rank - 1.0) / float(numeric.notna().sum() - 1)


def _spearman(x: pd.Series, y: pd.Series) -> float | None:
    part = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(part) < 3 or part["x"].nunique() < 2 or part["y"].nunique() < 2:
        return None
    return float(part["x"].rank().corr(part["y"].rank()))


def _load_oracle(path: Path, events: set[str]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    if not path.exists():
        return {}, {"available": False, "reason": "candidate_oracle_file_missing"}
    frame = pd.read_csv(path, low_memory=False)
    overlap = frame[frame["event_id"].astype(str).isin(events)].copy()
    if overlap.empty:
        return {}, {
            "available": False,
            "reason": "no_overlapping_candidate_trajectories_for_current_events",
            "candidate_source_events": int(frame["event_id"].nunique()),
            "overlap_events": 0,
        }
    output: dict[str, dict[str, Any]] = {}
    for event_id, group in overlap.groupby("event_id"):
        safe = group[group["pfv_safe"].astype(bool)]
        non_hold = safe[safe["candidate_non_hold"].astype(bool)]
        best = pd.to_numeric(non_hold["tfv_delta_m3"], errors="coerce").min()
        output[str(event_id)] = {
            "oracle_status": "available",
            "oracle_candidate_count": int(len(group)),
            "actual_safe_candidate_fraction": float(group["pfv_safe"].mean()),
            "actual_safe_non_hold_count": int(len(non_hold)),
            "oracle_best_tfv_delta_m3": _num(best),
            "oracle_tfv_gain_m3": _num(-best) if _num(best) is not None else None,
        }
    return output, {"available": True, "overlap_events": len(output), "source": str(path)}


def _plot(event_table: pd.DataFrame, path: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - environment dependent
        path.write_text("plot unavailable: " + repr(exc), encoding="utf-8")
        return
    colors = {
        "LOW_LOAD": "#2ca02c",
        "MODERATE_LOAD": "#1f77b4",
        "NEAR_CAPACITY": "#ff7f0e",
        "SEVERE_OVERLOAD": "#d62728",
    }
    fig, ax = plt.subplots(figsize=(8, 5))
    for label, part in event_table.groupby("load_category", dropna=False):
        ax.scatter(part["HYDRAULIC_LOAD_INDEX"], part["TFV_reduction_pct"],
                   label=str(label), color=colors.get(str(label), "#666666"), s=55)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xlabel("HYDRAULIC_LOAD_INDEX (development rank composite)")
    ax.set_ylabel("TFV reduction vs Internal (%)")
    ax.set_title("Rainfall/load versus TFV benefit")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--comparison", type=Path, default=None)
    parser.add_argument("--oracle-candidates", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.project_root.resolve()
    run_root = args.run_root.resolve()
    out = (args.output_dir or run_root / "load_controllability_analysis").resolve()
    out.mkdir(parents=True, exist_ok=True)
    comparison_path = args.comparison or run_root / "CORE_RTC_CHALLENGE_V2_COMPARISON.json"
    comparison = _read_json(comparison_path)
    event_rows = comparison.get("event_rows", [])
    event_ids = {str(row["event_id"]) for row in event_rows}
    manifest_path = root / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/evaluation_inputs/challenge_case_manifest.csv"
    manifest = pd.read_csv(manifest_path, dtype=str)
    manifest = manifest[manifest["event_id"].isin(event_ids)].drop_duplicates("event_id")

    records: list[dict[str, Any]] = []
    for event_id in sorted(event_ids):
        proposed = next((r for r in event_rows if r.get("event_id") == event_id and r.get("strategy") == "Proposed"), None)
        no_control = next((r for r in event_rows if r.get("event_id") == event_id and r.get("strategy") == "No-control"), None)
        internal = next((r for r in event_rows if r.get("event_id") == event_id and r.get("strategy") == "Internal"), None)
        if not proposed or not no_control or not internal:
            continue
        meta = manifest[manifest["event_id"] == event_id]
        inp = Path(str(meta.iloc[0]["inp_path"])) if len(meta) else Path()
        no_detail = Path(str(no_control["detail_path"]))
        int_detail = Path(str(internal["detail_path"]))
        p_detail = Path(str(proposed["detail_path"]))
        limits = _parse_inp_limits(inp)
        load = _detail_metrics(no_detail, limits) if no_detail.exists() else {}
        internal_load = _detail_metrics(int_detail, limits) if int_detail.exists() else {}
        proposed_dir = p_detail.parent
        decision = _decision_metrics(proposed_dir, p_detail) if p_detail.exists() else {}
        match = EVENT_RE.match(event_id)
        record: dict[str, Any] = {
            "event_id": event_id,
            "rainfall_sha256": str(meta.iloc[0]["rainfall_sha256"]) if len(meta) else None,
            "rainfall_family": match.group("family") if match else None,
            "return_period_metadata": match.group("rp") if match else None,
            "rain_duration_metadata_min": _num(meta.iloc[0]["rain_duration_min"]) if len(meta) else None,
            "simulation_duration_min": _num(meta.iloc[0]["simulation_duration_min"]) if len(meta) else None,
            **{f"{key}_no_control": value for key, value in load.items() if key != "detail_path"},
            **{f"{key}_internal": value for key, value in internal_load.items() if key not in {"detail_path", "total_rainfall_mm", "peak_rainfall_mm_h", "mean_wet_rainfall_mm_h", "wet_duration_min", "rainfall_peak_position_fraction"}},
            "PFV_proposed_m3": _num(proposed.get("PFV_m3")),
            "PFV_no_control_m3": _num(no_control.get("PFV_m3")),
            "PFV_budget_m3": _num(proposed.get("PFV_budget_m3")),
            "PFV_margin_m3": _num(proposed.get("PFV_margin_m3")),
            "PFV_pass": bool(proposed.get("PFV_pass")),
            "TFV_proposed_m3": _num(proposed.get("TFV_m3")),
            "TFV_internal_m3": _num(internal.get("TFV_m3")),
            "TFV_gain_m3": _num(internal.get("TFV_m3")) - _num(proposed.get("TFV_m3")) if _num(internal.get("TFV_m3")) is not None and _num(proposed.get("TFV_m3")) is not None else None,
            "TFV_reduction_pct": (100.0 * (_num(internal.get("TFV_m3")) - _num(proposed.get("TFV_m3"))) / _num(internal.get("TFV_m3"))) if _num(internal.get("TFV_m3")) not in (None, 0) and _num(proposed.get("TFV_m3")) is not None else None,
            "fallback_rate": _num(proposed.get("fallback_rate")),
            "active_decision_fraction": 1.0 - _num(proposed.get("fallback_rate")) if _num(proposed.get("fallback_rate")) is not None else None,
            "decision_count": int(proposed.get("decision_count", 0)),
            "action_changes": _num(proposed.get("action_changes")),
            "mean_executed_K": decision.get("mean_executed_K"),
            "max_executed_K": decision.get("max_executed_K"),
            "Global_Peak_proposed": _num(proposed.get("peak_TFV_rate")),
            "Global_Peak_internal": _num(internal.get("peak_TFV_rate")),
            "Global_Peak_no_control": _num(no_control.get("peak_TFV_rate")),
            "Global_Peak_change_vs_internal": _num(proposed.get("peak_TFV_rate")) - _num(internal.get("peak_TFV_rate")) if _num(proposed.get("peak_TFV_rate")) is not None and _num(internal.get("peak_TFV_rate")) is not None else None,
            "oracle_source_detail": "actual candidate trajectories required; runtime predicted audits are not oracle evidence",
        }
        records.append(record)
    table = pd.DataFrame(records)
    if table.empty:
        raise RuntimeError("no complete Proposed/No-control/Internal event rows found")

    load_vars = [
        "total_rainfall_mm_no_control", "peak_rainfall_mm_h_no_control",
        "PFV_no_control_m3", "TFV_internal_m3", "Global_Peak_internal",
        "max_flooded_node_fraction_no_control", "max_storage_utilization_no_control",
    ]
    components = []
    for variable in load_vars:
        if variable in table:
            col = f"load_component_{variable}"
            table[col] = _rank_percentile(table[variable])
            components.append(col)
    table["HYDRAULIC_LOAD_INDEX"] = table[components].mean(axis=1) if components else np.nan
    table["load_index_method"] = "mean rank-percentiles of available No-control/Internal reference hydraulic variables"
    table["load_category"] = pd.qcut(
        table["HYDRAULIC_LOAD_INDEX"].rank(method="first"),
        q=4,
        labels=["LOW_LOAD", "MODERATE_LOAD", "NEAR_CAPACITY", "SEVERE_OVERLOAD"],
    )
    table["load_category"] = table["load_category"].astype(str)

    oracle_path = args.oracle_candidates or root / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2/calibration/CALIBRATION_ORACLE_CANDIDATE_ROWS.csv"
    oracle_by_event, oracle_audit = _load_oracle(oracle_path, event_ids)
    for index, row in table.iterrows():
        oracle = oracle_by_event.get(row["event_id"], {"oracle_status": oracle_audit.get("reason", "unavailable")})
        for key, value in oracle.items():
            table.loc[index, key] = value
        if row["event_id"] not in oracle_by_event:
            table.loc[index, "oracle_TFV_gain_m3"] = np.nan
            table.loc[index, "selection_regret_m3"] = np.nan
            table.loc[index, "controllability_type"] = "UNAVAILABLE_ORACLE_EVIDENCE"

    correlations = {
        "HYDRAULIC_LOAD_INDEX": _spearman(table["HYDRAULIC_LOAD_INDEX"], table["TFV_reduction_pct"]),
        **{variable: _spearman(table[variable], table["TFV_reduction_pct"]) for variable in load_vars if variable in table},
    }
    load_vs = table[["event_id", "HYDRAULIC_LOAD_INDEX", "load_category", "TFV_reduction_pct", "TFV_gain_m3", "PFV_pass"] + [v for v in load_vars if v in table]].copy()
    decision_rows: list[dict[str, Any]] = []
    for _, event in table.iterrows():
        for step in range(int(event["decision_count"] or 0)):
            decision_rows.append({
                "event_id": event["event_id"], "decision_index": step,
                "load_category": event["load_category"], "HYDRAULIC_LOAD_INDEX": event["HYDRAULIC_LOAD_INDEX"],
                "PFV_pass_event": event["PFV_pass"], "selected_TFV_gain_m3": np.nan,
                "actual_safe_candidate_fraction": np.nan, "oracle_TFV_gain_m3": np.nan,
                "selection_regret_m3": np.nan, "controllability_type": "UNAVAILABLE_ORACLE_EVIDENCE",
                "oracle_status": oracle_audit.get("reason", "unavailable"),
            })
    decisions = pd.DataFrame(decision_rows)
    strata = table.groupby("load_category", dropna=False).agg(
        events=("event_id", "nunique"), decisions=("decision_count", "sum"),
        selected_TFV_gain_m3=("TFV_gain_m3", "mean"), fallback_rate=("fallback_rate", "mean"),
        PFV_pass_rate=("PFV_pass", "mean"), active_control_rate=("active_decision_fraction", "mean"),
        oracle_TFV_gain_m3=("oracle_TFV_gain_m3", "mean"),
    ).reset_index()
    strata["actual_safe_candidate_fraction"] = np.nan
    strata["selection_regret_m3"] = np.nan
    strata["oracle_status"] = oracle_audit.get("reason", "unavailable")
    table.to_csv(out / "RAIN_LOAD_EVENT_TABLE.csv", index=False)
    load_vs.to_csv(out / "LOAD_VS_TFV_BENEFIT.csv", index=False)
    decisions.to_csv(out / "RTC_CONTROLLABILITY_DECISION_AUDIT.csv", index=False)
    strata.to_csv(out / "RTC_CONTROLLABILITY_STRATUM_SUMMARY.csv", index=False)
    _plot(table, out / "LOAD_VS_TFV_BENEFIT.png")
    audit = {
        "analysis_id": "RAIN_LOAD_CONTROLLABILITY_ANALYSIS_V1",
        "read_only": True,
        "new_swmm_started": False,
        "run_root": str(run_root),
        "event_count": int(len(table)),
        "load_variables": load_vars,
        "load_index_method": table["load_index_method"].iloc[0],
        "spearman_load_vs_TFV_reduction_pct": correlations,
        "oracle_audit": oracle_audit,
        "oracle_available_for_current_events": bool(oracle_audit.get("overlap_events", 0)),
        "decision_type_counts": decisions["controllability_type"].value_counts().to_dict() if not decisions.empty else {},
        "physical_limit_fraction": None,
        "model_selector_limit_fraction": None,
        "successful_control_fraction": None,
        "interpretation_guard": "No capacity/model/selector type is assigned without same-event authoritative candidate oracle trajectories.",
        "event_rows": table.to_dict(orient="records"),
        "stratum_rows": strata.to_dict(orient="records"),
        "correlations": correlations,
        "outputs": [str(p) for p in sorted(out.iterdir())],
    }
    (out / "RAIN_LOAD_EVENT_TABLE.json").write_text(json.dumps(_json_value(audit), indent=2), encoding="utf-8")
    print(json.dumps(_json_value(audit), indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
