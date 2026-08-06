"""Read-only PFV--TFV oracle frontier audit.

This audit uses only recorded authoritative candidate trajectories.  It never
trains a model, starts SWMM, or changes the online PFV contract.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sewerrtc.v4.v42_priority_contract import get_pfv_core_node_indices
from sewerrtc.v4.v42_trajectory_builder import _load_graph_topology
from sewerrtc.control.authoritative_control_metrics_v42 import trajectory_metrics


DT_SEC = 600.0
RELATIVE_GRID = (0.0, 0.025, 0.05, 0.075, 0.10, 0.15)
ABSOLUTE_GRID = (0.0, 100.0, 250.0, 500.0, 1000.0, 2000.0)
TARGETS = (5.0, 10.0, 20.0, 30.0)


def _array(value: Any) -> np.ndarray:
    if isinstance(value, str):
        value = json.loads(value)
    return np.asarray(value, dtype=np.float64)


def _finite(value: Any) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _safe_stats(values: Sequence[float]) -> Dict[str, Optional[float]]:
    arr = np.asarray([float(x) for x in values if math.isfinite(float(x))], dtype=float)
    if arr.size == 0:
        return {key: None for key in ("mean", "median", "p25", "p75", "p90", "max")}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def _percentile_rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(np.argsort(values, kind="stable"), kind="stable")
    if len(values) <= 1:
        return np.zeros(len(values), dtype=float)
    return order.astype(float) / float(len(values) - 1)


def _non_hold(row: pd.Series) -> bool:
    try:
        candidate = _array(row["action_candidate_readback"])
        hold = _array(row["action_hold_previous_readback"])
        return bool(np.max(np.abs(candidate[:3] - hold[:3])) > 1.0e-6)
    except (KeyError, TypeError, ValueError):
        return False


def _reference_metrics(row: pd.Series, priority: Sequence[int]) -> Dict[str, Any]:
    no_control_flood = _array(row["trajectory_flood_no_control"])
    internal_flood = _array(row["trajectory_flood_dynamic_internal"])
    candidate_flood = _array(row["trajectory_flood_candidate"])
    no_control_metrics = trajectory_metrics(no_control_flood, priority, dt_sec=DT_SEC)
    internal_metrics = trajectory_metrics(internal_flood, priority, dt_sec=DT_SEC)
    candidate_metrics = trajectory_metrics(candidate_flood, priority, dt_sec=DT_SEC)
    no_control_pfv = no_control_metrics["PFV"]
    internal_tfv = internal_metrics["TFV"]
    internal_peak = internal_metrics["peak_TFV_rate"]
    flooded_count = (internal_flood > 0.0).sum(axis=1)
    node_count = max(1, int(internal_flood.shape[1]))
    storage_max = None
    storage_column = "trajectory_storage_volume_no_control"
    if storage_column in row.index and bool(row.get("trajectory_storage_volume_no_control_available", False)):
        try:
            storage_max = float(np.sum(_array(row[storage_column]), axis=1).max())
        except (TypeError, ValueError):
            storage_max = None
    candidate_pfv = candidate_metrics["PFV"]
    candidate_tfv = candidate_metrics["TFV"]
    candidate_peak = candidate_metrics["peak_TFV_rate"]
    return {
        "event_id": str(row["event_id"]),
        "rainfall_sha256": str(row["rainfall_sha256"]),
        "state_key": str(row["state_key"]),
        "checkpoint_min": float(row["checkpoint_min"]),
        "pfv_no_control_m3": no_control_pfv,
        "pfv_candidate_m3": candidate_pfv,
        "pfv_excess_m3": candidate_pfv - no_control_pfv,
        "tfv_internal_m3": internal_tfv,
        "tfv_candidate_m3": candidate_tfv,
        "tfv_reduction_pct": 100.0 * (internal_tfv - candidate_tfv) / max(abs(internal_tfv), 1.0e-9),
        "global_peak_internal_rate": internal_peak,
        "global_peak_candidate_rate": candidate_peak,
        "max_flooded_node_count_internal": int(flooded_count.max()),
        "max_flooded_node_fraction_internal": float(flooded_count.max() / node_count),
        "storage_volume_sum_max_no_control": storage_max,
        "candidate_non_hold": _non_hold(row),
        "candidate_action_sha256": str(row.get("candidate_action_sha256", "")),
    }


def _load_states(manifest: Path, project_root: Path) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    frame = pd.read_parquet(manifest).reset_index(drop=True)
    required = {
        "state_key",
        "event_id",
        "rainfall_sha256",
        "checkpoint_min",
        "pfv_delta",
        "tfv_delta",
        "peak_delta",
        "trajectory_flood_no_control",
        "trajectory_flood_dynamic_internal",
        "trajectory_flood_candidate",
        "action_candidate_readback",
        "action_hold_previous_readback",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"manifest missing required columns: {missing}")
    graph = _load_graph_topology(project_root)
    priority = get_pfv_core_node_indices(list(graph["node_ids"]))
    records: List[Dict[str, Any]] = []
    for state_key, group in frame.groupby("state_key", sort=True):
        row = group.iloc[0]
        metrics = _reference_metrics(row, priority)
        metrics["candidate_count"] = int(len(group))
        metrics["state_key"] = str(state_key)
        records.append(metrics)
    states = pd.DataFrame(records)
    variables = [
        "pfv_no_control_m3",
        "tfv_internal_m3",
        "global_peak_internal_rate",
        "max_flooded_node_fraction_internal",
        "storage_volume_sum_max_no_control",
    ]
    ranks = []
    for name in variables:
        values = pd.to_numeric(states[name], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(values)
        if mask.any():
            rank = np.full(len(values), np.nan, dtype=float)
            rank[mask] = _percentile_rank(values[mask])
            ranks.append(rank)
    states["hydraulic_load_index"] = np.nanmean(np.vstack(ranks), axis=0)
    states["load_regime"] = pd.qcut(
        states["hydraulic_load_index"].rank(method="first"),
        q=4,
        labels=["LOW_LOAD", "MODERATE_LOAD", "NEAR_CAPACITY", "SEVERE_OVERLOAD"],
    ).astype(str)
    states = states.drop(columns=["candidate_count"], errors="ignore")
    # Keep row-level candidate metrics from the current manifest row.  The
    # state table is built from the first row only for load stratification;
    # copying the whole record here silently replaced every candidate's PFV
    # and TFV with that first candidate.
    state_lookup = states.set_index("state_key")[["hydraulic_load_index", "load_regime"]].to_dict(orient="index")
    row_records: List[Dict[str, Any]] = []
    for _, row in frame.iterrows():
        item = _reference_metrics(row, priority)
        item.update(state_lookup[item["state_key"]])
        row_records.append(item)
    rows = pd.DataFrame(row_records)
    return rows, {
        "input_rows": int(len(frame)),
        "input_states": int(len(states)),
        "input_rainfall_groups": int(frame["rainfall_sha256"].nunique()),
        "priority_node_count": int(len(priority)),
        "state_reference_table": states,
    }


def _state_results(rows: pd.DataFrame, relative: float, absolute: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rows = rows.copy()
    rows["pfv_limit_m3"] = (1.0 + relative) * rows["pfv_no_control_m3"] + absolute
    rows["actual_safe"] = rows["pfv_candidate_m3"] <= rows["pfv_limit_m3"] + 1.0e-9
    rows["safe_non_hold"] = rows["actual_safe"] & rows["candidate_non_hold"]
    rows["safe_tfv_improving"] = rows["actual_safe"] & (rows["tfv_reduction_pct"] > 0.0)
    state_rows: List[Dict[str, Any]] = []
    for state_key, group in rows.groupby("state_key", sort=True):
        safe = group[group["actual_safe"]]
        safe_non_hold = safe[safe["candidate_non_hold"]]
        if safe.empty:
            best = None
        else:
            best = safe.sort_values(["tfv_candidate_m3", "candidate_action_sha256"], kind="stable").iloc[0]
        result: Dict[str, Any] = {
            "state_key": str(state_key),
            "event_id": str(group.iloc[0]["event_id"]),
            "rainfall_sha256": str(group.iloc[0]["rainfall_sha256"]),
            "load_regime": str(group.iloc[0]["load_regime"]),
            "hydraulic_load_index": float(group.iloc[0]["hydraulic_load_index"]),
            "candidate_count": int(len(group)),
            "actual_safe_candidate_count": int(len(safe)),
            "actual_safe_non_hold_count": int(len(safe_non_hold)),
            "actual_safe_tfv_improving_count": int((safe["tfv_reduction_pct"] > 0.0).sum()),
            "oracle_available": bool(best is not None),
            "oracle_tfv_reduction_pct": float(best["tfv_reduction_pct"]) if best is not None else None,
            "oracle_tfv_candidate_m3": float(best["tfv_candidate_m3"]) if best is not None else None,
            "oracle_pfv_excess_m3": float(best["pfv_excess_m3"]) if best is not None else None,
            "oracle_candidate_action_sha256": str(best["candidate_action_sha256"]) if best is not None else None,
        }
        state_rows.append(result)
    return rows, pd.DataFrame(state_rows)


def _aggregate(states: pd.DataFrame, rows: pd.DataFrame, relative: float, absolute: float) -> Dict[str, Any]:
    available = states[states["oracle_available"]]
    reductions = available["oracle_tfv_reduction_pct"].astype(float).tolist()
    event_means = available.groupby("event_id")["oracle_tfv_reduction_pct"].mean()
    safe_rows = rows[rows["actual_safe"]]
    result: Dict[str, Any] = {
        "relative_margin_fraction": relative,
        "absolute_margin_m3": absolute,
        "states": int(len(states)),
        "actual_safe_rows": int(rows["actual_safe"].sum()),
        "actual_safe_states": int(states["oracle_available"].sum()),
        "states_with_safe_non_hold": int((states["actual_safe_non_hold_count"] > 0).sum()),
        "states_with_safe_tfv_improvement": int((states["actual_safe_tfv_improving_count"] > 0).sum()),
        "oracle_tfv_reduction_pct": _safe_stats(reductions),
        "oracle_tfv_reduction_event_balanced_mean_pct": float(event_means.mean()) if len(event_means) else None,
        "fraction_states_with_oracle": float(len(available) / len(states)) if len(states) else 0.0,
        "fraction_states_tfv_reduction_ge_5_pct": float((available["oracle_tfv_reduction_pct"] >= 5.0).sum() / len(states)) if len(states) else 0.0,
        "fraction_states_tfv_reduction_ge_10_pct": float((available["oracle_tfv_reduction_pct"] >= 10.0).sum() / len(states)) if len(states) else 0.0,
        "fraction_states_tfv_reduction_ge_20_pct": float((available["oracle_tfv_reduction_pct"] >= 20.0).sum() / len(states)) if len(states) else 0.0,
        "fraction_states_tfv_reduction_ge_30_pct": float((available["oracle_tfv_reduction_pct"] >= 30.0).sum() / len(states)) if len(states) else 0.0,
        "safe_pfv_excess_m3": _safe_stats(safe_rows["pfv_excess_m3"].tolist()),
        "all_candidate_pfv_excess_m3": _safe_stats(rows["pfv_excess_m3"].tolist()),
    }
    for regime in ("LOW_LOAD", "MODERATE_LOAD", "NEAR_CAPACITY", "SEVERE_OVERLOAD"):
        all_subset = states[states["load_regime"] == regime]
        subset = available[available["load_regime"] == regime]
        result[regime] = {
            "states": int(len(all_subset)),
            "oracle_available_states": int(len(subset)),
            "oracle_available_fraction": float(len(subset) / len(all_subset)) if len(all_subset) else None,
            "oracle_mean_reduction_pct": float(subset["oracle_tfv_reduction_pct"].mean()) if len(subset) else None,
            "oracle_median_reduction_pct": float(subset["oracle_tfv_reduction_pct"].median()) if len(subset) else None,
            "fraction_ge_20_pct_of_all_states": float((subset["oracle_tfv_reduction_pct"] >= 20.0).sum() / len(all_subset)) if len(all_subset) else None,
            "fraction_ge_20_pct_of_oracle_states": float((subset["oracle_tfv_reduction_pct"] >= 20.0).mean()) if len(subset) else None,
            "safe_non_hold_state_fraction": float((all_subset["actual_safe_non_hold_count"] > 0).mean()) if len(all_subset) else None,
        }
    return result


def _rolling_semantics(project_root: Path) -> Dict[str, Any]:
    runtime = (project_root / "sewerrtc/v4/v42_formal_runtime.py").read_text(encoding="utf-8")
    safe_runtime = (project_root / "sewerrtc/v4/v42_formal_runtime_safe.py").read_text(encoding="utf-8")
    patch = (project_root / "sewerrtc/v4/v42_pfv_tfv_runtime_patch.py").read_text(encoding="utf-8")
    rolling = (project_root / "sewerrtc/control/rolling_pfv_budget_v42.py").read_text(encoding="utf-8")
    integrated = all(
        token in safe_runtime + patch + rolling
        for token in ("RollingPfvBudgetState", "rolling_pfv_budget_state", "realised_prefix_budget_metric_m3")
    )
    return {
        "online_pfv_candidate_scope": "realised_causal_prefix_plus_predicted_H120" if integrated else "predicted_H120_candidate_branch",
        "online_pfv_no_control_scope": "realised_causal_prefix_plus_predicted_H120" if integrated else "predicted_H120_no_control_branch",
        "realized_prefix_in_online_pfv": integrated,
        "decision_replan_interval": "10_min",
        "allowance_reinitialized_each_decision": False if integrated else "_pfv_budget_metric_ucb is recomputed per predict_and_decide call",
        "cumulative_event_pfv_budget_tracking": integrated,
        "whole_event_authoritative_pfv_recomputed_after_run": "compute_kpis" in runtime and "PFV" in runtime,
        "online_future_hydraulic_truth_used": False,
        "online_realized_future_rainfall_used": False,
        "semantic_alignment": integrated,
        "status": "IMPLEMENTED_CODE_AUDIT_ONLY" if integrated else "FAIL_CLOSED_REVIEW_REQUIRED",
        "reason": (
            "Cumulative PFV state includes realised causal prefix and predicted future allowance; "
            "this audit did not start SWMM, so runtime evidence remains pending."
            if integrated else
            "H120 PFV admission is reset at each decision and no cumulative event-level allowance is tracked online."
        ),
    }


def _split_authority(project_root: Path, manifest: Path) -> Dict[str, Any]:
    """Report model split metadata separately from PFV calibration authority."""
    formal = project_root / "outputs/project6_dual_reference_v4/final_v4/v42_paper/formal_f2"
    model_root = formal / "step2/repair_full_control_core_v1/models_action_effect_residual_v4"
    reports: List[Dict[str, Any]] = []
    for seed in (17, 42, 73):
        path = model_root / f"seed_{seed}/formal_step2_report.json"
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        groups = {
            key: sorted({str(x) for x in payload.get(key, [])})
            for key in ("train_rainfall_groups", "validation_rainfall_groups", "calibration_rainfall_groups")
        }
        reports.append({
            "seed": seed,
            "path": str(path),
            "model_sha256": payload.get("surrogate_model_sha256"),
            "counts": {key: len(value) for key, value in groups.items()},
            "groups": groups,
            "within_report_overlap": {
                "train__validation": len(set(groups["train_rainfall_groups"]) & set(groups["validation_rainfall_groups"])),
                "train__calibration": len(set(groups["train_rainfall_groups"]) & set(groups["calibration_rainfall_groups"])),
                "validation__calibration": len(set(groups["validation_rainfall_groups"]) & set(groups["calibration_rainfall_groups"])),
            },
        })

    frame_groups = set(map(str, pd.read_parquet(manifest, columns=["rainfall_sha256"])["rainfall_sha256"].dropna().unique()))
    calibration_paths = [
        formal / "diagnostics/PFV_ONLY_SAFETY_CALIBRATION_ACTION_EFFECT_RESIDUAL_V4_ABSOLUTE.json",
        formal / "pfv_only_v2/FRESH_PFV_ONLY_SAFETY_CALIBRATION.json",
        formal / "calibration/PFV_ONLY_SAFETY_CALIBRATION.json",
    ]
    artifacts: List[Dict[str, Any]] = []
    for path in calibration_paths:
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        groups = sorted({str(x) for x in payload.get("calibration_rainfall_groups", [])})
        artifacts.append({
            "path": str(path),
            "status": payload.get("status"),
            "method": payload.get("pfv_budget_metric_ucb_method"),
            "calibration_group_count": len(groups),
            "calibration_rainfall_groups": groups,
            "manifest_group_overlap": len(set(groups) & frame_groups),
            "training_or_internal_validation_overlap_count": payload.get("training_or_internal_validation_overlap_count"),
            "model_hashes": payload.get("model_hashes"),
            "model_sha256_by_seed": payload.get("model_sha256_by_seed"),
        })
    selected = next(
        (item for item in artifacts if "ACTION_EFFECT_RESIDUAL_V4_ABSOLUTE" in item["path"]),
        artifacts[0] if artifacts else None,
    )
    model_train_val = set()
    if reports:
        model_train_val = set(reports[0]["groups"]["train_rainfall_groups"]) | set(reports[0]["groups"]["validation_rainfall_groups"])
    selected_groups = set(selected["calibration_rainfall_groups"]) if selected else set()
    expected_model_hashes = {
        str(item["seed"]): str(item.get("model_sha256") or "")
        for item in reports
    }
    selected_model_hashes = {
        str(key): str(value)
        for key, value in dict(selected.get("model_hashes") or {}).items()
    } if selected else {}
    return {
        "manifest": str(manifest),
        "manifest_rainfall_group_count": len(frame_groups),
        "model_reports": reports,
        "model_report_count": len(reports),
        "model_split_consistent_across_seeds": bool(
            reports and all(item["groups"] == reports[0]["groups"] for item in reports[1:])
        ),
        "operational_pfv_calibration": selected,
        "calibration_artifacts_found": artifacts,
        "model_split_vs_operational_calibration": {
            "model_train_groups": len(model_train_val & set(reports[0]["groups"]["train_rainfall_groups"])) if reports else 0,
            "model_validation_groups": len(set(reports[0]["groups"]["validation_rainfall_groups"])) if reports else 0,
            "operational_calibration_groups": len(selected_groups),
            "overlap_with_model_train_or_validation": len(selected_groups & model_train_val),
            "model_report_calibration_group_count": reports[0]["counts"]["calibration_rainfall_groups"] if reports else None,
            "operational_calibration_group_count": len(selected_groups),
            "model_hashes_match": bool(expected_model_hashes and expected_model_hashes == selected_model_hashes),
        },
        "authority_consistent": bool(
            reports
            and selected
            and all(not any(item["within_report_overlap"].values()) for item in reports)
            and len(selected_groups & model_train_val) == 0
            and len(selected_groups) >= 12
            and expected_model_hashes == selected_model_hashes
        ),
        "interpretation": (
            "Model reports use an internal 65/8/8 train/validation/calibration split; "
            "the operational PFV calibration artifact uses an independent 12-group set. "
            "These authorities are recorded separately and must not be silently merged."
        ),
    }


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_markdown(path: Path, summary: Dict[str, Any], rolling: Dict[str, Any]) -> None:
    lines = [
        "# PFV–TFV Authoritative Pareto Frontier",
        "",
        "Read-only audit of recorded authoritative candidate trajectories; no training or SWMM was started.",
        "",
        f"- Input rows/states/groups: {summary['input_rows']} / {summary['input_states']} / {summary['input_rainfall_groups']}",
        f"- Path decision: **{summary['path_decision']}**",
        f"- 20% target physically supported in current candidate space: **{summary['twenty_pct_physically_supported']}**",
        "",
        "## Grid summary",
        "",
        "| Relative δ | Absolute B (m³) | Safe rows | Safe states | Safe non-Hold states | States ≥20% TFV reduction | Oracle mean reduction | LOW/MODERATE median |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in summary["grid_summary"]:
        low = item.get("LOW_LOAD", {}).get("oracle_median_reduction_pct")
        moderate = item.get("MODERATE_LOAD", {}).get("oracle_median_reduction_pct")
        low_moderate = [x for x in (low, moderate) if x is not None]
        lm = float(np.mean(low_moderate)) if low_moderate else None
        lines.append(
            f"| {item['relative_margin_fraction']:.3f} | {item['absolute_margin_m3']:.0f} | {item['actual_safe_rows']} | {item['actual_safe_states']} | {item['states_with_safe_non_hold']} | {item['fraction_states_tfv_reduction_ge_20_pct']:.3f} | {item['oracle_tfv_reduction_pct']['mean']} | {lm} |"
        )
    lines += [
        "",
        "## Rolling PFV semantics",
        "",
        f"- Online candidate scope: `{rolling['online_pfv_candidate_scope']}`",
        f"- Allowance reset per decision: `{rolling['allowance_reinitialized_each_decision']}`",
        f"- Cumulative event budget tracked online: `{rolling['cumulative_event_pfv_budget_tracking']}`",
        f"- Semantic result: **{rolling['status']}**",
        "",
        "The whole-event authoritative PFV is recomputed after a run, but that does not provide an online cumulative budget certificate.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plot(path: Path, grid: Sequence[Dict[str, Any]]) -> Optional[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        path.with_suffix(".plot_warning.txt").write_text(str(exc), encoding="utf-8")
        return str(exc)
    rel = sorted({float(x["relative_margin_fraction"]) for x in grid})
    abs_values = sorted({float(x["absolute_margin_m3"]) for x in grid})
    matrix = np.full((len(rel), len(abs_values)), np.nan)
    for item in grid:
        matrix[rel.index(float(item["relative_margin_fraction"])), abs_values.index(float(item["absolute_margin_m3"]))] = float(item["oracle_tfv_reduction_pct"]["median"] or np.nan)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    im = ax.imshow(matrix, aspect="auto", origin="lower", cmap="viridis")
    ax.set_xticks(range(len(abs_values)), [str(int(x)) for x in abs_values])
    ax.set_yticks(range(len(rel)), [f"{100*x:.1f}%" for x in rel])
    ax.set_xlabel("Absolute PFV margin B (m³)")
    ax.set_ylabel("Relative PFV margin δ")
    ax.set_title("Oracle median TFV reduction under PFV margin grid (%)")
    fig.colorbar(im, ax=ax, label="Median TFV reduction (%)")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows, metadata = _load_states(args.manifest, args.project_root)
    grid_rows: List[Dict[str, Any]] = []
    state_rows_all: List[pd.DataFrame] = []
    candidate_rows_all: List[pd.DataFrame] = []
    for relative in RELATIVE_GRID:
        for absolute in ABSOLUTE_GRID:
            candidate_rows, state_rows = _state_results(rows, relative, absolute)
            aggregate = _aggregate(state_rows, candidate_rows, relative, absolute)
            grid_rows.append(aggregate)
            candidate_rows = candidate_rows.copy()
            candidate_rows.insert(0, "absolute_margin_m3", absolute)
            candidate_rows.insert(0, "relative_margin_fraction", relative)
            state_rows = state_rows.copy()
            state_rows.insert(0, "absolute_margin_m3", absolute)
            state_rows.insert(0, "relative_margin_fraction", relative)
            candidate_rows_all.append(candidate_rows)
            state_rows_all.append(state_rows)

    grid_rows.sort(key=lambda x: (x["relative_margin_fraction"], x["absolute_margin_m3"]))
    first_any_20 = next((x for x in grid_rows if x["fraction_states_tfv_reduction_ge_20_pct"] > 0.0), None)
    low_moderate_20 = next(
        (
            x for x in grid_rows
            if all(x.get(regime, {}).get("oracle_median_reduction_pct") is not None and x[regime]["oracle_median_reduction_pct"] >= 20.0 for regime in ("LOW_LOAD", "MODERATE_LOAD"))
        ),
        None,
    )
    if first_any_20 is None:
        path_decision = "PATH_C"
    else:
        # A few favorable states do not establish the stated low/moderate
        # loading target.  Require both strata to reach the 20% median at one
        # tested margin; otherwise the next step is candidate expansion.
        path_decision = "PATH_A" if low_moderate_20 is not None else "PATH_B"
    rolling = _rolling_semantics(args.project_root)
    split_authority = _split_authority(args.project_root, args.manifest)
    summary: Dict[str, Any] = {
        "audit_id": "V42_PFV_TFV_AUTHORITATIVE_PARETO_FRONTIER_V1",
        "read_only": True,
        "new_swmm_started": False,
        "manifest": str(args.manifest),
        "input_rows": metadata["input_rows"],
        "input_states": metadata["input_states"],
        "input_rainfall_groups": metadata["input_rainfall_groups"],
        "action_map_authority": "recorded_authoritative_candidate_trajectories",
        "relative_margin_grid": list(RELATIVE_GRID),
        "absolute_margin_grid_m3": list(ABSOLUTE_GRID),
        "grid_summary": grid_rows,
        "twenty_pct_first_any_state_margin": first_any_20,
        "twenty_pct_low_moderate_median_margin": low_moderate_20,
        "twenty_pct_physically_supported": low_moderate_20 is not None,
        "path_decision": path_decision,
        "engineering_acceptability": "not_accepted_automatically; inspect safe PFV excess P90/max and rolling semantics before any contract change",
        "rolling_semantics_path": "PFV_ROLLING_NONINFERIORITY_SEMANTICS.json",
        "split_authority_path": "STEP2_SPLIT_AND_CALIBRATION_AUTHORITY.json",
    }
    candidate_out = pd.concat(candidate_rows_all, ignore_index=True)
    state_out = pd.concat(state_rows_all, ignore_index=True)
    candidate_out.to_csv(args.output_dir / "PFV_TFV_PARETO_FRONTIER_ROWS.csv", index=False)
    state_out.to_csv(args.output_dir / "PFV_TFV_PARETO_FRONTIER_STATES.csv", index=False)
    payload = _clean(summary)
    (args.output_dir / "PFV_TFV_PARETO_FRONTIER.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "PFV_ROLLING_NONINFERIORITY_SEMANTICS.json").write_text(json.dumps(_clean(rolling), indent=2, ensure_ascii=False), encoding="utf-8")
    (args.output_dir / "STEP2_SPLIT_AND_CALIBRATION_AUTHORITY.json").write_text(json.dumps(_clean(split_authority), indent=2, ensure_ascii=False), encoding="utf-8")
    _write_markdown(args.output_dir / "PFV_TFV_PARETO_FRONTIER.md", payload, rolling)
    plot_warning = _write_plot(args.output_dir / "PFV_TFV_PARETO_FRONTIER.png", grid_rows)
    if plot_warning:
        payload["plot_warning"] = plot_warning
    print(json.dumps({key: value for key, value in payload.items() if key != "grid_summary"}, indent=2, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
