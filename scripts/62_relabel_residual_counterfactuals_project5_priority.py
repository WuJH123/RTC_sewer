from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sewerrtc.io.priority_config import configured_priority_nodes, configured_priority_sentinel_nodes, read_node_list
from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config


def _safe_float(value: object, default: float = 0.0) -> float:
    try:
        v = float(value)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _priority_metrics_from_detail(path: Path, priority_nodes: list[str], dt_sec: int) -> dict[str, object]:
    if not path.exists():
        return {
            "priority_PFV": np.nan,
            "priority_duration_min": np.nan,
            "priority_peak_rate": np.nan,
            "priority_nodes_present": 0,
            "priority_nodes_missing": len(priority_nodes),
            "missing_priority_nodes": ",".join(priority_nodes),
        }

    try:
        header = pd.read_csv(path, nrows=0)
    except Exception:
        return {
            "priority_PFV": np.nan,
            "priority_duration_min": np.nan,
            "priority_peak_rate": np.nan,
            "priority_nodes_present": 0,
            "priority_nodes_missing": len(priority_nodes),
            "missing_priority_nodes": ",".join(priority_nodes),
        }

    pr_cols = [f"flood:{n}" for n in priority_nodes if f"flood:{n}" in header.columns]
    missing = [n for n in priority_nodes if f"flood:{n}" not in header.columns]
    if not pr_cols:
        return {
            "priority_PFV": 0.0,
            "priority_duration_min": 0.0,
            "priority_peak_rate": 0.0,
            "priority_nodes_present": 0,
            "priority_nodes_missing": len(missing),
            "missing_priority_nodes": ",".join(missing),
        }

    detail = pd.read_csv(path, usecols=pr_cols)
    rate = detail.apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(float).sum(axis=1)
    return {
        "priority_PFV": float(rate.sum() * int(dt_sec)),
        "priority_duration_min": float((rate > 1e-9).sum() * int(dt_sec) / 60.0),
        "priority_peak_rate": float(rate.max()) if len(rate) else 0.0,
        "priority_nodes_present": len(pr_cols),
        "priority_nodes_missing": len(missing),
        "missing_priority_nodes": ",".join(missing),
    }


def _event_baseline_metrics(
    event_id: str,
    residual_root: Path,
    priority_nodes: list[str],
    dt_sec: int,
    cache: dict[str, dict[str, object]],
) -> dict[str, object]:
    if event_id in cache:
        return cache[event_id]
    path = residual_root / "internal_rules" / f"{event_id}__internal_rules_detail.csv"
    metrics = _priority_metrics_from_detail(path, priority_nodes, dt_sec)
    metrics["baseline_detail_file_project5_priority"] = str(path)
    cache[event_id] = metrics
    return metrics


def _candidate_metrics_job(args: tuple[str, list[str], int]) -> dict[str, object]:
    detail_file, priority_nodes, dt_sec = args
    return _priority_metrics_from_detail(Path(detail_file), priority_nodes, int(dt_sec))


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Relabel Project4 residual counterfactuals for the Project5 PFV-core priority zone."
    )
    ap.add_argument("--config", default="configs/wuhan.yaml")
    ap.add_argument("--source-residual-dir", default="")
    ap.add_argument("--priority-file", default="")
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--dt-sec", type=int, default=300)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-rows", type=int, default=0, help="Optional smoke-test row limit; 0 means all rows.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    project_root = cfg_path(cfg, "project_root")
    source_residual_dir = (
        Path(args.source_residual_dir)
        if args.source_residual_dir
        else Path(cfg.get("reuse_sources", {}).get("project4_residual_counterfactual_dir", ""))
    )
    if not source_residual_dir:
        raise ValueError("Missing source residual directory. Set reuse_sources.project4_residual_counterfactual_dir or --source-residual-dir.")
    rows_path = source_residual_dir / "residual_counterfactual_results.csv"
    if not rows_path.exists():
        raise FileNotFoundError(f"Missing residual counterfactual results: {rows_path}")

    priority_file = Path(args.priority_file) if args.priority_file else cfg_path(cfg, "network.priority_nodes_file")
    priority_nodes = read_node_list(priority_file) if args.priority_file else configured_priority_nodes(cfg)
    sentinel_nodes = configured_priority_sentinel_nodes(cfg)
    out_dir = ensure_dir(
        Path(args.out_dir)
        if args.out_dir
        else cfg_path(cfg, "outputs.closed_loop") / "internal_residual_counterfactuals"
    )

    df = pd.read_csv(rows_path)
    if int(args.max_rows) > 0:
        df = df.head(int(args.max_rows)).copy()
    for col in ["baseline_PFV", "PFV", "delta_PFV", "y_pfv_improve", "y_safe", "y_safe_guarded"]:
        if col in df and f"project4_original_{col}" not in df:
            df[f"project4_original_{col}"] = df[col]

    baseline_cache: dict[str, dict[str, object]] = {}
    detail_files = df["detail_file"].fillna("").astype(str).tolist()
    jobs = [(p, priority_nodes, int(args.dt_sec)) for p in detail_files]
    if int(args.workers) <= 1:
        candidate_metrics = [_candidate_metrics_job(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=max(1, int(args.workers))) as ex:
            candidate_metrics = list(ex.map(_candidate_metrics_job, jobs, chunksize=16))

    relabeled_rows: list[dict[str, object]] = []
    missing_detail = sum(1 for p in detail_files if not Path(p).exists())
    for (_, row), candidate in zip(df.iterrows(), candidate_metrics):
        event_id = str(row.get("event_id", ""))
        baseline = _event_baseline_metrics(event_id, source_residual_dir, priority_nodes, int(args.dt_sec), baseline_cache)

        baseline_pfv = _safe_float(baseline.get("priority_PFV"), np.nan)
        candidate_pfv = _safe_float(candidate.get("priority_PFV"), np.nan)
        delta_pfv = candidate_pfv - baseline_pfv if np.isfinite(candidate_pfv) and np.isfinite(baseline_pfv) else np.nan

        baseline_tfv = _safe_float(row.get("baseline_TFV"), np.nan)
        baseline_peak = _safe_float(row.get("baseline_peak_TFV_rate"), np.nan)
        delta_tfv = _safe_float(row.get("TFV"), np.nan) - baseline_tfv
        delta_peak = _safe_float(row.get("peak_TFV_rate"), np.nan) - baseline_peak
        tfv_guard = float(cfg.get("experiment", {}).get("tfv_guard_pct", 0.005)) * baseline_tfv if baseline_tfv > 1.1 else 0.0
        peak_guard = float(cfg.get("experiment", {}).get("peak_guard_pct", 0.010)) * baseline_peak if baseline_peak > 1.1 else 0.0
        safe = bool(delta_tfv <= tfv_guard and delta_peak <= peak_guard)

        relabeled_rows.append(
            {
                "baseline_PFV": baseline_pfv,
                "PFV": candidate_pfv,
                "delta_PFV": delta_pfv,
                "y_pfv_improve": int(delta_pfv < 0.0) if np.isfinite(delta_pfv) else 0,
                "y_safe": int(safe),
                "tfv_guard": tfv_guard,
                "peak_guard": peak_guard,
                "y_safe_guarded": int(safe),
                "project5_priority_duration_min": candidate.get("priority_duration_min", np.nan),
                "project5_baseline_priority_duration_min": baseline.get("priority_duration_min", np.nan),
                "project5_priority_peak_rate": candidate.get("priority_peak_rate", np.nan),
                "project5_baseline_priority_peak_rate": baseline.get("priority_peak_rate", np.nan),
                "project5_priority_nodes_present": candidate.get("priority_nodes_present", 0),
                "project5_priority_nodes_missing": candidate.get("priority_nodes_missing", len(priority_nodes)),
                "project5_missing_priority_nodes": candidate.get("missing_priority_nodes", ""),
                "baseline_detail_file_project5_priority": baseline.get("baseline_detail_file_project5_priority", ""),
            }
        )

    relabeled = pd.DataFrame(relabeled_rows, index=df.index)
    for col in relabeled.columns:
        df[col] = relabeled[col]

    output_csv = out_dir / "residual_counterfactual_results.csv"
    df.to_csv(output_csv, index=False, encoding="utf-8-sig")
    summary = {
        "source_rows": str(rows_path),
        "output_rows": str(output_csv),
        "priority_source": str(priority_file),
        "priority_role": "pfv_core",
        "priority_nodes": priority_nodes,
        "sentinel_role": "depth_surcharge_monitoring_only",
        "sentinel_nodes": sentinel_nodes,
        "rows": int(len(df)),
        "events": int(df["event_id"].nunique()) if "event_id" in df else 0,
        "missing_candidate_detail_files": int(missing_detail),
        "baseline_events_recomputed": int(len(baseline_cache)),
        "pfv_improve_n": int(pd.to_numeric(df["y_pfv_improve"], errors="coerce").fillna(0).sum()),
        "safe_n": int(pd.to_numeric(df["y_safe"], errors="coerce").fillna(0).sum()),
        "mean_delta_PFV": float(pd.to_numeric(df["delta_PFV"], errors="coerce").mean(skipna=True)),
        "median_delta_PFV": float(pd.to_numeric(df["delta_PFV"], errors="coerce").median(skipna=True)),
        "note": "SWMM detail files are reused from Project4; labels are recomputed for Project5 priority nodes.",
    }
    (out_dir / "project5_residual_relabel_report.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
