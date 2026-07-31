from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sewerrtc.io.project_paths import load_config


def _priority_nodes(cfg: dict) -> list[str]:
    priority = ((cfg.get("project5_priority", {}) or {}).get("priority_nodes", []))
    if priority:
        return [str(x) for x in priority]
    pfile = Path(cfg.get("project_root", ".")) / "data" / "project5_design" / "priority_pfv_core_nodes.txt"
    if pfile.exists():
        return [line.strip() for line in pfile.read_text(encoding="utf-8").splitlines() if line.strip()]
    return [str(x) for x in ((cfg.get("evaluation", {}) or {}).get("priority_nodes", []))]


def _event_id_from_name(path: Path) -> str:
    return path.name.split("__", 1)[0]


def _elapsed_array(detail: pd.DataFrame) -> np.ndarray:
    return pd.to_numeric(detail["elapsed_min"], errors="coerce").to_numpy(float)


def _window_start(times: np.ndarray, elapsed_min: float) -> int:
    return int(np.argmin(np.abs(times - float(elapsed_min))))


def _window_metrics(
    detail: pd.DataFrame,
    elapsed_min: float,
    horizon_steps: int,
    dt_sec: int,
    priority_nodes: list[str],
) -> dict[str, float]:
    times = _elapsed_array(detail)
    start = _window_start(times, elapsed_min)
    idx = np.arange(start, min(len(detail), start + int(horizon_steps)), dtype=int)
    flood_cols = [c for c in detail.columns if c.startswith("flood:")]
    priority_cols = [f"flood:{node}" for node in priority_nodes if f"flood:{node}" in detail.columns]
    if not len(idx) or not flood_cols:
        return {"PFV": 0.0, "TFV": 0.0, "peak": 0.0}
    total_rate = detail.iloc[idx][flood_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)
    if priority_cols:
        pfv_rate = detail.iloc[idx][priority_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)
    else:
        pfv_rate = pd.Series(np.zeros(len(idx), dtype=float))
    return {
        "PFV": float(pfv_rate.sum() * int(dt_sec)),
        "TFV": float(total_rate.sum() * int(dt_sec)),
        "peak": float(total_rate.max() if len(total_rate) else 0.0),
    }


def _direction(delta: float, tol: float = 1.0e-9) -> str:
    if delta < -tol:
        return "improved"
    if delta > tol:
        return "worsened"
    return "neutral"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--reference-policy", default="no_control")
    parser.add_argument("--horizons-min", default="30,60,120")
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dt_sec = int((cfg.get("experiment", {}) or {}).get("control_step_sec", 300))
    priority = _priority_nodes(cfg)
    horizons = [int(x.strip()) for x in str(args.horizons_min).split(",") if x.strip()]
    horizon_steps = {h: max(1, int(round(h * 60 / dt_sec))) for h in horizons}

    rows: list[dict[str, object]] = []
    proposed_dir = run_dir / "proposed"
    reference_dir = run_dir / "baselines" / str(args.reference_policy)
    for history_path in sorted(proposed_dir.glob("*__controller_history.csv")):
        event_id = _event_id_from_name(history_path)
        prop_detail_path = proposed_dir / f"{event_id}__proposed_detail.csv"
        ref_detail_path = reference_dir / f"{event_id}__{args.reference_policy}_detail.csv"
        if not prop_detail_path.exists() or not ref_detail_path.exists():
            continue
        history = pd.read_csv(history_path)
        prop = pd.read_csv(prop_detail_path)
        ref = pd.read_csv(ref_detail_path)
        accepted = history[
            ~history.get("fallback_to_no_control", pd.Series(False, index=history.index)).fillna(False).astype(bool)
        ].copy()
        for _, item in accepted.iterrows():
            elapsed = float(item.get("elapsed_min", 0.0))
            base = {
                "event_id": event_id,
                "elapsed_min": elapsed,
                "phase": item.get("phase", ""),
                "selected_label": item.get("selected_label", item.get("selected_sequence_label", "")),
                "selected_tier": item.get("selected_tier", ""),
            }
            metrics: dict[str, float | str] = {}
            directions: dict[str, str] = {}
            for horizon_min, steps in horizon_steps.items():
                pm = _window_metrics(prop, elapsed, steps, dt_sec, priority)
                rm = _window_metrics(ref, elapsed, steps, dt_sec, priority)
                for name, pval in pm.items():
                    delta = float(pval - rm[name])
                    metrics[f"H{horizon_min}_{name}_delta"] = delta
                    directions[f"H{horizon_min}_{name}_direction"] = _direction(delta)
            pfv_dirs = [directions.get(f"H{h}_PFV_direction", "neutral") for h in horizons]
            tfv_dirs = [directions.get(f"H{h}_TFV_direction", "neutral") for h in horizons]
            peak_dirs = [directions.get(f"H{h}_peak_direction", "neutral") for h in horizons]
            rows.append(
                {
                    **base,
                    **metrics,
                    **directions,
                    "PFV_direction_reversal": len(set(pfv_dirs)) > 1,
                    "TFV_direction_reversal": len(set(tfv_dirs)) > 1,
                    "peak_direction_reversal": len(set(peak_dirs)) > 1,
                }
            )

    audit = pd.DataFrame(rows)
    audit_path = out_dir / "horizon_reversal_audit.csv"
    audit.to_csv(audit_path, index=False)
    summary = {
        "run_dir": str(run_dir),
        "reference_policy": str(args.reference_policy),
        "rows": int(len(audit)),
        "horizons_min": horizons,
        "PFV_direction_reversal_frac": float(audit["PFV_direction_reversal"].mean()) if len(audit) else None,
        "TFV_direction_reversal_frac": float(audit["TFV_direction_reversal"].mean()) if len(audit) else None,
        "peak_direction_reversal_frac": float(audit["peak_direction_reversal"].mean()) if len(audit) else None,
        "audit_csv": str(audit_path),
    }
    (out_dir / "horizon_reversal_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
