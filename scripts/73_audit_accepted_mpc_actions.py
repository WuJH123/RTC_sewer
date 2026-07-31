from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config


AUDIT_COLUMNS = [
    "event_id",
    "rain_id",
    "pattern",
    "duration_min",
    "elapsed_min",
    "phase",
    "selected_sequence_label",
    "selected_tier",
    "target_actuators",
    "reference_PFV_H",
    "proposed_PFV_H",
    "reference_TFV_H",
    "proposed_TFV_H",
    "reference_peak_TFV_rate_H",
    "proposed_peak_TFV_rate_H",
    "predicted_PFV_gain",
    "true_PFV_gain",
    "predicted_TFV_delta",
    "true_TFV_delta",
    "predicted_peak_delta",
    "true_peak_delta",
    "PFV_direction_correct",
    "TFV_safety_predicted",
    "TFV_safety_true",
    "peak_safety_predicted",
    "peak_safety_true",
]


def _event_parts(event_id: str) -> tuple[str, int, str]:
    parts = str(event_id).split("_", 2)
    rain_id = parts[0] if parts else ""
    duration = 0
    pattern = parts[2] if len(parts) >= 3 else ""
    if len(parts) >= 2 and parts[1].startswith("D"):
        try:
            duration = int(parts[1][1:])
        except ValueError:
            duration = 0
    return rain_id, duration, pattern


def _horizon_metrics(detail: pd.DataFrame, elapsed_min: float, horizon_steps: int, dt_sec: int, priority_nodes: list[str]) -> dict:
    if detail.empty or "elapsed_min" not in detail:
        return {"PFV_H": 0.0, "TFV_H": 0.0, "peak_TFV_rate_H": 0.0}
    frame = detail.copy()
    frame["elapsed_min"] = pd.to_numeric(frame["elapsed_min"], errors="coerce")
    frame = frame.dropna(subset=["elapsed_min"]).sort_values("elapsed_min").reset_index(drop=True)
    flood_cols = [c for c in frame.columns if c.startswith("flood:")]
    pr_cols = [f"flood:{n}" for n in priority_nodes if f"flood:{n}" in frame.columns]
    if not flood_cols:
        return {"PFV_H": 0.0, "TFV_H": 0.0, "peak_TFV_rate_H": 0.0}
    times = frame["elapsed_min"].to_numpy(float)
    start_idx = int(np.searchsorted(times, float(elapsed_min), side="left")) + 1
    idx = np.arange(start_idx, start_idx + int(horizon_steps), dtype=int)
    idx = idx[idx < len(frame)]
    if idx.size == 0:
        return {"PFV_H": 0.0, "TFV_H": 0.0, "peak_TFV_rate_H": 0.0}
    total_rate = frame.iloc[idx][flood_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)
    if pr_cols:
        priority_rate = frame.iloc[idx][pr_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1)
    else:
        priority_rate = pd.Series(0.0, index=idx)
    return {
        "PFV_H": float(priority_rate.sum() * int(dt_sec)),
        "TFV_H": float(total_rate.sum() * int(dt_sec)),
        "peak_TFV_rate_H": float(total_rate.max()) if len(total_rate) else 0.0,
    }


def _prepare_horizon_arrays(detail: pd.DataFrame, priority_nodes: list[str]) -> dict[str, np.ndarray]:
    if detail.empty or "elapsed_min" not in detail:
        return {
            "times": np.zeros(0, dtype=float),
            "total_rate": np.zeros(0, dtype=float),
            "priority_rate": np.zeros(0, dtype=float),
        }
    frame = detail.copy()
    frame["elapsed_min"] = pd.to_numeric(frame["elapsed_min"], errors="coerce")
    frame = frame.dropna(subset=["elapsed_min"]).sort_values("elapsed_min").reset_index(drop=True)
    flood_cols = [c for c in frame.columns if c.startswith("flood:")]
    pr_cols = [f"flood:{n}" for n in priority_nodes if f"flood:{n}" in frame.columns]
    if flood_cols:
        total_rate = frame[flood_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1).to_numpy(float)
    else:
        total_rate = np.zeros(len(frame), dtype=float)
    if pr_cols:
        priority_rate = frame[pr_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1).to_numpy(float)
    else:
        priority_rate = np.zeros(len(frame), dtype=float)
    return {
        "times": frame["elapsed_min"].to_numpy(float),
        "total_rate": total_rate,
        "priority_rate": priority_rate,
    }


def _horizon_metrics_from_arrays(arrays: dict[str, np.ndarray], elapsed_min: float, horizon_steps: int, dt_sec: int) -> dict:
    times = arrays["times"]
    if times.size == 0:
        return {"PFV_H": 0.0, "TFV_H": 0.0, "peak_TFV_rate_H": 0.0}
    start_idx = int(np.searchsorted(times, float(elapsed_min), side="left")) + 1
    idx = np.arange(start_idx, start_idx + int(horizon_steps), dtype=int)
    idx = idx[idx < times.size]
    if idx.size == 0:
        return {"PFV_H": 0.0, "TFV_H": 0.0, "peak_TFV_rate_H": 0.0}
    total_rate = arrays["total_rate"][idx]
    priority_rate = arrays["priority_rate"][idx]
    return {
        "PFV_H": float(priority_rate.sum() * int(dt_sec)),
        "TFV_H": float(total_rate.sum() * int(dt_sec)),
        "peak_TFV_rate_H": float(total_rate.max()) if total_rate.size else 0.0,
    }


def _read_detail(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    return pd.read_csv(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/open_pystorms_beta.yaml")
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--reference-policy", default="no_control")
    ap.add_argument("--out-dir", default="")
    args = ap.parse_args()

    cfg = load_config(args.config)
    run_dir = Path(args.run_dir)
    out_dir = ensure_dir(Path(args.out_dir) if args.out_dir else run_dir / "accepted_action_audit")
    priority = (cfg_path(cfg, "outputs.design") / "priority_nodes.txt").read_text(encoding="utf-8").splitlines()
    h = int((cfg.get("controller", {}) or {}).get("horizon_steps", (cfg.get("horizon_surrogate", {}) or {}).get("horizon_steps", 6)))
    dt_sec = int(cfg["experiment"]["control_step_sec"])
    rows: list[dict] = []
    for hist_path in sorted((run_dir / "proposed").glob("*__controller_history.csv")):
        event_id = hist_path.name.replace("__controller_history.csv", "")
        hist = pd.read_csv(hist_path)
        if hist.empty:
            continue
        proposed_detail = _read_detail(run_dir / "proposed" / f"{event_id}__proposed_detail.csv")
        reference_detail = _read_detail(
            run_dir / "baselines" / args.reference_policy / f"{event_id}__{args.reference_policy}_detail.csv"
        )
        if proposed_detail.empty or reference_detail.empty:
            continue
        proposed_arrays = _prepare_horizon_arrays(proposed_detail, priority)
        reference_arrays = _prepare_horizon_arrays(reference_detail, priority)
        label_col = "selected_sequence_label" if "selected_sequence_label" in hist.columns else "selected_label"
        gate_col = "selected_gate_pass" if "selected_gate_pass" in hist.columns else ""
        gate = hist[gate_col].astype(str).str.lower().isin(["true", "1", "yes"]) if gate_col else pd.Series(True, index=hist.index)
        labels = hist[label_col].astype(str) if label_col in hist.columns else pd.Series("", index=hist.index)
        accepted = hist[
            gate
            & ~labels.isin(["hold_native", "reference_no_control", "deployment_reliability_no_control"])
        ].copy()
        rain_id, duration_min, pattern = _event_parts(event_id)
        for _, row in accepted.iterrows():
            elapsed = float(row.get("elapsed_min", 0.0))
            prop = _horizon_metrics_from_arrays(proposed_arrays, elapsed, h, dt_sec)
            ref = _horizon_metrics_from_arrays(reference_arrays, elapsed, h, dt_sec)
            pred_pfv_gain = float(row.get("selected_reference_pfv_horizon", np.nan)) - float(row.get("selected_pfv_horizon", np.nan))
            pred_tfv_delta = float(row.get("selected_tfv_horizon", np.nan)) - float(row.get("selected_reference_tfv_horizon", np.nan))
            pred_peak_delta = float(row.get("selected_peak_tfv_rate", np.nan)) - float(row.get("selected_reference_peak_tfv_rate", np.nan))
            true_pfv_gain = float(ref["PFV_H"]) - float(prop["PFV_H"])
            true_tfv_delta = float(prop["TFV_H"]) - float(ref["TFV_H"])
            true_peak_delta = float(prop["peak_TFV_rate_H"]) - float(ref["peak_TFV_rate_H"])
            rows.append(
                {
                    "event_id": event_id,
                    "rain_id": rain_id,
                    "pattern": pattern,
                    "duration_min": duration_min,
                    "elapsed_min": elapsed,
                    "phase": row.get("phase", ""),
                    "selected_sequence_label": row.get(label_col, ""),
                    "selected_tier": row.get("selected_tier", ""),
                    "target_actuators": row.get("target_actuators", ""),
                    "reference_PFV_H": float(ref["PFV_H"]),
                    "proposed_PFV_H": float(prop["PFV_H"]),
                    "reference_TFV_H": float(ref["TFV_H"]),
                    "proposed_TFV_H": float(prop["TFV_H"]),
                    "reference_peak_TFV_rate_H": float(ref["peak_TFV_rate_H"]),
                    "proposed_peak_TFV_rate_H": float(prop["peak_TFV_rate_H"]),
                    "predicted_PFV_gain": pred_pfv_gain,
                    "true_PFV_gain": true_pfv_gain,
                    "predicted_TFV_delta": pred_tfv_delta,
                    "true_TFV_delta": true_tfv_delta,
                    "predicted_peak_delta": pred_peak_delta,
                    "true_peak_delta": true_peak_delta,
                    "PFV_direction_correct": bool(np.sign(pred_pfv_gain) == np.sign(true_pfv_gain)) if np.isfinite(pred_pfv_gain) else False,
                    "TFV_safety_predicted": bool(pred_tfv_delta <= float(row.get("selected_tfv_tolerance", 0.0))),
                    "TFV_safety_true": bool(true_tfv_delta <= float(row.get("selected_tfv_tolerance", 0.0))),
                    "peak_safety_predicted": bool(pred_peak_delta <= float(row.get("selected_peak_tolerance", 0.0))),
                    "peak_safety_true": bool(true_peak_delta <= float(row.get("selected_peak_tolerance", 0.0))),
                }
            )
    audit = pd.DataFrame(rows, columns=AUDIT_COLUMNS)
    audit.to_csv(out_dir / "accepted_action_horizon_audit.csv", index=False, encoding="utf-8-sig")
    report = {
        "run_dir": str(run_dir),
        "reference_policy": args.reference_policy,
        "horizon_steps": h,
        "accepted_actions": int(len(audit)),
        "PFV_direction_accuracy": float(audit["PFV_direction_correct"].mean()) if not audit.empty else None,
        "TFV_true_safe_frac": float(audit["TFV_safety_true"].mean()) if not audit.empty else None,
        "peak_true_safe_frac": float(audit["peak_safety_true"].mean()) if not audit.empty else None,
        "audit_csv": str(out_dir / "accepted_action_horizon_audit.csv"),
    }
    (out_dir / "accepted_action_horizon_audit_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
