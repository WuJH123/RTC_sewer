from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sewerrtc.data.peak_label_semantics import repair_paired_risk_rate_sequences
from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config


def _window(frame: pd.DataFrame, start_min: float, horizon: int) -> pd.DataFrame:
    times = pd.to_numeric(frame["elapsed_min"], errors="coerce").to_numpy(float)
    start = int(np.searchsorted(times, float(start_min), side="left"))
    out = frame.iloc[start : start + horizon].copy()
    if len(out) != horizon:
        raise ValueError(f"incomplete horizon at {start_min}: {len(out)}/{horizon}")
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/wuhan_project6_36_temporal_joint.yaml")
    parser.add_argument("--case-dir", default="outputs/project6_36_temporal_joint_v1/paired_cases")
    parser.add_argument("--out-dir", default="outputs/project6_36_temporal_joint_v1/effect_dataset")
    parser.add_argument("--dataset-name", default="same_state_raw_joint_36.npz")
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = cfg_path(cfg, "project_root")
    case_dir = root / args.case_dir if not Path(args.case_dir).is_absolute() else Path(args.case_dir)
    results_path = case_dir / "paired_candidate_results.csv"
    if not results_path.exists():
        raise FileNotFoundError("Paired SWMM results do not exist; run stage BuildPairedData first")
    results = pd.read_csv(results_path)
    horizon = int((cfg.get("controller", {}) or {}).get("horizon_steps", 6))
    node_table = pd.read_csv(cfg_path(cfg, "outputs.audit") / "node_table.csv")
    node_ids = node_table["node_id"].astype(str).tolist()
    storage_ids = node_table[node_table["node_type"].astype(str).str.lower().eq("storage")]["node_id"].astype(str).tolist()
    action_ids = pd.read_csv(cfg_path(cfg, "outputs.audit") / "actuator_table.csv")
    enabled = set((cfg_path(cfg, "network.control_enabled_actuator_ids_file")).read_text().split())
    action_ids = action_ids[action_ids["actuator_id"].astype(str).isin(enabled)]["actuator_id"].astype(str).tolist()
    priority = [item.strip() for item in (cfg_path(cfg, "outputs.design") / "priority_nodes.txt").read_text().splitlines() if item.strip()]
    arrays: dict[str, list] = {key: [] for key in (
        "event_ids", "pair_ids", "state", "candidate_action_seq", "reference_action_seq", "rain_seq",
        "reference_risk_rate_seq", "delta_risk_rate_seq", "priority_depth_seq", "storage_level_seq", "target_state_seq",
        "split", "candidate_kind", "candidate_family", "phase", "checkpoint_id", "source_dataset",
    )}
    failures = []
    for row in results.itertuples(index=False):
        try:
            candidate = pd.read_csv(row.candidate_detail)
            reference = pd.read_csv(row.reference_detail)
            cw = _window(candidate, float(row.override_start_min), horizon)
            rw = _window(reference, float(row.override_start_min), horizon)
            cstate = cw[[f"h:{node}" for node in node_ids]].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32)
            rstate = rw[[f"h:{node}" for node in node_ids]].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32)
            caction = cw[[f"a:{aid}" for aid in action_ids]].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32)
            raction = rw[[f"a:{aid}" for aid in action_ids]].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32)
            flood_cols = [column for column in rw.columns if column.startswith("flood:")]
            priority_cols = [f"flood:{node}" for node in priority if f"flood:{node}" in rw]
            ref_pfv = rw[priority_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1).to_numpy(np.float32)
            can_pfv = cw[priority_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1).to_numpy(np.float32)
            ref_tfv = rw[flood_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1).to_numpy(np.float32)
            can_tfv = cw[flood_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1).to_numpy(np.float32)
            ref_risk = np.stack([ref_pfv, ref_tfv, ref_tfv], axis=1)[None, ...]
            can_risk = np.stack([can_pfv, can_tfv, can_tfv], axis=1)[None, ...]
            ref_risk, delta_risk = repair_paired_risk_rate_sequences(ref_risk, can_risk - ref_risk)
            ref_risk, delta_risk = ref_risk[0], delta_risk[0]
            residual_action = caction - raction
            if not bool(np.any(np.abs(residual_action) > 1.0e-7)):
                raise ValueError("candidate is a no-op after clipping")
            specification = json.loads(row.executed_action_sequence)
            arrays["event_ids"].append(str(row.event_id)); arrays["pair_ids"].append(str(row.pair_id))
            arrays["state"].append(rstate[0]); arrays["candidate_action_seq"].append(caction); arrays["reference_action_seq"].append(raction)
            arrays["rain_seq"].append(rw[["rainfall_mm_h"]].to_numpy(np.float32))
            arrays["reference_risk_rate_seq"].append(ref_risk); arrays["delta_risk_rate_seq"].append(delta_risk)
            arrays["priority_depth_seq"].append(cw[[f"h:{node}" for node in priority if f"h:{node}" in cw]].mean(axis=1).to_numpy(np.float32))
            storage_cols = [f"h:{node}" for node in storage_ids if f"h:{node}" in cw]
            arrays["storage_level_seq"].append(cw[storage_cols].mean(axis=1).to_numpy(np.float32) if storage_cols else np.zeros(horizon, np.float32))
            arrays["target_state_seq"].append(cstate)
            arrays["split"].append(str(getattr(row, "split", "train")))
            arrays["candidate_kind"].append(str(specification.get("kind", "unknown")))
            arrays["candidate_family"].append(str(specification.get("family", specification.get("kind", "unknown"))))
            arrays["phase"].append(str(row.phase))
            arrays["checkpoint_id"].append(f"{row.event_id}|{row.phase}|{float(row.override_start_min):.3f}")
            arrays["source_dataset"].append(str(case_dir))
        except Exception as exc:
            failures.append({"case_id": str(row.case_id), "error": repr(exc)})
    if failures:
        raise RuntimeError(f"Dataset build rejected {len(failures)} cases; first={failures[0]}")
    out = ensure_dir(root / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir))
    dataset = out / str(args.dataset_name)
    payload = {key: np.asarray(value) for key, value in arrays.items()}
    payload.update({
        "node_ids": np.asarray(node_ids), "action_ids": np.asarray(action_ids),
        "label_semantics": np.asarray("same_state_candidate_minus_no_control"),
        "horizon_steps": np.asarray(horizon),
        "risk_label_channels": np.asarray(["PFV_rate", "TFV_rate", "running_peak_TFV_rate"]),
        "peak_label_definition": np.asarray("max(candidate_TFV_rate)-max(reference_TFV_rate)"),
    })
    np.savez_compressed(dataset, **payload)
    report = {
        "dataset": str(dataset), "samples": len(arrays["event_ids"]),
        "events": len(set(arrays["event_ids"])), "nodes": len(node_ids), "actions": len(action_ids),
        "horizon_steps": horizon, "label_semantics": "same_state_candidate_minus_no_control",
        "peak_label_definition": "difference between candidate and reference running TFV-rate peaks",
        "formal_calibration_leakage": False,
    }
    (out / "dataset_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
