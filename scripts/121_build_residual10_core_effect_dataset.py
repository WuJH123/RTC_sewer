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
    out = frame.iloc[start : start + int(horizon)].copy()
    if len(out) != int(horizon):
        raise ValueError(f"incomplete horizon at {start_min}: {len(out)}/{horizon}")
    return out


def _window_until(frame: pd.DataFrame, start_min: float, minutes: float) -> pd.DataFrame:
    elapsed = pd.to_numeric(frame["elapsed_min"], errors="coerce")
    return frame[(elapsed >= float(start_min)) & (elapsed < float(start_min) + float(minutes))].copy()


def _flood_rate(frame: pd.DataFrame, cols: list[str]) -> np.ndarray:
    if not cols:
        return np.zeros(len(frame), dtype=np.float32)
    return frame[cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).sum(axis=1).to_numpy(np.float32)


def _volume_delta(candidate: pd.DataFrame, reference: pd.DataFrame, cols: list[str], *, dt_sec: int) -> float:
    n = min(len(candidate), len(reference))
    if n <= 0:
        return float("nan")
    can = _flood_rate(candidate.iloc[:n], cols)
    ref = _flood_rate(reference.iloc[:n], cols)
    return float((can - ref).sum() * float(dt_sec))


def _peak_delta(candidate: pd.DataFrame, reference: pd.DataFrame, cols: list[str]) -> float:
    n = min(len(candidate), len(reference))
    if n <= 0:
        return float("nan")
    can = _flood_rate(candidate.iloc[:n], cols)
    ref = _flood_rate(reference.iloc[:n], cols)
    return float(can.max(initial=0.0) - ref.max(initial=0.0))


def _noninferiority_margin(reference_volume: pd.Series, *, absolute_margin: float, relative_margin: float) -> pd.Series:
    reference = pd.to_numeric(reference_volume, errors="coerce").fillna(0.0).clip(lower=0.0)
    return pd.Series(
        np.maximum(float(absolute_margin), reference.to_numpy(float) * float(relative_margin)),
        index=reference.index,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Residual10 mixed-reference effect dataset.")
    parser.add_argument("--config", default="configs/wuhan_project6_36_hierarchical_eventbudget_h120_v2.yaml")
    parser.add_argument("--case-dir", default="outputs/project6_36_residual10_core_paired_h120_v1/paired_cases")
    parser.add_argument("--out-dir", default="outputs/project6_36_residual10_core_paired_h120_v1/effect_dataset")
    parser.add_argument("--dataset-name", default="same_state_residual10_core_h120_v1.npz")
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = cfg_path(cfg, "project_root")
    case_dir = root / args.case_dir if not Path(args.case_dir).is_absolute() else Path(args.case_dir)
    results_path = case_dir / "paired_candidate_results.csv"
    if not results_path.exists():
        raise FileNotFoundError(results_path)
    results = pd.read_csv(results_path)
    manifest_path = root / "outputs/project6_36_residual10_core_paired_h120_v1/paired_plan/residual10_core_paired_manifest.csv"
    manifest = pd.read_csv(manifest_path) if manifest_path.exists() else pd.DataFrame()
    horizon = int((cfg.get("controller", {}) or {}).get("horizon_steps", 6))
    dt_sec = int(cfg["experiment"]["control_step_sec"])
    node_table = pd.read_csv(cfg_path(cfg, "outputs.audit") / "node_table.csv")
    node_ids = node_table["node_id"].astype(str).tolist()
    storage_ids = node_table[node_table["node_type"].astype(str).str.lower().eq("storage")]["node_id"].astype(str).tolist()
    actuator_table = pd.read_csv(cfg_path(cfg, "outputs.audit") / "actuator_table.csv")
    enabled = set(cfg_path(cfg, "network.control_enabled_actuator_ids_file").read_text(encoding="utf-8").split())
    action_ids = actuator_table[actuator_table["actuator_id"].astype(str).isin(enabled)]["actuator_id"].astype(str).tolist()
    priority = [line.strip() for line in (cfg_path(cfg, "outputs.design") / "priority_nodes.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    arrays: dict[str, list] = {key: [] for key in (
        "event_ids", "pair_ids", "state", "candidate_action_seq", "reference_action_seq", "rain_seq",
        "reference_risk_rate_seq", "delta_risk_rate_seq", "priority_depth_seq", "storage_level_seq", "target_state_seq",
        "split", "candidate_kind", "candidate_family", "phase", "checkpoint_id", "source_dataset",
        "reference_pfv_h30", "reference_pfv_h60", "reference_pfv_h90", "reference_pfv_h120", "reference_pfv_event_end",
        "delta_pfv_h30", "delta_pfv_h60", "delta_pfv_h90", "delta_pfv_h120", "delta_pfv_event_end",
        "delta_tfv_h30", "delta_tfv_h60", "delta_tfv_h90", "delta_tfv_h120", "delta_tfv_event_end",
        "delta_peak_h30", "delta_peak_h60", "delta_peak_h90", "delta_peak_h120", "delta_peak_event_end",
        "direction_reversal_tfv_h30_h120", "direction_reversal_peak_h30_h120",
        "terminal_storage_volume", "terminal_available_capacity", "terminal_downstream_flow",
    )}
    failures: list[dict[str, str]] = []
    grouped = {str(row.pair_id): row for row in results.itertuples(index=False)}
    pair_ids = sorted(set(results["pair_id"].astype(str)))
    for pair_id in pair_ids:
        branches = results[results["pair_id"].astype(str).eq(pair_id)].copy()
        if set(branches["branch"].astype(str)) != {"A", "B"}:
            failures.append({"pair_id": pair_id, "error": "missing A/B branch"})
            continue
        try:
            row_a = branches[branches["branch"].astype(str).eq("A")].iloc[0]
            row_b = branches[branches["branch"].astype(str).eq("B")].iloc[0]
            no_control = pd.read_csv(row_b["reference_detail"])
            core = pd.read_csv(row_a["candidate_detail"])
            candidate = pd.read_csv(row_b["candidate_detail"])
            start_min = float(row_b["override_start_min"])
            nw = _window(no_control, start_min, horizon)
            aw = _window(core, start_min, horizon)
            bw = _window(candidate, start_min, horizon)
            node_cols = [f"h:{node}" for node in node_ids]
            action_cols = [f"a:{aid}" for aid in action_ids]
            missing = [col for col in node_cols + action_cols if col not in bw]
            if missing:
                raise KeyError(f"detail missing required columns: {missing[:3]}")
            priority_cols = [f"flood:{node}" for node in priority if f"flood:{node}" in bw]
            flood_cols = [col for col in bw.columns if col.startswith("flood:")]
            no_pfv = _flood_rate(nw, priority_cols)
            can_pfv = _flood_rate(bw, priority_cols)
            core_tfv = _flood_rate(aw, flood_cols)
            can_tfv = _flood_rate(bw, flood_cols)
            ref_risk = np.stack([no_pfv, core_tfv, core_tfv], axis=1)[None, ...]
            can_risk = np.stack([can_pfv, can_tfv, can_tfv], axis=1)[None, ...]
            ref_risk, delta_risk = repair_paired_risk_rate_sequences(ref_risk, can_risk - ref_risk)
            ref_risk, delta_risk = ref_risk[0], delta_risk[0]
            reference_action = aw[action_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32)
            candidate_action = bw[action_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32)
            if not np.any(np.abs(candidate_action - reference_action) > 1.0e-7):
                raise ValueError("residual branch is no-op against core26 branch")
            metadata = manifest[manifest["pair_id"].astype(str).eq(pair_id)]
            meta_b = metadata[metadata["branch"].astype(str).eq("B")].iloc[0] if len(metadata) else None
            for minutes in (30, 60, 90, 120):
                no_h = _window_until(no_control, start_min, minutes)
                core_h = _window_until(core, start_min, minutes)
                can_h = _window_until(candidate, start_min, minutes)
                arrays[f"reference_pfv_h{minutes}"].append(float(_flood_rate(no_h, priority_cols).sum() * float(dt_sec)))
                arrays[f"delta_pfv_h{minutes}"].append(_volume_delta(can_h, no_h, priority_cols, dt_sec=dt_sec))
                arrays[f"delta_tfv_h{minutes}"].append(_volume_delta(can_h, core_h, flood_cols, dt_sec=dt_sec))
                arrays[f"delta_peak_h{minutes}"].append(_peak_delta(can_h, core_h, flood_cols))
            no_tail = no_control[pd.to_numeric(no_control["elapsed_min"], errors="coerce") >= start_min]
            core_tail = core[pd.to_numeric(core["elapsed_min"], errors="coerce") >= start_min]
            can_tail = candidate[pd.to_numeric(candidate["elapsed_min"], errors="coerce") >= start_min]
            arrays["reference_pfv_event_end"].append(float(_flood_rate(no_tail, priority_cols).sum() * float(dt_sec)))
            arrays["delta_pfv_event_end"].append(_volume_delta(can_tail, no_tail, priority_cols, dt_sec=dt_sec))
            arrays["delta_tfv_event_end"].append(_volume_delta(can_tail, core_tail, flood_cols, dt_sec=dt_sec))
            arrays["delta_peak_event_end"].append(_peak_delta(can_tail, core_tail, flood_cols))
            arrays["direction_reversal_tfv_h30_h120"].append(bool(np.sign(arrays["delta_tfv_h30"][-1]) != np.sign(arrays["delta_tfv_h120"][-1]) and abs(arrays["delta_tfv_h30"][-1]) > 100.0 and abs(arrays["delta_tfv_h120"][-1]) > 100.0))
            arrays["direction_reversal_peak_h30_h120"].append(bool(np.sign(arrays["delta_peak_h30"][-1]) != np.sign(arrays["delta_peak_h120"][-1]) and abs(arrays["delta_peak_h30"][-1]) > 0.1 and abs(arrays["delta_peak_h120"][-1]) > 0.1))
            storage_cols = [f"h:{node}" for node in storage_ids if f"h:{node}" in can_tail]
            arrays["terminal_storage_volume"].append(float(can_tail[storage_cols].tail(1).mean(axis=1).iloc[0]) if storage_cols and len(can_tail) else np.nan)
            arrays["terminal_available_capacity"].append(np.nan)
            arrays["terminal_downstream_flow"].append(np.nan)
            arrays["event_ids"].append(str(row_b["event_id"]))
            arrays["pair_ids"].append(pair_id)
            arrays["state"].append(nw[node_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32)[0])
            arrays["candidate_action_seq"].append(candidate_action)
            arrays["reference_action_seq"].append(reference_action)
            arrays["rain_seq"].append(nw[["rainfall_mm_h"]].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32))
            arrays["reference_risk_rate_seq"].append(ref_risk)
            arrays["delta_risk_rate_seq"].append(delta_risk)
            arrays["priority_depth_seq"].append(bw[[f"h:{node}" for node in priority if f"h:{node}" in bw]].mean(axis=1).to_numpy(np.float32))
            arrays["storage_level_seq"].append(bw[storage_cols[: max(1, len(storage_cols))]].mean(axis=1).to_numpy(np.float32) if storage_cols else np.zeros(horizon, np.float32))
            arrays["target_state_seq"].append(bw[node_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32))
            arrays["split"].append(str(row_b.get("split", "train")))
            arrays["candidate_kind"].append("core26_plus_residual10")
            arrays["candidate_family"].append("residual10_core_conditioned")
            arrays["phase"].append(str(row_b.get("phase", "unknown")))
            arrays["checkpoint_id"].append(str(meta_b["checkpoint_id"]) if meta_b is not None and "checkpoint_id" in meta_b else f"{row_b['event_id']}|{row_b.get('phase', 'unknown')}|{start_min:.3f}")
            arrays["source_dataset"].append(str(case_dir))
        except Exception as exc:
            failures.append({"pair_id": pair_id, "error": repr(exc)})
    if failures:
        fail_path = ensure_dir(root / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir)) / "dataset_failures.csv"
        pd.DataFrame(failures).to_csv(fail_path, index=False)
        raise RuntimeError(f"Dataset build rejected {len(failures)} pairs; first={failures[0]}")
    out = ensure_dir(root / args.out_dir if not Path(args.out_dir).is_absolute() else Path(args.out_dir))
    dataset = out / args.dataset_name
    payload = {key: np.asarray(value) for key, value in arrays.items()}
    payload.update({
        "node_ids": np.asarray(node_ids),
        "action_ids": np.asarray(action_ids),
        "label_semantics": np.asarray("mixed_reference_residual10_core_conditioned"),
        "pfv_label_reference": np.asarray("candidate_minus_no_control"),
        "tfv_peak_label_reference": np.asarray("candidate_minus_core26"),
        "horizon_steps": np.asarray(horizon),
        "risk_label_channels": np.asarray(["PFV_rate_vs_no_control", "TFV_rate_vs_core26", "running_peak_TFV_rate_vs_core26"]),
        "peak_label_definition": np.asarray("candidate running TFV-rate peak minus core26 running TFV-rate peak"),
    })
    np.savez_compressed(dataset, **payload)
    audit = pd.DataFrame({
        "event_id": arrays["event_ids"],
        "pair_id": arrays["pair_ids"],
        "split": arrays["split"],
        "phase": arrays["phase"],
        "reference_pfv_h120": arrays["reference_pfv_h120"],
        "delta_pfv_h120": arrays["delta_pfv_h120"],
        "delta_tfv_h120": arrays["delta_tfv_h120"],
        "delta_peak_h120": arrays["delta_peak_h120"],
        "reference_pfv_event_end": arrays["reference_pfv_event_end"],
        "delta_pfv_event_end": arrays["delta_pfv_event_end"],
        "delta_tfv_event_end": arrays["delta_tfv_event_end"],
        "delta_peak_event_end": arrays["delta_peak_event_end"],
        "direction_reversal_tfv_h30_h120": arrays["direction_reversal_tfv_h30_h120"],
        "direction_reversal_peak_h30_h120": arrays["direction_reversal_peak_h30_h120"],
    })
    temporal = (((cfg.get("controller", {}) or {}).get("temporal_joint", {}) or {}))
    safety = temporal.get("safety", {}) or {}
    pfv_abs_margin = float(safety.get("pfv_abs_margin_m3", 100.0))
    pfv_rel_margin = float(safety.get("pfv_rel_margin", 0.02))
    audit["pfv_margin_h120"] = _noninferiority_margin(
        audit["reference_pfv_h120"],
        absolute_margin=pfv_abs_margin,
        relative_margin=pfv_rel_margin,
    )
    pfv_safe = audit["delta_pfv_h120"] <= audit["pfv_margin_h120"]
    audit["pfv_safe_h120"] = pfv_safe
    audit.to_csv(out / "residual10_core_effect_audit.csv", index=False)
    report = {
        "dataset": str(dataset),
        "samples": int(len(arrays["event_ids"])),
        "events": int(len(set(arrays["event_ids"]))),
        "nodes": int(len(node_ids)),
        "actions": int(len(action_ids)),
        "horizon_steps": int(horizon),
        "label_semantics": "PFV=candidate-no_control; TFV/peak=candidate-core26",
        "splits": pd.Series(arrays["split"]).value_counts().astype(int).to_dict() if arrays["split"] else {},
        "pfv_abs_margin_m3": pfv_abs_margin,
        "pfv_rel_margin": pfv_rel_margin,
        "pfv_safe_h120_count": int(pfv_safe.sum()) if len(audit) else 0,
        "pfv_unsafe_h120_count": int((~pfv_safe).sum()) if len(audit) else 0,
        "pfv_unsafe_h120_definition": "delta_pfv_h120 > max(pfv_abs_margin_m3, reference_pfv_h120 * pfv_rel_margin)",
        "tfv_improved_h120_count": int((audit["delta_tfv_h120"] < -100.0).sum()) if len(audit) else 0,
        "peak_unsafe_h120_count": int((audit["delta_peak_h120"] > 0.1).sum()) if len(audit) else 0,
        "audit": str(out / "residual10_core_effect_audit.csv"),
    }
    (out / "dataset_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
