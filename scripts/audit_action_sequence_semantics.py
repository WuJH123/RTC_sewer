from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sewerrtc.control.actuator_scope import select_actuators_for_scope
from sewerrtc.io.project_paths import cfg_path, load_config
from sewerrtc.simulation.pyswmm_runner import (
    _make_horizon_surrogate_predictor,
    _reference_horizon_arrays_from_detail,
)


TARGETS = ("PFV", "TFV", "peak_TFV_rate")


def totals(pred: dict[str, np.ndarray]) -> np.ndarray:
    return np.asarray([
        np.sum(pred["pfv"]), np.sum(pred["tfv"]), np.max(pred["peak_tfv_rate"])
    ], dtype=float)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wuhan_project6.yaml")
    ap.add_argument("--samples", type=int, default=6)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-dir", default="outputs/audits")
    ap.add_argument("--gate-summary", default="", help="Evaluate an existing semantics summary against Phase-2 thresholds and exit.")
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = cfg_path(cfg, "project_root")
    if args.gate_summary:
        source = Path(args.gate_summary)
        summary = json.loads(source.read_text(encoding="utf-8"))
        relative = summary.get("mean_relative_absolute_difference", {})
        zero = max((relative.get("zero_action_consistency", {}) or {}).values() or [float("inf")])
        identity = max((relative.get("actuator_identity_swap", {}) or {}).values() or [0.0])
        temporal = max((relative.get("temporal_order", {}) or {}).values() or [0.0])
        report = {
            "source": str(source), "zero_action_relative_error": zero,
            "identity_swap_max_relative_change": identity, "temporal_order_max_relative_change": temporal,
            "thresholds": {"zero_action_relative_error_lt": 0.005, "identity_or_temporal_relative_change_gt": 0.005},
            "passed": bool(zero < 0.005 and (identity > 0.005 or temporal > 0.005)),
            "note": "This gate evaluates the supplied model summary only; targeted SWMM sign agreement remains unavailable until targeted cases are run.",
        }
        out = root / args.out_dir; out.mkdir(parents=True, exist_ok=True)
        path = out / "action_sequence_acceptance_gate.json"; path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report, indent=2)); return
    h = int((cfg.get("horizon_surrogate", {}) or {}).get("horizon_steps", 6))
    actuators = pd.read_csv(cfg_path(cfg, "outputs.audit") / "actuator_table.csv")
    actuators = select_actuators_for_scope(actuators, str((cfg.get("controller", {}) or {}).get("actuator_scope", "existing_plus_retrofit")))
    actuator_ids = actuators["actuator_id"].astype(str).tolist()
    node_table = pd.read_csv(cfg_path(cfg, "outputs.audit") / "node_table.csv")
    node_ids = node_table["node_id"].astype(str).tolist()
    node_index = {node: i for i, node in enumerate(node_ids)}
    priority_nodes = [x.strip() for x in (cfg_path(cfg, "outputs.design") / "priority_nodes.txt").read_text().splitlines() if x.strip()]
    priority_idx = [node_index[x] for x in priority_nodes if x in node_index]
    influence_path = root / "outputs" / "network" / "priority_to_actuator_candidates.csv"
    influence = pd.read_csv(influence_path) if influence_path.exists() else None
    model_path = cfg_path(cfg, "outputs.models") / "horizon_temporal_gnn.pt"
    predictor = _make_horizon_surrogate_predictor(model_path, h, priority_idx, actuators, influence, args.device)

    val_path = root / "outputs" / "surrogate_all109" / "horizon_surrogate_val_events.csv"
    val_events = pd.read_csv(val_path)["event_id"].astype(str).tolist()
    detail_dir = cfg_path(cfg, "outputs.data_bank_train") / "trajectories"
    rows = []
    selected = 0
    for event_id in val_events:
        path = detail_dir / f"{event_id}__no_control_detail.csv"
        if not path.exists():
            continue
        detail = pd.read_csv(path, low_memory=False)
        depth_cols = [f"h:{node}" for node in node_ids if f"h:{node}" in detail]
        action_cols = [f"a:{aid}" for aid in actuator_ids if f"a:{aid}" in detail]
        if len(action_cols) != len(actuator_ids) or len(detail) <= h + 2:
            continue
        start = max(1, min(len(detail) - h - 1, len(detail) // 2))
        state = np.zeros(len(node_ids), dtype=np.float32)
        for idx, node in enumerate(node_ids):
            col = f"h:{node}"
            if col in detail:
                state[idx] = float(pd.to_numeric(detail[col], errors="coerce").iloc[start] or 0.0)
        rain = pd.to_numeric(detail.get("rainfall_mm_h", pd.Series(0.0, index=detail.index)), errors="coerce").fillna(0.0).to_numpy(float)[start:start+h]
        reference = detail[action_cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(np.float32)[start:start+h]
        ref_risk = _reference_horizon_arrays_from_detail(
            detail,
            elapsed_min=float(pd.to_numeric(detail["elapsed_min"], errors="coerce").iloc[start]),
            horizon_steps=h,
            dt_sec=300,
            priority_nodes=priority_nodes,
        )
        context = {
            "reconstructed_state": state,
            "rainfall_window": rain,
            "current_action": reference[0],
            "reference_action_sequence": reference,
            "reference_pfv": ref_risk["pfv"],
            "reference_tfv": ref_risk["tfv"],
            "reference_peak": ref_risk["peak_tfv_rate"],
        }
        a, b = 0, min(1, len(actuator_ids) - 1)
        low, high = 0.2, 0.8
        seq_a = reference.copy(); seq_a[:, a] = high
        seq_b = reference.copy(); seq_b[:, b] = high
        seq_ab = reference.copy(); seq_ab[:, [a, b]] = high
        early = reference.copy(); early[:, a] = np.asarray([high, high, low, low, low, low])[:h]
        late = reference.copy(); late[:, a] = np.asarray([low, low, low, low, high, high])[:h]
        preds = predictor.predict_many([reference, seq_a, seq_b, seq_ab, early, late], [context] * 6)
        values = [totals(pred) for pred in preds]
        effect_a, effect_b, effect_ab = values[1]-values[0], values[2]-values[0], values[3]-values[0]
        true_reference = np.asarray([
            np.sum(ref_risk["pfv"]), np.sum(ref_risk["tfv"]), np.max(ref_risk["peak_tfv_rate"])
        ], float)
        tests = {
            "zero_action_consistency": (values[0] - true_reference, np.maximum(np.abs(true_reference), 1.0)),
            "actuator_identity_swap": (values[1] - values[2], np.maximum.reduce([np.abs(values[1]), np.abs(values[2]), np.ones(3)])),
            "temporal_order": (values[4] - values[5], np.maximum.reduce([np.abs(values[4]), np.abs(values[5]), np.ones(3)])),
            "joint_interaction": (effect_ab - effect_a - effect_b, np.maximum.reduce([np.abs(effect_ab), np.abs(effect_a), np.abs(effect_b), np.ones(3)])),
        }
        for test, (differences, scales) in tests.items():
            for target, difference, scale in zip(TARGETS, differences, scales):
                rows.append({
                    "event_id": event_id, "row_index": start, "test": test, "target": target,
                    "actuator_a": actuator_ids[a], "actuator_b": actuator_ids[b],
                    "prediction_difference": float(difference), "absolute_difference": abs(float(difference)),
                    "comparison_scale": float(scale),
                    "relative_absolute_difference": abs(float(difference)) / float(scale),
                })
        selected += 1
        if selected >= args.samples:
            break
    result = pd.DataFrame(rows)
    out = root / args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "action_sequence_semantics.csv"
    result.to_csv(csv_path, index=False)
    summary = {
        "samples": selected,
        "model": str(model_path),
        "action_tensor_source_shape": [h, len(actuator_ids)],
        "model_receives_raw_action_tensor": False,
        "aggregation": "build_action_feature_map statistics, group summaries, and 16-bin hashed signatures",
        "mean_absolute_difference": result.groupby(["test", "target"])["absolute_difference"].mean().unstack().to_dict(orient="index") if not result.empty else {},
        "mean_relative_absolute_difference": result.groupby(["test", "target"])["relative_absolute_difference"].mean().unstack().to_dict(orient="index") if not result.empty else {},
        "interpretation": {
            "identity": "Near-zero identity-swap differences indicate actuator identity is not resolved by the compressed representation.",
            "order": "Near-zero early-vs-late differences indicate temporal order is not resolved.",
            "joint": "Non-zero residual denotes learned non-additivity in the compressed feature model, not validated hydraulic interaction.",
            "zero": "Non-zero values are predicted effect for an unchanged No-control action sequence.",
        },
    }
    summary_path = out / "action_sequence_semantics_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"csv": str(csv_path), "summary": str(summary_path), **summary}, indent=2))


if __name__ == "__main__":
    main()
