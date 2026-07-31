from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sewerrtc.control.uncertainty_gate import evaluate_uncertainty_gate
from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config
from sewerrtc.models.temporal_graph_surrogate import TARGET_COLUMNS, load_horizon_surrogate
from sewerrtc.models.uncertainty import ResidualQuantileUncertainty


def _configured_dir(cfg: dict, key: str, default: str) -> Path:
    raw = (cfg.get("outputs", {}) or {}).get(key, default)
    path = Path(raw)
    if not path.is_absolute():
        path = cfg_path(cfg, "project_root") / path
    return path


def _read_dataset(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def _effect_truth(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {col: pd.to_numeric(df[f"effect_{col}"], errors="coerce").fillna(0.0) for col in TARGET_COLUMNS},
        index=df.index,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wuhan.yaml")
    ap.add_argument("--dataset", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--exact-effect-dataset", default="")
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = cfg_path(cfg, "project_root")
    hcfg = cfg.get("horizon_surrogate", {}) or {}
    dataset_path = Path(args.dataset) if args.dataset else root / hcfg.get("output_dataset", "data/surrogate/horizon_mpc_dataset.parquet")
    if not dataset_path.exists():
        dataset_path = root / hcfg.get("fallback_output_dataset", "data/surrogate/horizon_mpc_dataset.csv")
    if args.model:
        model_path = Path(args.model)
    else:
        model_dir = _configured_dir(cfg, "models", "outputs/models_paired_no_controls")
        temporal_path = model_dir / "horizon_temporal_gnn.pt"
        ridge_path = model_dir / "horizon_ridge_surrogate.npz"
        model_kind = str(hcfg.get("model_kind", hcfg.get("formal_model_kind", "ridge_baseline")))
        if model_kind == "temporal_gnn" and temporal_path.exists():
            model_path = temporal_path
        elif temporal_path.exists() and not ridge_path.exists():
            model_path = temporal_path
        else:
            model_path = ridge_path
    model = load_horizon_surrogate(model_path)
    model_dir = _configured_dir(cfg, "models", "outputs/models_paired_no_controls")
    unc = ResidualQuantileUncertainty.load(model_dir / "horizon_residual_quantile_uncertainty.npz")
    effect_path = Path(args.exact_effect_dataset) if args.exact_effect_dataset else None
    if effect_path is not None and effect_path.exists():
        df = _read_dataset(effect_path)
        pred = model.predict_effect(df)
        validation_scope = "exact_no_control_action_effect"
    elif all(f"effect_{c}" in _read_dataset(dataset_path).columns for c in getattr(model, "target_columns", [])) and callable(getattr(model, "predict_effect", None)):
        df = _read_dataset(dataset_path)
        pred = model.predict_effect(df)
        validation_scope = "paired_no_control_policy_effect"
    else:
        df = _read_dataset(dataset_path)
        pred = model.predict(df)
        validation_scope = "absolute_horizon_legacy"
    is_effect_scope = "effect" in validation_scope
    q = unc.predict_quantiles(pred, clip_lower=not is_effect_scope)
    ucfg = cfg.get("uncertainty_gate", {}) or {}
    decisions = []
    for idx, row in q.iterrows():
        reference_tfv = (
            float(pd.to_numeric(df.loc[idx, "reference_TFV_H"], errors="coerce"))
            if is_effect_scope and "reference_TFV_H" in df.columns
            else max(1.0, abs(float(row["TFV_H_p90"])))
        )
        reference_peak = (
            float(pd.to_numeric(df.loc[idx, "reference_peak_TFV_rate_H"], errors="coerce"))
            if is_effect_scope and "reference_peak_TFV_rate_H" in df.columns
            else max(1.0, abs(float(row["peak_TFV_rate_H_p90"])))
        )
        d = evaluate_uncertainty_gate(
            delta_pfv_p50=float(row["PFV_H_p50"]),
            delta_tfv_p90=float(row["TFV_H_p90"]),
            delta_peak_p90=float(row["peak_TFV_rate_H_p90"]),
            uncertainty_score=float(row["uncertainty_score"]),
            event_risk_class="medium_risk_event",
            min_pfv_gain=1.0,
            epsilon_tfv=float(ucfg.get("epsilon_tfv_pct", 0.005)) * max(1.0, reference_tfv),
            epsilon_peak=float(ucfg.get("epsilon_peak_pct", 0.010)) * max(1.0, reference_peak),
            max_uncertainty=float(ucfg.get("max_uncertainty", 1.0)),
        )
        decisions.append(d)
    out = q.copy()
    if is_effect_scope:
        truth = _effect_truth(df)
        for col in TARGET_COLUMNS:
            out[f"true_delta_{col}"] = truth[col].to_numpy(float)
        out["reference_TFV_H"] = pd.to_numeric(df["reference_TFV_H"], errors="coerce").fillna(0.0).to_numpy(float)
        out["reference_peak_TFV_rate_H"] = pd.to_numeric(df["reference_peak_TFV_rate_H"], errors="coerce").fillna(0.0).to_numpy(float)
    out["uncertainty_gate_pass"] = [d.pass_gate for d in decisions]
    out["uncertainty_gate_reason"] = [d.reason for d in decisions]
    out_dir = ensure_dir(_configured_dir(cfg, "surrogate", "outputs/surrogate"))
    out.to_csv(out_dir / "uncertainty_gate_validation.csv", index=False)
    summary = {
        "dataset": str(effect_path if validation_scope.startswith("exact") else dataset_path),
        "validation_scope": validation_scope,
        "prediction_semantics": "candidate_minus_no_control_effect" if is_effect_scope else "absolute_horizon_risk",
        "samples": int(len(out)),
        "gate_pass_rate": float(out["uncertainty_gate_pass"].mean()) if not out.empty else 0.0,
        "can_output": ["delta_PFV_p50", "delta_PFV_p90", "delta_TFV_p90", "delta_peak_p90", "uncertainty_score"],
        "output": str(out_dir / "uncertainty_gate_validation.csv"),
    }
    (out_dir / "uncertainty_gate_validation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
