from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

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
    if path.suffix.lower() == ".parquet":
        try:
            return pd.read_parquet(path)
        except ImportError:
            csv_path = path.with_suffix(".csv")
            if csv_path.exists():
                return pd.read_csv(csv_path)
            raise
    return pd.read_csv(path)


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
    df = _read_dataset(dataset_path)
    model = load_horizon_surrogate(model_path)
    effect_path = Path(args.exact_effect_dataset) if args.exact_effect_dataset else None
    calibration_scope = "absolute_horizon_legacy"
    if effect_path is not None and effect_path.exists():
        df = _read_dataset(effect_path)
        true = pd.DataFrame({c: pd.to_numeric(df[f"effect_{c}"], errors="coerce").fillna(0.0) for c in TARGET_COLUMNS})
        pred = model.predict_effect(df)
        calibration_scope = "exact_no_control_action_effect"
    elif all(f"effect_{c}" in df.columns for c in TARGET_COLUMNS) and callable(getattr(model, "predict_effect", None)):
        # The horizon dataset is paired to no_control.  Calibrating absolute
        # risks here would make the safety gate interpret hydraulic volume as
        # a candidate penalty rather than a relative action effect.
        true = pd.DataFrame({c: pd.to_numeric(df[f"effect_{c}"], errors="coerce").fillna(0.0) for c in TARGET_COLUMNS})
        pred = model.predict_effect(df)
        calibration_scope = "paired_no_control_policy_effect"
    else:
        true = df[TARGET_COLUMNS]
        pred = model.predict(df)
    unc = ResidualQuantileUncertainty.fit(true, pred, TARGET_COLUMNS)
    model_dir = _configured_dir(cfg, "models", "outputs/models_paired_no_controls")
    ensure_dir(model_dir)
    out_path = model_dir / "horizon_residual_quantile_uncertainty.npz"
    unc.save(out_path)
    quantiles = {f"{col}_q50": float(unc.q50[i]) for i, col in enumerate(TARGET_COLUMNS)}
    quantiles.update({f"{col}_q90": float(unc.q90[i]) for i, col in enumerate(TARGET_COLUMNS)})
    report = {
        "dataset": str(effect_path if calibration_scope.startswith("exact") else dataset_path),
        "model": str(model_path),
        "uncertainty_model": str(out_path),
        "calibration_scope": calibration_scope,
        "prediction_semantics": "candidate_minus_no_control_effect" if "effect" in calibration_scope else "absolute_horizon_risk",
        "samples": int(len(df)),
        "quantiles": quantiles,
    }
    out_dir = ensure_dir(_configured_dir(cfg, "surrogate", "outputs/surrogate"))
    (out_dir / "uncertainty_train_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
