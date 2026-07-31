from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config
from sewerrtc.models.temporal_graph_surrogate import (
    HorizonRidgeSurrogate,
    TARGET_COLUMNS,
    TemporalGraphHorizonSurrogate,
    regression_report,
    select_horizon_feature_columns,
)


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


def _row_random_split(df: pd.DataFrame, val_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rng = np.random.default_rng(int(seed))
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_val = max(1, int(len(df) * float(val_fraction)))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    if len(train_idx) == 0:
        train_idx = val_idx
    train = df.iloc[train_idx].copy()
    val = df.iloc[val_idx].copy()
    return train, val, {
        "split_strategy": "row_random_fallback",
        "train_events": [],
        "val_events": [],
    }


def _event_grouped_split(df: pd.DataFrame, val_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    if "event_id" not in df:
        return _row_random_split(df, val_fraction, seed)
    events = sorted(x for x in df["event_id"].dropna().astype(str).unique() if x)
    if len(events) < 2:
        train, val, split = _row_random_split(df, val_fraction, seed)
        split["split_strategy"] = "row_random_fallback_less_than_two_events"
        return train, val, split
    rng = np.random.default_rng(int(seed))
    shuffled = np.asarray(events, dtype=object)
    rng.shuffle(shuffled)
    n_val = max(1, int(round(len(shuffled) * float(val_fraction))))
    n_val = min(len(shuffled) - 1, n_val)
    val_events = sorted(str(x) for x in shuffled[:n_val])
    train_events = sorted(str(x) for x in shuffled[n_val:])
    event_series = df["event_id"].astype(str)
    train = df[event_series.isin(train_events)].copy()
    val = df[event_series.isin(val_events)].copy()
    if train.empty or val.empty:
        train, val, split = _row_random_split(df, val_fraction, seed)
        split["split_strategy"] = "row_random_fallback_empty_group"
        return train, val, split
    return train, val, {
        "split_strategy": "event_id_grouped",
        "train_events": train_events,
        "val_events": val_events,
    }


def _write_event_list(path: Path, role: str, event_ids: list[str]) -> Path:
    pd.DataFrame({"split": role, "event_id": event_ids}).to_csv(path, index=False)
    return path


def _attach_no_control_reference_targets(df: pd.DataFrame) -> pd.DataFrame:
    required = {"event_id", "policy_id", "row_index"}
    if not required.issubset(df.columns):
        return df
    targets = [c for c in TARGET_COLUMNS if c in df.columns]
    if targets and all(f"reference_{c}" in df.columns for c in targets):
        return df
    ref = df[df["policy_id"].astype(str).eq("no_control")][["event_id", "row_index"] + targets].copy()
    if ref.empty:
        return df
    ref = ref.drop_duplicates(["event_id", "row_index"])
    ref = ref.rename(columns={c: f"reference_{c}" for c in targets})
    out = df.merge(ref, on=["event_id", "row_index"], how="left", validate="many_to_one")
    for c in targets:
        out[f"reference_{c}"] = pd.to_numeric(out[f"reference_{c}"], errors="coerce").fillna(
            pd.to_numeric(out[c], errors="coerce").fillna(0.0)
        )
    return out


def _combine_effect_predictions(
    frame: pd.DataFrame,
    absolute: pd.DataFrame,
    effect: pd.DataFrame,
    reference_policy: str = "no_control",
) -> pd.DataFrame:
    """Calibrate uncertainty on the same paired effect prediction used online."""
    if not {"event_id", "row_index", "policy_id"}.issubset(frame.columns):
        return absolute
    work = frame.reset_index(drop=True)
    out = absolute.reset_index(drop=True).copy()
    eff = effect.reset_index(drop=True)
    ref_rows = work[work["policy_id"].astype(str).eq(reference_policy)]
    ref_indices = np.flatnonzero(work["policy_id"].astype(str).eq(reference_policy))
    ref_map = {
        (str(row.event_id), int(row.row_index)): int(idx)
        for row, idx in zip(ref_rows.itertuples(index=False), ref_indices)
    }
    for target in TARGET_COLUMNS:
        col = f"pred_{target}"
        values = out[col].to_numpy(float).copy()
        effects = pd.to_numeric(eff[col], errors="coerce").fillna(0.0).to_numpy(float)
        for i, row in work.iterrows():
            base_i = ref_map.get((str(row["event_id"]), int(row["row_index"])), i)
            values[i] = values[base_i] + effects[i]
        out[col] = np.maximum(values, 0.0)
    return out


def _write_effect_reports(
    val: pd.DataFrame,
    pred: pd.DataFrame,
    feature_columns: list[str],
    target_columns: list[str],
    out_dir: Path,
) -> tuple[Path, Path]:
    effect = val[["event_id", "policy_id", "phase"]].copy() if {"event_id", "policy_id", "phase"}.issubset(val.columns) else pd.DataFrame(index=val.index)
    if "policy_id" not in effect:
        effect["policy_id"] = "unknown"
    for col in target_columns:
        effect[f"true_{col}"] = pd.to_numeric(val[col], errors="coerce").fillna(0.0) if col in val else 0.0
        effect[f"pred_{col}"] = pd.to_numeric(pred[f"pred_{col}"], errors="coerce").fillna(0.0) if f"pred_{col}" in pred else 0.0
    policy_report = (
        effect.groupby("policy_id", as_index=False)
        .agg({f"true_{c}": "mean" for c in target_columns} | {f"pred_{c}": "mean" for c in target_columns})
        .sort_values("pred_PFV_H", ascending=False)
        if not effect.empty
        else pd.DataFrame()
    )
    policy_path = out_dir / "horizon_surrogate_policy_effect_report.csv"
    policy_report.to_csv(policy_path, index=False)

    sensitivity_rows = []
    action_like = [
        c
        for c in feature_columns
        if any(token in c for token in ("action", "delta", "retain", "release", "pump", "storage"))
    ]
    for feat in action_like:
        x = pd.to_numeric(val.get(feat, pd.Series(0.0, index=val.index)), errors="coerce").fillna(0.0)
        if float(x.std()) <= 1e-12:
            continue
        for col in target_columns:
            y = pd.to_numeric(val[col], errors="coerce").fillna(0.0) if col in val else pd.Series(0.0, index=val.index)
            if float(y.std()) <= 1e-12:
                continue
            corr = float(x.corr(y))
            if np.isfinite(corr):
                sensitivity_rows.append({"feature": feat, "target": col, "pearson_corr": corr, "abs_corr": abs(corr)})
    sens = pd.DataFrame(sensitivity_rows)
    if not sens.empty:
        sens = sens.sort_values(["target", "abs_corr"], ascending=[True, False])
    sensitivity_path = out_dir / "horizon_surrogate_action_sensitivity.csv"
    sens.to_csv(sensitivity_path, index=False)
    return policy_path, sensitivity_path


def _configured_output_dir(cfg: dict, key: str, default: str) -> Path:
    raw = (cfg.get("outputs", {}) or {}).get(key, default)
    path = Path(raw)
    if not path.is_absolute():
        path = cfg_path(cfg, "project_root") / path
    return ensure_dir(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wuhan.yaml")
    ap.add_argument("--dataset", default="")
    ap.add_argument(
        "--exact-effect-dataset",
        default="",
        help="Optional exact No-control replay counterfactual rows used by the action-effect head.",
    )
    ap.add_argument("--alpha", type=float, default=1e-2)
    ap.add_argument("--val-fraction", type=float, default=0.2)
    ap.add_argument("--allow-small-dataset", action="store_true")
    ap.add_argument("--min-samples", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.10)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--model-output", default="")
    ap.add_argument("--report-dir", default="")
    ap.add_argument(
        "--model-kind",
        choices=["ridge_baseline", "temporal_gnn"],
        default="",
        help="Horizon surrogate family. ridge_baseline is a smoke/baseline model; temporal_gnn is the formal GAT-MPC target.",
    )
    args = ap.parse_args()
    cfg = load_config(args.config)
    root = cfg_path(cfg, "project_root")
    hcfg = cfg.get("horizon_surrogate", {}) or {}
    model_kind = str(args.model_kind or hcfg.get("model_kind", "ridge_baseline"))
    dataset_path = Path(args.dataset) if args.dataset else root / hcfg.get("output_dataset", "data/surrogate/horizon_mpc_dataset.parquet")
    if not dataset_path.exists():
        fallback = root / hcfg.get("fallback_output_dataset", "data/surrogate/horizon_mpc_dataset.csv")
        if fallback.exists():
            dataset_path = fallback
    if not dataset_path.exists():
        raise FileNotFoundError(f"Missing horizon dataset: {dataset_path}")
    df = _read_dataset(dataset_path)
    if df.empty:
        raise ValueError("Horizon dataset is empty")
    df = _attach_no_control_reference_targets(df)
    exact_effect_path = Path(args.exact_effect_dataset) if args.exact_effect_dataset else None
    exact_effect_rows = 0
    if exact_effect_path is not None and exact_effect_path.exists():
        exact = _read_dataset(exact_effect_path)
        if not exact.empty:
            required = [f"effect_{c}" for c in TARGET_COLUMNS]
            missing = [c for c in required if c not in exact.columns]
            if missing:
                raise ValueError(f"Exact effect dataset is missing columns: {missing}")
            exact_effect_rows = int(len(exact))
            df = pd.concat([df, exact], ignore_index=True, sort=False)
    if len(df) < int(args.min_samples) and not args.allow_small_dataset:
        raise ValueError(
            f"Horizon dataset has only {len(df)} samples at {dataset_path}. "
            "This looks like a smoke/debug dataset. Rebuild Stage 2 without --max-detail-files "
            "for formal training, or pass --allow-small-dataset only for smoke validation."
        )
    feature_columns = select_horizon_feature_columns(df)
    if not feature_columns:
        raise ValueError("No numeric horizon surrogate features were found in the dataset")
    model_dir = _configured_output_dir(cfg, "models", "outputs/models_paired_no_controls")
    out_dir = ensure_dir(Path(args.report_dir)) if args.report_dir else _configured_output_dir(cfg, "surrogate", "outputs/surrogate")
    train, val, split = _event_grouped_split(
        df,
        val_fraction=float(args.val_fraction),
        seed=int(cfg["experiment"].get("random_seed", 2026)),
    )
    train_events_path = _write_event_list(out_dir / "horizon_surrogate_train_events.csv", "train", split["train_events"])
    val_events_path = _write_event_list(out_dir / "horizon_surrogate_val_events.csv", "val", split["val_events"])
    if model_kind == "ridge_baseline":
        model = HorizonRidgeSurrogate(alpha=float(args.alpha)).fit(train, feature_columns, TARGET_COLUMNS)
        model_path = Path(args.model_output) if args.model_output else model_dir / "horizon_ridge_surrogate.npz"
        formal_ready = False
        interpretation = "ridge_baseline is for smoke/baseline validation, not the formal transferable GAT-MPC surrogate."
    else:
        model = TemporalGraphHorizonSurrogate(
            hidden_dim=int(args.hidden_dim),
            layers=int(args.layers),
            dropout=float(args.dropout),
            seed=int(cfg["experiment"].get("random_seed", 2026)),
        ).fit(
            train,
            feature_columns,
            TARGET_COLUMNS,
            epochs=int(args.epochs),
            batch_size=int(args.batch_size),
            lr=float(args.lr),
            weight_decay=float(args.weight_decay),
            val_df=val,
            device=str(args.device),
            patience=int(args.patience),
        )
        model_path = Path(args.model_output) if args.model_output else model_dir / "horizon_temporal_gnn.pt"
        formal_ready = True
        interpretation = (
            "temporal_gnn is the formal Project5 neural horizon surrogate over graph-derived state, "
            "rainfall-window, influence-domain and action-sequence features."
        )
    pred = model.predict(val)
    effect_pred = None
    if model_kind == "temporal_gnn" and callable(getattr(model, "predict_effect", None)):
        effect_pred = model.predict_effect(val)
        pred = _combine_effect_predictions(val, pred, effect_pred)
    calibration_quantile = float(hcfg.get("conformal_calibration_quantile", 0.90))
    calibration_quantile = min(0.999, max(0.50, calibration_quantile))
    model.calibration_margins = {
        col: float(
            np.quantile(
                np.abs(
                    pd.to_numeric(val[col], errors="coerce").fillna(0.0).to_numpy(float)
                    - pd.to_numeric(pred[f"pred_{col}"], errors="coerce").fillna(0.0).to_numpy(float)
                ),
                calibration_quantile,
            )
        )
        for col in TARGET_COLUMNS
    }
    if effect_pred is not None and all(f"reference_{c}" in val.columns for c in TARGET_COLUMNS):
        effect_errors = {
            col: np.abs(
                (
                    pd.to_numeric(val[col], errors="coerce").fillna(0.0).to_numpy(float)
                    - pd.to_numeric(val[f"reference_{col}"], errors="coerce").fillna(0.0).to_numpy(float)
                )
                - pd.to_numeric(effect_pred[f"pred_{col}"], errors="coerce").fillna(0.0).to_numpy(float)
            )
            for col in TARGET_COLUMNS
        }
        quantiles = sorted({0.70, 0.80, calibration_quantile})
        model.effect_calibration_margins_by_quantile = {
            f"{q:.2f}": {col: float(np.quantile(errors, q)) for col, errors in effect_errors.items()}
            for q in quantiles
        }
        controller_q = float(hcfg.get("controller_uncertainty_quantile", calibration_quantile))
        controller_q = min(max(controller_q, 0.50), calibration_quantile)
        nearest = min(quantiles, key=lambda q: abs(q - controller_q))
        model.effect_calibration_margins = model.effect_calibration_margins_by_quantile[f"{nearest:.2f}"]
    # Calibration metadata is part of the deployed model and must be saved
    # after it has been estimated on event-held-out validation data.
    model.save(model_path)
    report = regression_report(val[TARGET_COLUMNS], pred, TARGET_COLUMNS)
    report.to_csv(out_dir / "horizon_surrogate_validation.csv", index=False)
    policy_effect_path, sensitivity_path = _write_effect_reports(val, pred, feature_columns, TARGET_COLUMNS, out_dir)
    meta = {
        "model_kind": model_kind,
        "formal_surrogate_ready": bool(formal_ready),
        "interpretation": interpretation,
        "dataset": str(dataset_path),
        "exact_effect_dataset": str(exact_effect_path) if exact_effect_path is not None else "",
        "exact_effect_rows": int(exact_effect_rows),
        "samples": int(len(df)),
        "conformal_calibration_quantile": calibration_quantile,
        "calibration_margins": model.calibration_margins,
        "effect_calibration_margins": model.effect_calibration_margins,
        "effect_calibration_margins_by_quantile": model.effect_calibration_margins_by_quantile,
        "train_samples": int(len(train)),
        "val_samples": int(len(val)),
        "split_strategy": split["split_strategy"],
        "train_events": split["train_events"],
        "val_events": split["val_events"],
        "train_event_count": int(len(split["train_events"])),
        "val_event_count": int(len(split["val_events"])),
        "train_events_path": str(train_events_path),
        "val_events_path": str(val_events_path),
        "feature_count": int(len(feature_columns)),
        "effect_supervision": bool(all(f"reference_{c}" in df.columns for c in TARGET_COLUMNS)),
        "target_columns": TARGET_COLUMNS,
        "model_path": str(model_path),
        "validation_report": str(out_dir / "horizon_surrogate_validation.csv"),
        "policy_effect_report": str(policy_effect_path),
        "action_sensitivity_report": str(sensitivity_path),
        "epochs": int(args.epochs) if model_kind == "temporal_gnn" else 0,
        "best_val_loss": float(getattr(model, "best_val_loss", float("nan"))) if model_kind == "temporal_gnn" else None,
    }
    (out_dir / "horizon_surrogate_train_report.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(json.dumps(meta, indent=2))
    print(report.to_string(index=False))


if __name__ == "__main__":
    main()
