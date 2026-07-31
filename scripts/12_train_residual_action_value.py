from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config
from sewerrtc.control.candidate_generator import _slug, parse_candidate_label
from sewerrtc.models.residual_value import ResidualActionValueNet, feature_columns_from_frame


def _safe_threshold_for_precision(
    safe_prob: np.ndarray,
    y_safe: np.ndarray,
    min_precision: float = 0.70,
) -> tuple[float, float, float]:
    """Choose a conservative probability threshold for safe-action filtering.

    The controller can always reject more candidates, so a calibrated safe
    threshold is more meaningful than a fixed 0.5 cutoff. We prefer the lowest
    threshold that reaches the target precision and still keeps some recall.
    """
    safe_prob = np.asarray(safe_prob, dtype=float).reshape(-1)
    y_safe = np.asarray(y_safe, dtype=bool).reshape(-1)
    if safe_prob.size == 0:
        return 1.0, 0.0, 0.0
    best = (0.5, 0.0, 0.0)
    for thr in np.linspace(0.05, 0.95, 91):
        pred = safe_prob >= thr
        tp = int((pred & y_safe).sum())
        fp = int((pred & ~y_safe).sum())
        fn = int((~pred & y_safe).sum())
        if tp + fp == 0:
            continue
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        if precision >= min_precision:
            # Keep the highest recall among thresholds satisfying precision.
            if recall > best[2] or (np.isclose(recall, best[2]) and thr < best[0]):
                best = (float(thr), float(precision), float(recall))
    if best[1] > 0:
        return best
    # Fall back to the threshold with maximum precision, then recall.
    for thr in np.linspace(0.05, 0.95, 91):
        pred = safe_prob >= thr
        tp = int((pred & y_safe).sum())
        fp = int((pred & ~y_safe).sum())
        fn = int((~pred & y_safe).sum())
        if tp + fp == 0:
            continue
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        if precision > best[1] or (np.isclose(precision, best[1]) and recall > best[2]):
            best = (float(thr), float(precision), float(recall))
    return best


def _metrics(
    pred_delta: np.ndarray,
    true_delta: np.ndarray,
    probs: np.ndarray,
    y_safe_true: np.ndarray,
    y_pfv_true: np.ndarray | None = None,
    y_tfv_nonworse_true: np.ndarray | None = None,
    y_peak_nonworse_true: np.ndarray | None = None,
    safe_threshold: float | None = None,
) -> dict:
    if y_pfv_true is None:
        y_pfv = true_delta[:, 0] < 0
    else:
        y_pfv = np.asarray(y_pfv_true, dtype=bool).reshape(-1)
    if y_tfv_nonworse_true is None:
        y_tfv = true_delta[:, 1] <= 0
    else:
        y_tfv = np.asarray(y_tfv_nonworse_true, dtype=bool).reshape(-1)
    if y_peak_nonworse_true is None:
        y_peak = true_delta[:, 2] <= 0
    else:
        y_peak = np.asarray(y_peak_nonworse_true, dtype=bool).reshape(-1)
    # Prefer explicit classification heads when available. The delta
    # regression remains useful for ranking magnitude, but safety gates depend
    # on correctly classifying risk directions.
    pfv_pred = probs[:, 0] >= 0.5 if probs.shape[1] > 0 else pred_delta[:, 0] < 0
    tfv_pred = probs[:, 3] >= 0.5 if probs.shape[1] > 3 else pred_delta[:, 1] <= 0
    peak_pred = probs[:, 4] >= 0.5 if probs.shape[1] > 4 else pred_delta[:, 2] <= 0
    pfv_dir = np.mean(pfv_pred == y_pfv)
    tfv_dir = np.mean(tfv_pred == y_tfv)
    peak_dir = np.mean(peak_pred == y_peak)
    y_safe = np.asarray(y_safe_true, dtype=bool).reshape(-1)
    if safe_threshold is None:
        safe_threshold, calibrated_precision, calibrated_recall = _safe_threshold_for_precision(probs[:, 1], y_safe)
    else:
        pred_tmp = probs[:, 1] >= float(safe_threshold)
        tp_tmp = int((pred_tmp & y_safe).sum())
        fp_tmp = int((pred_tmp & ~y_safe).sum())
        fn_tmp = int((~pred_tmp & y_safe).sum())
        calibrated_precision = tp_tmp / max(1, tp_tmp + fp_tmp)
        calibrated_recall = tp_tmp / max(1, tp_tmp + fn_tmp)
    pred_safe = probs[:, 1] >= float(safe_threshold)
    safe_tp = int(((pred_safe == 1) & (y_safe == 1)).sum())
    safe_fp = int(((pred_safe == 1) & (y_safe == 0)).sum())
    safe_fn = int(((pred_safe == 0) & (y_safe == 1)).sum())
    return {
        "MAE_delta_PFV": float(np.mean(np.abs(pred_delta[:, 0] - true_delta[:, 0]))),
        "MAE_delta_TFV": float(np.mean(np.abs(pred_delta[:, 1] - true_delta[:, 1]))),
        "MAE_delta_peak": float(np.mean(np.abs(pred_delta[:, 2] - true_delta[:, 2]))),
        "PFV_direction_accuracy": float(pfv_dir),
        "TFV_direction_accuracy": float(tfv_dir),
        "peak_direction_accuracy": float(peak_dir),
        "safe_precision": float(safe_tp / max(1, safe_tp + safe_fp)),
        "safe_recall": float(safe_tp / max(1, safe_tp + safe_fn)),
        "safe_threshold": float(safe_threshold),
        "safe_precision_calibrated": float(calibrated_precision),
        "safe_recall_calibrated": float(calibrated_recall),
        "samples": int(len(true_delta)),
    }


def _tier_from_delta(df: pd.DataFrame) -> pd.Series:
    if "residual_delta" in df:
        d_abs = pd.to_numeric(df["residual_delta"], errors="coerce").abs()
    else:
        d_abs = pd.Series(np.nan, index=df.index)
    if "feat_delta_abs_max" in df:
        fb = pd.to_numeric(df["feat_delta_abs_max"], errors="coerce").abs()
        d_abs = d_abs.fillna(fb)
        d_abs = d_abs.mask(d_abs <= 1e-9, fb)
    return pd.Series(
        np.where(d_abs <= 0.080001, "small", np.where(d_abs <= 0.160001, "medium", "large")),
        index=df.index,
    )


def _zero_identity_action_features(df: pd.DataFrame) -> pd.DataFrame:
    """Force a copied residual row to represent candidate == native.

    These rows are not extra SWMM evidence; they are a hard identity constraint
    for the action-value model. In any physically consistent residual model,
    a zero action delta must imply zero PFV/TFV/peak delta and a safe but
    non-beneficial action.
    """
    out = df.copy()
    out["template_name"] = "identity"
    out["residual_delta"] = 0.0
    out["residual_delta_tier"] = "identity"
    out["case_id"] = out.get("case_id", pd.Series(np.arange(len(out)), index=out.index)).astype(str) + "__identity"
    for col in list(out.columns):
        lower = col.lower()
        if col.startswith("feat_") and (
            "delta" in lower
            or "changed_count" in lower
        ):
            out[col] = 0.0
    for col in ["delta_PFV", "delta_TFV", "delta_peak", "TFV", "PFV", "peak_TFV_rate"]:
        if col in out:
            if col.startswith("delta_"):
                out[col] = 0.0
    if "baseline_PFV" in out:
        out["PFV"] = out["baseline_PFV"]
    if "baseline_TFV" in out:
        out["TFV"] = out["baseline_TFV"]
    if "baseline_peak_TFV_rate" in out:
        out["peak_TFV_rate"] = out["baseline_peak_TFV_rate"]
    out["y_pfv_improve"] = 0
    out["y_safe"] = 1
    out["y_safe_guarded"] = 1
    return out


def _augment_identity_rows(
    df: pd.DataFrame,
    rng: np.random.Generator,
    frac: float,
    min_samples: int,
    max_samples: int,
) -> pd.DataFrame:
    if frac <= 0 and min_samples <= 0:
        return df
    n_target = max(int(round(len(df) * max(0.0, float(frac)))), int(min_samples))
    if max_samples > 0:
        n_target = min(n_target, int(max_samples))
    if n_target <= 0 or df.empty:
        return df
    source = df.copy()
    # Prefer actual high-risk event rows so identity constraints live in the
    # same feature distribution where the controller will evaluate actions.
    if "baseline_PFV" in source:
        source["_identity_rank"] = pd.to_numeric(source["baseline_PFV"], errors="coerce").fillna(0.0)
        source = source.sort_values("_identity_rank", ascending=False).drop(columns=["_identity_rank"])
    replace = n_target > len(source)
    sample_idx = rng.choice(source.index.to_numpy(), size=n_target, replace=replace)
    identity = _zero_identity_action_features(source.loc[sample_idx].reset_index(drop=True))
    return pd.concat([df, identity], ignore_index=True)


def _group_report(
    df: pd.DataFrame,
    indices: np.ndarray,
    pred_delta: np.ndarray,
    true_delta: np.ndarray,
    probs: np.ndarray,
    y_safe_true: np.ndarray,
    y_tfv_nonworse_true: np.ndarray | None,
    y_peak_nonworse_true: np.ndarray | None,
    safe_threshold: float,
) -> pd.DataFrame:
    meta = df.iloc[indices].copy().reset_index(drop=True)
    if "template_name" not in meta:
        meta["template_name"] = meta.get("template", "unknown")
    if "residual_delta" not in meta:
        meta["residual_delta"] = np.nan
    inferred = _tier_from_delta(meta)
    if "residual_delta_tier" not in meta:
        meta["residual_delta_tier"] = inferred
    else:
        tier = meta["residual_delta_tier"].fillna("").astype(str).str.strip()
        meta["residual_delta_tier"] = tier.mask(tier.eq(""), inferred)
    meta["_pred_delta_pfv"] = pred_delta[:, 0]
    meta["_pred_delta_tfv"] = pred_delta[:, 1]
    meta["_pred_delta_peak"] = pred_delta[:, 2]
    meta["_true_delta_pfv"] = true_delta[:, 0]
    meta["_true_delta_tfv"] = true_delta[:, 1]
    meta["_true_delta_peak"] = true_delta[:, 2]
    meta["_prob_pfv"] = probs[:, 0]
    meta["_prob_safe"] = probs[:, 1]
    meta["_prob_tfv_nonworse"] = probs[:, 3] if probs.shape[1] > 3 else 1.0
    meta["_prob_peak_nonworse"] = probs[:, 4] if probs.shape[1] > 4 else 1.0
    meta["_safe_true"] = np.asarray(y_safe_true, dtype=bool).reshape(-1)
    meta["_pfv_true"] = meta["_true_delta_pfv"] < 0
    if y_tfv_nonworse_true is None:
        meta["_tfv_true"] = meta["_true_delta_tfv"] <= 0
    else:
        meta["_tfv_true"] = np.asarray(y_tfv_nonworse_true, dtype=bool).reshape(-1)
    if y_peak_nonworse_true is None:
        meta["_peak_true"] = meta["_true_delta_peak"] <= 0
    else:
        meta["_peak_true"] = np.asarray(y_peak_nonworse_true, dtype=bool).reshape(-1)
    meta["_pfv_dir_ok"] = (meta["_prob_pfv"] >= 0.5) == meta["_pfv_true"]
    meta["_tfv_dir_ok"] = (meta["_prob_tfv_nonworse"] >= 0.5) == meta["_tfv_true"]
    meta["_peak_dir_ok"] = (meta["_prob_peak_nonworse"] >= 0.5) == meta["_peak_true"]
    meta["_safe_pred"] = meta["_prob_safe"] >= float(safe_threshold)
    meta["_safe_tp"] = meta["_safe_true"] & meta["_safe_pred"]
    meta["_safe_fp"] = (~meta["_safe_true"]) & meta["_safe_pred"]
    rows = []
    for keys, g in meta.groupby(["template_name", "residual_delta_tier", "residual_delta"], dropna=False):
        tp = int(g["_safe_tp"].sum())
        fp = int(g["_safe_fp"].sum())
        rows.append(
            {
                "template_name": keys[0],
                "residual_delta_tier": keys[1],
                "residual_delta": keys[2],
                "samples": int(len(g)),
                "events": int(g["event_id"].nunique()) if "event_id" in g else 0,
                "true_pfv_improve_frac": float((g["_true_delta_pfv"] < 0).mean()),
                "true_safe_frac": float(g["_safe_true"].mean()),
                "PFV_direction_accuracy": float(g["_pfv_dir_ok"].mean()),
                "TFV_direction_accuracy": float(g["_tfv_dir_ok"].mean()),
                "peak_direction_accuracy": float(g["_peak_dir_ok"].mean()),
                "safe_precision": float(tp / max(1, tp + fp)),
                "mean_abs_delta_PFV_error": float(np.mean(np.abs(g["_pred_delta_pfv"] - g["_true_delta_pfv"]))),
                "mean_true_delta_PFV": float(g["_true_delta_pfv"].mean()),
                "mean_pred_delta_PFV": float(g["_pred_delta_pfv"].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(["samples", "template_name"], ascending=[False, True])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wuhan.yaml")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden-dim", type=int, default=128)
    ap.add_argument("--cls-weight", type=float, default=0.35)
    ap.add_argument("--unsafe-loss-weight", type=float, default=2.5)
    ap.add_argument("--peak-loss-weight", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tfv-guard-pct", type=float, default=0.005)
    ap.add_argument("--peak-guard-pct", type=float, default=0.010)
    ap.add_argument("--gate-pfv-direction", type=float, default=0.70)
    ap.add_argument("--gate-safe-precision", type=float, default=0.80)
    ap.add_argument("--gate-peak-direction", type=float, default=0.80)
    ap.add_argument("--keep-no-change", action="store_true")
    ap.add_argument(
        "--identity-augmentation-frac",
        type=float,
        default=0.12,
        help="Add synthetic candidate==native rows as a hard zero-delta identity constraint.",
    )
    ap.add_argument("--identity-min-samples", type=int, default=200)
    ap.add_argument("--identity-max-samples", type=int, default=800)
    args = ap.parse_args()
    cfg = load_config(args.config)
    seed = int(args.seed if args.seed is not None else cfg["experiment"].get("random_seed", 2026))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    data_path = cfg_path(cfg, "outputs.closed_loop") / "internal_residual_counterfactuals" / "residual_counterfactual_results.csv"
    if not data_path.exists():
        raise FileNotFoundError(f"Missing residual counterfactual dataset: {data_path}")
    df = pd.read_csv(data_path)
    if "template_name" not in df:
        df["template_name"] = df.get("template", "unknown")
    df["template_name"] = df["template_name"].fillna("unknown").astype(str)
    if "candidate_scope" not in df:
        df["candidate_scope"] = "all"
    df["candidate_scope"] = df["candidate_scope"].fillna("all").astype(str)
    label_series = pd.Series("", index=df.index, dtype=object)
    for c in ["selected_candidate_label", "candidate_label", "template_name"]:
        if c not in df:
            continue
        s = df[c].fillna("").astype(str)
        label_series = label_series.mask(~label_series.str.contains(r"\|scope=", regex=True, na=False), s)
    needs_parse = label_series.str.contains(r"\|scope=", regex=True, na=False)
    if needs_parse.any():
        parsed = label_series[needs_parse].map(parse_candidate_label)
        idx = df.index[needs_parse]
        df.loc[idx, "template_name"] = [str(x.get("template", "unknown")) for x in parsed]
        df.loc[idx, "candidate_scope"] = [str(x.get("scope", "all")) for x in parsed]
        parsed_delta = pd.Series([float(x.get("delta", 0.0) or 0.0) for x in parsed], index=idx)
        parsed_hold = pd.Series([int(x.get("hold_steps", 1) or 1) for x in parsed], index=idx)
        if "residual_delta" not in df:
            df["residual_delta"] = np.nan
        missing_delta = pd.to_numeric(df.loc[idx, "residual_delta"], errors="coerce").isna()
        if missing_delta.any():
            df.loc[idx[missing_delta.to_numpy()], "residual_delta"] = parsed_delta.loc[missing_delta].to_numpy()
        df.loc[idx, "_parsed_hold_steps"] = parsed_hold
    df = _augment_identity_rows(
        df,
        np.random.default_rng(seed),
        float(args.identity_augmentation_frac),
        int(args.identity_min_samples),
        int(args.identity_max_samples),
    )
    inferred_tier = _tier_from_delta(df)
    if "residual_delta_tier" not in df:
        df["residual_delta_tier"] = inferred_tier
    else:
        tier = df["residual_delta_tier"].fillna("").astype(str).str.strip()
        df["residual_delta_tier"] = tier.mask(tier.eq(""), inferred_tier)
    # Encode the counterfactual action semantics explicitly. Without these
    # one-hot features, the value model must infer whether a residual action is
    # pump throttling or storage inlet restriction only from aggregate deltas,
    # which is too weak for safe action filtering.
    #
    # The residual counterfactual bank is intentionally assembled from multiple
    # generations of experiments. Older rows do not contain every newer
    # one-hot/metadata feature. Treat missing feature values as "feature not
    # active" (0.0); only labels are allowed to remove rows. This is critical
    # for reusing all SWMM evidence instead of silently training on the newest
    # narrow slice.
    original_rows = len(df)
    if "residual_delta" in df:
        residual_delta = pd.to_numeric(df["residual_delta"], errors="coerce")
    else:
        residual_delta = pd.Series(np.nan, index=df.index)
    if "feat_delta_abs_max" in df:
        fallback_delta = pd.to_numeric(df["feat_delta_abs_max"], errors="coerce")
        residual_delta = residual_delta.fillna(fallback_delta)
        residual_delta = residual_delta.mask(residual_delta.abs() <= 1e-9, fallback_delta)
    residual_delta = residual_delta.fillna(0.0).astype(float)
    residual_delta_abs = residual_delta.abs().astype(float)
    feature_parts = [
        pd.DataFrame(
            {
                "feat_residual_delta": residual_delta,
                "feat_residual_delta_abs": residual_delta_abs,
            },
            index=df.index,
        )
    ]
    if "feat_delta_abs_max" in df.columns:
        feat_delta_abs_max = pd.to_numeric(df["feat_delta_abs_max"], errors="coerce")
        df["feat_delta_abs_max"] = feat_delta_abs_max.fillna(residual_delta_abs).astype(float)
    else:
        feature_parts.append(pd.DataFrame({"feat_delta_abs_max": residual_delta_abs}, index=df.index))
    if "override_steps" in df:
        hold_steps = pd.to_numeric(df["override_steps"], errors="coerce").fillna(1.0).clip(lower=1.0)
    elif "_parsed_hold_steps" in df:
        hold_steps = pd.to_numeric(df["_parsed_hold_steps"], errors="coerce").fillna(1.0).clip(lower=1.0)
    elif "feat_hold_steps" not in df:
        hold_steps = pd.Series(1.0, index=df.index)
    else:
        hold_steps = pd.to_numeric(df["feat_hold_steps"], errors="coerce").fillna(1.0).clip(lower=1.0)
    feature_parts.append(pd.DataFrame({"feat_hold_steps": hold_steps.astype(float)}, index=df.index))

    template_series = df["template_name"].fillna("unknown").astype(str)
    template_features = {
        f"feat_template_{_slug(name)}": (template_series == name).astype(float)
        for name in sorted(template_series.unique())
    }
    if template_features:
        feature_parts.append(pd.DataFrame(template_features, index=df.index))

    tier_series = df["residual_delta_tier"].fillna("").astype(str)
    tier_names = sorted(set(["small", "medium", "large", "identity"]) | set(tier_series.unique()))
    tier_features = {
        f"feat_delta_tier_{_slug(name)}": (tier_series == name).astype(float)
        for name in tier_names
        if name
    }
    if tier_features:
        feature_parts.append(pd.DataFrame(tier_features, index=df.index))

    if "candidate_scope" in df:
        scopes = df["candidate_scope"].fillna("all").astype(str)
    else:
        scopes = pd.Series("all", index=df.index)
    scope_features = {
        f"feat_candidate_scope_{_slug(name)}": (scopes == name).astype(float)
        for name in sorted(scopes.unique())
    }
    if scope_features:
        feature_parts.append(pd.DataFrame(scope_features, index=df.index))

    if feature_parts:
        df = pd.concat([df, *feature_parts], axis=1)
        df = df.loc[:, ~df.columns.duplicated(keep="last")].copy()
    feature_cols = feature_columns_from_frame(df)
    if not feature_cols:
        raise RuntimeError("No feature columns found. Run scripts/11_generate_internal_residual_counterfactuals.py first.")
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[feature_cols] = df[feature_cols].fillna(0.0)
    for c in ["delta_PFV", "delta_TFV", "delta_peak"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    before_label_drop = len(df)
    df = df.dropna(subset=["delta_PFV", "delta_TFV", "delta_peak"]).reset_index(drop=True)
    dropped_label_rows = before_label_drop - len(df)
    if dropped_label_rows:
        print(f"[residual_value] dropped label-missing rows: {dropped_label_rows}; kept={len(df)}")
    if not args.keep_no_change and "feat_delta_abs_max" in df.columns:
        before = len(df)
        keep_identity = df.get("template_name", pd.Series("", index=df.index)).astype(str).eq("identity")
        keep_changed = pd.to_numeric(df["feat_delta_abs_max"], errors="coerce").fillna(0.0) > 1e-6
        df = df[keep_changed | keep_identity].reset_index(drop=True)
        print(f"[residual_value] dropped no-change rows: {before - len(df)}; kept={len(df)}")
    print(
        f"[residual_value] raw_rows={original_rows} usable_rows={len(df)} "
        f"feature_count={len(feature_cols)}"
    )
    if len(df) < 20:
        raise RuntimeError(f"Too few residual samples ({len(df)}). Generate more counterfactual cases first.")
    X = df[feature_cols].to_numpy(np.float32)
    Y = df[["delta_PFV", "delta_TFV", "delta_peak"]].to_numpy(np.float32)
    y_pfv = (Y[:, 0] < 0).astype(np.float32)
    if "baseline_TFV" in df.columns:
        baseline_tfv = pd.to_numeric(df["baseline_TFV"], errors="coerce").fillna(0.0).to_numpy(np.float32)
    else:
        baseline_tfv = np.zeros(len(df), dtype=np.float32)
    if "baseline_peak_TFV_rate" in df.columns:
        baseline_peak = pd.to_numeric(df["baseline_peak_TFV_rate"], errors="coerce").fillna(0.0).to_numpy(np.float32)
    else:
        baseline_peak = np.zeros(len(df), dtype=np.float32)
    tfv_guard = np.maximum(0.0, float(args.tfv_guard_pct) * baseline_tfv)
    peak_guard = np.maximum(0.0, float(args.peak_guard_pct) * baseline_peak)
    y_safe = ((Y[:, 1] <= tfv_guard) & (Y[:, 2] <= peak_guard)).astype(np.float32)
    y_nonzero = (np.abs(Y[:, 0]) > 1.0).astype(np.float32)
    y_tfv_nonworse = (Y[:, 1] <= tfv_guard).astype(np.float32)
    y_peak_nonworse = (Y[:, 2] <= peak_guard).astype(np.float32)
    Ycls = np.stack([y_pfv, y_safe, y_nonzero, y_tfv_nonworse, y_peak_nonworse], axis=1).astype(np.float32)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    val_n = max(4, int(0.2 * len(X)))
    val_idx, train_idx = idx[:val_n], idx[val_n:]
    x_mean = X[train_idx].mean(axis=0)
    x_std = X[train_idx].std(axis=0)
    x_std[x_std < 1e-6] = 1.0
    y_mean = Y[train_idx].mean(axis=0)
    y_std = Y[train_idx].std(axis=0)
    y_std[y_std < 1.0] = 1.0
    Xn = (X - x_mean) / x_std
    Yn = (Y - y_mean) / y_std
    dev = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    model = ResidualActionValueNet(X.shape[1], int(args.hidden_dim), output_dim=8).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=1e-4)
    train_ds = TensorDataset(
        torch.tensor(Xn[train_idx], dtype=torch.float32),
        torch.tensor(Yn[train_idx], dtype=torch.float32),
        torch.tensor(Ycls[train_idx], dtype=torch.float32),
    )
    weights = np.ones(len(train_idx), dtype=np.float32)
    weights[Ycls[train_idx, 0] > 0] *= 3.0
    weights[Ycls[train_idx, 1] == 0] *= 2.0
    sampler = WeightedRandomSampler(torch.tensor(weights, dtype=torch.double), len(weights), replacement=True)
    dl = DataLoader(train_ds, batch_size=int(args.batch_size), sampler=sampler)
    best_score = float("inf")
    out_dir = ensure_dir(cfg_path(cfg, "outputs.models"))
    diag_dir = ensure_dir(cfg_path(cfg, "outputs.diagnostics"))
    best_path = out_dir / "residual_action_value.pt"
    hist = []
    pos = torch.tensor(
        [
            max(1.0, float((Ycls[train_idx, i] == 0).sum()) / max(1.0, float((Ycls[train_idx, i] == 1).sum())))
            for i in range(Ycls.shape[1])
        ],
        dtype=torch.float32,
        device=dev,
    )
    for ep in range(1, int(args.epochs) + 1):
        model.train()
        losses = []
        for xb, yb, cb in dl:
            xb, yb, cb = xb.to(dev), yb.to(dev), cb.to(dev)
            out = model(xb)
            loss_delta = nn.functional.smooth_l1_loss(out["delta"], yb)
            bce = nn.functional.binary_cross_entropy_with_logits(
                out["logits"],
                cb,
                pos_weight=pos,
                reduction="none",
            )
            cls_weights = torch.ones_like(bce)
            # A false-safe action is much more harmful than rejecting a useful
            # action in the NativeShield controller. Penalize unsafe examples
            # on the safe head to improve precision rather than raw recall.
            cls_weights[:, 1] = torch.where(
                cb[:, 1] > 0.5,
                torch.ones_like(cb[:, 1]),
                torch.full_like(cb[:, 1], float(args.unsafe_loss_weight)),
            )
            cls_weights[:, 4] = float(args.peak_loss_weight)
            loss_cls = (bce * cls_weights).mean()
            loss = loss_delta + float(args.cls_weight) * loss_cls
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            losses.append(float(loss.detach().cpu()))
        if ep == 1 or ep % 10 == 0 or ep == int(args.epochs):
            model.eval()
            with torch.no_grad():
                xv = torch.tensor(Xn[val_idx], dtype=torch.float32, device=dev)
                out = model(xv)
                pred = (out["delta"].cpu().numpy() * y_std[None, :]) + y_mean[None, :]
                probs = torch.sigmoid(out["logits"]).cpu().numpy()
            m = _metrics(
                pred,
                Y[val_idx],
                probs,
                Ycls[val_idx, 1],
                y_pfv_true=Ycls[val_idx, 0],
                y_tfv_nonworse_true=Ycls[val_idx, 3],
                y_peak_nonworse_true=Ycls[val_idx, 4],
            )
            # For NativeShield, the saved checkpoint must first satisfy the
            # control-evidence gates. A lower MAE is not useful if it admits
            # unsafe residual actions or loses peak-direction skill.
            gate_shortfall = (
                max(0.0, float(args.gate_pfv_direction) - m["PFV_direction_accuracy"])
                + max(0.0, float(args.gate_safe_precision) - m["safe_precision"])
                + max(0.0, float(args.gate_peak_direction) - m["peak_direction_accuracy"])
            )
            score = (
                100000.0 * gate_shortfall
                + 8000.0 * (1.0 - m["PFV_direction_accuracy"])
                + 8000.0 * (1.0 - m["safe_precision"])
                + 6000.0 * (1.0 - m["peak_direction_accuracy"])
                + 0.005 * m["MAE_delta_PFV"]
                + 0.05 * m["MAE_delta_peak"]
            )
            m.update({"epoch": ep, "loss": float(np.mean(losses)), "score": float(score)})
            hist.append(m)
            marker = ""
            if score < best_score:
                best_score = score
                marker = " best"
                torch.save(
                    {
                        "model": model.state_dict(),
                        "feature_cols": feature_cols,
                        "x_mean": x_mean.astype(np.float32),
                        "x_std": x_std.astype(np.float32),
                        "y_mean": y_mean.astype(np.float32),
                        "y_std": y_std.astype(np.float32),
                        "hidden_dim": int(args.hidden_dim),
                        "output_dim": 8,
                        "metrics": m,
                        "safe_threshold": float(m.get("safe_threshold", 0.5)),
                        "train_samples": int(len(train_idx)),
                        "val_samples": int(len(val_idx)),
                    },
                    best_path,
                )
            print(
                f"epoch={ep:03d} loss={m['loss']:.5f} "
                f"PFV_dir={m['PFV_direction_accuracy']:.3f} "
                f"peak_dir={m['peak_direction_accuracy']:.3f} "
                f"safe_precision={m['safe_precision']:.3f}{marker}"
            )
    pd.DataFrame(hist).to_csv(diag_dir / "residual_action_value_training_report.csv", index=False)
    if best_path.exists():
        ckpt = torch.load(best_path, map_location=dev, weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.eval()
        with torch.no_grad():
            xv = torch.tensor(Xn[val_idx], dtype=torch.float32, device=dev)
            out = model(xv)
            pred = (out["delta"].cpu().numpy() * y_std[None, :]) + y_mean[None, :]
            probs = torch.sigmoid(out["logits"]).cpu().numpy()
        safe_threshold = float(ckpt.get("safe_threshold", ckpt.get("metrics", {}).get("safe_threshold", 0.5)))
        _group_report(
            df,
            val_idx,
            pred,
            Y[val_idx],
            probs,
            Ycls[val_idx, 1],
            Ycls[val_idx, 3],
            Ycls[val_idx, 4],
            safe_threshold,
        ).to_csv(
            diag_dir / "residual_action_value_group_report.csv",
            index=False,
        )
    balance = {
        "samples": int(len(df)),
        "train_samples": int(len(train_idx)),
        "val_samples": int(len(val_idx)),
        "pfv_improve": int(y_pfv.sum()),
        "safe": int(y_safe.sum()),
        "pfv_nonzero": int(y_nonzero.sum()),
        "events": int(df["event_id"].nunique()) if "event_id" in df else 0,
        "templates": int(df["template_name"].nunique()) if "template_name" in df else 0,
        "delta_tiers": sorted(df["residual_delta_tier"].dropna().astype(str).unique().tolist()),
        "tfv_guard_pct": float(args.tfv_guard_pct),
        "peak_guard_pct": float(args.peak_guard_pct),
        "kept_no_change_rows": bool(args.keep_no_change),
        "model_path": str(best_path),
        "feature_count": int(len(feature_cols)),
        "cls_weight": float(args.cls_weight),
        "unsafe_loss_weight": float(args.unsafe_loss_weight),
        "peak_loss_weight": float(args.peak_loss_weight),
        "seed": int(seed),
        "identity_augmentation_frac": float(args.identity_augmentation_frac),
        "identity_min_samples": int(args.identity_min_samples),
        "identity_max_samples": int(args.identity_max_samples),
        "identity_rows": int((df.get("template_name", pd.Series("", index=df.index)).astype(str) == "identity").sum()),
    }
    (diag_dir / "residual_action_value_balance.json").write_text(json.dumps(balance, indent=2), encoding="utf-8")
    print(json.dumps(balance, indent=2))


if __name__ == "__main__":
    main()
