"""Pilot Dataset v2, Baselines v2 and Gate v2 (contract project6_v4_pilot_gate_v2).

Merges the immutable Pilot400 v1 samples with the Joint Coverage Extension
and the conditional Flat Auxiliary samples into Dataset v2, audits it under
the versioned Gate v2 contract (docs/contracts/PROJECT6_V4_PILOT_GATE_V2.json),
trains the real sklearn baseline models and produces the Gate v2 verdict.

Hard rules enforced here fail-closed:

* primary400 rows are never dropped, relabelled or rerun;
* extension / flat-auxiliary rows only ever enter ``pilot_train``;
* evaluation splits (calibration / validation / challenge) stay pure
  primary400 and are never used for fitting, tuning or selection;
* actual-schedule uniqueness is per checkpoint state (the v1 hard gate) and
  every new sample must be globally disjoint from the v1 actual SHAs; the
  cross-state global uniqueness fraction is reported as a statistic only,
  because the frozen v1 evidence itself is 388/400 globally unique and the
  Gate v2 contract is not retroactive;
* models only see pre-run information (projected schedule vs anchor,
  projection K, family); no future SWMM results leak into features.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .partial_audit import HARD_AUTHENTICITY_COLUMNS

GATE_V2_CONTRACT_VERSION = "project6_v4_pilot_gate_v2"
PRIMARY_PHASE = "primary400"
SOURCE_PHASES = ("primary400", "joint_extension", "flat_auxiliary")
TRAIN_SPLIT = "pilot_train"
EVAL_SPLITS = ("pilot_calibration", "pilot_validation", "pilot_challenge")
REQUIRED_MODELS_V2 = (
    "zero_predictor",
    "majority_classifier",
    "ridge",
    "logistic_regression",
    "hist_gradient_boosting",
)
DELTA_TARGETS = (
    "delta_pfv_h120_vs_no_control",
    "delta_tfv_h120_vs_dynamic_internal",
    "delta_peak_h120_vs_dynamic_internal",
)
PRIMARY_REGRESSION_TARGET = "delta_tfv_h120_vs_dynamic_internal"
CLASS_TARGETS = ("joint_noninferior", "pfv_safe", "peak_noninferior")
PRIMARY_CLASS_TARGET = "joint_noninferior"
_STATE_KEYS = ["event_id", "checkpoint_id"]

_AUDIT_LABEL_COLUMNS = (
    "locally_responsive",
    "confirmed_flat",
    "materially_beneficial",
    "joint_noninferior",
    "pfv_safe",
    "tfv_noninferior",
    "peak_noninferior",
    "neutral",
)


def _finite_or_none(payload):
    """Recursively map non-finite floats (NaN/Inf) to None.

    The status writer uses ``allow_nan=False``; a NaN metric (for example the
    Spearman rank of a constant zero-predictor) means "not computable" and
    is reported honestly as null instead of crashing the report write.
    """
    if isinstance(payload, dict):
        return {key: _finite_or_none(value) for key, value in payload.items()}
    if isinstance(payload, (list, tuple)):
        return [_finite_or_none(value) for value in payload]
    if isinstance(payload, float) and not np.isfinite(payload):
        return None
    return payload


def _require(frame: pd.DataFrame, columns, *, name: str) -> None:
    missing = set(columns) - set(frame)
    if missing:
        raise ValueError(f"{name} missing columns: {sorted(missing)}")


def _with_phase_columns(
    frame: pd.DataFrame,
    *,
    source_phase: str,
    source_contract_version: str,
    selected_after_pilot_v1: bool,
) -> pd.DataFrame:
    out = frame.copy()
    if "source_phase" not in out:
        out["source_phase"] = source_phase
    out["source_phase"] = out["source_phase"].astype(str)
    if not out["source_phase"].eq(source_phase).all():
        raise ValueError(
            f"expected uniform source_phase={source_phase}, got "
            f"{sorted(out['source_phase'].unique())}"
        )
    out["source_contract_version"] = source_contract_version
    out["selected_after_pilot_v1"] = bool(selected_after_pilot_v1)
    return out


def build_pilot_dataset_v2(
    primary: pd.DataFrame,
    extension: pd.DataFrame | None = None,
    flat_auxiliary: pd.DataFrame | None = None,
    *,
    branch_frames: list[pd.DataFrame] | None = None,
) -> dict:
    """Merge primary400 + extension (+ flat auxiliary) into Dataset v2.

    The v1 rows are copied verbatim (never dropped or relabelled).  New rows
    must sit in ``pilot_train`` and must not repeat any actual schedule that
    already exists in the same checkpoint state or anywhere in v1; violators
    go to ``actual_duplicates`` and never into the sample manifest.
    """
    _require(
        primary,
        list(_STATE_KEYS)
        + ["sample_id", "split", "actual_schedule_sha256", "checkpoint_role"],
        name="primary sample manifest",
    )
    if len(primary) != 400:
        raise ValueError(
            f"primary400 manifest must hold 400 rows, got {len(primary)}"
        )
    primary = _with_phase_columns(
        primary,
        source_phase=PRIMARY_PHASE,
        source_contract_version="project6_v4_pilot_gate_v1",
        selected_after_pilot_v1=False,
    )
    v1_actual = set(primary["actual_schedule_sha256"].astype(str))
    state_actual: dict[tuple, set] = {
        key: set(group["actual_schedule_sha256"].astype(str))
        for key, group in primary.groupby(_STATE_KEYS)
    }
    parts: list[pd.DataFrame] = [primary]
    duplicates: list[dict] = []
    new_sources = (
        ("joint_extension", extension),
        ("flat_auxiliary", flat_auxiliary),
    )
    for phase, frame in new_sources:
        if frame is None or frame.empty:
            continue
        _require(
            frame,
            list(_STATE_KEYS)
            + [
                "sample_id",
                "split",
                "actual_schedule_sha256",
                "checkpoint_role",
            ],
            name=f"{phase} sample manifest",
        )
        frame = _with_phase_columns(
            frame,
            source_phase=phase,
            source_contract_version=GATE_V2_CONTRACT_VERSION,
            selected_after_pilot_v1=True,
        )
        if not frame["split"].astype(str).eq(TRAIN_SPLIT).all():
            raise ValueError(
                f"{phase} samples must all sit in {TRAIN_SPLIT}"
            )
        kept_rows: list[pd.Series] = []
        for _, row in frame.iterrows():
            key = (row["event_id"], row["checkpoint_id"])
            actual = str(row["actual_schedule_sha256"])
            reason = ""
            if actual in v1_actual:
                reason = "duplicate_actual_vs_primary400"
            elif actual in state_actual.setdefault(key, set()):
                reason = "duplicate_actual_within_state_v2"
            if reason:
                duplicates.append(
                    {
                        "sample_id": row["sample_id"],
                        "event_id": row["event_id"],
                        "checkpoint_id": row["checkpoint_id"],
                        "source_phase": phase,
                        "actual_schedule_sha256": actual,
                        "reason": reason,
                    }
                )
                continue
            state_actual[key].add(actual)
            kept_rows.append(row)
        if kept_rows:
            parts.append(pd.DataFrame(kept_rows))
    samples = pd.concat(parts, ignore_index=True, sort=False)
    if samples["sample_id"].astype(str).duplicated().any():
        raise ValueError("dataset v2 sample_id collision")
    split = samples["split"].astype(str)
    phase = samples["source_phase"].astype(str)
    samples["eligible_for_training"] = split.eq(TRAIN_SPLIT)
    samples["eligible_for_calibration"] = split.eq(
        "pilot_calibration"
    ) & phase.eq(PRIMARY_PHASE)
    samples["eligible_for_validation"] = split.eq(
        "pilot_validation"
    ) & phase.eq(PRIMARY_PHASE)
    samples["eligible_for_challenge"] = split.eq(
        "pilot_challenge"
    ) & phase.eq(PRIMARY_PHASE)

    split_manifest = (
        samples.groupby(["split", "source_phase"], as_index=False)
        .agg(
            samples=("sample_id", "size"),
            events=("event_id", "nunique"),
            states=("checkpoint_id", "nunique"),
        )
        .sort_values(["split", "source_phase"])
        .reset_index(drop=True)
    )
    source_rows = []
    for name, group in samples.groupby("source_phase"):
        source_rows.append(
            {
                "source_phase": name,
                "samples": int(len(group)),
                "events": int(group["event_id"].nunique()),
                "states": int(
                    group.drop_duplicates(_STATE_KEYS).shape[0]
                ),
                "joint_noninferior": int(
                    group.get(
                        "joint_noninferior",
                        pd.Series(False, index=group.index),
                    )
                    .astype(bool)
                    .sum()
                ),
                "confirmed_flat": int(
                    group.get(
                        "confirmed_flat",
                        pd.Series(False, index=group.index),
                    )
                    .astype(bool)
                    .sum()
                ),
            }
        )
    branch_manifest = (
        pd.concat(branch_frames, ignore_index=True, sort=False)
        if branch_frames
        else pd.DataFrame()
    )
    accounting = {
        "primary_rows": int(len(primary)),
        "primary_rows_kept": int(phase.eq(PRIMARY_PHASE).sum()),
        "extension_rows_kept": int(phase.eq("joint_extension").sum()),
        "flat_auxiliary_rows_kept": int(phase.eq("flat_auxiliary").sum()),
        "duplicates_removed": int(len(duplicates)),
        "total_samples": int(len(samples)),
        "accounting_closed": bool(
            int(phase.eq(PRIMARY_PHASE).sum()) == 400
            and len(samples) + len(duplicates)
            == 400
            + sum(
                len(frame)
                for _, frame in new_sources
                if frame is not None and not frame.empty
            )
        ),
    }
    return {
        "sample_manifest": samples,
        "branch_manifest": branch_manifest,
        "split_manifest": split_manifest,
        "source_manifest": pd.DataFrame(source_rows),
        "actual_duplicates": pd.DataFrame(duplicates),
        "accounting": accounting,
    }


def audit_pilot_dataset_v2(
    samples: pd.DataFrame,
    *,
    responsive_state_total: int = 32,
) -> dict:
    """Gate v2 dataset audit (versioned revision of the v1 audit).

    Hard authenticity gates are byte-identical to v1; the flat fraction band
    is replaced by the Flat v2 criteria and the joint coverage check gains
    the 10/32 minimum plus event support, per the Gate v2 contract.
    """
    required = set(_STATE_KEYS) | {
        "rainfall_sha256",
        "checkpoint_role",
        "actual_schedule_sha256",
        "state_hash_match",
        "readback_ok",
        "split",
        "source_phase",
        *_AUDIT_LABEL_COLUMNS,
    }
    missing = required - set(samples)
    if missing:
        return {"status": "blocked", "missing_columns": sorted(missing)}
    hard_missing = sorted(
        set(HARD_AUTHENTICITY_COLUMNS) - set(samples)
    )
    if hard_missing:
        return {"status": "blocked", "missing_columns": hard_missing}
    phase = samples["source_phase"].astype(str)
    primary = samples[phase.eq(PRIMARY_PHASE)]
    added = samples[~phase.eq(PRIMARY_PHASE)]
    responsive = samples[samples["checkpoint_role"] == "responsive"]
    flat = samples[samples["confirmed_flat"].astype(bool)]
    low_opp_states = samples[
        samples["checkpoint_role"] != "responsive"
    ].drop_duplicates(_STATE_KEYS)
    peak_degraded = samples[~samples["peak_noninferior"].astype(bool)]
    safe_peak_negative = peak_degraded[
        peak_degraded["pfv_safe"].astype(bool)
    ]
    checkpoint_joint = responsive.groupby(_STATE_KEYS)[
        "joint_noninferior"
    ].any()
    joint_states = checkpoint_joint[checkpoint_joint]
    joint_event_support = (
        joint_states.reset_index()["event_id"].nunique()
        if len(joint_states)
        else 0
    )
    flat_fraction = float(samples["confirmed_flat"].mean())

    def cross_event_both_sides(column: str) -> bool:
        values = samples.groupby("event_id")[column].agg(["any", "all"])
        positive_events = int(values["any"].sum())
        negative_events = int((~values["all"]).sum())
        return positive_events >= 3 and negative_events >= 3

    hard_matrix = samples[list(HARD_AUTHENTICITY_COLUMNS)].fillna(False)
    checks = {
        # -- structure and immutability -------------------------------
        "primary400_intact_400": len(primary) == 400,
        "accepted_events_8": samples["event_id"].nunique() == 8,
        "responsive_checkpoints_at_least_32": responsive.groupby(
            _STATE_KEYS
        ).ngroups
        >= responsive_state_total,
        "source_phase_values_valid": phase.isin(SOURCE_PHASES).all(),
        "extension_samples_train_only": (
            added["split"].astype(str).eq(TRAIN_SPLIT).all()
            if len(added)
            else True
        ),
        "evaluation_splits_primary_only": samples[
            samples["split"].astype(str).isin(EVAL_SPLITS)
        ]["source_phase"]
        .eq(PRIMARY_PHASE)
        .all(),
        "extension_states_subset_of_primary": (
            added.drop_duplicates(_STATE_KEYS)
            .set_index(_STATE_KEYS)
            .index.isin(
                primary.drop_duplicates(_STATE_KEYS)
                .set_index(_STATE_KEYS)
                .index
            )
            .all()
            if len(added)
            else True
        ),
        # -- hard execution authenticity (unchanged from v1) ----------
        "same_state_100pct": bool(samples["state_hash_match"].all()),
        "readback_100pct": bool(samples["readback_ok"].all()),
        "hard_authenticity_100pct": bool(
            hard_matrix.astype(bool).all().all()
        ),
        "actual_duplicates_0": not samples.duplicated(
            _STATE_KEYS + ["actual_schedule_sha256"]
        ).any(),
        "extension_actual_sha_disjoint_from_v1": (
            not added["actual_schedule_sha256"]
            .astype(str)
            .isin(set(primary["actual_schedule_sha256"].astype(str)))
            .any()
            if len(added)
            else True
        ),
        # -- informativeness -------------------------------------------
        "informative_actual_unique_at_least_300": len(
            samples.drop_duplicates(
                _STATE_KEYS + ["actual_schedule_sha256"]
            )
        )
        >= 300,
        "responsive_local_response_at_least_70pct": (
            float(responsive["locally_responsive"].mean()) >= 0.70
            if len(responsive)
            else False
        ),
        # -- Flat v2 (versioned revision, gate_change_control_v1) ------
        "confirmed_flat_at_least_8": len(flat) >= 8,
        "confirmed_flat_event_support_at_least_3": flat[
            "event_id"
        ].nunique()
        >= 3,
        "low_opportunity_state_support_at_least_8": len(low_opp_states)
        >= 8,
        "flat_fraction_at_most_20pct": flat_fraction <= 0.20,
        # -- Joint v2 ---------------------------------------------------
        "joint_at_30pct_responsive_checkpoints": (
            float(checkpoint_joint.mean()) >= 0.30
            if len(checkpoint_joint)
            else False
        ),
        "joint_state_count_at_least_10": int(checkpoint_joint.sum())
        >= 10,
        "joint_event_support_at_least_4": int(joint_event_support) >= 4,
        # -- label boundary coverage (unchanged from v1) ---------------
        "material_benefit_3_events": samples[
            samples["materially_beneficial"].astype(bool)
        ]["event_id"].nunique()
        >= 3,
        "pfv_both_sides_3_events": cross_event_both_sides("pfv_safe"),
        "tfv_both_sides_3_events": cross_event_both_sides(
            "tfv_noninferior"
        ),
        "peak_both_sides_3_events": cross_event_both_sides(
            "peak_noninferior"
        ),
        "peak_degraded_at_least_30": len(peak_degraded) >= 30,
        "pfv_safe_peak_negative_at_least_10": len(safe_peak_negative)
        >= 10,
        "neutral_present": bool(samples["neutral"].astype(bool).any()),
        "rainfall_sha_split_isolated": not samples.groupby(
            "rainfall_sha256"
        )["split"]
        .nunique()
        .gt(1)
        .any(),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    return {
        "status": "pass" if all(checks.values()) else "scientific_fail",
        "gate_version": "v2",
        "contract": GATE_V2_CONTRACT_VERSION,
        "checks": checks,
        "headline": {
            "total_samples": int(len(samples)),
            "samples_by_source_phase": {
                str(k): int(v)
                for k, v in phase.value_counts().items()
            },
            "confirmed_flat_count": int(len(flat)),
            "confirmed_flat_event_support": int(
                flat["event_id"].nunique()
            ),
            "flat_fraction_observed": flat_fraction,
            "joint_state_count": int(checkpoint_joint.sum()),
            "responsive_state_count": int(len(checkpoint_joint)),
            "joint_responsive_state_fraction": (
                float(checkpoint_joint.mean())
                if len(checkpoint_joint)
                else 0.0
            ),
            "joint_event_support": int(joint_event_support),
            "joint_target_12_reached": int(checkpoint_joint.sum()) >= 12,
            "actual_sha_global_unique_fraction": (
                float(
                    samples["actual_schedule_sha256"].nunique()
                    / len(samples)
                )
                if len(samples)
                else 0.0
            ),
        },
        "notes": [
            "actual-schedule uniqueness is enforced per checkpoint state "
            "(v1 hard gate) plus zero overlap of new samples with v1; the "
            "cross-state global uniqueness fraction is a report-only "
            "statistic because frozen v1 evidence is itself 388/400 and "
            "the Gate v2 contract is not retroactive",
        ],
    }


# --------------------------------------------------------------------------
# Baselines v2
# --------------------------------------------------------------------------


def _schedule_matrix(text: str) -> np.ndarray:
    return np.asarray(json.loads(str(text)), dtype=float)


def candidate_features(
    samples: pd.DataFrame,
    *,
    family_categories: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Pre-run-only features: projected-vs-anchor geometry + K + family.

    Never touches deltas, labels, residuals or any post-run column, so no
    future SWMM information can leak into the models.
    """
    _require(
        samples,
        [
            "projected_schedule_json",
            "anchor_schedule_json",
            "k_target",
            "k_actual",
            "checkpoint_min",
            "candidate_family",
        ],
        name="feature source",
    )
    rows = []
    for _, row in samples.iterrows():
        projected = _schedule_matrix(row["projected_schedule_json"])
        anchor = _schedule_matrix(row["anchor_schedule_json"])
        diff = projected - anchor
        abs_diff = np.abs(diff)
        rows.append(
            {
                "k_target": float(row["k_target"]),
                "k_actual": float(row["k_actual"]),
                "checkpoint_min": float(row["checkpoint_min"]),
                "move_l1_mean": float(abs_diff.mean()),
                "move_max": float(abs_diff.max()),
                "facilities_moved": float(
                    (abs_diff.max(axis=0) > 1e-9).sum()
                ),
                "steps_changed": float(
                    (abs_diff.max(axis=1) > 1e-9).sum()
                ),
                "net_direction": float(diff.mean()),
            }
        )
    numeric = pd.DataFrame(rows, index=samples.index).fillna(0.0)
    families = samples["candidate_family"].astype(str)
    if family_categories is None:
        family_categories = sorted(families.unique())
    for name in family_categories:
        numeric[f"family__{name}"] = families.eq(name).astype(float)
    return numeric, list(family_categories)


def _regression_metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    error = y_pred - y_true
    rank_corr = (
        float(
            pd.Series(y_pred).rank().corr(pd.Series(y_true).rank())
        )
        if len(y_true) > 2 and pd.Series(y_true).nunique() > 1
        else None
    )
    return {
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "spearman": rank_corr,
        "n": int(len(y_true)),
    }


def _classification_metrics(y_true, y_pred, y_score) -> dict:
    from sklearn.metrics import (
        average_precision_score,
        balanced_accuracy_score,
        f1_score,
        matthews_corrcoef,
        roc_auc_score,
    )

    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)
    both_classes = len(np.unique(y_true)) > 1
    metrics = {
        "balanced_accuracy": float(
            balanced_accuracy_score(y_true, y_pred)
        ),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "mcc": float(matthews_corrcoef(y_true, y_pred))
        if both_classes
        else None,
        "auroc": float(roc_auc_score(y_true, y_score))
        if both_classes and y_score is not None
        else None,
        "auprc": float(average_precision_score(y_true, y_score))
        if both_classes and y_score is not None
        else None,
        "n": int(len(y_true)),
        "positives": int(y_true.sum()),
    }
    predicted_safe = int(y_pred.sum())
    metrics["false_safe_rate"] = (
        float((y_pred & ~y_true).sum() / predicted_safe)
        if predicted_safe
        else None
    )
    negatives = int((~y_true).sum())
    metrics["degraded_recall"] = (
        float((~y_pred & ~y_true).sum() / negatives)
        if negatives
        else None
    )
    return metrics


def _topk_and_regret(
    frame: pd.DataFrame,
    scores: np.ndarray,
    *,
    ks: tuple[int, ...] = (1, 3, 5),
) -> dict:
    """Per-state Top-K feasible recall and decision regret on eval rows."""
    work = frame.copy()
    work["_score"] = scores
    hits = {k: [] for k in ks}
    regrets = []
    for _, group in work.groupby(_STATE_KEYS):
        feasible = group["joint_noninferior"].astype(bool)
        ranked = group.sort_values("_score", ascending=False)
        if feasible.any():
            for k in ks:
                hits[k].append(
                    bool(
                        ranked.head(k)["joint_noninferior"]
                        .astype(bool)
                        .any()
                    )
                )
        tfv = pd.to_numeric(
            group[PRIMARY_REGRESSION_TARGET], errors="coerce"
        )
        if tfv.notna().all() and len(group) > 1:
            best = float(tfv.min())
            chosen = float(
                pd.to_numeric(
                    ranked[PRIMARY_REGRESSION_TARGET], errors="coerce"
                ).iloc[0]
            )
            regrets.append(chosen - best)
    return {
        **{
            f"top{k}_feasible_recall": (
                float(np.mean(hits[k])) if hits[k] else None
            )
            for k in ks
        },
        "states_with_feasible": int(len(hits[ks[0]])),
        "decision_regret_mean": (
            float(np.mean(regrets)) if regrets else None
        ),
        "decision_regret_worst": (
            float(np.max(regrets)) if regrets else None
        ),
    }


def _bootstrap_interval(
    values_true,
    values_pred,
    metric,
    *,
    n_boot: int,
    seed: int,
) -> dict | None:
    values_true = np.asarray(values_true)
    values_pred = np.asarray(values_pred)
    if len(values_true) < 5:
        return None
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(int(n_boot)):
        index = rng.integers(0, len(values_true), len(values_true))
        try:
            stats.append(metric(values_true[index], values_pred[index]))
        except ValueError:
            continue
    if not stats:
        return None
    return {
        "p2_5": float(np.percentile(stats, 2.5)),
        "p97_5": float(np.percentile(stats, 97.5)),
        "n_boot": int(len(stats)),
    }


def train_pilot_baselines_v2(
    samples: pd.DataFrame,
    *,
    seed: int = 20260723,
    n_boot: int = 200,
) -> dict:
    """Fit the five real baseline models and evaluate on primary400 splits.

    Training rows: every ``eligible_for_training`` sample (primary400
    pilot_train + joint extension + flat auxiliary).  Evaluation rows: only
    primary400 calibration / validation / challenge, never used for fitting
    or tuning; hyperparameters are fixed constants from
    ``training.build_baseline_models``.
    """
    from sklearn.metrics import balanced_accuracy_score

    from .training import build_baseline_models

    _require(
        samples,
        ["eligible_for_training", "split", "source_phase"]
        + list(DELTA_TARGETS)
        + list(CLASS_TARGETS),
        name="dataset v2 samples",
    )
    train = samples[samples["eligible_for_training"].astype(bool)]
    if train.empty:
        raise ValueError("no eligible training rows")
    evals = {
        split: samples[
            samples["split"].astype(str).eq(split)
            & samples["source_phase"].astype(str).eq(PRIMARY_PHASE)
        ]
        for split in EVAL_SPLITS
    }
    x_train, family_categories = candidate_features(train)
    x_eval = {
        split: candidate_features(
            frame, family_categories=family_categories
        )[0]
        for split, frame in evals.items()
        if len(frame)
    }
    report: dict = {
        "contract": GATE_V2_CONTRACT_VERSION,
        "seed": int(seed),
        "n_boot": int(n_boot),
        "feature_columns": list(x_train.columns),
        "split_policy": {
            "train_rows": int(len(train)),
            "train_source_phases": {
                str(k): int(v)
                for k, v in train["source_phase"]
                .value_counts()
                .items()
            },
            "train_only_pilot_train": bool(
                train["split"].astype(str).eq(TRAIN_SPLIT).all()
            ),
            "eval_rows": {
                split: int(len(frame))
                for split, frame in evals.items()
            },
            "eval_primary400_only": bool(
                all(
                    frame["source_phase"].eq(PRIMARY_PHASE).all()
                    for frame in evals.values()
                    if len(frame)
                )
            ),
            "no_calibration_fitting": True,
            "no_validation_tuning": True,
            "no_challenge_selection": True,
            "extension_rows_never_evaluated": True,
        },
        "regression": {},
        "classification": {},
        "ranking": {},
        "models": {},
    }
    models = build_baseline_models()

    # ---- regression: zero predictor + ridge per delta target ----------
    for target in DELTA_TARGETS:
        y_train = pd.to_numeric(train[target], errors="coerce")
        mask = y_train.notna()
        entry: dict = {}
        for name in ("zero_predictor", "ridge"):
            model = build_baseline_models()[name]
            model.fit(x_train[mask.values], y_train[mask.values])
            per_split = {}
            for split, frame in evals.items():
                if not len(frame):
                    continue
                y_true = pd.to_numeric(frame[target], errors="coerce")
                keep = y_true.notna()
                if not keep.any():
                    continue
                pred = model.predict(x_eval[split][keep.values])
                per_split[split] = _regression_metrics(
                    y_true[keep.values], pred
                )
                per_split[split]["per_event_worst_rmse"] = float(
                    max(
                        _regression_metrics(
                            y_true[keep.values][
                                frame.loc[keep.values, "event_id"]
                                == event
                            ],
                            pred[
                                (
                                    frame.loc[
                                        keep.values, "event_id"
                                    ]
                                    == event
                                ).to_numpy()
                            ],
                        )["rmse"]
                        for event in frame.loc[
                            keep.values, "event_id"
                        ].unique()
                    )
                )
                if (
                    split == "pilot_validation"
                    and target == PRIMARY_REGRESSION_TARGET
                ):
                    per_split[split]["rmse_bootstrap"] = (
                        _bootstrap_interval(
                            y_true[keep.values].to_numpy(),
                            pred,
                            lambda t, p: float(
                                np.sqrt(np.mean((p - t) ** 2))
                            ),
                            n_boot=n_boot,
                            seed=seed,
                        )
                    )
            entry[name] = per_split
        report["regression"][target] = entry

    # ---- classification tasks ------------------------------------------
    class_models = (
        "majority_classifier",
        "logistic_regression",
        "hist_gradient_boosting",
    )
    trained_flags: dict[str, bool] = {}
    for target in CLASS_TARGETS:
        y_train = train[target].astype(bool)
        entry = {}
        degenerate = y_train.nunique() < 2
        for name in class_models:
            if degenerate:
                entry[name] = {"trained": False, "reason": "single_class"}
                trained_flags.setdefault(name, False)
                continue
            model = build_baseline_models()[name]
            model.fit(x_train, y_train)
            trained_flags[name] = trained_flags.get(name, True) and True
            per_split = {}
            for split, frame in evals.items():
                if not len(frame):
                    continue
                y_true = frame[target].astype(bool)
                pred = model.predict(x_eval[split]).astype(bool)
                score = None
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(x_eval[split])
                    if proba.shape[1] == 2:
                        score = proba[:, 1]
                per_split[split] = _classification_metrics(
                    y_true, pred, score
                )
                per_event = {
                    str(event): float(
                        balanced_accuracy_score(
                            y_true[frame["event_id"] == event],
                            pred[
                                (
                                    frame["event_id"] == event
                                ).to_numpy()
                            ],
                        )
                    )
                    for event in frame["event_id"].unique()
                    if y_true[frame["event_id"] == event].nunique()
                    > 1
                }
                if per_event:
                    per_split[split]["per_event_worst_balanced_accuracy"] = (
                        float(min(per_event.values()))
                    )
                if (
                    split == "pilot_validation"
                    and target == PRIMARY_CLASS_TARGET
                ):
                    per_split[split]["balanced_accuracy_bootstrap"] = (
                        _bootstrap_interval(
                            y_true.to_numpy(),
                            pred,
                            lambda t, p: float(
                                balanced_accuracy_score(t, p)
                            ),
                            n_boot=n_boot,
                            seed=seed,
                        )
                    )
            entry[name] = {"trained": True, "splits": per_split}
        report["classification"][target] = entry

    # ---- ranking: Top-K feasible recall + decision regret ---------------
    hist_entry = report["classification"][PRIMARY_CLASS_TARGET].get(
        "hist_gradient_boosting", {}
    )
    if hist_entry.get("trained"):
        ranker = build_baseline_models()["hist_gradient_boosting"]
        ranker.fit(x_train, train[PRIMARY_CLASS_TARGET].astype(bool))
        for split, frame in evals.items():
            if not len(frame):
                continue
            scores = ranker.predict_proba(x_eval[split])[:, 1]
            report["ranking"][split] = _topk_and_regret(frame, scores)

    # ---- model summary flags for the gate --------------------------------
    def _validation(block, name, key):
        return (
            block.get(name, {})
            .get("splits", block.get(name, {}))
            .get("pilot_validation", {})
            .get(key)
        )

    zero_rmse = (
        report["regression"][PRIMARY_REGRESSION_TARGET]["zero_predictor"]
        .get("pilot_validation", {})
        .get("rmse")
    )
    ridge_rmse = (
        report["regression"][PRIMARY_REGRESSION_TARGET]["ridge"]
        .get("pilot_validation", {})
        .get("rmse")
    )
    joint_block = report["classification"][PRIMARY_CLASS_TARGET]
    majority_ba = _validation(
        joint_block, "majority_classifier", "balanced_accuracy"
    )
    logistic_ba = _validation(
        joint_block, "logistic_regression", "balanced_accuracy"
    )
    hist_ba = _validation(
        joint_block, "hist_gradient_boosting", "balanced_accuracy"
    )
    report["models"] = {
        "zero_predictor": {
            "role": "trivial_reference",
            "validation_rmse": zero_rmse,
        },
        "majority_classifier": {
            "role": "trivial_reference",
            "validation_balanced_accuracy": majority_ba,
        },
        "ridge": {
            "validation_rmse": ridge_rmse,
            "beats_zero_prediction": bool(
                ridge_rmse is not None
                and zero_rmse is not None
                and ridge_rmse < zero_rmse
            ),
        },
        "logistic_regression": {
            "validation_balanced_accuracy": logistic_ba,
            "beats_majority_class": bool(
                logistic_ba is not None
                and majority_ba is not None
                and logistic_ba > majority_ba
                and logistic_ba > 0.5
            ),
        },
        "hist_gradient_boosting": {
            "validation_balanced_accuracy": hist_ba,
            "beats_majority_class": bool(
                hist_ba is not None
                and majority_ba is not None
                and hist_ba > majority_ba
                and hist_ba > 0.5
            ),
        },
    }
    return _finite_or_none(report)


def evaluate_pilot_gate_v2(
    dataset_audit: dict, baseline_report: dict
) -> dict:
    """Gate v2 verdict; never authorises Train1600 automatically."""
    dataset_pass = dataset_audit.get("status") == "pass"
    models = baseline_report.get("models", {})
    split_policy = baseline_report.get("split_policy", {})
    checks = {
        "dataset_audit_v2_pass": dataset_pass,
        "required_models_present": set(REQUIRED_MODELS_V2).issubset(
            set(models)
        ),
        "train_only_pilot_train": bool(
            split_policy.get("train_only_pilot_train", False)
        ),
        "eval_primary400_only": bool(
            split_policy.get("eval_primary400_only", False)
        ),
        "ridge_beats_zero": bool(
            models.get("ridge", {}).get("beats_zero_prediction", False)
        ),
        "logistic_beats_majority": bool(
            models.get("logistic_regression", {}).get(
                "beats_majority_class", False
            )
        ),
        "hist_gradient_boosting_beats_majority": bool(
            models.get("hist_gradient_boosting", {}).get(
                "beats_majority_class", False
            )
        ),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    scientific_pass = all(checks.values())
    return {
        "status": "pass" if scientific_pass else "scientific_fail",
        "gate_version": "v2",
        "contract": GATE_V2_CONTRACT_VERSION,
        "scientific_pass": scientific_pass,
        "exit_code": 0 if scientific_pass else 5,
        "checks": checks,
        "dataset_checks": {
            key: bool(value)
            for key, value in dataset_audit.get("checks", {}).items()
        },
        "dataset_headline": dataset_audit.get("headline", {}),
        "train1600_authorization": "manual_decision_required",
        "auto_entry_into_train1600_prohibited": True,
    }
