"""Pilot Dataset v3, Baselines v3 and Gate v3 (contract project6_v4_learning_task_v3).

Implements the two-tier learning task of docs/contracts/
PROJECT6_V4_LEARNING_TASK_V3.json on top of the frozen Pilot v2 dataset
(479 rows) and the Gate P3 feasibility map dataset (653 rows):

* state-level head: when does an intervention opportunity exist
  (labels mapped from the frozen feasibility classes);
* candidate-level heads: continuous deltas, safety heads and a composed
  joint noninferiority (no standalone joint binary classifier gate).

Hard rules enforced here fail-closed:

* v2 rows (primary400 + joint extension + flat auxiliary) are copied
  verbatim, never dropped or relabelled;
* exact-search samples from calibration/validation/challenge states never
  enter training (only ``pilot_train`` search rows are training-eligible);
* oracle-revealed states are excluded from unseen evaluation; their
  metrics are reported separately as diagnostics only;
* models only see pre-run information (projected schedule vs anchor,
  projection K, family) and, for the state head, online-available state
  descriptors (checkpoint time, rainfall phase); no exact-search outcome
  and no future SWMM result is ever a feature.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .partial_audit import HARD_AUTHENTICITY_COLUMNS
from .pilot_v2 import (
    DELTA_TARGETS,
    EVAL_SPLITS,
    PRIMARY_PHASE,
    PRIMARY_REGRESSION_TARGET,
    SOURCE_PHASES,
    TRAIN_SPLIT,
    _bootstrap_interval,
    _classification_metrics,
    _finite_or_none,
    _regression_metrics,
    _require,
    _topk_and_regret,
    candidate_features,
)

LEARNING_TASK_V3_CONTRACT = "project6_v4_learning_task_v3"
FEASIBILITY_PHASE = "feasibility_p3"
SOURCE_PHASES_V3 = tuple(SOURCE_PHASES) + (FEASIBILITY_PHASE,)
NOT_SEARCHED_CLASS = "not_searched"
UNLABELED_STATE = "unlabeled_not_searched"
SAFETY_TARGETS = ("pfv_safe", "tfv_noninferior", "peak_noninferior")
STATE_LEVEL_LABELS = (
    "intervention_feasible_found",
    "fallback_only_under_budget",
    "boundary_or_uncertain",
    "ood_or_unresolved",
)
CLASS_TO_STATE_LABEL = {
    "joint_feasible_found": "intervention_feasible_found",
    "joint_feasible_robust": "intervention_feasible_found",
    "joint_boundary_found": "boundary_or_uncertain",
    "no_joint_found_under_budget": "fallback_only_under_budget",
    "no_pfv_safe_found": "fallback_only_under_budget",
    "execution_unresolved": "ood_or_unresolved",
}
FALLBACK_ONLY_CLASSES = ("no_joint_found_under_budget", "no_pfv_safe_found")
CLASS_TO_FALLBACK_TARGET = {
    "joint_feasible_found": "intervention_candidate",
    "joint_feasible_robust": "intervention_candidate",
    "joint_boundary_found": "fallback_dynamic_internal_boundary_watch",
    "no_joint_found_under_budget": "fallback_dynamic_internal",
    "no_pfv_safe_found": "fallback_dynamic_internal",
    "execution_unresolved": "fallback_dynamic_internal",
    NOT_SEARCHED_CLASS: "fallback_dynamic_internal_not_searched",
}
FORBIDDEN_CLASS_TERMS = ("physically_infeasible", "impossible", "uncontrollable")
CONTRACT_STATE_COLUMNS = (
    "state_feasibility_class",
    "oracle_search_budget",
    "exact_joint_found",
    "online_joint_found",
    "candidate_generator_hit",
    "fallback_target",
    "feasibility_label_validity",
)
_STATE_KEYS = ["event_id", "checkpoint_id"]
_MAP_REQUIRED = [
    "event_id",
    "checkpoint_id",
    "split",
    "rainfall_phase",
    "checkpoint_min",
    "state_feasibility_class",
    "exact_joint_found",
    "online_joint_found",
    "candidate_generator_hit",
    "oracle_revealed",
    "feasibility_samples",
]


def _state_key(row) -> tuple[str, str]:
    return (str(row["event_id"]), str(row["checkpoint_id"]))


def build_pilot_dataset_v3(
    v2_samples: pd.DataFrame,
    feasibility_samples: pd.DataFrame,
    feasibility_map: pd.DataFrame,
) -> dict:
    """Merge frozen v2 rows with the feasibility-map rows into Dataset v3.

    Every row gains the seven contract state columns plus the state-level
    label; training/evaluation eligibility is recomputed under the v3
    rules (search rows train-only-if-pilot_train, oracle-revealed states
    removed from unseen evaluation).
    """
    _require(
        v2_samples,
        list(_STATE_KEYS)
        + ["sample_id", "split", "actual_schedule_sha256", "source_phase"],
        name="dataset v2 sample manifest",
    )
    _require(
        feasibility_samples,
        list(_STATE_KEYS)
        + [
            "sample_id",
            "split",
            "actual_schedule_sha256",
            "source_phase",
            "search_result_training_eligible",
        ],
        name="feasibility sample manifest",
    )
    _require(feasibility_map, _MAP_REQUIRED, name="feasibility map")
    v2_phase = v2_samples["source_phase"].astype(str)
    if int(v2_phase.eq(PRIMARY_PHASE).sum()) != 400:
        raise ValueError(
            "dataset v2 manifest must hold exactly 400 primary400 rows, "
            f"got {int(v2_phase.eq(PRIMARY_PHASE).sum())}"
        )
    if not v2_phase.isin(SOURCE_PHASES).all():
        raise ValueError(
            f"unexpected v2 source phases: {sorted(v2_phase.unique())}"
        )
    feas_phase = feasibility_samples["source_phase"].astype(str)
    if not feas_phase.eq(FEASIBILITY_PHASE).all():
        raise ValueError(
            "feasibility manifest must be uniformly source_phase="
            f"{FEASIBILITY_PHASE}, got {sorted(feas_phase.unique())}"
        )
    overlap = set(v2_samples["sample_id"].astype(str)) & set(
        feasibility_samples["sample_id"].astype(str)
    )
    if overlap:
        raise ValueError(
            f"sample_id overlap between v2 and feasibility: {sorted(overlap)[:5]}"
        )
    samples = pd.concat(
        [v2_samples, feasibility_samples], ignore_index=True, sort=False
    )
    if samples["sample_id"].astype(str).duplicated().any():
        raise ValueError("dataset v3 sample_id collision")

    map_by_state = {
        _state_key(row): row for _, row in feasibility_map.iterrows()
    }

    def _map_value(key, column, default):
        row = map_by_state.get(key)
        if row is None:
            return default
        return row[column]

    keys = [_state_key(row) for _, row in samples.iterrows()]
    classes = [
        str(_map_value(key, "state_feasibility_class", NOT_SEARCHED_CLASS))
        for key in keys
    ]
    samples["state_feasibility_class"] = classes
    samples["oracle_search_budget"] = [
        int(_map_value(key, "feasibility_samples", 0)) for key in keys
    ]
    samples["exact_joint_found"] = [
        bool(_map_value(key, "exact_joint_found", False)) for key in keys
    ]
    samples["online_joint_found"] = [
        bool(_map_value(key, "online_joint_found", False)) for key in keys
    ]
    samples["candidate_generator_hit"] = [
        bool(_map_value(key, "candidate_generator_hit", False)) for key in keys
    ]
    samples["fallback_target"] = [
        CLASS_TO_FALLBACK_TARGET.get(cls, "fallback_dynamic_internal")
        for cls in classes
    ]
    samples["feasibility_label_validity"] = [
        "not_searched"
        if cls == NOT_SEARCHED_CLASS
        else ("invalid_unresolved" if cls == "execution_unresolved" else "valid")
        for cls in classes
    ]
    samples["state_level_label"] = [
        CLASS_TO_STATE_LABEL.get(cls, UNLABELED_STATE) for cls in classes
    ]
    samples["oracle_revealed_state"] = [
        bool(_map_value(key, "oracle_revealed", False)) for key in keys
    ]

    split = samples["split"].astype(str)
    phase = samples["source_phase"].astype(str)
    is_feas = phase.eq(FEASIBILITY_PHASE)
    train_ok = pd.Series(True, index=samples.index)
    train_ok[is_feas] = (
        samples.loc[is_feas, "search_result_training_eligible"]
        .astype("boolean")
        .fillna(False)
        .astype(bool)
    )
    revealed = samples["oracle_revealed_state"].astype(bool)
    samples["eligible_for_training"] = split.eq(TRAIN_SPLIT) & train_ok
    samples["eligible_for_evaluation"] = (
        split.isin(EVAL_SPLITS) & phase.eq(PRIMARY_PHASE) & ~revealed
    )
    samples["oracle_revealed_evaluation_only"] = (
        split.isin(EVAL_SPLITS) & phase.eq(PRIMARY_PHASE) & revealed
    )

    state_rows = []
    for key, group in samples.groupby(_STATE_KEYS):
        first = group.iloc[0]
        map_row = map_by_state.get((str(key[0]), str(key[1])))
        state_rows.append(
            {
                "event_id": key[0],
                "checkpoint_id": key[1],
                "split": str(first["split"]),
                "rainfall_phase": (
                    str(map_row["rainfall_phase"]) if map_row is not None else ""
                ),
                "checkpoint_min": (
                    float(map_row["checkpoint_min"])
                    if map_row is not None
                    else float(first.get("checkpoint_min", np.nan))
                ),
                "state_feasibility_class": str(
                    first["state_feasibility_class"]
                ),
                "state_level_label": str(first["state_level_label"]),
                "oracle_search_budget": int(first["oracle_search_budget"]),
                "exact_joint_found": bool(first["exact_joint_found"]),
                "online_joint_found": bool(first["online_joint_found"]),
                "candidate_generator_hit": bool(
                    first["candidate_generator_hit"]
                ),
                "fallback_target": str(first["fallback_target"]),
                "feasibility_label_validity": str(
                    first["feasibility_label_validity"]
                ),
                "oracle_revealed_state": bool(first["oracle_revealed_state"]),
                "sample_rows": int(len(group)),
            }
        )
    state_manifest = pd.DataFrame(state_rows).sort_values(
        _STATE_KEYS
    ).reset_index(drop=True)

    split_manifest = (
        samples.groupby(["split", "source_phase"], as_index=False)
        .agg(
            samples=("sample_id", "size"),
            events=("event_id", "nunique"),
            states=("checkpoint_id", "nunique"),
            training_eligible=("eligible_for_training", "sum"),
            evaluation_eligible=("eligible_for_evaluation", "sum"),
        )
        .sort_values(["split", "source_phase"])
        .reset_index(drop=True)
    )
    source_manifest = (
        samples.groupby("source_phase", as_index=False)
        .agg(
            samples=("sample_id", "size"),
            events=("event_id", "nunique"),
            training_eligible=("eligible_for_training", "sum"),
        )
        .sort_values("source_phase")
        .reset_index(drop=True)
    )
    accounting = {
        "v2_rows": int(len(v2_samples)),
        "feasibility_rows": int(len(feasibility_samples)),
        "total_samples": int(len(samples)),
        "training_rows": int(samples["eligible_for_training"].sum()),
        "evaluation_rows": int(samples["eligible_for_evaluation"].sum()),
        "oracle_revealed_evaluation_rows": int(
            samples["oracle_revealed_evaluation_only"].sum()
        ),
        "states_total": int(len(state_manifest)),
        "searched_states": int(
            (state_manifest["state_feasibility_class"] != NOT_SEARCHED_CLASS)
            .sum()
        ),
        "accounting_closed": bool(
            len(samples) == len(v2_samples) + len(feasibility_samples)
        ),
    }
    return {
        "sample_manifest": samples,
        "state_manifest": state_manifest,
        "split_manifest": split_manifest,
        "source_manifest": source_manifest,
        "accounting": accounting,
    }


def audit_pilot_dataset_v3(
    samples: pd.DataFrame,
    *,
    expected_v2_counts: dict | None = None,
    expected_feasibility_rows: int = 653,
) -> dict:
    """Mechanical Dataset v3 audit (scientific gates live in Gate v3).

    Verifies frozen-source integrity, hard authenticity, the replay-aware
    actual-uniqueness policy and the v3 training / evaluation isolation
    rules from the LEARNING_TASK_V3 contract.
    """
    if expected_v2_counts is None:
        expected_v2_counts = {
            "primary400": 400,
            "joint_extension": 60,
            "flat_auxiliary": 19,
        }
    required = set(_STATE_KEYS) | {
        "sample_id",
        "split",
        "source_phase",
        "actual_schedule_sha256",
        "eligible_for_training",
        "eligible_for_evaluation",
        "oracle_revealed_state",
        "state_level_label",
        "joint_noninferior",
        "confirmed_flat",
        "rainfall_sha256",
        *CONTRACT_STATE_COLUMNS,
    }
    missing = sorted(required - set(samples))
    if missing:
        return {"status": "blocked", "missing_columns": missing}
    hard_missing = sorted(set(HARD_AUTHENTICITY_COLUMNS) - set(samples))
    if hard_missing:
        return {"status": "blocked", "missing_columns": hard_missing}
    phase = samples["source_phase"].astype(str)
    split = samples["split"].astype(str)
    v2 = samples[phase.isin(SOURCE_PHASES)]
    feas = samples[phase.eq(FEASIBILITY_PHASE)]
    replay = (
        feas["expected_replay_of"].notna()
        if "expected_replay_of" in feas
        else pd.Series(False, index=feas.index)
    )
    v2_state_actual = {
        key: set(group["actual_schedule_sha256"].astype(str))
        for key, group in v2.groupby(_STATE_KEYS)
    }
    nonreplay = feas[~replay]
    nonreplay_collisions = int(
        sum(
            str(row["actual_schedule_sha256"])
            in v2_state_actual.get(_state_key(row), set())
            for _, row in nonreplay.iterrows()
        )
    )
    dedupe_pool = pd.concat([v2, nonreplay], ignore_index=True, sort=False)
    classes = samples["state_feasibility_class"].astype(str)
    searched = samples[classes != NOT_SEARCHED_CLASS]
    labels = samples["state_level_label"].astype(str)
    revealed = samples["oracle_revealed_state"].astype(bool)
    eval_rows = samples[samples["eligible_for_evaluation"].astype(bool)]
    train_rows = samples[samples["eligible_for_training"].astype(bool)]
    hard_matrix = samples[list(HARD_AUTHENTICITY_COLUMNS)].fillna(False)
    flat = samples[samples["confirmed_flat"].astype(bool)]

    def _feasible_states(frame: pd.DataFrame) -> int:
        if frame.empty:
            return 0
        return int(
            frame.groupby(_STATE_KEYS)["joint_noninferior"]
            .any()
            .sum()
        )

    locked_validation_feasible = _feasible_states(
        eval_rows[eval_rows["split"].astype(str).eq("pilot_validation")]
    )
    evaluation_feasible = _feasible_states(eval_rows)
    fallback_only_states = samples[
        classes.isin(FALLBACK_ONLY_CLASSES)
    ].drop_duplicates(_STATE_KEYS)
    checks = {
        # -- frozen source integrity -----------------------------------
        "primary400_intact_400": int(phase.eq(PRIMARY_PHASE).sum())
        == expected_v2_counts["primary400"],
        "joint_extension_intact": int(phase.eq("joint_extension").sum())
        == expected_v2_counts["joint_extension"],
        "flat_auxiliary_intact": int(phase.eq("flat_auxiliary").sum())
        == expected_v2_counts["flat_auxiliary"],
        "feasibility_rows_expected": int(len(feas))
        == int(expected_feasibility_rows),
        "sample_id_unique": not samples["sample_id"]
        .astype(str)
        .duplicated()
        .any(),
        "source_phase_values_valid": phase.isin(SOURCE_PHASES_V3).all(),
        # -- hard execution authenticity --------------------------------
        "hard_authenticity_100pct": bool(
            hard_matrix.astype(bool).all().all()
        ),
        "nonreplay_actual_disjoint_from_v2": nonreplay_collisions == 0,
        "actual_duplicates_0": not dedupe_pool.duplicated(
            _STATE_KEYS + ["actual_schedule_sha256"]
        ).any(),
        # -- training isolation ------------------------------------------
        "training_only_pilot_train": bool(
            train_rows["split"].astype(str).eq(TRAIN_SPLIT).all()
            if len(train_rows)
            else False
        ),
        "no_eval_split_search_rows_trainable": not bool(
            (
                feas["eligible_for_training"].astype(bool)
                & ~feas["split"].astype(str).eq(TRAIN_SPLIT)
            ).any()
        ),
        "feasibility_training_rows_search_eligible": bool(
            feas.loc[
                feas["eligible_for_training"].astype(bool),
                "search_result_training_eligible",
            ]
            .astype("boolean")
            .fillna(False)
            .astype(bool)
            .all()
            if "search_result_training_eligible" in feas
            else False
        ),
        # -- evaluation isolation -----------------------------------------
        "evaluation_primary400_only": bool(
            eval_rows["source_phase"].astype(str).eq(PRIMARY_PHASE).all()
            if len(eval_rows)
            else True
        ),
        "evaluation_excludes_oracle_revealed": not bool(
            (eval_rows["oracle_revealed_state"].astype(bool)).any()
        ),
        # -- frozen classification vocabulary ------------------------------
        "state_class_vocabulary_valid": searched[
            "state_feasibility_class"
        ]
        .astype(str)
        .isin(set(CLASS_TO_STATE_LABEL))
        .all()
        if len(searched)
        else True,
        "no_forbidden_class_terms": not classes.isin(
            FORBIDDEN_CLASS_TERMS
        ).any(),
        "state_level_labels_valid": labels[
            classes != NOT_SEARCHED_CLASS
        ]
        .isin(STATE_LEVEL_LABELS)
        .all()
        if len(searched)
        else True,
        "execution_unresolved_zero": not classes.eq(
            "execution_unresolved"
        ).any(),
        "contract_state_columns_present": all(
            column in samples for column in CONTRACT_STATE_COLUMNS
        ),
        # -- flat policy (report-only elsewhere; count preserved here) ----
        "confirmed_flat_14_preserved": int(len(flat)) == 14,
        "rainfall_sha_split_isolated": not samples.groupby(
            "rainfall_sha256"
        )["split"]
        .nunique()
        .gt(1)
        .any(),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    state_label_by_split = {
        str(split_name): {
            str(k): int(v)
            for k, v in group.drop_duplicates(_STATE_KEYS)[
                "state_level_label"
            ]
            .value_counts()
            .items()
        }
        for split_name, group in samples[
            classes != NOT_SEARCHED_CLASS
        ].groupby("split")
    }
    return {
        "status": "pass" if all(checks.values()) else "scientific_fail",
        "gate_version": "v3",
        "contract": LEARNING_TASK_V3_CONTRACT,
        "checks": checks,
        "headline": {
            "total_samples": int(len(samples)),
            "samples_by_source_phase": {
                str(k): int(v) for k, v in phase.value_counts().items()
            },
            "training_rows": int(len(train_rows)),
            "evaluation_rows": int(len(eval_rows)),
            "oracle_revealed_states": int(
                samples[revealed].drop_duplicates(_STATE_KEYS).shape[0]
            ),
            "oracle_revealed_evaluation_rows_excluded": int(
                samples.get(
                    "oracle_revealed_evaluation_only",
                    pd.Series(False, index=samples.index),
                )
                .astype(bool)
                .sum()
            ),
            "state_feasibility_class_counts": {
                str(k): int(v)
                for k, v in searched.drop_duplicates(_STATE_KEYS)[
                    "state_feasibility_class"
                ]
                .value_counts()
                .items()
            },
            "state_level_label_distribution_by_split": state_label_by_split,
            "locked_validation_feasible_states_unrevealed": int(
                locked_validation_feasible
            ),
            "evaluation_feasible_states_unrevealed": int(
                evaluation_feasible
            ),
            "fallback_only_states": int(len(fallback_only_states)),
            "fallback_only_event_support": int(
                fallback_only_states["event_id"].nunique()
            ),
            "confirmed_flat_count": int(len(flat)),
            "confirmed_flat_event_support": int(
                flat["event_id"].nunique()
            ),
            "flat_head_enabled": bool(flat["event_id"].nunique() >= 3),
            "replay_rows": int(replay.sum()),
        },
        "notes": [
            "flat fraction is not a gating quantity under Gate v3; the flat "
            "head stays disabled while flat event support < 3",
            "oracle-revealed evaluation rows are excluded from unseen "
            "evaluation and only reported as diagnostics",
        ],
    }


# --------------------------------------------------------------------------
# Baselines v3
# --------------------------------------------------------------------------


def _composed_scale(train_tfv: pd.Series) -> float:
    values = pd.to_numeric(train_tfv, errors="coerce").dropna()
    scale = float(values.std()) if len(values) > 1 else 0.0
    return scale if scale > 0.0 else 1.0


def train_pilot_baselines_v3(
    samples: pd.DataFrame,
    state_manifest: pd.DataFrame,
    *,
    seed: int = 20260728,
    n_boot: int = 200,
    tfv_margin: float = 0.0,
) -> dict:
    """Fit the two-tier v3 baselines on Dataset v3.

    Candidate tier: zero/ridge regression on the three delta targets plus
    majority/logistic/HGB safety heads; joint noninferiority is composed
    from ``pfv_safe`` and ``peak_noninferior`` heads and the ridge TFV
    delta (frozen margin), never from a standalone joint classifier.
    State tier: majority + logistic on online-available descriptors only
    (checkpoint time, rainfall phase); resubstitution diagnostics because
    every searched non-train state is oracle-revealed.
    """
    from sklearn.metrics import balanced_accuracy_score

    from .training import build_baseline_models

    _require(
        samples,
        [
            "eligible_for_training",
            "eligible_for_evaluation",
            "oracle_revealed_evaluation_only",
            "split",
            "source_phase",
            "joint_noninferior",
        ]
        + list(DELTA_TARGETS)
        + list(SAFETY_TARGETS),
        name="dataset v3 samples",
    )
    _require(
        state_manifest,
        [
            "split",
            "checkpoint_min",
            "rainfall_phase",
            "state_level_label",
            "feasibility_label_validity",
            "oracle_revealed_state",
        ],
        name="state manifest v3",
    )
    train = samples[samples["eligible_for_training"].astype(bool)]
    if train.empty:
        raise ValueError("no eligible training rows")
    evals = {
        split: samples[
            samples["eligible_for_evaluation"].astype(bool)
            & samples["split"].astype(str).eq(split)
        ]
        for split in EVAL_SPLITS
    }
    diagnostics = {
        split: samples[
            samples["oracle_revealed_evaluation_only"].astype(bool)
            & samples["split"].astype(str).eq(split)
        ]
        for split in EVAL_SPLITS
    }
    x_train, family_categories = candidate_features(train)
    x_eval = {
        split: candidate_features(frame, family_categories=family_categories)[0]
        for split, frame in evals.items()
        if len(frame)
    }
    x_diag = {
        split: candidate_features(frame, family_categories=family_categories)[0]
        for split, frame in diagnostics.items()
        if len(frame)
    }
    report: dict = {
        "contract": LEARNING_TASK_V3_CONTRACT,
        "seed": int(seed),
        "n_boot": int(n_boot),
        "tfv_margin": float(tfv_margin),
        "feature_columns": list(x_train.columns),
        "split_policy": {
            "train_rows": int(len(train)),
            "train_source_phases": {
                str(k): int(v)
                for k, v in train["source_phase"].value_counts().items()
            },
            "train_only_pilot_train": bool(
                train["split"].astype(str).eq(TRAIN_SPLIT).all()
            ),
            "eval_rows": {
                split: int(len(frame)) for split, frame in evals.items()
            },
            "oracle_revealed_diagnostic_rows": {
                split: int(len(frame))
                for split, frame in diagnostics.items()
            },
            "eval_primary400_only": bool(
                all(
                    frame["source_phase"].eq(PRIMARY_PHASE).all()
                    for frame in evals.values()
                    if len(frame)
                )
            ),
            "eval_excludes_oracle_revealed": True,
            "no_calibration_fitting": True,
            "no_validation_tuning": True,
            "no_challenge_selection": True,
            "no_exact_search_features": True,
        },
        "regression": {},
        "classification": {},
        "composed_joint": {},
        "ranking": {},
        "ranking_oracle_revealed_diagnostic": {},
        "state_level": {},
        "models": {},
    }

    # ---- regression: zero predictor + ridge per delta target ----------
    ridge_tfv_model = None
    for target in DELTA_TARGETS:
        y_train = pd.to_numeric(train[target], errors="coerce")
        mask = y_train.notna()
        entry: dict = {}
        for name in ("zero_predictor", "ridge"):
            model = build_baseline_models()[name]
            model.fit(x_train[mask.values], y_train[mask.values])
            if name == "ridge" and target == PRIMARY_REGRESSION_TARGET:
                ridge_tfv_model = model
            per_split: dict = {}
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
                if (
                    split == "pilot_validation"
                    and target == PRIMARY_REGRESSION_TARGET
                ):
                    per_split[split]["rmse_bootstrap"] = _bootstrap_interval(
                        y_true[keep.values].to_numpy(),
                        pred,
                        lambda t, p: float(np.sqrt(np.mean((p - t) ** 2))),
                        n_boot=n_boot,
                        seed=seed,
                    )
            diag_split: dict = {}
            for split, frame in diagnostics.items():
                if not len(frame):
                    continue
                y_true = pd.to_numeric(frame[target], errors="coerce")
                keep = y_true.notna()
                if not keep.any():
                    continue
                diag_split[split] = _regression_metrics(
                    y_true[keep.values],
                    model.predict(x_diag[split][keep.values]),
                )
            entry[name] = {
                "splits": per_split,
                "oracle_revealed_diagnostic": diag_split,
            }
        report["regression"][target] = entry

    # ---- safety classification heads ------------------------------------
    class_models = (
        "majority_classifier",
        "logistic_regression",
        "hist_gradient_boosting",
    )
    fitted: dict[tuple[str, str], object] = {}
    for target in SAFETY_TARGETS:
        y_train = train[target].astype(bool)
        entry = {}
        degenerate = y_train.nunique() < 2
        for name in class_models:
            if degenerate:
                entry[name] = {"trained": False, "reason": "single_class"}
                continue
            model = build_baseline_models()[name]
            model.fit(x_train, y_train)
            fitted[(target, name)] = model
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
            entry[name] = {"trained": True, "splits": per_split}
        report["classification"][target] = entry

    # ---- composed joint noninferiority -----------------------------------
    def _proba(model, features) -> np.ndarray | None:
        if not hasattr(model, "predict_proba"):
            return None
        proba = model.predict_proba(features)
        return proba[:, 1] if proba.shape[1] == 2 else None

    scale = _composed_scale(train[PRIMARY_REGRESSION_TARGET])
    for name in ("logistic_regression", "hist_gradient_boosting"):
        pfv_model = fitted.get(("pfv_safe", name))
        peak_model = fitted.get(("peak_noninferior", name))
        if pfv_model is None or peak_model is None or ridge_tfv_model is None:
            report["composed_joint"][name] = {
                "trained": False,
                "reason": "missing_component_head",
            }
            continue
        per_split = {}
        diag_split = {}
        for split_map, feature_map, sink in (
            (evals, x_eval, per_split),
            (diagnostics, x_diag, diag_split),
        ):
            for split, frame in split_map.items():
                if not len(frame):
                    continue
                features = feature_map[split]
                tfv_pred = ridge_tfv_model.predict(features)
                pfv_pred = pfv_model.predict(features).astype(bool)
                peak_pred = peak_model.predict(features).astype(bool)
                hard = pfv_pred & peak_pred & (tfv_pred <= float(tfv_margin))
                p_pfv = _proba(pfv_model, features)
                p_peak = _proba(peak_model, features)
                soft = None
                if p_pfv is not None and p_peak is not None:
                    soft = (
                        p_pfv
                        * p_peak
                        * (1.0 / (1.0 + np.exp(tfv_pred / scale)))
                    )
                sink[split] = _classification_metrics(
                    frame["joint_noninferior"].astype(bool), hard, soft
                )
        report["composed_joint"][name] = {
            "trained": True,
            "policy": (
                "pfv_safe_head AND peak_noninferior_head AND "
                "ridge_tfv_delta <= frozen margin; no standalone joint "
                "binary classifier"
            ),
            "splits": per_split,
            "oracle_revealed_diagnostic": diag_split,
        }

    # ---- ranking: composed soft score -------------------------------------
    hgb_entry = report["composed_joint"].get("hist_gradient_boosting", {})
    if hgb_entry.get("trained"):
        pfv_model = fitted[("pfv_safe", "hist_gradient_boosting")]
        peak_model = fitted[("peak_noninferior", "hist_gradient_boosting")]
        for split_map, feature_map, sink_key in (
            (evals, x_eval, "ranking"),
            (diagnostics, x_diag, "ranking_oracle_revealed_diagnostic"),
        ):
            for split, frame in split_map.items():
                if not len(frame):
                    continue
                features = feature_map[split]
                p_pfv = _proba(pfv_model, features)
                p_peak = _proba(peak_model, features)
                if p_pfv is None or p_peak is None:
                    continue
                tfv_pred = ridge_tfv_model.predict(features)
                scores = (
                    p_pfv
                    * p_peak
                    * (1.0 / (1.0 + np.exp(tfv_pred / scale)))
                )
                report[sink_key][split] = _topk_and_regret(frame, scores)

    # ---- state-level head ---------------------------------------------------
    valid_states = state_manifest[
        state_manifest["feasibility_label_validity"].astype(str).eq("valid")
    ]
    train_states = valid_states[
        valid_states["split"].astype(str).eq(TRAIN_SPLIT)
    ]
    unseen_states = valid_states[
        ~valid_states["split"].astype(str).eq(TRAIN_SPLIT)
        & ~valid_states["oracle_revealed_state"].astype(bool)
    ]
    state_entry: dict = {
        "labeled_states": int(len(valid_states)),
        "train_states": int(len(train_states)),
        "label_distribution": {
            str(k): int(v)
            for k, v in valid_states["state_level_label"]
            .value_counts()
            .items()
        },
        "train_label_distribution": {
            str(k): int(v)
            for k, v in train_states["state_level_label"]
            .value_counts()
            .items()
        },
        "allowed_inputs": [
            "checkpoint_min",
            "rainfall_phase (one-hot)",
        ],
        "forbidden_inputs_excluded": [
            "exact search outcomes",
            "search budgets and per-state search counts",
            "future SWMM results",
        ],
    }
    if len(train_states) >= 4 and train_states[
        "state_level_label"
    ].nunique() > 1:
        phases = sorted(
            valid_states["rainfall_phase"].astype(str).unique()
        )

        def _state_features(frame: pd.DataFrame) -> pd.DataFrame:
            out = pd.DataFrame(
                {
                    "checkpoint_min": pd.to_numeric(
                        frame["checkpoint_min"], errors="coerce"
                    ).fillna(0.0)
                },
                index=frame.index,
            )
            for phase_name in phases:
                out[f"phase__{phase_name}"] = (
                    frame["rainfall_phase"].astype(str).eq(phase_name)
                ).astype(float)
            return out

        y_state = train_states["state_level_label"].astype(str)
        majority_label = y_state.mode().iloc[0]
        logistic = build_baseline_models()["logistic_regression"]
        logistic.fit(_state_features(train_states), y_state)
        resub_pred = logistic.predict(_state_features(train_states))
        state_entry["majority_label"] = str(majority_label)
        state_entry["resubstitution_diagnostic"] = {
            "majority_accuracy": float(
                y_state.eq(majority_label).mean()
            ),
            "logistic_balanced_accuracy": float(
                balanced_accuracy_score(y_state, resub_pred)
            ),
            "diagnostic_only": True,
        }
        if len(unseen_states) and unseen_states[
            "state_level_label"
        ].nunique() >= 1:
            unseen_pred = logistic.predict(_state_features(unseen_states))
            state_entry["unseen_state_evaluation"] = {
                "states": int(len(unseen_states)),
                "accuracy": float(
                    (
                        unseen_states["state_level_label"].astype(str)
                        == unseen_pred
                    ).mean()
                ),
            }
        else:
            state_entry["unseen_state_evaluation"] = None
            state_entry["unseen_state_evaluation_reason"] = (
                "every searched non-train state is oracle_revealed; no "
                "unseen labeled states exist under the frozen P3 evidence"
            )
    else:
        state_entry["trained"] = False
        state_entry["reason"] = "insufficient_or_degenerate_train_states"
    report["state_level"] = state_entry

    # ---- model summary for the gate -----------------------------------------
    def _split_metric(block, split, key):
        return block.get("splits", {}).get(split, {}).get(key)

    zero_block = report["regression"][PRIMARY_REGRESSION_TARGET][
        "zero_predictor"
    ]
    ridge_block = report["regression"][PRIMARY_REGRESSION_TARGET]["ridge"]
    zero_rmse = _split_metric(zero_block, "pilot_validation", "rmse")
    ridge_rmse = _split_metric(ridge_block, "pilot_validation", "rmse")
    improvement = (
        (zero_rmse - ridge_rmse) / zero_rmse
        if zero_rmse not in (None, 0) and ridge_rmse is not None
        else None
    )
    composed = report["composed_joint"].get("hist_gradient_boosting", {})
    composed_validation = (
        composed.get("splits", {}).get("pilot_validation", {})
        if composed.get("trained")
        else {}
    )
    report["models"] = {
        "zero_predictor": {"validation_rmse": zero_rmse},
        "ridge": {
            "validation_rmse": ridge_rmse,
            "validation_rmse_improvement_vs_zero": improvement,
        },
        "composed_joint_hgb": {
            "validation_balanced_accuracy": composed_validation.get(
                "balanced_accuracy"
            ),
            "validation_mcc": composed_validation.get("mcc"),
            "validation_auprc": composed_validation.get("auprc"),
            "validation_false_safe_rate": composed_validation.get(
                "false_safe_rate"
            ),
            "validation_positives": composed_validation.get("positives"),
            "validation_n": composed_validation.get("n"),
        },
        "top5_feasible_recall_validation": report["ranking"]
        .get("pilot_validation", {})
        .get("top5_feasible_recall"),
    }
    return _finite_or_none(report)


def evaluate_pilot_gate_v3(
    dataset_audit: dict,
    baseline_report: dict,
    feasibility_audit: dict,
) -> dict:
    """Gate v3 verdict under the LEARNING_TASK_V3 contract.

    Verdicts: ``pass`` / ``underpowered_validation`` / ``scientific_fail``.
    ``train1600_authorized`` is true only on a full pass and PlanTrain1600
    is never run automatically in any case.
    """
    dataset_checks = dataset_audit.get("checks", {})
    headline = dataset_audit.get("headline", {})
    recall = feasibility_audit.get("recall_report", {})
    p3_gate = feasibility_audit.get("p3_gate", {})
    models = baseline_report.get("models", {})
    composed = models.get("composed_joint_hgb", {})
    ridge_improvement = models.get("ridge", {}).get(
        "validation_rmse_improvement_vs_zero"
    )
    positives = composed.get("validation_positives")
    n_val = composed.get("validation_n")
    prevalence = (
        float(positives) / float(n_val)
        if positives is not None and n_val
        else None
    )
    auprc = composed.get("validation_auprc")
    false_safe = composed.get("validation_false_safe_rate")
    top5 = models.get("top5_feasible_recall_validation")
    locked_validation_feasible = int(
        headline.get("locked_validation_feasible_states_unrevealed", 0)
    )
    checks = {
        "dataset_audit_v3_pass": dataset_audit.get("status") == "pass",
        "hard_authenticity_100pct": bool(
            dataset_checks.get("hard_authenticity_100pct", False)
        ),
        "exact_joint_feasible_states_at_least_8": int(
            recall.get("exact_joint_feasible_states", 0)
        )
        >= 8,
        "joint_event_support_at_least_3": int(
            recall.get("event_support", 0)
        )
        >= 3,
        "candidate_generator_recall_at_least_0p80": float(
            recall.get("candidate_generator_state_recall", 0.0) or 0.0
        )
        >= 0.80,
        "unresolved_states_zero": int(
            p3_gate.get("unresolved_states", 1)
        )
        == 0,
        "positive_control_replay_100pct": float(
            feasibility_audit.get("replay_success_rate", 0.0) or 0.0
        )
        == 1.0,
        "ridge_rmse_improvement_at_least_10pct": bool(
            ridge_improvement is not None and ridge_improvement >= 0.10
        ),
        "joint_safety_balanced_accuracy_at_least_0p60": bool(
            composed.get("validation_balanced_accuracy") is not None
            and composed["validation_balanced_accuracy"] >= 0.60
        ),
        "joint_safety_mcc_positive": bool(
            composed.get("validation_mcc") is not None
            and composed["validation_mcc"] > 0.0
        ),
        "joint_safety_auprc_above_prevalence": bool(
            auprc is not None
            and prevalence is not None
            and auprc > prevalence
        ),
        "false_safe_at_most_0p20": bool(
            false_safe is not None and false_safe <= 0.20
        ),
        "top5_feasible_recall_at_least_0p80": bool(
            top5 is not None and top5 >= 0.80
        ),
        "feasible_states_in_evaluation_at_least_8": int(
            headline.get("evaluation_feasible_states_unrevealed", 0)
        )
        >= 8,
        "fallback_only_cross_event_support": int(
            headline.get("fallback_only_event_support", 0)
        )
        >= 3,
    }
    checks = {key: bool(value) for key, value in checks.items()}
    underpowered = locked_validation_feasible < 5
    scientific_pass = all(checks.values()) and not underpowered
    if underpowered:
        status = "underpowered_validation"
    elif scientific_pass:
        status = "pass"
    else:
        status = "scientific_fail"
    return {
        "status": status,
        "gate_version": "v3",
        "contract": LEARNING_TASK_V3_CONTRACT,
        "scientific_pass": bool(scientific_pass),
        "exit_code": 0 if scientific_pass else 5,
        "checks": checks,
        "underpowered_validation": {
            "triggered": bool(underpowered),
            "locked_validation_feasible_states": locked_validation_feasible,
            "threshold": 5,
            "consequence": (
                "train1600 not authorized; add independent unseen "
                "development validation events from the standard pool; "
                "never reuse oracle-searched states"
            )
            if underpowered
            else None,
        },
        "metrics_used": _finite_or_none(
            {
                "exact_joint_feasible_states": recall.get(
                    "exact_joint_feasible_states"
                ),
                "online_generator_joint_states": recall.get(
                    "online_generator_joint_states"
                ),
                "candidate_generator_state_recall": recall.get(
                    "candidate_generator_state_recall"
                ),
                "joint_event_support": recall.get("event_support"),
                "replay_success_rate": feasibility_audit.get(
                    "replay_success_rate"
                ),
                "ridge_rmse_improvement_vs_zero": ridge_improvement,
                "composed_joint_validation": composed,
                "composed_joint_positive_prevalence": prevalence,
                "top5_feasible_recall_validation": top5,
                "evaluation_feasible_states_unrevealed": headline.get(
                    "evaluation_feasible_states_unrevealed"
                ),
                "fallback_only_event_support": headline.get(
                    "fallback_only_event_support"
                ),
            }
        ),
        "dataset_headline": headline,
        "flat_fraction_core_blocking_gate": False,
        "train1600_authorized": bool(scientific_pass),
        "train1600_authorization": (
            "authorized_manual_start_only"
            if scientific_pass
            else "not_authorized"
        ),
        "auto_entry_into_train1600_prohibited": True,
    }
