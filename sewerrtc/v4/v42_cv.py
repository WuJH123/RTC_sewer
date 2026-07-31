"""V4.2 nested event-grouped cross-validation planner.

Implements a two-level nested CV scheme where:
  - Outer CV (5-fold, GroupKFold): held-out evaluation; each event appears in
    exactly one test fold.  Groups are defined by ``rainfall_sha256`` so that
    all candidates / checkpoints / references sharing the same rainfall family
    stay together.
  - Inner CV (3-fold, GroupKFold): hyper-parameter selection within the
    current outer training set.

All randomness is captured up-front as frozen seeds — no runtime ``random``
calls are permitted.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_OUTER_SPLITS = 5
DEFAULT_INNER_SPLITS = 3
DEFAULT_SEED = 42
MIN_EVENTS_PER_FOLD = 3

# Output sub-directory (relative to output_root)
NESTED_CV_DIR = "models/v42_nested_cv"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class V42CVPlan:
    """Immutable description of a nested CV layout."""

    outer_n_splits: int
    inner_n_splits: int
    n_repeats: int
    group_column: str
    frozen_seeds: tuple[int, ...]
    outer_fold_assignment: dict[str, int]  # event_id -> fold_idx


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_event_ledger(ledger_path: Path) -> pd.DataFrame:
    """Read the event usage ledger CSV."""
    if not ledger_path.exists():
        raise FileNotFoundError(f"event_usage_ledger.csv not found at {ledger_path}")
    df = pd.read_csv(ledger_path)
    required = {"event_id", "rainfall_sha256"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"ledger missing required columns: {missing}")
    return df


def _build_event_table(
    event_ids: list[str] | np.ndarray,
    rainfall_shas: list[str] | np.ndarray,
) -> pd.DataFrame:
    """Build a de-duplicated event table from parallel arrays."""
    df = pd.DataFrame({
        "event_id": np.asarray(event_ids, dtype=str),
        "rainfall_sha256": np.asarray(rainfall_shas, dtype=str),
    })
    # One row per unique event_id (should already be unique).
    df = df.drop_duplicates(subset="event_id", keep="first").reset_index(drop=True)
    return df


def _check_one_to_one_mapping(event_df: pd.DataFrame) -> bool:
    """Return True when event_id <-> rainfall_sha256 is strictly 1:1."""
    sha_per_event = event_df.groupby("event_id")["rainfall_sha256"].nunique()
    event_per_sha = event_df.groupby("rainfall_sha256")["event_id"].nunique()
    return bool((sha_per_event <= 1).all() and (event_per_sha <= 1).all())


def _freeze_seeds(base_seed: int, n: int) -> tuple[int, ...]:
    """Generate *n* deterministic seeds from *base_seed*."""
    rng = np.random.RandomState(base_seed)
    return tuple(int(rng.randint(0, 2**31 - 1)) for _ in range(n))


# ---------------------------------------------------------------------------
# Core planner
# ---------------------------------------------------------------------------

def plan_v42_nested_grouped_cv(
    event_ids: list[str] | np.ndarray,
    rainfall_shas: list[str] | np.ndarray,
    *,
    n_outer: int = DEFAULT_OUTER_SPLITS,
    n_inner: int = DEFAULT_INNER_SPLITS,
    seed: int = DEFAULT_SEED,
) -> V42CVPlan:
    """Compute a deterministic nested grouped CV plan.

    Parameters
    ----------
    event_ids : array-like of str
        One entry per event (unique event identifiers).
    rainfall_shas : array-like of str
        Parallel array — the ``rainfall_sha256`` for each event.
    n_outer : int
        Number of outer folds (default 5).
    n_inner : int
        Number of inner folds (default 3).
    seed : int
        Master seed used to freeze all downstream randomness.

    Returns
    -------
    V42CVPlan
        Frozen plan with outer fold assignments.
    """
    event_df = _build_event_table(event_ids, rainfall_shas)
    unique_events = event_df["event_id"].values
    groups = event_df["rainfall_sha256"].values

    is_one_to_one = _check_one_to_one_mapping(event_df)
    log.info(
        "Planning nested CV: %d events, %d unique rainfall SHAs, "
        "event<->SHA 1:1 = %s",
        len(unique_events),
        event_df["rainfall_sha256"].nunique(),
        is_one_to_one,
    )

    # --- Outer split (GroupKFold on rainfall_sha256) ---
    outer_gkf = GroupKFold(n_splits=n_outer)
    pseudo_y = np.zeros(len(event_df))  # dummy — GroupKFold ignores y
    outer_assignment: dict[str, int] = {}

    for fold_idx, (_train_idx, test_idx) in enumerate(
        outer_gkf.split(np.arange(len(event_df)), pseudo_y, groups)
    ):
        for idx in test_idx:
            eid = str(unique_events[idx])
            if eid in outer_assignment:
                raise RuntimeError(
                    f"event {eid} assigned to multiple outer folds"
                )
            outer_assignment[eid] = int(fold_idx)

    # Every event must be assigned exactly once.
    unassigned = set(unique_events) - set(outer_assignment.keys())
    if unassigned:
        raise RuntimeError(f"{len(unassigned)} events not assigned to any fold")

    frozen_seeds = _freeze_seeds(seed, n_outer * n_inner)

    plan = V42CVPlan(
        outer_n_splits=n_outer,
        inner_n_splits=n_inner,
        n_repeats=1,
        group_column="rainfall_sha256",
        frozen_seeds=frozen_seeds,
        outer_fold_assignment=outer_assignment,
    )
    log.info("Nested CV plan created with %d outer × %d inner folds", n_outer, n_inner)
    return plan


# ---------------------------------------------------------------------------
# Inner-fold helper
# ---------------------------------------------------------------------------

def compute_inner_folds(
    event_df: pd.DataFrame,
    plan: V42CVPlan,
    outer_fold: int,
) -> pd.DataFrame:
    """Compute inner 3-fold split for the training portion of *outer_fold*.

    Returns a DataFrame with columns ``[event_id, outer_fold, inner_fold]``.
    """
    outer_train_mask = np.array([
        plan.outer_fold_assignment.get(eid, -1) != outer_fold
        for eid in event_df["event_id"].values
    ])
    train_df = event_df.loc[outer_train_mask].reset_index(drop=True)

    if len(train_df) == 0:
        return pd.DataFrame(columns=["event_id", "outer_fold", "inner_fold"])

    inner_gkf = GroupKFold(n_splits=plan.inner_n_splits)
    pseudo_y = np.zeros(len(train_df))
    inner_groups = train_df["rainfall_sha256"].values

    inner_fold_col = np.full(len(train_df), -1, dtype=int)
    for inner_idx, (_tr, te) in enumerate(
        inner_gkf.split(np.arange(len(train_df)), pseudo_y, inner_groups)
    ):
        inner_fold_col[te] = inner_idx

    result = train_df[["event_id"]].copy()
    result["outer_fold"] = outer_fold
    result["inner_fold"] = inner_fold_col
    return result


# ---------------------------------------------------------------------------
# Artifact writers
# ---------------------------------------------------------------------------

def write_nested_cv_artifacts(
    plan: V42CVPlan,
    event_df: pd.DataFrame,
    output_root: Path,
    *,
    class_columns: tuple[str, ...] | None = None,
) -> dict[str, Path]:
    """Persist all CV plan artifacts under *output_root*.

    Parameters
    ----------
    plan : V42CVPlan
    event_df : pd.DataFrame
        Must contain ``event_id``, ``rainfall_sha256``.  May optionally
        contain classification columns (``pfv_safe``, ``tfv_improved``, …).
    output_root : Path
    class_columns : tuple of str, optional
        Classification target column names for class-support reporting.

    Returns dict mapping artifact name → path.
    """
    cv_dir = output_root / NESTED_CV_DIR
    cv_dir.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}

    # --- outer_fold_assignment.csv ---
    outer_df = event_df[["event_id", "rainfall_sha256"]].copy()
    outer_df["outer_fold"] = outer_df["event_id"].map(plan.outer_fold_assignment)
    p = cv_dir / "outer_fold_assignment.csv"
    outer_df.to_csv(p, index=False)
    paths["outer_fold_assignment"] = p

    # --- inner_fold_assignment.csv ---
    inner_rows: list[pd.DataFrame] = []
    for fold_idx in range(plan.outer_n_splits):
        inner_rows.append(compute_inner_folds(event_df, plan, fold_idx))
    inner_df = pd.concat(inner_rows, ignore_index=True) if inner_rows else pd.DataFrame(
        columns=["event_id", "outer_fold", "inner_fold"]
    )
    p = cv_dir / "inner_fold_assignment.csv"
    inner_df.to_csv(p, index=False)
    paths["inner_fold_assignment"] = p

    # --- group_balance_report.csv ---
    balance_rows: list[dict[str, Any]] = []
    for fold_idx in range(plan.outer_n_splits):
        test_mask = outer_df["outer_fold"] == fold_idx
        test_events = set(outer_df.loc[test_mask, "event_id"])
        n_events = int(test_mask.sum())
        # Count samples per fold (using event_df rows whose event_id is in test).
        sample_mask = event_df["event_id"].isin(test_events)
        n_samples = int(sample_mask.sum())
        rain_families = sorted(
            event_df.loc[sample_mask, "rainfall_sha256"].unique().tolist()
        )
        balance_rows.append({
            "fold": fold_idx,
            "n_events": n_events,
            "n_samples": n_samples,
            "rainfall_families": ";".join(rain_families),
            "n_rainfall_families": len(rain_families),
        })
    p = cv_dir / "group_balance_report.csv"
    pd.DataFrame(balance_rows).to_csv(p, index=False)
    paths["group_balance_report"] = p

    # --- class_support_by_fold.csv ---
    if class_columns is not None:
        present_cols = [c for c in class_columns if c in event_df.columns]
        if present_cols:
            class_rows: list[dict[str, Any]] = {"fold": []}
            for col in present_cols:
                class_rows[col] = []
            for fold_idx in range(plan.outer_n_splits):
                test_mask = outer_df["outer_fold"] == fold_idx
                test_events = set(outer_df.loc[test_mask, "event_id"])
                fold_samples = event_df.loc[event_df["event_id"].isin(test_events)]
                class_rows["fold"].append(fold_idx)
                for col in present_cols:
                    class_rows[col].append(int(fold_samples[col].sum()))
            p = cv_dir / "class_support_by_fold.csv"
            pd.DataFrame(class_rows).to_csv(p, index=False)
            paths["class_support_by_fold"] = p

    # --- rainfall_distribution_by_fold.csv ---
    rain_rows: list[dict[str, Any]] = []
    for fold_idx in range(plan.outer_n_splits):
        test_mask = outer_df["outer_fold"] == fold_idx
        test_events = set(outer_df.loc[test_mask, "event_id"])
        fold_samples = event_df.loc[event_df["event_id"].isin(test_events)]
        family_counts = fold_samples["rainfall_sha256"].value_counts()
        for family, count in family_counts.items():
            rain_rows.append({
                "fold": fold_idx,
                "family": family,
                "count": int(count),
            })
    p = cv_dir / "rainfall_distribution_by_fold.csv"
    pd.DataFrame(rain_rows).to_csv(p, index=False)
    paths["rainfall_distribution_by_fold"] = p

    # --- nested_cv_plan.json ---
    plan_dict = {
        "outer_n_splits": plan.outer_n_splits,
        "inner_n_splits": plan.inner_n_splits,
        "n_repeats": plan.n_repeats,
        "group_column": plan.group_column,
        "frozen_seeds": list(plan.frozen_seeds),
        "outer_fold_assignment": plan.outer_fold_assignment,
        "algorithm": "GroupKFold",
        "n_events": len(event_df),
        "n_unique_rainfall_shas": int(event_df["rainfall_sha256"].nunique()),
    }
    p = cv_dir / "nested_cv_plan.json"
    p.write_text(json.dumps(plan_dict, indent=2, ensure_ascii=False), encoding="utf-8")
    paths["nested_cv_plan"] = p

    log.info("Wrote %d nested CV artifacts to %s", len(paths), cv_dir)
    return paths


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def audit_v42_nested_cv_plan(
    output_root: Path,
    *,
    min_events: int = MIN_EVENTS_PER_FOLD,
    class_columns: tuple[str, ...] = ("pfv_safe", "tfv_improved"),
) -> dict[str, Any]:
    """Validate a previously written nested CV plan.

    Checks
    ------
    1. No event appears in more than one outer fold.
    2. No rainfall SHA spans multiple outer folds.
    3. Every fold has at least *min_events* events.
    4. Each fold has at least 1 positive sample for every class column.

    Returns a dict with ``passed`` (bool) and ``violations`` (list of str).
    """
    cv_dir = output_root / NESTED_CV_DIR
    violations: list[str] = []

    # Load artifacts
    outer_path = cv_dir / "outer_fold_assignment.csv"
    if not outer_path.exists():
        return {"passed": False, "violations": ["outer_fold_assignment.csv not found"]}
    outer_df = pd.read_csv(outer_path)

    # 1. No event in multiple folds
    dup = outer_df.groupby("event_id")["outer_fold"].nunique()
    multi = dup[dup > 1]
    if len(multi):
        violations.append(
            f"{len(multi)} events assigned to multiple outer folds"
        )

    # 2. Rainfall SHA integrity
    sha_folds = outer_df.groupby("rainfall_sha256")["outer_fold"].nunique()
    split_shas = sha_folds[sha_folds > 1]
    if len(split_shas):
        violations.append(
            f"{len(split_shas)} rainfall SHAs span multiple folds"
        )

    # 3. Minimum events per fold
    fold_counts = outer_df["outer_fold"].value_counts()
    small_folds = fold_counts[fold_counts < min_events]
    if len(small_folds):
        violations.append(
            f"{len(small_folds)} folds have fewer than {min_events} events"
        )

    # 4. Class support
    class_path = cv_dir / "class_support_by_fold.csv"
    if class_path.exists():
        class_df = pd.read_csv(class_path)
        for col in class_columns:
            if col in class_df.columns:
                zero_folds = (class_df[col] == 0).sum()
                if zero_folds > 0:
                    violations.append(
                        f"fold(s) with 0 positive '{col}' samples: "
                        f"{zero_folds} fold(s)"
                    )
    else:
        violations.append("class_support_by_fold.csv not found")

    passed = len(violations) == 0
    log.info(
        "Nested CV audit: %s (%d violations)",
        "PASSED" if passed else "FAILED",
        len(violations),
    )
    return {"passed": passed, "violations": violations}


# ---------------------------------------------------------------------------
# Convenience: plan from ledger file
# ---------------------------------------------------------------------------

def plan_from_ledger(
    ledger_path: Path,
    *,
    n_outer: int = DEFAULT_OUTER_SPLITS,
    n_inner: int = DEFAULT_INNER_SPLITS,
    seed: int = DEFAULT_SEED,
) -> tuple[V42CVPlan, pd.DataFrame]:
    """Build a plan directly from an ``event_usage_ledger.csv``.

    Returns the plan and the de-duplicated event DataFrame.
    """
    ledger = _load_event_ledger(ledger_path)
    event_df = ledger[["event_id", "rainfall_sha256"]].drop_duplicates(
        subset="event_id", keep="first"
    ).reset_index(drop=True)

    plan = plan_v42_nested_grouped_cv(
        event_df["event_id"].values,
        event_df["rainfall_sha256"].values,
        n_outer=n_outer,
        n_inner=n_inner,
        seed=seed,
    )
    return plan, event_df
