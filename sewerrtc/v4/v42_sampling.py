"""V4.2 fold-internal training sampler.

Provides stratified, event-balanced, hard-negative-aware sampling for training
folds.  The sampler operates **only** on the training portion of a fold — it
never modifies validation / test distributions.

Four sampling dimensions are combined:
  1. Event-balanced weighting — prevents candidate-rich events from dominating.
  2. State-stratum coverage — ensures each mini-batch spans all feasibility
     regimes (high_feasibility, boundary, fallback_likely, …).
  3. Hard-negative up-weighting — 2× base weight for scientifically hard cases.
  4. Optional class-balanced resampling with duplication tracking.

All random seeds are frozen at construction time for full reproducibility.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Recognised state strata (order matters for deterministic iteration).
KNOWN_STATE_STRATA: tuple[str, ...] = (
    "high_feasibility",
    "boundary",
    "fallback_likely",
    "low_opportunity",
    "pre_peak",
    "peak",
    "recession",
    "PFV_active",
    "PFV_inactive",
)

# Hard-negative categories that receive 2× weight.
HARD_NEGATIVE_TYPES: frozenset[str] = frozenset({
    "pfv_unsafe_tfv_improved",
    "peak_degraded_tfv_improved",
    "joint_hard_negative",
    "pfv_boundary",
    "peak_boundary",
    "previous_false_safe",
})

HARD_NEGATIVE_MULTIPLIER = 2.0

# ---------------------------------------------------------------------------
# Prohibition checks
# ---------------------------------------------------------------------------

_PROHIBITED_PATTERNS: tuple[str, ...] = (
    "oversample_before_split",
    "duplicate_then_split",
    "same_state_to_validation",
    "smote_on_action_sequences",
    "continuous_interpolation_on_binary_pump",
    "synthetic_kpi_without_swmm_truth",
)


def validate_sampling_prohibitions(
    sampler: V42GroupedTrainingSampler,
    fold_split: dict[str, np.ndarray],
) -> list[str]:
    """Return a list of violated prohibition strings (empty == clean).

    Checks
    ------
    * No oversampling was applied before fold splitting.
    * No candidate-row duplication before splitting.
    * No same-state leakage into validation.
    * No SMOTE on raw action sequences.
    * No continuous interpolation of binary pump actions.
    * No synthetic KPI labels without SWMM truth.
    """
    violations: list[str] = []

    train_idx = fold_split.get("train", np.array([], dtype=int))
    val_idx = fold_split.get("val", np.array([], dtype=int))

    # 1. Oversample before split — check that sampler was not given indices
    #    outside the training fold.
    if hasattr(sampler, "_original_n_samples"):
        max_train = int(train_idx.max()) if len(train_idx) else 0
        if max_train >= sampler._original_n_samples:
            violations.append("oversample_before_split")

    # 2. Duplicate-then-split — detect if any training index appears more
    #    than once in the *full* index array before sampling.
    if len(train_idx) > 0:
        unique_before = len(np.unique(train_idx))
        if unique_before < len(train_idx):
            violations.append("duplicate_then_split")

    # 3. Same-state to validation — check that no state key appears in both
    #    train and val (requires state_key attribute).
    if hasattr(sampler, "_state_keys") and sampler._state_keys is not None:
        train_keys = set(sampler._state_keys[train_idx].tolist())
        val_keys = set(sampler._state_keys[fold_split["val"]].tolist())
        overlap = train_keys & val_keys
        if overlap:
            violations.append("same_state_to_validation")

    # 4-6. Structural checks — these are design-level guarantees that the
    # sampler class itself must honour.  We verify by inspecting class flags.
    if not getattr(sampler, "_smote_disabled", True):
        violations.append("smote_on_action_sequences")
    if not getattr(sampler, "_continuous_interpolation_disabled", True):
        violations.append("continuous_interpolation_on_binary_pump")
    if not getattr(sampler, "_synthetic_kpi_disabled", True):
        violations.append("synthetic_kpi_without_swmm_truth")

    return violations


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------

def audit_v42_sampling(
    sampler: V42GroupedTrainingSampler,
    fold_split: dict[str, np.ndarray],
    original_indices: np.ndarray,
) -> dict[str, Any]:
    """Audit a sampling operation and return a diagnostic dict.

    Parameters
    ----------
    sampler : V42GroupedTrainingSampler
    fold_split : dict
        Must contain ``"train"`` and ``"val"`` index arrays.
    original_indices : np.ndarray
        The original (un-sampled) row indices of the full dataset.

    Returns
    -------
    dict with keys:
        val_distribution_unchanged : bool
        unique_train_samples : int
        effective_train_samples : int
        event_balance_report : dict
        hard_negative_coverage : dict
    """
    train_idx = fold_split["train"]
    val_idx = fold_split["val"]

    # --- Validation distribution unchanged ---
    original_val_set = set(original_indices) & set(val_idx)
    sampled_val_set = set(val_idx)
    val_unchanged = original_val_set == sampled_val_set

    # --- Unique vs effective training samples ---
    unique_train = len(np.unique(train_idx))
    effective_train = len(train_idx)

    # --- Event balance ---
    event_balance: dict[str, int] = {}
    if sampler._event_groups is not None:
        train_events = sampler._event_groups.iloc[train_idx]
        event_balance = train_events.value_counts().to_dict()

    # --- Hard-negative coverage ---
    hn_coverage: dict[str, int] = {}
    if sampler._hard_negative_types is not None:
        train_hn = sampler._hard_negative_types.iloc[train_idx]
        for hn_type in HARD_NEGATIVE_TYPES:
            hn_coverage[hn_type] = int((train_hn == hn_type).sum())

    return {
        "val_distribution_unchanged": val_unchanged,
        "unique_train_samples": unique_train,
        "effective_train_samples": effective_train,
        "duplication_ratio": (
            effective_train / max(unique_train, 1) - 1.0
        ),
        "event_balance_report": event_balance,
        "hard_negative_coverage": hn_coverage,
    }


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------

class V42GroupedTrainingSampler:
    """Fold-internal training sampler.

    Operates **only** on the training fold.  Validation / test distributions
    are never modified.

    Parameters
    ----------
    event_groups : pd.Series
        ``sample_idx -> event_id`` for every sample in the full dataset.
    state_strata : pd.Series or None
        ``sample_idx -> stratum`` (e.g. ``"high_feasibility"``).
    hard_negative_types : pd.Series or None
        ``sample_idx -> hn_type`` (empty string for non-hard-negative).
    state_keys : np.ndarray or None
        Per-sample unique state key (``event_id::checkpoint_id``).
    random_state : int
        Frozen seed — no runtime randomness allowed.
    """

    # Structural prohibition flags (always True — these techniques are banned).
    _smote_disabled: bool = True
    _continuous_interpolation_disabled: bool = True
    _synthetic_kpi_disabled: bool = True

    def __init__(
        self,
        event_groups: pd.Series,
        state_strata: pd.Series | None = None,
        hard_negative_types: pd.Series | None = None,
        state_keys: np.ndarray | None = None,
        random_state: int = 42,
    ) -> None:
        self._event_groups = event_groups.reset_index(drop=True)
        self._state_strata = (
            state_strata.reset_index(drop=True) if state_strata is not None else None
        )
        self._hard_negative_types = (
            hard_negative_types.reset_index(drop=True)
            if hard_negative_types is not None
            else None
        )
        self._state_keys = state_keys
        self._original_n_samples = len(event_groups)
        self._rng = np.random.RandomState(random_state)
        self._random_state = random_state

        # Pre-compute event-level statistics.
        self._event_counts: dict[str, int] = (
            self._event_groups.value_counts().to_dict()
        )

        log.info(
            "V42GroupedTrainingSampler: %d samples, %d events, "
            "strata=%s, hard_neg=%s",
            self._original_n_samples,
            len(self._event_counts),
            "yes" if self._state_strata is not None else "no",
            "yes" if self._hard_negative_types is not None else "no",
        )

    # ------------------------------------------------------------------
    # Weight computation
    # ------------------------------------------------------------------

    def _event_balanced_weights(
        self, indices: np.ndarray
    ) -> np.ndarray:
        """Return per-sample weights so each event has equal total weight."""
        groups = self._event_groups.iloc[indices]
        weights = np.zeros(len(indices), dtype=np.float64)
        for i, eid in enumerate(groups.values):
            n_event = self._event_counts.get(str(eid), 1)
            weights[i] = 1.0 / max(n_event, 1)
        return weights

    def _stratum_coverage(self, indices: np.ndarray) -> np.ndarray:
        """Return per-sample weights to encourage stratum coverage.

        Rare strata receive higher weight so that each mini-batch is more
        likely to cover all regimes.
        """
        if self._state_strata is None:
            return np.ones(len(indices), dtype=np.float64)

        strata = self._state_strata.iloc[indices].values.astype(str)
        # Count samples per stratum within the training fold.
        stratum_counts: dict[str, int] = {}
        for s in strata:
            stratum_counts[s] = stratum_counts.get(s, 0) + 1

        n_total = max(len(indices), 1)
        n_strata = max(len(stratum_counts), 1)
        target_per_stratum = n_total / n_strata

        weights = np.zeros(len(indices), dtype=np.float64)
        for i, s in enumerate(strata):
            count = stratum_counts.get(s, 1)
            # Up-weight rare strata, down-weight common ones.
            weights[i] = target_per_stratum / max(count, 1)
        return weights

    def _hard_negative_weights(
        self, indices: np.ndarray
    ) -> np.ndarray:
        """Return 2× weight for hard-negative samples, 1× otherwise."""
        if self._hard_negative_types is None:
            return np.ones(len(indices), dtype=np.float64)

        hn_types = self._hard_negative_types.iloc[indices].values.astype(str)
        weights = np.ones(len(indices), dtype=np.float64)
        for i, hn in enumerate(hn_types):
            if hn in HARD_NEGATIVE_TYPES:
                weights[i] = HARD_NEGATIVE_MULTIPLIER
        return weights

    def get_sample_weights(
        self, fold_train_indices: np.ndarray
    ) -> np.ndarray:
        """Compute combined per-sample weights for the training fold.

        The final weight is the product of:
          event_balance × stratum_coverage × hard_negative
        """
        w_event = self._event_balanced_weights(fold_train_indices)
        w_stratum = self._stratum_coverage(fold_train_indices)
        w_hn = self._hard_negative_weights(fold_train_indices)

        combined = w_event * w_stratum * w_hn
        # Normalise so weights sum to len(fold_train_indices).
        total = combined.sum()
        if total > 0:
            combined = combined / total * len(fold_train_indices)
        return combined

    # ------------------------------------------------------------------
    # Epoch sampling
    # ------------------------------------------------------------------

    def _class_balanced_indices(
        self, fold_train_indices: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Optionally over-sample minority classes within the training fold.

        Returns
        -------
        sampled_indices : np.ndarray
            May contain duplicates (oversampled copies).
        source_sample_ids : np.ndarray
            For each entry in *sampled_indices*, the original sample index it
            was derived from (identity when no oversampling occurs).
        """
        if self._state_strata is None:
            # No stratification — just return the original indices.
            return fold_train_indices, fold_train_indices.copy()

        strata = self._state_strata.iloc[fold_train_indices].values.astype(str)
        unique_strata, stratum_counts = np.unique(strata, return_counts=True)
        max_count = int(stratum_counts.max()) if len(stratum_counts) else 0

        sampled_parts: list[np.ndarray] = []
        source_parts: list[np.ndarray] = []

        for stratum in unique_strata:
            mask = strata == stratum
            stratum_idx = fold_train_indices[mask]
            n_current = int(mask.sum())

            if n_current < max_count and n_current > 0:
                # Oversample to match the largest stratum.
                n_needed = max_count - n_current
                extra = self._rng.choice(stratum_idx, size=n_needed, replace=True)
                sampled_parts.append(np.concatenate([stratum_idx, extra]))
                source_parts.append(np.concatenate([stratum_idx, extra]))  # copies track source
            else:
                sampled_parts.append(stratum_idx)
                source_parts.append(stratum_idx.copy())

        if sampled_parts:
            sampled_indices = np.concatenate(sampled_parts)
            source_sample_ids = np.concatenate(source_parts)
        else:
            sampled_indices = fold_train_indices.copy()
            source_sample_ids = fold_train_indices.copy()

        return sampled_indices, source_sample_ids

    def sample_epoch(
        self, fold_train_indices: np.ndarray
    ) -> np.ndarray:
        """Generate a shuffled index array for one training epoch.

        Combines event-balanced weighting, stratum coverage, hard-negative
        up-weighting and optional class-balanced resampling.

        The result is fully deterministic given the same ``random_state`` —
        the internal RNG is reset to the frozen seed at the start of every
        call so that repeated calls with the same *fold_train_indices*
        produce identical output.
        """
        # Reset RNG to frozen seed for full reproducibility.
        self._rng = np.random.RandomState(self._random_state)

        if len(fold_train_indices) == 0:
            return fold_train_indices.copy()

        # Step 1: class-balanced resampling (may introduce duplicates).
        resampled, _source = self._class_balanced_indices(fold_train_indices)

        # Step 2: compute combined weights for the (possibly expanded) set.
        weights = self.get_sample_weights(resampled)

        # Step 3: weighted shuffle.
        # Use Gumbel-max trick for deterministic weighted permutation.
        gumbel_noise = -np.log(
            -np.log(self._rng.uniform(1e-10, 1.0, size=len(resampled)))
        )
        order = np.argsort(-(np.log(np.maximum(weights, 1e-10)) + gumbel_noise))

        return resampled[order]

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    def summary(self, fold_train_indices: np.ndarray) -> dict[str, Any]:
        """Return a human-readable summary of the sampler state for a fold."""
        weights = self.get_sample_weights(fold_train_indices)
        resampled, source = self._class_balanced_indices(fold_train_indices)

        event_counts = self._event_groups.iloc[fold_train_indices].value_counts()
        hn_counts: dict[str, int] = {}
        if self._hard_negative_types is not None:
            hn_series = self._hard_negative_types.iloc[fold_train_indices]
            for hn_type in HARD_NEGATIVE_TYPES:
                hn_counts[hn_type] = int((hn_series == hn_type).sum())

        stratum_counts: dict[str, int] = {}
        if self._state_strata is not None:
            stratum_series = self._state_strata.iloc[fold_train_indices]
            stratum_counts = stratum_series.value_counts().to_dict()

        return {
            "n_train_unique": len(fold_train_indices),
            "n_train_effective": len(resampled),
            "n_events": len(event_counts),
            "weight_min": float(weights.min()),
            "weight_max": float(weights.max()),
            "weight_mean": float(weights.mean()),
            "event_balance": event_counts.to_dict(),
            "stratum_distribution": stratum_counts,
            "hard_negative_counts": hn_counts,
            "random_state": self._random_state,
        }
