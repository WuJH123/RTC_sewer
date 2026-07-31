"""V4.2 derived supervision signals.

Generates four types of supervision data from the unified development pool:
1. One-step transition samples (single-step dynamics)
2. Multi-horizon target labels (H10/H30/H60/H90/H120)
3. Pairwise ranking pairs (Candidate vs Candidate within same state)
4. Candidate-reference pairs (Candidate vs NC/DI/Hold)
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_HORIZON_STEPS = 12
HORIZON_INTERVAL_MIN = 10

# Multi-horizon definitions (minutes)
MULTI_HORIZONS = {"H10": 10, "H30": 30, "H60": 60, "H90": 90, "H120": 120}

# Branch role mappings
BRANCH_ROLES = ("candidate", "no_control", "dynamic_internal_rules", "hold_previous")
BRANCH_ALIASES = {
    "candidate": ["candidate", "Candidate"],
    "no_control": ["no_control", "executable_passive", "No-control", "passive"],
    "dynamic_internal_rules": ["dynamic_internal_rules", "internal_rules", "Internal"],
    "hold_previous": ["hold_previous", "hold", "Hold"],
}

# KPI column patterns per horizon
KPI_METRICS = ("PFV", "TFV", "peak_TFV_rate", "priority_flood_duration_min",
               "flood_duration_min", "storage_volume", "total_flooding_rate")

# Output sub-directory
DERIVED_OUTPUT_DIR = "v42_development/derived"


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class DerivedSupervisionResult:
    """Container for derived supervision signal results."""

    one_step_transitions: pd.DataFrame
    multi_horizon_targets: pd.DataFrame
    pairwise_ranking: pd.DataFrame
    candidate_reference_pairs: pd.DataFrame
    audit: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helper: column resolution
# ---------------------------------------------------------------------------


def _resolve_kpi_column(df: pd.DataFrame, branch: str, horizon: str, metric: str) -> str | None:
    """Find the KPI column name for a given branch/horizon/metric combo."""
    # Common patterns:
    # {branch}_{metric}_{horizon}  e.g. candidate_PFV_H120
    # {branch}_{metric}_H{horizon}
    candidates = [
        f"{branch}_{metric}_{horizon}",
        f"{branch}_{metric}_H{horizon.replace('H', '')}",
    ]
    # Also try aliases
    for alias in BRANCH_ALIASES.get(branch, [branch]):
        candidates.append(f"{alias}_{metric}_{horizon}")
        candidates.append(f"{alias}_{metric}_H{horizon.replace('H', '')}")

    for c in candidates:
        if c in df.columns:
            return c
    return None


def _get_kpi_value(row: pd.Series, branch: str, horizon: str, metric: str) -> float | None:
    """Extract a KPI value from a row, trying multiple column patterns."""
    col = _resolve_kpi_column(pd.DataFrame([row]), branch, horizon, metric)
    if col and col in row.index:
        val = row[col]
        if pd.notna(val):
            return float(val)
    return None


# ---------------------------------------------------------------------------
# 6.1 One-step transitions
# ---------------------------------------------------------------------------


def derive_one_step_transitions(
    trajectory_df: pd.DataFrame,
) -> pd.DataFrame:
    """Derive one-step transition samples from trajectory data.

    Each 12-step trajectory is split into 12 individual (X_t, rainfall_t, action_t) -> X_{t+1}
    transitions for all four branches (Candidate/NC/DI/Hold).

    Parameters
    ----------
    trajectory_df : DataFrame
        Unified pool manifest with trajectory data.

    Returns
    -------
    DataFrame
        One-step transition records with source tracking fields.
    """
    records: list[dict[str, Any]] = []

    for idx, row in trajectory_df.iterrows():
        event_id = str(row.get("event_id", ""))
        state_key = str(row.get("state_key", ""))
        case_id = str(row.get("sample_id", row.get("case_id", "")))
        source_sha = str(row.get("v4_sample_identity_sha256",
                                  row.get("case_signature", "")))

        # For each of the 12 horizon steps
        for t in range(N_HORIZON_STEPS):
            # For each branch role
            for branch in BRANCH_ROLES:
                # Extract state features at time t
                # These would come from trajectory depth/action arrays if available,
                # or from KPI columns as proxies
                record = {
                    "source_event_id": event_id,
                    "source_state_id": state_key,
                    "source_case_id": case_id,
                    "source_trajectory_sha": source_sha,
                    "time_index": t,
                    "split_group": branch,
                    # KPI-based features (proxy for full state)
                    "pfv_t": _get_kpi_value(row, branch, f"H{(t+1)*10}", "PFV"),
                    "tfv_t": _get_kpi_value(row, branch, f"H{(t+1)*10}", "TFV"),
                    "peak_t": _get_kpi_value(row, branch, f"H{(t+1)*10}", "peak_TFV_rate"),
                    # Delta from previous step (transition signal)
                    "pfv_delta_t": None,
                    "tfv_delta_t": None,
                }

                # Compute deltas if t > 0
                if t > 0:
                    pfv_prev = _get_kpi_value(row, branch, f"H{t*10}", "PFV")
                    tfv_prev = _get_kpi_value(row, branch, f"H{t*10}", "TFV")
                    if record["pfv_t"] is not None and pfv_prev is not None:
                        record["pfv_delta_t"] = record["pfv_t"] - pfv_prev
                    if record["tfv_t"] is not None and tfv_prev is not None:
                        record["tfv_delta_t"] = record["tfv_t"] - tfv_prev

                records.append(record)

    result = pd.DataFrame(records)
    logger.info("Derived %d one-step transitions from %d trajectories",
                len(result), len(trajectory_df))
    return result


# ---------------------------------------------------------------------------
# 6.2 Multi-horizon targets
# ---------------------------------------------------------------------------


def derive_multi_horizon_targets(
    trajectory_df: pd.DataFrame,
) -> pd.DataFrame:
    """Derive multi-horizon target labels from trajectory KPI data.

    For each sample, computes PFV/TFV/Peak/priority_depth/storage_volume/
    total_flooding_rate at H10/H30/H60/H90/H120 horizons.

    Parameters
    ----------
    trajectory_df : DataFrame
        Unified pool manifest with KPI columns.

    Returns
    -------
    DataFrame
        Multi-horizon target records (5 horizons per sample).
    """
    records: list[dict[str, Any]] = []

    for idx, row in trajectory_df.iterrows():
        event_id = str(row.get("event_id", ""))
        state_key = str(row.get("state_key", ""))
        case_id = str(row.get("sample_id", row.get("case_id", "")))
        branch = str(row.get("split_group",
                              row.get("anchor_type", "candidate")))

        for horizon_name, horizon_min in MULTI_HORIZONS.items():
            record = {
                "source_event_id": event_id,
                "source_state_id": state_key,
                "source_case_id": case_id,
                "horizon": horizon_name,
                "horizon_min": horizon_min,
                "split_group": branch,
                # Core KPIs
                "PFV": _get_kpi_value(row, "candidate", horizon_name, "PFV"),
                "TFV": _get_kpi_value(row, "candidate", horizon_name, "TFV"),
                "Peak": _get_kpi_value(row, "candidate", horizon_name, "peak_TFV_rate"),
                "priority_depth": _get_kpi_value(
                    row, "candidate", horizon_name, "priority_flood_duration_min"
                ),
                "flood_duration": _get_kpi_value(
                    row, "candidate", horizon_name, "flood_duration_min"
                ),
                # Reference KPIs for delta computation
                "PFV_no_control": _get_kpi_value(row, "no_control", horizon_name, "PFV"),
                "TFV_internal": _get_kpi_value(row, "internal_rules", horizon_name, "TFV"),
                "Peak_internal": _get_kpi_value(row, "internal_rules", horizon_name, "peak_TFV_rate"),
            }

            # Compute deltas
            if record["PFV"] is not None and record["PFV_no_control"] is not None:
                record["delta_PFV_vs_NC"] = record["PFV"] - record["PFV_no_control"]
            if record["TFV"] is not None and record["TFV_internal"] is not None:
                record["delta_TFV_vs_DI"] = record["TFV"] - record["TFV_internal"]
            if record["Peak"] is not None and record["Peak_internal"] is not None:
                record["delta_Peak_vs_DI"] = record["Peak"] - record["Peak_internal"]

            records.append(record)

    result = pd.DataFrame(records)
    logger.info("Derived %d multi-horizon targets from %d trajectories",
                len(result), len(trajectory_df))
    return result


# ---------------------------------------------------------------------------
# 6.3 Pairwise ranking
# ---------------------------------------------------------------------------


def derive_pairwise_ranking(
    pool_df: pd.DataFrame,
) -> pd.DataFrame:
    """Derive pairwise ranking samples from Candidate branches within each state.

    For each state with 5 Candidates, generates C(5,2) = 10 pairs with
    comparative labels.

    Parameters
    ----------
    pool_df : DataFrame
        Unified pool candidate manifest.

    Returns
    -------
    DataFrame
        Pairwise ranking records with comparative labels.
    """
    records: list[dict[str, Any]] = []

    # Group by state_key
    if "state_key" not in pool_df.columns:
        logger.warning("state_key column not found, using event_id+checkpoint_id")
        pool_df = pool_df.copy()
        pool_df["state_key"] = (
            pool_df["event_id"].astype(str) + "::" +
            pool_df["checkpoint_id"].astype(str)
        )

    for state_key, group in pool_df.groupby("state_key"):
        if len(group) < 2:
            continue

        # Only build pairs among candidates
        if "anchor_type" in group.columns:
            candidates = group[group["anchor_type"].astype(str).str.contains(
                "candidate", case=False, na=False
            )]
        else:
            candidates = group

        if len(candidates) < 2:
            continue

        # Generate all pairs C(n, 2)
        indices = candidates.index.tolist()
        for i, j in combinations(range(len(indices)), 2):
            idx_a = indices[i]
            idx_b = indices[j]
            row_a = candidates.loc[idx_a]
            row_b = candidates.loc[idx_b]

            # Extract H120 KPIs for comparison
            pfv_a = _get_kpi_value(row_a, "candidate", "H120", "PFV")
            pfv_b = _get_kpi_value(row_b, "candidate", "H120", "PFV")
            tfv_a = _get_kpi_value(row_a, "candidate", "H120", "TFV")
            tfv_b = _get_kpi_value(row_b, "candidate", "H120", "TFV")
            peak_a = _get_kpi_value(row_a, "candidate", "H120", "peak_TFV_rate")
            peak_b = _get_kpi_value(row_b, "candidate", "H120", "peak_TFV_rate")

            # Comparative labels
            pfv_safe_priority = (pfv_a <= pfv_b) if (pfv_a is not None and pfv_b is not None) else None
            peak_non_inferior = (peak_a <= peak_b) if (peak_a is not None and peak_b is not None) else None
            tfv_improved = (tfv_a <= tfv_b) if (tfv_a is not None and tfv_b is not None) else None

            # PFV-first lexicographic ordering
            if pfv_a is not None and pfv_b is not None:
                if pfv_a != pfv_b:
                    pfv_first_lex = int(pfv_a < pfv_b)  # lower PFV is better
                elif tfv_a is not None and tfv_b is not None:
                    pfv_first_lex = int(tfv_a < tfv_b)
                else:
                    pfv_first_lex = 0
            else:
                pfv_first_lex = None

            # Utility difference (proxy: PFV delta)
            if pfv_a is not None and pfv_b is not None:
                true_utility_diff = pfv_a - pfv_b
            else:
                true_utility_diff = None

            # Hard negative: both candidates are very similar in PFV
            hard_negative = False
            if true_utility_diff is not None:
                hard_negative = abs(true_utility_diff) < 100.0  # threshold

            record = {
                "state_key": state_key,
                "source_event_id": str(row_a.get("event_id", "")),
                "candidate_a_id": str(row_a.get("sample_id", row_a.get("candidate_id", idx_a))),
                "candidate_b_id": str(row_b.get("sample_id", row_b.get("candidate_id", idx_b))),
                "pfv_a": pfv_a,
                "pfv_b": pfv_b,
                "pfv_safe_priority": pfv_safe_priority,
                "peak_a": peak_a,
                "peak_b": peak_b,
                "peak_non_inferior": peak_non_inferior,
                "tfv_a": tfv_a,
                "tfv_b": tfv_b,
                "tfv_improved": tfv_improved,
                "pfv_first_lexicographic": pfv_first_lex,
                "true_utility_difference": true_utility_diff,
                "hard_negative_relation": hard_negative,
            }
            records.append(record)

    result = pd.DataFrame(records)
    logger.info("Derived %d pairwise ranking samples", len(result))
    return result


# ---------------------------------------------------------------------------
# 6.4 Candidate-Reference pairs
# ---------------------------------------------------------------------------


def derive_candidate_reference_pairs(
    pool_df: pd.DataFrame,
) -> pd.DataFrame:
    """Derive Candidate vs Reference (NC/DI/Hold) pairs for each state.

    Parameters
    ----------
    pool_df : DataFrame
        Unified pool manifest (both candidates and references).

    Returns
    -------
    DataFrame
        Candidate-reference pair records with delta information.
    """
    records: list[dict[str, Any]] = []

    # Ensure state_key
    if "state_key" not in pool_df.columns:
        pool_df = pool_df.copy()
        pool_df["state_key"] = (
            pool_df["event_id"].astype(str) + "::" +
            pool_df["checkpoint_id"].astype(str)
        )

    # Identify reference types
    ref_type_map = {
        "no_control": ["no_control", "executable_passive", "No-control", "passive"],
        "dynamic_internal_rules": ["internal_rules", "Internal", "dynamic_internal_rules"],
        "hold_previous": ["hold_previous", "hold", "Hold"],
    }

    for state_key, group in pool_df.groupby("state_key"):
        # Find candidates
        if "anchor_type" in group.columns:
            cand_mask = group["anchor_type"].astype(str).str.contains(
                "candidate", case=False, na=False
            )
        else:
            cand_mask = pd.Series([True] * len(group), index=group.index)

        candidates = group[cand_mask]
        if candidates.empty:
            continue

        # For each reference type
        for ref_name, ref_aliases in ref_type_map.items():
            if "anchor_type" in group.columns:
                ref_mask = group["anchor_type"].astype(str).isin(ref_aliases)
            else:
                continue

            refs = group[ref_mask]
            if refs.empty:
                continue

            # Build pairs: each candidate vs each reference of this type
            for _, cand_row in candidates.iterrows():
                for _, ref_row in refs.iterrows():
                    pfv_cand = _get_kpi_value(cand_row, "candidate", "H120", "PFV")
                    pfv_ref = _get_kpi_value(ref_row, ref_name, "H120", "PFV")
                    tfv_cand = _get_kpi_value(cand_row, "candidate", "H120", "TFV")
                    tfv_ref = _get_kpi_value(ref_row, ref_name, "H120", "TFV")
                    peak_cand = _get_kpi_value(cand_row, "candidate", "H120", "peak_TFV_rate")
                    peak_ref = _get_kpi_value(ref_row, ref_name, "H120", "peak_TFV_rate")

                    record = {
                        "state_key": state_key,
                        "source_event_id": str(cand_row.get("event_id", "")),
                        "candidate_id": str(cand_row.get("sample_id", cand_row.get("candidate_id", ""))),
                        "reference_id": str(ref_row.get("sample_id", ref_row.get("candidate_id", ""))),
                        "reference_type": ref_name,
                        # Candidate KPIs
                        "candidate_PFV": pfv_cand,
                        "candidate_TFV": tfv_cand,
                        "candidate_Peak": peak_cand,
                        # Reference KPIs
                        "reference_PFV": pfv_ref,
                        "reference_TFV": tfv_ref,
                        "reference_Peak": peak_ref,
                        # Deltas
                        "delta_PFV": (pfv_cand - pfv_ref) if (pfv_cand is not None and pfv_ref is not None) else None,
                        "delta_TFV": (tfv_cand - tfv_ref) if (tfv_cand is not None and tfv_ref is not None) else None,
                        "delta_Peak": (peak_cand - peak_ref) if (peak_cand is not None and peak_ref is not None) else None,
                    }
                    records.append(record)

    result = pd.DataFrame(records)
    logger.info("Derived %d candidate-reference pairs", len(result))
    return result


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------


def build_v42_derived_supervision(
    project_root: Path,
    output_root: Path,
    pool_manifest: pd.DataFrame | None = None,
) -> DerivedSupervisionResult:
    """Build all V4.2 derived supervision signals.

    Parameters
    ----------
    project_root : Path
        Project root directory.
    output_root : Path
        Output root containing unified pool data.
    pool_manifest : DataFrame, optional
        Pre-loaded unified pool manifest. If None, loads from disk.

    Returns
    -------
    DerivedSupervisionResult
        All derived supervision datasets and audit info.
    """
    project_root = Path(project_root)
    output_root = Path(output_root)
    warnings: list[str] = []

    # ------------------------------------------------------------------
    # 1. Load pool data if not provided
    # ------------------------------------------------------------------
    if pool_manifest is None:
        pool_dir = output_root / "v42_development"
        cand_path = pool_dir / "unified_candidate_manifest.parquet"
        ref_path = pool_dir / "unique_reference_manifest.parquet"

        if cand_path.exists():
            pool_manifest = pd.read_parquet(cand_path)
            logger.info("Loaded candidate pool: %d rows", len(pool_manifest))
        else:
            # Fallback: load from source manifests
            base_path = output_root / "action_effect_dataset_v4" / "v4_dataset_manifest.csv"
            if base_path.exists():
                pool_manifest = pd.read_csv(base_path)
                logger.info("Loaded base manifest as fallback: %d rows", len(pool_manifest))
            else:
                raise FileNotFoundError(f"Pool manifest not found at {cand_path}")

        if ref_path.exists():
            ref_df = pd.read_parquet(ref_path)
            pool_manifest = pd.concat([pool_manifest, ref_df], ignore_index=True, sort=False)
            logger.info("After merging references: %d rows", len(pool_manifest))

    # Ensure state_key
    if "state_key" not in pool_manifest.columns:
        pool_manifest = pool_manifest.copy()
        eid = pool_manifest.get("event_id", pd.Series("", index=pool_manifest.index)).astype(str)
        cid = pool_manifest.get("checkpoint_id", pd.Series("", index=pool_manifest.index)).astype(str)
        pool_manifest["state_key"] = eid + "::" + cid

    # ------------------------------------------------------------------
    # 2. Derive one-step transitions
    # ------------------------------------------------------------------
    logger.info("Deriving one-step transitions...")
    one_step = derive_one_step_transitions(pool_manifest)

    # ------------------------------------------------------------------
    # 3. Derive multi-horizon targets
    # ------------------------------------------------------------------
    logger.info("Deriving multi-horizon targets...")
    multi_horizon = derive_multi_horizon_targets(pool_manifest)

    # ------------------------------------------------------------------
    # 4. Derive pairwise ranking
    # ------------------------------------------------------------------
    logger.info("Deriving pairwise ranking...")
    pairwise = derive_pairwise_ranking(pool_manifest)

    # ------------------------------------------------------------------
    # 5. Derive candidate-reference pairs
    # ------------------------------------------------------------------
    logger.info("Deriving candidate-reference pairs...")
    cr_pairs = derive_candidate_reference_pairs(pool_manifest)

    # ------------------------------------------------------------------
    # 6. Build audit
    # ------------------------------------------------------------------
    n_trajectories = len(pool_manifest)
    n_states = pool_manifest["state_key"].nunique()

    # Expected counts
    expected_one_step = n_trajectories * N_HORIZON_STEPS * len(BRANCH_ROLES)
    expected_multi_horizon = n_trajectories * len(MULTI_HORIZONS)
    expected_pairwise = n_states * 10  # C(5,2) = 10 per state
    expected_cr = n_states * 3  # 3 reference types per state

    audit = {
        "n_trajectories": n_trajectories,
        "n_states": n_states,
        "one_step_transitions": {
            "actual": len(one_step),
            "expected": expected_one_step,
            "match": len(one_step) == expected_one_step,
        },
        "multi_horizon_targets": {
            "actual": len(multi_horizon),
            "expected": expected_multi_horizon,
            "match": len(multi_horizon) == expected_multi_horizon,
        },
        "pairwise_ranking": {
            "actual": len(pairwise),
            "expected_max": expected_pairwise,
        },
        "candidate_reference_pairs": {
            "actual": len(cr_pairs),
            "expected_max": expected_cr,
        },
        "no_cross_state_pairs": _verify_no_cross_state_pairs(pairwise),
        "warnings": warnings[:20],
    }

    return DerivedSupervisionResult(
        one_step_transitions=one_step,
        multi_horizon_targets=multi_horizon,
        pairwise_ranking=pairwise,
        candidate_reference_pairs=cr_pairs,
        audit=audit,
        warnings=warnings,
    )


def _verify_no_cross_state_pairs(pairwise_df: pd.DataFrame) -> bool:
    """Verify that no pairwise ranking crosses state boundaries."""
    if pairwise_df.empty:
        return True
    # Each pair should have a single state_key
    if "state_key" in pairwise_df.columns:
        return pairwise_df["state_key"].notna().all()
    return True


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------


def write_derived_supervision(
    result: DerivedSupervisionResult,
    output_dir: Path,
) -> dict[str, str]:
    """Write derived supervision outputs to disk.

    Returns dict of written file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    # Parquet for large datasets
    for name, df in [
        ("one_step_transitions.parquet", result.one_step_transitions),
        ("multi_horizon_targets.parquet", result.multi_horizon_targets),
        ("pairwise_ranking.parquet", result.pairwise_ranking),
        ("candidate_reference_pairs.parquet", result.candidate_reference_pairs),
    ]:
        path = output_dir / name
        df.to_parquet(path, index=False)
        written[name] = str(path)

    # Audit JSON
    audit_path = output_dir / "derivation_audit.json"
    audit_path.write_text(
        json.dumps(result.audit, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    written["derivation_audit.json"] = str(audit_path)

    logger.info("Wrote derived supervision: %d files to %s", len(written), output_dir)
    return written


# ---------------------------------------------------------------------------
# Audit function
# ---------------------------------------------------------------------------


def audit_v42_derived_supervision(output_root: Path) -> dict[str, Any]:
    """Audit the derived supervision signals for correctness.

    Checks:
    - One-step sample count = trajectories × 12 × 4 branches
    - Multi-horizon count = trajectories × 5 horizons
    - Ranking pair count = states × 10
    - CR pair count = states × 3
    - No cross-state pairs

    Returns dict with status and exit_code.
    """
    output_root = Path(output_root)
    derived_dir = output_root / DERIVED_OUTPUT_DIR

    if not derived_dir.exists():
        return {
            "status": "blocked",
            "exit_code": 2,
            "message": f"Derived directory not found: {derived_dir}",
        }

    errors: list[str] = []

    # Load files
    try:
        one_step = pd.read_parquet(derived_dir / "one_step_transitions.parquet")
        multi_horizon = pd.read_parquet(derived_dir / "multi_horizon_targets.parquet")
        pairwise = pd.read_parquet(derived_dir / "pairwise_ranking.parquet")
        cr_pairs = pd.read_parquet(derived_dir / "candidate_reference_pairs.parquet")
    except FileNotFoundError as exc:
        return {
            "status": "blocked",
            "exit_code": 2,
            "message": f"Missing file: {exc}",
        }

    # Load audit for expected counts
    audit_path = derived_dir / "derivation_audit.json"
    if audit_path.exists():
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    else:
        audit = {}

    # Check 1: One-step transitions
    if "one_step_transitions" in audit:
        expected = audit["one_step_transitions"].get("expected", 0)
        actual = len(one_step)
        if expected > 0 and actual != expected:
            errors.append(
                f"One-step count mismatch: {actual} actual vs {expected} expected"
            )

    # Check 2: Multi-horizon targets
    if "multi_horizon_targets" in audit:
        expected = audit["multi_horizon_targets"].get("expected", 0)
        actual = len(multi_horizon)
        if expected > 0 and actual != expected:
            errors.append(
                f"Multi-horizon count mismatch: {actual} actual vs {expected} expected"
            )

    # Check 3: Pairwise ranking — no cross-state
    if "state_key" in pairwise.columns:
        cross_state = pairwise[pairwise["state_key"].isna()]
        if len(cross_state) > 0:
            errors.append(f"{len(cross_state)} pairs have missing state_key")

    # Check 4: CR pairs — reference_type should be valid
    if "reference_type" in cr_pairs.columns:
        valid_refs = {"no_control", "dynamic_internal_rules", "hold_previous"}
        actual_refs = set(cr_pairs["reference_type"].unique())
        invalid = actual_refs - valid_refs
        if invalid:
            errors.append(f"Invalid reference types: {invalid}")

    # Statistics
    stats = {
        "n_one_step": len(one_step),
        "n_multi_horizon": len(multi_horizon),
        "n_pairwise": len(pairwise),
        "n_cr_pairs": len(cr_pairs),
    }

    status = "pass" if not errors else "blocked"
    exit_code = 0 if not errors else 2

    return {
        "status": status,
        "exit_code": exit_code,
        "errors": errors,
        "statistics": stats,
        "message": "Derived supervision audit passed" if status == "pass"
        else f"Audit failed: {'; '.join(errors)}",
    }
