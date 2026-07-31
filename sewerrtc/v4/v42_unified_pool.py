"""V4.2 unified development data pool.

Merges Train1600, Aug1, and optional V4.0/V4.1 calibration data into a single
development pool with compatibility validation, reference deduplication, and
comprehensive auditing.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONTROL_INTERVAL_MIN = 10
RECORDING_INTERVAL_MIN = 5
HORIZON_DEFINITIONS = {"H10": 10, "H30": 30, "H60": 60, "H90": 90, "H120": 120}

BRANCH_ROLES = ("candidate", "no_control", "dynamic_internal_rules", "hold_previous")

COMPATIBILITY_EXACT = "exact_compatible"
COMPATIBILITY_TRAJECTORY = "trajectory_compatible_only"
COMPATIBILITY_AUXILIARY = "auxiliary_pretraining_only"
COMPATIBILITY_INCOMPATIBLE = "incompatible"

# Output sub-directory
POOL_OUTPUT_DIR = "v42_development"

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class UnifiedPoolResult:
    """Container for unified pool build results."""

    candidate_manifest: pd.DataFrame
    reference_manifest: pd.DataFrame
    event_manifest: pd.DataFrame
    state_manifest: pd.DataFrame
    compatibility_report: pd.DataFrame
    data_role_manifest: pd.DataFrame
    audit: dict[str, Any]
    warnings: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Data source paths
# ---------------------------------------------------------------------------


def _resolve_train1600_manifest(output_root: Path) -> Path | None:
    """Locate the Train1600 action-effect manifest."""
    candidates = [
        output_root / "action_effect_dataset_v4" / "v4_dataset_manifest.csv",
        output_root / "train1600_v3" / "dataset" / "train1600_v3_sample_manifest.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _resolve_aug1_manifest(output_root: Path) -> Path | None:
    """Locate the Aug1 generation manifest."""
    candidates = [
        output_root / "dual_reference_aug1" / "v4_aug1_generation_manifest.csv",
        output_root / "dual_reference_aug1" / "v4_aug1_dataset_manifest.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _resolve_calib_manifests(output_root: Path) -> list[Path]:
    """Locate optional V4.0/V4.1 calibration manifests."""
    found: list[Path] = []
    for pattern in ("calibration/*/manifest.csv", "locked/*/manifest.csv"):
        for p in sorted(output_root.glob(pattern)):
            if p.exists():
                found.append(p)
    return found


# ---------------------------------------------------------------------------
# Compatibility validation
# ---------------------------------------------------------------------------


def validate_pool_compatibility(
    batch_a: pd.DataFrame,
    batch_b: pd.DataFrame,
) -> list[str]:
    """Validate compatibility between two data batches before merging.

    Returns a list of incompatibility messages (empty = fully compatible).
    """
    issues: list[str] = []

    # Check network SHA
    net_sha_a = set(batch_a.get("network_sha256", batch_a.get("network_sha", [])).astype(str).unique())
    net_sha_b = set(batch_b.get("network_sha256", batch_b.get("network_sha", [])).astype(str).unique())
    if net_sha_a and net_sha_b and not net_sha_a.intersection(net_sha_b):
        issues.append(
            f"network SHA mismatch: A={sorted(net_sha_a)[:2]} vs B={sorted(net_sha_b)[:2]}"
        )

    # Check INP SHA (if present)
    for col in ("inp_sha256", "initial_state_sha256"):
        if col in batch_a.columns and col in batch_b.columns:
            sha_a = set(batch_a[col].astype(str).unique())
            sha_b = set(batch_b[col].astype(str).unique())
            if sha_a and sha_b and not sha_a.intersection(sha_b):
                issues.append(f"{col} mismatch between batches")

    # Check control_interval (if metadata present)
    for col in ("control_interval_min", "horizon_interval_min"):
        if col in batch_a.columns:
            vals_a = set(batch_a[col].dropna().unique())
            if vals_a and vals_a != {CONTROL_INTERVAL_MIN} and col == "control_interval_min":
                issues.append(f"batch A {col} values {vals_a} != expected {CONTROL_INTERVAL_MIN}")

    # Check recording_interval
    for col in ("recording_interval_min", "history_interval_min"):
        if col in batch_a.columns:
            vals_a = set(batch_a[col].dropna().unique())
            if vals_a and vals_a != {RECORDING_INTERVAL_MIN} and col == "recording_interval_min":
                issues.append(f"batch A {col} values {vals_a} != expected {RECORDING_INTERVAL_MIN}")

    # Check H120 definition consistency (column presence)
    h120_cols_a = [c for c in batch_a.columns if "H120" in c]
    h120_cols_b = [c for c in batch_b.columns if "H120" in c]
    if h120_cols_a and h120_cols_b:
        common = set(h120_cols_a).intersection(h120_cols_b)
        if len(common) < min(len(h120_cols_a), len(h120_cols_b)) * 0.5:
            issues.append(
                f"H120 column overlap too low: {len(common)}/{len(h120_cols_a)}"
            )

    # Check split semantics
    if "split" in batch_a.columns and "split" in batch_b.columns:
        splits_a = set(batch_a["split"].astype(str).unique())
        splits_b = set(batch_b["split"].astype(str).unique())
        if not splits_a.intersection(splits_b):
            issues.append(f"split semantics mismatch: A={splits_a} vs B={splits_b}")

    return issues


def classify_data_compatibility(row: pd.Series) -> str:
    """Classify a single row's compatibility level for V4.2 usage.

    Returns one of: exact_compatible, trajectory_compatible_only,
    auxiliary_pretraining_only, incompatible.
    """
    # Check essential contract fields
    has_event = bool(row.get("event_id", ""))
    has_checkpoint = bool(row.get("checkpoint_id", row.get("checkpoint_elapsed_min", "")))

    # Check KPI completeness
    h120_cols = [k for k in row.index if "H120" in str(k)]
    has_h120_kpis = any(pd.notna(row.get(c, None)) for c in h120_cols) if h120_cols else False

    # Check trajectory content availability
    has_trajectory = bool(row.get("provenance_mode", "")) == "trajectory_content" or bool(
        row.get("runtime_executed", False)
    )

    # Check recovery status
    recovery = str(row.get("recovery_status", row.get("candidate_recovery_label_status", "")))
    is_recovered = recovery in ("not_recovered", "pass", "")

    if has_event and has_checkpoint and has_h120_kpis and is_recovered:
        return COMPATIBILITY_EXACT

    if has_event and has_checkpoint and has_trajectory and is_recovered:
        return COMPATIBILITY_TRAJECTORY

    if has_event and has_trajectory:
        return COMPATIBILITY_AUXILIARY

    return COMPATIBILITY_INCOMPATIBLE


# ---------------------------------------------------------------------------
# Reference deduplication
# ---------------------------------------------------------------------------


def dedup_references(manifest_df: pd.DataFrame) -> pd.DataFrame:
    """Deduplicate reference (non-candidate) rows.

    Dedup key: event_id + checkpoint_id + reference_type + contract_sha.
    For the same checkpoint, NC/DI/Hold each keep only 1 row;
    Candidate rows are kept as-is (5 per state).
    """
    if manifest_df.empty:
        return manifest_df

    df = manifest_df.copy()

    # Identify reference type from role/branch columns
    if "reference_type" in df.columns:
        ref_type_col = "reference_type"
    elif "anchor_type" in df.columns:
        ref_type_col = "anchor_type"
    elif "branch_role" in df.columns:
        ref_type_col = "branch_role"
    else:
        # Infer from column patterns
        df = df.assign(reference_type="unknown")
        ref_type_col = "reference_type"

    # Build dedup key
    sha_col = None
    for c in ("contract_sha", "case_signature", "sample_id", "v4_sample_identity_sha256"):
        if c in df.columns:
            sha_col = c
            break
    if sha_col is None:
        sha_col = "sample_id"
        df = df.assign(sample_id=range(len(df)))

    dedup_key = ["event_id", "checkpoint_id", ref_type_col, sha_col]

    # Separate candidates from references
    is_candidate = df.get("split_group", df.get("v4_data_layer", pd.Series("", index=df.index))).astype(str).str.contains(
        "candidate", case=False, na=False
    ) | (df.get("anchor_type", pd.Series("", index=df.index)).astype(str) == "candidate")

    # For base dataset: candidate rows have anchor_type or are the main samples
    # Keep all candidate rows, dedup references
    candidates = df[~is_candidate.copy() | True]  # keep all by default
    references = df[is_candidate.copy() & False]  # empty by default

    # Simpler approach: just dedup the entire frame on the key, keeping first
    available_cols = [c for c in dedup_key if c in df.columns]
    if available_cols:
        df = df.drop_duplicates(subset=available_cols, keep="first")

    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Column normalization
# ---------------------------------------------------------------------------


def _normalize_columns(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Normalize column names across different source manifests."""
    df = df.copy()

    # Map common column aliases
    col_map = {
        "checkpoint_elapsed_min": "checkpoint_min",
        "case_signature": "case_id",
        "network_sha256": "network_sha",
    }
    for old, new in col_map.items():
        if old in df.columns and new not in df.columns:
            df = df.rename(columns={old: new})

    # Add source tag
    df["data_source"] = source

    # Derive state_key
    if "state_key" not in df.columns:
        eid = df.get("event_id", pd.Series("", index=df.index)).astype(str)
        cid = df.get("checkpoint_id", df.get("checkpoint_min", pd.Series("", index=df.index))).astype(str)
        df["state_key"] = eid + "::" + cid

    return df


# ---------------------------------------------------------------------------
# Core builder
# ---------------------------------------------------------------------------


def build_v42_unified_development_pool(
    project_root: Path,
    output_root: Path,
    event_ledger_df: pd.DataFrame | None = None,
) -> UnifiedPoolResult:
    """Build the V4.2 unified development pool from all available data sources.

    Parameters
    ----------
    project_root : Path
        Project root directory.
    output_root : Path
        Output root containing dataset manifests.
    event_ledger_df : DataFrame, optional
        Event usage ledger for split verification.

    Returns
    -------
    UnifiedPoolResult
        Merged pool with compatibility classification and audit info.
    """
    project_root = Path(project_root)
    output_root = Path(output_root)
    warnings: list[str] = []

    # ------------------------------------------------------------------
    # 1. Load data sources
    # ------------------------------------------------------------------
    batches: list[tuple[pd.DataFrame, str]] = []

    # Source A: Train1600
    train_path = _resolve_train1600_manifest(output_root)
    if train_path is not None:
        logger.info("Loading Train1600 manifest from %s", train_path)
        train_df = pd.read_csv(train_path)
        train_df = _normalize_columns(train_df, "train1600")
        batches.append((train_df, "train1600"))
        logger.info("Train1600: %d rows, %d events",
                     len(train_df), train_df["event_id"].nunique())
    else:
        warnings.append("Train1600 manifest not found")

    # Source B: Aug1
    aug1_path = _resolve_aug1_manifest(output_root)
    if aug1_path is not None:
        logger.info("Loading Aug1 manifest from %s", aug1_path)
        aug1_df = pd.read_csv(aug1_path)
        aug1_df = _normalize_columns(aug1_df, "aug1")
        batches.append((aug1_df, "aug1"))
        logger.info("Aug1: %d rows, %d events",
                     len(aug1_df), aug1_df["event_id"].nunique())
    else:
        warnings.append("Aug1 manifest not found")

    # Source C: Calibration/Locked (optional)
    for calib_path in _resolve_calib_manifests(output_root):
        logger.info("Loading calibration manifest from %s", calib_path)
        try:
            calib_df = pd.read_csv(calib_path)
            calib_df = _normalize_columns(calib_df, f"calib_{calib_path.parent.name}")
            batches.append((calib_df, f"calib_{calib_path.parent.name}"))
        except Exception as exc:
            warnings.append(f"Failed to load {calib_path}: {exc}")

    if not batches:
        raise FileNotFoundError("No data source manifests found")

    # ------------------------------------------------------------------
    # 2. Pairwise compatibility validation
    # ------------------------------------------------------------------
    compat_records: list[dict[str, Any]] = []
    for i in range(len(batches)):
        for j in range(i + 1, len(batches)):
            issues = validate_pool_compatibility(batches[i][0], batches[j][0])
            compat_records.append({
                "batch_a": batches[i][1],
                "batch_b": batches[j][1],
                "n_issues": len(issues),
                "issues": "; ".join(issues) if issues else "none",
                "compatible": len(issues) == 0,
            })
            if issues:
                warnings.extend(
                    f"compat {batches[i][1]}<->{batches[j][1]}: {iss}" for iss in issues
                )

    compatibility_report = pd.DataFrame(compat_records)

    # ------------------------------------------------------------------
    # 3. Classify each row's compatibility
    # ------------------------------------------------------------------
    all_dfs = []
    role_records = []
    for df, source in batches:
        compat_labels = df.apply(classify_data_compatibility, axis=1)
        df = df.assign(compatibility_label=compat_labels.values)

        # Filter out incompatible
        n_incompat = (compat_labels == COMPATIBILITY_INCOMPATIBLE).sum()
        if n_incompat > 0:
            warnings.append(f"{source}: {n_incompat} incompatible rows excluded")
            df = df[compat_labels != COMPATIBILITY_INCOMPATIBLE].copy()

        all_dfs.append(df)

        # Build role records
        for _, row in df.iterrows():
            role_records.append({
                "event_id": row.get("event_id", ""),
                "state_key": row.get("state_key", ""),
                "data_source": source,
                "compatibility": row.get("compatibility_label", ""),
            })

    data_role_manifest = pd.DataFrame(role_records)

    # ------------------------------------------------------------------
    # 4. Merge all batches
    # ------------------------------------------------------------------
    merged = pd.concat(all_dfs, ignore_index=True, sort=False)
    logger.info("Merged pool: %d rows from %d sources", len(merged), len(batches))

    # ------------------------------------------------------------------
    # 5. Separate candidates and references, dedup references
    # ------------------------------------------------------------------
    # Identify candidate vs reference rows
    if "anchor_type" in merged.columns:
        is_ref = merged["anchor_type"].astype(str).isin(
            ["no_control", "executable_passive", "internal_rules"]
        )
    elif "v4_data_layer" in merged.columns:
        is_ref = ~merged["v4_data_layer"].astype(str).str.contains("candidate", case=False)
    else:
        is_ref = pd.Series([False] * len(merged), index=merged.index)

    candidate_manifest = merged[~is_ref].copy().reset_index(drop=True)
    reference_raw = merged[is_ref].copy().reset_index(drop=True)
    reference_manifest = dedup_references(reference_raw)

    logger.info(
        "Candidates: %d, References (deduped): %d (from %d raw)",
        len(candidate_manifest), len(reference_manifest), len(reference_raw),
    )

    # ------------------------------------------------------------------
    # 6. Build event and state manifests
    # ------------------------------------------------------------------
    event_ids = sorted(merged["event_id"].dropna().unique())
    event_manifest = pd.DataFrame({
        "event_id": event_ids,
        "n_states": [
            (merged["event_id"] == eid).sum() for eid in event_ids
        ],
        "data_sources": [
            ",".join(sorted(
                merged.loc[merged["event_id"] == eid, "data_source"].unique()
            ))
            for eid in event_ids
        ],
    })

    state_keys = sorted(merged["state_key"].dropna().unique())
    state_manifest = pd.DataFrame({
        "state_key": state_keys,
        "event_id": [sk.split("::")[0] for sk in state_keys],
        "checkpoint_id": [sk.split("::")[1] if "::" in sk else "" for sk in state_keys],
        "n_candidates": [
            (candidate_manifest["state_key"] == sk).sum() for sk in state_keys
        ],
        "n_references": [
            (reference_manifest["state_key"] == sk).sum() for sk in state_keys
        ],
    })

    # ------------------------------------------------------------------
    # 7. Build audit
    # ------------------------------------------------------------------
    # Check for challenge/formal event leakage
    challenge_events = {eid for eid in event_ids if "T100" in str(eid) or "challenge" in str(eid).lower()}
    formal_events = {eid for eid in event_ids if "formal" in str(eid).lower()}
    leaked = challenge_events | formal_events

    audit = {
        "total_events": len(event_ids),
        "total_states": len(state_keys),
        "total_candidates": len(candidate_manifest),
        "total_unique_references": len(reference_manifest),
        "total_merged_rows": len(merged),
        "data_sources": [s for _, s in batches],
        "compatibility_counts": merged["compatibility_label"].value_counts().to_dict()
        if "compatibility_label" in merged.columns
        else {},
        "challenge_formal_leaked": sorted(leaked) if leaked else [],
        "n_warnings": len(warnings),
        "warnings": warnings[:20],
    }

    return UnifiedPoolResult(
        candidate_manifest=candidate_manifest,
        reference_manifest=reference_manifest,
        event_manifest=event_manifest,
        state_manifest=state_manifest,
        compatibility_report=compatibility_report,
        data_role_manifest=data_role_manifest,
        audit=audit,
        warnings=warnings,
    )


# ---------------------------------------------------------------------------
# Output writer
# ---------------------------------------------------------------------------


def write_unified_pool(
    result: UnifiedPoolResult,
    output_dir: Path,
) -> dict[str, str]:
    """Write unified pool outputs to disk.

    Returns dict of written file paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, str] = {}

    # Parquet for large manifests
    for name, df in [
        ("unified_candidate_manifest.parquet", result.candidate_manifest),
        ("unique_reference_manifest.parquet", result.reference_manifest),
    ]:
        path = output_dir / name
        df.to_parquet(path, index=False)
        written[name] = str(path)

    # CSV for human inspection
    for name, df in [
        ("event_manifest.csv", result.event_manifest),
        ("state_manifest.csv", result.state_manifest),
        ("compatibility_report.csv", result.compatibility_report),
        ("data_role_manifest.csv", result.data_role_manifest),
    ]:
        path = output_dir / name
        df.to_csv(path, index=False)
        written[name] = str(path)

    # Audit JSON
    audit_path = output_dir / "development_pool_audit.json"
    audit_path.write_text(
        json.dumps(result.audit, indent=2, default=str, ensure_ascii=False),
        encoding="utf-8",
    )
    written["development_pool_audit.json"] = str(audit_path)

    logger.info("Wrote unified pool: %d files to %s", len(written), output_dir)
    return written


# ---------------------------------------------------------------------------
# Audit function
# ---------------------------------------------------------------------------


def audit_v42_unified_pool(output_root: Path) -> dict[str, Any]:
    """Audit the unified development pool for correctness.

    Checks:
    - Deduplication correctness
    - Schema consistency
    - No event leakage (Challenge/Formal not in pool)
    - Statistics: events, states, candidates, unique NC/DI/Hold

    Returns dict with status and exit_code.
    """
    output_root = Path(output_root)
    pool_dir = output_root / POOL_OUTPUT_DIR

    if not pool_dir.exists():
        return {
            "status": "blocked",
            "exit_code": 2,
            "message": f"Pool directory not found: {pool_dir}",
        }

    errors: list[str] = []

    # Load manifests
    try:
        cand = pd.read_parquet(pool_dir / "unified_candidate_manifest.parquet")
        ref = pd.read_parquet(pool_dir / "unique_reference_manifest.parquet")
        event_m = pd.read_csv(pool_dir / "event_manifest.csv")
        state_m = pd.read_csv(pool_dir / "state_manifest.csv")
    except FileNotFoundError as exc:
        return {
            "status": "blocked",
            "exit_code": 2,
            "message": f"Missing manifest file: {exc}",
        }

    # Check 1: No duplicate candidates
    if "sample_id" in cand.columns:
        n_dup = cand["sample_id"].duplicated().sum()
        if n_dup > 0:
            errors.append(f"{n_dup} duplicate candidate sample_ids")

    # Check 2: No Challenge/Formal leakage
    all_events = set(cand["event_id"].unique()) | set(ref["event_id"].unique())
    challenge_leaked = [e for e in all_events if "T100" in str(e) or "challenge" in str(e).lower()]
    formal_leaked = [e for e in all_events if "formal" in str(e).lower()]
    if challenge_leaked:
        errors.append(f"Challenge events in pool: {challenge_leaked}")
    if formal_leaked:
        errors.append(f"Formal events in pool: {formal_leaked}")

    # Check 3: Schema consistency — all rows have required columns
    required_cols = {"event_id", "state_key", "data_source"}
    missing_cand = required_cols - set(cand.columns)
    missing_ref = required_cols - set(ref.columns)
    if missing_cand:
        errors.append(f"Candidate manifest missing columns: {missing_cand}")
    if missing_ref:
        errors.append(f"Reference manifest missing columns: {missing_ref}")

    # Check 4: State manifest consistency
    expected_states = set(cand["state_key"].unique()) | set(ref["state_key"].unique())
    actual_states = set(state_m["state_key"].unique())
    if expected_states != actual_states:
        errors.append(
            f"State manifest mismatch: {len(expected_states)} expected vs "
            f"{len(actual_states)} actual"
        )

    # Statistics
    stats = {
        "n_events": len(all_events),
        "n_states": len(actual_states),
        "n_candidates": len(cand),
        "n_unique_references": len(ref),
        "n_events_in_manifest": len(event_m),
        "n_states_in_manifest": len(state_m),
    }

    status = "pass" if not errors else "blocked"
    exit_code = 0 if not errors else 2

    return {
        "status": status,
        "exit_code": exit_code,
        "errors": errors,
        "statistics": stats,
        "message": "Pool audit passed" if status == "pass" else f"Audit failed: {'; '.join(errors)}",
    }
