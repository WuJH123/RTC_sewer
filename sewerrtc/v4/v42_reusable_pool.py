"""Build task-specific reusable manifests from the project-wide SWMM audit.

The strict :mod:`v42_paper_dataset` builder remains unchanged and continues to
require every formal target.  This module is the complementary *reuse* mode:
it retains real partial hydraulic supervision through explicit availability
masks, never through zero filling or synthetic labels.

Whole-event legacy details (for example historical PFV-first/closed-loop files)
may not carry a checkpoint manifest.  They are not discarded solely for that
reason: if their timestamps contain at least one complete 60-min history plus
H120 future window, they can contribute to generic hydraulic/action-effect
pretraining, while remaining ineligible for same-state four-reference formal
counterfactual supervision until the checkpoint/reference lineage is proven.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sewerrtc.v4.v42_existing_pool_audit import ReuseClassification
from sewerrtc.v4.v42_trajectory_builder import (
    HISTORY_INTERVAL_MIN,
    HORIZON_INTERVAL_MIN,
    N_HISTORY_FRAMES,
    N_HORIZON_STEPS,
)


TIME_ATOL_MIN = 1.0e-6


@dataclass(frozen=True)
class ReusablePoolResult:
    physical_manifest_path: Path
    case_manifest_path: Path
    audit_path: Path
    physical_row_count: int
    case_row_count: int


def _read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _write_table(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def _bool_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)
    return df[column].fillna(False).astype(bool)


def _string_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series("", index=df.index, dtype=str)
    return df[column].fillna("").astype(str)


def _window_anchor_count(detail_path: str | Path) -> int:
    """Count valid centers with 13x5-min history and 12x10-min future.

    This enables whole-event historical trajectories to be reused for generic
    dynamics pretraining without inventing a formal counterfactual checkpoint.
    """
    path = Path(detail_path)
    if not path.exists():
        return 0
    header = pd.read_csv(path, nrows=0)
    if "elapsed_min" not in header.columns:
        return 0
    elapsed = pd.to_numeric(
        pd.read_csv(path, usecols=["elapsed_min"])["elapsed_min"], errors="coerce"
    ).to_numpy(float)
    if not np.isfinite(elapsed).all() or len(elapsed) == 0:
        return 0
    values = np.unique(elapsed)

    def has_time(value: float) -> bool:
        return bool(np.any(np.isclose(values, value, atol=TIME_ATOL_MIN, rtol=0.0)))

    count = 0
    for center in values:
        history = [
            center - (N_HISTORY_FRAMES - 1 - i) * HISTORY_INTERVAL_MIN
            for i in range(N_HISTORY_FRAMES)
        ]
        future = [center + (i + 1) * HORIZON_INTERVAL_MIN for i in range(N_HORIZON_STEPS)]
        if all(has_time(float(x)) for x in history) and all(has_time(float(x)) for x in future):
            count += 1
    return count


def build_reusable_paper_pool(
    *,
    physical_inventory: str | Path,
    case_inventory: str | Path,
    output_physical_manifest: str | Path,
    output_case_manifest: str | Path,
    audit_output: str | Path,
    include_source_domain: bool = True,
    include_consumed_development: bool = True,
) -> ReusablePoolResult:
    """Create masked task views without pretending that missing targets exist.

    ``physical_inventory`` must come from :func:`audit_existing_swmm_pool`.
    Every target mask is copied from the evidence audit.  No target tensor or
    label is materialised here, so there is no opportunity to silently fill a
    missing variable with zero.
    """
    physical_path = Path(physical_inventory)
    case_path = Path(case_inventory)
    physical = _read_table(physical_path)
    cases = _read_table(case_path)
    if physical.empty:
        raise ValueError("physical inventory is empty")
    if cases.empty:
        raise ValueError("case inventory is empty")

    allowed_classes = {
        ReuseClassification.FULL_REUSE.value,
        ReuseClassification.REUSE_AFTER_EXTRACTION.value,
        ReuseClassification.PARTIAL_AUX_REUSE.value,
    }
    if include_source_domain:
        allowed_classes.add(ReuseClassification.SOURCE_DOMAIN_REUSE.value)

    case_keep = cases[cases["classification"].isin(allowed_classes)].copy()
    if not include_consumed_development and "source_role" in case_keep.columns:
        case_keep = case_keep[case_keep["source_role"] != "consumed_development"].copy()
    if "source_role" in case_keep.columns:
        case_keep = case_keep[case_keep["source_role"] != "reserved_evaluation"].copy()

    mask_sources = {
        "mask_depth": "available_node_depth",
        "mask_flood": "available_node_flooding_rate",
        "mask_storage": "available_storage_volume",
        "mask_facility_flow": "available_managed_facility_flow",
        "mask_outfall_flow": "available_outfall_flow",
        "mask_readback": "available_readback_setting",
        "mask_rainfall": "available_rainfall",
        "mask_history": "available_history_complete",
        "mask_horizon": "available_horizon_complete",
    }
    missing_columns = [source for source in mask_sources.values() if source not in physical.columns]
    if missing_columns:
        raise KeyError(f"physical inventory missing availability columns: {missing_columns}")

    physical_view = physical.copy()
    for target, source in mask_sources.items():
        physical_view[target] = _bool_series(physical_view, source)

    # Formal same-state rows already have a checkpoint-specific complete window.
    # Legacy whole-event details may still expose many valid windows; count them
    # once here so they can be used for generic dynamics pretraining.
    anchor_counts: list[int] = []
    for row in physical_view.itertuples(index=False):
        if bool(getattr(row, "mask_history")) and bool(getattr(row, "mask_horizon")):
            anchor_counts.append(1)
        else:
            anchor_counts.append(_window_anchor_count(str(getattr(row, "detail_path"))))
    physical_view["window_anchor_count"] = anchor_counts
    physical_view["windowable_13x12"] = physical_view["window_anchor_count"].astype(int) > 0

    physical_view["eligible_dynamics_pretrain"] = (
        physical_view["mask_depth"]
        & physical_view["mask_flood"]
        & physical_view["mask_readback"]
        & physical_view["mask_rainfall"]
        & physical_view["windowable_13x12"]
    )
    physical_view["eligible_actuator_effect"] = (
        physical_view["mask_facility_flow"]
        & physical_view["mask_readback"]
        & physical_view["windowable_13x12"]
    )
    physical_view["eligible_storage_supervision"] = (
        physical_view["mask_storage"] & physical_view["windowable_13x12"]
    )
    physical_view["eligible_outfall_supervision"] = physical_view["mask_outfall_flow"]
    physical_view["formal_complete_branch"] = (
        physical_view["mask_history"]
        & physical_view["mask_horizon"]
        & physical_view["eligible_dynamics_pretrain"]
        & physical_view["mask_storage"]
        & physical_view["mask_facility_flow"]
        & physical_view["mask_outfall_flow"]
    )

    physical_view = physical_view[
        physical_view[
            [
                "eligible_dynamics_pretrain",
                "eligible_actuator_effect",
                "eligible_storage_supervision",
                "eligible_outfall_supervision",
            ]
        ].any(axis=1)
    ].copy()
    if "source_role" in physical_view.columns:
        physical_view = physical_view[physical_view["source_role"] != "reserved_evaluation"].copy()
        if not include_consumed_development:
            physical_view = physical_view[physical_view["source_role"] != "consumed_development"].copy()
    if not include_source_domain and "domain_id" in physical_view.columns:
        physical_view = physical_view[~physical_view["domain_id"].astype(str).str.startswith("source_")].copy()

    if "available_outfall_reconstruction_candidate" in physical_view.columns:
        physical_view["outfall_reconstruction_candidate"] = _bool_series(
            physical_view, "available_outfall_reconstruction_candidate"
        )
        physical_view["outfall_requires_validation_before_reconstruction"] = (
            physical_view["outfall_reconstruction_candidate"] & ~physical_view["mask_outfall_flow"]
        )
    else:
        physical_view["outfall_reconstruction_candidate"] = False
        physical_view["outfall_requires_validation_before_reconstruction"] = False

    four = _bool_series(case_keep, "four_reference_complete")
    core = _bool_series(case_keep, "core_trajectory_targets")
    full = _bool_series(case_keep, "full_reuse_targets")
    case_keep["eligible_counterfactual_flood"] = four & core
    case_keep["eligible_formal_all_target"] = four & full
    case_keep["eligible_target_no_dwf"] = _string_series(case_keep, "domain_id").str.startswith("target_no_dwf")
    case_keep["eligible_source_domain"] = _string_series(case_keep, "domain_id").str.startswith("source_")

    output_physical_manifest = Path(output_physical_manifest)
    output_case_manifest = Path(output_case_manifest)
    audit_output = Path(audit_output)
    _write_table(physical_view, output_physical_manifest)
    _write_table(case_keep, output_case_manifest)

    task_counts = {
        "physical_rows": int(len(physical_view)),
        "case_rows": int(len(case_keep)),
        "dynamics_pretrain_physical_runs": int(physical_view["eligible_dynamics_pretrain"].sum()),
        "actuator_effect_physical_runs": int(physical_view["eligible_actuator_effect"].sum()),
        "storage_supervision_physical_runs": int(physical_view["eligible_storage_supervision"].sum()),
        "explicit_outfall_supervision_physical_runs": int(physical_view["eligible_outfall_supervision"].sum()),
        "windowable_legacy_physical_runs": int((physical_view["window_anchor_count"] > 1).sum()),
        "counterfactual_flood_cases": int(case_keep["eligible_counterfactual_flood"].sum()),
        "formal_all_target_cases": int(case_keep["eligible_formal_all_target"].sum()),
        "target_no_dwf_cases": int(case_keep["eligible_target_no_dwf"].sum()),
        "source_domain_cases": int(case_keep["eligible_source_domain"].sum()),
    }
    audit: dict[str, Any] = {
        "contract": "PROJECT6_V42_REUSABLE_POOL_V1",
        "physical_inventory": str(physical_path),
        "case_inventory": str(case_path),
        "missing_targets_are_imputed": False,
        "formal_strict_builder_preserved": True,
        "availability_mask_authority": "existing_pool_audit_only",
        "legacy_window_policy": "whole-event detail may contribute only to generic pretraining when a real 13x5 + 12x10 window exists",
        "task_counts": task_counts,
        "outfall_policy": (
            "explicit labels train outfall head; incoming-link reconstruction remains disabled "
            "until independently validated"
        ),
    }
    audit_output.parent.mkdir(parents=True, exist_ok=True)
    audit_output.write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")
    return ReusablePoolResult(
        physical_manifest_path=output_physical_manifest,
        case_manifest_path=output_case_manifest,
        audit_path=audit_output,
        physical_row_count=int(len(physical_view)),
        case_row_count=int(len(case_keep)),
    )
