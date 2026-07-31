"""Build task-specific reusable manifests from the project-wide SWMM audit.

The strict :mod:`v42_paper_dataset` builder remains unchanged and continues to
require every formal target.  This module is the complementary *reuse* mode:
it retains real partial hydraulic supervision through explicit availability
masks, never through zero filling or synthetic labels.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from sewerrtc.v4.v42_existing_pool_audit import ReuseClassification


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

    case_keep = cases["classification"].isin(allowed_classes).copy()
    if not include_consumed_development and "source_role" in case_keep.columns:
        case_keep = case_keep[case_keep["source_role"] != "consumed_development"].copy()
    if "source_role" in case_keep.columns:
        case_keep = case_keep[case_keep["source_role"] != "reserved_evaluation"].copy()

    # These masks are evidence authority.  Do not derive a mask from a model
    # expectation or from a target column that has been filled with zeros.
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
        physical_view[target] = physical_view[source].astype(bool)
    physical_view["eligible_dynamics_pretrain"] = (
        physical_view["mask_depth"]
        & physical_view["mask_flood"]
        & physical_view["mask_readback"]
        & physical_view["mask_rainfall"]
        & physical_view["mask_history"]
        & physical_view["mask_horizon"]
    )
    physical_view["eligible_actuator_effect"] = (
        physical_view["mask_facility_flow"]
        & physical_view["mask_readback"]
        & physical_view["mask_history"]
        & physical_view["mask_horizon"]
    )
    physical_view["eligible_storage_supervision"] = (
        physical_view["mask_storage"]
        & physical_view["mask_history"]
        & physical_view["mask_horizon"]
    )
    physical_view["eligible_outfall_supervision"] = physical_view["mask_outfall_flow"]
    physical_view["formal_complete_branch"] = (
        physical_view["eligible_dynamics_pretrain"]
        & physical_view["mask_storage"]
        & physical_view["mask_facility_flow"]
        & physical_view["mask_outfall_flow"]
    )

    # Preserve every physical branch with at least one legitimate task.  This is
    # intentionally broader than the strict all-target dataset.
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

    # Never claim that an outfall label exists merely because an incoming-link
    # reconstruction is structurally possible.
    if "available_outfall_reconstruction_candidate" in physical_view.columns:
        invalid_promotion = physical_view["available_outfall_reconstruction_candidate"].astype(bool) & physical_view["mask_outfall_flow"]
        # Explicit outfall columns can coexist with incoming-link candidates on
        # new recorder runs; that is legitimate.  The guard here is therefore
        # informational rather than destructive.
        physical_view["outfall_reconstruction_candidate"] = physical_view["available_outfall_reconstruction_candidate"].astype(bool)
        physical_view["outfall_requires_validation_before_reconstruction"] = (
            physical_view["outfall_reconstruction_candidate"] & ~physical_view["mask_outfall_flow"]
        )
    else:
        physical_view["outfall_reconstruction_candidate"] = False
        physical_view["outfall_requires_validation_before_reconstruction"] = False

    # Case-level task eligibility remains separate from branch-level masks.
    case_keep["eligible_counterfactual_flood"] = (
        case_keep.get("four_reference_complete", False).astype(bool)
        & case_keep.get("core_trajectory_targets", False).astype(bool)
    )
    case_keep["eligible_formal_all_target"] = (
        case_keep.get("four_reference_complete", False).astype(bool)
        & case_keep.get("full_reuse_targets", False).astype(bool)
    )
    case_keep["eligible_target_no_dwf"] = case_keep.get("domain_id", "").astype(str).str.startswith("target_no_dwf")
    case_keep["eligible_source_domain"] = case_keep.get("domain_id", "").astype(str).str.startswith("source_")

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
