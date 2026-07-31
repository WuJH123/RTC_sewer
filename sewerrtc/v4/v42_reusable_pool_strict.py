"""Strict post-processing for the V4.2 reusable evidence pool.

The generic reuse builder intentionally supports partial supervision, but formal
counterfactual eligibility must additionally prove that each required reference
role resolves to a finite physical trajectory.  Task-specific labels also need
the same causal model context (depth/flood/readback/rainfall/window) rather than
being counted as trainable merely because an isolated target column exists.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .v42_reusable_pool import ReusablePoolResult, build_reusable_paper_pool


FOUR_ROLES = ("candidate", "no_control", "dynamic_internal", "hold_previous")


def _read(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    return pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)


def _write(frame: pd.DataFrame, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".parquet":
        frame.to_parquet(p, index=False)
    else:
        frame.to_csv(p, index=False)


def _bool(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[col].fillna(False).astype(bool)


def _ids(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    parsed = json.loads(str(value))
    return [str(x) for x in parsed]


def _strict_case_finite(
    cases: pd.DataFrame,
    original_physical: pd.DataFrame,
) -> tuple[pd.Series, pd.Series]:
    by_id = {
        str(row.physical_identity_sha256): row
        for row in original_physical.itertuples(index=False)
    }
    finite_ok: list[bool] = []
    formal_branch_ok: list[bool] = []
    for case in cases.itertuples(index=False):
        ids = _ids(getattr(case, "branch_physical_ids", "[]"))
        role_rows: dict[str, list[Any]] = {role: [] for role in FOUR_ROLES}
        for pid in ids:
            row = by_id.get(pid)
            if row is None:
                continue
            role = str(getattr(row, "branch_role", ""))
            if role in role_rows:
                role_rows[role].append(row)

        all_roles_finite = True
        all_roles_formal = True
        for role in FOUR_ROLES:
            rows = role_rows[role]
            finite_rows = [
                r
                for r in rows
                if bool(getattr(r, "available_finite_checked", False))
                and bool(getattr(r, "available_finite_pass", False))
            ]
            if not finite_rows:
                all_roles_finite = False
                all_roles_formal = False
                continue
            if not any(bool(getattr(r, "formal_all_target_complete", False)) for r in finite_rows):
                all_roles_formal = False
        finite_ok.append(all_roles_finite)
        formal_branch_ok.append(all_roles_formal)
    return (
        pd.Series(finite_ok, index=cases.index, dtype=bool),
        pd.Series(formal_branch_ok, index=cases.index, dtype=bool),
    )


def build_reusable_paper_pool_strict(
    *,
    physical_inventory: str | Path,
    case_inventory: str | Path,
    output_physical_manifest: str | Path,
    output_case_manifest: str | Path,
    audit_output: str | Path,
    alignment_inventory: str | Path | None = None,
    include_source_domain: bool = True,
    include_consumed_development: bool = True,
    require_finite_audit: bool = True,
) -> ReusablePoolResult:
    """Build reusable views and then enforce finite, causal task admission."""
    original_physical = _read(physical_inventory)
    result = build_reusable_paper_pool(
        physical_inventory=physical_inventory,
        case_inventory=case_inventory,
        output_physical_manifest=output_physical_manifest,
        output_case_manifest=output_case_manifest,
        audit_output=audit_output,
        alignment_inventory=alignment_inventory,
        include_source_domain=include_source_domain,
        include_consumed_development=include_consumed_development,
        require_finite_audit=require_finite_audit,
    )
    physical = _read(output_physical_manifest)
    cases = _read(output_case_manifest)

    # One common causal-context gate feeds all trajectory-supervision tasks.
    causal_context = (
        _bool(physical, "mask_depth")
        & _bool(physical, "mask_flood")
        & _bool(physical, "mask_readback")
        & _bool(physical, "mask_rainfall")
        & _bool(physical, "windowable_13x12")
        & _bool(physical, "mask_finite")
    )
    physical["eligible_dynamics_pretrain"] = causal_context
    physical["eligible_actuator_effect"] = causal_context & _bool(
        physical, "mask_facility_flow"
    )
    physical["eligible_storage_supervision"] = causal_context & _bool(
        physical, "mask_storage"
    )
    physical["eligible_outfall_supervision"] = causal_context & _bool(
        physical, "mask_outfall_flow"
    )
    physical["formal_complete_branch"] = (
        causal_context
        & _bool(physical, "mask_history")
        & _bool(physical, "mask_horizon")
        & _bool(physical, "mask_storage")
        & _bool(physical, "mask_facility_flow")
        & _bool(physical, "mask_outfall_flow")
    )

    four_branch_finite, four_branch_formal = _strict_case_finite(
        cases, original_physical
    )
    cases["four_reference_finite_pass"] = four_branch_finite
    cases["four_reference_formal_branch_pass"] = four_branch_formal
    aligned = _bool(cases, "same_state_numeric_pass") & _bool(
        cases, "same_forcing_pass"
    )
    four = _bool(cases, "four_reference_complete")
    core = _bool(cases, "core_trajectory_targets")
    full = _bool(cases, "full_reuse_targets")
    cases["eligible_counterfactual_flood"] = (
        four & core & aligned & four_branch_finite
    )
    cases["eligible_formal_all_target"] = (
        four & full & aligned & four_branch_finite & four_branch_formal
    )

    _write(physical, output_physical_manifest)
    _write(cases, output_case_manifest)

    audit_path = Path(audit_output)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["strict_scientific_admission"] = True
    audit["counterfactual_requires_all_four_roles_finite"] = True
    audit["task_labels_require_common_causal_context"] = True
    audit["task_counts"] = {
        "physical_rows": int(len(physical)),
        "case_rows": int(len(cases)),
        "dynamics_pretrain_physical_runs": int(physical["eligible_dynamics_pretrain"].sum()),
        "actuator_effect_physical_runs": int(physical["eligible_actuator_effect"].sum()),
        "storage_supervision_physical_runs": int(physical["eligible_storage_supervision"].sum()),
        "explicit_outfall_supervision_physical_runs": int(physical["eligible_outfall_supervision"].sum()),
        "counterfactual_flood_cases": int(cases["eligible_counterfactual_flood"].sum()),
        "formal_all_target_cases": int(cases["eligible_formal_all_target"].sum()),
    }
    audit_path.write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")

    return ReusablePoolResult(
        physical_manifest_path=Path(output_physical_manifest),
        case_manifest_path=Path(output_case_manifest),
        audit_path=audit_path,
        physical_row_count=int(len(physical)),
        case_row_count=int(len(cases)),
    )
