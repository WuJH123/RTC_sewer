"""Build strict target-masked reusable V4.2 task manifests.

Formal counterfactual admission is target-domain only. Historical source-domain
(DWF/unknown-domain) trajectories may still contribute auxiliary dynamics or
representation learning, but they cannot silently become the formal Wuhan
rainfall-only four-reference population.
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


def _scalar_bool(value: Any, *, name: str) -> bool:
    """Parse a persisted boolean scalar without Python truthiness surprises."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return float(value) != 0.0
    text = str(value).strip().casefold()
    if text in {"true", "1", "yes", "y", "t"}:
        return True
    if text in {"false", "0", "no", "n", "f", "", "none", "nan"}:
        return False
    raise ValueError(f"boolean scalar {name!r} has unsupported value: {value!r}")


def _bool(frame: pd.DataFrame, col: str) -> pd.Series:
    if col not in frame.columns:
        return pd.Series(False, index=frame.index, dtype=bool)
    series = frame[col]
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series.dtype):
        return series.fillna(0).astype(float).ne(0.0)
    text = series.fillna("").astype(str).str.strip().str.casefold()
    true_values = {"true", "1", "yes", "y", "t"}
    false_values = {"false", "0", "no", "n", "f", "", "none", "nan"}
    unknown = sorted(set(text.unique()) - true_values - false_values)
    if unknown:
        raise ValueError(f"boolean column {col!r} has unsupported values: {unknown[:10]}")
    return text.isin(true_values)


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
                if _scalar_bool(
                    getattr(r, "available_finite_checked", False),
                    name="available_finite_checked",
                )
                and _scalar_bool(
                    getattr(r, "available_finite_pass", False),
                    name="available_finite_pass",
                )
            ]
            if not finite_rows:
                all_roles_finite = False
                all_roles_formal = False
                continue
            if not any(
                _scalar_bool(
                    getattr(r, "formal_all_target_complete", False),
                    name="formal_all_target_complete",
                )
                for r in finite_rows
            ):
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
    """Build reusable views and enforce finite, causal, domain-safe admission."""
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
    # ``build_reusable_paper_pool`` can preserve both the inventory and
    # alignment copies of this field; pandas names them *_x/*_y on merge.
    # Prefer the alignment copy and fall back to the canonical/left copy.
    forcing = (
        _bool(cases, "same_forcing_pass_y")
        if "same_forcing_pass_y" in cases.columns
        else _bool(cases, "same_forcing_pass")
    )
    aligned = _bool(cases, "same_state_numeric_pass") & forcing
    four = _bool(cases, "four_reference_complete")
    core = _bool(cases, "core_trajectory_targets")
    full = _bool(cases, "full_reuse_targets")
    domain = cases.get("domain_id", pd.Series("", index=cases.index)).fillna("").astype(str)
    target_no_dwf = domain.str.startswith("target_no_dwf")
    source_domain = domain.str.startswith("source_")

    formal_cf_base = four & aligned & four_branch_finite
    cases["eligible_counterfactual_flood"] = (
        formal_cf_base & core & target_no_dwf
    )
    cases["eligible_formal_all_target"] = (
        formal_cf_base & full & four_branch_formal & target_no_dwf
    )
    cases["eligible_source_domain_counterfactual_aux"] = (
        formal_cf_base & core & source_domain
    )
    cases["formal_target_domain"] = target_no_dwf

    _write(physical, output_physical_manifest)
    _write(cases, output_case_manifest)

    audit_path = Path(audit_output)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit["strict_scientific_admission"] = True
    audit["counterfactual_requires_all_four_roles_finite"] = True
    audit["task_labels_require_common_causal_context"] = True
    audit["formal_counterfactual_requires_target_no_dwf"] = True
    audit["source_domain_formal_admission_forbidden"] = True
    audit["persisted_boolean_parsing_fail_closed"] = True
    audit["task_counts"] = {
        "physical_rows": int(len(physical)),
        "case_rows": int(len(cases)),
        "dynamics_pretrain_physical_runs": int(physical["eligible_dynamics_pretrain"].sum()),
        "actuator_effect_physical_runs": int(physical["eligible_actuator_effect"].sum()),
        "storage_supervision_physical_runs": int(physical["eligible_storage_supervision"].sum()),
        "explicit_outfall_supervision_physical_runs": int(physical["eligible_outfall_supervision"].sum()),
        "counterfactual_flood_cases": int(cases["eligible_counterfactual_flood"].sum()),
        "formal_all_target_cases": int(cases["eligible_formal_all_target"].sum()),
        "source_domain_counterfactual_aux_cases": int(cases["eligible_source_domain_counterfactual_aux"].sum()),
        "formal_target_domain_cases": int(target_no_dwf.sum()),
    }
    audit_path.write_text(
        json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8"
    )

    return ReusablePoolResult(
        physical_manifest_path=Path(output_physical_manifest),
        case_manifest_path=Path(output_case_manifest),
        audit_path=audit_path,
        physical_row_count=int(len(physical)),
        case_row_count=int(len(cases)),
    )
