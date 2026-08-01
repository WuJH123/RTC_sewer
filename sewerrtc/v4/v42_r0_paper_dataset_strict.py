"""Fail-closed wrapper for the R0 -> formal Step-2 dataset bridge.

The generic materializer already enforces four-reference finite/aligned target
coverage. This wrapper adds the formal Wuhan target-domain invariant so historical
DWF/unknown-domain evidence can never be promoted into the formal Step-2
population by a stale or hand-edited case manifest.

Formal manifests are required to be Parquet so persisted boolean dtypes remain
unambiguous end-to-end. CSV remains acceptable for human-readable audit exports,
but not as a formal training-admission transport.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .v42_r0_paper_dataset import R0PaperDatasetResult, build_r0_paper_dataset


def _read(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    return pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)


def _bool_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise KeyError(column)
    series = frame[column]
    if pd.api.types.is_bool_dtype(series.dtype):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series.dtype):
        return series.fillna(0).astype(float).ne(0.0)
    text = series.fillna("").astype(str).str.strip().str.casefold()
    true_values = {"true", "1", "yes", "y", "t"}
    false_values = {"false", "0", "no", "n", "f", "", "none", "nan"}
    unknown = sorted(set(text.unique()) - true_values - false_values)
    if unknown:
        raise ValueError(
            f"formal Step-2 boolean column {column!r} has unsupported values: {unknown[:10]}"
        )
    return text.isin(true_values)


def _require_typed_parquet(name: str, value: str | Path) -> Path:
    path = Path(value)
    if path.suffix.lower() != ".parquet":
        raise ValueError(
            f"formal Step-2 {name} must be a typed Parquet manifest, got {path}; "
            "CSV audit exports cannot authorize formal training"
        )
    return path


def build_r0_paper_dataset_strict(
    *,
    project_root: str | Path,
    physical_manifest: str | Path,
    case_manifest: str | Path,
    split_manifest: str | Path,
    output_manifest: str | Path,
    audit_output: str | Path,
) -> R0PaperDatasetResult:
    physical_manifest = _require_typed_parquet("physical_manifest", physical_manifest)
    case_manifest = _require_typed_parquet("case_manifest", case_manifest)
    split_manifest = _require_typed_parquet("split_manifest", split_manifest)
    cases = _read(case_manifest)
    required = {"eligible_formal_all_target", "domain_id", "source_role"}
    missing = required - set(cases.columns)
    if missing:
        raise KeyError(f"strict Step-2 case manifest missing fields: {sorted(missing)}")

    eligible = _bool_series(cases, "eligible_formal_all_target")
    admitted = cases[eligible].copy()
    if admitted.empty:
        raise ValueError("strict R0 has no formal Step-2 cases")
    bad_domain = ~admitted["domain_id"].fillna("").astype(str).str.startswith("target_no_dwf")
    if bool(bad_domain.any()):
        examples = admitted.loc[bad_domain, ["case_uid", "domain_id"]].head(10).to_dict("records")
        raise RuntimeError(
            "formal Step-2 admission contains source/unknown-domain cases; rebuild "
            f"the strict R0 reusable pool. examples={examples}"
        )
    reserved = admitted["source_role"].fillna("").astype(str).eq("reserved_evaluation")
    if bool(reserved.any()):
        raise RuntimeError("formal Step-2 admission contains reserved evaluation cases")
    if "formal_target_domain" in admitted.columns and not bool(
        _bool_series(admitted, "formal_target_domain").all()
    ):
        raise RuntimeError("formal_target_domain evidence contradicts domain_id")

    return build_r0_paper_dataset(
        project_root=project_root,
        physical_manifest=physical_manifest,
        case_manifest=case_manifest,
        split_manifest=split_manifest,
        output_manifest=output_manifest,
        audit_output=audit_output,
    )
