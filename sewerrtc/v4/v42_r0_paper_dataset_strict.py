"""Fail-closed wrapper for the R0 -> formal Step-2 dataset bridge.

The generic materializer already enforces four-reference finite/aligned target
coverage. This wrapper adds the formal Wuhan target-domain invariant so historical
DWF/unknown-domain evidence can never be promoted into the formal Step-2
population by a stale or hand-edited case manifest.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .v42_r0_paper_dataset import R0PaperDatasetResult, build_r0_paper_dataset


def _read(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    return pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)


def build_r0_paper_dataset_strict(
    *,
    project_root: str | Path,
    physical_manifest: str | Path,
    case_manifest: str | Path,
    split_manifest: str | Path,
    output_manifest: str | Path,
    audit_output: str | Path,
) -> R0PaperDatasetResult:
    cases = _read(case_manifest)
    required = {"eligible_formal_all_target", "domain_id", "source_role"}
    missing = required - set(cases.columns)
    if missing:
        raise KeyError(f"strict Step-2 case manifest missing fields: {sorted(missing)}")

    eligible = cases["eligible_formal_all_target"].fillna(False).astype(bool)
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
        admitted["formal_target_domain"].fillna(False).astype(bool).all()
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
