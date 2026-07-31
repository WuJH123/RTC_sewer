"""Strict preparation of the V4.2 fast-track evidence core."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from sewerrtc.v4.v42_case_alignment_audit import audit_case_alignment
from sewerrtc.v4.v42_fasttrack import (
    CONTRACT_ID,
    FORMAL_CONTRACT_ID,
    _json_ids,
    _read_table,
    _write_table,
    select_fasttrack_core,
    targeted_finite_audit,
)
from sewerrtc.v4.v42_reusable_pool import build_reusable_paper_pool


def prepare_fasttrack_core_strict(
    *,
    project_root: str | Path,
    r01_audit_dir: str | Path,
    output_dir: str | Path,
    max_events: int = 16,
    cases_per_event: int = 3,
    seed: int = 42,
    min_events: int = 8,
    min_aligned_cases: int = 12,
    min_finite_fraction: float = 0.95,
) -> dict[str, Any]:
    """Prepare a development core where every admitted case has four finite branches."""
    r01_audit_dir = Path(r01_audit_dir)
    output_dir = Path(output_dir)
    core = select_fasttrack_core(
        physical_inventory=r01_audit_dir / "physical_run_inventory.parquet",
        case_inventory=r01_audit_dir / "target_coverage_by_case.csv",
        output_dir=output_dir,
        max_events=max_events,
        cases_per_event=cases_per_event,
        seed=seed,
    )
    finite = targeted_finite_audit(
        project_root=project_root,
        physical_manifest=core.physical_manifest,
    )
    finite_by_id = {
        str(row.physical_identity_sha256): bool(row.available_finite_pass)
        for row in finite.itertuples(index=False)
    }

    cases = _read_table(core.case_manifest).copy()
    cases["all_branches_finite"] = [
        bool(ids) and all(finite_by_id.get(pid, False) for pid in ids)
        for ids in (_json_ids(value) for value in cases["branch_physical_ids"])
    ]
    cases = cases[cases["all_branches_finite"]].reset_index(drop=True)
    _write_table(cases, core.case_manifest)

    alignment_path = output_dir / "case_alignment_audit.csv"
    if cases.empty:
        alignment = pd.DataFrame(
            columns=["case_uid", "same_state_numeric_pass", "same_forcing_pass", "error"]
        )
        alignment.to_csv(alignment_path, index=False)
        aligned = 0
    else:
        alignment = audit_case_alignment(
            project_root=project_root,
            physical_inventory=core.physical_manifest,
            case_inventory=core.case_manifest,
            output_path=alignment_path,
        )
        aligned = int(
            (
                alignment["same_state_numeric_pass"].fillna(False).astype(bool)
                & alignment["same_forcing_pass"].fillna(False).astype(bool)
            ).sum()
        )

    reusable_physical = output_dir / "reusable_pool_manifest.parquet"
    reusable_cases = output_dir / "reusable_case_manifest.parquet"
    reusable_summary = output_dir / "reusable_pool_summary.json"
    reusable_physical_rows = 0
    reusable_case_rows = 0
    if not cases.empty:
        reusable = build_reusable_paper_pool(
            physical_inventory=core.physical_manifest,
            case_inventory=core.case_manifest,
            alignment_inventory=alignment_path,
            output_physical_manifest=reusable_physical,
            output_case_manifest=reusable_cases,
            audit_output=reusable_summary,
            include_source_domain=False,
            include_consumed_development=True,
            require_finite_audit=True,
        )
        reusable_physical_rows = int(reusable.physical_row_count)
        reusable_case_rows = int(reusable.case_row_count)

    finite_fraction = float(finite["available_finite_pass"].mean()) if len(finite) else 0.0
    finite_event_count = int(cases["fasttrack_group"].nunique()) if not cases.empty else 0
    passed = bool(
        finite_event_count >= int(min_events)
        and aligned >= int(min_aligned_cases)
        and finite_fraction >= float(min_finite_fraction)
    )
    evidence = {
        "contract_id": CONTRACT_ID,
        "formal_contract_id": FORMAL_CONTRACT_ID,
        "stage": "core_pool",
        "status": "pass" if passed else "fail",
        "development_only": True,
        "formal_authorization": False,
        "metrics": {
            "independent_rainfall_groups": finite_event_count,
            "selected_cases_before_finite_filter": int(core.selected_cases),
            "selected_cases": int(len(cases)),
            "selected_physical_runs": int(core.selected_physical_runs),
            "finite_pass_fraction": finite_fraction,
            "all_branch_finite_cases": int(len(cases)),
            "aligned_cases": aligned,
            "reusable_physical_rows": reusable_physical_rows,
            "reusable_case_rows": reusable_case_rows,
        },
    }
    (output_dir / "evidence.json").write_text(
        json.dumps(evidence, indent=2, allow_nan=False), encoding="utf-8"
    )
    return evidence
