"""Raw Independent Oracle for the project-wide R0-derived Step-2 population.

Unlike the legacy oracle resolver, this implementation does not rediscover rows
under Train1600.  It validates the exact authoritative raw detail paths recorded
by :mod:`v42_r0_paper_dataset`, preserving one population lineage from R0 to
formal training admission.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .v42_independent_oracle import (
    _load_graph_node_ids,
    _priority_indices,
    _raw_branch_kpis,
    _format_kpis,
)


def _read(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    return pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)


def audit_r0_manifest_raw(
    *,
    project_root: str | Path,
    manifest_path: str | Path,
    atol_pfv_m3: float = 1.0e-3,
    atol_tfv_m3: float = 1.0e-3,
    atol_peak_m3s: float = 1.0e-6,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    df = _read(manifest_path)
    if df.empty:
        raise ValueError("R0 paper manifest is empty")
    required = {
        "case_uid",
        "checkpoint_min",
        "source_detail_path_candidate",
        "source_detail_path_no_control",
        "source_detail_path_dynamic_internal",
        "pfv_delta",
        "tfv_delta",
        "peak_delta",
    }
    missing = required - set(df.columns)
    if missing:
        raise KeyError(f"R0 paper manifest missing columns: {sorted(missing)}")

    node_ids = _load_graph_node_ids(Path(project_root))
    priority_idx = _priority_indices(node_ids)
    rows: list[dict[str, Any]] = []
    failures = 0
    for idx, row in df.iterrows():
        rec: dict[str, Any] = {
            "row_index": int(idx),
            "case_uid": str(row.get("case_uid", "")),
            "case_id": str(row.get("case_id", "")),
            "event_id": str(row.get("event_id", "")),
            "audit_mode": "raw",
        }
        try:
            checkpoint = float(row["checkpoint_min"])
            pfv_c, tfv_c, peak_c, n_c = _raw_branch_kpis(
                Path(str(row["source_detail_path_candidate"])),
                checkpoint,
                node_ids,
                priority_idx,
            )
            pfv_nc, _, _, n_nc = _raw_branch_kpis(
                Path(str(row["source_detail_path_no_control"])),
                checkpoint,
                node_ids,
                priority_idx,
            )
            _, tfv_di, peak_di, n_di = _raw_branch_kpis(
                Path(str(row["source_detail_path_dynamic_internal"])),
                checkpoint,
                node_ids,
                priority_idx,
            )
            rec.update(_format_kpis(pfv_c, pfv_nc, tfv_c, tfv_di, peak_c, peak_di))
            rec.update(
                raw_candidate_rows=n_c,
                raw_no_control_rows=n_nc,
                raw_dynamic_internal_rows=n_di,
            )
            stored_pfv = float(row["pfv_delta"])
            stored_tfv = float(row["tfv_delta"])
            stored_peak = float(row["peak_delta"])
            rec["stored_pfv_delta_m3"] = stored_pfv
            rec["stored_tfv_delta_m3"] = stored_tfv
            rec["stored_peak_delta_m3s"] = stored_peak
            rec["pfv_abs_error_m3"] = abs(stored_pfv - rec["recomputed_pfv_delta_m3"])
            rec["tfv_abs_error_m3"] = abs(stored_tfv - rec["recomputed_tfv_delta_m3"])
            rec["peak_abs_error_m3s"] = abs(stored_peak - rec["recomputed_peak_delta_m3s"])
            rec["pfv_pass"] = rec["pfv_abs_error_m3"] <= atol_pfv_m3
            rec["tfv_pass"] = rec["tfv_abs_error_m3"] <= atol_tfv_m3
            rec["peak_pass"] = rec["peak_abs_error_m3s"] <= atol_peak_m3s
            rec["row_pass"] = bool(rec["pfv_pass"] and rec["tfv_pass"] and rec["peak_pass"])
        except Exception as exc:
            rec["row_pass"] = False
            rec["error"] = f"{type(exc).__name__}: {exc}"
        failures += int(not rec["row_pass"])
        rows.append(rec)

    lineages = []
    if "sample_lineage_sha256" in df.columns:
        lineages = [str(x) for x in df["sample_lineage_sha256"].dropna().unique() if str(x)]
    summary = {
        "contract_id": "PROJECT6_V42_PAPER_WORKFLOW_V1",
        "manifest": str(manifest_path),
        "audit_mode": "raw",
        "row_count": int(len(df)),
        "pass_count": int(len(df) - failures),
        "fail_count": int(failures),
        "all_pass": failures == 0,
        "expected_count": None,
        "sample_lineage_sha256": lineages[0] if len(lineages) == 1 else "",
        "population_lineage_unique": len(lineages) <= 1,
        "tolerances": {
            "pfv_m3": atol_pfv_m3,
            "tfv_m3": atol_tfv_m3,
            "peak_m3s": atol_peak_m3s,
        },
    }
    if len(lineages) > 1:
        summary["all_pass"] = False
        summary["fail_count"] = int(summary["fail_count"]) + 1
        summary["population_error"] = "multiple_sample_lineage_sha256_values"
    return pd.DataFrame(rows), summary
