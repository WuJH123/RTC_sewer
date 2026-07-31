"""Raw Independent Oracle for the project-wide R0-derived Step-2 population.

Unlike the legacy resolver, this implementation validates the exact raw detail
paths recorded by :mod:`v42_r0_paper_dataset`.  The oracle samples the raw
5-minute recorder at the formal model targets t+10,...,t+120, then independently
recomputes the H12 KPI labels.  Final scientific performance remains evaluated
from authoritative SWMM in Step 4; this oracle specifically verifies that the
Step-2 training labels match the frozen 12x10-minute surrogate discretization.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .v42_independent_oracle import (
    _format_kpis,
    _load_graph_node_ids,
    _priority_indices,
)


HORIZON_STEPS = 12
CONTROL_INTERVAL_MIN = 10.0
TIME_ATOL_MIN = 1.0e-6


def _read(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    return pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)


def _raw_branch_kpis_at_model_steps(
    detail_path: Path,
    checkpoint_min: float,
    node_ids: list[str],
    priority_indices: list[int],
) -> tuple[float, float, float, int]:
    header = pd.read_csv(detail_path, nrows=0)
    lookup = {
        str(c)[len("flood:"):].casefold(): str(c)
        for c in header.columns
        if str(c).startswith("flood:")
    }
    missing = [node for node in node_ids if node.casefold() not in lookup]
    if missing:
        raise ValueError(f"raw detail missing flood nodes: {missing[:10]}")
    usecols = ["elapsed_min"] + [lookup[node.casefold()] for node in node_ids]
    detail = pd.read_csv(detail_path, usecols=usecols)
    elapsed = pd.to_numeric(detail["elapsed_min"], errors="coerce").to_numpy(float)
    if not np.isfinite(elapsed).all():
        raise ValueError("elapsed_min contains NaN/Inf")
    target_times = np.asarray(
        [checkpoint_min + (i + 1) * CONTROL_INTERVAL_MIN for i in range(HORIZON_STEPS)],
        dtype=np.float64,
    )
    idx: list[int] = []
    for target in target_times:
        matches = np.flatnonzero(
            np.isclose(elapsed, target, atol=TIME_ATOL_MIN, rtol=0.0)
        )
        if len(matches) != 1:
            raise ValueError(
                f"expected one raw row at {target} min, found {len(matches)}"
            )
        idx.append(int(matches[0]))
    flood = (
        detail[[lookup[node.casefold()] for node in node_ids]]
        .iloc[idx]
        .apply(pd.to_numeric, errors="coerce")
        .to_numpy(dtype=np.float64)
    )
    if not np.isfinite(flood).all():
        raise ValueError("raw flooding-rate targets contain NaN/Inf")
    dt = np.diff(np.concatenate([[checkpoint_min], target_times])) * 60.0
    total = flood.sum(axis=1)
    priority = flood[:, priority_indices].sum(axis=1)
    return (
        float(np.sum(priority * dt)),
        float(np.sum(total * dt)),
        float(np.max(total)),
        int(len(idx)),
    )


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
            "oracle_discretization": "t+10,...,t+120_from_raw_recorder",
        }
        try:
            checkpoint = float(row["checkpoint_min"])
            pfv_c, tfv_c, peak_c, n_c = _raw_branch_kpis_at_model_steps(
                Path(str(row["source_detail_path_candidate"])),
                checkpoint,
                node_ids,
                priority_idx,
            )
            pfv_nc, _, _, n_nc = _raw_branch_kpis_at_model_steps(
                Path(str(row["source_detail_path_no_control"])),
                checkpoint,
                node_ids,
                priority_idx,
            )
            _, tfv_di, peak_di, n_di = _raw_branch_kpis_at_model_steps(
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

    lineages: list[str] = []
    if "sample_lineage_sha256" in df.columns:
        lineages = [
            str(x)
            for x in df["sample_lineage_sha256"].dropna().unique()
            if str(x)
        ]
    summary = {
        "contract_id": "PROJECT6_V42_PAPER_WORKFLOW_V1",
        "manifest": str(manifest_path),
        "audit_mode": "raw",
        "oracle_discretization": "formal_H12_exact_rows_from_raw_5min_detail",
        "final_performance_authority": "Step4_authoritative_SWMM",
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
