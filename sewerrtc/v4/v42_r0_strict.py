"""Strict, resumable Phase-R0 audit for the V4.2 paper workflow.

The high-throughput file reader lives in :mod:`v42_existing_pool_audit`.  This
module owns the *formal* R0 semantics around that reader:

* Engineering36 action hashes include ``elapsed_min`` when available;
* historical revealed Calibration/Locked/Formal evidence is discovered and
  tagged as consumed development instead of being silently pruned;
* TargetAvailability serialization is treated as a versioned schema: every
  field written by ``PhysicalRunRecord.as_dict()`` is named
  ``available_<field>`` and consumers never guess an unprefixed name;
* case classification is deterministic when duplicate lineage contains more
  than one physical row for the same branch role;
* an expensive 20k-file scan can be checkpointed *before* case classification,
  so a post-processing bug never requires another full disk audit.

The canonical CLI ``scripts/audit_v42_existing_swmm_pool.py`` calls this module.
"""
from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import v42_existing_pool_audit as base


_PATCH_LOCK = threading.Lock()
LOGICAL_CACHE_SCHEMA = "PROJECT6_V42_R0_LOGICAL_CACHE_V1"
AVAILABILITY_PREFIX = "available_"
AVAILABILITY_COLUMNS = {
    item.name: f"{AVAILABILITY_PREFIX}{item.name}"
    for item in fields(base.TargetAvailability)
}
_AVAILABILITY_BOOL_FIELDS = tuple(
    name for name in AVAILABILITY_COLUMNS if name != "depth_semantics"
)
_CASE_GROUP_COLUMNS = [
    "case_id",
    "event_id",
    "rainfall_sha256",
    "checkpoint_min",
    "network_sha256",
    "domain_id",
    "source_experiment",
    "source_role",
]


def _action_sha_semantic_compatible(
    path: Path,
    facility_ids: list[str],
    _df: pd.DataFrame | None = None,
) -> str:
    """Hash readback action using the frozen pre-optimization byte semantics."""
    if _df is not None:
        header = _df.iloc[:0]
    else:
        header = pd.read_csv(path, nrows=0)
    lookup = {
        str(c)[len("setting:"):].casefold(): str(c)
        for c in header.columns
        if str(c).startswith("setting:")
    }
    required = [lookup.get(fid.casefold()) for fid in facility_ids]
    if any(c is None for c in required):
        return ""
    read_cols = [str(c) for c in required]
    if "elapsed_min" in header.columns:
        read_cols = ["elapsed_min"] + read_cols
    if _df is not None:
        if not all(c in _df.columns for c in read_cols):
            return ""
        numeric = _df[read_cols].apply(pd.to_numeric, errors="coerce")
    else:
        numeric = pd.read_csv(path, usecols=read_cols).apply(
            pd.to_numeric, errors="coerce"
        )
    if numeric.isna().any().any():
        return ""
    arr = numeric.to_numpy(dtype=np.float64)
    return hashlib.sha256(arr.tobytes(order="C")).hexdigest()


def _coerce_bool_series(series: pd.Series, *, column: str) -> pd.Series:
    """Coerce persisted boolean columns without the ``astype(bool)`` string trap."""
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
            f"R0 boolean column {column!r} contains unsupported values: {unknown[:10]}"
        )
    return text.isin(true_values)


def _require_columns(frame: pd.DataFrame, required: set[str], *, context: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"{context} missing columns: {missing}")


def _enrich_physical_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate the serialized TargetAvailability schema and derive case columns."""
    required = set(_CASE_GROUP_COLUMNS) | {
        "branch_role",
        "completion_status",
        "physical_identity_sha256",
        "detail_path",
    } | set(AVAILABILITY_COLUMNS.values())
    _require_columns(frame, required, context="R0 physical serialization")
    out = frame.copy()

    for field_name in _AVAILABILITY_BOOL_FIELDS:
        col = AVAILABILITY_COLUMNS[field_name]
        out[col] = _coerce_bool_series(out[col], column=col)
    semantics_col = AVAILABILITY_COLUMNS["depth_semantics"]
    out[semantics_col] = out[semantics_col].fillna("unknown").astype(str)

    out["core_trajectory_complete"] = (
        out[AVAILABILITY_COLUMNS["node_depth"]]
        & out[AVAILABILITY_COLUMNS["node_flooding_rate"]]
        & out[AVAILABILITY_COLUMNS["storage_volume"]]
        & out[AVAILABILITY_COLUMNS["managed_facility_flow"]]
        & out[AVAILABILITY_COLUMNS["readback_setting"]]
        & out[AVAILABILITY_COLUMNS["rainfall"]]
        & out[AVAILABILITY_COLUMNS["history_complete"]]
        & out[AVAILABILITY_COLUMNS["horizon_complete"]]
    )
    out["formal_all_target_complete"] = (
        out["core_trajectory_complete"]
        & out[AVAILABILITY_COLUMNS["outfall_flow"]]
    )
    out["missing_outfall_only"] = (
        out["core_trajectory_complete"]
        & ~out[AVAILABILITY_COLUMNS["outfall_flow"]]
    )
    # Backward-compatible convenience column retained in output manifests.  The
    # classifier itself uses the authoritative prefixed schema directly.
    out["outfall_reconstruction_candidate"] = out[
        AVAILABILITY_COLUMNS["outfall_reconstruction_candidate"]
    ]
    return out


def _role_any(group: pd.DataFrame, role: str, column: str) -> bool:
    part = group[group["branch_role"].astype(str) == str(role)]
    if part.empty:
        return False
    return bool(_coerce_bool_series(part[column], column=column).any())


def _role_outfall_reconstructable(group: pd.DataFrame, role: str) -> bool:
    part = group[group["branch_role"].astype(str) == str(role)]
    if part.empty:
        return False
    missing = _coerce_bool_series(part["missing_outfall_only"], column="missing_outfall_only")
    candidate = _coerce_bool_series(
        part[AVAILABILITY_COLUMNS["outfall_reconstruction_candidate"]],
        column=AVAILABILITY_COLUMNS["outfall_reconstruction_candidate"],
    )
    return bool((missing & candidate).any())


def _classify_case_schema_safe(group: pd.DataFrame) -> base.CaseReuseRecord:
    """Classify one logical case using the exact serialized physical schema.

    Duplicate physical lineage is handled per role with ``any(valid row)``.
    Therefore thread completion order cannot decide which duplicate row wins.
    """
    required = set(_CASE_GROUP_COLUMNS) | {
        "branch_role",
        "completion_status",
        "physical_identity_sha256",
        "core_trajectory_complete",
        "formal_all_target_complete",
        "missing_outfall_only",
        AVAILABILITY_COLUMNS["depth_semantics"],
        AVAILABILITY_COLUMNS["outfall_reconstruction_candidate"],
        AVAILABILITY_COLUMNS["node_depth"],
        AVAILABILITY_COLUMNS["node_flooding_rate"],
    }
    _require_columns(group, required, context="R0 case classification")

    first = group.iloc[0]
    roles = set(group["branch_role"].fillna("").astype(str))
    recognized_roles = {role for role in roles if role in base.FOUR_BRANCH_ROLES}
    four_complete = all(role in recognized_roles for role in base.FOUR_BRANCH_ROLES)

    full_targets = bool(
        four_complete
        and all(
            _role_any(group, role, "formal_all_target_complete")
            for role in base.FOUR_BRANCH_ROLES
        )
    )
    core_targets = bool(
        four_complete
        and all(
            _role_any(group, role, "core_trajectory_complete")
            for role in base.FOUR_BRANCH_ROLES
        )
    )
    outfall_only = bool(
        core_targets
        and not full_targets
        and all(
            _role_any(group, role, "missing_outfall_only")
            for role in base.FOUR_BRANCH_ROLES
        )
    )
    outfall_reconstruct = bool(
        outfall_only
        and all(
            _role_outfall_reconstructable(group, role)
            for role in base.FOUR_BRANCH_ROLES
        )
    )

    reasons: list[str] = []
    domain = str(first["domain_id"])
    source_role = str(first["source_role"])
    source_experiment = str(first["source_experiment"])
    completion_bad = bool(
        group["completion_status"].fillna("").astype(str).str.casefold().isin({"failed", "error"}).any()
    )
    ambiguous_depth = bool(
        group[AVAILABILITY_COLUMNS["depth_semantics"]]
        .fillna("unknown")
        .astype(str)
        .eq("ambiguous")
        .any()
    )
    if completion_bad:
        reasons.append("failed_completion")
    if ambiguous_depth:
        reasons.append("ambiguous_depth_head_semantics")
    if source_role == "reserved_evaluation":
        reasons.append("reserved_evaluation")
    if not four_complete:
        reasons.append("four_reference_incomplete")
    if outfall_only:
        reasons.append("outfall_only_missing")
    if outfall_reconstruct:
        reasons.append("outfall_link_reconstruction_requires_validation")

    any_core = bool(
        _coerce_bool_series(
            group["core_trajectory_complete"], column="core_trajectory_complete"
        ).any()
    )
    any_depth = bool(
        _coerce_bool_series(
            group[AVAILABILITY_COLUMNS["node_depth"]],
            column=AVAILABILITY_COLUMNS["node_depth"],
        ).any()
    )
    any_flood = bool(
        _coerce_bool_series(
            group[AVAILABILITY_COLUMNS["node_flooding_rate"]],
            column=AVAILABILITY_COLUMNS["node_flooding_rate"],
        ).any()
    )

    if completion_bad or ambiguous_depth or source_role == "reserved_evaluation":
        classification = base.ReuseClassification.INVALID_OR_INCOMPATIBLE
    elif domain == "source_dwf":
        classification = (
            base.ReuseClassification.SOURCE_DOMAIN_REUSE
            if core_targets or any_core
            else base.ReuseClassification.RERUN_REQUIRED
        )
    elif full_targets:
        classification = base.ReuseClassification.FULL_REUSE
    elif outfall_only or any_core or (any_depth and any_flood):
        classification = base.ReuseClassification.PARTIAL_AUX_REUSE
    else:
        classification = base.ReuseClassification.RERUN_REQUIRED

    checkpoint = None if pd.isna(first["checkpoint_min"]) else float(first["checkpoint_min"])
    # Include every grouping dimension.  The previous UID omitted
    # source_experiment/source_role, so two different groups could collide and
    # later break one-to-one alignment merges.
    case_uid = base._sha256_text(
        json.dumps(
            {
                "case_id": str(first["case_id"]),
                "event_id": str(first["event_id"]),
                "rainfall_sha": str(first["rainfall_sha256"]),
                "checkpoint_min": checkpoint,
                "network_sha": str(first["network_sha256"]),
                "domain_id": domain,
                "source_experiment": source_experiment,
                "source_role": source_role,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return base.CaseReuseRecord(
        case_uid=case_uid,
        case_id=str(first["case_id"]),
        event_id=str(first["event_id"]),
        rainfall_sha256=str(first["rainfall_sha256"]),
        checkpoint_min=checkpoint,
        network_sha256=str(first["network_sha256"]),
        domain_id=domain,
        source_role=source_role,
        source_experiment=source_experiment,
        branch_count=int(len(recognized_roles)),
        four_reference_complete=bool(four_complete),
        same_forcing_pass=None,
        full_reuse_targets=bool(full_targets),
        core_trajectory_targets=bool(core_targets),
        outfall_only_blocker=bool(outfall_only),
        outfall_reconstruction_candidate=bool(outfall_reconstruct),
        classification=classification,
        reason_codes=tuple(reasons),
        branch_physical_ids=tuple(
            sorted(set(group["physical_identity_sha256"].fillna("").astype(str)) - {""})
        ),
    )


@contextmanager
def _strict_runtime_semantics():
    """Temporarily restore strict semantics inside the optimized audit module."""
    with _PATCH_LOCK:
        old_action = base._action_sha
        old_skip = base._SKIP_DIR_NAMES
        old_classifier = base._classify_case
        try:
            base._action_sha = _action_sha_semantic_compatible
            base._SKIP_DIR_NAMES = frozenset()
            # Protect callers that still delegate to the base audit while the
            # strict context is active.
            base._classify_case = _classify_case_schema_safe
            yield
        finally:
            base._action_sha = old_action
            base._SKIP_DIR_NAMES = old_skip
            base._classify_case = old_classifier


def _sort_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    keys = [c for c in columns if c in frame.columns]
    if frame.empty or not keys:
        return frame.reset_index(drop=True)
    return frame.sort_values(keys, kind="mergesort").reset_index(drop=True)


def _read_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _write_table_atomic(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    if path.suffix.lower() == ".parquet":
        frame.to_parquet(tmp, index=False)
    else:
        frame.to_csv(tmp, index=False)
    tmp.replace(path)


def _cache_meta_path(cache_path: Path) -> Path:
    return cache_path.with_name(cache_path.name + ".meta.json")


def _write_scan_cache(
    frame: pd.DataFrame,
    *,
    cache_path: Path,
    project_root: Path,
    outputs_root: Path,
    full_finite_check: bool,
) -> None:
    checked = _enrich_physical_frame(frame)
    # Persist only serialized/derived values; no Python dataclass objects.
    _write_table_atomic(checked, cache_path)
    meta = {
        "schema": LOGICAL_CACHE_SCHEMA,
        "record_count": int(len(checked)),
        "full_finite_check": bool(full_finite_check),
        "project_root": str(project_root.resolve()),
        "outputs_root": str(outputs_root.resolve()),
        "active_network_sha256": base._active_network_sha(project_root),
        "availability_columns": sorted(AVAILABILITY_COLUMNS.values()),
        "created_at_unix": time.time(),
    }
    meta_path = _cache_meta_path(cache_path)
    tmp = meta_path.with_name(meta_path.name + ".tmp")
    tmp.write_text(json.dumps(meta, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(meta_path)


def _load_scan_cache(
    *,
    cache_path: Path,
    project_root: Path,
    outputs_root: Path,
    full_finite_check: bool,
) -> pd.DataFrame:
    meta_path = _cache_meta_path(cache_path)
    if not cache_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"R0 scan cache is incomplete: {cache_path} / {meta_path}"
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    if meta.get("schema") != LOGICAL_CACHE_SCHEMA:
        raise RuntimeError("R0 scan cache schema mismatch")
    if bool(meta.get("full_finite_check")) != bool(full_finite_check):
        raise RuntimeError("R0 scan cache finite-audit mode mismatch")
    if str(meta.get("project_root")) != str(project_root.resolve()):
        raise RuntimeError("R0 scan cache project-root mismatch")
    if str(meta.get("outputs_root")) != str(outputs_root.resolve()):
        raise RuntimeError("R0 scan cache outputs-root mismatch")
    if str(meta.get("active_network_sha256", "")) != base._active_network_sha(project_root):
        raise RuntimeError("R0 scan cache network SHA mismatch")
    frame = _read_table(cache_path)
    if int(meta.get("record_count", -1)) != int(len(frame)):
        raise RuntimeError("R0 scan cache row-count mismatch")
    return _enrich_physical_frame(frame)


def _scan_logical_records(
    *,
    project_root: Path,
    outputs_root: Path,
    full_finite_check: bool,
    max_workers: int,
) -> pd.DataFrame:
    t0 = time.monotonic()
    print("[R0] discovering details …", file=sys.stderr, flush=True)
    discovery = base.discover_existing_details(outputs_root)
    if discovery.empty:
        raise FileNotFoundError(f"no detail.csv or *_detail.csv found under {outputs_root}")
    print(
        f"[R0] discovered {len(discovery)} details in {time.monotonic()-t0:.1f}s",
        file=sys.stderr,
        flush=True,
    )

    graph = base._load_graph_topology(project_root)
    node_ids = list(graph["node_ids"])
    facility_ids = base._load_engineering36_ids(project_root)
    nodes, _ = base._parse_inp_topology(
        project_root / "data" / "wuhan_v8_storage_retrofit.inp"
    )
    storage_ids = [
        str(x)
        for x in nodes.loc[nodes["node_type"] == "storage", "node_id"].tolist()
    ]
    outfall_ids = [
        str(x)
        for x in nodes.loc[nodes["node_type"] == "outfall", "node_id"].tolist()
    ]
    invert_by_id = {
        str(row.node_id).casefold(): float(row.invert)
        for row in nodes.itertuples(index=False)
    }
    incoming_links = base._incoming_links_by_outfall(project_root, outfall_ids)
    active_sha = base._active_network_sha(project_root)

    records: list[base.PhysicalRunRecord] = []
    detail_cache: dict[
        tuple[str, float | None],
        tuple[base.TargetAvailability, tuple[str, ...], str, int],
    ] = {}
    cache_lock = threading.Lock()
    rows_list = list(discovery.itertuples(index=False))
    total = len(rows_list)
    done_count = 0
    t1 = time.monotonic()
    print(
        f"[R0] auditing {total} files with {max_workers} threads …",
        file=sys.stderr,
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                base._audit_single_row,
                row,
                outputs_root=outputs_root,
                node_ids=node_ids,
                storage_ids=storage_ids,
                facility_ids=facility_ids,
                outfall_ids=outfall_ids,
                invert_by_id=invert_by_id,
                incoming_links=incoming_links,
                active_sha=active_sha,
                full_finite_check=full_finite_check,
                detail_cache=detail_cache,
                cache_lock=cache_lock,
            ): idx
            for idx, row in enumerate(rows_list)
        }
        for future in as_completed(futures):
            records.append(future.result())
            done_count += 1
            if done_count % 200 == 0 or done_count == total:
                elapsed = time.monotonic() - t1
                rate = done_count / elapsed if elapsed > 0 else 0.0
                eta = (total - done_count) / rate if rate > 0 else 0.0
                print(
                    f"[R0] {done_count}/{total}  elapsed={elapsed:.0f}s  "
                    f"rate={rate:.1f}/s  eta={eta:.0f}s  cache={len(detail_cache)}",
                    file=sys.stderr,
                    flush=True,
                )
    frame = pd.DataFrame([record.as_dict() for record in records])
    return _sort_frame(
        _enrich_physical_frame(frame),
        ["physical_identity_sha256", "source_role", "detail_path", "case_id", "branch_role"],
    )


def _postprocess_physical_frame(
    physical_all: pd.DataFrame,
    *,
    full_finite_check: bool,
) -> base.ExistingPoolAuditResult:
    physical_all = _enrich_physical_frame(physical_all)
    duplicate_mask = physical_all.duplicated("physical_identity_sha256", keep=False)
    duplicate_lineage = physical_all.loc[duplicate_mask].copy()
    canonical = (
        physical_all.sort_values(
            ["physical_identity_sha256", "source_role", "detail_path"],
            kind="mergesort",
        )
        .drop_duplicates("physical_identity_sha256", keep="first")
        .reset_index(drop=True)
    )

    cases: list[base.CaseReuseRecord] = []
    for _, group in physical_all.groupby(_CASE_GROUP_COLUMNS, dropna=False, sort=False):
        cases.append(_classify_case_schema_safe(group))
    case_df = pd.DataFrame([case.as_dict() for case in cases])
    if not case_df.empty and case_df["case_uid"].duplicated().any():
        duplicates = case_df.loc[case_df["case_uid"].duplicated(False), "case_uid"].tolist()
        raise RuntimeError(f"R0 case_uid collision after full grouping keys: {duplicates[:10]}")

    source_summary = (
        case_df.groupby(["source_experiment", "classification"], dropna=False)
        .size()
        .rename("case_count")
        .reset_index()
        if not case_df.empty
        else pd.DataFrame(columns=["source_experiment", "classification", "case_count"])
    )
    target_cols = [
        AVAILABILITY_COLUMNS["node_depth"],
        AVAILABILITY_COLUMNS["node_flooding_rate"],
        AVAILABILITY_COLUMNS["storage_volume"],
        AVAILABILITY_COLUMNS["managed_facility_flow"],
        AVAILABILITY_COLUMNS["outfall_flow"],
        AVAILABILITY_COLUMNS["readback_setting"],
        AVAILABILITY_COLUMNS["rainfall"],
        AVAILABILITY_COLUMNS["history_complete"],
        AVAILABILITY_COLUMNS["horizon_complete"],
    ]
    target_summary = pd.DataFrame(
        {
            "target": target_cols,
            "available_physical_runs": [int(canonical[col].sum()) for col in target_cols],
            "total_physical_runs": [int(len(canonical))] * len(target_cols),
        }
    )
    target_summary["available_fraction"] = np.where(
        target_summary["total_physical_runs"] > 0,
        target_summary["available_physical_runs"]
        / target_summary["total_physical_runs"],
        np.nan,
    )
    class_counts = (
        case_df["classification"].value_counts().to_dict() if not case_df.empty else {}
    )
    summary = {
        "contract": "PROJECT6_V42_EXISTING_POOL_REUSE_V1",
        "logical_detail_records": int(len(physical_all)),
        "unique_physical_runs": int(len(canonical)),
        "duplicate_lineage_rows": int(len(duplicate_lineage)),
        "case_groups": int(len(case_df)),
        "unique_events": int(case_df["event_id"].nunique()) if not case_df.empty else 0,
        "classification_counts": {str(k): int(v) for k, v in class_counts.items()},
        "full_finite_check": bool(full_finite_check),
        "fixed_sample_cap": None,
        "outfall_policy": (
            "explicit target required for FULL_REUSE; incoming-link coverage is "
            "candidate-only until independent validation"
        ),
        "missing_targets_are_imputed": False,
        "availability_schema": "TargetAvailability->available_<field>",
        "case_uid_uses_all_grouping_dimensions": True,
        "duplicate_role_classification_order_invariant": True,
    }
    return base.ExistingPoolAuditResult(
        physical_runs=canonical,
        cases=case_df,
        duplicate_lineage=duplicate_lineage,
        source_summary=source_summary,
        target_summary=target_summary,
        summary=summary,
    )


def audit_existing_swmm_pool_strict(
    *,
    project_root: str | Path,
    outputs_root: str | Path,
    full_finite_check: bool = True,
    max_workers: int = 16,
    logical_cache_path: str | Path | None = None,
    resume_from_logical_cache: bool = False,
) -> base.ExistingPoolAuditResult:
    """Run strict R0 with an optional post-scan resume checkpoint."""
    project_root = Path(project_root)
    outputs_root = Path(outputs_root)
    cache_path = Path(logical_cache_path) if logical_cache_path is not None else None
    with _strict_runtime_semantics():
        if resume_from_logical_cache:
            if cache_path is None:
                raise ValueError("resume_from_logical_cache requires logical_cache_path")
            physical_all = _load_scan_cache(
                cache_path=cache_path,
                project_root=project_root,
                outputs_root=outputs_root,
                full_finite_check=bool(full_finite_check),
            )
            print(
                f"[R0] resumed {len(physical_all)} logical detail records from {cache_path}; "
                "skipping expensive CSV audit",
                file=sys.stderr,
                flush=True,
            )
        else:
            physical_all = _scan_logical_records(
                project_root=project_root,
                outputs_root=outputs_root,
                full_finite_check=bool(full_finite_check),
                max_workers=max(1, int(max_workers)),
            )
            if cache_path is not None:
                _write_scan_cache(
                    physical_all,
                    cache_path=cache_path,
                    project_root=project_root,
                    outputs_root=outputs_root,
                    full_finite_check=bool(full_finite_check),
                )
                print(
                    f"[R0] scan checkpoint written before case classification: {cache_path}",
                    file=sys.stderr,
                    flush=True,
                )
        result = _postprocess_physical_frame(
            physical_all,
            full_finite_check=bool(full_finite_check),
        )

    return base.ExistingPoolAuditResult(
        physical_runs=_sort_frame(
            result.physical_runs,
            ["physical_identity_sha256", "source_role", "detail_path"],
        ),
        cases=_sort_frame(result.cases, ["case_uid", "source_role"]),
        duplicate_lineage=_sort_frame(
            result.duplicate_lineage,
            ["physical_identity_sha256", "source_role", "detail_path"],
        ),
        source_summary=_sort_frame(
            result.source_summary, ["source_experiment", "classification"]
        ),
        target_summary=_sort_frame(result.target_summary, ["target"]),
        summary={
            **result.summary,
            "strict_semantics_wrapper": True,
            "action_hash_includes_elapsed_when_available": True,
            "historical_revealed_evidence_policy": (
                "discover_then_tag_consumed_development"
            ),
            "threaded_output_deterministically_sorted": True,
            "post_scan_cache_enabled": cache_path is not None,
            "post_scan_cache_path": "" if cache_path is None else str(cache_path),
            "resumed_from_post_scan_cache": bool(resume_from_logical_cache),
        },
    )


write_existing_pool_audit = base.write_existing_pool_audit
