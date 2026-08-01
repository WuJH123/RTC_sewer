"""Discovery-safe and incrementally refreshable Phase-R0 audit.

This module protects two formal-evidence invariants:

1. continuation trajectories such as ``candidate_then_internal.csv`` and
   ``candidate_then_passive.csv`` are useful physical evidence, but their file
   names alone do not prove that they are the canonical Dynamic-Internal or
   Hold-Previous references. They therefore enter R0 with explicit auxiliary
   roles until authoritative provenance proves otherwise;
2. a persisted expensive scan cache may be resumed only if it covers the
   *current* discovery population. When new historical files are found, refresh
   audits only new/changed logical rows and reuses unchanged cached rows.

Raw SWMM files remain authoritative and read-only.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import pandas as pd

from . import v42_existing_pool_audit as base
from . import v42_r0_strict as strict


DISCOVERY_CACHE_SCHEMA = "PROJECT6_V42_R0_DISCOVERY_V2"
AUX_CANDIDATE_THEN_INTERNAL = "candidate_then_internal_aux"
AUX_CANDIDATE_THEN_PASSIVE = "candidate_then_passive_aux"


def _formal_role_from_filename(path: Path) -> str:
    """Infer only branch roles proven by the filename convention.

    PFV-first continuation files are deliberately not promoted to canonical
    reference roles. A later provenance importer may relabel them only after
    verifying the generator/manifest semantics.
    """
    name = path.name.lower()
    if "candidate_then_internal" in name:
        return AUX_CANDIDATE_THEN_INTERNAL
    if "candidate_then_passive" in name:
        return AUX_CANDIDATE_THEN_PASSIVE
    ordered = (
        ("dynamic_internal_rules", "dynamic_internal"),
        ("dynamic_internal", "dynamic_internal"),
        ("hold_previous", "hold_previous"),
        ("no_control", "no_control"),
        ("candidate", "candidate"),
        ("proposed", "candidate"),
    )
    for token, role in ordered:
        if token in name:
            return role
    return "unknown"


@contextmanager
def _formal_discovery_semantics():
    old = base._infer_role_from_filename
    base._infer_role_from_filename = _formal_role_from_filename
    try:
        yield
    finally:
        base._infer_role_from_filename = old


def discover_formal_existing_details(outputs_root: str | Path) -> pd.DataFrame:
    """Discover heterogeneous details with fail-closed PFV-first role semantics."""
    with _formal_discovery_semantics():
        return base.discover_existing_details(outputs_root)


def _missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def _text(value: Any) -> str:
    return "" if _missing(value) else str(value)


def _normal_checkpoint(value: Any) -> str:
    if _missing(value):
        return ""
    try:
        return f"{float(value):.9f}"
    except (TypeError, ValueError):
        return str(value)


def _logical_key(row: Any) -> tuple[str, ...]:
    def value(name: str) -> Any:
        if isinstance(row, dict):
            return row.get(name)
        if isinstance(row, pd.Series):
            return row.get(name)
        return getattr(row, name, None)

    detail_text = _text(value("detail_path"))
    if not detail_text:
        raise ValueError("logical R0 row has no detail_path")
    return (
        str(Path(detail_text).resolve()),
        _text(value("case_id")),
        _text(value("event_id")),
        _normal_checkpoint(value("checkpoint_min")),
        _text(value("network_sha256")),
        _text(value("rainfall_sha256")),
        _text(value("branch_role")),
        _text(value("completion_path")),
    )


def _discovery_index(discovery: pd.DataFrame) -> tuple[dict[tuple[str, ...], dict[str, Any]], str]:
    """Return logical-key metadata and a deterministic current-population hash."""
    index: dict[tuple[str, ...], dict[str, Any]] = {}
    fingerprint_rows: list[dict[str, Any]] = []
    for row in discovery.itertuples(index=False):
        key = _logical_key(row)
        if key in index:
            raise RuntimeError(f"duplicate logical discovery key: {key}")
        path = Path(key[0])
        stat = path.stat()
        info = {
            "detail_path": str(path),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
        index[key] = info
        fingerprint_rows.append(
            {
                "logical_key": list(key),
                "size": info["size"],
                "mtime_ns": info["mtime_ns"],
            }
        )
    fingerprint_rows.sort(key=lambda item: tuple(item["logical_key"]))
    payload = json.dumps(fingerprint_rows, sort_keys=True, separators=(",", ":"))
    return index, hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _cache_meta_path(cache_path: Path) -> Path:
    return cache_path.with_name(cache_path.name + ".meta.json")


def _read_cache_meta(cache_path: Path) -> dict[str, Any]:
    meta_path = _cache_meta_path(cache_path)
    if not meta_path.exists():
        raise FileNotFoundError(meta_path)
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("R0 cache metadata root must be an object")
    return payload


def _stamp_discovery_meta(
    *,
    cache_path: Path,
    discovery_count: int,
    fingerprint: str,
) -> None:
    meta_path = _cache_meta_path(cache_path)
    meta = _read_cache_meta(cache_path)
    meta.update(
        {
            "discovery_schema": DISCOVERY_CACHE_SCHEMA,
            "discovery_record_count": int(discovery_count),
            "discovery_fingerprint_sha256": str(fingerprint),
            "discovery_checked_at_unix": time.time(),
        }
    )
    tmp = meta_path.with_name(meta_path.name + ".tmp")
    tmp.write_text(json.dumps(meta, indent=2, allow_nan=False), encoding="utf-8")
    tmp.replace(meta_path)


def _cache_matches_current_discovery(
    *,
    cache_path: Path,
    discovery_count: int,
    fingerprint: str,
) -> tuple[bool, str]:
    meta = _read_cache_meta(cache_path)
    if meta.get("discovery_schema") != DISCOVERY_CACHE_SCHEMA:
        return False, "cache_has_no_current_discovery_contract"
    if int(meta.get("discovery_record_count", -1)) != int(discovery_count):
        return False, "discovery_record_count_changed"
    if str(meta.get("discovery_fingerprint_sha256", "")) != str(fingerprint):
        return False, "discovery_fingerprint_changed"
    return True, ""


def _file_is_reusable_from_cache(
    cached: Any,
    current: dict[str, Any],
    cache_meta: dict[str, Any],
) -> bool:
    if int(getattr(cached, "detail_size_bytes", -1)) != int(current["size"]):
        return False
    cached_mtime = getattr(cached, "source_mtime_ns", None)
    if not _missing(cached_mtime):
        return int(cached_mtime) == int(current["mtime_ns"])
    # Backward compatibility for the 20,002-row cache created before mtime was
    # persisted: reuse it only when the source file predates that cache write.
    created = float(cache_meta.get("created_at_unix", 0.0))
    return created > 0.0 and int(current["mtime_ns"]) <= int(created * 1e9)


def _prepare_audit_context(project_root: Path) -> dict[str, Any]:
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
    return {
        "node_ids": node_ids,
        "facility_ids": facility_ids,
        "storage_ids": storage_ids,
        "outfall_ids": outfall_ids,
        "invert_by_id": {
            str(row.node_id).casefold(): float(row.invert)
            for row in nodes.itertuples(index=False)
        },
        "incoming_links": base._incoming_links_by_outfall(project_root, outfall_ids),
        "active_sha": base._active_network_sha(project_root),
    }


def _audit_subset(
    *,
    rows: list[Any],
    project_root: Path,
    outputs_root: Path,
    full_finite_check: bool,
    max_workers: int,
) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    ctx = _prepare_audit_context(project_root)
    detail_cache: dict = {}
    cache_lock = threading.Lock()
    records: list[base.PhysicalRunRecord] = []
    total = len(rows)
    done = 0
    t0 = time.monotonic()
    print(
        f"[R0-refresh] auditing {total} new/changed logical rows with {max_workers} threads ...",
        file=sys.stderr,
        flush=True,
    )
    with strict._strict_runtime_semantics():
        with ThreadPoolExecutor(max_workers=max(1, int(max_workers))) as pool:
            futures = {
                pool.submit(
                    base._audit_single_row,
                    row,
                    outputs_root=outputs_root,
                    node_ids=ctx["node_ids"],
                    storage_ids=ctx["storage_ids"],
                    facility_ids=ctx["facility_ids"],
                    outfall_ids=ctx["outfall_ids"],
                    invert_by_id=ctx["invert_by_id"],
                    incoming_links=ctx["incoming_links"],
                    active_sha=ctx["active_sha"],
                    full_finite_check=full_finite_check,
                    detail_cache=detail_cache,
                    cache_lock=cache_lock,
                ): row
                for row in rows
            }
            for future in as_completed(futures):
                records.append(future.result())
                done += 1
                if done % 200 == 0 or done == total:
                    elapsed = time.monotonic() - t0
                    rate = done / elapsed if elapsed else 0.0
                    eta = (total - done) / rate if rate else 0.0
                    print(
                        f"[R0-refresh] {done}/{total} elapsed={elapsed:.0f}s "
                        f"rate={rate:.1f}/s eta={eta:.0f}s",
                        file=sys.stderr,
                        flush=True,
                    )
    frame = pd.DataFrame([record.as_dict() for record in records])
    if frame.empty:
        return frame
    frame["source_mtime_ns"] = [
        int(Path(str(path)).stat().st_mtime_ns) for path in frame["detail_path"]
    ]
    return strict._enrich_physical_frame(frame)


def _finalize(
    result: base.ExistingPoolAuditResult,
    *,
    cache_path: Path,
    resume: bool,
    refresh_stats: dict[str, int] | None,
) -> base.ExistingPoolAuditResult:
    summary = {
        **result.summary,
        "strict_semantics_wrapper": True,
        "action_hash_includes_elapsed_when_available": True,
        "historical_revealed_evidence_policy": "discover_then_tag_consumed_development",
        "threaded_output_deterministically_sorted": True,
        "post_scan_cache_enabled": True,
        "post_scan_cache_path": str(cache_path),
        "resumed_from_post_scan_cache": bool(resume),
        "discovery_cache_current": True,
        "pfvfirst_continuation_role_policy": "auxiliary_until_proven_by_provenance",
    }
    if refresh_stats is not None:
        summary["incremental_refresh"] = refresh_stats
    return base.ExistingPoolAuditResult(
        physical_runs=strict._sort_frame(
            result.physical_runs,
            ["physical_identity_sha256", "source_role", "detail_path"],
        ),
        cases=strict._sort_frame(result.cases, ["case_uid", "source_role"]),
        duplicate_lineage=strict._sort_frame(
            result.duplicate_lineage,
            ["physical_identity_sha256", "source_role", "detail_path"],
        ),
        source_summary=strict._sort_frame(
            result.source_summary, ["source_experiment", "classification"]
        ),
        target_summary=strict._sort_frame(result.target_summary, ["target"]),
        summary=summary,
    )


def audit_existing_swmm_pool_refreshable(
    *,
    project_root: str | Path,
    outputs_root: str | Path,
    cache_path: str | Path,
    full_finite_check: bool = True,
    max_workers: int = 16,
    resume: bool = False,
    refresh: bool = False,
) -> base.ExistingPoolAuditResult:
    """Run full/resume/refresh R0 while proving cache population completeness."""
    if resume and refresh:
        raise ValueError("choose either resume or refresh, not both")
    project_root = Path(project_root)
    outputs_root = Path(outputs_root)
    cache_path = Path(cache_path)

    discovery = discover_formal_existing_details(outputs_root)
    discovery_index, fingerprint = _discovery_index(discovery)
    print(
        f"[R0] formal discovery sees {len(discovery)} logical detail rows; "
        f"fingerprint={fingerprint[:12]}",
        file=sys.stderr,
        flush=True,
    )

    if not resume and not refresh:
        with _formal_discovery_semantics():
            result = strict.audit_existing_swmm_pool_strict(
                project_root=project_root,
                outputs_root=outputs_root,
                full_finite_check=full_finite_check,
                max_workers=max_workers,
                logical_cache_path=cache_path,
                resume_from_logical_cache=False,
            )
        cached = strict._read_table(cache_path)
        cached["source_mtime_ns"] = [
            int(Path(str(path)).stat().st_mtime_ns) for path in cached["detail_path"]
        ]
        strict._write_scan_cache(
            cached,
            cache_path=cache_path,
            project_root=project_root,
            outputs_root=outputs_root,
            full_finite_check=full_finite_check,
        )
        _stamp_discovery_meta(
            cache_path=cache_path,
            discovery_count=len(discovery),
            fingerprint=fingerprint,
        )
        return _finalize(result, cache_path=cache_path, resume=False, refresh_stats=None)

    cache_meta = _read_cache_meta(cache_path)
    with strict._strict_runtime_semantics():
        cached = strict._load_scan_cache(
            cache_path=cache_path,
            project_root=project_root,
            outputs_root=outputs_root,
            full_finite_check=full_finite_check,
        )

    if resume:
        matches, reason = _cache_matches_current_discovery(
            cache_path=cache_path,
            discovery_count=len(discovery),
            fingerprint=fingerprint,
        )
        if not matches:
            raise RuntimeError(
                f"R0 scan cache is stale relative to current discovery ({reason}); "
                "run with --refresh-scan-cache instead of silently omitting new data"
            )
        result = strict._postprocess_physical_frame(
            cached, full_finite_check=full_finite_check
        )
        return _finalize(result, cache_path=cache_path, resume=True, refresh_stats=None)

    discovery_rows = {
        _logical_key(row): row for row in discovery.itertuples(index=False)
    }
    cached_rows = {
        _logical_key(row): row for row in cached.itertuples(index=False)
    }

    reused_records: list[dict[str, Any]] = []
    to_audit: list[Any] = []
    changed = 0
    for key, discovery_row in discovery_rows.items():
        cached_row = cached_rows.get(key)
        info = discovery_index[key]
        if cached_row is not None and _file_is_reusable_from_cache(
            cached_row, info, cache_meta
        ):
            reused_records.append(dict(cached_row._asdict()))
        else:
            if cached_row is not None:
                changed += 1
            to_audit.append(discovery_row)

    new_frame = _audit_subset(
        rows=to_audit,
        project_root=project_root,
        outputs_root=outputs_root,
        full_finite_check=full_finite_check,
        max_workers=max_workers,
    )
    reused = pd.DataFrame(reused_records)
    combined = pd.concat([reused, new_frame], ignore_index=True, sort=False)
    if combined.empty:
        raise RuntimeError("incremental R0 refresh produced an empty cache")
    if len(combined) != len(discovery):
        raise RuntimeError(
            f"incremental R0 refresh population mismatch: {len(combined)} != {len(discovery)}"
        )

    if "source_mtime_ns" not in combined.columns:
        combined["source_mtime_ns"] = [
            int(Path(str(path)).stat().st_mtime_ns) for path in combined["detail_path"]
        ]
    else:
        missing_mtime = combined["source_mtime_ns"].isna()
        if bool(missing_mtime.any()):
            combined.loc[missing_mtime, "source_mtime_ns"] = [
                int(Path(str(path)).stat().st_mtime_ns)
                for path in combined.loc[missing_mtime, "detail_path"]
            ]

    strict._write_scan_cache(
        combined,
        cache_path=cache_path,
        project_root=project_root,
        outputs_root=outputs_root,
        full_finite_check=full_finite_check,
    )
    _stamp_discovery_meta(
        cache_path=cache_path,
        discovery_count=len(discovery),
        fingerprint=fingerprint,
    )
    result = strict._postprocess_physical_frame(
        combined, full_finite_check=full_finite_check
    )
    removed = len(set(cached_rows) - set(discovery_rows))
    stats = {
        "current_discovery_rows": int(len(discovery)),
        "reused_cached_rows": int(len(reused)),
        "audited_new_or_changed_rows": int(len(new_frame)),
        "changed_cached_rows": int(changed),
        "removed_logical_rows": int(removed),
    }
    print(
        f"[R0-refresh] completed: {json.dumps(stats, sort_keys=True)}",
        file=sys.stderr,
        flush=True,
    )
    return _finalize(result, cache_path=cache_path, resume=False, refresh_stats=stats)
