"""Project-wide discovery, semantic audit, de-duplication and reuse classification.

This module deliberately does **not** assume that Train1600 is the complete
training pool.  It discovers physical SWMM evidence across historical Project6
outputs and normalises heterogeneous layouts into a canonical evidence table.

Scientific rules implemented here:

* a manifest row is not a physical run;
* file/network hashes are lineage evidence, not sufficient admission criteria;
* Candidate/NC/DI/Hold roles are preserved explicitly;
* missing targets are never filled with zero;
* DWF histories are source-domain evidence, not automatically invalid;
* outfall flow is never guessed from a neighbouring link.  Incoming-link flow
  coverage is reported only as a *reconstruction candidate* until independently
  validated against a recorder that stores explicit outfall flow;
* old calibration/locked/formal results that have already been revealed are
  treated as consumed development, not as fresh blind evidence.

The audit is intentionally read-only.  It never runs SWMM, deletes outputs, or
modifies historical evidence.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Iterable, Iterator

import numpy as np
import pandas as pd

from sewerrtc.v4.v42_trajectory_builder import (
    HISTORY_INTERVAL_MIN,
    HORIZON_INTERVAL_MIN,
    N_HISTORY_FRAMES,
    N_HORIZON_STEPS,
    _load_engineering36_ids,
    _load_graph_topology,
    _parse_inp_topology,
)


TIME_ATOL_MIN = 1.0e-6
BRANCH_ALIASES = {
    "candidate": "candidate",
    "no_control": "no_control",
    "dynamic_internal": "dynamic_internal",
    "dynamic_internal_rules": "dynamic_internal",
    "hold_previous": "hold_previous",
}
FOUR_BRANCH_ROLES = ("candidate", "no_control", "dynamic_internal", "hold_previous")


class ReuseClassification(str, Enum):
    """Admission class for existing physical evidence."""

    FULL_REUSE = "FULL_REUSE"
    REUSE_AFTER_EXTRACTION = "REUSE_AFTER_EXTRACTION"
    PARTIAL_AUX_REUSE = "PARTIAL_AUX_REUSE"
    SOURCE_DOMAIN_REUSE = "SOURCE_DOMAIN_REUSE"
    RERUN_REQUIRED = "RERUN_REQUIRED"
    INVALID_OR_INCOMPATIBLE = "INVALID_OR_INCOMPATIBLE"


@dataclass(frozen=True)
class TargetAvailability:
    node_depth: bool = False
    hydraulic_head: bool = False
    node_flooding_rate: bool = False
    storage_volume: bool = False
    managed_facility_flow: bool = False
    outfall_flow: bool = False
    readback_setting: bool = False
    rainfall: bool = False
    history_complete: bool = False
    horizon_complete: bool = False
    finite_checked: bool = False
    finite_pass: bool = False
    depth_semantics: str = "unknown"
    outfall_reconstruction_candidate: bool = False

    @property
    def core_trajectory_complete(self) -> bool:
        return bool(
            self.node_depth
            and self.node_flooding_rate
            and self.storage_volume
            and self.managed_facility_flow
            and self.readback_setting
            and self.rainfall
            and self.history_complete
            and self.horizon_complete
        )

    @property
    def formal_all_target_complete(self) -> bool:
        return bool(self.core_trajectory_complete and self.outfall_flow)


@dataclass(frozen=True)
class PhysicalRunRecord:
    source_root: str
    source_experiment: str
    run_dir: str
    completion_path: str | None
    detail_path: str
    detail_sha256: str
    detail_size_bytes: int
    case_id: str
    event_id: str
    rainfall_sha256: str
    checkpoint_min: float | None
    branch_role: str
    network_sha256: str
    active_network_sha_match: bool | None
    domain_id: str
    source_role: str
    action_readback_sha256: str
    physical_identity_sha256: str
    completion_status: str
    prefix_hash_match: bool | None
    checkpoint_hash_match: bool | None
    target: TargetAvailability
    missing_target_groups: tuple[str, ...]
    audit_reasons: tuple[str, ...]
    window_anchor_count: int = 0

    def as_dict(self) -> dict[str, Any]:
        row = asdict(self)
        target = row.pop("target")
        row.update({f"available_{k}": v for k, v in target.items()})
        row["missing_target_groups"] = json.dumps(list(self.missing_target_groups))
        row["audit_reasons"] = json.dumps(list(self.audit_reasons))
        return row


@dataclass(frozen=True)
class CaseReuseRecord:
    case_uid: str
    case_id: str
    event_id: str
    rainfall_sha256: str
    checkpoint_min: float | None
    network_sha256: str
    domain_id: str
    source_role: str
    source_experiment: str
    branch_count: int
    four_reference_complete: bool
    same_forcing_pass: bool | None
    full_reuse_targets: bool
    core_trajectory_targets: bool
    outfall_only_blocker: bool
    outfall_reconstruction_candidate: bool
    classification: ReuseClassification
    reason_codes: tuple[str, ...]
    branch_physical_ids: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["classification"] = self.classification.value
        row["reason_codes"] = json.dumps(list(self.reason_codes))
        row["branch_physical_ids"] = json.dumps(list(self.branch_physical_ids))
        return row


@dataclass(frozen=True)
class ExistingPoolAuditResult:
    physical_runs: pd.DataFrame
    cases: pd.DataFrame
    duplicate_lineage: pd.DataFrame
    source_summary: pd.DataFrame
    target_summary: pd.DataFrame
    summary: dict[str, Any]


# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------


def _sha256_bytes(chunks: Iterable[bytes]) -> str:
    h = hashlib.sha256()
    for chunk in chunks:
        h.update(chunk)
    return h.hexdigest()


def _sha256_file(path: Path, chunk_size: int = 4 * 1024 * 1024) -> str:
    def chunks() -> Iterator[bytes]:
        with path.open("rb") as f:
            while True:
                block = f.read(chunk_size)
                if not block:
                    break
                yield block

    return _sha256_bytes(chunks())


def _window_anchor_count_from_elapsed(elapsed: np.ndarray) -> int:
    """Count valid 13x5+12x10 window centers from an in-memory elapsed array."""
    if elapsed is None or len(elapsed) == 0:
        return 0
    if not np.isfinite(elapsed).all():
        return 0
    values = np.unique(elapsed)
    if len(values) == 0:
        return 0
    history_offsets = np.array(
        [-(N_HISTORY_FRAMES - 1 - i) * HISTORY_INTERVAL_MIN for i in range(N_HISTORY_FRAMES)]
    )
    future_offsets = np.array([(i + 1) * HORIZON_INTERVAL_MIN for i in range(N_HORIZON_STEPS)])
    needed = np.concatenate([
        values[:, None] + history_offsets[None, :],
        values[:, None] + future_offsets[None, :],
    ], axis=1)
    sorted_vals = np.sort(values)
    indices = np.searchsorted(sorted_vals, needed)
    # Check both idx and idx-1 since the closest value might be on either side.
    def _close_at(idx_arr: np.ndarray) -> np.ndarray:
        clipped = np.clip(idx_arr, 0, len(sorted_vals) - 1)
        return np.isclose(sorted_vals[clipped], needed, atol=TIME_ATOL_MIN, rtol=0.0)
    valid = _close_at(indices)
    valid_prev = _close_at(indices - 1)
    valid = valid | valid_prev
    return int(np.all(valid, axis=1).sum())


def _read_file_once(path: Path, chunk_size: int = 4 * 1024 * 1024) -> tuple[str, bytes]:
    """Read file once, return (sha256, raw_bytes)."""
    h = hashlib.sha256()
    parts: list[bytes] = []
    with path.open("rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            h.update(block)
            parts.append(block)
    return h.hexdigest(), b"".join(parts)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _active_network_sha(project_root: Path) -> str:
    path = project_root / "data" / "wuhan_v8_storage_retrofit.inp"
    if not path.exists():
        raise FileNotFoundError(path)
    return _sha256_file(path)


def _normalise_role(raw: str) -> str:
    text = str(raw).strip().lower()
    return BRANCH_ALIASES.get(text, text)


def _infer_role_from_filename(path: Path) -> str:
    name = path.name.lower()
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


def _source_experiment(outputs_root: Path, path: Path) -> str:
    try:
        rel = path.resolve().relative_to(outputs_root.resolve())
        parts = rel.parts
    except Exception:
        parts = path.parts
    if not parts:
        return "unknown"
    # Preserve enough hierarchy to distinguish final_v4/pilot from frozen copies.
    return "/".join(parts[: min(4, len(parts))])


def _source_role(path: Path) -> str:
    lowered = "/".join(part.lower() for part in path.parts)
    # Only the new V4.2 paper-workflow evaluation tree can remain reserved.
    if "v42_paper" in lowered and ("formal_blind" in lowered or "challenge" in lowered):
        return "reserved_evaluation"
    if any(token in lowered for token in ("formal_blind", "locked_validation", "calibration")):
        return "consumed_development"
    if "frozen_evidence" in lowered:
        return "frozen_lineage_copy"
    return "development"


def _infer_checkpoint_min(payload: dict[str, Any], case_id: str) -> float | None:
    for key in ("checkpoint_min", "checkpoint_elapsed_min", "checkpoint"):
        value = payload.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                pass
    match = re.search(r"__(-?\d+(?:\.\d+)?)__", case_id)
    if match:
        return float(match.group(1))
    return None


def _branch_detail_from_completion(
    completion_path: Path,
    payload: dict[str, Any],
) -> list[tuple[str, Path]]:
    branches = payload.get("branches")
    out: list[tuple[str, Path]] = []
    if not isinstance(branches, dict):
        return out
    for raw_role, value in branches.items():
        role = _normalise_role(raw_role)
        if isinstance(value, str):
            text = value
        elif isinstance(value, dict):
            text = str(
                value.get("detail_path")
                or value.get("path")
                or value.get("detail")
                or ""
            )
        else:
            text = ""
        if not text:
            continue
        candidate = Path(text)
        if not candidate.is_absolute():
            candidate = completion_path.parent / candidate
        if not candidate.exists():
            local = completion_path.parent / Path(text).name
            if local.exists():
                candidate = local
        if candidate.exists():
            out.append((role, candidate.resolve()))
    return out


def _action_sha(path: Path, facility_ids: list[str], _df: pd.DataFrame | None = None) -> str:
    """Hash the authoritative readback action sequence if available.

    Empty string means the historical file has no complete Engineering36
    readback contract; it is never replaced by requested actions.
    """
    if _df is not None:
        header = _df.iloc[:0]
    else:
        header = pd.read_csv(path, nrows=0)
    lookup = {str(c)[len("setting:"):].casefold(): str(c) for c in header.columns if str(c).startswith("setting:")}
    required = [lookup.get(fid.casefold()) for fid in facility_ids]
    if any(c is None for c in required):
        return ""
    usecols = [str(c) for c in required]
    if _df is not None:
        if not all(c in _df.columns for c in usecols):
            return ""
        numeric = _df[usecols].apply(pd.to_numeric, errors="coerce")
    else:
        read_cols = usecols[:]
        if "elapsed_min" in header.columns:
            read_cols = ["elapsed_min"] + read_cols
        df = pd.read_csv(path, usecols=read_cols)
        numeric = df.apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        return ""
    arr = numeric.to_numpy(dtype=np.float64)
    return hashlib.sha256(arr.tobytes(order="C")).hexdigest()


def _target_columns(
    *,
    node_ids: list[str],
    storage_ids: list[str],
    facility_ids: list[str],
    outfall_ids: list[str],
) -> dict[str, list[str]]:
    return {
        "node_depth": [f"h:{x}" for x in node_ids],
        "hydraulic_head": [f"head:{x}" for x in node_ids],
        "node_flooding_rate": [f"flood:{x}" for x in node_ids],
        "storage_volume": [f"storage_volume:{x}" for x in storage_ids],
        "managed_facility_flow": [f"flow:{x}" for x in facility_ids],
        "outfall_flow": [f"outfall_flow:{x}" for x in outfall_ids],
        "readback_setting": [f"setting:{x}" for x in facility_ids],
        "rainfall": ["rainfall_mm_h"],
    }


def _casefold_columns(columns: Iterable[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for col in columns:
        text = str(col)
        key = text.casefold()
        if key not in out:
            out[key] = text
    return out


def _all_columns_present(lookup: dict[str, str], expected: Iterable[str]) -> bool:
    expected_list = list(expected)
    return bool(expected_list) and all(str(x).casefold() in lookup for x in expected_list)


def _depth_semantics(
    path: Path,
    *,
    node_ids: list[str],
    invert_by_id: dict[str, float],
    _df: pd.DataFrame | None = None,
) -> str:
    """Determine whether h is depth and head is hydraulic head from real values."""
    if _df is not None:
        lookup = _casefold_columns(_df.columns)
    else:
        header = pd.read_csv(path, nrows=0)
        lookup = _casefold_columns(header.columns)
    sample_ids = [x for x in node_ids if f"h:{x}".casefold() in lookup and f"head:{x}".casefold() in lookup][:8]
    if not sample_ids:
        return "unknown"
    usecols: list[str] = []
    for nid in sample_ids:
        usecols.extend([lookup[f"h:{nid}".casefold()], lookup[f"head:{nid}".casefold()]])
    if _df is not None:
        df = _df[usecols].head(32)
    else:
        df = pd.read_csv(path, usecols=usecols, nrows=32)
    diffs_a: list[float] = []
    diffs_b: list[float] = []
    for nid in sample_ids:
        h = pd.to_numeric(df[lookup[f"h:{nid}".casefold()]], errors="coerce").to_numpy(float)
        head = pd.to_numeric(df[lookup[f"head:{nid}".casefold()]], errors="coerce").to_numpy(float)
        inv = float(invert_by_id[nid.casefold()])
        finite = np.isfinite(h) & np.isfinite(head)
        if finite.any():
            diffs_a.extend(np.abs((head[finite] - h[finite]) - inv).tolist())
            diffs_b.extend(np.abs((h[finite] - head[finite]) - inv).tolist())
    if not diffs_a:
        return "unknown"
    med_a = float(np.median(diffs_a))
    med_b = float(np.median(diffs_b))
    if med_a <= 1.0e-3 and med_a < med_b:
        return "h_is_depth_head_is_hydraulic_head"
    if med_b <= 1.0e-3 and med_b < med_a:
        return "head_is_depth_h_is_hydraulic_head"
    return "ambiguous"


def _time_coverage(path: Path, checkpoint_min: float | None, _elapsed: np.ndarray | None = None) -> tuple[bool, bool]:
    if checkpoint_min is None:
        return False, False
    if _elapsed is not None:
        elapsed = _elapsed
    else:
        header = pd.read_csv(path, nrows=0)
        if "elapsed_min" not in header.columns:
            return False, False
        elapsed = pd.to_numeric(pd.read_csv(path, usecols=["elapsed_min"])["elapsed_min"], errors="coerce").to_numpy(float)
    if not np.isfinite(elapsed).all():
        return False, False
    history_times = np.asarray(
        [checkpoint_min - (N_HISTORY_FRAMES - 1 - i) * HISTORY_INTERVAL_MIN for i in range(N_HISTORY_FRAMES)],
        dtype=float,
    )
    future_times = np.asarray(
        [checkpoint_min + (i + 1) * HORIZON_INTERVAL_MIN for i in range(N_HORIZON_STEPS)],
        dtype=float,
    )

    def present(targets: np.ndarray) -> bool:
        for value in targets:
            if int(np.sum(np.isclose(elapsed, value, atol=TIME_ATOL_MIN, rtol=0.0))) != 1:
                return False
        return True

    return present(history_times), present(future_times)


def _finite_target_check(path: Path, expected: dict[str, list[str]], _df: pd.DataFrame | None = None) -> bool:
    """Full finite check for columns that are actually present.

    Callers can disable this expensive pass during inventory discovery.  A
    metadata-only result is never represented as a finite scientific pass.
    """
    if _df is not None:
        lookup = _casefold_columns(_df.columns)
    else:
        header = pd.read_csv(path, nrows=0)
        lookup = _casefold_columns(header.columns)
    cols: list[str] = []
    for names in expected.values():
        for name in names:
            actual = lookup.get(name.casefold())
            if actual is not None:
                cols.append(actual)
    cols = sorted(set(cols))
    if not cols:
        return False
    if _df is not None:
        numeric = _df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        return bool(np.isfinite(numeric).all())
    for chunk in pd.read_csv(path, usecols=cols, chunksize=64):
        numeric = chunk.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(numeric).all():
            return False
    return True


def _incoming_links_by_outfall(project_root: Path, outfall_ids: list[str]) -> dict[str, list[str]]:
    _, links = _parse_inp_topology(project_root / "data" / "wuhan_v8_storage_retrofit.inp")
    out: dict[str, list[str]] = {x.casefold(): [] for x in outfall_ids}
    for row in links.itertuples(index=False):
        key = str(row.to_node).casefold()
        if key in out:
            out[key].append(str(row.link_id))
    return out


def _audit_target_availability(
    path: Path,
    *,
    node_ids: list[str],
    storage_ids: list[str],
    facility_ids: list[str],
    outfall_ids: list[str],
    invert_by_id: dict[str, float],
    incoming_links: dict[str, list[str]],
    checkpoint_min: float | None,
    full_finite_check: bool,
    _df: pd.DataFrame | None = None,
) -> tuple[TargetAvailability, tuple[str, ...]]:
    if _df is not None:
        lookup = _casefold_columns(_df.columns)
    else:
        header = pd.read_csv(path, nrows=0)
        lookup = _casefold_columns(header.columns)
    expected = _target_columns(
        node_ids=node_ids,
        storage_ids=storage_ids,
        facility_ids=facility_ids,
        outfall_ids=outfall_ids,
    )
    present = {name: _all_columns_present(lookup, cols) for name, cols in expected.items()}
    elapsed = None
    if _df is not None and "elapsed_min" in _df.columns:
        elapsed = pd.to_numeric(_df["elapsed_min"], errors="coerce").to_numpy(float)
    history_complete, horizon_complete = _time_coverage(path, checkpoint_min, _elapsed=elapsed)
    semantics = _depth_semantics(path, node_ids=node_ids, invert_by_id=invert_by_id, _df=_df)
    incoming_ok = bool(outfall_ids)
    for outfall in outfall_ids:
        links = incoming_links.get(outfall.casefold(), [])
        if not links or not all(f"flow:{link}".casefold() in lookup for link in links):
            incoming_ok = False
            break
    finite_pass = _finite_target_check(path, expected, _df=_df) if full_finite_check else False
    target = TargetAvailability(
        node_depth=present["node_depth"],
        hydraulic_head=present["hydraulic_head"],
        node_flooding_rate=present["node_flooding_rate"],
        storage_volume=present["storage_volume"],
        managed_facility_flow=present["managed_facility_flow"],
        outfall_flow=present["outfall_flow"],
        readback_setting=present["readback_setting"],
        rainfall=present["rainfall"],
        history_complete=history_complete,
        horizon_complete=horizon_complete,
        finite_checked=bool(full_finite_check),
        finite_pass=bool(finite_pass),
        depth_semantics=semantics,
        outfall_reconstruction_candidate=bool((not present["outfall_flow"]) and incoming_ok),
    )
    missing = tuple(name for name, ok in present.items() if not ok)
    return target, missing


def _infer_domain_id(
    path: Path,
    *,
    network_sha: str,
    active_network_sha: str,
) -> str:
    lowered = "/".join(part.lower() for part in path.parts)
    if network_sha and network_sha == active_network_sha:
        return "target_no_dwf"
    if "no_dwf" in lowered:
        return "target_no_dwf_variant"
    # Do not mistake the token 'dwf' inside 'no_dwf'.
    if "dwf" in lowered and "no_dwf" not in lowered:
        return "source_dwf"
    return "source_domain_unknown"


def _identity_sha(
    *,
    detail_sha: str,
    network_sha: str,
    rainfall_sha: str,
    checkpoint_min: float | None,
    branch_role: str,
    action_sha: str,
    domain_id: str,
) -> str:
    payload = {
        "detail_sha256": detail_sha,
        "network_sha256": network_sha,
        "rainfall_sha256": rainfall_sha,
        "checkpoint_min": checkpoint_min,
        "branch_role": branch_role,
        "actual_readback_action_sha256": action_sha,
        "domain_id": domain_id,
    }
    return _sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":")))


# ---------------------------------------------------------------------------
# Discovery adapters
# ---------------------------------------------------------------------------


def _completion_payload(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}


# Directories that must never contribute to the reusable pool.
_SKIP_DIR_NAMES = frozenset({
    "formal_blind", "challenge", "formal_evaluation",
    "locked_validation_b_timeseries", "calibration_a_timeseries",
})


def _walk_skip_dir(dirpath: str) -> bool:
    """Return True if this directory should be pruned from discovery."""
    parts = dirpath.replace("\\", "/").lower().split("/")
    return bool(_SKIP_DIR_NAMES & set(parts))


def _fast_rglob(root: Path, pattern: str) -> list[Path]:
    """Fast recursive glob that prunes reserved directories."""
    results: list[Path] = []
    root_str = str(root.resolve())
    for dirpath, dirnames, filenames in os.walk(root_str):
        # Prune reserved / irrelevant subtrees in-place.
        dirnames[:] = [
            d for d in dirnames
            if d.lower() not in _SKIP_DIR_NAMES
            and not _walk_skip_dir(os.path.join(dirpath, d))
        ]
        for fn in filenames:
            if fn == pattern or (pattern.startswith("*") and fn.endswith(pattern[1:])):
                results.append(Path(dirpath) / fn)
    return results


def _discover_completion_details(outputs_root: Path) -> tuple[list[dict[str, Any]], set[Path]]:
    rows: list[dict[str, Any]] = []
    referenced: set[Path] = set()
    for completion_path in _fast_rglob(outputs_root, "completion.json"):
        payload = _completion_payload(completion_path)
        case_id = str(payload.get("case_id") or payload.get("event_id") or completion_path.parent.name)
        event_id = str(payload.get("event_id") or case_id.split("__")[0])
        checkpoint_min = _infer_checkpoint_min(payload, case_id)
        network_sha = str(payload.get("network_sha256") or payload.get("network_sha") or "")
        rainfall_sha = str(payload.get("rainfall_sha256") or payload.get("rainfall_sha") or "")
        branch_paths = _branch_detail_from_completion(completion_path, payload)
        for role, detail in branch_paths:
            referenced.add(detail.resolve())
            rows.append(
                {
                    "completion_path": completion_path.resolve(),
                    "detail_path": detail.resolve(),
                    "case_id": case_id,
                    "event_id": event_id,
                    "checkpoint_min": checkpoint_min,
                    "network_sha256": network_sha,
                    "rainfall_sha256": rainfall_sha,
                    "branch_role": role,
                    "completion_status": str(payload.get("status") or ("pass" if payload.get("gate3_h120_pass") else "unknown")),
                    "prefix_hash_match": payload.get("prefix_history_hash_match"),
                    "checkpoint_hash_match": payload.get("checkpoint_pre_action_hash_match"),
                }
            )
    return rows, referenced


def _discover_orphan_details(outputs_root: Path, referenced: set[Path]) -> list[dict[str, Any]]:
    """Discover legacy/PFV-first/closed-loop details not represented by completion.json."""
    rows: list[dict[str, Any]] = []
    seen: set[Path] = set()
    root_str = str(outputs_root.resolve())
    for dirpath, dirnames, filenames in os.walk(root_str):
        dirnames[:] = [
            d for d in dirnames
            if d.lower() not in _SKIP_DIR_NAMES
            and not _walk_skip_dir(os.path.join(dirpath, d))
        ]
        for fn in filenames:
            if fn == "detail.csv" or fn.endswith("_detail.csv"):
                detail = Path(dirpath) / fn
                path = detail.resolve()
                if path in seen or path in referenced:
                    continue
                seen.add(path)
                role = _infer_role_from_filename(path)
                rows.append(
                    {
                        "completion_path": None,
                        "detail_path": path,
                        "case_id": path.parent.name,
                        "event_id": path.parent.name,
                        "checkpoint_min": None,
                        "network_sha256": "",
                        "rainfall_sha256": "",
                        "branch_role": role,
                        "completion_status": "legacy_detail_only",
                        "prefix_hash_match": None,
                        "checkpoint_hash_match": None,
                    }
                )
    return rows


def discover_existing_details(outputs_root: str | Path) -> pd.DataFrame:
    """Return all discovered physical-detail candidates across heterogeneous layouts."""
    root = Path(outputs_root)
    if not root.exists():
        raise FileNotFoundError(root)
    completion_rows, referenced = _discover_completion_details(root)
    orphan_rows = _discover_orphan_details(root, referenced)
    rows = completion_rows + orphan_rows
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Audit and classification
# ---------------------------------------------------------------------------


def _classify_case(group: pd.DataFrame) -> CaseReuseRecord:
    first = group.iloc[0]
    roles = set(str(x) for x in group["branch_role"])
    four_complete = all(role in roles for role in FOUR_BRANCH_ROLES)
    role_rows = {str(row.branch_role): row for row in group.itertuples(index=False)}
    required_rows = [role_rows.get(role) for role in FOUR_BRANCH_ROLES]
    full_targets = bool(four_complete and all(bool(getattr(row, "formal_all_target_complete")) for row in required_rows if row is not None))
    core_targets = bool(four_complete and all(bool(getattr(row, "core_trajectory_complete")) for row in required_rows if row is not None))
    outfall_only = bool(
        core_targets
        and not full_targets
        and all(
            bool(getattr(row, "missing_outfall_only"))
            for row in required_rows
            if row is not None
        )
    )
    outfall_reconstruct = bool(
        outfall_only
        and all(
            bool(getattr(row, "outfall_reconstruction_candidate"))
            for row in required_rows
            if row is not None
        )
    )

    reasons: list[str] = []
    domain = str(first["domain_id"])
    source_role = str(first["source_role"])
    completion_bad = any(str(x).lower() in {"failed", "error"} for x in group["completion_status"])
    ambiguous_depth = any(str(x) == "ambiguous" for x in group["depth_semantics"])
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

    if completion_bad or ambiguous_depth or source_role == "reserved_evaluation":
        classification = ReuseClassification.INVALID_OR_INCOMPATIBLE
    elif domain == "source_dwf":
        if core_targets or any(bool(x) for x in group["core_trajectory_complete"]):
            classification = ReuseClassification.SOURCE_DOMAIN_REUSE
        else:
            classification = ReuseClassification.RERUN_REQUIRED
    elif full_targets:
        classification = ReuseClassification.FULL_REUSE
    elif outfall_only:
        # Even when all incoming-link flows exist, do not silently promote to
        # REUSE_AFTER_EXTRACTION until a new explicit-outfall recorder validates
        # the deterministic reconstruction rule.
        classification = ReuseClassification.PARTIAL_AUX_REUSE
    elif any(bool(x) for x in group["core_trajectory_complete"]):
        classification = ReuseClassification.PARTIAL_AUX_REUSE
    elif any(bool(x) for x in group["available_node_depth"]) and any(bool(x) for x in group["available_node_flooding_rate"]):
        classification = ReuseClassification.PARTIAL_AUX_REUSE
    else:
        classification = ReuseClassification.RERUN_REQUIRED

    case_uid = _sha256_text(
        json.dumps(
            {
                "case_id": str(first["case_id"]),
                "event_id": str(first["event_id"]),
                "rainfall_sha": str(first["rainfall_sha256"]),
                "checkpoint_min": None if pd.isna(first["checkpoint_min"]) else float(first["checkpoint_min"]),
                "network_sha": str(first["network_sha256"]),
                "domain_id": domain,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return CaseReuseRecord(
        case_uid=case_uid,
        case_id=str(first["case_id"]),
        event_id=str(first["event_id"]),
        rainfall_sha256=str(first["rainfall_sha256"]),
        checkpoint_min=None if pd.isna(first["checkpoint_min"]) else float(first["checkpoint_min"]),
        network_sha256=str(first["network_sha256"]),
        domain_id=domain,
        source_role=source_role,
        source_experiment=str(first["source_experiment"]),
        branch_count=int(len(group)),
        four_reference_complete=bool(four_complete),
        same_forcing_pass=None,
        full_reuse_targets=bool(full_targets),
        core_trajectory_targets=bool(core_targets),
        outfall_only_blocker=bool(outfall_only),
        outfall_reconstruction_candidate=bool(outfall_reconstruct),
        classification=classification,
        reason_codes=tuple(reasons),
        branch_physical_ids=tuple(sorted(set(str(x) for x in group["physical_identity_sha256"]))),
    )


def _audit_single_row(
    row,
    *,
    outputs_root: Path,
    node_ids: list[str],
    storage_ids: list[str],
    facility_ids: list[str],
    outfall_ids: list[str],
    invert_by_id: dict[str, float],
    incoming_links: dict[str, list[str]],
    active_sha: str,
    full_finite_check: bool,
    detail_cache: dict,
    cache_lock: threading.Lock,
) -> PhysicalRunRecord:
    """Audit one discovery row — designed to run in a thread pool."""
    path = Path(row.detail_path)
    reasons: list[str] = []
    window_anchor = 0
    try:
        detail_sha, raw_bytes = _read_file_once(path)
        df = pd.read_csv(io.BytesIO(raw_bytes))
        # Pre-compute window_anchor_count while df is in memory.
        if "elapsed_min" in df.columns:
            elapsed_arr = pd.to_numeric(df["elapsed_min"], errors="coerce").to_numpy(float)
            window_anchor = _window_anchor_count_from_elapsed(elapsed_arr)
        action_sha = _action_sha(path, facility_ids, _df=df)
        checkpoint = None if row.checkpoint_min is None or (isinstance(row.checkpoint_min, float) and math.isnan(row.checkpoint_min)) else float(row.checkpoint_min)
        cache_key = (detail_sha, checkpoint)
        with cache_lock:
            cached = detail_cache.get(cache_key)
        if cached is not None:
            target, missing, cached_action, cached_wac = cached
            if not action_sha:
                action_sha = cached_action
            if not window_anchor:
                window_anchor = cached_wac
        else:
            target, missing = _audit_target_availability(
                path,
                node_ids=node_ids,
                storage_ids=storage_ids,
                facility_ids=facility_ids,
                outfall_ids=outfall_ids,
                invert_by_id=invert_by_id,
                incoming_links=incoming_links,
                checkpoint_min=checkpoint,
                full_finite_check=full_finite_check,
                _df=df,
            )
            with cache_lock:
                detail_cache[cache_key] = (target, missing, action_sha, window_anchor)
    except Exception as exc:
        detail_sha = ""
        action_sha = ""
        target = TargetAvailability()
        missing = (
            "node_depth",
            "node_flooding_rate",
            "storage_volume",
            "managed_facility_flow",
            "outfall_flow",
            "readback_setting",
            "rainfall",
        )
        reasons.append(f"detail_audit_error:{type(exc).__name__}:{exc}")
        checkpoint = None if row.checkpoint_min is None else row.checkpoint_min

    network_sha = str(row.network_sha256 or "")
    domain_id = _infer_domain_id(path, network_sha=network_sha, active_network_sha=active_sha)
    active_match = None if not network_sha else bool(network_sha == active_sha)
    if target.depth_semantics == "ambiguous":
        reasons.append("ambiguous_h_head_semantics")
    if not action_sha:
        reasons.append("engineering36_readback_incomplete")
    physical_id = _identity_sha(
        detail_sha=detail_sha,
        network_sha=network_sha,
        rainfall_sha=str(row.rainfall_sha256 or ""),
        checkpoint_min=checkpoint,
        branch_role=str(row.branch_role),
        action_sha=action_sha,
        domain_id=domain_id,
    )
    return PhysicalRunRecord(
        source_root=str(outputs_root),
        source_experiment=_source_experiment(outputs_root, path),
        run_dir=str(path.parent),
        completion_path=None if row.completion_path is None else str(row.completion_path),
        detail_path=str(path),
        detail_sha256=detail_sha,
        detail_size_bytes=int(path.stat().st_size) if path.exists() else 0,
        case_id=str(row.case_id),
        event_id=str(row.event_id),
        rainfall_sha256=str(row.rainfall_sha256 or ""),
        checkpoint_min=checkpoint,
        branch_role=str(row.branch_role),
        network_sha256=network_sha,
        active_network_sha_match=active_match,
        domain_id=domain_id,
        source_role=_source_role(path),
        action_readback_sha256=action_sha,
        physical_identity_sha256=physical_id,
        completion_status=str(row.completion_status),
        prefix_hash_match=row.prefix_hash_match,
        checkpoint_hash_match=row.checkpoint_hash_match,
        target=target,
        missing_target_groups=missing,
        audit_reasons=tuple(reasons),
        window_anchor_count=window_anchor,
    )


def audit_existing_swmm_pool(
    *,
    project_root: str | Path,
    outputs_root: str | Path,
    full_finite_check: bool = False,
    max_workers: int = 16,
) -> ExistingPoolAuditResult:
    """Discover and audit all historical Project6 SWMM detail evidence.

    ``full_finite_check=False`` is an efficient metadata pass.  Scientific
    admission must later run with ``True`` for the rows selected for training.
    """
    project_root = Path(project_root)
    outputs_root = Path(outputs_root)
    t0 = time.monotonic()
    print("[R0] discovering details …", file=sys.stderr, flush=True)
    discovery = discover_existing_details(outputs_root)
    if discovery.empty:
        raise FileNotFoundError(f"no detail.csv or *_detail.csv found under {outputs_root}")
    print(f"[R0] discovered {len(discovery)} details in {time.monotonic()-t0:.1f}s", file=sys.stderr, flush=True)

    graph = _load_graph_topology(project_root)
    node_ids = list(graph["node_ids"])
    facility_ids = _load_engineering36_ids(project_root)
    nodes, _ = _parse_inp_topology(project_root / "data" / "wuhan_v8_storage_retrofit.inp")
    storage_ids = [str(x) for x in nodes.loc[nodes["node_type"] == "storage", "node_id"].tolist()]
    outfall_ids = [str(x) for x in nodes.loc[nodes["node_type"] == "outfall", "node_id"].tolist()]
    invert_by_id = {str(row.node_id).casefold(): float(row.invert) for row in nodes.itertuples(index=False)}
    incoming_links = _incoming_links_by_outfall(project_root, outfall_ids)
    active_sha = _active_network_sha(project_root)

    records: list[PhysicalRunRecord] = []
    detail_cache: dict[tuple[str, float | None], tuple[TargetAvailability, tuple[str, ...], str]] = {}
    cache_lock = threading.Lock()

    rows_list = list(discovery.itertuples(index=False))
    total = len(rows_list)
    done_count = 0
    t1 = time.monotonic()
    print(f"[R0] auditing {total} files with {max_workers} threads …", file=sys.stderr, flush=True)

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(
                _audit_single_row,
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
                rate = done_count / elapsed if elapsed > 0 else 0
                eta = (total - done_count) / rate if rate > 0 else 0
                print(
                    f"[R0] {done_count}/{total}  elapsed={elapsed:.0f}s  "
                    f"rate={rate:.1f}/s  eta={eta:.0f}s  "
                    f"cache={len(detail_cache)}",
                    file=sys.stderr, flush=True,
                )

    physical_all = pd.DataFrame([record.as_dict() for record in records])
    # Add convenient target-level columns used by case classification.
    physical_all["core_trajectory_complete"] = (
        physical_all["available_node_depth"]
        & physical_all["available_node_flooding_rate"]
        & physical_all["available_storage_volume"]
        & physical_all["available_managed_facility_flow"]
        & physical_all["available_readback_setting"]
        & physical_all["available_rainfall"]
        & physical_all["available_history_complete"]
        & physical_all["available_horizon_complete"]
    )
    physical_all["formal_all_target_complete"] = physical_all["core_trajectory_complete"] & physical_all["available_outfall_flow"]
    physical_all["missing_outfall_only"] = physical_all["core_trajectory_complete"] & ~physical_all["available_outfall_flow"]
    physical_all["outfall_reconstruction_candidate"] = physical_all["available_outfall_reconstruction_candidate"]

    duplicate_mask = physical_all.duplicated("physical_identity_sha256", keep=False)
    duplicate_lineage = physical_all.loc[duplicate_mask].copy()
    # Canonical physical-run table keeps one row while duplicate_lineage preserves provenance.
    canonical = physical_all.sort_values(["physical_identity_sha256", "source_role", "detail_path"]).drop_duplicates("physical_identity_sha256", keep="first").reset_index(drop=True)

    # Case grouping uses logical evidence (not canonical rows) so four references remain visible.
    case_group_cols = [
        "case_id",
        "event_id",
        "rainfall_sha256",
        "checkpoint_min",
        "network_sha256",
        "domain_id",
        "source_experiment",
        "source_role",
    ]
    cases: list[CaseReuseRecord] = []
    for _, group in physical_all.groupby(case_group_cols, dropna=False, sort=False):
        cases.append(_classify_case(group))
    case_df = pd.DataFrame([case.as_dict() for case in cases])

    source_summary = (
        case_df.groupby(["source_experiment", "classification"], dropna=False)
        .size()
        .rename("case_count")
        .reset_index()
        if not case_df.empty
        else pd.DataFrame(columns=["source_experiment", "classification", "case_count"])
    )
    target_cols = [
        "available_node_depth",
        "available_node_flooding_rate",
        "available_storage_volume",
        "available_managed_facility_flow",
        "available_outfall_flow",
        "available_readback_setting",
        "available_rainfall",
        "available_history_complete",
        "available_horizon_complete",
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
        target_summary["available_physical_runs"] / target_summary["total_physical_runs"],
        np.nan,
    )

    class_counts = case_df["classification"].value_counts().to_dict() if not case_df.empty else {}
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
        "outfall_policy": "explicit target required for FULL_REUSE; incoming-link coverage is candidate-only until independent validation",
        "missing_targets_are_imputed": False,
    }
    return ExistingPoolAuditResult(
        physical_runs=canonical,
        cases=case_df,
        duplicate_lineage=duplicate_lineage,
        source_summary=source_summary,
        target_summary=target_summary,
        summary=summary,
    )


def write_existing_pool_audit(result: ExistingPoolAuditResult, output_dir: str | Path) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "physical_csv": output_dir / "physical_run_inventory.csv",
        "physical_parquet": output_dir / "physical_run_inventory.parquet",
        "case_csv": output_dir / "target_coverage_by_case.csv",
        "duplicate_csv": output_dir / "duplicate_lineage_audit.csv",
        "source_summary_csv": output_dir / "source_reuse_summary.csv",
        "target_summary_csv": output_dir / "target_coverage_by_branch.csv",
        "summary_json": output_dir / "data_reuse_audit.json",
        "partial_csv": output_dir / "partial_aux_reuse.csv",
        "rerun_csv": output_dir / "rerun_required.csv",
        "invalid_csv": output_dir / "invalid_or_incompatible.csv",
        "source_domain_csv": output_dir / "source_domain_reuse.csv",
    }
    result.physical_runs.to_csv(paths["physical_csv"], index=False)
    result.physical_runs.to_parquet(paths["physical_parquet"], index=False)
    result.cases.to_csv(paths["case_csv"], index=False)
    result.duplicate_lineage.to_csv(paths["duplicate_csv"], index=False)
    result.source_summary.to_csv(paths["source_summary_csv"], index=False)
    result.target_summary.to_csv(paths["target_summary_csv"], index=False)
    paths["summary_json"].write_text(json.dumps(result.summary, indent=2, allow_nan=False), encoding="utf-8")
    result.cases.loc[result.cases["classification"] == ReuseClassification.PARTIAL_AUX_REUSE.value].to_csv(paths["partial_csv"], index=False)
    result.cases.loc[result.cases["classification"] == ReuseClassification.RERUN_REQUIRED.value].to_csv(paths["rerun_csv"], index=False)
    result.cases.loc[result.cases["classification"] == ReuseClassification.INVALID_OR_INCOMPATIBLE.value].to_csv(paths["invalid_csv"], index=False)
    result.cases.loc[result.cases["classification"] == ReuseClassification.SOURCE_DOMAIN_REUSE.value].to_csv(paths["source_domain_csv"], index=False)
    return paths
