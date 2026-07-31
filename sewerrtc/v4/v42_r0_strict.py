"""Strict compatibility wrapper for the optimized V4.2 Phase-R0 audit.

The optimized reader in :mod:`v42_existing_pool_audit` is intentionally kept as
the high-throughput implementation.  This module restores two scientific
semantics that must not change as a consequence of I/O optimization:

* the authoritative Engineering36 action hash includes ``elapsed_min`` when the
  historical file contains it, exactly as the pre-optimization implementation;
* previously revealed Calibration/Locked/Formal evidence is discovered and then
  classified as ``consumed_development``.  It is not silently pruned from the
  evidence inventory.  New V4.2 Challenge/Formal-Blind material is still
  rejected by the existing ``source_role`` classification.

It also sorts all returned tables deterministically so a threaded scan does not
make scientific manifests depend on future-completion order.
"""
from __future__ import annotations

import hashlib
import threading
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from . import v42_existing_pool_audit as base


_PATCH_LOCK = threading.Lock()


def _action_sha_semantic_compatible(
    path: Path,
    facility_ids: list[str],
    _df: pd.DataFrame | None = None,
) -> str:
    """Hash readback action using the frozen pre-optimization byte semantics.

    The old implementation hashed ``elapsed_min`` followed by Engineering36
    readback settings whenever ``elapsed_min`` existed.  The optimized in-memory
    branch accidentally omitted time, changing physical identities.  Keep the
    original semantic contract while still avoiding an extra disk read.
    """
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


@contextmanager
def _strict_runtime_semantics():
    """Temporarily restore strict semantics inside the optimized audit module."""
    with _PATCH_LOCK:
        old_action = base._action_sha
        old_skip = base._SKIP_DIR_NAMES
        try:
            base._action_sha = _action_sha_semantic_compatible
            # Do not prune historical revealed evaluation folders.  The base
            # _source_role() already distinguishes new reserved V4.2 evidence
            # from old consumed-development evidence and the pool builder later
            # excludes reserved_evaluation fail-closed.
            base._SKIP_DIR_NAMES = frozenset()
            yield
        finally:
            base._action_sha = old_action
            base._SKIP_DIR_NAMES = old_skip


def _sort_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    keys = [c for c in columns if c in frame.columns]
    if frame.empty or not keys:
        return frame.reset_index(drop=True)
    return frame.sort_values(keys, kind="mergesort").reset_index(drop=True)


def audit_existing_swmm_pool_strict(
    *,
    project_root: str | Path,
    outputs_root: str | Path,
    full_finite_check: bool = True,
    max_workers: int = 16,
) -> base.ExistingPoolAuditResult:
    """Run the optimized audit without changing frozen scientific semantics."""
    with _strict_runtime_semantics():
        result = base.audit_existing_swmm_pool(
            project_root=project_root,
            outputs_root=outputs_root,
            full_finite_check=bool(full_finite_check),
            max_workers=max(1, int(max_workers)),
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
            "historical_revealed_evidence_policy": "discover_then_tag_consumed_development",
            "threaded_output_deterministically_sorted": True,
        },
    )


write_existing_pool_audit = base.write_existing_pool_audit
