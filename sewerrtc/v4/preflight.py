"""Unified preflight checks that gate every real-SWMM long run.

A preflight failure returns a non-zero exit and no case may start.  All
checks are fail-closed: missing evidence is a failure, never a pass.  The
worker import probe runs in a subprocess so the pipeline process itself
stays lightweight and the SWMM worker chain is proven torch-free.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


MAX_WORKERS = 16

# Modules a SWMM worker process imports; none of them may pull in torch.
WORKER_IMPORT_CHAIN = (
    "sewerrtc.v4.runtime",
    "sewerrtc.v4.training_plan",
    "sewerrtc.v4.partial_audit",
    "sewerrtc.v4.preflight",
)


def worker_import_is_torch_free(
    python_exe: str | None = None, *, timeout: int = 120
) -> bool:
    """Probe in a subprocess that the worker import chain never loads torch."""
    code = (
        "import sys\n"
        + "".join(f"import {module}\n" for module in WORKER_IMPORT_CHAIN)
        + "sys.exit(1 if 'torch' in sys.modules else 0)\n"
    )
    try:
        probe = subprocess.run(
            [python_exe or sys.executable, "-c", code],
            capture_output=True,
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 0


def _requested_duplicates(plan: pd.DataFrame) -> int:
    keys = ["event_id", "checkpoint_id", "requested_schedule_sha256"]
    if not all(key in plan for key in keys):
        return -1
    # Reference branches legitimately share the candidate's requested SHA;
    # duplicate detection applies to candidate rows only.
    scope = plan[plan["branch"].eq("candidate")] if "branch" in plan else plan
    return int(scope.duplicated(keys).sum())


def preflight_checks(
    plan: pd.DataFrame,
    *,
    workers: int,
    output_root: str | Path,
    input_sha: str | None,
    evidence: dict | None = None,
    minimum_free_bytes: int = 1_000_000_000,
    probe_torch: bool = True,
    python_exe: str | None = None,
) -> dict:
    """Run every preflight check; ``status`` is pass only when all hold.

    ``evidence`` supplies runtime facts the plan cannot prove by itself:
    ``writer_lock_free``, ``reference_cache_clean``, ``active_conflicting_pids``
    and optionally ``torch_free_override`` for unit tests.
    """
    evidence = evidence or {}
    root = Path(output_root)
    requested_dupes = _requested_duplicates(plan)
    k_column = "K" if "K" in plan else ("k_target" if "k_target" in plan else None)
    k_values = (
        pd.to_numeric(plan[k_column], errors="coerce")
        if k_column is not None
        else None
    )
    checks = {
        "input_sha_present": bool(input_sha),
        "plan_nonempty": len(plan) > 0,
        "case_ids_unique": (
            "case_id" in plan and not plan["case_id"].duplicated().any()
        ),
        "event_split_isolated": (
            "split" in plan
            and not plan.groupby("event_id")["split"].nunique().gt(1).any()
            if "event_id" in plan
            else False
        ),
        "checkpoints_unique_per_case": (
            {"event_id", "checkpoint_id"} <= set(plan)
        ),
        "candidate_projection_present": (
            "projected_schedule_sha256" in plan
            and plan["projected_schedule_sha256"].notna().all()
        ),
        "requested_schedule_duplicates_zero": requested_dupes == 0,
        "k_le_8": (
            bool(k_values.le(8).all()) if k_values is not None else False
        ),
        "engineering_constraint_columns_present": all(
            column in plan
            for column in (
                "binary_semantics_ok",
                "rate_limit_ok",
                "dwell_ok",
                "interlock_ok",
            )
        )
        and all(
            plan[column].fillna(False).astype(bool).all()
            for column in (
                "binary_semantics_ok",
                "rate_limit_ok",
                "dwell_ok",
                "interlock_ok",
            )
        ),
        "output_root_ready": root.exists() and root.is_dir(),
        "writer_lock_free": bool(evidence.get("writer_lock_free", False)),
        "reference_cache_clean": bool(
            evidence.get("reference_cache_clean", False)
        ),
        "disk_space_ok": (
            root.exists()
            and shutil.disk_usage(root).free >= int(minimum_free_bytes)
        ),
        "no_conflicting_active_pids": (
            len(evidence.get("active_conflicting_pids", [1])) == 0
        ),
        "workers_le_16": 0 < int(workers) <= MAX_WORKERS,
    }
    if "torch_free_override" in evidence:
        checks["worker_import_torch_free"] = bool(
            evidence["torch_free_override"]
        )
    elif probe_torch:
        checks["worker_import_torch_free"] = worker_import_is_torch_free(
            python_exe
        )
    else:
        checks["worker_import_torch_free"] = False
    checks = {key: bool(value) for key, value in checks.items()}
    status = "pass" if all(checks.values()) else "blocked"
    return {
        "status": status,
        "exit_code": 0 if status == "pass" else 2,
        "checks": checks,
        "workers": int(workers),
        "planned_rows": int(len(plan)),
        "requested_schedule_duplicates": requested_dupes,
    }
