"""Pilot400 sample-task execution (spec section VI).

One RunPilot400 plan row is one four-branch sample.  The worker (1) makes
sure the three reference branches for the sample's checkpoint state exist in
the single-writer reference cache, (2) runs exactly one candidate branch,
(3) verifies all four branch artifacts, and (4) returns a per-branch payload
for the sample-level completion marker.  Cross-branch labels are never
computed inside the worker; reducers own every comparison.

Reference dedup keeps physical SWMM runs at 400 candidates plus at most 120
reference runs for the 1600 logical branch rows.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .pilot_candidates import PILOT_BRANCH_ROLES
from .reference_cache import (
    REFERENCE_BRANCHES,
    read_reference_completion,
    reference_dir,
    wait_for_reference,
    write_reference_completion,
)
from .runtime import ReferenceWriteLock, atomic_write_json
from .simulation import run_prepared_case


RESULT_NAME = "result.json"

BRANCH_CONTEXT_COLUMNS = (
    "branch_rows_json",
    "reference_root",
    "reference_identity_json",
)

def attach_branch_context(
    candidate_plan: pd.DataFrame,
    branch_plan: pd.DataFrame,
    *,
    reference_root: str | Path,
) -> pd.DataFrame:
    """Embed each sample's four branch rows into its executable plan row.

    The run worker executes in a separate process and only sees its own row,
    so the reference identity and the serialized branch requests must travel
    inside the row itself.
    """
    required = {
        "sample_id",
        "case_id",
        "branch_role",
        "runner_function",
        "runner_kwargs",
    }
    absent = required - set(branch_plan)
    if absent:
        raise ValueError(f"branch plan missing: {sorted(absent)}")
    grouped = {
        str(sample_id): group
        for sample_id, group in branch_plan.groupby("sample_id")
    }
    plan = candidate_plan.copy()
    contexts: list[str] = []
    identities: list[str] = []
    for _, row in plan.iterrows():
        sample_id = str(row["sample_id"])
        group = grouped.get(sample_id)
        if group is None or set(group["branch_role"].astype(str)) != set(
            PILOT_BRANCH_ROLES
        ):
            raise ValueError(
                f"sample is missing four-branch plan rows: {sample_id}"
            )
        contexts.append(
            json.dumps(
                [
                    {
                        "branch_role": str(branch["branch_role"]),
                        "case_id": str(branch["case_id"]),
                        "runner_function": str(branch["runner_function"]),
                        "runner_kwargs": str(branch["runner_kwargs"]),
                    }
                    for _, branch in group.iterrows()
                ]
            )
        )
        identities.append(
            json.dumps(
                {
                    "network_sha256": str(row.get("network_sha256", "")),
                    "config_sha256": str(row.get("config_sha256", "")),
                    "contract_sha256": str(row.get("contract_sha256", "")),
                    "checkpoint_state_sha256": str(
                        row.get("checkpoint_state_sha256", "")
                    ),
                },
                sort_keys=True,
            )
        )
    plan["branch_rows_json"] = contexts
    plan["reference_root"] = str(reference_root)
    plan["reference_identity_json"] = identities
    return plan


def _branch_paths(directory: Path) -> dict[str, str]:
    return {
        "directory": str(directory),
        "inp": str(directory / "case.inp"),
        "rpt": str(directory / "case.rpt"),
        "out": str(directory / "case.out"),
        "log": str(directory / "case.log"),
        "temporary": str(directory / "tmp"),
    }


def ensure_reference_cache(
    root: str | Path,
    event_id: str,
    checkpoint_id: str,
    *,
    identity: dict,
    branch_rows: dict[str, dict],
) -> tuple[dict, int]:
    """Guarantee the three reference branches; single writer, waiting readers.

    Returns ``(completion_payload, new_physical_runs)`` where the run count
    is 3 for the one writer that materializes the cache and 0 for everyone
    who reads or waits.
    """
    try:
        payload = read_reference_completion(
            root, event_id, checkpoint_id, expected_identity=identity
        )
        return payload, 0
    except FileNotFoundError:
        pass
    directory = reference_dir(root, event_id, checkpoint_id)
    directory.mkdir(parents=True, exist_ok=True)
    writer = ReferenceWriteLock(directory / ".writer.lock")
    if not writer.acquire():
        payload = wait_for_reference(
            root, event_id, checkpoint_id, expected_identity=identity
        )
        return payload, 0
    try:
        try:
            payload = read_reference_completion(
                root, event_id, checkpoint_id, expected_identity=identity
            )
            return payload, 0
        except FileNotFoundError:
            pass
        for branch in REFERENCE_BRANCHES:
            row = branch_rows.get(branch)
            if row is None:
                raise ValueError(f"missing reference branch row: {branch}")
            branch_dir = directory / branch
            branch_dir.mkdir(parents=True, exist_ok=True)
            outcome = run_prepared_case(row, _branch_paths(branch_dir))
            atomic_write_json(
                branch_dir / RESULT_NAME,
                {
                    "runner_function": outcome["runner_function"],
                    "result": outcome["result"],
                },
            )
        write_reference_completion(
            root,
            event_id,
            checkpoint_id,
            identity=identity,
            branch_artifacts={
                branch: ["detail.csv", RESULT_NAME]
                for branch in REFERENCE_BRANCHES
            },
        )
        payload = read_reference_completion(
            root, event_id, checkpoint_id, expected_identity=identity
        )
        return payload, len(REFERENCE_BRANCHES)
    finally:
        writer.release()


def run_pilot_sample(row: dict, paths: dict[str, str]) -> dict:
    """Sample task: 3 cached references + 1 candidate + artifact checks."""
    branch_rows = {
        str(item["branch_role"]): item
        for item in json.loads(str(row["branch_rows_json"]))
    }
    missing = set(PILOT_BRANCH_ROLES) - set(branch_rows)
    if missing:
        raise ValueError(f"sample task missing branches: {sorted(missing)}")
    root = str(row["reference_root"])
    identity = json.loads(str(row["reference_identity_json"]))
    event_id = str(row["event_id"])
    checkpoint_id = str(row["checkpoint_id"])
    _, new_reference_runs = ensure_reference_cache(
        root,
        event_id,
        checkpoint_id,
        identity=identity,
        branch_rows=branch_rows,
    )
    candidate = run_prepared_case(row, paths)
    branches: dict[str, dict] = {
        "candidate": {
            "status": "pass",
            "detail_path": str(candidate["detail_path"]),
            "result": candidate["result"],
            "runner_kwargs": str(row.get("runner_kwargs", "")),
        }
    }
    directory = reference_dir(root, event_id, checkpoint_id)
    for branch in REFERENCE_BRANCHES:
        branch_dir = directory / branch
        stored = json.loads(
            (branch_dir / RESULT_NAME).read_text(encoding="utf-8")
        )
        branches[branch] = {
            "status": "pass",
            "detail_path": str(branch_dir / "detail.csv"),
            "result": stored.get("result", {}),
            "runner_kwargs": str(branch_rows[branch].get("runner_kwargs", "")),
        }
    for branch, info in branches.items():
        artifact = Path(str(info["detail_path"]))
        if not artifact.exists() or artifact.stat().st_size == 0:
            raise ValueError(
                f"branch artifact missing or empty: {branch}: {artifact}"
            )
    return {
        "runner_function": candidate["runner_function"],
        "result": candidate["result"],
        "detail_path": str(candidate["detail_path"]),
        "branches": branches,
        "physical_swmm_runs": 1 + int(new_reference_runs),
    }


def expand_pilot_completions(completions: pd.DataFrame) -> pd.DataFrame:
    """Expand sample-level completions into per-branch reducer rows.

    A completion without a valid ``branches`` payload (failed or foreign
    marker) contributes nothing, so its sample simply stays pending.
    """
    if completions is None or completions.empty:
        return pd.DataFrame(columns=["case_id"])
    rows: list[dict] = []
    for _, row in completions.iterrows():
        branches = row.get("branches")
        if not isinstance(branches, dict):
            continue
        sample_id = str(row.get("sample_id", row.get("case_id", "")))
        sample_status = str(row.get("status", ""))
        for branch, info in branches.items():
            if not isinstance(info, dict):
                continue
            branch_status = (
                str(info.get("status", ""))
                if sample_status == "pass"
                else sample_status
            )
            rows.append(
                {
                    "case_id": f"{sample_id}__{branch}",
                    "sample_id": sample_id,
                    "branch_role": str(branch),
                    "status": branch_status,
                    "detail_path": str(info.get("detail_path", "")),
                    "result": info.get("result", {}),
                    "runner_kwargs": str(info.get("runner_kwargs", "")),
                    "rainfall_sha256": row.get("rainfall_sha256", ""),
                    "input_sha": row.get("input_sha", ""),
                }
            )
    return pd.DataFrame(rows)
