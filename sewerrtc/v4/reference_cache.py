"""Single-writer reference cache for Final V4 branch trajectories.

Every ``event_id + checkpoint_id + network/config/contract SHA`` identity runs
``no_control``, ``dynamic_internal_rules`` and ``hold_previous`` exactly once.  One
process writes the cache under an exclusive lock; everyone else waits and
reads.  Incomplete or identity-mismatched caches are never readable, and
reference branches are never counted as Pilot400/Train1600 samples.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

from .runtime import ReferenceWriteLock, atomic_write_json


REFERENCE_BRANCHES = ("no_control", "dynamic_internal_rules", "hold_previous")

COMPLETION_NAME = "reference_completion.json"


def reference_identity(
    *, network_sha256: str, config_sha256: str, contract_sha256: str
) -> dict:
    return {
        "network_sha256": str(network_sha256),
        "config_sha256": str(config_sha256),
        "contract_sha256": str(contract_sha256),
    }


def reference_identity_sha(identity: dict) -> str:
    canonical = json.dumps(
        identity, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def reference_dir(
    root: str | Path, event_id: str, checkpoint_id: str
) -> Path:
    return Path(root) / "references" / str(event_id) / str(checkpoint_id)


def write_reference_completion(
    root: str | Path,
    event_id: str,
    checkpoint_id: str,
    *,
    identity: dict,
    branch_artifacts: dict[str, list[str]],
) -> Path:
    """Freeze a finished reference cache atomically under a writer lock.

    ``branch_artifacts`` maps each branch to its artifact file names relative
    to the branch directory.  All three branches must exist with non-empty
    artifacts before the completion marker may be written.
    """
    missing_branches = set(REFERENCE_BRANCHES) - set(branch_artifacts)
    if missing_branches:
        raise ValueError(
            f"reference cache missing branches: {sorted(missing_branches)}"
        )
    directory = reference_dir(root, event_id, checkpoint_id)
    branches: dict[str, dict] = {}
    for branch in REFERENCE_BRANCHES:
        branch_dir = directory / branch
        artifacts = {}
        for name in branch_artifacts[branch]:
            artifact = branch_dir / name
            if not artifact.exists() or artifact.stat().st_size == 0:
                raise ValueError(
                    f"reference branch artifact missing or empty: {artifact}"
                )
            artifacts[name] = hashlib.sha256(
                artifact.read_bytes()
            ).hexdigest()
        if not artifacts:
            raise ValueError(f"reference branch has no artifacts: {branch}")
        branches[branch] = artifacts
    lock = ReferenceWriteLock(directory / ".reference.lock")
    with lock:
        atomic_write_json(
            directory / COMPLETION_NAME,
            {
                "status": "complete",
                "event_id": str(event_id),
                "checkpoint_id": str(checkpoint_id),
                "identity": dict(identity),
                "identity_sha256": reference_identity_sha(identity),
                "branches": branches,
                "counted_as_sample": False,
            },
        )
    return directory / COMPLETION_NAME


def read_reference_completion(
    root: str | Path,
    event_id: str,
    checkpoint_id: str,
    *,
    expected_identity: dict,
) -> dict:
    """Read a completed cache; refuse incomplete or mismatched caches."""
    directory = reference_dir(root, event_id, checkpoint_id)
    marker = directory / COMPLETION_NAME
    if not marker.exists():
        raise FileNotFoundError(f"reference cache incomplete: {marker}")
    payload = json.loads(marker.read_text(encoding="utf-8"))
    if payload.get("status") != "complete":
        raise ValueError(f"reference cache not complete: {marker}")
    expected_sha = reference_identity_sha(expected_identity)
    if payload.get("identity_sha256") != expected_sha:
        raise ValueError(
            "reference cache identity mismatch: "
            f"{payload.get('identity_sha256')} != {expected_sha}"
        )
    branches = payload.get("branches", {})
    if set(branches) != set(REFERENCE_BRANCHES):
        raise ValueError("reference cache branch set incomplete")
    for branch, artifacts in branches.items():
        for name, digest in artifacts.items():
            artifact = directory / branch / name
            if not artifact.exists():
                raise ValueError(f"reference artifact vanished: {artifact}")
            if hashlib.sha256(artifact.read_bytes()).hexdigest() != digest:
                raise ValueError(f"reference artifact SHA mismatch: {artifact}")
    return payload


def wait_for_reference(
    root: str | Path,
    event_id: str,
    checkpoint_id: str,
    *,
    expected_identity: dict,
    timeout_sec: float = 3600.0,
    poll_sec: float = 5.0,
) -> dict:
    """Non-writers block until the single writer publishes the cache."""
    deadline = time.monotonic() + float(timeout_sec)
    while True:
        try:
            return read_reference_completion(
                root,
                event_id,
                checkpoint_id,
                expected_identity=expected_identity,
            )
        except FileNotFoundError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(float(poll_sec))
