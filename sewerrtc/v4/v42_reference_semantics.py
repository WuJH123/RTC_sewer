"""Reference-branch semantics and equivalence diagnostics for Project6 V4.2.

The current Project6 No-control definition is *all Engineering36 facilities fully
open/on* for the complete prediction horizon.  This module makes that assumption
explicit and auditable.  It also detects action/hydraulic equivalence across the
four surrogate branches.  Equivalent branches are not automatically invalid:
Dynamic Internal can legitimately equal Hold when native rules are inactive, and
Candidate can equal Hold when the candidate makes no effective change.  The
important requirement is that the equality is real and reported, not silently
assumed.
"""
from __future__ import annotations

import hashlib
from itertools import combinations
from typing import Mapping

import numpy as np

BRANCHES = ("candidate", "no_control", "dynamic_internal", "hold_previous")


def no_control_all_open(action: np.ndarray, *, atol: float = 1.0e-7) -> bool:
    arr = np.asarray(action, dtype=float)
    return bool(
        arr.ndim == 2
        and arr.shape[0] >= 1
        and arr.shape[1] == 36
        and np.isfinite(arr).all()
        and np.allclose(arr, 1.0, atol=atol, rtol=0.0)
    )


def array_sha256(value: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(value, dtype=np.float64))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def branch_equivalence(
    actions: Mapping[str, np.ndarray],
    *,
    depths: Mapping[str, np.ndarray] | None = None,
    floods: Mapping[str, np.ndarray] | None = None,
    atol: float = 1.0e-7,
) -> dict:
    missing = [name for name in BRANCHES if name not in actions]
    if missing:
        raise KeyError(f"missing branch actions: {missing}")

    action_pairs: list[str] = []
    hydraulic_pairs: list[str] = []
    action_hashes = {name: array_sha256(actions[name]) for name in BRANCHES}
    hydraulic_hashes: dict[str, str] = {}

    for a, b in combinations(BRANCHES, 2):
        if np.allclose(actions[a], actions[b], atol=atol, rtol=0.0):
            action_pairs.append(f"{a}=={b}")

    if depths is not None and floods is not None:
        for name in BRANCHES:
            if name not in depths or name not in floods:
                raise KeyError(f"missing hydraulic branch={name}")
            digest = hashlib.sha256()
            digest.update(np.ascontiguousarray(np.asarray(depths[name], dtype=np.float64)).tobytes())
            digest.update(np.ascontiguousarray(np.asarray(floods[name], dtype=np.float64)).tobytes())
            hydraulic_hashes[name] = digest.hexdigest()
        for a, b in combinations(BRANCHES, 2):
            same_depth = np.allclose(depths[a], depths[b], atol=atol, rtol=0.0)
            same_flood = np.allclose(floods[a], floods[b], atol=atol, rtol=0.0)
            if same_depth and same_flood:
                hydraulic_pairs.append(f"{a}=={b}")

    return {
        "no_control_all_open_verified": no_control_all_open(actions["no_control"], atol=atol),
        "action_equivalent_pairs": action_pairs,
        "hydraulic_equivalent_pairs": hydraulic_pairs,
        "unique_action_branch_count": len(set(action_hashes.values())),
        "unique_hydraulic_branch_count": len(set(hydraulic_hashes.values())) if hydraulic_hashes else None,
        "action_hashes": action_hashes,
        "hydraulic_hashes": hydraulic_hashes,
        "dynamic_internal_equals_hold_action": "dynamic_internal==hold_previous" in action_pairs,
        "dynamic_internal_equals_hold_hydraulics": "dynamic_internal==hold_previous" in hydraulic_pairs,
    }
