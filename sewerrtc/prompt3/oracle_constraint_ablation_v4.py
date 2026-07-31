#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Oracle constraint-ablation projection for V4 bottleneck diagnosis.

Provides ``project_schedule_ablation`` which is a generalisation of the
``project_schedule`` function in ``206_oracle_pareto_v4.py``.  Instead of the
binary *relaxed / constrained* choice it accepts a *constraint mask* that
controls each operational constraint independently:

    rate      – per-actuator step-delta limit
    dwell     – binary minimum-hold (dwell) steps
    topk      – max-K simultaneous actuator changes
    interlock – storage inlet / outlet mutual exclusion

Nine ablation modes (A0–A8) are defined as combinations of these flags.

This module is intentionally standalone so that it can be imported from both
the diagnostic script (207) and the test suite without pulling in the full
Oracle runner.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Import shared helpers from the 206 oracle script via direct path loading.
# We avoid ``import scripts.206_…`` because the filename starts with a digit.
import importlib.util as _ilu

_206_PATH = _PROJECT_ROOT / "scripts" / "206_oracle_pareto_v4.py"
_spec = _ilu.spec_from_file_location("_oracle206", str(_206_PATH))
_oracle206 = _ilu.module_from_spec(_spec)
sys.modules["_oracle206"] = _oracle206  # required for dataclass on py3.9
_spec.loader.exec_module(_oracle206)  # type: ignore[union-attr]

# Re-export helpers we need from 206
_actuator_ids = _oracle206._actuator_ids
binary_pump_ids = _oracle206.binary_pump_ids
nested_get = _oracle206.nested_get
_step_limit_vector = _oracle206._step_limit_vector
_min_hold_vector = _oracle206._min_hold_vector
_storage_groups = _oracle206._storage_groups


# ---------------------------------------------------------------------------
# Constraint mask and ablation mode table
# ---------------------------------------------------------------------------

ABLATION_MODES: dict[str, dict[str, bool]] = {
    "A0_full_constraints":    {"rate": True,  "dwell": True,  "topk": True,  "interlock": True},
    "A1_relax_rate":           {"rate": False, "dwell": True,  "topk": True,  "interlock": True},
    "A2_relax_dwell":          {"rate": True,  "dwell": False, "topk": True,  "interlock": True},
    "A3_relax_K":              {"rate": True,  "dwell": True,  "topk": False, "interlock": True},
    "A4_relax_rate_dwell":     {"rate": False, "dwell": False, "topk": True,  "interlock": True},
    "A5_relax_rate_K":         {"rate": False, "dwell": True,  "topk": False, "interlock": True},
    "A6_relax_dwell_K":        {"rate": True,  "dwell": False, "topk": False, "interlock": True},
    "A7_relax_rate_dwell_K":   {"rate": False, "dwell": False, "topk": False, "interlock": True},
    "A8_operational_relaxed":  {"rate": False, "dwell": False, "topk": False, "interlock": False},
}


# ---------------------------------------------------------------------------
# Core projection
# ---------------------------------------------------------------------------


def project_schedule_ablation(
    matrix: np.ndarray,
    *,
    anchor: np.ndarray,
    actuators: pd.DataFrame,
    cfg: Mapping[str, Any],
    engineering_cfg: Mapping[str, Any],
    constraint_mask: Mapping[str, bool],
    max_k: int | None,
) -> np.ndarray:
    """Project a raw action schedule with fine-grained constraint control.

    Parameters
    ----------
    matrix : (T, N) array of requested actions in [0, 1].
    anchor : (T, N) passive / fallback trajectory used as reference for top-K
        and interlock computations.
    actuators : DataFrame with at least an ``actuator_id`` column.
    cfg, engineering_cfg : project configuration dicts.
    constraint_mask : dict with keys ``rate``, ``dwell``, ``topk``,
        ``interlock``.  ``True`` means the constraint is *enforced*.
    max_k : maximum number of simultaneous actuator changes (used only when
        ``constraint_mask["topk"]`` is True).

    Returns
    -------
    out : (T, N) projected action in [0, 1].
    """
    ids = _actuator_ids(actuators)
    out = np.clip(np.asarray(matrix, dtype=float).copy(), 0.0, 1.0)

    # --- Binary pump quantisation (always enforced — physical semantics) ---
    binary = binary_pump_ids(actuators, cfg, engineering_cfg)
    binary_idx = [ids.index(aid) for aid in binary if aid in ids]
    if binary_idx:
        out[:, binary_idx] = (out[:, binary_idx] >= 0.5).astype(float)

    # --- Rate limit ---
    apply_rate = bool(constraint_mask.get("rate", True))
    apply_dwell = bool(constraint_mask.get("dwell", True))
    apply_topk = bool(constraint_mask.get("topk", True))
    apply_interlock = bool(constraint_mask.get("interlock", True))

    limits = _step_limit_vector(actuators, cfg) if apply_rate else np.full(len(ids), 1.0)
    hold = _min_hold_vector(actuators, cfg) if apply_dwell else np.zeros(len(ids), dtype=int)

    last_change = np.full(len(ids), -10_000, dtype=int)
    previous = np.asarray(anchor[0], dtype=float).copy()

    for t in range(len(out)):
        requested = out[t].copy()

        # Rate: clip per-step delta
        if apply_rate:
            delta = np.clip(requested - previous, -limits, limits)
            current = np.clip(previous + delta, 0.0, 1.0)
        else:
            current = np.clip(requested, 0.0, 1.0)

        # Binary quantisation (always)
        for j in binary_idx:
            current[j] = float(current[j] >= 0.5)

        # Dwell: enforce minimum hold for binary pumps
        if apply_dwell:
            for j in binary_idx:
                if current[j] != previous[j]:
                    if t - last_change[j] < max(1, int(hold[j])):
                        current[j] = previous[j]
                    else:
                        last_change[j] = t

        # Top-K: limit simultaneous changes from anchor
        if apply_topk and max_k is not None and max_k >= 0:
            dev = np.abs(current - anchor[t])
            changed = np.flatnonzero(dev > 1e-9)
            if len(changed) > max_k:
                keep = changed[np.argsort(dev[changed])[-max_k:]]
                reset = np.setdiff1d(changed, keep, assume_unique=False)
                current[reset] = anchor[t, reset]

        out[t] = current
        previous = current

    # --- Interlock: storage inlet / outlet mutual exclusion ---
    if apply_interlock and bool(nested_get(cfg, "controller.storage_retrofit.inlet_outlet_incompatible_action_constraint", True)):
        for _, group in _storage_groups(actuators).items():
            for t in range(len(out)):
                inlet = group["storage_inlet"]
                outlet = group["storage_outlet"]
                if not inlet or not outlet:
                    continue
                inlet_dev = max(abs(out[t, j] - anchor[t, j]) for j in inlet)
                outlet_dev = max(abs(out[t, j] - anchor[t, j]) for j in outlet)
                if inlet_dev > 1e-9 and outlet_dev > 1e-9:
                    reset = outlet if inlet_dev >= outlet_dev else inlet
                    for j in reset:
                        out[t, j] = anchor[t, j]

    return np.clip(out, 0.0, 1.0)


def ablation_mode_for_constraint_mode(constraint_mode: str) -> str:
    """Map the legacy 206 constraint_mode string to an ablation mode name."""
    if constraint_mode == "constrained":
        return "A0_full_constraints"
    if constraint_mode == "relaxed":
        return "A8_operational_relaxed"
    raise ValueError(f"Unknown constraint_mode: {constraint_mode!r}")


def constraint_mode_for_ablation(ablation_mode: str) -> str:
    """Map an ablation mode name back to the closest legacy label."""
    mask = ABLATION_MODES[ablation_mode]
    if all(mask.values()):
        return "constrained"
    if not any(mask.values()):
        return "relaxed"
    return f"ablation_{ablation_mode}"
