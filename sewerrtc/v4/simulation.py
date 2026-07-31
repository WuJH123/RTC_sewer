"""V4 simulation facade.

No SWMM implementation is copied here.  All authoritative execution is routed
to :mod:`sewerrtc.simulation.pyswmm_runner`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sewerrtc.io.swmm_mutation import mutate_inp_for_event
from sewerrtc.simulation import pyswmm_runner


BRANCHES = (
    "candidate",
    "no_control",
    "dynamic_internal_rules",
    "hold_previous",
)


def authoritative_runner_module():
    return pyswmm_runner


def run_prepared_case(row: dict, paths: dict[str, str]) -> dict:
    """Run a fully prepared, serialized runner request.

    The planner must provide ``runner_function`` plus keyword arguments whose
    paths and arrays have already been materialized. Only existing authoritative
    runner functions are accepted; this facade never mutates DWF.
    """
    function_name = str(row.get("runner_function", ""))
    allowed = {
        "run_swmm_fixed_action": pyswmm_runner.run_swmm_fixed_action,
        "run_swmm_dynamic_internal": pyswmm_runner.run_swmm_dynamic_internal,
    }
    if function_name not in allowed:
        raise ValueError(f"unsupported authoritative runner: {function_name}")
    raw_kwargs = row.get("runner_kwargs", {})
    if isinstance(raw_kwargs, str):
        raw_kwargs = json.loads(raw_kwargs)
    kwargs = dict(raw_kwargs)
    actuators_csv = kwargs.pop("actuators_csv", None)
    if actuators_csv is not None:
        actuators = pd.read_csv(actuators_csv)
        if "actuator_id" not in actuators:
            actuators["actuator_id"] = actuators["facility_id"]
        if "link_type" not in actuators and "actuator_type" in actuators:
            actuators["link_type"] = actuators["actuator_type"]
        kwargs["actuators"] = actuators
    priority_file = kwargs.pop("priority_nodes_file", None)
    if priority_file is not None:
        kwargs["priority_nodes"] = [
            line.strip()
            for line in Path(priority_file)
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip() and not line.lstrip().startswith(("#", ";"))
        ]
    rainfall_path = kwargs.pop("rainfall_path", None)
    if rainfall_path is not None:
        base_inp = Path(kwargs["inp_path"])
        simulation_duration = int(
            kwargs.get(
                "simulation_duration_min", kwargs.get("duration_min", 0)
            )
        )
        kwargs["inp_path"] = mutate_inp_for_event(
            base_inp,
            rainfall_path,
            paths["inp"],
            simulation_duration,
            strip_controls=False,
        )
    if isinstance(kwargs.get("post_action"), list):
        kwargs["post_action"] = np.asarray(
            kwargs["post_action"], dtype=float
        )
    if isinstance(kwargs.get("prefix_schedule"), dict):
        kwargs["prefix_schedule"] = {
            float(key): np.asarray(value, dtype=float)
            for key, value in kwargs["prefix_schedule"].items()
        }
    detail_path = Path(paths["directory"]) / "detail.csv"
    kwargs["out_detail_csv"] = str(detail_path)
    # Fail-closed hot-start contract: None or empty markers are metadata that
    # planners set to declare a no-hotstart run and are simply removed; any
    # non-empty path is a real hot-start request and must never be silently
    # ignored. ``kwargs`` is already a copy, so caller kwargs stay untouched.
    hotstart_dir = kwargs.pop("hotstart_dir", None)
    if hotstart_dir:
        raise ValueError(
            f"Final V4 prohibits hot-start, got hotstart_dir={hotstart_dir!r}"
        )
    result = allowed[function_name](**kwargs)
    return {
        "runner_function": function_name,
        "result": result,
        "detail_path": str(detail_path),
    }
