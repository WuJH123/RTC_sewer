"""Safety wrapper around the V4.2 Formal authoritative runtime.

For Proposed and every non-Internal baseline, Engineering36 must be controlled by
the evaluated policy, not simultaneously by the INP's native [CONTROLS] rules.
This module creates a short-lived rule-free runtime INP while preserving the
physical network/rainfall definition.  ``Internal`` alone runs the original INP
with its native rules.

The wrapper exists so Formal No-control is physically *all-open* and Formal
All-close is physically *all-zero*, rather than merely writing a target that can
be overwritten by a native rule during the same SWMM run.
"""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path
from typing import Any

from sewerrtc.simulation.pyswmm_runner import physical_network_sha256
from sewerrtc.v4.v42_formal_runtime import (
    FormalEventInput,
    run_baseline_event as _run_baseline_event,
    run_proposed_event as _run_proposed_event,
)


def build_rule_free_runtime_inp(source: str | Path, target: str | Path) -> Path:
    source = Path(source)
    target = Path(target)
    lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    output: list[str] = []
    in_controls = False
    controls_seen = False
    for raw in lines:
        stripped = raw.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            name = stripped[1:-1].strip().upper()
            if name == "CONTROLS":
                in_controls = True
                controls_seen = True
                output.append(raw)
                output.append("; Formal runtime: native control rules disabled; evaluated policy owns Engineering36.")
                continue
            if in_controls:
                in_controls = False
            output.append(raw)
            continue
        if in_controls:
            # Preserve comments only; executable RULE/IF/THEN/ELSE/AND/OR lines
            # are intentionally removed from this runtime clone.
            if stripped.startswith(";"):
                output.append(raw)
            continue
        output.append(raw)
    if not controls_seen:
        output.extend(
            [
                "",
                "[CONTROLS]",
                "; Formal runtime: no native control rules in source INP.",
            ]
        )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(output) + "\n", encoding="utf-8")
    if physical_network_sha256(source) != physical_network_sha256(target):
        raise RuntimeError(
            "removing [CONTROLS] changed the physical-network SHA; refusing Formal runtime clone"
        )
    return target.resolve()


def _runtime_event(event: FormalEventInput, output_dir: Path) -> FormalEventInput:
    digest = hashlib.sha256(
        f"{event.event_id}|{event.rainfall_sha256}|{event.input_sha256}".encode("utf-8")
    ).hexdigest()[:16]
    runtime_inp = build_rule_free_runtime_inp(
        event.inp_path,
        output_dir / "runtime_inp" / f"{digest}__no_native_controls.inp",
    )
    return replace(event, inp_path=runtime_inp)


def run_baseline_event(
    event: FormalEventInput,
    *,
    strategy: str,
    project_root: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    out = Path(output_dir)
    if strategy == "Internal":
        # Internal is the sole baseline whose scientific definition is the
        # frozen native SWMM dynamic-rule behavior.
        return _run_baseline_event(
            event,
            strategy=strategy,
            project_root=project_root,
            output_dir=out,
        )
    runtime_event = _runtime_event(event, out)
    result = _run_baseline_event(
        runtime_event,
        strategy=strategy,
        project_root=project_root,
        output_dir=out,
    )
    result["source_input_sha256"] = event.input_sha256
    result["runtime_rule_free_inp_sha256"] = runtime_event.input_sha256
    result["native_controls_disabled"] = True
    return result


def run_proposed_event(
    event: FormalEventInput,
    *,
    project_root: str | Path,
    output_dir: str | Path,
    state_source: str = "gat_sparse_reconstruction",
    device: str = "auto",
    max_candidate_sequences: int = 64,
) -> dict[str, Any]:
    out = Path(output_dir)
    runtime_event = _runtime_event(event, out)
    # The inner Proposed runtime opens a second shadow simulation using its
    # supplied event INP.  We need that shadow to retain native rules, so a
    # dedicated Proposed implementation must receive the original path for the
    # shadow.  Until the inner API exposes separate plant/shadow paths, fail
    # closed rather than run a rule-free object as the Dynamic-Internal shadow.
    #
    # This guard is intentionally explicit: a future patch must add
    # ``internal_shadow_inp_path`` to the canonical runtime before Formal
    # Proposed execution is authorized.
    raise RuntimeError(
        "Formal Proposed runtime requires separate rule-free plant and native-rule Internal shadow INPs. "
        "The production orchestrator must use run_proposed_event_dual_inp; refusing ambiguous execution."
    )
