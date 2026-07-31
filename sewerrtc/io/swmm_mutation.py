from __future__ import annotations

from datetime import datetime, timedelta
import os
from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd

from .inp_parser import read_sections


def _clock_from_minutes(minutes: float) -> str:
    seconds = int(round(float(minutes) * 60.0))
    if seconds < 0:
        raise ValueError("control rule time must be non-negative")
    hours, remainder = divmod(seconds, 3600)
    minute, second = divmod(remainder, 60)
    return f"{hours:02d}:{minute:02d}:{second:02d}"


def inject_time_gated_control_schedule(
    source_inp: str | Path,
    out_inp: str | Path,
    facility_semantics: pd.DataFrame,
    schedule: np.ndarray,
    checkpoint_min: float,
    decision_interval_sec: int = 600,
    priority: int = 100,
    rule_prefix: str = "GATE5R",
) -> Path:
    """Append inactive-before-checkpoint, high-priority native SWMM rules."""
    source = Path(source_inp)
    output = Path(out_inp)
    if int(decision_interval_sec) <= 0:
        raise ValueError("decision_interval_sec must be positive")
    if "facility_id" not in facility_semantics.columns:
        raise ValueError("facility_semantics is missing facility_id")
    values = np.asarray(schedule, dtype=float)
    if values.ndim != 2 or values.shape[1] != len(facility_semantics):
        raise ValueError("schedule must be steps x facilities in semantics order")
    if not np.isfinite(values).all():
        raise ValueError("schedule contains non-finite settings")
    safe_prefix = re.sub(r"[^A-Za-z0-9_]", "_", str(rule_prefix))
    declared_types = (
        facility_semantics.get(
            "actuator_type", pd.Series(["link"] * len(facility_semantics))
        )
        .astype(str)
        .str.upper()
        .replace({"": "LINK", "NAN": "LINK"})
        .tolist()
    )
    facilities = facility_semantics["facility_id"].astype(str).tolist()
    sections = read_sections(source)
    inferred_types: dict[str, str] = {}
    for section, kind in (
        ("PUMPS", "PUMP"),
        ("ORIFICES", "ORIFICE"),
        ("WEIRS", "WEIR"),
        ("OUTLETS", "OUTLET"),
    ):
        for raw in sections.get(section, []):
            parts = str(raw).split()
            if parts and not parts[0].startswith(";"):
                inferred_types[parts[0]] = kind
    types = [
        inferred_types.get(facility_id, declared_type)
        for facility_id, declared_type in zip(facilities, declared_types)
    ]
    blocks: list[str] = []
    interval_min = float(decision_interval_sec) / 60.0
    for step in range(values.shape[0]):
        start = float(checkpoint_min) + step * interval_min
        stop = start + interval_min
        block = [
            f"RULE {safe_prefix}_STEP_{step:02d}",
            f"IF SIMULATION TIME >= {_clock_from_minutes(start)}",
            f"AND SIMULATION TIME < {_clock_from_minutes(stop)}",
        ]
        for index, (facility_id, actuator_type) in enumerate(
            zip(facilities, types)
        ):
            keyword = "THEN" if index == 0 else "AND"
            kind = actuator_type if actuator_type in {
                "PUMP",
                "ORIFICE",
                "WEIR",
                "OUTLET",
                "LINK",
            } else "LINK"
            block.append(
                f"{keyword} {kind} {facility_id} SETTING = "
                f"{float(np.clip(values[step, index], 0.0, 1.0)):.6f}"
            )
        block.append(f"PRIORITY {int(priority)}")
        blocks.extend(["", *block])

    lines = source.read_text(encoding="utf-8", errors="ignore").splitlines()
    control_index = next(
        (index for index, line in enumerate(lines) if line.strip().upper() == "[CONTROLS]"),
        None,
    )
    if control_index is None:
        raise ValueError("source INP is missing [CONTROLS]")
    insert_index = len(lines)
    for index in range(control_index + 1, len(lines)):
        if lines[index].strip().startswith("["):
            insert_index = index
            break
    updated = lines[:insert_index] + [
        "",
        ";; Project6 Gate5R V3 time-gated Engineering36 override rules.",
        *blocks,
        "",
    ] + lines[insert_index:]
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.name}.tmp")
    temporary.write_text("\n".join(updated) + "\n", encoding="utf-8")
    os.replace(temporary, output)
    return output


def mutate_inp_for_event(
    base_inp: str | Path,
    rainfall_csv: str | Path,
    out_inp: str | Path,
    simulation_duration_min: int,
    series_name: str = "RTC_RAIN_TS",
    gage_name: str = "RG_PROJECT4",
    strip_controls: bool = False,
    disabled_control_targets: Iterable[str] | None = None,
) -> Path:
    base_inp = Path(base_inp)
    out_inp = Path(out_inp)
    out_inp.parent.mkdir(parents=True, exist_ok=True)
    rain = pd.read_csv(rainfall_csv)
    start = datetime(2022, 8, 11, 0, 0, 0)
    end = start + timedelta(minutes=int(simulation_duration_min))
    lines = base_inp.read_text(encoding="utf-8", errors="ignore").splitlines()
    disabled_targets = {str(value) for value in (disabled_control_targets or [])}
    action_re = re.compile(
        r"^(\s*)(THEN|ELSE|AND)(\s+)(LINK|PUMP|ORIFICE|WEIR|OUTLET)"
        r"(\s+)(\S+)(.*)$",
        re.IGNORECASE,
    )
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        upper = line.strip().upper()
        if upper == "[OPTIONS]":
            result.append(line)
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("["):
                key = lines[i].split()[0].upper() if lines[i].split() else ""
                if key == "START_DATE":
                    result.append(f"START_DATE           {start:%m/%d/%Y}")
                elif key == "START_TIME":
                    result.append(f"START_TIME           {start:%H:%M:%S}")
                elif key == "END_DATE":
                    result.append(f"END_DATE             {end:%m/%d/%Y}")
                elif key == "END_TIME":
                    result.append(f"END_TIME             {end:%H:%M:%S}")
                elif key == "REPORT_STEP":
                    result.append("REPORT_STEP          00:05:00")
                else:
                    result.append(lines[i])
                i += 1
            continue
        if upper == "[RAINGAGES]":
            result.append("[RAINGAGES]")
            result.append(";;Name           Format    Interval SCF      Source    Series")
            result.append(f"{gage_name:<16} INTENSITY 0:05     1.0      TIMESERIES {series_name}")
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("["):
                i += 1
            continue
        if upper == "[SUBCATCHMENTS]":
            result.append(line)
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("["):
                raw = lines[i]
                stripped = raw.strip()
                if not stripped or stripped.startswith(";"):
                    result.append(raw)
                else:
                    parts = raw.split()
                    if len(parts) >= 2:
                        parts[1] = gage_name
                        result.append(" ".join(parts))
                    else:
                        result.append(raw)
                i += 1
            continue
        if upper == "[TIMESERIES]":
            result.append(line)
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("["):
                raw = lines[i]
                stripped = raw.strip()
                first = stripped.split()[0] if stripped and not stripped.startswith(";") else ""
                if first != series_name:
                    result.append(raw)
                i += 1
            if result and result[-1].strip():
                result.append("")
            result.append(";; Project5 injected rainfall timeseries; existing boundary/control timeseries above are preserved.")
            result.append(";;Name           Date       Time     Value")
            for _, r in rain.iterrows():
                ts = start + timedelta(minutes=float(r["elapsed_min"]))
                result.append(f"{series_name:<16} {ts:%m/%d/%Y} {ts:%H:%M} {float(r['intensity_mm_h']):.6f}")
            continue
        if strip_controls and upper == "[CONTROLS]":
            result.append("[CONTROLS]")
            result.append(";; Native SWMM controls stripped by Project4 for generic action exploration / Proposed RTC.")
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("["):
                i += 1
            continue
        if disabled_targets and upper == "[CONTROLS]":
            result.append(line)
            i += 1
            control_lines: list[str] = []
            while i < len(lines) and not lines[i].strip().startswith("["):
                control_lines.append(lines[i])
                i += 1
            preamble: list[str] = []
            rule_blocks: list[list[str]] = []
            current: list[str] = []
            for raw in control_lines:
                if raw.strip().upper().startswith("RULE "):
                    if current:
                        rule_blocks.append(current)
                    current = [raw]
                elif current:
                    current.append(raw)
                else:
                    preamble.append(raw)
            if current:
                rule_blocks.append(current)
            result.extend(preamble)
            result.append(
                ";; Engineering36 native actions removed for deterministic "
                "prefix replay and external post-checkpoint control."
            )
            for block in rule_blocks:
                retained: list[str] = []
                removed_action = False
                retained_action_positions: list[int] = []
                for raw in block:
                    match = action_re.match(raw)
                    if match and match.group(6) in disabled_targets:
                        removed_action = True
                        continue
                    retained.append(raw)
                    if match:
                        retained_action_positions.append(len(retained) - 1)
                if removed_action and not retained_action_positions:
                    continue
                if removed_action and retained_action_positions:
                    first_index = retained_action_positions[0]
                    first = retained[first_index]
                    match = action_re.match(first)
                    if match and match.group(2).upper() != "THEN":
                        retained[first_index] = (
                            f"{match.group(1)}THEN{match.group(3)}"
                            f"{match.group(4)}{match.group(5)}"
                            f"{match.group(6)}{match.group(7)}"
                        )
                result.extend(retained)
            continue
        result.append(line)
        i += 1
    temporary = out_inp.with_name(f"{out_inp.name}.tmp")
    temporary.write_text("\n".join(result) + "\n", encoding="utf-8")
    os.replace(temporary, out_inp)
    return out_inp
