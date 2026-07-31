from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


SECTION_RE = re.compile(r"^\s*\[([^\]]+)\]\s*$")


def _clean(line: str) -> str:
    return line.split(";", 1)[0].strip()


def read_sections(inp_path: str | Path) -> Dict[str, List[str]]:
    sections: Dict[str, List[str]] = {}
    current = None
    with Path(inp_path).open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            m = SECTION_RE.match(raw)
            if m:
                current = m.group(1).strip().upper()
                sections.setdefault(current, [])
                continue
            if current is not None:
                sections[current].append(raw.rstrip("\n"))
    return sections


def section_rows(sections: Dict[str, List[str]], name: str) -> List[List[str]]:
    rows: List[List[str]] = []
    for raw in sections.get(name.upper(), []):
        line = _clean(raw)
        if not line:
            continue
        rows.append(line.split())
    return rows


def options_dict(sections: Dict[str, List[str]]) -> Dict[str, str]:
    out = {}
    for row in section_rows(sections, "OPTIONS"):
        if len(row) >= 2:
            out[row[0].upper()] = " ".join(row[1:])
    return out


def parse_nodes(sections: Dict[str, List[str]]) -> pd.DataFrame:
    records = []
    for r in section_rows(sections, "JUNCTIONS"):
        if len(r) >= 3:
            records.append(
                {
                    "node_id": r[0],
                    "node_type": "junction",
                    "invert": float(r[1]),
                    "max_depth": float(r[2]),
                    "init_depth": float(r[3]) if len(r) > 3 else 0.0,
                    "ponded_area": float(r[5]) if len(r) > 5 else 0.0,
                }
            )
    for r in section_rows(sections, "OUTFALLS"):
        if len(r) >= 3:
            records.append(
                {
                    "node_id": r[0],
                    "node_type": "outfall",
                    "invert": float(r[1]),
                    "max_depth": 0.0,
                    "init_depth": 0.0,
                    "ponded_area": 0.0,
                }
            )
    for r in section_rows(sections, "STORAGE"):
        if len(r) >= 3:
            records.append(
                {
                    "node_id": r[0],
                    "node_type": "storage",
                    "invert": float(r[1]),
                    "max_depth": float(r[2]),
                    "init_depth": float(r[3]) if len(r) > 3 else 0.0,
                    "ponded_area": 0.0,
                    "storage_shape": r[4] if len(r) > 4 else "",
                }
            )
    return pd.DataFrame.from_records(records)


def parse_links(sections: Dict[str, List[str]]) -> pd.DataFrame:
    records = []

    def add_row(r: List[str], link_type: str) -> None:
        if len(r) < 3:
            return
        records.append(
            {
                "link_id": r[0],
                "link_type": link_type,
                "from_node": r[1],
                "to_node": r[2],
                "length": float(r[3]) if link_type == "conduit" and len(r) > 3 else 0.0,
                "roughness": float(r[4]) if link_type == "conduit" and len(r) > 4 else 0.0,
                "raw": " ".join(r),
            }
        )

    for name, typ in [
        ("CONDUITS", "conduit"),
        ("PUMPS", "pump"),
        ("ORIFICES", "orifice"),
        ("WEIRS", "weir"),
        ("OUTLETS", "outlet"),
    ]:
        for r in section_rows(sections, name):
            add_row(r, typ)
    links = pd.DataFrame.from_records(records)
    xsections = []
    for r in section_rows(sections, "XSECTIONS"):
        if len(r) >= 2:
            xsections.append(
                {
                    "link_id": r[0],
                    "shape": r[1],
                    "geom1": float(r[2]) if len(r) > 2 else 0.0,
                    "geom2": float(r[3]) if len(r) > 3 else 0.0,
                }
            )
    if not links.empty and xsections:
        links = links.merge(pd.DataFrame(xsections), on="link_id", how="left")
    return links


def parse_subcatchments(sections: Dict[str, List[str]]) -> pd.DataFrame:
    records = []
    for r in section_rows(sections, "SUBCATCHMENTS"):
        if len(r) >= 8:
            records.append(
                {
                    "subcatchment_id": r[0],
                    "raingage": r[1],
                    "outlet": r[2],
                    "area_ha": float(r[3]),
                    "imperv_pct": float(r[4]),
                    "width": float(r[5]),
                    "slope_pct": float(r[6]),
                }
            )
    return pd.DataFrame.from_records(records)


def parse_controls(sections: Dict[str, List[str]]) -> pd.DataFrame:
    rows = []
    rule = None
    setting_re = re.compile(r"\b(?:LINK|PUMP|ORIFICE|WEIR|OUTLET)\s+(\S+)\s+SETTING\s*=\s*([0-9.+-Ee]+)", re.I)
    for raw in sections.get("CONTROLS", []):
        line = _clean(raw)
        if not line:
            continue
        parts = line.split()
        if parts and parts[0].upper() == "RULE":
            rule = parts[1] if len(parts) > 1 else f"rule_{len(rows)}"
        m = setting_re.search(line)
        if m:
            rows.append({"rule": rule, "link_id": m.group(1), "setting": float(m.group(2)), "line": line})
    return pd.DataFrame.from_records(rows)


def parse_raingages(sections: Dict[str, List[str]]) -> pd.DataFrame:
    rows = []
    for r in section_rows(sections, "RAINGAGES"):
        if len(r) >= 6:
            rows.append(
                {
                    "gage_id": r[0],
                    "format": r[1],
                    "interval": r[2],
                    "snow_catch": r[3],
                    "source": r[4],
                    "series": r[5],
                }
            )
    return pd.DataFrame.from_records(rows)


def parse_timeseries(sections: Dict[str, List[str]]) -> pd.DataFrame:
    rows = []
    for r in section_rows(sections, "TIMESERIES"):
        if len(r) >= 4:
            rows.append({"series": r[0], "date": r[1], "time": r[2], "value": float(r[3])})
        elif len(r) >= 3:
            rows.append({"series": r[0], "date": "", "time": r[1], "value": float(r[2])})
    return pd.DataFrame.from_records(rows)


def audit_inp(inp_path: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, str]]:
    sections = read_sections(inp_path)
    nodes = parse_nodes(sections)
    links = parse_links(sections)
    subcatchments = parse_subcatchments(sections)
    options = options_dict(sections)
    controls = parse_controls(sections)
    if not links.empty:
        controlled = set(controls["link_id"]) if not controls.empty else set()
        links["has_internal_rule"] = links["link_id"].isin(controlled)
        links["is_actuator"] = links["link_type"].isin(["pump", "orifice", "weir", "outlet"])
    return nodes, links, subcatchments, options
