"""Validate whether legacy incoming-link flows can reconstruct outfall flow.

Historical Project6 detail files predate the explicit ``outfall_flow:<node>``
recorder.  This module does not assume that an adjacent link is an outfall-flow
label.  It first derives the complete set of incoming links from the physical
INP, then requires a *new* detail file containing both explicit outfall flow and
all incoming-link flows to validate the deterministic relationship.

Only a passing validation report may be used to promote historical rows from
``PARTIAL_AUX_REUSE`` to ``REUSE_AFTER_EXTRACTION``.  No SWMM execution occurs
in this module.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from sewerrtc.v4.v42_trajectory_builder import _parse_inp_topology


@dataclass(frozen=True)
class OutfallValidationRow:
    outfall_id: str
    incoming_links: tuple[str, ...]
    sample_count: int
    max_abs_error_m3s: float
    rmse_m3s: float
    pass_tolerance: bool


@dataclass(frozen=True)
class OutfallValidationResult:
    detail_path: str
    inp_path: str
    atol_m3s: float
    rtol: float
    status: str
    rows: tuple[OutfallValidationRow, ...]

    def as_dict(self) -> dict:
        return {
            "detail_path": self.detail_path,
            "inp_path": self.inp_path,
            "atol_m3s": self.atol_m3s,
            "rtol": self.rtol,
            "status": self.status,
            "rows": [asdict(row) for row in self.rows],
        }


def incoming_links_by_outfall(inp_path: str | Path) -> dict[str, list[str]]:
    nodes, links = _parse_inp_topology(Path(inp_path))
    outfalls = {
        str(x).casefold(): str(x)
        for x in nodes.loc[nodes["node_type"] == "outfall", "node_id"].tolist()
    }
    result: dict[str, list[str]] = {raw: [] for raw in outfalls.values()}
    for row in links.itertuples(index=False):
        key = str(row.to_node).casefold()
        if key in outfalls:
            result[outfalls[key]].append(str(row.link_id))
    return result


def _column_lookup(columns: Iterable[str]) -> dict[str, str]:
    return {str(c).casefold(): str(c) for c in columns}


def reconstruction_candidates(
    detail_path: str | Path,
    *,
    inp_path: str | Path,
) -> dict[str, list[str]]:
    """Return outfalls whose *complete* incoming-link flow set exists.

    This is only a structural candidate test; it is not scientific validation.
    """
    path = Path(detail_path)
    header = pd.read_csv(path, nrows=0)
    lookup = _column_lookup(header.columns)
    candidates: dict[str, list[str]] = {}
    for outfall, links in incoming_links_by_outfall(inp_path).items():
        if links and all(f"flow:{link}".casefold() in lookup for link in links):
            candidates[outfall] = links
    return candidates


def validate_outfall_reconstruction(
    detail_path: str | Path,
    *,
    inp_path: str | Path,
    atol_m3s: float = 1.0e-5,
    rtol: float = 1.0e-5,
) -> OutfallValidationResult:
    """Compare explicit recorder output against the full incoming-link sum.

    The detail file must contain ``outfall_flow:<outfall_id>``.  A historical
    file without explicit outfall output cannot validate itself.
    """
    detail_path = Path(detail_path)
    inp_path = Path(inp_path)
    header = pd.read_csv(detail_path, nrows=0)
    lookup = _column_lookup(header.columns)
    rows: list[OutfallValidationRow] = []
    for outfall, links in incoming_links_by_outfall(inp_path).items():
        explicit_key = f"outfall_flow:{outfall}".casefold()
        if explicit_key not in lookup:
            raise KeyError(f"missing explicit outfall recorder column for {outfall}")
        if not links:
            raise ValueError(f"outfall {outfall} has no incoming links in INP")
        link_keys = [f"flow:{link}".casefold() for link in links]
        missing = [key for key in link_keys if key not in lookup]
        if missing:
            raise KeyError(f"missing incoming-link flow columns for {outfall}: {missing}")
        usecols = [lookup[explicit_key]] + [lookup[key] for key in link_keys]
        df = pd.read_csv(detail_path, usecols=usecols)
        numeric = df.apply(pd.to_numeric, errors="coerce")
        if numeric.isna().any().any():
            raise ValueError(f"non-finite outfall validation data for {outfall}")
        explicit = numeric[lookup[explicit_key]].to_numpy(dtype=float)
        link_sum = numeric[[lookup[key] for key in link_keys]].sum(axis=1).to_numpy(dtype=float)
        error = link_sum - explicit
        max_abs = float(np.max(np.abs(error))) if len(error) else 0.0
        rmse = float(np.sqrt(np.mean(error * error))) if len(error) else 0.0
        scale = float(np.max(np.abs(explicit))) if len(explicit) else 0.0
        passed = bool(max_abs <= atol_m3s + rtol * scale)
        rows.append(
            OutfallValidationRow(
                outfall_id=outfall,
                incoming_links=tuple(links),
                sample_count=int(len(error)),
                max_abs_error_m3s=max_abs,
                rmse_m3s=rmse,
                pass_tolerance=passed,
            )
        )
    status = "pass" if rows and all(row.pass_tolerance for row in rows) else "fail"
    return OutfallValidationResult(
        detail_path=str(detail_path),
        inp_path=str(inp_path),
        atol_m3s=float(atol_m3s),
        rtol=float(rtol),
        status=status,
        rows=tuple(rows),
    )


def reconstruct_outfall_flow(
    detail_path: str | Path,
    *,
    inp_path: str | Path,
    validated_result: OutfallValidationResult,
) -> pd.DataFrame:
    """Derive historical outfall trajectories only after external validation."""
    if validated_result.status != "pass":
        raise RuntimeError("outfall reconstruction is not validated")
    detail_path = Path(detail_path)
    header = pd.read_csv(detail_path, nrows=0)
    lookup = _column_lookup(header.columns)
    output: dict[str, np.ndarray] = {}
    for outfall, links in incoming_links_by_outfall(inp_path).items():
        keys = [f"flow:{link}".casefold() for link in links]
        missing = [key for key in keys if key not in lookup]
        if missing:
            raise KeyError(f"historical detail lacks incoming links for {outfall}: {missing}")
        df = pd.read_csv(detail_path, usecols=[lookup[key] for key in keys])
        numeric = df.apply(pd.to_numeric, errors="coerce")
        if numeric.isna().any().any():
            raise ValueError(f"non-finite incoming-link flow for {outfall}")
        output[f"outfall_flow:{outfall}"] = numeric.sum(axis=1).to_numpy(dtype=float)
    return pd.DataFrame(output)


def write_validation(path: str | Path, result: OutfallValidationResult) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result.as_dict(), indent=2, allow_nan=False), encoding="utf-8")
