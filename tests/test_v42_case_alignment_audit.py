from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import sewerrtc.v4.v42_case_alignment_audit as mod


def _detail(path: Path, *, rain_shift: float = 0.0) -> None:
    elapsed = np.arange(0.0, 181.0, 5.0)
    frame = pd.DataFrame(
        {
            "elapsed_min": elapsed,
            "h:N1": np.linspace(0.1, 0.2, len(elapsed)),
            "storage_volume:S1": np.linspace(10.0, 12.0, len(elapsed)),
            "flow:F1": np.linspace(1.0, 1.2, len(elapsed)),
            "setting:F1": np.ones(len(elapsed)),
            "rainfall_mm_h": np.where(elapsed >= 60.0, 10.0 + rain_shift, 0.0),
        }
    )
    frame.to_csv(path, index=False)


def _inventories(tmp_path: Path, *, rain_shift_di: float = 0.0) -> tuple[Path, Path]:
    roles = ["candidate", "no_control", "dynamic_internal", "hold_previous"]
    rows = []
    ids = []
    for i, role in enumerate(roles):
        path = tmp_path / f"{role}_detail.csv"
        _detail(path, rain_shift=rain_shift_di if role == "dynamic_internal" else 0.0)
        pid = f"p{i}"
        ids.append(pid)
        rows.append(
            {
                "physical_identity_sha256": pid,
                "branch_role": role,
                "detail_path": str(path),
            }
        )
    physical = tmp_path / "physical.csv"
    pd.DataFrame(rows).to_csv(physical, index=False)
    cases = tmp_path / "cases.csv"
    pd.DataFrame(
        [
            {
                "case_uid": "c1",
                "checkpoint_min": 60.0,
                "branch_physical_ids": json.dumps(ids),
            }
        ]
    ).to_csv(cases, index=False)
    return physical, cases


def _patch_graph(monkeypatch) -> None:
    monkeypatch.setattr(mod, "_load_graph_topology", lambda root: {"node_ids": ["N1"]})
    monkeypatch.setattr(mod, "_load_engineering36_ids", lambda root: ["F1"])
    nodes = pd.DataFrame(
        [
            {"node_id": "N1", "invert": 0.0, "max_depth": 1.0, "ponded_area": 0.0, "node_type": "junction"},
            {"node_id": "S1", "invert": 0.0, "max_depth": 1.0, "ponded_area": 1.0, "node_type": "storage"},
        ]
    )
    links = pd.DataFrame([{"link_id": "F1", "from_node": "N1", "to_node": "S1", "link_type": "pump"}])
    monkeypatch.setattr(mod, "_parse_inp_topology", lambda path: (nodes, links))


def test_same_state_and_same_rainfall_pass(monkeypatch, tmp_path: Path):
    _patch_graph(monkeypatch)
    physical, cases = _inventories(tmp_path)
    result = mod.audit_case_alignment(
        project_root=tmp_path,
        physical_inventory=physical,
        case_inventory=cases,
        output_path=tmp_path / "alignment.csv",
    )
    assert bool(result.loc[0, "same_state_numeric_pass"])
    assert bool(result.loc[0, "same_forcing_pass"])


def test_future_rainfall_mismatch_fails_forcing(monkeypatch, tmp_path: Path):
    _patch_graph(monkeypatch)
    physical, cases = _inventories(tmp_path, rain_shift_di=1.0)
    result = mod.audit_case_alignment(
        project_root=tmp_path,
        physical_inventory=physical,
        case_inventory=cases,
        output_path=tmp_path / "alignment.csv",
    )
    assert bool(result.loc[0, "same_state_numeric_pass"])
    assert not bool(result.loc[0, "same_forcing_pass"])
