from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from sewerrtc.simulation.v42_hydraulic_recorder import (
    formal_target_columns,
    record_v42_hydraulic_targets,
    storage_volume_from_depth_v42,
)
from sewerrtc.v4.v42_hydraulic_target_audit import audit_detail_targets


class _Node:
    def __init__(
        self,
        *,
        depth=1.0,
        head=11.0,
        flooding=0.0,
        volume=0.0,
        total_inflow=0.0,
    ):
        self.depth = depth
        self.head = head
        self.flooding = flooding
        self.volume = volume
        self.total_inflow = total_inflow


class _Link:
    def __init__(self, *, flow=1.0, setting=0.5):
        self.flow = flow
        self.current_setting = setting


def test_recorder_writes_explicit_outfall_total_inflow():
    row = {"elapsed_min": 10.0}
    nodes = {
        "J1": _Node(depth=1.2, head=11.2, flooding=0.3),
        "S1": _Node(depth=2.0, head=12.0, flooding=0.0, volume=123.0),
        "O1": _Node(depth=0.1, head=10.1, flooding=0.0, total_inflow=4.2),
    }
    links = {"P1": _Link(flow=3.5, setting=0.8)}
    record_v42_hydraulic_targets(
        row=row,
        node_objects=nodes,
        facility_link_objects=links,
        graph_node_ids=["J1", "S1", "O1"],
        storage_node_ids=["S1"],
        facility_ids=["P1"],
        outfall_node_ids=["O1"],
    )
    assert row["h:J1"] == 1.2
    assert row["flood:J1"] == 0.3
    assert row["storage_volume:S1"] == 123.0
    assert row["flow:P1"] == 3.5
    assert row["outfall_flow:O1"] == 4.2


def test_recorder_never_zero_fills_missing_authoritative_object():
    row = {}
    record_v42_hydraulic_targets(
        row=row,
        node_objects={},
        facility_link_objects={},
        graph_node_ids=["J1", "O1"],
        storage_node_ids=[],
        facility_ids=["P1"],
        outfall_node_ids=["O1"],
    )
    assert np.isnan(row["h:J1"])
    assert np.isnan(row["flood:J1"])
    assert np.isnan(row["outfall_flow:O1"])
    assert np.isnan(row["flow:P1"])


def _detail(with_outfall: bool) -> pd.DataFrame:
    data = {
        "elapsed_min": [10.0, 20.0],
        "h:J1": [1.0, 1.1],
        "h:S1": [2.0, 2.1],
        "h:O1": [0.1, 0.1],
        "flood:J1": [0.2, 0.1],
        "flood:S1": [0.0, 0.0],
        "flood:O1": [0.0, 0.0],
        "storage_volume:S1": [100.0, 101.0],
        "flow:P1": [2.0, 2.2],
    }
    if with_outfall:
        data["outfall_flow:O1"] = [4.0, 4.1]
    return pd.DataFrame(data)


def test_target_audit_blocks_legacy_detail_without_outfall_flow(tmp_path: Path):
    path = tmp_path / "legacy.csv"
    _detail(False).to_csv(path, index=False)
    audit = audit_detail_targets(
        path,
        node_ids=["J1", "S1", "O1"],
        storage_node_ids=["S1"],
        facility_ids=["P1"],
        outfall_node_ids=["O1"],
    )
    assert audit.node_depth
    assert audit.node_flooding_rate
    assert audit.storage_volume
    assert audit.managed_facility_flow
    assert not audit.outfall_flow
    assert not audit.formal_complete
    assert audit.missing_columns["outfall_flow"] == ["outfall_flow:O1"]


def test_target_audit_passes_only_when_all_physical_targets_exist(tmp_path: Path):
    path = tmp_path / "formal.csv"
    _detail(True).to_csv(path, index=False)
    audit = audit_detail_targets(
        path,
        node_ids=["J1", "S1", "O1"],
        storage_node_ids=["S1"],
        facility_ids=["P1"],
        outfall_node_ids=["O1"],
    )
    assert audit.formal_complete


def test_formal_column_contract_is_explicit():
    columns = formal_target_columns(
        graph_node_ids=["J1", "O1"],
        storage_node_ids=[],
        facility_ids=["P1"],
        outfall_node_ids=["O1"],
    )
    assert columns["node_depth"] == ["h:J1", "h:O1"]
    assert columns["node_flooding_rate"] == ["flood:J1", "flood:O1"]
    assert columns["managed_facility_flow"] == ["flow:P1"]
    assert columns["outfall_flow"] == ["outfall_flow:O1"]


def test_storage_volume_recovery_matches_functional_and_tabular_geometry():
    depth = np.asarray([0.0, 1.0, 2.0], dtype=float)
    functional = storage_volume_from_depth_v42(
        depth, shape="FUNCTIONAL", functional_params=[0.0, 0.0, 4932.7]
    )
    tabular = storage_volume_from_depth_v42(
        depth, shape="TABULAR", curve_depth=[0.0, 2.0], curve_area=[100.0, 100.0]
    )
    np.testing.assert_allclose(functional, [0.0, 4932.7, 9865.4])
    np.testing.assert_allclose(tabular, [0.0, 100.0, 200.0])
