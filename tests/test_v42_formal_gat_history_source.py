import pandas as pd

from scripts.build_v42_formal_gat_history_source_f2 import _json_safe, _loader, _raw_gates_pass
from scripts.materialize_v42_formal_gat_history_f2 import (
    _detail,
    _read_input_manifest,
)


def test_raw_gate_tuple_is_selected_as_columns():
    columns = {
        name: [True]
        for name in (
            "training_admission_authorized",
            "raw_independent_oracle_all_pass",
            "same_state_raw_verified",
            "same_forcing_raw_verified",
            "actual_readback_verified",
            "h120_window_complete",
            "kpi_recompute_ok",
        )
    }
    assert _raw_gates_pass(pd.DataFrame(columns))
    columns["kpi_recompute_ok"] = [False]
    assert not _raw_gates_pass(pd.DataFrame(columns))


def test_history_audit_json_replaces_nonfinite_values():
    assert _json_safe({"bad": float("nan"), "nested": [float("inf")]}) == {
        "bad": None,
        "nested": [None],
    }


def test_windowed_loader_uses_real_windows_path(tmp_path):
    path = tmp_path / "detail.csv"
    pd.DataFrame(
        {
            "elapsed_min": [0.0, 5.0, 10.0, 15.0],
            "rainfall_mm_h": [1.0, 2.0, 3.0, 4.0],
            "h:N1": [0.1, 0.2, 0.3, 0.4],
            "setting:F1": [0.0, 1.0, 0.0, 1.0],
        }
    ).to_csv(path, index=False)
    loader = _loader(type("Graph", (), {"node_ids": ["N1"], "facility_ids": ["F1"]})(), 1)
    window = loader(path, 5.0, 10.0)
    assert window["elapsed_min"].tolist() == [5.0, 10.0]


def test_formal_gat_detail_reader_can_project_window(tmp_path):
    path = tmp_path / "detail.csv"
    pd.DataFrame(
        {
            "elapsed_min": [0.0, 5.0, 10.0, 15.0],
            "rainfall_mm_h": [1.0, 2.0, 3.0, 4.0],
            "h:N1": [0.1, 0.2, 0.3, 0.4],
            "setting:F1": [0.0, 1.0, 0.0, 1.0],
        }
    ).to_csv(path, index=False)
    projected = _detail(path, ["elapsed_min", "rainfall_mm_h", "h:N1", "setting:F1"], 5.0, 10.0)
    assert projected["elapsed_min"].tolist() == [5.0, 10.0]


def test_raw_manifest_projection_drops_truth_history_columns(tmp_path):
    path = tmp_path / "raw.parquet"
    pd.DataFrame(
        {
            "state_key": ["s1"],
            "history_depth": ["truth"],
            "history_depth_swmm_truth_diagnostic": ["truth"],
            "rainfall_forecast": ["future"],
            "trajectory_depth_candidate": ["candidate"],
        }
    ).to_parquet(path, index=False)
    frame, projected_out = _read_input_manifest(path)
    assert set(projected_out) == {
        "history_depth",
        "history_depth_swmm_truth_diagnostic",
        "rainfall_forecast",
    }
    assert list(frame.columns) == ["state_key", "trajectory_depth_candidate"]
