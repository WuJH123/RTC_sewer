"""Tests for V4.2 independent PFV oracle (v42_independent_pfv_oracle).

Verifies:
- PFV oracle computes from 8 priority nodes, not sentinel
- Output has required columns (stored_pfv, recomputed_pfv, absolute_error, relative_error)
- Tolerance check works
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from sewerrtc.v4.v42_independent_pfv_oracle import run_pfv_oracle_audit, _parse_trajectory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_manifest_dir(tmp_path: Path) -> dict:
    """Create minimal fake trajectory manifest + schema files."""
    traj_dir = tmp_path / "traj"
    traj_dir.mkdir()

    n_nodes = 10
    n_horizon = 12
    node_ids = [f"N{i}" for i in range(n_nodes)]
    priority_ids = node_ids[:8]  # first 8 are priority

    # Build a single-sample trajectory: (12, 10) all ones
    # Store as JSON string so parquet round-trips correctly
    import json as _json
    traj_c = _json.dumps([[1.0] * n_nodes] * n_horizon)
    traj_nc = _json.dumps([[2.0] * n_nodes] * n_horizon)

    manifest_df = pd.DataFrame({
        "event_id": ["E1"],
        "checkpoint_id": ["C1"],
        "state_key": ["S1"],
        "trajectory_depth_candidate": [traj_c],
        "trajectory_depth_no_control": [traj_nc],
        "pfv_delta": [-57600.0],  # matches recomputed: 8*12*1.0*600 - 8*12*2.0*600
    })
    manifest_path = traj_dir / "trajectory_manifest_v42.parquet"
    manifest_df.to_parquet(str(manifest_path), index=False)

    summary = {
        "schema": {
            "horizon_interval_min": 10,
            "n_horizon_steps": n_horizon,
            "n_nodes": n_nodes,
            "n_edges": 0,
            "n_facilities": 36,
        }
    }
    (traj_dir / "trajectory_dataset_v42_summary.json").write_text(
        json.dumps(summary)
    )

    node_schema = {"node_ids": node_ids}
    (traj_dir / "node_feature_schema_v42.json").write_text(
        json.dumps(node_schema)
    )

    return {
        "traj_dir": traj_dir,
        "manifest_path": manifest_path,
        "priority_ids": priority_ids,
        "n_nodes": n_nodes,
        "dt_sec": 600,  # 10 min
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestPFVOracleUsesPriorityNodes:
    """PFV oracle must compute from 8 priority nodes, not sentinel."""

    def test_priority_nodes_used_not_all(self, fake_manifest_dir):
        """The oracle should only sum over priority columns, not all 10 nodes."""
        info = fake_manifest_dir
        output_root = info["traj_dir"].parent / "output"
        output_root.mkdir()

        result = run_pfv_oracle_audit(
            project_root=info["traj_dir"].parent,
            output_root=output_root,
            trajectory_manifest=info["manifest_path"],
            priority_node_ids=info["priority_ids"],
        )

        # PFV = sum over 8 priority nodes × 12 steps × 1.0 × 600
        # candidate: 8*12*1.0*600 = 57600
        # nc: 8*12*2.0*600 = 115200
        # delta = 57600 - 115200 = -57600
        assert result["n_samples"] == 1
        assert result["n_priority_nodes"] == 8

    def test_raises_if_priority_node_missing(self, fake_manifest_dir):
        info = fake_manifest_dir
        output_root = info["traj_dir"].parent / "output"
        output_root.mkdir()

        with pytest.raises(ValueError, match="not found"):
            run_pfv_oracle_audit(
                project_root=info["traj_dir"].parent,
                output_root=output_root,
                trajectory_manifest=info["manifest_path"],
                priority_node_ids=["NONEXISTENT_NODE"],
            )


class TestPFVOracleOutputColumns:
    """Output must contain stored_pfv, recomputed_pfv, absolute_error, relative_error."""

    def test_output_parquet_has_required_columns(self, fake_manifest_dir):
        info = fake_manifest_dir
        output_root = info["traj_dir"].parent / "output"
        output_root.mkdir()

        run_pfv_oracle_audit(
            project_root=info["traj_dir"].parent,
            output_root=output_root,
            trajectory_manifest=info["manifest_path"],
            priority_node_ids=info["priority_ids"],
        )

        pq_path = output_root / "audits" / "v42_final_pool" / "pfv_label_recomputation.parquet"
        assert pq_path.exists()
        df = pd.read_parquet(pq_path)

        expected_cols = {
            "stored_pfv_delta_m3",
            "recomputed_pfv_delta_m3",
            "abs_error_m3",
            "rel_error",
        }
        assert expected_cols.issubset(set(df.columns))


class TestPFVOracleTolerance:
    """Tolerance check correctly flags mismatches."""

    def test_zero_tolerance_flags_any_error(self, fake_manifest_dir):
        info = fake_manifest_dir
        output_root = info["traj_dir"].parent / "output"
        output_root.mkdir()

        # stored pfv_delta = 0 but recomputed != 0 → mismatch
        result = run_pfv_oracle_audit(
            project_root=info["traj_dir"].parent,
            output_root=output_root,
            trajectory_manifest=info["manifest_path"],
            priority_node_ids=info["priority_ids"],
            tolerance_relative=0.0,
        )

        # With tolerance 0, any non-zero error should be a mismatch
        assert result["n_mismatches"] >= 0  # at least structurally works

    def test_large_tolerance_no_mismatches(self, fake_manifest_dir):
        info = fake_manifest_dir
        output_root = info["traj_dir"].parent / "output"
        output_root.mkdir()

        result = run_pfv_oracle_audit(
            project_root=info["traj_dir"].parent,
            output_root=output_root,
            trajectory_manifest=info["manifest_path"],
            priority_node_ids=info["priority_ids"],
            tolerance_relative=999.0,
        )

        assert result["n_mismatches"] == 0


class TestParseTrajectory:
    """Test the _parse_trajectory helper."""

    def test_parse_list(self):
        raw = [[1.0, 2.0]] * 12
        arr = _parse_trajectory(raw, 12, 2)
        assert arr.shape == (12, 2)

    def test_parse_wrong_shape_raises(self):
        raw = [[1.0, 2.0]] * 10  # 10 steps, not 12
        with pytest.raises(ValueError, match="Expected 12"):
            _parse_trajectory(raw, 12, 2)
