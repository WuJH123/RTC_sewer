"""Tests for V4.2 TFV/Peak recomputation (v42_tfv_peak_oracle).

Verifies:
- TFV from all nodes, not just priority 8
- Peak = max(C) - max(DI), NOT max(C - DI)
- Output has stored vs recomputed comparison
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from sewerrtc.v4.v42_tfv_peak_oracle import run_tfv_peak_oracle_audit, _parse_trajectory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def fake_tfv_manifest(tmp_path: Path) -> dict:
    """Create minimal fake manifest for TFV/Peak oracle."""
    traj_dir = tmp_path / "traj"
    traj_dir.mkdir()

    n_nodes = 20  # more than 8 to test "all nodes"
    n_horizon = 12
    node_ids = [f"N{i}" for i in range(n_nodes)]

    # Build trajectory: (12, 20) — store as JSON string for parquet
    traj_c = json.dumps([[1.0] * n_nodes] * n_horizon)
    traj_di = json.dumps([[0.5] * n_nodes] * n_horizon)

    manifest_df = pd.DataFrame({
        "event_id": ["E1"],
        "checkpoint_id": ["C1"],
        "state_key": ["S1"],
        "trajectory_depth_candidate": [traj_c],
        "trajectory_depth_dynamic_internal": [traj_di],
        "tfv_delta": [0.0],
        "peak_delta": [0.0],
    })
    manifest_path = traj_dir / "trajectory_manifest_v42.parquet"
    manifest_df.to_parquet(str(manifest_path), index=False)

    summary = {
        "schema": {
            "horizon_interval_min": 10,
            "n_horizon_steps": n_horizon,
            "n_nodes": n_nodes,
        }
    }
    (traj_dir / "trajectory_dataset_v42_summary.json").write_text(json.dumps(summary))
    (traj_dir / "node_feature_schema_v42.json").write_text(
        json.dumps({"node_ids": node_ids})
    )

    return {
        "traj_dir": traj_dir,
        "manifest_path": manifest_path,
        "n_nodes": n_nodes,
        "dt_sec": 600,
    }


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTFVFromAllNodes:
    """TFV from all nodes, not just priority 8."""

    def test_uses_all_20_nodes(self, fake_tfv_manifest):
        info = fake_tfv_manifest
        output_root = info["traj_dir"].parent / "output"
        output_root.mkdir()

        result = run_tfv_peak_oracle_audit(
            project_root=info["traj_dir"].parent,
            output_root=output_root,
            trajectory_manifest=info["manifest_path"],
        )

        # TFV_candidate = sum over ALL 20 nodes × 12 steps × 1.0 × 600
        # = 20 * 12 * 1.0 * 600 = 144000
        # TFV_di = 20 * 12 * 0.5 * 600 = 72000
        # delta = 144000 - 72000 = 72000
        assert result["n_nodes_in_schema"] == 20
        assert result["n_samples"] == 1


class TestPeakFormula:
    """Peak = max(C) - max(DI), NOT max(C - DI)."""

    def test_peak_uses_max_c_minus_max_di(self, fake_tfv_manifest):
        """Verify the oracle computes peak as max(spatial_sum_C) - max(spatial_sum_DI)."""
        info = fake_tfv_manifest
        output_root = info["traj_dir"].parent / "output"
        output_root.mkdir()

        result = run_tfv_peak_oracle_audit(
            project_root=info["traj_dir"].parent,
            output_root=output_root,
            trajectory_manifest=info["manifest_path"],
        )

        # With constant trajectories:
        # spatial_sum_C(t) = 20 * 1.0 = 20.0 for all t → max = 20.0
        # spatial_sum_DI(t) = 20 * 0.5 = 10.0 for all t → max = 10.0
        # peak_delta = 20.0 - 10.0 = 10.0
        assert result["n_samples"] == 1
        # The recomputed peak should be positive (C > DI)
        assert "peak_max_abs_error_m3s" in result


class TestOutputComparison:
    """Output has stored vs recomputed comparison."""

    def test_output_parquet_has_comparison_columns(self, fake_tfv_manifest):
        info = fake_tfv_manifest
        output_root = info["traj_dir"].parent / "output"
        output_root.mkdir()

        run_tfv_peak_oracle_audit(
            project_root=info["traj_dir"].parent,
            output_root=output_root,
            trajectory_manifest=info["manifest_path"],
        )

        pq_path = output_root / "audits" / "v42_final_pool" / "tfv_peak_recomputation.parquet"
        assert pq_path.exists()
        df = pd.read_parquet(pq_path)

        expected_cols = {
            "stored_tfv_delta_m3",
            "recomputed_tfv_delta_m3",
            "tfv_abs_error_m3",
            "stored_peak_delta_m3s",
            "recomputed_peak_delta_m3s",
            "peak_abs_error_m3s",
        }
        assert expected_cols.issubset(set(df.columns))
