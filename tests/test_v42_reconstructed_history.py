from __future__ import annotations

import numpy as np
import pytest

from sewerrtc.state.v42_reconstructed_history import (
    CONTRACT_ID,
    CausalReconstructedStateBufferV42,
)


def test_causal_history_requires_13_real_past_reconstructions():
    buffer = CausalReconstructedStateBufferV42(n_nodes=4, gat_model_sha256="gat")
    for i in range(12):
        buffer.append(
            elapsed_min=5.0 * i,
            node_depth_m=np.full(4, i, dtype=float),
            node_uncertainty_m=np.full(4, 0.1, dtype=float),
        )
    assert not buffer.ready
    with pytest.raises(RuntimeError, match="not ready"):
        buffer.snapshot()

    buffer.append(
        elapsed_min=60.0,
        node_depth_m=np.full(4, 12.0),
        node_uncertainty_m=np.full(4, 0.1),
    )
    snap = buffer.snapshot()
    assert snap.ready
    assert snap.node_depth_m.shape == (13, 4)
    assert snap.elapsed_min.tolist() == [5.0 * i for i in range(13)]
    assert snap.contract_id == CONTRACT_ID


def test_causal_history_rejects_gap_and_model_swap():
    buffer = CausalReconstructedStateBufferV42(n_nodes=2, gat_model_sha256="gat-a")
    buffer.append(
        elapsed_min=0.0,
        node_depth_m=[1.0, 2.0],
        node_uncertainty_m=[0.1, 0.2],
    )
    with pytest.raises(RuntimeError, match="contiguous"):
        buffer.append(
            elapsed_min=10.0,
            node_depth_m=[1.1, 2.1],
            node_uncertainty_m=[0.1, 0.2],
        )
    with pytest.raises(RuntimeError, match="model hash changed"):
        buffer.append(
            elapsed_min=5.0,
            node_depth_m=[1.1, 2.1],
            node_uncertainty_m=[0.1, 0.2],
            gat_model_sha256="gat-b",
        )


def test_history_metadata_proves_no_truth_or_current_frame_repetition():
    buffer = CausalReconstructedStateBufferV42(n_nodes=2, gat_model_sha256="gat")
    meta = buffer.metadata()
    assert meta["future_hydraulic_truth_used"] is False
    assert meta["authoritative_swmm_history_used_as_online_input"] is False
    assert meta["current_frame_repetition_used"] is False
