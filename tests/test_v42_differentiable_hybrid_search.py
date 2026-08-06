from __future__ import annotations

import numpy as np

from sewerrtc.control.differentiable_hybrid_search_v42 import (
    DifferentiableSearchConfig,
    _prepare_starts,
)


def test_prepare_starts_keeps_h4_h12_at_current_readback() -> None:
    ids = ["ADD301.2", "x", "ADD301.3"]
    current = np.asarray([1.0, 0.4, 0.0], dtype=np.float32)
    warm = np.repeat(current[None, :], 12, axis=0)
    warm[:3, 1] = 0.9
    warm[3:, 1] = 0.1
    starts = _prepare_starts(
        current_action=current,
        actuator_ids=ids,
        warm_starts=[warm],
        horizon=12,
        config=DifferentiableSearchConfig(max_warm_starts=4),
    )
    assert starts
    for sequence in starts:
        np.testing.assert_allclose(
            sequence[3:], np.broadcast_to(current[None, :], sequence[3:].shape)
        )


def test_prepare_starts_enforces_binary_outer_modes() -> None:
    ids = ["ADD301.2", "continuous", "ADD301.3"]
    current = np.asarray([0.4, 0.5, 0.6], dtype=np.float32)
    starts = _prepare_starts(
        current_action=current,
        actuator_ids=ids,
        warm_starts=[],
        horizon=12,
        config=DifferentiableSearchConfig(include_all_binary_modes_for_current=True),
    )
    assert len(starts) >= 4
    for sequence in starts:
        assert np.isin(sequence[:3, 0], [0.0, 1.0]).all()
        assert np.isin(sequence[:3, 2], [0.0, 1.0]).all()


def test_prepare_starts_deduplicates_sequences() -> None:
    ids = ["x", "y"]
    current = np.asarray([0.2, 0.8], dtype=np.float32)
    warm = np.repeat(current[None, :], 12, axis=0)
    starts = _prepare_starts(
        current_action=current,
        actuator_ids=ids,
        warm_starts=[warm, warm.copy()],
        horizon=12,
        config=DifferentiableSearchConfig(max_warm_starts=8),
    )
    assert len(starts) == 1
