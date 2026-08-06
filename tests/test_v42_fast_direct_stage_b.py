import numpy as np

from scripts.run_v42_fast_direct_stage_b import build_stage_b_sequences


def test_stage_b_is_bounded_and_preserves_tail():
    base = np.full((12, 6), 0.5, dtype=np.float32)
    rows = build_stage_b_sequences([base], [0, 1, 2, 3])
    assert 0 < len(rows) <= 32
    assert all(row.shape == (12, 6) for row in rows)
    assert all(np.array_equal(row[3:], base[3:]) for row in rows)
    assert all(np.isfinite(row).all() and (row >= 0).all() and (row <= 1).all() for row in rows)
