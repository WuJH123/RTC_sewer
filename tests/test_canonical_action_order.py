from __future__ import annotations

import numpy as np

from sewerrtc.control.canonical_action_order import CanonicalActionOrder


def _mapping() -> CanonicalActionOrder:
    global_ids = [f"A{i:03d}" for i in range(109)]
    old_mask = [global_ids[i] for i in [90, 1, 53, 10, 108, 3, 42, 7, 88, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 54, 55, 56, 57]]
    return CanonicalActionOrder.from_global_registry(global_ids, old_mask)


def test_global_projection_and_expansion_round_trip_preserve_unique_values():
    order = _mapping()
    global_values = np.arange(109, dtype=np.float32)[None, None, :]
    canonical = order.project_global109(global_values)
    expanded = order.expand_to_global109(canonical, fill_value=-1.0)
    assert len(order.canonical_ids) == 36
    assert len(set(order.canonical_ids)) == 36
    assert np.array_equal(expanded[..., list(order.canonical_global_indices)], canonical)
    assert np.max(np.abs(order.project_global109(expanded) - canonical)) == 0.0


def test_old_mask_and_canonical_round_trip_is_exact():
    order = _mapping()
    old_values = np.arange(36, dtype=np.float32)[None, :]
    canonical = order.old_mask_to_canonical(old_values)
    recovered = order.canonical_to_old_mask(canonical)
    assert np.max(np.abs(recovered - old_values)) == 0.0


def test_candidate_reference_alignment_and_semantics_preservation():
    order = _mapping()
    candidate = np.arange(36 * 6, dtype=np.float32).reshape(1, 6, 36)
    reference = candidate + 1000.0
    assert order.canonical_to_old_mask(order.old_mask_to_canonical(candidate)).shape == candidate.shape
    assert order.canonical_to_old_mask(order.old_mask_to_canonical(reference)).shape == reference.shape
    # The mapping preserves every named column independently for candidate and reference.
    assert np.max(np.abs(order.old_mask_to_canonical(candidate) - order.old_mask_to_canonical(reference) + 1000.0)) == 0.0


def test_binary_and_continuous_values_are_not_coerced_by_permutation():
    order = _mapping()
    values = {aid: i / 35.0 for i, aid in enumerate(order.canonical_ids)}
    values[order.canonical_ids[0]] = 0.0
    values[order.canonical_ids[-1]] = 1.0
    aligned = order.align_action_dict(values)
    assert aligned[0] == 0.0
    assert aligned[-1] == 1.0
