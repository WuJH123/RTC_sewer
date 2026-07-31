"""Loader contract tests (spec section 4 / defect 2)."""
from __future__ import annotations

import pytest

from v4_model_helpers import make_catalog, make_manifest
from sewerrtc.v4.train_v4_loader import (
    LeakageError,
    PROCESS_RESIDUAL_COLUMNS,
    build_feature_matrix,
    build_training_data,
    compute_acceptance,
    full_event_heads_enabled,
    load_accepted_frame,
)


def _total(manifest):
    return len(manifest)


def test_acceptance_ignores_status_planned():
    m = make_manifest()
    n = _total(m)
    # Every row is status='planned'; acceptance must still keep all of them.
    accepted = load_accepted_frame(m, require_count=n)
    assert len(accepted) == n
    assert (m["status"] == "planned").all()
    # Flipping status to arbitrary values must not change acceptance.
    m2 = m.copy()
    m2.loc[m2.index[:10], "status"] = "pass"
    m2.loc[m2.index[10:20], "status"] = "failed"
    assert len(load_accepted_frame(m2, require_count=n)) == n


def test_exact_required_count_enforced():
    m = make_manifest()
    n = _total(m)
    with pytest.raises(Exception):
        load_accepted_frame(m, require_count=n + 1)


def test_acceptance_is_and_of_gates():
    m = make_manifest()
    m.loc[m.index[0], "kpi_recompute_ok"] = False
    mask = compute_acceptance(m)
    assert mask.sum() == len(m) - 1


def test_no_feature_leakage():
    m = make_manifest()
    cat = make_catalog(m)
    frame = load_accepted_frame(m, require_count=None)
    _, names = build_feature_matrix(frame, cat)
    prohibited = set(PROCESS_RESIDUAL_COLUMNS) | {
        "delta_pfv_h120_vs_no_control",
        "pfv_safe",
        "feasible_rank",
        "regret_to_exact_best",
    }
    assert not (set(names) & prohibited)


def test_leakage_guard_rejects_label_name():
    from sewerrtc.v4.train_v4_loader import assert_no_leakage

    with pytest.raises(LeakageError):
        assert_no_leakage(["ok_feature", "pfv_safe"])


def test_full_event_head_disabled():
    m = make_manifest()
    frame = load_accepted_frame(m, require_count=None)
    assert full_event_heads_enabled(frame) is False
    data = build_training_data(m, make_catalog(m), require_count=None)
    assert data.full_event_enabled is False
    assert not data.full_event_mask.any()


def test_event_split_isolation():
    m = make_manifest()
    data = build_training_data(m, make_catalog(m), require_count=None)
    # No event id appears in more than one split.
    import pandas as pd

    df = pd.DataFrame({"event": data.event_id, "split": data.split})
    spans = df.groupby("event")["split"].nunique()
    assert int((spans > 1).sum()) == 0


def test_peak_hard_negative_preserved():
    m = make_manifest()
    data = build_training_data(m, make_catalog(m), require_count=None)
    tr = data.split_index("train")
    peak = int((data.hard_negative_type[tr] == "Peak_hard_negative").sum())
    assert peak > 0
    # All Peak hard negatives from the source train rows survive the loader.
    src = int(
        ((m["split"] == "train") & (m["hard_negative_type"] == "Peak_hard_negative")).sum()
    )
    assert peak == src


def test_only_future_rainfall_forecast_as_future_input():
    m = make_manifest()
    _, names = build_feature_matrix(load_accepted_frame(m, require_count=None), make_catalog(m))
    future_like = [
        n for n in names if n.startswith("forecast_")
    ]
    assert set(future_like) <= {
        "forecast_rain_depth_120min_mm",
        "forecast_rain_peak_120min_mm_h",
    }
