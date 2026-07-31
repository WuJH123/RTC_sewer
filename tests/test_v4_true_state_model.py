"""True-state model / calibration tests (spec sections 5-8)."""
from __future__ import annotations

import numpy as np

from v4_model_helpers import make_catalog, make_manifest
from sewerrtc.v4.train_v4_loader import build_training_data
from sewerrtc.v4.train_v4_models import (
    ModelConfig,
    TrueStateEnsemble,
    calibrate,
    evaluate_split,
    event_equal_weights,
    hard_negative_weights,
)
from sewerrtc.v4.pipeline_train_v4_model import (
    ONLINE_ALLOWED_K,
    ONLINE_DISABLED_K,
)


def _data():
    m = make_manifest()
    return build_training_data(m, make_catalog(m), require_count=None)


def _fit(data):
    cfg = ModelConfig().light()
    return TrueStateEnsemble(cfg=cfg, pfv_dead_zone=1.0).fit(data), cfg


def test_pfv_hurdle_structure():
    data = _data()
    model, _ = _fit(data)
    # Hurdle = a gate model plus an active-only regressor per seed.
    assert model.pfv_gate_ and model.pfv_active_
    assert len(model.pfv_gate_) == len(model.pfv_active_)
    pred = model.predict(data, data.split_index("locked_validation"))
    assert "pfv_active_prob" in pred
    p = pred["pfv_active_prob"]
    assert np.all((p >= 0) & (p <= 1))


def test_calibration_uses_calibration_split_only():
    data = _data()
    model, cfg = _fit(data)
    cal = calibrate(model, data, cfg=cfg)
    assert cal["split_used"] == "calibration"
    assert cal["calibration_n"] == int(data.split_index("calibration").size)


def test_calibration_does_not_read_locked():
    data = _data()
    model, cfg = _fit(data)
    cal_before = calibrate(model, data, cfg=cfg)
    # Corrupt the Locked labels drastically; calibration must be unchanged.
    lk = data.split_index("locked_validation")
    for head in data.continuous:
        data.continuous[head][lk] += 1e6
    for col in data.classification:
        data.classification[col][lk] = 1 - data.classification[col][lk]
    cal_after = calibrate(model, data, cfg=cfg)
    assert cal_before["temperatures"] == cal_after["temperatures"]
    assert cal_before["conformal_abs_q90"] == cal_after["conformal_abs_q90"]
    assert (
        cal_before["abstain_uncertainty_threshold"]
        == cal_after["abstain_uncertainty_threshold"]
    )


def test_event_equal_weights_balance_events():
    ev = np.array(["a", "a", "a", "b"])
    w = event_equal_weights(ev)
    # Event 'a' (3 rows) and event 'b' (1 row) carry equal total weight.
    assert np.isclose(w[:3].sum(), w[3])


def test_hard_negative_upweight():
    hn = np.array(["", "Peak_hard_negative", ""])
    w = hard_negative_weights(hn, weight=2.0)
    assert w[1] == 2.0 and w[0] == 1.0


def test_online_k1_k2_disabled_policy():
    assert ONLINE_DISABLED_K == [1, 2]
    assert set(ONLINE_ALLOWED_K) == {4, 6, 8}
    assert not (set(ONLINE_ALLOWED_K) & set(ONLINE_DISABLED_K))


def test_locked_evaluation_reports_all_heads():
    data = _data()
    model, cfg = _fit(data)
    cal = calibrate(model, data, cfg=cfg)
    rep = evaluate_split(model, data, "locked_validation", calibration=cal)
    assert rep["n"] == int(data.split_index("locked_validation").size)
    assert set(rep["continuous"]) == {"pfv", "tfv", "peak"}
    assert len(rep["residual_mae"]) == 7
    assert "abstain_rate" in rep
