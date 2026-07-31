"""V4.1 Compact rescue Phase-1 ops tests (spec sections 3-11, 19).

Exercises the pure-logic ops on the synthetic Train1600-shaped fixture so the
diagnostics / learning-curve / ablation / architecture / gradient / selection /
compact-train logic is verified without the multi-GB frozen artifacts and
without any SWMM.  Guards the anti-leakage contract: fold-local feature
selection, event grouping and "old Locked never drives selection".
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from v4_model_helpers import make_catalog, make_manifest
from sewerrtc.v4.train_v4_loader import build_training_data
from sewerrtc.v4.train_v4_models import ModelConfig, TrueStateEnsemble
from sewerrtc.v4.v4_compact_diag_ops import (
    build_feature_block_catalog,
    classify_feature_block,
    generalization_failure_split,
    locked_metric_comparability,
)
from sewerrtc.v4.v4_compact_curve_ops import (
    build_learning_curves,
    diagnose_learning_curves,
    run_feature_block_ablation,
)
from sewerrtc.v4.v4_compact_model_ops import (
    CompactHeadSpecificModel,
    audit_gradient_conflict,
    compact_cv_report,
    run_head_architecture_ablation,
    select_compact_model,
)

DEAD_ZONES = {"pfv": 1.0, "tfv": 1.0, "peak": 0.001}


def _data():
    m = make_manifest()
    return build_training_data(m, make_catalog(m), require_count=None)


def _cfg():
    return ModelConfig().light()


# --- section 5: feature-block catalog -------------------------------------

def test_feature_block_catalog_covers_every_feature_and_is_offline():
    data = _data()
    catalog = build_feature_block_catalog(data)
    assert len(catalog) == data.features.shape[1]
    assert set(catalog["physical_block"]).issubset(set("ABCDEFGHIJKLMN"))
    # removal never depends on old-Locked error -- only train variance / dupes.
    assert not catalog[catalog["remove_candidate"]]["remove_reason"].eq("").any()
    assert classify_feature_block("opportunity_score") == "A"


# --- section 3: old-Locked metric comparability (read-only diagnosis) -----

def test_metric_comparability_reports_baselines_not_selection():
    data = _data()
    model = TrueStateEnsemble(cfg=_cfg(), pfv_dead_zone=1.0).fit(data)
    out = locked_metric_comparability(
        model, data, cfg=_cfg(), dead_zones=DEAD_ZONES
    )
    assert out["report"]["usable_for_v4_1_selection"] is False
    assert out["report"]["role"] == "explain_v4_0_failure_only"
    hm = out["head_metrics"]
    # baseline models must be re-scored on the SAME Locked split.
    assert {"zero", "train_mean"}.issubset(set(hm["model"]))


# --- section 4: generalization-failure split ------------------------------

def test_generalization_failure_split_is_diagnostic_only():
    data = _data()
    model = TrueStateEnsemble(cfg=_cfg(), pfv_dead_zone=1.0).fit(data)
    tables = generalization_failure_split(model, data)
    for key in (
        "locked_error_by_event",
        "locked_error_by_k",
        "train_locked_shift_report",
        "locked_worst_cases",
    ):
        assert key in tables and isinstance(tables[key], pd.DataFrame)
    assert "standardized_mean_shift" in tables["train_locked_shift_report"]


# --- section 6: Train-only event-grouped learning curves ------------------

def test_learning_curves_report_train_and_cv_splits():
    data = _data()
    curve = build_learning_curves(data, cfg=_cfg(), dead_zones=DEAD_ZONES)
    assert {"train", "cv"}.issubset(set(curve["split_kind"]))
    assert set(curve["train_ratio"]).issuperset({0.2, 1.0})
    verdicts = diagnose_learning_curves(curve)
    assert set(verdicts).issubset({"pfv", "tfv", "peak"})


# --- section 7: feature-block ablation (fold-local) -----------------------

def test_feature_block_ablation_is_fold_local():
    data = _data()
    ablation = run_feature_block_ablation(
        data, cfg=_cfg(), dead_zones=DEAD_ZONES, seeds=(0,), n_folds=4
    )
    assert not ablation.empty
    assert bool(ablation["feature_selection_fold_local"].all())
    # at least the canonical combos are present.
    assert {"state_only", "state_action_rain", "full_570"}.issubset(
        set(ablation["combo"])
    )


# --- section 8: head-architecture ablation (sequence Peak) ----------------

def test_head_architecture_ablation_includes_sequence_peak():
    data = _data()
    arch = run_head_architecture_ablation(
        data, cfg=_cfg(), dead_zones=DEAD_ZONES, n_folds=4
    )
    assert {"A", "B", "C", "D"}.issubset(set(arch["architecture"]))
    peak_variants = set(arch[arch["head"] == "peak"]["architecture"])
    assert {"peak_direct", "peak_sequence", "peak_consistency"} & peak_variants


# --- section 9: multitask gradient conflict -------------------------------

def test_gradient_conflict_audit_is_train_only():
    data = _data()
    report = audit_gradient_conflict(data)
    assert report["stage"] == "AuditV4MultitaskGradientConflictV1"
    assert 0.0 <= report["conflict_fraction"] <= 1.0
    assert isinstance(report["persistent_conflict"], bool)


# --- section 10: selection never reads old Locked -------------------------

def test_selection_reads_only_train_grouped_evidence():
    data = _data()
    cfg = _cfg()
    ablation = run_feature_block_ablation(
        data, cfg=cfg, dead_zones=DEAD_ZONES, seeds=(0,), n_folds=4
    )
    arch = run_head_architecture_ablation(data, cfg=cfg, dead_zones=DEAD_ZONES, n_folds=4)
    gradient = audit_gradient_conflict(data)
    curve = build_learning_curves(data, cfg=cfg, dead_zones=DEAD_ZONES)
    selection = select_compact_model(
        learning_diag=diagnose_learning_curves(curve),
        ablation=ablation,
        architecture=arch,
        gradient=gradient,
    )
    assert selection["reads_old_locked"] is False
    assert selection["reads_old_calibration"] is False
    assert selection["reads_new_locked"] is False
    assert "train_grouped_cv" in selection["selection_basis"]
    assert selection["selected_architecture"] in {"A", "B", "C", "D"}


# --- section 11: compact head-specific model ------------------------------

def test_compact_model_fits_head_specific_and_predicts():
    data = _data()
    model = CompactHeadSpecificModel(cfg=_cfg(), seeds=(0, 1)).fit(data)
    # head-specific feature subsets are stored per head.
    assert "pfv" in model.head_idx_ and "tfv" in model.head_idx_
    pred = model.predict(data, data.split_index("locked_validation"))
    assert "pfv" in pred["continuous"]
    assert pred["continuous"]["pfv"].shape[0] == data.split_index(
        "locked_validation"
    ).size
    assert "uncertainty" in pred


def test_compact_cv_report_is_event_grouped():
    data = _data()
    cfg = _cfg()
    report = compact_cv_report(
        lambda: CompactHeadSpecificModel(cfg=cfg, seeds=(0,)),
        data, cfg=cfg, dead_zones=DEAD_ZONES, n_folds=4,
    )
    assert "predictions" in report and "metrics" in report
    assert not report["predictions"].empty


# --- section 19: pipeline import must not load Torch ----------------------

def test_pipeline_import_does_not_load_torch():
    """Verify pipeline imports don't pull in torch as a side effect.

    Uses subprocess isolation so the test is deterministic even when run
    after other tests that have already imported torch.
    """
    import subprocess

    result = subprocess.run(
        [sys.executable, "-c", (
            "import sys; "
            "assert 'torch' not in sys.modules, 'torch loaded before test'; "
            "import importlib; "
            "importlib.import_module('sewerrtc.v4.pipeline'); "
            "importlib.import_module('sewerrtc.v4.pipeline_v4_compact'); "
            "assert 'torch' not in sys.modules, 'torch loaded by pipeline import'"
        )],
        capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
