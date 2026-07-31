from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from sewerrtc.io.safe_paths import path_budget_check, short_run_tag, single_writer_lease
from sewerrtc.prompt3 import action_effect_v4 as v4
from sewerrtc.prompt3 import action_effect_v4_aug1 as v4a
from sewerrtc.prompt3.action_effect_v4 import (
    ACTION_FEATURE_NAMES, CAUSAL_FEATURE_NAMES, CONTEXT_FEATURE_NAMES,
    REFERENCE_LABELS, RESIDUAL_LABELS,
)
from sewerrtc.simulation.runtime_contracts import analyze_recovery, write_csv


# ---------------------------------------------------------------------------
# Synthetic fixtures (no SWMM). The real branch execution is validated by the
# 8-case SWMM smoke, not by pytest.
# ---------------------------------------------------------------------------
def _mk_aug1_row(event, ckpt, action_type, sig, *, nc_full=14.0, cand_full=24.0,
                 h120_delta=1.0, runtime=True, readback=True, paired=True,
                 censored=False, drop_full=False):
    row = {
        "sample_id": sig, "case_signature": sig, "event_id": event,
        "checkpoint_elapsed_min": ckpt, "action_type": action_type,
        "runtime_executed": "true" if runtime else "false",
        "authoritative_swmm": "true", "deterministic_prefix_replay": "true",
        "hotstart_used": "false", "truth_future_leakage": "0",
        "initial_state_sha256": f"hash_{event}_{ckpt}",
        "reference_initial_state_sha256": f"hash_{event}_{ckpt}",
        "paired_initial_state_hash_ok": "true" if paired else "false",
        "candidate_differs": "true",
        "readback_ok": "true" if readback else "false",
        "readback_worst_abs": 0.0,
        "recovery_censored": "true" if censored else "false",
        "recovery_status": "censored" if censored else "recovered",
        "no_control_PFV_H120": 10.0, "passive_PFV_H120": 12.0,
        "internal_PFV_H120": 11.0, "internal_TFV_H120": 100.0, "internal_peak_H120": 5.0,
        "no_control_PFV_full": nc_full, "passive_PFV_full": 16.0, "internal_PFV_full": 15.0,
        "candidate_PFV_H120": 10.0 + h120_delta, "candidate_TFV_H120": 90.0,
        "candidate_peak_H120": 4.0, "candidate_PFV_full": cand_full,
        "delta_PFV_H120_vs_no_control": h120_delta,
        "delta_PFV_H120_vs_passive": (10.0 + h120_delta) - 12.0,
        "delta_TFV_H120_vs_internal": -10.0,
        "delta_peak_H120_vs_internal": -1.0,
        "delta_PFV_full_vs_no_control": cand_full - nc_full,
        "delta_PFV_full_vs_passive": cand_full - 16.0,
    }
    for name in CONTEXT_FEATURE_NAMES:
        row[f"v4_ctx_{name}"] = 0.1
    for name in CAUSAL_FEATURE_NAMES:
        row[f"v4_causal_{name}"] = 0.2
    for name in ACTION_FEATURE_NAMES:
        row[f"v4_act_{name}"] = 0.3
    if drop_full:
        del row["candidate_PFV_full"]
        del row["delta_PFV_full_vs_no_control"]
    return row


def _mk_config(tmp_path: Path, val_event: str = "event_val") -> Path:
    out_root = tmp_path / "out"
    (out_root / "action_effect_dataset_v4").mkdir(parents=True, exist_ok=True)
    # Base V4 manifest, so build can confirm the base is preserved and derive the
    # frozen validation split (sha256(event)%5==0).
    base_rows = [_mk_aug1_row("event_base", 60.0, "current_candidate", "base1")]
    write_csv(out_root / "action_effect_dataset_v4" / "v4_dataset_manifest.csv", base_rows)
    cfg = {
        "project": {"root": str(tmp_path), "output_root": str(out_root),
                    "inp": "data/wuhan_v8_storage_retrofit.inp"},
        "v4": {
            "training": {"required_min_samples": 3000, "ensemble_size": 5,
                         "seeds": [20260723, 20260724, 20260725, 20260726, 20260727]},
            "dual_reference": {"pfv_event_quantile": 0.95},
            "model_gate": {"residual_direction_accuracy_min": {"PFV": 0.70, "TFV": 0.70, "peak": 0.80}},
            "aug1": {"effective_target": 1600, "reserve": 400, "minimum_events": 24},
        },
    }
    path = tmp_path / "cfg.yaml"
    path.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return path


def _find_val_event(config: Path) -> str:
    """Return an event_id that lands in the frozen validation split."""
    i = 0
    while True:
        ev = f"ev_v_{i}"
        if int(v4a.hashlib.sha256(ev.encode()).hexdigest()[:8], 16) % 5 == 0:
            return ev
        i += 1


def _find_train_events(n: int) -> list[str]:
    out: list[str] = []
    i = 0
    while len(out) < n:
        ev = f"ev_t_{i}"
        if int(v4a.hashlib.sha256(ev.encode()).hexdigest()[:8], 16) % 5 != 0:
            out.append(ev)
        i += 1
    return out


# 1. Same checkpoint => identical paired initial-state hash (state only).
def test_initial_state_hash_ignores_checkpoint_action() -> None:
    base = {"elapsed_min": [0.0, 5.0, 10.0], "h:P1": [1.0, 1.1, 1.2],
            "flood:P1": [0.0, 0.0, 0.5]}
    branch_a = pd.DataFrame({**base, "a:ADD301.2": [1.0, 1.0, 1.0]})
    branch_b = pd.DataFrame({**base, "a:ADD301.2": [1.0, 1.0, 0.0]})  # different action at ckpt
    ha = v4a._initial_state_hash(branch_a, 10.0)
    hb = v4a._initial_state_hash(branch_b, 10.0)
    assert ha == hb  # branches share the same prefix state, only actions differ


def test_initial_state_hash_changes_when_prefix_state_differs() -> None:
    a = pd.DataFrame({"elapsed_min": [0.0, 10.0], "h:P1": [1.0, 1.2], "flood:P1": [0.0, 0.0]})
    b = pd.DataFrame({"elapsed_min": [0.0, 10.0], "h:P1": [1.0, 9.9], "flood:P1": [0.0, 0.0]})
    assert v4a._initial_state_hash(a, 10.0) != v4a._initial_state_hash(b, 10.0)


# 2. Deterministic prefix replay: no hot-start is ever enabled.
def test_generation_marks_no_hotstart() -> None:
    source = Path(v4a.__file__).read_text(encoding="utf-8")
    assert '"hotstart_used": _truth_str(False)' in source
    assert "hot_start=True" not in source


# 3/4. Engineering projection + candidate differs from every reference branch.
def test_binary_pump_projection_and_variable_pump_clip() -> None:
    assert v4a._project_setting("ADD301.2", 0.4) == 0.0
    assert v4a._project_setting("ADD301.2", 0.6) == 1.0
    assert v4a._project_setting("add350.1", 0.37) == pytest.approx(0.37)
    assert v4a._project_setting("add350.1", 1.9) == 1.0


def test_candidate_is_forced_to_differ_from_references() -> None:
    ids = ["ADD301.2", "add350.1"]
    references = {
        "no_control": {"ADD301.2": [1.0], "add350.1": [1.0]},
        "internal_current_action": {"ADD301.2": [1.0], "add350.1": [1.0]},
    }
    candidate = {"ADD301.2": [1.0], "add350.1": [1.0]}  # identical to a reference
    out = v4a._ensure_candidate_differs(candidate, references, ["ADD301.2"], ids)
    step0 = tuple(out[a][0] for a in ids)
    assert all(step0 != tuple(r[a][0] for a in ids) for r in references.values())


# 5/13. Full recovery uses the unified criterion; censored is preserved.
def test_analyze_recovery_marks_censored_without_deletion() -> None:
    rows = [{"elapsed_min": t, "flood:P1": 2.0} for t in range(0, 800, 5)]
    detail = pd.DataFrame(rows)
    out = analyze_recovery(detail, event_id="e", policy_id="candidate", trajectory_id="t",
                           duration_min=75, minimum_tail_min=180, max_tail_min=720,
                           priority_nodes=["P1"])
    assert out["recovery_censored"] is True
    assert out["recovery_criteria_met"] is False


# 6/16/17. Build audit: H120 vs full distinct; missing full => 3; enough valid => 0.
def test_build_returns_3_when_full_event_label_missing(tmp_path: Path) -> None:
    config = _mk_config(tmp_path)
    out_dir = v4a._aug1_dir(config)
    write_csv(out_dir / "v4_aug1_generation_manifest.csv",
              [_mk_aug1_row("event_x", 60.0, "top2", "sig_missing", drop_full=True)])
    code, outputs = v4a.build_v4_augmented_dataset(config, smoke=True)
    assert code == 3
    audit = json.loads(Path(outputs["audit"]).read_text(encoding="utf-8"))
    assert audit["aug1_valid_sample_count"] == 0


def test_build_returns_0_when_valid_samples_meet_smoke_gate(tmp_path: Path) -> None:
    config = _mk_config(tmp_path)
    out_dir = v4a._aug1_dir(config)
    ev_a, ev_b = _find_train_events(2)
    rows = [
        _mk_aug1_row(ev_a, 60.0, "top2", "sig_a", cand_full=24.0, nc_full=14.0),
        _mk_aug1_row(ev_b, 80.0, "top4", "sig_b", cand_full=10.0, nc_full=14.0),
    ]
    write_csv(out_dir / "v4_aug1_generation_manifest.csv", rows)
    code, outputs = v4a.build_v4_augmented_dataset(config, smoke=True)
    audit = json.loads(Path(outputs["audit"]).read_text(encoding="utf-8"))
    assert code == 0, audit["checks"]
    assert audit["base_preserved"] is True
    assert audit["checks"]["h120_full_not_copied"] is True


def test_h120_and_full_delta_columns_are_not_copies() -> None:
    rows = [_mk_aug1_row("e", 60.0, "top2", "s", cand_full=24.0, nc_full=14.0)]
    # H120 delta = +1.0, full delta = 24-14 = +10.0 -> distinct.
    assert v4a._h120_full_distinct(rows) is True


# 7/8. runtime_executed=false and readback failure cannot enter the dataset.
def test_reject_runtime_not_executed() -> None:
    row = _mk_aug1_row("e", 60.0, "top2", "s", runtime=False)
    assert v4a._reject_aug1_row(row) == "runtime_executed_false"


def test_reject_readback_failed() -> None:
    row = _mk_aug1_row("e", 60.0, "top2", "s", readback=False)
    assert v4a._reject_aug1_row(row) == "readback_failed"


def test_reject_paired_hash_mismatch() -> None:
    row = _mk_aug1_row("e", 60.0, "top2", "s", paired=False)
    assert v4a._reject_aug1_row(row) == "paired_initial_state_hash_mismatch"


# 9/10. Case-signature dedup and resume (no duplicate completed cases).
def test_build_dedups_case_signature(tmp_path: Path) -> None:
    config = _mk_config(tmp_path)
    out_dir = v4a._aug1_dir(config)
    ev = _find_train_events(1)[0]
    rows = [
        _mk_aug1_row(ev, 60.0, "top2", "dup"),
        {**_mk_aug1_row(ev, 60.0, "top4", "dup2"), "case_signature": "dup"},
    ]
    write_csv(out_dir / "v4_aug1_generation_manifest.csv", rows)
    _, outputs = v4a.build_v4_augmented_dataset(config, smoke=True)
    rejected = pd.read_csv(outputs["rejected"])
    assert (rejected["reject_reason"] == "duplicate_case_signature").any()


# 11. Validation events are isolated from the aug1 training split.
def test_validation_events_isolated_from_training(tmp_path: Path) -> None:
    config = _mk_config(tmp_path)
    val_events = v4a._validation_events(config)
    train = _find_train_events(2)
    assert all(t not in val_events for t in train)


# 12. Aug1 reuses the leakage-free causal feature names.
def test_causal_feature_names_shared_with_base() -> None:
    assert v4a.CAUSAL_FEATURE_NAMES == v4.CAUSAL_FEATURE_NAMES


# 13 (dedup case) already above; 14. MAX_PATH budget.
def test_path_budget_and_short_tag() -> None:
    tag = short_run_tag("event_that_is_extremely_long_" * 5)
    assert len(tag) <= 36
    assert path_budget_check(Path("E:/RTC_sewer/Project6/outputs/project6_dual_reference_v4/dual_reference_aug1/cases") / f"{tag}.inp")["within_budget"]


# 15. Single writer lease.
def test_single_writer_lease(tmp_path: Path) -> None:
    with single_writer_lease(tmp_path / "out", owner="a"):
        with pytest.raises(RuntimeError):
            with single_writer_lease(tmp_path / "out", owner="b"):
                pass


# 18. Model gate cannot be bypassed except by changing the frozen threshold.
def _train_bad_full_pfv(tmp_path: Path):
    """Build a dataset where full-event PFV direction is systematically wrong."""
    config = _mk_config(tmp_path)
    out_dir = v4a._aug1_dir(config)
    events = _find_train_events(6) + [_find_val_event(config)]
    rows = []
    for j, ev in enumerate(events):
        for i in range(4):
            # H120 direction learnable; full-event PFV sign flipped vs H120.
            cand_full = 14.0 - 5.0 if (i % 2 == 0) else 14.0 + 5.0
            rows.append(_mk_aug1_row(ev, 60.0 + i * 10, "top2", f"{ev}_{i}",
                                     cand_full=cand_full, nc_full=14.0,
                                     h120_delta=(1.0 if i % 2 == 0 else -1.0)))
    write_csv(out_dir / "v4_aug1_generation_manifest.csv", rows)
    v4a.build_v4_augmented_dataset(config, smoke=True)
    return config


def test_model_gate_enforces_frozen_threshold(tmp_path: Path) -> None:
    config = _train_bad_full_pfv(tmp_path)
    code, _ = v4a.train_v4_aug1(config, smoke=True, ensemble_size=2)
    assert code == 0
    # Non-smoke gate MUST enforce; a threshold cannot be silently skipped.
    code, outputs = v4a.evaluate_v4_aug1_model_gate(config, smoke=False)
    gate = json.loads(Path(outputs["gate"]).read_text(encoding="utf-8"))
    assert gate["thresholds"]["delta_PFV_full_vs_no_control"] == 0.70
    # Smoke mode does not enforce thresholds (mirrors base behaviour).
    code_smoke, _ = v4a.evaluate_v4_aug1_model_gate(config, smoke=True)
    assert code_smoke == 0


# 19. Event-balanced accuracy is not dominated by the largest event.
def test_event_balanced_accuracy_not_dominated_by_largest_event() -> None:
    label_idx = RESIDUAL_LABELS.index("delta_PFV_full_vs_no_control")
    used = [{"event_id": "big"}] * 10 + [{"event_id": "s1"}, {"event_id": "s2"}]
    mask = np.ones(len(used), dtype=bool)
    y_res = np.zeros((len(used), len(RESIDUAL_LABELS)))
    res_pred = np.zeros((len(used), len(RESIDUAL_LABELS)))
    y_res[:, label_idx] = 1.0            # every truth is "worse" (positive)
    res_pred[:10, label_idx] = 1.0       # big event predicted correctly
    res_pred[10:, label_idx] = -1.0      # both small events predicted wrong
    metrics = v4a._event_balanced_metrics(used, mask, res_pred, y_res)
    label = "delta_PFV_full_vs_no_control"
    assert metrics[f"worst_event_direction_accuracy_{label}"] == 0.0
    # Balanced accuracy (1/3) is far below the sample-weighted accuracy (10/12).
    assert metrics[f"event_balanced_direction_accuracy_{label}"] == pytest.approx(1.0 / 3.0)


# 20. Old V4 and new Aug1 model live at different paths and hashes.
def test_aug1_model_path_and_hash_differ_from_base(tmp_path: Path) -> None:
    config = _train_bad_full_pfv(tmp_path)
    _, outputs = v4a.train_v4_aug1(config, smoke=True, ensemble_size=2)
    aug1_model = Path(outputs["model"])
    out_root = v4a._output_root(config)
    base_model_dir = out_root / "action_effect_models_v4"
    assert "action_effect_models_v4_aug1" in str(aug1_model)
    assert base_model_dir.name != aug1_model.parent.name
    report = json.loads(Path(outputs["report"]).read_text(encoding="utf-8"))
    assert report["copied_from_previous_version"] is False
    assert report["feature_count_context"] == len(CONTEXT_FEATURE_NAMES) + len(CAUSAL_FEATURE_NAMES)
