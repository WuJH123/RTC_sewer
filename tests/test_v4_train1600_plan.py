from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from sewerrtc.v4.event_splits import (
    EventShortfallError,
    build_event_usage_ledger,
    select_train1600_events,
)
from sewerrtc.v4.pipeline import _plan_train_handler
from sewerrtc.v4.runtime import RuntimeOptions
from sewerrtc.v4.training_plan import (
    audit_train1600_plan,
    build_round0_plan,
    build_train1600_target_plan,
    build_train_checkpoint_catalog,
)


FAMILIES = ("frontal", "convective", "typhoon")
RISKS = ("low", "medium", "high")

SPLIT_COUNTS = {"train": 48, "calibration": 8, "locked_validation": 8, "reserve": 16}


def make_standard_catalog(num_events: int) -> pd.DataFrame:
    rows = []
    for index in range(num_events):
        event = f"ev{index:03d}"
        sha = hashlib.sha256(event.encode()).hexdigest()
        for checkpoint_index in range(5):
            responsive = checkpoint_index < 4
            rows.append(
                {
                    "event_id": event,
                    "rainfall_sha256": sha,
                    "checkpoint_id": f"{event}_cp{checkpoint_index}",
                    "elapsed_min": (
                        30 + checkpoint_index * 40
                        if responsive
                        else 240 + (index % 3) * 120
                    ),
                    "checkpoint_role": (
                        "responsive" if responsive else "low_opportunity"
                    ),
                    "rainfall_phase": "rising",
                    "opportunity_score": float(10 - checkpoint_index),
                    "event_tier": "standard_4plus",
                    "checkpoint_state_source": "cold_start_prefix_replay",
                    "network_sha256": "net",
                    "config_sha256": "cfg",
                    "source_run_uuid": "uuid",
                    "rainfall_family": FAMILIES[index % 3],
                    "risk_level": RISKS[index % 3],
                }
            )
    return pd.DataFrame(rows)


def make_ledger(catalog: pd.DataFrame) -> pd.DataFrame:
    return build_event_usage_ledger(
        catalog[["event_id", "rainfall_sha256", "event_tier"]],
        scanned_event_ids=set(catalog["event_id"].astype(str)),
    )


def test_train_split_excludes_pilot_events_and_returns_48_8_8_16() -> None:
    catalog = make_standard_catalog(81)
    ledger = make_ledger(catalog)
    pilot_event = "ev080"
    mask = ledger["event_id"] == pilot_event
    ledger.loc[mask, "used_pilot"] = True
    ledger.loc[mask, "assigned_split"] = "pilot"

    selection = select_train1600_events(catalog, ledger, counts=SPLIT_COUNTS)

    assert {split: len(events) for split, events in selection.items()} == SPLIT_COUNTS
    all_selected = [event for events in selection.values() for event in events]
    assert pilot_event not in all_selected
    assert len(set(all_selected)) == 80


def test_train_catalog_320_reserve_80_target_1600_round0_400() -> None:
    catalog = make_standard_catalog(80)
    ledger = make_ledger(catalog)
    selection = select_train1600_events(catalog, ledger, counts=SPLIT_COUNTS)

    train_catalog, reserve_catalog = build_train_checkpoint_catalog(
        catalog, selection
    )
    assert len(train_catalog) == 320
    assert train_catalog["event_id"].nunique() == 64
    assert len(reserve_catalog) == 80
    assert reserve_catalog["event_id"].nunique() == 16

    target_plan = build_train1600_target_plan(train_catalog)
    assert len(target_plan) == 1600
    per_state = target_plan.groupby(["event_id", "checkpoint_id"]).size()
    assert per_state.eq(5).all()
    # All candidates of one checkpoint stay in the event's frozen split.
    assert (
        not target_plan.groupby(["event_id", "checkpoint_id"])["split"]
        .nunique()
        .gt(1)
        .any()
    )
    assert (
        not target_plan.groupby("rainfall_sha256")["split"].nunique().gt(1).any()
    )

    round0 = build_round0_plan(train_catalog)
    assert len(round0) == 400
    assert not round0["case_id"].duplicated().any()


def test_shortfall_fails_closed_with_report_and_no_padding() -> None:
    catalog = make_standard_catalog(79)
    ledger = make_ledger(catalog)

    with pytest.raises(EventShortfallError) as excinfo:
        select_train1600_events(catalog, ledger, counts=SPLIT_COUNTS)
    report = excinfo.value.report
    assert report["required_events"] == 80
    assert report["usable_events"] == 79
    assert report["shortfall"] == 1
    assert report["policy"]["no_short_event_padding"] is True
    assert report["policy"]["no_pilot_reuse"] is True


def test_audit_train1600_plan_uses_native_bool_json() -> None:
    catalog = make_standard_catalog(80)
    ledger = make_ledger(catalog)
    selection = select_train1600_events(catalog, ledger, counts=SPLIT_COUNTS)
    train_catalog, reserve_catalog = build_train_checkpoint_catalog(
        catalog, selection
    )

    audit = audit_train1600_plan(train_catalog, reserve_catalog, selection)

    assert audit["status"] == "pass"
    assert all(type(value) is bool for value in audit["checks"].values())
    json.dumps(audit, allow_nan=False)


def _setup_handler_env(tmp_path: Path, *, verdict: dict | None) -> Path:
    output = tmp_path / "out"
    if verdict is not None:
        verdict_path = output / "pilot" / "evaluation" / "pilot_gate_verdict.json"
        verdict_path.parent.mkdir(parents=True, exist_ok=True)
        verdict_path.write_text(json.dumps(verdict), encoding="utf-8")
    return output


def _write_canonical_inputs(output: Path, num_events: int = 80) -> pd.DataFrame:
    catalog = make_standard_catalog(num_events)
    ledger = make_ledger(catalog)
    (output / "opportunities").mkdir(parents=True, exist_ok=True)
    (output / "inventory").mkdir(parents=True, exist_ok=True)
    catalog.to_csv(
        output / "opportunities" / "standard_checkpoint_catalog.csv", index=False
    )
    ledger.to_csv(output / "inventory" / "event_usage_ledger.csv", index=False)
    return catalog


def test_train_handler_generates_catalog_resumes_and_rejects_stale(
    tmp_path: Path,
) -> None:
    output = _setup_handler_env(
        tmp_path, verdict={"scientific_pass": True, "exit_code": 0}
    )
    _write_canonical_inputs(output)
    handler = _plan_train_handler(
        tmp_path, output, {"train1600": {"split": SPLIT_COUNTS}}
    )
    options = RuntimeOptions(stage="PlanTrain1600")

    result = handler(options)
    assert result.exit_code == 0
    assert result.scope_complete
    planning = output / "train1600" / "planning"
    catalog = pd.read_csv(planning / "train_checkpoint_catalog.csv")
    assert len(catalog) == 320
    reserve = pd.read_csv(planning / "reserve_checkpoint_catalog.csv")
    assert len(reserve) == 80
    round0 = pd.read_csv(output / "train1600" / "round0" / "plan.csv")
    assert len(round0) == 400
    completion = json.loads(
        (planning / "completion.json").read_text(encoding="utf-8")
    )
    assert completion["input_identity_sha256"]
    ledger = pd.read_csv(output / "inventory" / "event_usage_ledger.csv")
    assert (ledger["assigned_split"] == "train").sum() == 48
    assert (ledger["assigned_split"] == "calibration").sum() == 8
    assert (ledger["assigned_split"] == "locked_validation").sum() == 8
    assert (ledger["assigned_split"] == "reserve").sum() == 16

    resumed = handler(options)
    assert resumed.exit_code == 0
    assert resumed.evidence["resumed"] is True

    # Any input mutation makes the frozen plan stale and fails closed.
    ledger.loc[0, "exclusion_reason"] = "tampered"
    ledger.to_csv(output / "inventory" / "event_usage_ledger.csv", index=False)
    stale = handler(options)
    assert stale.exit_code != 0
    assert stale.evidence["reason"] == "stale_train_plan_detected"


def test_train_handler_fails_closed_without_pilot_gate(tmp_path: Path) -> None:
    missing = _setup_handler_env(tmp_path, verdict=None)
    _write_canonical_inputs(missing)
    handler = _plan_train_handler(
        tmp_path, missing, {"train1600": {"split": SPLIT_COUNTS}}
    )
    result = handler(RuntimeOptions(stage="PlanTrain1600"))
    assert result.exit_code != 0
    assert result.evidence["reason"] == "pilot_gate_verdict_missing"
    assert not (
        missing / "train1600" / "planning" / "train_checkpoint_catalog.csv"
    ).exists()

    failed = _setup_handler_env(
        tmp_path / "second", verdict={"scientific_pass": False, "exit_code": 5}
    )
    _write_canonical_inputs(failed)
    handler = _plan_train_handler(
        tmp_path / "second", failed, {"train1600": {"split": SPLIT_COUNTS}}
    )
    result = handler(RuntimeOptions(stage="PlanTrain1600"))
    assert result.exit_code != 0
    assert result.evidence["reason"] == "pilot_gate_not_passed"
    assert not (
        failed / "train1600" / "planning" / "train_checkpoint_catalog.csv"
    ).exists()


def test_train_handler_never_creates_empty_files_for_missing_inputs(
    tmp_path: Path,
) -> None:
    output = _setup_handler_env(
        tmp_path, verdict={"scientific_pass": True, "exit_code": 0}
    )
    handler = _plan_train_handler(
        tmp_path, output, {"train1600": {"split": SPLIT_COUNTS}}
    )
    result = handler(RuntimeOptions(stage="PlanTrain1600"))
    assert result.exit_code != 0
    assert result.evidence["reason"] == "canonical_inputs_missing"
    assert not (output / "train1600").exists()
    assert not (
        output / "opportunities" / "standard_checkpoint_catalog.csv"
    ).exists()
