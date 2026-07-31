from __future__ import annotations

from pathlib import Path

import pandas as pd

from sewerrtc.v4.preflight import (
    MAX_WORKERS,
    WORKER_IMPORT_CHAIN,
    preflight_checks,
    worker_import_is_torch_free,
)


def make_plan(rows: int = 4) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "case_id": [f"case{i}" for i in range(rows)],
            "event_id": [f"e{i % 2}" for i in range(rows)],
            "checkpoint_id": [f"c{i}" for i in range(rows)],
            "split": ["train"] * rows,
            "projected_schedule_sha256": [f"proj{i}" for i in range(rows)],
            "requested_schedule_sha256": [f"req{i}" for i in range(rows)],
            "K": [2] * rows,
            "binary_semantics_ok": [True] * rows,
            "rate_limit_ok": [True] * rows,
            "dwell_ok": [True] * rows,
            "interlock_ok": [True] * rows,
        }
    )


def good_evidence() -> dict:
    return {
        "writer_lock_free": True,
        "reference_cache_clean": True,
        "active_conflicting_pids": [],
        "torch_free_override": True,
    }


def test_preflight_passes_with_full_evidence(tmp_path: Path) -> None:
    report = preflight_checks(
        make_plan(),
        workers=16,
        output_root=tmp_path,
        input_sha="sha",
        evidence=good_evidence(),
        minimum_free_bytes=1,
    )

    assert report["status"] == "pass"
    assert report["exit_code"] == 0
    assert all(report["checks"].values())


def test_preflight_failure_returns_nonzero(tmp_path: Path) -> None:
    plan = make_plan()
    plan.loc[plan.index[0], "K"] = 9
    report = preflight_checks(
        plan,
        workers=16,
        output_root=tmp_path,
        input_sha="sha",
        evidence=good_evidence(),
        minimum_free_bytes=1,
    )

    assert report["status"] == "blocked"
    assert report["exit_code"] != 0
    assert not report["checks"]["k_le_8"]


def test_preflight_missing_evidence_fails_closed(tmp_path: Path) -> None:
    report = preflight_checks(
        make_plan(),
        workers=16,
        output_root=tmp_path,
        input_sha="sha",
        evidence={"torch_free_override": True},
        minimum_free_bytes=1,
    )

    assert report["status"] == "blocked"
    assert not report["checks"]["writer_lock_free"]
    assert not report["checks"]["reference_cache_clean"]
    assert not report["checks"]["no_conflicting_active_pids"]


def test_preflight_rejects_more_than_16_workers(tmp_path: Path) -> None:
    assert MAX_WORKERS == 16
    report = preflight_checks(
        make_plan(),
        workers=17,
        output_root=tmp_path,
        input_sha="sha",
        evidence=good_evidence(),
        minimum_free_bytes=1,
    )

    assert report["status"] == "blocked"
    assert not report["checks"]["workers_le_16"]


def test_preflight_rejects_requested_schedule_duplicates(
    tmp_path: Path,
) -> None:
    plan = make_plan()
    plan.loc[plan.index[1], "requested_schedule_sha256"] = plan.loc[
        plan.index[0], "requested_schedule_sha256"
    ]
    plan.loc[plan.index[1], "event_id"] = plan.loc[plan.index[0], "event_id"]
    plan.loc[plan.index[1], "checkpoint_id"] = plan.loc[
        plan.index[0], "checkpoint_id"
    ]
    report = preflight_checks(
        plan,
        workers=16,
        output_root=tmp_path,
        input_sha="sha",
        evidence=good_evidence(),
        minimum_free_bytes=1,
    )

    assert report["status"] == "blocked"
    assert not report["checks"]["requested_schedule_duplicates_zero"]


def test_preflight_rejects_split_leakage(tmp_path: Path) -> None:
    plan = make_plan()
    plan.loc[plan.index[0], "split"] = "val"  # e0 now spans two splits
    report = preflight_checks(
        plan,
        workers=16,
        output_root=tmp_path,
        input_sha="sha",
        evidence=good_evidence(),
        minimum_free_bytes=1,
    )

    assert report["status"] == "blocked"
    assert not report["checks"]["event_split_isolated"]


def test_worker_import_chain_is_really_torch_free() -> None:
    # Real subprocess probe over the exact modules a 16-worker SWMM pool
    # imports; fails if any of them pulls torch in.
    assert "sewerrtc.v4.runtime" in WORKER_IMPORT_CHAIN
    assert worker_import_is_torch_free() is True


def test_torch_probe_failure_blocks_preflight(tmp_path: Path) -> None:
    evidence = good_evidence()
    evidence["torch_free_override"] = False
    report = preflight_checks(
        make_plan(),
        workers=16,
        output_root=tmp_path,
        input_sha="sha",
        evidence=evidence,
        minimum_free_bytes=1,
    )

    assert report["status"] == "blocked"
    assert not report["checks"]["worker_import_torch_free"]


def test_reference_branches_do_not_count_as_requested_duplicates(
    tmp_path: Path,
) -> None:
    # A four-branch peak plan repeats the candidate's requested SHA on its
    # reference branches; only candidate-branch duplicates may block.
    plan = pd.concat(
        [
            make_plan(2).assign(branch="candidate"),
            make_plan(2).assign(branch="no_control"),
            make_plan(2).assign(branch="dynamic_internal_rules"),
            make_plan(2).assign(branch="hold_previous"),
        ],
        ignore_index=True,
    )
    report = preflight_checks(
        plan,
        workers=16,
        output_root=tmp_path,
        input_sha="sha",
        evidence=good_evidence(),
        minimum_free_bytes=1,
    )
    assert report["checks"]["requested_schedule_duplicates_zero"]
    assert report["requested_schedule_duplicates"] == 0

    # But duplicates inside the candidate branch itself still block.
    dupe = plan.copy()
    candidates = dupe[dupe["branch"].eq("candidate")].index
    dupe.loc[candidates[1], "requested_schedule_sha256"] = dupe.loc[
        candidates[0], "requested_schedule_sha256"
    ]
    dupe.loc[candidates[1], "event_id"] = dupe.loc[candidates[0], "event_id"]
    dupe.loc[candidates[1], "checkpoint_id"] = dupe.loc[
        candidates[0], "checkpoint_id"
    ]
    report = preflight_checks(
        dupe,
        workers=16,
        output_root=tmp_path,
        input_sha="sha",
        evidence=good_evidence(),
        minimum_free_bytes=1,
    )
    assert not report["checks"]["requested_schedule_duplicates_zero"]


def test_k_target_column_satisfies_k_check(tmp_path: Path) -> None:
    plan = make_plan().rename(columns={"K": "k_target"})
    report = preflight_checks(
        plan,
        workers=16,
        output_root=tmp_path,
        input_sha="sha",
        evidence=good_evidence(),
        minimum_free_bytes=1,
    )
    assert report["checks"]["k_le_8"]

    plan.loc[plan.index[0], "k_target"] = 9
    report = preflight_checks(
        plan,
        workers=16,
        output_root=tmp_path,
        input_sha="sha",
        evidence=good_evidence(),
        minimum_free_bytes=1,
    )
    assert not report["checks"]["k_le_8"]
