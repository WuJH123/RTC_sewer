from pathlib import Path

import pandas as pd

from sewerrtc.v4.runtime import (
    ReferenceWriteLock,
    RuntimeOptions,
    StageRegistry,
    isolated_case_paths,
    run_parallel_cases,
    resources_available,
    select_pending,
)


def _tiny_echo_worker(row: dict, paths: dict[str, str]) -> dict:
    return {"value": row["value"], "case_directory": paths["directory"]}


def test_runtime_defaults_to_16_workers_and_unknown_stage_fails() -> None:
    assert RuntimeOptions().workers == 16
    assert StageRegistry().run("DoesNotExist", RuntimeOptions()).exit_code != 0


def test_case_paths_are_process_isolated(tmp_path: Path) -> None:
    first = isolated_case_paths(tmp_path, "case-a", "run-1")
    second = isolated_case_paths(tmp_path, "case-b", "run-1")

    assert first["directory"] != second["directory"]
    assert first["inp"] != second["inp"]
    assert first["rpt"] != second["rpt"]
    assert first["out"] != second["out"]
    assert first["log"] != second["log"]


def test_limit_is_applied_after_completed_cases_are_removed() -> None:
    plan = pd.DataFrame({"case_id": ["a", "b", "c", "d"]})
    pending = select_pending(plan, completed={"a", "b"}, limit=1)

    assert pending["case_id"].tolist() == ["c"]


def test_group_key_limit_keeps_whole_samples_atomically() -> None:
    # A four-branch Peak sample must stay atomic: Limit=1 with a group key
    # yields every branch of exactly one sample, never a stray branch.
    plan = pd.DataFrame(
        {
            "case_id": [
                "s1__candidate",
                "s1__no_control",
                "s1__dynamic_internal_rules",
                "s1__hold_previous",
                "s2__candidate",
                "s2__no_control",
                "s2__dynamic_internal_rules",
                "s2__hold_previous",
            ],
            "sample_id": ["s1"] * 4 + ["s2"] * 4,
            "branch": [
                "candidate",
                "no_control",
                "dynamic_internal_rules",
                "hold_previous",
            ]
            * 2,
        }
    )
    pending = select_pending(
        plan, completed=set(), limit=1, group_key="sample_id"
    )

    assert pending["sample_id"].nunique() == 1
    assert len(pending) == 4
    assert set(pending["branch"]) == {
        "candidate",
        "no_control",
        "dynamic_internal_rules",
        "hold_previous",
    }


def test_group_key_absent_falls_back_to_row_limit() -> None:
    plan = pd.DataFrame({"case_id": ["a", "b", "c"]})
    pending = select_pending(
        plan, completed=set(), limit=1, group_key="sample_id"
    )

    assert len(pending) == 1


def test_reference_cache_allows_only_one_writer(tmp_path: Path) -> None:
    first = ReferenceWriteLock(tmp_path / "reference.lock")
    second = ReferenceWriteLock(tmp_path / "reference.lock")

    assert first.acquire()
    assert not second.acquire()
    first.release()
    assert second.acquire()
    second.release()


def test_parallel_runtime_dry_run_does_not_start_cases(tmp_path: Path) -> None:
    plan = pd.DataFrame({"case_id": ["a", "b"]})
    result = run_parallel_cases(
        plan,
        run_root=tmp_path,
        worker=None,
        options=RuntimeOptions(dry_run=True, workers=16),
        input_sha="input",
    )

    assert result.scope_complete is False
    assert result.completed == 0
    assert result.remaining == 2
    assert not list(tmp_path.glob("*/completion.json"))


def test_parallel_runtime_executes_isolated_tiny_cases_and_closes_scope(
    tmp_path: Path,
) -> None:
    plan = pd.DataFrame(
        {"case_id": ["a", "b", "c", "d"], "value": [1, 2, 3, 4]}
    )
    result = run_parallel_cases(
        plan,
        run_root=tmp_path,
        worker=_tiny_echo_worker,
        options=RuntimeOptions(stage="Tiny", workers=16),
        input_sha="input",
        minimum_free_bytes=1,
        # Resource gating has its own dedicated test below; keep this
        # execution test deterministic under transient memory pressure.
        minimum_free_memory_bytes=1,
    )

    assert result.exit_code == 0
    assert result.completed == 4
    assert result.remaining == 0
    assert result.batch_complete
    assert result.scope_complete
    assert len(list(tmp_path.glob("*/completion.json"))) == 4


def test_resource_guard_rejects_impossible_disk_or_memory_requirement(
    tmp_path: Path,
) -> None:
    impossible = 10**30
    assert not resources_available(
        tmp_path,
        minimum_free_bytes=impossible,
        minimum_free_memory_bytes=1,
    )
    assert not resources_available(
        tmp_path,
        minimum_free_bytes=1,
        minimum_free_memory_bytes=impossible,
    )
