from pathlib import Path

import pandas as pd
import pytest

from sewerrtc.v4 import simulation
from sewerrtc.v4.pipeline import (
    LONG_RUN_STAGES,
    RUN_STAGE_GROUP_KEYS,
    RUN_STAGE_PLANS,
)
from sewerrtc.v4.runtime import (
    EXIT_RUNTIME_ERROR,
    RuntimeOptions,
    atomic_write_json,
    discover_completions,
    run_parallel_cases,
    select_pending,
)


def test_resume_reads_all_valid_completions_then_applies_limit(tmp_path: Path) -> None:
    for case_id in ("a", "b"):
        case_dir = tmp_path / case_id
        case_dir.mkdir()
        atomic_write_json(
            case_dir / "completion.json",
            {"case_id": case_id, "status": "pass", "input_sha": "i"},
        )
    plan = pd.DataFrame({"case_id": ["a", "b", "c", "d"]})
    completed = discover_completions(tmp_path, expected_input_sha="i")
    pending = select_pending(plan, completed=completed, limit=1)

    assert completed == {"a", "b"}
    assert pending["case_id"].tolist() == ["c"]


def test_resume_never_converts_terminal_failed_cases_into_success(
    tmp_path: Path,
) -> None:
    plan = pd.DataFrame({"case_id": ["failed-case"]})
    atomic_write_json(
        tmp_path / "failed-case" / "completion.json",
        {
            "case_id": "failed-case",
            "status": "failed",
            "input_sha": "input",
        },
    )

    result = run_parallel_cases(
        plan,
        run_root=tmp_path,
        worker=None,
        options=RuntimeOptions(stage="Resume", resume=True),
        input_sha="input",
        minimum_free_bytes=1,
        minimum_free_memory_bytes=1,
    )

    assert result.exit_code == EXIT_RUNTIME_ERROR
    assert not result.scope_complete
    assert result.evidence["terminal_failed"] == 1


def test_nonempty_hotstart_dir_is_rejected_fail_closed(tmp_path: Path) -> None:
    row = {
        "runner_function": "run_swmm_fixed_action",
        "runner_kwargs": {"hotstart_dir": "X:/real_hotstart"},
    }

    with pytest.raises(ValueError, match="prohibits hot-start"):
        simulation.run_prepared_case(row, {"directory": str(tmp_path)})


def test_none_hotstart_marker_is_stripped_without_mutating_caller_kwargs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured = {}

    def stub_runner(**kwargs):
        captured.update(kwargs)
        return {"status": "ok"}

    monkeypatch.setattr(
        simulation.pyswmm_runner, "run_swmm_fixed_action", stub_runner
    )
    raw_kwargs = {"hotstart_dir": None}
    row = {
        "runner_function": "run_swmm_fixed_action",
        "runner_kwargs": raw_kwargs,
    }

    result = simulation.run_prepared_case(row, {"directory": str(tmp_path)})

    assert result["runner_function"] == "run_swmm_fixed_action"
    assert "hotstart_dir" not in captured
    # Caller kwargs must never be mutated by the facade.
    assert raw_kwargs == {"hotstart_dir": None}


def test_limit_with_sample_group_key_keeps_pilot_samples_atomic() -> None:
    # RunPilot400 groups by sample_id: Limit=1 selects one whole sample
    # rather than one stray branch row.
    plan = pd.DataFrame(
        {
            "case_id": [f"s{i}__{b}" for i in (1, 2) for b in ("a", "b")],
            "sample_id": ["s1", "s1", "s2", "s2"],
        }
    )
    pending = select_pending(
        plan, completed=set(), limit=1, stratified=False, group_key="sample_id"
    )

    assert pending["sample_id"].unique().tolist() == ["s1"]
    assert len(pending) == 2


def test_completed_pilot_samples_are_not_rerun_on_resume(tmp_path: Path) -> None:
    # Sample-level completions (one marker per sample task) resume cleanly.
    atomic_write_json(
        tmp_path / "s1" / "completion.json",
        {"case_id": "s1", "status": "pass", "input_sha": "i"},
    )
    plan = pd.DataFrame({"case_id": ["s1", "s2"], "sample_id": ["s1", "s2"]})
    completed = discover_completions(tmp_path, expected_input_sha="i")
    pending = select_pending(
        plan, completed=completed, limit=0, stratified=False,
        group_key="sample_id",
    )

    assert completed == {"s1"}
    assert pending["case_id"].tolist() == ["s2"]


def test_train_v3_run_stages_resume_atomically_by_sample() -> None:
    # All five V3 SWMM segments resume with whole-sample batches and read
    # their plans from the train1600_v3 tree (the calibration and locked
    # plans being the frozen artifacts published by their Plan stages).
    for run_stage in (
        "RunTrainRound0V3",
        "RunTrainRound1V3",
        "RunTrainRound2V3",
        "RunCalibration200V3",
        "RunLockedValidation200V3",
    ):
        assert run_stage in LONG_RUN_STAGES
        assert RUN_STAGE_GROUP_KEYS[run_stage] == "sample_id"
        assert RUN_STAGE_PLANS[run_stage].startswith("train1600_v3/")
