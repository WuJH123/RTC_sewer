from __future__ import annotations

import json
import hashlib
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import pandas as pd


EXIT_PASS = 0
EXIT_BLOCKED = 2
EXIT_INCOMPLETE = 3
EXIT_RUNTIME_ERROR = 4
EXIT_SCIENTIFIC_FAIL = 5


def working_code_sha(root: str | Path) -> str:
    """Hash Git HEAD plus the active code set, including dirty changes."""
    project_root = Path(root)
    git_head = ""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        git_head = result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    paths = [
        *(project_root / "sewerrtc" / "v4").glob("*.py"),
        project_root / "scripts" / "project6_v4_final.py",
        project_root
        / "scripts"
        / "project6_runs"
        / "RUN_PROJECT6_V4_FINAL.ps1",
        project_root / "sewerrtc" / "simulation" / "pyswmm_runner.py",
        project_root / "sewerrtc" / "simulation" / "kpi_metrics.py",
        project_root / "sewerrtc" / "control" / "v4_candidate_generator.py",
    ]
    digest = hashlib.sha256()
    digest.update(git_head.encode())
    for path in sorted(item for item in paths if item.exists()):
        digest.update(str(path.relative_to(project_root)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class RuntimeOptions:
    stage: str = ""
    config: str = ""
    workers: int = 16
    limit: int = 0
    resume: bool = False
    retry_failed: bool = False
    dry_run: bool = False


@dataclass
class StageResult:
    stage: str
    status: str
    exit_code: int
    completed: int = 0
    remaining: int = 0
    batch_complete: bool = False
    scope_complete: bool = False
    evidence: dict | None = None


class StageRegistry:
    def __init__(self) -> None:
        self._handlers: dict[str, Callable[[RuntimeOptions], StageResult]] = {}

    def register(
        self, name: str, handler: Callable[[RuntimeOptions], StageResult]
    ) -> None:
        if not name or name in self._handlers:
            raise ValueError(f"invalid or duplicate stage: {name}")
        self._handlers[name] = handler

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._handlers)

    def run(self, name: str, options: RuntimeOptions) -> StageResult:
        handler = self._handlers.get(name)
        if handler is None:
            return StageResult(name, "blocked", EXIT_BLOCKED)
        return handler(options)


class ReferenceWriteLock:
    """Atomic single-writer lock for an event-checkpoint reference cache."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._owned = False

    def acquire(self) -> bool:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(
                str(self.path), os.O_CREAT | os.O_EXCL | os.O_WRONLY
            )
        except FileExistsError:
            return False
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(f"pid={os.getpid()}\n")
        self._owned = True
        return True

    def release(self) -> None:
        if self._owned:
            self.path.unlink(missing_ok=True)
            self._owned = False

    def __enter__(self) -> "ReferenceWriteLock":
        if not self.acquire():
            raise FileExistsError(str(self.path))
        return self

    def __exit__(self, *_args) -> None:
        self.release()


def atomic_write_json(path: str | Path, payload: dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".tmp", dir=str(target.parent)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, allow_nan=False)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)


def isolated_case_paths(
    root: str | Path, case_id: str, run_uuid: str
) -> dict[str, Path]:
    directory = Path(root) / str(case_id) / str(run_uuid)
    return {
        "directory": directory,
        "inp": directory / "case.inp",
        "rpt": directory / "case.rpt",
        "out": directory / "case.out",
        "log": directory / "case.log",
        "temporary": directory / "tmp",
    }


def discover_completions(
    run_root: str | Path,
    *,
    expected_input_sha: str | None = None,
    include_failed: bool = False,
) -> set[str]:
    completed: set[str] = set()
    for marker in Path(run_root).glob("*/completion.json"):
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        valid_status = payload.get("status") == "pass" or (
            include_failed and payload.get("status") == "failed"
        )
        if not valid_status:
            continue
        if (
            expected_input_sha is not None
            and payload.get("input_sha") != expected_input_sha
        ):
            continue
        completed.add(str(payload.get("case_id", marker.parent.name)))
    return completed


def completion_manifest(run_root: str | Path) -> pd.DataFrame:
    rows = []
    for marker in sorted(Path(run_root).glob("*/completion.json")):
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            continue
        payload["completion_path"] = str(marker)
        rows.append(payload)
    return pd.DataFrame(rows)


STRATIFY_COLUMNS = (
    "event_id",
    "checkpoint_id",
    "candidate_family",
    "K",
    "candidate_priority",
)


def stratified_order(plan: pd.DataFrame) -> pd.DataFrame:
    """Deterministic round-robin over event -> checkpoint -> family -> K.

    Never runs one event's rows back-to-back in CSV order: cases are dealt
    out one per event, then one per checkpoint within the event, so any
    prefix batch (1 / 16 / 40 / 64) spans as many events and states as the
    plan allows. Ties keep candidate priority then original plan order.
    """
    if "case_id" not in plan:
        raise ValueError("plan is missing case_id")
    if plan.empty or "event_id" not in plan:
        return plan.copy()
    ordered = plan.copy()
    ordered["_plan_order"] = range(len(ordered))
    levels = [
        column
        for column in ("event_id", "checkpoint_id", "candidate_family", "K")
        if column in ordered
    ]
    tie = (
        ["candidate_priority", "_plan_order"]
        if "candidate_priority" in ordered
        else ["_plan_order"]
    )
    ordered = ordered.sort_values(tie, kind="stable")
    # Deal bottom-up: rank rows inside the finest stratum, then at every
    # coarser level interleave the child sequences one card at a time.
    rank = ordered.groupby(levels, dropna=False).cumcount()
    for depth in range(len(levels) - 1, 0, -1):
        ordered["_rank"] = rank
        ordered = ordered.sort_values(
            ["_rank", "_plan_order"], kind="stable"
        )
        rank = ordered.groupby(levels[:depth], dropna=False).cumcount()
        if levels[:depth] == ["event_id", "checkpoint_id"]:
            # Rotate each state's starting card so consecutive states lead
            # with different candidate families instead of every state
            # picking its lowest-priority candidate first.
            grouped = ordered.groupby(levels[:depth], dropna=False)
            ordinal = grouped.ngroup()
            size = grouped["case_id"].transform("size")
            rank = (rank + ordinal) % size
    ordered["_rank"] = rank
    ordered = ordered.sort_values(["_rank", "_plan_order"], kind="stable")
    return ordered.drop(columns=["_rank", "_plan_order"]).reset_index(
        drop=True
    )


def select_pending(
    plan: pd.DataFrame,
    completed: Iterable[str],
    limit: int = 0,
    *,
    stratified: bool = True,
    group_key: str | None = None,
) -> pd.DataFrame:
    """Filter completed first, stratify the pending set, then apply Limit.

    ``group_key`` makes Limit count whole groups instead of individual case
    rows: a four-branch Peak sample stays atomic, so ``Limit=1`` yields one
    complete sample (all its branches) rather than a single stray branch.
    """
    if "case_id" not in plan:
        raise ValueError("plan is missing case_id")
    completed_set = {str(item) for item in completed}
    pending = plan[~plan["case_id"].astype(str).isin(completed_set)].copy()
    if stratified:
        pending = stratified_order(pending)
    if int(limit) <= 0:
        return pending
    if group_key and group_key in pending:
        keep = pending[group_key].drop_duplicates().head(int(limit))
        return pending[pending[group_key].isin(keep)].reset_index(drop=True)
    return pending.head(int(limit))


def resources_available(
    path: str | Path,
    *,
    minimum_free_bytes: int = 1_000_000_000,
    minimum_free_memory_bytes: int = 1_000_000_000,
) -> bool:
    disk_ok = shutil.disk_usage(Path(path)).free >= int(minimum_free_bytes)
    if not disk_ok:
        return False
    try:
        import psutil

        return (
            int(psutil.virtual_memory().available)
            >= int(minimum_free_memory_bytes)
        )
    except ImportError:
        # psutil is optional; disk remains a mandatory fail-closed guard.
        return True


def _invoke_case_worker(
    worker: Callable[[dict, dict[str, str]], dict],
    row: dict,
    paths: dict[str, Path],
) -> dict:
    serializable_paths = {key: str(value) for key, value in paths.items()}
    payload = dict(worker(row, serializable_paths))
    for key, value in row.items():
        if key not in payload:
            if isinstance(value, Path):
                payload[key] = str(value)
            elif pd.api.types.is_scalar(value) and pd.isna(value):
                payload[key] = None
            elif hasattr(value, "item"):
                payload[key] = value.item()
            else:
                payload[key] = value
    return payload


def run_parallel_cases(
    plan: pd.DataFrame,
    *,
    run_root: str | Path,
    worker: Callable[[dict, dict[str, str]], dict] | None,
    options: RuntimeOptions,
    input_sha: str,
    minimum_free_bytes: int = 1_000_000_000,
    minimum_free_memory_bytes: int = 1_000_000_000,
    group_key: str | None = None,
) -> StageResult:
    """Execute isolated cases with fail-closed accounting.

    Resume order is fixed: read the complete plan, remove valid completions,
    then apply ``Limit``. Failed cases are excluded unless ``RetryFailed`` is
    explicitly enabled. ``group_key`` makes ``Limit`` count whole groups (e.g.
    a four-branch Peak sample) so a partial batch never runs a stray branch.
    """
    root = Path(run_root)
    root.mkdir(parents=True, exist_ok=True)
    successful = discover_completions(
        root,
        expected_input_sha=input_sha,
    )
    successful_or_failed = discover_completions(
        root,
        expected_input_sha=input_sha,
        include_failed=True,
    )
    terminal_failed = successful_or_failed - successful
    excluded = (
        successful
        if options.retry_failed
        else successful | terminal_failed
    )
    pending = select_pending(
        plan, excluded, options.limit, group_key=group_key
    )
    total_remaining = len(
        select_pending(plan, excluded, limit=0, group_key=group_key)
    )
    if options.dry_run:
        return StageResult(
            options.stage,
            "incomplete",
            EXIT_INCOMPLETE,
            completed=0,
            remaining=int(total_remaining),
            batch_complete=False,
            scope_complete=False,
            evidence={
                "dry_run": True,
                "long_task_not_started": True,
                "selected": int(len(pending)),
                "terminal_failed": int(len(terminal_failed)),
            },
        )
    if pending.empty:
        if terminal_failed:
            return StageResult(
                options.stage,
                "runtime_error",
                EXIT_RUNTIME_ERROR,
                completed=len(successful),
                remaining=0,
                batch_complete=True,
                scope_complete=False,
                evidence={"terminal_failed": len(terminal_failed)},
            )
        return StageResult(
            options.stage,
            "pass",
            EXIT_PASS,
            completed=len(successful),
            remaining=0,
            batch_complete=True,
            scope_complete=True,
        )
    if worker is None:
        return StageResult(
            options.stage,
            "blocked",
            EXIT_BLOCKED,
            remaining=int(total_remaining),
            evidence={"reason": "case_worker_missing"},
        )
    run_uuid = str(uuid.uuid4())
    results: list[dict] = []
    failures: list[dict] = []
    maximum = min(max(1, int(options.workers)), 16)
    executor = ProcessPoolExecutor(max_workers=maximum)
    futures: dict = {}
    records = pending.to_dict(orient="records")
    next_record = 0
    submitted = 0
    resource_blocked = False
    interrupted = False

    def submit_until_full() -> None:
        nonlocal submitted, resource_blocked, next_record
        while len(futures) < maximum:
            if next_record >= len(records):
                return
            if not resources_available(
                root,
                minimum_free_bytes=minimum_free_bytes,
                minimum_free_memory_bytes=minimum_free_memory_bytes,
            ):
                resource_blocked = True
                return
            row = records[next_record]
            next_record += 1
            case_id = str(row["case_id"])
            paths = isolated_case_paths(root, case_id, run_uuid)
            paths["directory"].mkdir(parents=True, exist_ok=True)
            future = executor.submit(_invoke_case_worker, worker, row, paths)
            futures[future] = (case_id, paths)
            submitted += 1

    try:
        submit_until_full()
        while futures:
            finished, _ = wait(
                set(futures), return_when=FIRST_COMPLETED
            )
            for future in finished:
                case_id, paths = futures.pop(future)
                try:
                    payload = dict(future.result())
                    payload.update(
                        {
                            "case_id": case_id,
                            "status": "pass",
                            "input_sha": input_sha,
                        }
                    )
                    results.append(payload)
                except BaseException as exc:
                    payload = {
                        "case_id": case_id,
                        "status": "failed",
                        "input_sha": input_sha,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                    failures.append(payload)
                atomic_write_json(
                    paths["directory"].parent / "completion.json", payload
                )
            if not resource_blocked:
                submit_until_full()
    except KeyboardInterrupt:
        interrupted = True
        for future in futures:
            future.cancel()
        executor.shutdown(wait=False, cancel_futures=True)
        return StageResult(
            options.stage,
            "incomplete",
            EXIT_INCOMPLETE,
            completed=len(results),
            remaining=max(
                0, total_remaining - len(results) - len(failures)
            ),
            evidence={"interrupted": True, "failed": len(failures)},
        )
    finally:
        if not interrupted:
            executor.shutdown(wait=True, cancel_futures=False)
    valid_completed = discover_completions(root, expected_input_sha=input_sha)
    remaining = len(select_pending(plan, valid_completed, limit=0))
    status = "pass" if remaining == 0 and not failures else (
        "runtime_error" if failures else "incomplete"
    )
    exit_code = {
        "pass": EXIT_PASS,
        "incomplete": EXIT_INCOMPLETE,
        "runtime_error": EXIT_RUNTIME_ERROR,
    }[status]
    return StageResult(
        options.stage,
        status,
        exit_code,
        completed=len(valid_completed),
        remaining=remaining,
        batch_complete=(
            len(results) + len(failures) == submitted
            and not resource_blocked
        ),
        scope_complete=remaining == 0 and not failures,
        evidence={
            "failed": len(failures),
            "submitted": submitted,
            "resource_blocked": resource_blocked,
        },
    )


def stage_record(
    result: StageResult,
    *,
    config_sha: str,
    code_sha: str,
    input_sha: str,
    started_at: float,
    finished_at: float | None = None,
    run_uuid: str | None = None,
) -> dict:
    payload = asdict(result)
    payload.update(
        {
            "run_uuid": run_uuid or str(uuid.uuid4()),
            "config_sha": config_sha,
            "code_git_sha": code_sha,
            "input_sha": input_sha,
            "started_at": float(started_at),
            "finished_at": float(finished_at or time.time()),
            "completion_marker": result.scope_complete,
        }
    )
    return payload
