from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.run_v42_fast_core_pipeline import _archive, _lock


def test_fast_lock_is_exclusive_and_cleanup(tmp_path: Path) -> None:
    lock = tmp_path / ".fast.lock"
    with _lock(lock, {"pid": 1, "mode": "smoke"}):
        assert lock.exists()
        with pytest.raises(RuntimeError):
            with _lock(lock, {"pid": 2, "mode": "potential"}):
                pass
    assert not lock.exists()


def test_archive_moves_previous_shared_run(tmp_path: Path) -> None:
    fast = tmp_path / "fast_e2e_64plus"
    archive_root = tmp_path / "fast_runs"
    fast.mkdir()
    (fast / "FAST_RUN_METADATA.json").write_text(json.dumps({"mode": "smoke"}), encoding="utf-8")
    (fast / "step2_fast_e2e_core_manifest.parquet").write_bytes(b"old-smoke-artifact")
    archived = _archive(fast, archive_root)
    assert archived is not None
    archived_path = Path(archived)
    assert archived_path.exists()
    assert (archived_path / "step2_fast_e2e_core_manifest.parquet").read_bytes() == b"old-smoke-artifact"
    assert fast.exists()
    assert list(fast.iterdir()) == []
