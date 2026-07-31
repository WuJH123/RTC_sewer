from pathlib import Path

import pandas as pd
import pytest

from sewerrtc.v4.pipeline_train_v3 import (
    FREEZE_POINTER_REL,
    FREEZE_ROOT_REL,
    _aggregate_sha,
    _reference_cache_check,
    _sha_file,
)
from sewerrtc.v4.reference_cache import (
    REFERENCE_BRANCHES,
    read_reference_completion,
    reference_dir,
    write_reference_completion,
)


IDENTITY = {
    "network_sha256": "net",
    "config_sha256": "cfg",
    "contract_sha256": "contract",
}


def _materialize_branches(root: Path) -> dict[str, list[str]]:
    directory = reference_dir(root, "e1", "c1")
    artifacts = {}
    for branch in REFERENCE_BRANCHES:
        branch_dir = directory / branch
        branch_dir.mkdir(parents=True, exist_ok=True)
        (branch_dir / "detail.csv").write_text("t,v\n0,1\n", encoding="utf-8")
        artifacts[branch] = ["detail.csv"]
    return artifacts


def test_reference_cache_roundtrip_never_counts_as_sample(tmp_path: Path) -> None:
    artifacts = _materialize_branches(tmp_path)
    write_reference_completion(
        tmp_path, "e1", "c1", identity=IDENTITY, branch_artifacts=artifacts
    )
    payload = read_reference_completion(
        tmp_path, "e1", "c1", expected_identity=IDENTITY
    )

    assert payload["status"] == "complete"
    assert payload["counted_as_sample"] is False
    assert set(payload["branches"]) == set(REFERENCE_BRANCHES)


def test_incomplete_or_mismatched_reference_cache_is_unreadable(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError):
        read_reference_completion(
            tmp_path, "e1", "c1", expected_identity=IDENTITY
        )

    artifacts = _materialize_branches(tmp_path)
    # Missing branch artifacts refuse to publish a completion marker.
    partial = {branch: artifacts[branch] for branch in REFERENCE_BRANCHES[:2]}
    with pytest.raises(ValueError, match="missing branches"):
        write_reference_completion(
            tmp_path, "e1", "c1", identity=IDENTITY, branch_artifacts=partial
        )

    write_reference_completion(
        tmp_path, "e1", "c1", identity=IDENTITY, branch_artifacts=artifacts
    )
    with pytest.raises(ValueError, match="identity mismatch"):
        read_reference_completion(
            tmp_path,
            "e1",
            "c1",
            expected_identity={**IDENTITY, "network_sha256": "other"},
        )

    # SHA-verified artifacts: silent tampering makes the cache unreadable.
    tampered = reference_dir(tmp_path, "e1", "c1") / "no_control" / "detail.csv"
    tampered.write_text("t,v\n0,999\n", encoding="utf-8")
    with pytest.raises(ValueError, match="SHA mismatch"):
        read_reference_completion(
            tmp_path, "e1", "c1", expected_identity=IDENTITY
        )


def _freeze_reference_manifest(output_root: Path) -> Path:
    """Freeze-time manifest over the current pilot/references files."""
    import json

    reference_root = output_root / "pilot" / "references"
    reference_root.mkdir(parents=True, exist_ok=True)
    for name in ("a.csv", "b.csv"):
        (reference_root / name).write_text(f"t,v\n0,{name}\n", "utf-8")
    manifest = {
        name: _sha_file(reference_root / name) for name in ("a.csv", "b.csv")
    }
    frozen = output_root / FREEZE_ROOT_REL / "codesha"
    frozen.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {"path": list(manifest), "sha256": list(manifest.values())}
    ).to_csv(frozen / "reference_cache_sha_manifest.csv", index=False)
    (frozen / "pilot_feasibility_p3_freeze.json").write_text(
        json.dumps({"reference_cache_sha256": _aggregate_sha(manifest)}),
        "utf-8",
    )
    pointer = output_root / FREEZE_POINTER_REL
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(
        json.dumps(
            {"frozen_dir_rel": f"{FREEZE_ROOT_REL}/codesha"}
        ),
        "utf-8",
    )
    return reference_root


def test_frozen_reference_cache_check_tolerates_new_files_only(
    tmp_path: Path,
) -> None:
    reference_root = _freeze_reference_manifest(tmp_path)

    expected, actual, report = _reference_cache_check(tmp_path)
    assert report["frozen_manifest_available"]
    assert expected == actual
    assert report["frozen_files_missing"] == 0

    # New cache entries for new V3 states never break consistency.
    (reference_root / "new_state.csv").write_text("t,v\n0,9\n", "utf-8")
    expected, actual, report = _reference_cache_check(tmp_path)
    assert expected == actual

    # Touching a freeze-time file breaks consistency (hard block upstream).
    (reference_root / "a.csv").write_text("t,v\n0,tampered\n", "utf-8")
    expected, actual, _report = _reference_cache_check(tmp_path)
    assert expected != actual


def test_frozen_reference_cache_check_counts_missing_files(
    tmp_path: Path,
) -> None:
    reference_root = _freeze_reference_manifest(tmp_path)
    (reference_root / "b.csv").unlink()

    expected, actual, report = _reference_cache_check(tmp_path)

    assert report["frozen_files_missing"] == 1
    assert expected != actual
