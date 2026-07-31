"""Integration tests for the V4 model stages: freeze re-stamp on code-SHA
change (defect 1) and Locked one-shot protection (spec section 8)."""
from __future__ import annotations

import json
from pathlib import Path

from v4_model_helpers import make_catalog, make_manifest
from sewerrtc.v4.runtime import RuntimeOptions
from sewerrtc.v4.pipeline_train_v4 import (
    FREEZE_NAME,
    FREEZE_POINTER_REL,
    FREEZE_ROOT_REL,
    build_train_v4_handlers,
)
from sewerrtc.v4.pipeline_train_v4_model import (
    LOCKED_INTENT_REL,
    LOCKED_RESULT_REL,
    build_train_v4_model_handlers,
)

CONFIG = {
    "v4_true_state": {"light": True, "require_accepted_count": None},
    "thresholds": {
        "dead_zone": {"pfv_m3": 1.0, "tfv_m3": 1.0, "peak_m3s": 0.001},
        "scientific_margin": {"pfv_m3": 0.0, "tfv_m3": 0.0, "peak_m3s": 0.0},
    },
}
OPTS = RuntimeOptions(stage="", config="")


def _write_source_tree(output: Path) -> None:
    m = make_manifest()
    ds = output / "train1600_v3" / "dataset"
    pl = output / "train1600_v3" / "planning"
    ds.mkdir(parents=True, exist_ok=True)
    pl.mkdir(parents=True, exist_ok=True)
    m.to_csv(ds / "train1600_v3_sample_manifest.csv", index=False)
    make_catalog(m).to_csv(pl / "train_checkpoint_catalog_v3.csv", index=False)
    (ds / "train1600_v3_dataset_audit.json").write_text(
        json.dumps({"status": "pass", "checks": {}}), encoding="utf-8"
    )


def _freeze(tmp_path: Path, project_root: Path):
    handlers = build_train_v4_handlers(
        project_root=project_root, output_root=tmp_path, config=CONFIG
    )
    return handlers["FreezeTrain1600V3Evidence"](OPTS)


def test_freeze_restamps_on_code_sha_change(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    project_root = tmp_path  # working_code_sha is computed from this root
    _write_source_tree(output)

    # 1) First freeze creates a record + pointer under the current code sha.
    res1 = _freeze(output, project_root)
    assert res1.exit_code == 0 and res1.scope_complete
    pointer = json.loads((output / FREEZE_POINTER_REL).read_text())
    real_sha = pointer["code_sha256"]
    frozen_dir = output / pointer["frozen_dir_rel"]
    assert (frozen_dir / FREEZE_NAME).exists()

    # 2) Idempotent reuse under the same code sha.
    res2 = _freeze(output, project_root)
    assert res2.exit_code == 0
    assert res2.evidence.get("already_frozen") is True

    # 3) Simulate a code-sha change by rewriting the pointer to a stale sha
    #    (frozen files remain intact).  Freeze must re-verify and re-stamp.
    stale = dict(pointer)
    stale["code_sha256"] = "STALE" * 12
    (output / FREEZE_POINTER_REL).write_text(json.dumps(stale), encoding="utf-8")
    res3 = _freeze(output, project_root)
    assert res3.exit_code == 0 and res3.scope_complete
    assert res3.evidence.get("code_sha_rotated") is True
    new_pointer = json.loads((output / FREEZE_POINTER_REL).read_text())
    assert new_pointer["code_sha256"] == real_sha
    assert new_pointer["code_sha_rotation"]["previous_code_sha256"] == stale[
        "code_sha256"
    ]


def test_freeze_blocks_on_tampered_prior_evidence(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    project_root = tmp_path
    _write_source_tree(output)
    res1 = _freeze(output, project_root)
    assert res1.exit_code == 0
    pointer = json.loads((output / FREEZE_POINTER_REL).read_text())
    frozen_dir = output / pointer["frozen_dir_rel"]
    # Tamper a frozen file and mark the pointer stale.
    manifest_copy = frozen_dir / "dataset" / "train1600_v3_sample_manifest.csv"
    manifest_copy.write_text("event_id\ncorrupted\n", encoding="utf-8")
    stale = dict(pointer)
    stale["code_sha256"] = "STALE" * 12
    (output / FREEZE_POINTER_REL).write_text(json.dumps(stale), encoding="utf-8")
    res = _freeze(output, project_root)
    assert res.exit_code != 0
    assert res.evidence["reason"] == "prior_frozen_evidence_sha_mismatch"


def _setup_frozen_for_model(output: Path) -> None:
    """Create a frozen evidence dir + pointer the model stages can read."""
    m = make_manifest()
    frozen = output / FREEZE_ROOT_REL / "testsha"
    (frozen / "dataset").mkdir(parents=True, exist_ok=True)
    (frozen / "planning").mkdir(parents=True, exist_ok=True)
    m.to_csv(
        frozen / "dataset" / "train1600_v3_sample_manifest.csv", index=False
    )
    make_catalog(m).to_csv(
        frozen / "planning" / "train_checkpoint_catalog_v3.csv", index=False
    )
    (frozen / FREEZE_NAME).write_text(
        json.dumps({"freeze_id": "x", "file_sha256": {}}), encoding="utf-8"
    )
    pointer = output / FREEZE_POINTER_REL
    pointer.parent.mkdir(parents=True, exist_ok=True)
    pointer.write_text(
        json.dumps(
            {
                "frozen_dir_rel": f"{FREEZE_ROOT_REL}/testsha",
                "code_sha256": "testsha",
                "immutable": True,
            }
        ),
        encoding="utf-8",
    )


def test_locked_one_shot_protection(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    project_root = tmp_path
    _setup_frozen_for_model(output)
    h = build_train_v4_model_handlers(
        project_root=project_root, output_root=output, config=CONFIG
    )
    # Drive the offline chain up to Locked.
    assert h["TrainV4Baselines"](OPTS).exit_code == 0
    assert h["EvaluateV4Baselines"](OPTS).exit_code == 0
    assert h["TrainV4TrueState"](OPTS).exit_code == 0
    assert h["CalibrateV4TrueState"](OPTS).exit_code == 0

    first = h["EvaluateV4TrueStateLocked"](OPTS)
    assert first.exit_code == 0
    assert (output / LOCKED_INTENT_REL).exists()
    assert (output / LOCKED_RESULT_REL).exists()

    # Second attempt must be refused (one-shot).
    second = h["EvaluateV4TrueStateLocked"](OPTS)
    assert second.exit_code != 0
    assert second.evidence["reason"] == "locked_evaluation_already_executed"

    # Offline gate passes and keeps the Model Safety Gate deferred.
    gate = h["AuditV4OfflineSafetyGate"](OPTS)
    assert gate.exit_code == 0
    assert gate.evidence["model_safety_gate_status"] == "deferred"


def test_locked_refused_when_only_intent_exists(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    project_root = tmp_path
    _setup_frozen_for_model(output)
    intent = output / LOCKED_INTENT_REL
    intent.parent.mkdir(parents=True, exist_ok=True)
    intent.write_text(json.dumps({"one_shot": True}), encoding="utf-8")
    h = build_train_v4_model_handlers(
        project_root=project_root, output_root=output, config=CONFIG
    )
    res = h["EvaluateV4TrueStateLocked"](OPTS)
    assert res.exit_code != 0
    assert res.evidence["reason"] == "locked_evaluation_already_executed"
