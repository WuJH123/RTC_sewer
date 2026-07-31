from __future__ import annotations

import json
from pathlib import Path

from sewerrtc.state.hotstart_acceleration import (
    cache_key,
    object_order_signature,
    validate_cache_manifest,
)


def test_cache_key_changes_when_any_hash_changes() -> None:
    base = {
        "network_sha256": "a",
        "rainfall_sha256": "b",
        "policy_hash": "c",
        "checkpoint_id": "cp",
        "checkpoint_phase": "post_hydraulic_step_pre_rtc_decision",
        "engine_hash": "d",
        "config_hash": "e",
        "controller_prefix_hash": "f",
    }
    changed = dict(base)
    changed["controller_prefix_hash"] = "different"

    assert cache_key(base) != cache_key(changed)


def test_object_count_same_order_different_is_not_eligible(tmp_path: Path) -> None:
    source = tmp_path / "source.inp"
    clone = tmp_path / "clone.inp"
    source.write_text("[JUNCTIONS]\nA 0 0 0\nB 0 0 0\n[CONDUITS]\nL1 A B 1 1 1 1\n", encoding="utf-8")
    clone.write_text("[JUNCTIONS]\nB 0 0 0\nA 0 0 0\n[CONDUITS]\nL1 A B 1 1 1 1\n", encoding="utf-8")

    signature = object_order_signature(source, clone)

    assert signature["hotstart_eligible"] is False
    assert any(row["same_order"] is False for row in signature["sections"])


def test_validate_cache_manifest_rejects_engine_hash_change(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    payload = {"cache_key": "x", "engine_hash": "engine-a", "validation_status": "created"}
    manifest.write_text(json.dumps(payload), encoding="utf-8")

    result = validate_cache_manifest(manifest, {"engine_hash": "engine-b"})

    assert result["status"] == "stale"
    assert "engine_hash" in result["stale_fields"]
