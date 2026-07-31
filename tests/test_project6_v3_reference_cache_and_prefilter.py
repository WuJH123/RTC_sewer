from __future__ import annotations

from sewerrtc.data.candidate_prefilter import prefilter_candidate
from sewerrtc.state.interface_cache import reference_cache_validity


def test_reference_cache_rejects_candidate_branch() -> None:
    result = reference_cache_validity({"branch_type": "candidate", "status": "completed"}, {})

    assert result["status"] == "invalid"
    assert result["reason"] == "candidate_branch_not_reference_cacheable"


def test_reference_cache_hash_change_is_stale() -> None:
    result = reference_cache_validity(
        {"branch_type": "reference", "status": "completed", "network_sha256": "a"},
        {"network_sha256": "b"},
    )

    assert result["status"] == "stale"
    assert result["stale_fields"] == ["network_sha256"]


def test_binary_pumps_reject_intermediate_values() -> None:
    keep, reason = prefilter_candidate({"binary_pump_values": {"ADD301.2": 0.5}, "override_count": 1})

    assert keep is False
    assert reason == "binary_pump_intermediate_value"


def test_same_state_fingerprint_failure_excludes_dataset_entry() -> None:
    keep, reason = prefilter_candidate({"checkpoint_fingerprint_status": "failed_gate", "override_count": 1})

    assert keep is False
    assert reason == "same_state_fingerprint_failed"
