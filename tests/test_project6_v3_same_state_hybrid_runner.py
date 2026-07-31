from __future__ import annotations

from sewerrtc.state.hotstart_acceleration import run_same_state_branch


def test_certified_checkpoint_uses_hotstart() -> None:
    result = run_same_state_branch(
        {"checkpoint_id": "cp1"},
        {"candidate_id": "cand1"},
        certifications={"cp1": {"certification_status": "pass", "certification_id": "cert1"}},
    )

    assert result["actual_same_state_method"] == "verified_hotstart"
    assert result["fallback_to_replay"] is False


def test_uncertified_checkpoint_uses_replay() -> None:
    result = run_same_state_branch(
        {"checkpoint_id": "cp2"},
        {"candidate_id": "cand1"},
        certifications={"cp2": {"certification_status": "failed_gate", "failure_reason": "fingerprint"}},
    )

    assert result["actual_same_state_method"] == "deterministic_prefix_replay"
    assert result["fallback_to_replay"] is True


def test_post_load_fingerprint_mismatch_falls_back_to_replay() -> None:
    result = run_same_state_branch(
        {"checkpoint_id": "cp3"},
        {"candidate_id": "cand1"},
        certifications={"cp3": {"certification_status": "pass", "certification_id": "cert3"}},
        post_load_fingerprint_status="failed_gate",
    )

    assert result["actual_same_state_method"] == "deterministic_prefix_replay"
    assert result["fallback_reason"] == "post_load_fingerprint_failed"
