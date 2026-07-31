from __future__ import annotations

from pathlib import Path

import pandas as pd

from sewerrtc.state.interface_cache import (
    cache_validity,
    compare_interface_frames,
    runoff_cache_eligible,
)


def test_hydrology_hash_change_invalidates_cache() -> None:
    manifest = {"hydrology_contract_hash": "a", "rainfall_hash": "b"}
    expected = {"hydrology_contract_hash": "changed", "rainfall_hash": "b"}

    result = cache_validity(manifest, expected)

    assert result["status"] == "stale"
    assert result["stale_fields"] == ["hydrology_contract_hash"]


def test_runoff_cache_not_eligible_when_candidate_changes_hydrology() -> None:
    event = {
        "rainfall_path_exists": True,
        "subcatchment_parameters_fixed": True,
        "candidate_modifies_hydrology": True,
    }

    result = runoff_cache_eligible(event)

    assert result["runoff_cache_eligible"] is False
    assert "candidate_modifies_hydrology" in result["blocking_reasons"]


def test_rainfall_or_hydraulic_mismatch_fails_equivalence() -> None:
    reference = pd.DataFrame({"elapsed_min": [5.0], "rainfall_mm_h": [1.0], "h:A": [2.0]})
    cached = pd.DataFrame({"elapsed_min": [5.0], "rainfall_mm_h": [2.0], "h:A": [2.0]})

    result = compare_interface_frames(reference, cached)

    assert result["status"] == "failed_gate"
    assert result["rainfall_equivalence"] == "failed_gate"


def test_matching_frames_pass_equivalence() -> None:
    reference = pd.DataFrame({"elapsed_min": [5.0], "rainfall_mm_h": [1.0], "h:A": [2.0], "flow:L": [3.0]})
    cached = reference.copy()

    result = compare_interface_frames(reference, cached)

    assert result["status"] == "pass"
    assert result["hydraulic_continuation"] == "pass"
