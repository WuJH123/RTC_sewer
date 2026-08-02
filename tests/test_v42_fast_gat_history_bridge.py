import pandas as pd

from scripts.materialize_v42_fast_gat_history import _resolve_history_detail


def test_candidate_window_is_not_accepted_as_history_source():
    manifest = pd.DataFrame(
        [
            {"event_id": "e1", "rainfall_sha256": "r1", "detail_path": "candidate.csv", "history_start_min": 70, "history_end_min": 130},
        ]
    )
    assert _resolve_history_detail(
        window_manifest=manifest,
        event_id="e1",
        rainfall_sha="r1",
        checkpoint=130,
        candidate_path="candidate.csv",
    ) is None


def test_same_state_prefix_source_covers_gat_warmup():
    manifest = pd.DataFrame(
        [
            {"event_id": "e1", "rainfall_sha256": "r1", "detail_path": "whole_event.csv", "history_start_min": 10, "history_end_min": 70},
            {"event_id": "e1", "rainfall_sha256": "r1", "detail_path": "candidate.csv", "history_start_min": 70, "history_end_min": 130},
        ]
    )
    assert _resolve_history_detail(
        window_manifest=manifest,
        event_id="e1",
        rainfall_sha="r1",
        checkpoint=130,
        candidate_path="candidate.csv",
    ).name == "whole_event.csv"
