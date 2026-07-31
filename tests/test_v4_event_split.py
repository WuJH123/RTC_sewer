import pandas as pd

from sewerrtc.v4.event_splits import select_train1600_events
from sewerrtc.v4.inventory import partition_events
from train_v3_helpers import SPLIT_COUNTS, make_ledger, make_standard_catalog


def test_event_partition_is_rainfall_hash_isolated_and_reserves_are_excluded() -> None:
    events = pd.DataFrame(
        {
            "event_id": [f"e{i}" for i in range(80)],
            "rainfall_sha256": [f"r{i}" for i in range(80)],
            "eligible": True,
            "revealed": False,
        }
    )
    partition = partition_events(
        events,
        {"train": 48, "calibration": 8, "locked_validation": 8, "reserve": 16},
    )

    assert partition["split"].value_counts().to_dict() == {
        "train": 48,
        "reserve": 16,
        "calibration": 8,
        "locked_validation": 8,
    }
    assert not partition.groupby("rainfall_sha256")["split"].nunique().gt(1).any()


def test_v3_selection_48_8_8_16_is_rainfall_sha_isolated() -> None:
    catalog = make_standard_catalog(80)
    ledger = make_ledger(catalog)

    selection = select_train1600_events(catalog, ledger, counts=SPLIT_COUNTS)

    assert {k: len(v) for k, v in selection.items()} == SPLIT_COUNTS
    sha_of_event = (
        catalog.drop_duplicates("event_id")
        .set_index("event_id")["rainfall_sha256"]
        .astype(str)
    )
    sha_split: dict[str, str] = {}
    for split, events in selection.items():
        for event in events:
            sha = sha_of_event[str(event)]
            # One event id / rainfall SHA belongs to exactly one split.
            assert sha_split.setdefault(sha, split) == split
    assert len(sha_split) == 80
