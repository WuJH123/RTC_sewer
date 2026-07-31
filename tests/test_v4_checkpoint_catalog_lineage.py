import hashlib
from pathlib import Path

import pandas as pd
import pytest

from sewerrtc.v4.event_splits import build_event_usage_ledger
from sewerrtc.v4.opportunity import (
    STANDARD_CATALOG_REQUIRED_COLUMNS,
    build_canonical_catalogs,
)
from sewerrtc.v4.pilot import build_pilot_planning_bundle


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "project6_dual_reference_v4"
    / "final_v4"
    / "opportunities"
)


def _sha(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()


def make_pool() -> pd.DataFrame:
    """One standard_4plus, one short_3 and one short_2 event."""
    rows = []

    def add(event: str, responsive: int, low: int, tier: str) -> None:
        for index in range(responsive + low):
            rows.append(
                {
                    "event_id": event,
                    "rainfall_sha256": _sha(event),
                    "checkpoint_id": f"{event}_cp{index}",
                    "elapsed_min": 30 + index * 40,
                    "opportunity_class": (
                        "responsive" if index < responsive else "low_opportunity"
                    ),
                    "phase": "rising",
                    "opportunity_score": float(responsive - index),
                    "event_tier": tier,
                    "source_detail": "cold_start_prefix_replay",
                    "rainfall_family": "frontal",
                    "risk_level": "high",
                }
            )

    add("std_event", 4, 1, "standard_4plus")
    add("short3_event", 3, 1, "short_3")
    add("short2_event", 2, 1, "short_2")
    return pd.DataFrame(rows)


def test_canonical_builder_isolates_short_events_from_standard_catalog() -> None:
    catalogs = build_canonical_catalogs(
        make_pool(),
        network_sha256="net",
        config_sha256="cfg",
        source_run_uuid="uuid",
    )
    standard = catalogs["standard_checkpoint_catalog"]
    short = catalogs["short_event_checkpoint_catalog"]

    assert set(standard["event_id"]) == {"std_event"}
    assert standard["event_tier"].eq("standard_4plus").all()
    assert len(standard) == 5
    assert set(short["event_id"]) == {"short3_event", "short2_event"}
    assert set(short["event_tier"]) == {"short_3", "short_2"}
    assert set(STANDARD_CATALOG_REQUIRED_COLUMNS).issubset(standard.columns)


def test_short_events_never_enter_pilot_planning() -> None:
    catalogs = build_canonical_catalogs(
        make_pool(),
        network_sha256="net",
        config_sha256="cfg",
        source_run_uuid="uuid",
    )
    mixed = pd.concat(
        [
            catalogs["standard_checkpoint_catalog"],
            catalogs["short_event_checkpoint_catalog"],
        ],
        ignore_index=True,
    )
    ledger = build_event_usage_ledger(
        mixed[["event_id", "rainfall_sha256", "event_tier"]],
        scanned_event_ids=set(mixed["event_id"].astype(str)),
    )
    anchors = pd.DataFrame({"event_id": ["std_event"]})

    with pytest.raises(ValueError, match="standard_4plus"):
        build_pilot_planning_bundle(
            mixed,
            ledger,
            peak_anchor_library=anchors,
            gate5r_classification=anchors,
        )


def test_real_standard_catalog_is_182_events_910_rows() -> None:
    path = CANONICAL_DIR / "standard_checkpoint_catalog.csv"
    assert path.exists(), (
        "canonical standard catalog missing; run BuildOpportunityPool "
        f"to regenerate {path}"
    )
    frame = pd.read_csv(path)

    assert len(frame) == 910
    assert frame["event_id"].nunique() == 182
    assert set(STANDARD_CATALOG_REQUIRED_COLUMNS).issubset(frame.columns)
    assert frame["event_tier"].eq("standard_4plus").all()
    roles = frame.groupby("event_id")["checkpoint_role"].value_counts().unstack(
        fill_value=0
    )
    assert roles["responsive"].eq(4).all()
    assert roles["low_opportunity"].eq(1).all()
    assert not frame.duplicated(["event_id", "checkpoint_id"]).any()


def test_real_short_catalog_holds_short_2_and_short_3_only() -> None:
    path = CANONICAL_DIR / "short_event_checkpoint_catalog.csv"
    assert path.exists(), (
        "canonical short catalog missing; run BuildOpportunityPool "
        f"to regenerate {path}"
    )
    frame = pd.read_csv(path)

    assert set(frame["event_tier"]) <= {"short_2", "short_3"}
    assert frame["event_id"].nunique() == 62
