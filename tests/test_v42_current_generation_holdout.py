from __future__ import annotations

import pandas as pd

from sewerrtc.v4.formal_f2 import build_event_ledger, split_overlap_matrix


def test_historical_status_is_metadata_not_current_split_gate() -> None:
    source = pd.DataFrame(
        [
            {
                "source_id": "history",
                "rainfall_group_key": f"r{i}",
                "formal_step2_allowed": True,
                "step2_accepted_from_manifest": True,
                "raw_readmission_required": False,
                "historically_revealed": True,
            }
            for i in range(70)
        ]
    )
    # Inventory deliberately reuses the same historical groups plus independent
    # groups. The current generation may train or hold out a historically seen
    # group; only current train/evaluation overlap is forbidden.
    inventory = pd.DataFrame(
        [
            {"event_id": f"e{i}", "rainfall_sha256": f"r{i}"}
            for i in range(70)
        ]
        + [
            {"event_id": f"u{i}", "rainfall_sha256": f"u{i}"}
            for i in range(80)
        ]
    )
    ledger = build_event_ledger(source, inventory=inventory, minimum_train_groups=65, seed=42)
    assert all(value == 0 for value in split_overlap_matrix(ledger).values())
    assert ledger.loc[ledger.formal_f2_role.eq("train"), "rainfall_group_key"].nunique() >= 65
    assert not bool(ledger["historical_status_is_split_gate"].astype(bool).any())


def test_current_generation_holdouts_are_excluded_from_training() -> None:
    source = pd.DataFrame(
        [
            {
                "source_id": "trainable",
                "rainfall_group_key": f"r{i}",
                "formal_step2_allowed": True,
                "step2_accepted_from_manifest": True,
                "raw_readmission_required": False,
                "historically_revealed": bool(i % 2),
            }
            for i in range(130)
        ]
    )
    inventory = pd.DataFrame(
        [{"event_id": f"e{i}", "rainfall_sha256": f"r{i}"} for i in range(130)]
    )
    ledger = build_event_ledger(source, inventory=inventory, minimum_train_groups=65, seed=7)
    train = set(ledger.loc[ledger.formal_f2_role.eq("train"), "rainfall_group_key"].astype(str))
    evaluation = set(
        ledger.loc[
            ledger.formal_f2_role.isin(
                ["calibration", "locked_validation", "challenge", "formal_blind"]
            ),
            "rainfall_group_key",
        ].astype(str)
    )
    assert train
    assert evaluation
    assert not (train & evaluation)
