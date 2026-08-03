from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from scripts import build_v42_qualification_first_pass as builder
from scripts import materialize_v42_qualification_gat_history as gat_bridge
from scripts import run_v42_qualification_first_pass as runner


ROOT = Path(__file__).resolve().parents[1]


def _action(value: float) -> str:
    array = np.zeros((12, 36), dtype=float)
    array[:3, 0] = value
    return json.dumps(array.tolist())


def test_qualification_profile_never_authorizes_formal() -> None:
    payload = json.loads((ROOT / "configs/v42_qualification_first_pass.json").read_text())
    assert payload["qualification_only"] is True
    assert payload["development_only"] is True
    assert payload["formal_mainline_authorized"] is False
    assert payload["formal_outputs_must_not_be_overwritten"] is True
    assert payload["closed_loop_qualification"]["formal_untouched_events_must_not_be_consumed"] is True


def test_qualification_runner_registers_all_28_units() -> None:
    assert len(runner.STAGES) == 28
    assert runner.STAGES[0].startswith("01_")
    assert runner.STAGES[-1].startswith("28_")


def test_candidate_selection_uses_distinct_actual_h3_schedules() -> None:
    frame = pd.DataFrame(
        {
            "state_key": ["s1"] * 5,
            "checkpoint_min": [120.0] * 5,
            "action_candidate_readback": [
                _action(0.1),
                _action(0.1),
                _action(0.2),
                _action(0.3),
                _action(0.4),
            ],
        }
    )
    selected = builder._choose_step2_state(frame, candidates=3, seed=42)
    assert selected is not None
    assert len(selected) == 3
    assert selected["qualification_candidate_action_sha256"].nunique() == 3


def test_step2_selection_rejects_states_before_causal_warmup() -> None:
    frame = pd.DataFrame(
        {
            "state_key": ["early"] * 3 + ["warm"] * 3,
            "checkpoint_min": [90.0] * 3 + [120.0] * 3,
            "action_candidate_readback": [
                _action(0.1),
                _action(0.2),
                _action(0.3),
                _action(0.4),
                _action(0.5),
                _action(0.6),
            ],
        }
    )
    selected = builder._choose_step2_state(
        frame, candidates=3, seed=42, min_checkpoint_min=120.0
    )
    assert selected is not None
    assert selected["state_key"].unique().tolist() == ["warm"]
    assert (selected["checkpoint_min"] >= 120.0).all()


def test_step2_groups_are_limited_to_step1_history_groups() -> None:
    eligible = {"g1": pd.DataFrame(), "g2": pd.DataFrame(), "g3": pd.DataFrame()}
    selected = builder._select_step2_groups(
        eligible,
        ranked_groups=["g1", "g2", "g3"],
        step1_history_groups={"g1", "g3"},
        required_groups=2,
    )
    assert selected == ["g1", "g3"]


def test_step1_selection_prefers_step2_eligible_groups() -> None:
    ranked = builder._rank_preferred_groups(
        ["g1", "g2", "g3", "g4"],
        preferred={"g3", "g4"},
        seed=42,
        salt="step1-train",
    )
    assert set(ranked[:2]) == {"g3", "g4"}


def test_prepare_reuse_requires_matching_config_hash(tmp_path: Path) -> None:
    config = tmp_path / "config.json"
    config.write_text('{"step2_min_checkpoint_min": 120}', encoding="utf-8")
    audit = tmp_path / "audit.json"
    audit.write_text(json.dumps({"status": "pass", "config_sha256": "stale"}), encoding="utf-8")
    step1 = tmp_path / "step1.parquet"
    step2 = tmp_path / "step2.parquet"
    step1.write_bytes(b"step1")
    step2.write_bytes(b"step2")
    assert not runner._prepare_artifacts_reusable(audit, step1, step2, config)


def test_pre_action_signature_ignores_checkpoint_transition(monkeypatch) -> None:
    graph = SimpleNamespace(node_ids=["n1"], facility_ids=["f1"])
    first = {
        "depth_history": np.arange(13, dtype=np.float32).reshape(13, 1),
        "rainfall": np.arange(13, dtype=np.float32),
        "actions": np.zeros((13, 1), dtype=np.float32),
    }
    second = {key: value.copy() for key, value in first.items()}
    second["actions"][-1, 0] = 1.0
    current = {"value": first}

    def fake_extract(*_args, **_kwargs):
        return current["value"]

    monkeypatch.setattr(gat_bridge, "_detail_extract_window", fake_extract)
    signature_a = gat_bridge._pre_action_signature(pd.DataFrame(), 120.0, graph)
    current["value"] = second
    signature_b = gat_bridge._pre_action_signature(pd.DataFrame(), 120.0, graph)
    assert signature_a == signature_b

    third = {key: value.copy() for key, value in first.items()}
    third["actions"][-2, 0] = 1.0
    current["value"] = third
    signature_c = gat_bridge._pre_action_signature(pd.DataFrame(), 120.0, graph)
    assert signature_c != signature_a


def test_step1_spread_caps_each_physical_run_and_group() -> None:
    rows = []
    for run in ("r1", "r2"):
        for anchor in range(0, 100, 5):
            rows.append(
                {
                    "physical_identity_sha256": run,
                    "split_group_key": "g1",
                    "anchor_min": float(anchor),
                    "detail_path": f"{run}.csv",
                }
            )
    frame = pd.DataFrame(rows)
    capped = builder._cap_step1(frame, windows_per_run=2, windows_per_group=3)
    assert len(capped) <= 3
    assert capped.groupby("physical_identity_sha256").size().max() <= 2
