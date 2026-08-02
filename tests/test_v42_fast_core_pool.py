import json

import numpy as np
import pandas as pd

from scripts.build_v42_fast_core_pool import (
    _h3_schedule_sha,
    _resolved_forcing,
    _state_key,
)


def test_resolved_forcing_prefers_alignment_copy():
    row = pd.Series(
        {
            "same_forcing_pass_x": False,
            "same_forcing_pass_y": True,
            "same_forcing_pass": False,
        }
    )
    assert _resolved_forcing(row) is True


def test_state_key_does_not_depend_on_case_id():
    base = {
        "rainfall_sha256": "rain-a",
        "checkpoint_min": 180.0,
        "network_sha256": "net",
        "event_id": "event-1",
    }
    a = pd.Series({**base, "case_id": "candidate-a"})
    b = pd.Series({**base, "case_id": "candidate-b"})
    assert _state_key(a) == _state_key(b)


def test_h3_schedule_hash_ignores_uncontrollable_tail():
    a = np.zeros((12, 36), dtype=float)
    b = a.copy()
    b[3:, 0] = 1.0
    ra = pd.Series({"projected_schedule_json": json.dumps(a.tolist())})
    rb = pd.Series({"projected_schedule_json": json.dumps(b.tolist())})
    assert _h3_schedule_sha(ra) == _h3_schedule_sha(rb)


def test_h3_schedule_hash_detects_control_prefix_change():
    a = np.zeros((12, 36), dtype=float)
    b = a.copy()
    b[0, 0] = 1.0
    ra = pd.Series({"projected_schedule_json": json.dumps(a.tolist())})
    rb = pd.Series({"projected_schedule_json": json.dumps(b.tolist())})
    assert _h3_schedule_sha(ra) != _h3_schedule_sha(rb)
