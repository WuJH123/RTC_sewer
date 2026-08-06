from __future__ import annotations

import json

import numpy as np
import pandas as pd

from scripts.build_v42_authoritative_experience_bank import canonical_action_identity
from sewerrtc.control.authoritative_control_metrics_v42 import action_sha256


def test_legacy_action_sha_mismatch_keeps_canonical_identity() -> None:
    action = np.ones((12, 2), dtype=np.float32)
    row = pd.Series(
        {
            "action_candidate_readback": json.dumps(action.tolist()),
            "candidate_action_sha256": "legacy-format-sha",
        }
    )

    executed, canonical, legacy, matches = canonical_action_identity(row)

    assert np.array_equal(executed, action)
    assert canonical == action_sha256(action)
    assert legacy == "legacy-format-sha"
    assert matches is False
