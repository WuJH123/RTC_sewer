from __future__ import annotations

import pandas as pd

from scripts import audit_v42_experience_bank_contracts as audit


def test_no_control_audit_excludes_causal_prefix(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, "_load_engineering36_ids", lambda root: ["A1"])
    detail = tmp_path / "no_control.csv"
    pd.DataFrame(
        {"elapsed_min": [0.0, 5.0, 10.0], "setting:A1": [0.0, 1.0, 1.0]}
    ).to_csv(detail, index=False)
    candidates = pd.DataFrame(
        {
            "source_detail_path_no_control": [str(detail)],
            "checkpoint_min": [5.0],
        }
    )

    report = audit._audit_no_control(tmp_path, candidates, tmp_path / "audit")

    assert report["status"] == "pass"
    check = report["rows"][0]["checks"][0]
    assert check["prefix_rows_excluded"] == 1
    assert check["post_action_rows"] == 2
