from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from sewerrtc.v4.v42_fast_feasibility import (
    FAST_CONTRACT_ID,
    build_fast_step1_aux_allowlist,
)


def test_fast_aux_allowlist_is_deterministic_and_development_only(tmp_path: Path) -> None:
    rows = []
    for i in range(10):
        role = "target_formal" if i < 2 else "auxiliary_pretrain"
        rows.append(
            {
                "detail_path": f"x{i}.csv",
                "anchor_min": 60.0,
                "split_group_key": f"g{i}",
                "physical_identity_sha256": f"p{i}",
                "step1_domain_role": role,
            }
        )
    manifest = tmp_path / "windows.parquet"
    pd.DataFrame(rows).to_parquet(manifest, index=False)
    out1 = tmp_path / "a.json"
    out2 = tmp_path / "b.json"
    a = build_fast_step1_aux_allowlist(
        manifest_path=manifest, output_path=out1, max_groups=4, seed=42
    )
    b = build_fast_step1_aux_allowlist(
        manifest_path=manifest, output_path=out2, max_groups=4, seed=42
    )
    assert a == b
    assert a["contract_id"] == FAST_CONTRACT_ID
    assert a["development_only"] is True
    assert a["formal_target_upgrade"] is False
    assert a["selected_aux_groups"] == 4
    assert all(g not in {"g0", "g1"} for g in a["groups"])
    assert json.loads(out1.read_text(encoding="utf-8"))["groups"] == a["groups"]
