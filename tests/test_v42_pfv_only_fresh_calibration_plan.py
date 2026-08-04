from pathlib import Path

import pandas as pd

from scripts.plan_v42_pfv_only_fresh_calibration import select_plan


def test_fresh_plan_is_untouched_and_forcing_only(tmp_path: Path):
    rows = []
    inventory = []
    for i, depth in enumerate((10, 11, 20, 21, 30, 31), start=1):
        event = f"fresh_{i}"
        forcing = tmp_path / f"{event}.csv"
        pd.DataFrame(
            {"elapsed_min": [0, 5, 10], "intensity_mm_h": [depth, depth, depth], "event_id": [event] * 3}
        ).to_csv(forcing, index=False)
        rows.append({"inventory_event_id": event, "rainfall_sha256": f"sha_{i}", "rainfall_group_key": f"sha_{i}", "formal_f2_role": "unused_untouched"})
        inventory.append({"event_id": event, "rainfall_path": str(forcing), "storm_family_id": f"family_{i}"})

    plan = select_plan(pd.DataFrame(rows), pd.DataFrame(inventory), count=6)

    assert len(plan) == 6
    assert plan["rainfall_sha256"].nunique() == 6
    assert plan["selection_uses_control_outcome"].eq(False).all()
    assert plan["authoritative_swmm_required"].eq(True).all()
    assert set(plan["severity_stratum"]) == {"low", "medium", "high"}
