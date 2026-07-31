"""One-shot recovery: restore the first-round Round B directives file.

Context: BuildPilotFeasibilityMap used to overwrite map/round_b_directives.csv
on every invocation.  After the 653-sample rebuild it wrote the 7-state
residual, and a subsequent PlanPilotFeasibilityMap regenerated only 416 rows,
dropping the 255 executed Round B plan rows.  The directives are a pure
function of the frozen Round-A-only evidence, so this script recomputes them
and restores the frozen file after verifying they cover exactly the states
that actually ran Round B.  The overwritten residual is preserved as
round_b_directives_residual_diagnostic.csv.
"""

import pathlib
import shutil
import sys

import pandas as pd
import yaml

PROJ = pathlib.Path(r"E:\RTC_sewer\Project6")
sys.path.insert(0, str(PROJ))

from sewerrtc.v4.pilot_feasibility_map import (  # noqa: E402
    combine_state_samples,
    plan_feasibility_round_b_directives,
)
from sewerrtc.v4.pipeline_p3 import _boundary_band  # noqa: E402


def main() -> int:
    root = PROJ / "outputs" / "project6_dual_reference_v4" / "final_v4"
    p3 = root / "pilot_feasibility_p3"
    cfg = yaml.safe_load(
        (PROJ / "configs" / "wuhan_project6_v4_final.yaml").read_text(
            encoding="utf-8"
        )
    )
    manifest = pd.read_csv(p3 / "dataset" / "feasibility_sample_manifest.csv")
    round_b = manifest[manifest["sample_id"].str.contains("__round_b__")]
    executed_states = set(
        zip(
            round_b["event_id"].astype(str),
            round_b["checkpoint_id"].astype(str),
        )
    )
    print("manifest roundB rows:", len(round_b))
    print("manifest roundB states:", len(executed_states))
    catalog = pd.read_csv(p3 / "pilot_feasibility_state_catalog.csv")
    v2 = pd.read_csv(
        root / "pilot" / "dataset_v2" / "pilot_v2_sample_manifest.csv"
    )
    feas_round_a = manifest[manifest["sample_id"].str.contains("__round_a__")]
    plan = pd.read_csv(
        p3 / "planning" / "feasibility_candidate_plan.csv"
    )
    plan_round_a = plan[plan["search_round"].astype(str) == "round_a"]
    directives = plan_feasibility_round_b_directives(
        catalog,
        combine_state_samples(v2, feas_round_a),
        plan_round_a,
        scientific_margin=cfg["thresholds"]["scientific_margin"],
        boundary_band=_boundary_band(PROJ),
    )
    directive_states = set(
        zip(
            directives["event_id"].astype(str),
            directives["checkpoint_id"].astype(str),
        )
    )
    print("recomputed directives rows:", len(directives))
    if directive_states != executed_states:
        print("ABORT: directive states do not match executed Round B states")
        print("only in directives:", directive_states - executed_states)
        print("only in manifest:", executed_states - directive_states)
        return 1
    target = p3 / "map" / "round_b_directives.csv"
    residual = p3 / "map" / "round_b_directives_residual_diagnostic.csv"
    if not residual.exists():
        shutil.copy(target, residual)
        print("residual preserved:", residual.name)
    directives.to_csv(target, index=False)
    restored = pd.read_csv(target)
    print("restored rows:", len(restored))
    print(
        "budgets:", sorted(restored["round_b_budget"].astype(int).unique())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
