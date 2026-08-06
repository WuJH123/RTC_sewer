"""Audit PFV-relaxation / TFV-benefit exchange from the canonical experience bank."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sewerrtc.control.pfv_tfv_tradeoff_v42 import (
    TRADEOFF_CONTRACT,
    add_tradeoff_columns,
    contract_scan,
    pareto_exchange_rates,
    select_knee_points,
    state_pareto_frontier,
)


def _parse_floats(text: str) -> tuple[float, ...]:
    values = tuple(float(x.strip()) for x in str(text).split(",") if x.strip())
    if not values:
        raise ValueError("empty numeric grid")
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experience-bank", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--relative-grid", default="0,0.025,0.05,0.075,0.10,0.15")
    parser.add_argument("--absolute-grid", default="0,100,250,500,1000,2000")
    args = parser.parse_args()

    if args.experience_bank.suffix.lower() in {".parquet", ".pq"}:
        bank = pd.read_parquet(args.experience_bank)
    else:
        bank = pd.read_csv(args.experience_bank, low_memory=False)
    work = add_tradeoff_columns(bank)
    frontier = state_pareto_frontier(work)
    exchange = pareto_exchange_rates(frontier)
    knees = select_knee_points(frontier)
    state_grid, aggregate_grid = contract_scan(
        work,
        relative_margins=_parse_floats(args.relative_grid),
        absolute_margins_m3=_parse_floats(args.absolute_grid),
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    work.to_csv(args.output_dir / "PFV_TFV_ALL_AUTHORITATIVE_CANDIDATES.csv", index=False)
    frontier.to_csv(args.output_dir / "PFV_TFV_STATE_PARETO_FRONTIER.csv", index=False)
    exchange.to_csv(args.output_dir / "PFV_TFV_PARETO_EXCHANGE_RATES.csv", index=False)
    knees.to_csv(args.output_dir / "PFV_TFV_PARETO_KNEE_POINTS.csv", index=False)
    state_grid.to_csv(args.output_dir / "PFV_TFV_CONTRACT_STATE_GRID.csv", index=False)
    aggregate_grid.to_csv(args.output_dir / "PFV_TFV_CONTRACT_AGGREGATE_GRID.csv", index=False)

    strict = aggregate_grid[
        np.isclose(aggregate_grid["relative_margin_fraction"], 0.05)
        & np.isclose(aggregate_grid["absolute_margin_m3"], 100.0)
    ]
    best_mean = aggregate_grid.sort_values(
        ["all_state_zero_if_unavailable_mean_pct", "improving_fraction_all_states"],
        ascending=[False, False],
        kind="stable",
    ).head(1)
    summary = {
        "contract": TRADEOFF_CONTRACT,
        "development_only": True,
        "online_PFV_contract_changed": False,
        "new_SWMM_started": False,
        "candidate_rows": int(len(work)),
        "states": int(work["state_key"].astype(str).nunique()),
        "pareto_rows": int(len(frontier)),
        "knee_points": int(len(knees)),
        "strict_5pct_plus_100m3": strict.to_dict("records")[0] if len(strict) else None,
        "diagnostic_best_grid_by_all_state_mean": best_mean.to_dict("records")[0] if len(best_mean) else None,
        "interpretation": (
            "The grid quantifies opportunity made available by PFV relaxation; it does not authorise "
            "changing the online PFV contract.  State-wise Pareto fronts expose efficient risk-benefit "
            "actions and exchange rates quantify added TFV benefit per extra m3 of PFV cost."
        ),
    }
    (args.output_dir / "PFV_TFV_TRADEOFF_SUMMARY.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
