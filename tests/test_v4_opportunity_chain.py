from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sewerrtc.v4.opportunity import (
    build_opportunity_pool,
    plan_opportunity_scans,
)
from sewerrtc.v4.peak_boundary import (
    PEAK_FAMILIES,
    build_peak_boundary_dataset,
    build_peak_candidate_catalog,
    build_peak_boundary_plan,
)
from sewerrtc.v4.pipeline import ALL_STAGES


def _rainfall(path: Path, peak_at: int) -> None:
    elapsed = np.arange(0, 181, 5)
    intensity = np.maximum(0.0, 30.0 - np.abs(elapsed - peak_at))
    pd.DataFrame(
        {"elapsed_min": elapsed, "intensity_mm_h": intensity}
    ).to_csv(path, index=False)


def test_opportunity_scan_plan_is_generated_from_eligible_inventory(
    tmp_path: Path,
) -> None:
    rain_a = tmp_path / "a.csv"
    rain_b = tmp_path / "b.csv"
    _rainfall(rain_a, 60)
    _rainfall(rain_b, 120)
    inventory = pd.DataFrame(
        {
            "event_id": ["a", "b", "excluded"],
            "rainfall_path": [rain_a, rain_b, rain_a],
            "rainfall_sha256": ["a" * 64, "b" * 64, "c" * 64],
            "eligible": [True, True, False],
        }
    )
    config = {
        "project": {
            "network": "data/network.inp",
            "facility_semantics": "data/semantics.csv",
            "priority_nodes": "data/priority.txt",
        },
        "runtime": {
            "record_step_min": 5,
            "control_step_min": 10,
            "use_hotstart": False,
        },
        "opportunity": {"post_rain_buffer_min": 180},
    }

    plan = plan_opportunity_scans(inventory, config, tmp_path)

    assert plan["event_id"].tolist() == ["a", "b"]
    assert set(plan["runner_function"]) == {"run_swmm_fixed_action"}
    assert plan["case_id"].is_unique
    for encoded in plan["runner_kwargs"]:
        kwargs = json.loads(encoded)
        assert Path(kwargs["rainfall_path"]).is_absolute()
        assert Path(kwargs["inp_path"]).is_absolute()
        assert kwargs["hotstart_dir"] is None
        assert kwargs["control_step_sec"] == 300
        assert kwargs["decision_interval_sec"] == 600


def test_opportunity_pool_aggregates_scored_checkpoints_and_peak_inputs(
    tmp_path: Path,
) -> None:
    detail = tmp_path / "detail.csv"
    elapsed = np.arange(0, 301, 5, dtype=float)
    rain = np.maximum(0.0, 40.0 - np.abs(elapsed - 120.0))
    pd.DataFrame(
        {
            "elapsed_min": elapsed,
            "rainfall_mm_h": rain,
            "flow:P1": 0.2 + 0.01 * elapsed,
            "flow:O1": 0.1 + 0.005 * elapsed,
            "h:N1": 1.0 + 0.005 * elapsed,
            "h:N2": 0.2 + 0.001 * elapsed,
            "h:N3": 0.8 + 0.003 * elapsed,
            "storage_volume:S1": 100.0 + elapsed,
            "flood:PR1": np.where(elapsed >= 110, 0.01, 0.0),
            "system_inflow_m3s": 1.0 + rain / 20.0,
            "total_outfall_flow_m3s": 0.8 + rain / 25.0,
            "excess_fullness_p95": np.clip(rain / 50.0, 0.0, 1.0),
            "a:P1": np.where(elapsed < 100, 0.0, 1.0),
            "a:O1": 0.5,
        }
    ).to_csv(detail, index=False)
    run_manifest = pd.DataFrame(
        {
            "case_id": ["opportunity__event-a"],
            "event_id": ["event-a"],
            "status": ["pass"],
            "detail_path": [str(detail)],
        }
    )
    inventory = pd.DataFrame(
        {
            "event_id": ["event-a"],
            "storm_family_id": ["double_peak"],
            "rainfall_sha256": ["a" * 64],
        }
    )
    semantics = pd.DataFrame(
        {
            "facility_id": ["P1", "O1"],
            "actuator_type": ["pump", "orifice"],
            "binary_or_continuous": ["binary", "continuous"],
            "from_node": ["N1", "N3"],
            "to_node": ["N2", "N2"],
            "storage_role": ["none", "storage_outlet"],
            "lower_bound": [0.0, 0.0],
            "upper_bound": [1.0, 1.0],
        }
    )

    pool, diagnostics = build_opportunity_pool(
        run_manifest,
        inventory,
        facility_ids=["P1", "O1"],
        facility_semantics=semantics,
        responsive_threshold=0.25,
        checkpoint_spacing_min=30,
    )

    assert {
        "event_id",
        "checkpoint_id",
        "checkpoint_min",
        "opportunity_class",
        "phase",
        "rainfall_family",
        "risk_level",
        "source_detail",
        "anchor_action_json",
        "active_facility_ids_json",
    }.issubset(pool.columns)
    assert set(pool["opportunity_class"]).issubset(
        {"responsive", "low_opportunity"}
    )
    assert diagnostics["component"].nunique() > 3

    catalog = build_peak_candidate_catalog(
        pool,
        facility_ids=["P1", "O1"],
        facility_semantics=semantics,
        target_count=30,
    )
    assert len(catalog) == 30
    assert catalog["requested_schedule_sha256"].is_unique
    assert catalog["family"].nunique() >= 2
    assert set(catalog["family"]).issubset(PEAK_FAMILIES)
    assert {"runner_function", "runner_kwargs"}.issubset(catalog.columns)
    # Engineering constraint verification columns must be emitted and all
    # true; failing candidates are dropped fail-closed, never repaired.
    for column in (
        "binary_semantics_ok",
        "rate_limit_ok",
        "dwell_ok",
        "interlock_ok",
    ):
        assert column in catalog.columns
        assert catalog[column].astype(bool).all()
    branch_plan = build_peak_boundary_plan(catalog, minimum=30, maximum=60)
    assert len(branch_plan) == 120
    assert set(branch_plan["branch"]) == {
        "candidate",
        "no_control",
        "dynamic_internal_rules",
        "hold_previous",
    }
    # Peak plans carry a constant development split so split isolation is
    # provable from the plan alone.
    assert set(branch_plan["split"]) == {"development"}


def test_peak_dataset_requires_four_same_state_branches_and_keeps_failures(
    tmp_path: Path,
) -> None:
    elapsed = np.arange(0, 181, 5, dtype=float)
    manifests = []
    for branch, post_rate in {
        "candidate": 0.10,
        "no_control": 0.20,
        "dynamic_internal_rules": 0.05,
        "hold_previous": 0.10,
    }.items():
        post = elapsed > 60
        detail = pd.DataFrame(
            {
                "elapsed_min": elapsed,
                "h:N1": np.where(post, 1.0 + post_rate, 1.0),
                "flow:F1": np.where(post, post_rate, 0.0),
                "flood:PR1": np.where(post, post_rate, 0.0),
                "flood:N2": np.where(post, post_rate, 0.0),
                "a:F1": np.where(post, 1.0, 0.0),
                "actual_setting:F1": np.where(post, 1.0, 0.0),
                "readback_setting:F1": np.where(post, 1.0, 0.0),
            }
        )
        path = tmp_path / f"{branch}.csv"
        detail.to_csv(path, index=False)
        manifests.append(
            {
                "case_id": f"sample-1__{branch}",
                "sample_id": "sample-1",
                "event_id": "event-a",
                "checkpoint_id": "event-a__60",
                "checkpoint_min": 60.0,
                "branch": branch,
                "family": "synchronized_pump_starts",
                "status": "pass",
                "detail_path": str(path),
                "requested_schedule_sha256": "r" * 64,
                "projected_schedule_json": json.dumps([[1.0]] * 12),
                "anchor_schedule_json": json.dumps([[0.0]] * 12),
            }
        )

    samples, rejected = build_peak_boundary_dataset(
        pd.DataFrame(manifests),
        priority_nodes=["PR1"],
        facility_ids=["F1"],
        scientific_margin={"pfv_m3": 0.0, "tfv_m3": 0.0, "peak_m3s": 0.0},
        dead_zone={"pfv_m3": 1.0, "tfv_m3": 1.0, "peak_m3s": 0.001},
    )

    assert rejected.empty
    assert len(samples) == 1
    sample = samples.iloc[0]
    assert sample["state_hash_match"]
    assert sample["readback_ok"]
    assert sample["pfv_safe"]
    assert not sample["peak_noninferior"]
    assert sample["hard_negative_type"] == "Peak_hard_negative"


def test_opportunity_and_peak_catalog_stages_are_explicitly_registered() -> None:
    ordered = list(ALL_STAGES)
    assert ordered.index("BuildEventInventory") < ordered.index(
        "PlanOpportunityPool"
    )
    assert ordered.index("PlanOpportunityPool") < ordered.index(
        "ScanOpportunityPool"
    )
    assert ordered.index("ScanOpportunityPool") < ordered.index(
        "BuildOpportunityPool"
    )
    assert ordered.index("BuildOpportunityPool") < ordered.index(
        "AuditOpportunityCoverage"
    )
    assert ordered.index("AuditOpportunityCoverage") < ordered.index(
        "BuildPeakCandidateCatalog"
    )
    assert ordered.index("BuildPeakCandidateCatalog") < ordered.index(
        "PlanPeakBoundary"
    )
    assert ordered.index("RunPeakBoundary") < ordered.index(
        "BuildPeakBoundaryDataset"
    )
    assert ordered.index("BuildPeakBoundaryDataset") < ordered.index(
        "AuditPeakBoundary"
    )
