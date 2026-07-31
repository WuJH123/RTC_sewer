from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sewerrtc.control.v4_opportunity import scan_control_opportunities


ONLINE_COMPONENTS = (
    "history_flow",
    "history_head",
    "history_depth",
    "storage_headroom",
    "priority_sentinel_risk",
    "downstream_capacity",
    "native_rule_state",
    "forecast_rain_h120",
    "remaining_rainfall",
    "conduit_fullness",
    "inflow_outflow_imbalance",
    "current_action",
    "di_hold_difference",
)


def _rainfall_duration_min(path: str | Path) -> int:
    rainfall = pd.read_csv(path)
    for column in ("elapsed_min", "time_min", "minute"):
        if column in rainfall:
            values = pd.to_numeric(rainfall[column], errors="coerce")
            if values.notna().any():
                return max(10, int(np.ceil(float(values.max()))))
    return max(10, int(len(rainfall) * 5))


def plan_opportunity_scans(
    inventory: pd.DataFrame,
    config: dict,
    project_root: str | Path,
) -> pd.DataFrame:
    """Materialize one native-rule baseline request per eligible event."""
    required = {"event_id", "rainfall_path", "rainfall_sha256", "eligible"}
    missing = required - set(inventory)
    if missing:
        raise ValueError(f"event inventory missing: {sorted(missing)}")
    root = Path(project_root)
    project = config.get("project", {})
    runtime = config.get("runtime", {})
    opportunity = config.get("opportunity", {})
    if bool(runtime.get("use_hotstart", False)):
        raise ValueError("Final V4 Opportunity planning prohibits hot-start")

    def absolute(raw: str) -> str:
        path = Path(str(raw))
        return str((path if path.is_absolute() else root / path).resolve())

    selected = inventory[inventory["eligible"].astype(bool)].copy()
    selected = selected.drop_duplicates("rainfall_sha256").sort_values(
        ["event_id", "rainfall_sha256"]
    )
    limit = int(opportunity.get("scan_event_limit", 0))
    if limit > 0:
        selected = selected.head(limit)
    rows: list[dict] = []
    for _, event in selected.iterrows():
        rainfall_path = Path(str(event["rainfall_path"]))
        if not rainfall_path.is_absolute():
            rainfall_path = root / rainfall_path
        if not rainfall_path.exists():
            continue
        rain_duration = _rainfall_duration_min(rainfall_path)
        simulation_duration = rain_duration + int(
            opportunity.get("post_rain_buffer_min", 180)
        )
        kwargs = {
            "inp_path": absolute(project["network"]),
            "rainfall_path": str(rainfall_path.resolve()),
            "actuators_csv": absolute(project["facility_semantics"]),
            "priority_nodes_file": absolute(project["priority_nodes"]),
            "event_id": str(event["event_id"]),
            "duration_min": rain_duration,
            "simulation_duration_min": simulation_duration,
            "prefix_schedule": None,
            "override_start_min": float(simulation_duration + 1),
            "post_action": [1.0] * 36,
            "control_step_sec": int(runtime.get("record_step_min", 5)) * 60,
            "decision_interval_sec": int(
                runtime.get("control_step_min", 10)
            )
            * 60,
            "policy_id": "opportunity_dynamic_internal",
            "post_control_mode": "native_rules",
            "cleanup_swmm_artifacts": True,
            "hydraulic_summary_start_min": 0.0,
            "hotstart_dir": None,
        }
        rows.append(
            {
                "case_id": f"opportunity__{event['event_id']}",
                "event_id": str(event["event_id"]),
                "rainfall_sha256": str(event["rainfall_sha256"]),
                "runner_function": "run_swmm_fixed_action",
                "runner_kwargs": json.dumps(kwargs, separators=(",", ":")),
            }
        )
    if not rows:
        raise ValueError("no executable eligible Opportunity events")
    return pd.DataFrame(rows)


def _phase_at_checkpoint(
    elapsed: np.ndarray, rainfall: np.ndarray, checkpoint: float
) -> str:
    positive = rainfall > 1e-9
    if not positive.any():
        return "late_rain"
    peak_time = float(elapsed[int(np.nanargmax(rainfall))])
    rain_end = float(elapsed[np.flatnonzero(positive)[-1]])
    if checkpoint < 0.6 * peak_time:
        return "rising"
    if checkpoint < peak_time:
        return "pre_peak"
    if checkpoint <= min(rain_end, peak_time + 30.0):
        return "peak"
    return "late_rain"


def _spaced_top(
    frame: pd.DataFrame, count: int, spacing_min: float
) -> pd.DataFrame:
    chosen: list[int] = []
    # First preserve phase diversity, then fill by score.
    ordered_parts = []
    for phase in ("rising", "pre_peak", "peak", "late_rain"):
        group = frame[frame["phase"] == phase].sort_values(
            "opportunity_score", ascending=False
        )
        if len(group):
            ordered_parts.append(group.head(1))
    ordered_parts.append(frame.sort_values("opportunity_score", ascending=False))
    ordered = pd.concat(ordered_parts).drop_duplicates().copy()
    for index, row in ordered.iterrows():
        checkpoint = float(row["checkpoint_min"])
        if all(
            abs(checkpoint - float(frame.loc[item, "checkpoint_min"]))
            >= float(spacing_min)
            for item in chosen
        ):
            chosen.append(index)
        if len(chosen) == int(count):
            break
    return frame.loc[chosen].copy()


def build_opportunity_pool(
    run_manifest: pd.DataFrame,
    inventory: pd.DataFrame,
    *,
    facility_ids: list[str],
    facility_semantics: pd.DataFrame,
    responsive_threshold: float = 0.25,
    checkpoint_spacing_min: float = 30.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert successful native baselines into scored checkpoint evidence."""
    required = {"event_id", "status", "detail_path"}
    missing = required - set(run_manifest)
    if missing:
        raise ValueError(f"Opportunity run manifest missing: {sorted(missing)}")
    semantics = facility_semantics.set_index("facility_id", drop=False)
    facility_nodes = {
        facility_id: (
            str(semantics.loc[facility_id].get("from_node", "")),
            str(semantics.loc[facility_id].get("to_node", "")),
        )
        for facility_id in facility_ids
        if facility_id in semantics.index
    }
    inventory_by_event = inventory.drop_duplicates("event_id").set_index(
        "event_id"
    )
    pool_rows: list[dict] = []
    diagnostic_frames: list[pd.DataFrame] = []
    for _, run in run_manifest[
        run_manifest["status"].astype(str).eq("pass")
    ].iterrows():
        detail_path = Path(str(run["detail_path"]))
        if not detail_path.exists():
            continue
        detail = pd.read_csv(detail_path)
        scored = scan_control_opportunities(
            detail,
            facility_ids,
            facility_nodes=facility_nodes,
            responsive_threshold=float(responsive_threshold),
            weak_threshold=min(0.05, float(responsive_threshold)),
        )
        max_elapsed = float(scored["elapsed_min"].max())
        scored = scored[
            (scored["elapsed_min"] >= 60.0)
            & (scored["elapsed_min"] <= max_elapsed - 120.0)
            & np.isclose(np.mod(scored["elapsed_min"], 10.0), 0.0)
        ].copy()
        if scored.empty:
            continue
        elapsed = pd.to_numeric(detail["elapsed_min"], errors="coerce").to_numpy(
            float
        )
        rainfall = pd.to_numeric(
            detail.get("rainfall_mm_h", 0.0), errors="coerce"
        )
        if not isinstance(rainfall, pd.Series):
            rainfall = pd.Series(np.zeros(len(detail)))
        rainfall_values = rainfall.fillna(0.0).to_numpy(float)
        scored["phase"] = [
            _phase_at_checkpoint(elapsed, rainfall_values, float(value))
            for value in scored["elapsed_min"]
        ]
        scored["checkpoint_min"] = scored["elapsed_min"].astype(float)
        raw_responsive = scored[
            scored["opportunity_score"] >= float(responsive_threshold)
        ].copy()
        selected = _spaced_top(
            raw_responsive, 4, float(checkpoint_spacing_min)
        )
        selected["opportunity_class"] = "responsive"
        remaining = scored.drop(index=selected.index, errors="ignore")
        low = remaining.nsmallest(1, "opportunity_score").copy()
        low["opportunity_class"] = "low_opportunity"
        selected = pd.concat([selected, low], ignore_index=True)

        event_id = str(run["event_id"])
        family = (
            str(inventory_by_event.loc[event_id].get("storm_family_id", "unknown"))
            if event_id in inventory_by_event.index
            else "unknown"
        )
        rainfall_sha = (
            str(inventory_by_event.loc[event_id].get("rainfall_sha256", ""))
            if event_id in inventory_by_event.index
            else ""
        )
        runner_kwargs = str(run.get("runner_kwargs", "{}"))
        for _, checkpoint in selected.iterrows():
            checkpoint_min = float(checkpoint["elapsed_min"])
            position = int(np.argmin(np.abs(elapsed - checkpoint_min)))
            detail_row = detail.iloc[position]
            anchor = [
                float(detail_row.get(f"a:{facility_id}", 1.0))
                for facility_id in facility_ids
            ]
            active = sorted(
                facility_ids,
                key=lambda item: abs(
                    float(detail_row.get(f"flow:{item}", 0.0))
                ),
                reverse=True,
            )
            active = [
                item
                for item in active
                if abs(float(detail_row.get(f"flow:{item}", 0.0))) > 1e-8
            ]
            record = checkpoint.to_dict()
            record.update(
                {
                    "event_id": event_id,
                    "rainfall_sha256": rainfall_sha,
                    "checkpoint_min": checkpoint_min,
                    "checkpoint_id": f"{event_id}__{checkpoint_min:.0f}",
                    "rainfall_family": family,
                    "source_detail": str(detail_path.resolve()),
                    "source_runner_kwargs": runner_kwargs,
                    "anchor_action_json": json.dumps(anchor),
                    "active_facility_ids_json": json.dumps(active),
                }
            )
            pool_rows.append(record)
        normalized = pd.DataFrame(
            {
                column: robust_normalize(scored[column])
                for column in (
                    "opportunity_score",
                    "active_flow_signal",
                    "flood_signal",
                    "storage_signal",
                    "facility_head_difference_signal",
                    "downstream_capacity_signal",
                    "inflow_outflow_imbalance_signal",
                    "native_switch_signal",
                    "rainfall_signal",
                )
                if column in scored
            }
        )
        diagnostics = component_diagnostics(
            normalized, list(normalized.columns)
        )
        diagnostics["event_id"] = event_id
        diagnostic_frames.append(diagnostics)
    pool = pd.DataFrame(pool_rows)
    if pool.empty:
        raise ValueError("no valid Opportunity checkpoint evidence")
    risk_source = pd.to_numeric(
        pool["opportunity_score"], errors="coerce"
    ).rank(method="first", pct=True)
    pool["risk_level"] = np.select(
        [risk_source <= 1 / 3, risk_source <= 2 / 3],
        ["low", "medium"],
        default="high",
    )
    tiers = classify_event_tiers(pool)
    pool["event_tier"] = pool["event_id"].map(
        tiers.set_index("event_id")["event_tier"]
    )
    diagnostics = (
        pd.concat(diagnostic_frames, ignore_index=True)
        if diagnostic_frames
        else pd.DataFrame()
    )
    return pool.reset_index(drop=True), diagnostics


def robust_normalize(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    median = float(numeric.median())
    q1, q3 = numeric.quantile([0.25, 0.75])
    scale = max(float(q3 - q1), 1e-12)
    z = (numeric - median) / scale
    return pd.Series(1.0 / (1.0 + np.exp(-z)), index=values.index).fillna(0.0)


def component_diagnostics(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    for column in columns:
        values = pd.to_numeric(frame.get(column), errors="coerce")
        rows.append(
            {
                "component": column,
                "missing_fraction": float(values.isna().mean()),
                "saturation_fraction": float(
                    ((values <= 0.0) | (values >= 1.0)).mean()
                ),
                "minimum": float(values.min()) if values.notna().any() else np.nan,
                "maximum": float(values.max()) if values.notna().any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


REQUIRED_LOW_COUNT = 1

EVENT_TIERS = ("standard_4plus", "short_3", "short_2", "ineligible")


def classify_event_tiers(
    frame: pd.DataFrame, required_low_count: int = REQUIRED_LOW_COUNT
) -> pd.DataFrame:
    """Per-event duration-aware feasibility targets and tier labels.

    ``planned_checkpoint_count`` is the materialized per-event checkpoint
    total (planner emits 3-5 including one reserved low-opportunity control),
    so ``max_feasible_responsive = planned - required_low_count`` and the
    duration-aware target is ``min(4, max_feasible_responsive)``.
    """
    grouped = frame.groupby("event_id")
    tiers = pd.DataFrame(
        {
            "planned_checkpoint_count": grouped.size(),
            "responsive_count": grouped["opportunity_class"].apply(
                lambda values: int(values.eq("responsive").sum())
            ),
            "low_opportunity_count": grouped["opportunity_class"].apply(
                lambda values: int(values.eq("low_opportunity").sum())
            ),
        }
    ).reset_index()
    tiers["max_feasible_responsive"] = (
        tiers["planned_checkpoint_count"] - int(required_low_count)
    ).clip(lower=0)
    tiers["required_responsive"] = tiers["max_feasible_responsive"].clip(
        upper=4
    )
    tiers["meets_duration_aware_target"] = (
        tiers["responsive_count"] >= tiers["required_responsive"]
    )
    conditions = [
        (tiers["responsive_count"] >= 4)
        & (tiers["low_opportunity_count"] >= 1),
        (tiers["responsive_count"] == 3)
        & (tiers["low_opportunity_count"] >= 1),
        (tiers["responsive_count"] == 2)
        & (tiers["low_opportunity_count"] >= 1),
    ]
    tiers["event_tier"] = np.select(
        conditions,
        ["standard_4plus", "short_3", "short_2"],
        default="ineligible",
    )
    return tiers


def audit_opportunity_coverage(
    frame: pd.DataFrame, config: dict | None = None
) -> dict:
    required = {
        "event_id",
        "checkpoint_min",
        "opportunity_class",
        "phase",
        "rainfall_family",
        "risk_level",
    }
    missing = required - set(frame)
    if missing:
        return {"status": "blocked", "missing_columns": sorted(missing)}
    opportunity_cfg = (config or {}).get("opportunity", {})
    pilot_events = int(opportunity_cfg.get("pilot_events", 8))
    train_cv_events = int(
        opportunity_cfg.get("train_calibration_validation_events", 64)
    )
    reserve_events = int(opportunity_cfg.get("reserve_events", 16))
    min_standard_eligible = int(
        opportunity_cfg.get(
            "min_standard_eligible_events",
            pilot_events + train_cv_events + reserve_events,
        )
    )
    responsive = frame[frame["opportunity_class"] == "responsive"]
    low = frame[frame["opportunity_class"] == "low_opportunity"]
    by_event = responsive.groupby("event_id")
    spacing_ok = all(
        np.diff(sorted(group["checkpoint_min"].astype(float))).min() >= 30.0
        for _, group in by_event
        if len(group) > 1
    )
    tiers = classify_event_tiers(frame)
    standard_eligible_count = int(
        tiers["event_tier"].eq("standard_4plus").sum()
    )
    tier_counts = {
        tier: int(tiers["event_tier"].eq(tier).sum()) for tier in EVENT_TIERS
    }
    known_classes = set(frame["opportunity_class"]) <= {
        "responsive",
        "low_opportunity",
    }
    checks = {
        "development_events_at_least_8": responsive["event_id"].nunique() >= 8,
        "responsive_at_least_32": len(responsive) >= 32,
        "all_events_meet_duration_aware_target": tiers[
            "meets_duration_aware_target"
        ].all(),
        "standard_eligible_at_least_required": (
            standard_eligible_count >= min_standard_eligible
        ),
        "one_low_control_per_event": tiers["low_opportunity_count"]
        .ge(1)
        .all()
        and set(responsive["event_id"]).issubset(set(low["event_id"])),
        "min_two_responsive_per_event": tiers["responsive_count"].ge(2).all(),
        "checkpoint_spacing_30min": spacing_ok,
        "phases_covered": {
            "rising",
            "pre_peak",
            "peak",
            "late_rain",
        }.issubset(set(responsive["phase"])),
        "rainfall_families_at_least_3": frame["rainfall_family"].nunique() >= 3,
        "risk_levels_at_least_3": frame["risk_level"].nunique() >= 3,
        "accounting_closure": known_classes
        and len(frame) == len(responsive) + len(low)
        and int(tiers["planned_checkpoint_count"].sum()) == len(frame),
    }
    checks = {key: bool(value) for key, value in checks.items()}
    events_below_target = sorted(
        tiers.loc[~tiers["meets_duration_aware_target"], "event_id"]
        .astype(str)
        .tolist()
    )
    return {
        "status": "pass" if all(checks.values()) else "scientific_fail",
        "checks": checks,
        "responsive_checkpoints": int(len(responsive)),
        "events": int(responsive["event_id"].nunique()),
        "event_tiers": tier_counts,
        "standard_eligible_event_count": standard_eligible_count,
        "downstream_requirements": {
            "pilot_events": pilot_events,
            "train_calibration_validation_events": train_cv_events,
            "reserve_events": reserve_events,
            "min_standard_eligible_events": min_standard_eligible,
        },
        "events_below_duration_aware_target": events_below_target,
        "legacy_four_responsive_all_events": bool(by_event.size().ge(4).all()),
        "legacy_check_deprecated": True,
        "deprecated_reason": "planner_auditor_contract_conflict",
    }


CANONICAL_CATALOG_FILES = (
    "opportunity_pool.csv",
    "event_tier_catalog.csv",
    "standard_checkpoint_catalog.csv",
    "short_event_checkpoint_catalog.csv",
    "opportunity_coverage_audit.json",
)

STANDARD_CATALOG_REQUIRED_COLUMNS = (
    "event_id",
    "rainfall_sha256",
    "checkpoint_id",
    "elapsed_min",
    "checkpoint_role",
    "rainfall_phase",
    "opportunity_score",
    "event_tier",
    "checkpoint_state_source",
    "network_sha256",
    "config_sha256",
    "source_run_uuid",
)


def build_canonical_catalogs(
    pool: pd.DataFrame,
    *,
    network_sha256: str,
    config_sha256: str,
    source_run_uuid: str,
) -> dict[str, pd.DataFrame]:
    """Derive the canonical Opportunity catalogs from the scored pool.

    ``standard_checkpoint_catalog`` is the only permitted Pilot400/Train1600
    checkpoint source; ``short_event_checkpoint_catalog`` keeps short_2 and
    short_3 events for auxiliary use only.
    """
    required = {
        "event_id",
        "rainfall_sha256",
        "checkpoint_id",
        "elapsed_min",
        "opportunity_class",
        "phase",
        "opportunity_score",
        "event_tier",
        "source_detail",
    }
    missing = required - set(pool)
    if missing:
        raise ValueError(f"pool missing canonical columns: {sorted(missing)}")
    catalog = pool.copy()
    catalog["checkpoint_role"] = catalog["opportunity_class"].map(
        {"responsive": "responsive", "low_opportunity": "low_opportunity"}
    )
    if catalog["checkpoint_role"].isna().any():
        raise ValueError("pool contains unknown opportunity_class values")
    catalog["rainfall_phase"] = catalog["phase"].astype(str)
    catalog["checkpoint_state_source"] = catalog["source_detail"].astype(str)
    catalog["network_sha256"] = str(network_sha256)
    catalog["config_sha256"] = str(config_sha256)
    catalog["source_run_uuid"] = str(source_run_uuid)

    tiers = classify_event_tiers(catalog)
    event_meta = catalog.drop_duplicates("event_id")[
        [
            column
            for column in (
                "event_id",
                "rainfall_sha256",
                "rainfall_family",
                "event_tier",
            )
            if column in catalog
        ]
    ]
    tier_catalog = tiers.merge(
        event_meta.drop(columns="event_tier", errors="ignore"),
        on="event_id",
        how="left",
        validate="one_to_one",
    )

    standard = catalog[catalog["event_tier"] == "standard_4plus"].copy()
    if not standard.empty:
        grouped = standard.groupby("event_id")
        role_counts = grouped["checkpoint_role"].value_counts().unstack(
            fill_value=0
        )
        if not role_counts.get(
            "responsive", pd.Series(0, index=role_counts.index)
        ).eq(4).all():
            raise ValueError(
                "standard_4plus events must carry exactly 4 responsive checkpoints"
            )
        if not role_counts.get(
            "low_opportunity", pd.Series(0, index=role_counts.index)
        ).eq(1).all():
            raise ValueError(
                "standard_4plus events must carry exactly 1 low-opportunity checkpoint"
            )
        if standard.duplicated(["event_id", "checkpoint_id"]).any():
            raise ValueError("standard catalog contains duplicate checkpoints")
        if not grouped.size().eq(5).all():
            raise ValueError(
                "standard_4plus events must carry exactly 5 checkpoints"
            )
        if standard["rainfall_sha256"].astype(str).str.len().ne(64).any():
            raise ValueError("standard catalog carries invalid rainfall_sha256")

    short = catalog[catalog["event_tier"].isin(["short_2", "short_3"])].copy()
    missing_columns = set(STANDARD_CATALOG_REQUIRED_COLUMNS) - set(standard)
    if missing_columns:
        raise ValueError(
            f"standard catalog missing columns: {sorted(missing_columns)}"
        )
    return {
        "event_tier_catalog": tier_catalog.reset_index(drop=True),
        "standard_checkpoint_catalog": standard.reset_index(drop=True),
        "short_event_checkpoint_catalog": short.reset_index(drop=True),
    }
