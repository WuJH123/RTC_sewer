"""Gate 5R pipeline helpers and evidence builders.

Heavy authoritative SWMM runs are launched by the CLI entrypoint.  This module
keeps contract checks, accounting, event partitioning, and audit calculations
testable without starting SWMM.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from sewerrtc.control.v4_action_authority import classify_action_authority
from sewerrtc.control.v4_opportunity import scan_control_opportunities
from sewerrtc.simulation.kpi_metrics import compute_window_kpis


EXIT_PASS = 0
EXIT_BLOCKED = 2
EXIT_INCOMPLETE = 3
EXIT_RUNTIME_ERROR = 4
EXIT_SCIENTIFIC_FAIL = 5


def action_authority_reference_name() -> str:
    """Return the frozen action anchor used for action-authority diagnosis."""
    return "hold_previous"


def safe_repeat_noise_ranges(samples: pd.DataFrame) -> dict[str, float]:
    """Return JSON-safe repeat ranges, including for an empty accepted set."""
    mapping = {
        "pfv_m3": "repeat_pfv_range_m3",
        "tfv_m3": "repeat_tfv_range_m3",
        "peak_m3s": "repeat_peak_range_m3s",
    }
    result: dict[str, float] = {}
    for output_name, column in mapping.items():
        if column not in samples.columns or samples.empty:
            result[output_name] = 0.0
            continue
        values = pd.to_numeric(samples[column], errors="coerce")
        finite = values[np.isfinite(values)]
        result[output_name] = float(finite.max()) if len(finite) else 0.0
    return result


def canary_gate_status(pass_gate: bool, accepted: int) -> str:
    if bool(pass_gate):
        return "pass"
    return "incomplete" if int(accepted) == 0 else "scientific_fail"


def confirmed_flat_fraction_is_in_range(
    samples: pd.DataFrame, minimum: float, maximum: float
) -> bool:
    """Check the negative-control share in the accepted sample population."""
    if samples.empty or "confirmed_flat" not in samples.columns:
        return False
    fraction = float(samples["confirmed_flat"].astype(bool).mean())
    return float(minimum) <= fraction <= float(maximum)


def rebuild_run_manifest_from_completions(
    plan: pd.DataFrame,
    run_root: Path,
    reference_contract_hash: str | None = None,
) -> pd.DataFrame:
    """Reconstruct durable run accounting from every valid completion marker."""
    if "case_id" not in plan.columns:
        raise ValueError("plan is missing case_id")
    rows: list[dict] = []
    for case_id in plan["case_id"].astype(str):
        completion_path = Path(run_root) / case_id / "completion.json"
        if not completion_path.exists():
            continue
        try:
            completion = json.loads(completion_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue
        if completion.get("status") != "pass":
            continue
        if (
            reference_contract_hash is not None
            and completion.get("reference_contract_hash")
            != str(reference_contract_hash)
        ):
            continue
        rows.append(
            {
                "case_id": case_id,
                "status": "accepted",
                "error": "",
                "reused": True,
                "completion_path": str(completion_path),
                "reference_contract_hash": str(
                    completion.get("reference_contract_hash", "")
                ),
            }
        )
    return pd.DataFrame(rows)


def _hash_numeric_frame(frame: pd.DataFrame, columns: Sequence[str]) -> str:
    shared = [column for column in columns if column in frame.columns]
    if frame.empty or not shared:
        return ""
    values = (
        frame[shared]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(-9999.0)
        .to_numpy(dtype=np.float64)
    )
    return hashlib.sha256(np.round(values, 8).tobytes()).hexdigest()


def branch_state_hashes(
    detail: pd.DataFrame,
    checkpoint_min: float,
    facility_ids: Sequence[str],
    prefix_history_min: float = 60.0,
    horizon_min: float = 120.0,
) -> dict:
    """Hash the exact native prefix, pre-action state, and post schedules."""
    if "elapsed_min" not in detail.columns:
        raise ValueError("detail is missing elapsed_min")
    elapsed = pd.to_numeric(detail["elapsed_min"], errors="coerce")
    prefix = detail[
        (elapsed >= float(checkpoint_min) - float(prefix_history_min))
        & (elapsed < float(checkpoint_min))
    ].copy()
    checkpoint_rows = detail[
        np.isclose(elapsed.to_numpy(float), float(checkpoint_min), atol=1e-6)
    ].copy()
    pre_action = (
        checkpoint_rows.head(1).copy()
        if len(checkpoint_rows)
        else prefix.tail(1).copy()
    )
    post = detail[
        (elapsed > float(checkpoint_min))
        & (elapsed < float(checkpoint_min) + float(horizon_min))
    ].copy()
    hydraulic_columns = sorted(
        column
        for column in detail.columns
        if column.startswith(("h:", "head:", "flood:", "storage_volume:", "flow:"))
    )
    actual_columns = [
        f"actual_setting:{facility_id}"
        for facility_id in facility_ids
        if f"actual_setting:{facility_id}" in detail.columns
    ]
    readback_columns = [
        f"readback_setting:{facility_id}"
        for facility_id in facility_ids
        if f"readback_setting:{facility_id}" in detail.columns
    ]
    return {
        "prefix_history_rows": int(len(prefix)),
        "prefix_history_sha256": _hash_numeric_frame(prefix, hydraulic_columns),
        "checkpoint_pre_action_elapsed_min": (
            float(pre_action["elapsed_min"].iloc[0]) if len(pre_action) else None
        ),
        "checkpoint_pre_action_sha256": _hash_numeric_frame(
            pre_action, hydraulic_columns
        ),
        "post_actual_schedule_sha256": _hash_numeric_frame(post, actual_columns),
        "post_readback_schedule_sha256": _hash_numeric_frame(
            post, readback_columns
        ),
    }


def post_decision_readback_mask(
    elapsed_min: pd.Series,
    checkpoint_min: float,
    decision_interval_min: float = 10.0,
    sample_interval_min: float = 5.0,
    horizon_min: float = 120.0,
) -> pd.Series:
    """Select stable readback samples after native time-rule evaluation.

    A sample exactly on a 10-minute decision boundary represents the
    pre-transition routing state in PySWMM.  The following 5-minute sample is
    the first unambiguous observation of the new high-priority native rule.
    """
    elapsed = pd.to_numeric(elapsed_min, errors="coerce")
    offset = elapsed - float(checkpoint_min)
    in_window = (offset > 0.0) & (offset < float(horizon_min))
    phase = np.mod(offset.to_numpy(float), float(decision_interval_min))
    stable = np.isclose(phase, float(sample_interval_min), atol=1e-6)
    return pd.Series(in_window.to_numpy(bool) & stable, index=elapsed_min.index)


def hashes_match_across_branches(
    branch_hashes: dict[str, dict], keys: Sequence[str]
) -> bool:
    if not branch_hashes:
        return False
    for key in keys:
        values = [str(item.get(key, "")) for item in branch_hashes.values()]
        if not values or any(not value for value in values) or len(set(values)) != 1:
            return False
    return True


def gate_exit_code(status: str) -> int:
    return {
        "pass": EXIT_PASS,
        "blocked": EXIT_BLOCKED,
        "incomplete": EXIT_INCOMPLETE,
        "runtime_error": EXIT_RUNTIME_ERROR,
        "scientific_fail": EXIT_SCIENTIFIC_FAIL,
    }.get(str(status), EXIT_BLOCKED)


def reference_cache_key(
    event_id: str, checkpoint_min: float, contract_hash: str
) -> str:
    payload = f"{event_id}|{float(checkpoint_min):.6f}|{contract_hash}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


def reference_cache_is_ready(
    reference_dir: Path, reference_contract_hash: str
) -> bool:
    """Return True only for a complete, matching, read-only reference cache."""
    completion_path = Path(reference_dir) / "completion.json"
    required = (
        "no_control_detail.csv",
        "dynamic_internal_rules_detail.csv",
        "hold_previous_detail.csv",
    )
    if not completion_path.exists():
        return False
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False
    return bool(
        completion.get("status") == "pass"
        and completion.get("reference_contract_hash")
        == str(reference_contract_hash)
        and all(
            (Path(reference_dir) / filename).exists()
            and (Path(reference_dir) / filename).stat().st_size > 0
            for filename in required
        )
    )


def pending_case_ids(
    planned_case_ids: Iterable[str], completed_case_ids: Iterable[str]
) -> list[str]:
    completed = {str(case_id) for case_id in completed_case_ids}
    return [
        str(case_id)
        for case_id in planned_case_ids
        if str(case_id) not in completed
    ]


def select_pending_plan(
    plan: pd.DataFrame,
    completed_case_ids: Iterable[str],
    limit: int = 0,
) -> pd.DataFrame:
    """Select the next deterministic batch after excluding completed cases."""
    if "case_id" not in plan.columns:
        raise ValueError("plan is missing case_id")
    completed = {str(case_id) for case_id in completed_case_ids}
    pending = plan[
        ~plan["case_id"].astype(str).isin(completed)
    ].copy()
    if int(limit) > 0:
        pending = pending.head(int(limit)).copy()
    return pending.reset_index(drop=True)


def output_roots_are_isolated(new_root: Path, legacy_root: Path) -> bool:
    return new_root.resolve() != legacy_root.resolve() and legacy_root.resolve() not in new_root.resolve().parents


def accounting_is_closed(
    planned: int, accepted: int, rejected: int, pending: int, missing: int
) -> bool:
    return int(planned) == sum(
        int(value) for value in (accepted, rejected, pending, missing)
    )


def classify_candidate_result(
    delta_pfv_m3: float,
    delta_tfv_m3: float,
    delta_peak_m3s: float,
    action_cost: float,
    minimum_tfv_improvement_m3: float = 25.0,
    minimum_benefit_cost_ratio: float = 1.5,
) -> dict:
    """Apply the frozen zero-margin science gate and separate dead zones."""
    values = np.asarray(
        [delta_pfv_m3, delta_tfv_m3, delta_peak_m3s, action_cost],
        dtype=float,
    )
    if not np.isfinite(values).all() or action_cost < 0:
        raise ValueError("candidate result values must be finite and cost non-negative")
    pfv_noninferior = float(delta_pfv_m3) <= 0.0
    tfv_noninferior = float(delta_tfv_m3) <= 0.0
    peak_noninferior = float(delta_peak_m3s) <= 0.0
    joint_noninferior = pfv_noninferior and tfv_noninferior and peak_noninferior
    benefit = max(0.0, -float(delta_tfv_m3))
    ratio = (
        benefit / float(action_cost)
        if action_cost > 0
        else (float("inf") if benefit > 0 else 0.0)
    )
    materially_beneficial = (
        joint_noninferior
        and benefit >= float(minimum_tfv_improvement_m3)
        and ratio >= float(minimum_benefit_cost_ratio)
    )
    return {
        "pfv_noninferior": bool(pfv_noninferior),
        "tfv_noninferior": bool(tfv_noninferior),
        "peak_noninferior": bool(peak_noninferior),
        "joint_noninferior": bool(joint_noninferior),
        "tfv_benefit_m3": float(benefit),
        "benefit_cost_ratio": float(ratio),
        "materially_beneficial": bool(materially_beneficial),
    }


def schedule_action_cost(schedule: np.ndarray, anchor: np.ndarray) -> float:
    """Auditable simple cost: changed coordinates plus total variation."""
    candidate = np.asarray(schedule, dtype=float)
    reference = np.asarray(anchor, dtype=float)
    if candidate.shape != reference.shape or candidate.ndim != 2:
        raise ValueError("schedule and anchor must be equally shaped 2-D arrays")
    changed = float(np.any(np.abs(candidate - reference) > 1e-6, axis=0).sum())
    path = np.vstack([reference[0], candidate])
    variation = float(np.abs(np.diff(path, axis=0)).sum())
    return changed + variation


def audit_contract_values(
    recovery_contract: dict,
    dataset_contract: dict,
    v4_config: dict,
    facility_ids: list[str],
) -> dict:
    roles = recovery_contract.get("reference_roles", {})
    checks = {
        "engineering36_count": len(facility_ids) == 36,
        "state_sampling_300_sec": int(
            recovery_contract.get("control_step_sec", -1)
        )
        == 300,
        "control_interval_600_sec": int(
            recovery_contract.get("control_interval_sec", -1)
        )
        == 600,
        "horizon_120_min": int(
            recovery_contract.get("prediction_horizon_min", -1)
        )
        == 120,
        "horizon_12_steps": int(
            recovery_contract.get("prediction_horizon_steps", -1)
        )
        == 12,
        "max_k_8": int(recovery_contract.get("max_k", -1)) == 8
        and int(v4_config.get("v4", {}).get("adaptive_k", {}).get("max_k", -1))
        == 8,
        "pfv_reference_no_control": str(
            roles.get("PFV", {}).get("primary", "")
        ).lower()
        == "no-control",
        "tfv_reference_dynamic_internal": str(
            roles.get("TFV", {}).get("primary", "")
        ).lower()
        .replace(" (to be regenerated)", "")
        == "dynamic internal",
        "peak_reference_dynamic_internal": str(
            roles.get("Peak", {}).get("primary", "")
        ).lower()
        .replace(" (to be regenerated)", "")
        == "dynamic internal",
        "peak_unit_m3s": recovery_contract.get("kpi_definitions", {})
        .get("Peak", {})
        .get("unit")
        == "m3/s",
        "dataset_contract_v1_1_or_later": tuple(
            int(part)
            for part in str(dataset_contract.get("version", "0.0")).split(".")[:2]
        )
        >= (1, 1),
        "accepted_target_1600": int(
            v4_config.get("v4", {}).get("aug1", {}).get("effective_target", -1)
        )
        == 1600,
    }
    return {
        "status": "pass" if all(checks.values()) else "blocked",
        "checks": checks,
        "failed_checks": [name for name, passed in checks.items() if not passed],
    }


def build_formal_1600_plan(
    events: pd.DataFrame, seed: int = 20260726
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if "event_id" not in events.columns:
        raise ValueError("events is missing event_id")
    eligible = events.copy()
    if "eligible" in eligible.columns:
        eligible = eligible[eligible["eligible"].astype(bool)]
    eligible = eligible.drop_duplicates("event_id").reset_index(drop=True)
    if len(eligible) < 80:
        raise ValueError("at least 80 eligible independent events are required")
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(len(eligible))[:80]
    selected = eligible.iloc[order].copy().reset_index(drop=True)
    selected["split"] = (
        ["train"] * 48
        + ["model_validation"] * 8
        + ["challenge"] * 8
        + ["reserve"] * 16
    )
    selected["partition_seed"] = int(seed)

    checkpoint_roles = [
        "responsive_1",
        "responsive_2",
        "responsive_3",
        "responsive_4",
        "flat_action_probe",
    ]
    candidate_roles = [
        "best_safe",
        "pfv_boundary",
        "tfv_improved_pfv_unsafe",
        "peak_degraded",
        "uncertainty_or_coverage",
    ]
    rows: list[dict] = []
    for _, event in selected[selected["split"] != "reserve"].iterrows():
        for checkpoint_role in checkpoint_roles:
            for candidate_role in candidate_roles:
                rows.append(
                    {
                        "event_id": event["event_id"],
                        "split": event["split"],
                        "checkpoint_role": checkpoint_role,
                        "candidate_role": candidate_role,
                        "status": "planned",
                    }
                )
    plan = pd.DataFrame(rows)
    return plan, selected


def build_pilot_plan(
    opportunities: pd.DataFrame,
    events: int = 8,
    responsive_per_event: int = 4,
    flat_per_event: int = 1,
    seed: int = 20260726,
) -> pd.DataFrame:
    required = {"event_id", "checkpoint_min", "opportunity_class"}
    missing = required - set(opportunities.columns)
    if missing:
        raise ValueError(f"opportunities missing columns: {sorted(missing)}")
    source = opportunities.copy()
    source = source.drop_duplicates(["event_id", "checkpoint_min"])
    eligible_events: list[str] = []
    for event_id, group in source.groupby("event_id"):
        low_control = group.nsmallest(
            flat_per_event, ["opportunity_score", "checkpoint_min"]
        )
        if (
            int(group["opportunity_class"].eq("responsive").sum())
            >= responsive_per_event
            and len(low_control) >= flat_per_event
        ):
            eligible_events.append(str(event_id))
    if len(eligible_events) < events:
        raise ValueError(
            f"need {events} events with checkpoint quotas, found {len(eligible_events)}"
        )
    rng = np.random.default_rng(int(seed))
    selected_events = [
        eligible_events[index]
        for index in rng.permutation(len(eligible_events))[:events]
    ]
    rows: list[dict] = []
    for event_id in selected_events:
        group = source[source["event_id"].astype(str) == event_id]
        responsive = group[group["opportunity_class"] == "responsive"].sort_values(
            ["opportunity_score", "checkpoint_min"], ascending=[False, True]
        )
        low_control = group[
            ~group.index.isin(responsive.head(responsive_per_event).index)
        ].sort_values(
            ["opportunity_score", "checkpoint_min"], ascending=[True, True]
        )
        chosen = pd.concat(
            [
                responsive.head(responsive_per_event),
                low_control.head(flat_per_event),
            ],
            ignore_index=True,
        )
        for _, checkpoint in chosen.iterrows():
            role = (
                "flat_action_probe"
                if float(checkpoint["checkpoint_min"])
                in set(
                    low_control.head(flat_per_event)["checkpoint_min"].astype(float)
                )
                else "responsive"
            )
            rows.append(
                {
                    "event_id": event_id,
                    "checkpoint_min": float(checkpoint["checkpoint_min"]),
                    "checkpoint_role": role,
                    "source_detail": checkpoint.get("source_detail", ""),
                    "status": "planned",
                }
            )
    return pd.DataFrame(rows)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_existing_gate5(
    project_root: Path,
    output_root: Path,
    checkpoint_min: float = 4380.0,
    horizon_min: float = 120.0,
    dt_sec: int = 300,
) -> dict:
    """Recompute the legacy Gate 5 evidence with authoritative units."""
    legacy_root = (
        project_root
        / "outputs"
        / "project6_dual_reference_v4"
        / "recovery_capability_v2"
        / "gate4_h120_batch0"
    )
    gate5_root = legacy_root / "gate5_exact_diagnosis"
    candidate_files = sorted(
        (gate5_root / "parallel_runs").rglob("*_detail.csv")
    )
    work = legacy_root / "work"
    event_id = "V31_RP10_D2H_P65_v31_independent_gamma_084"
    di_path = work / f"batch0_{event_id}__dynamic_internal_rules_detail.csv"
    nc_path = work / f"batch0_{event_id}__no_control_detail.csv"
    output_root.mkdir(parents=True, exist_ok=True)
    if not candidate_files or not di_path.exists() or not nc_path.exists():
        audit = {
            "status": "incomplete",
            "candidate_files": len(candidate_files),
            "dynamic_internal_exists": di_path.exists(),
            "no_control_exists": nc_path.exists(),
        }
        (output_root / "reaudit_existing_gate5.json").write_text(
            json.dumps(audit, indent=2), encoding="utf-8"
        )
        return audit

    facility_ids = (
        pd.read_csv(project_root / "data" / "project6_v3_facility_semantics_36.csv")[
            "facility_id"
        ]
        .astype(str)
        .tolist()
    )

    def read_window(path: Path) -> pd.DataFrame:
        header = pd.read_csv(path, nrows=0).columns.tolist()
        prefixes = (
            "requested_setting:",
            "target_setting:",
            "a:",
            "flow:",
            "head:",
            "storage_volume:",
            "flood:",
        )
        usecols = [
            column
            for column in header
            if column == "elapsed_min" or column.startswith(prefixes)
        ]
        frame = pd.read_csv(path, usecols=usecols)
        return frame[
            (frame["elapsed_min"] >= checkpoint_min)
            & (frame["elapsed_min"] < checkpoint_min + horizon_min)
        ].reset_index(drop=True)

    di = read_window(di_path)
    nc = read_window(nc_path)
    di_kpi = compute_window_kpis(di, [], checkpoint_min, horizon_min, dt_sec)
    nc_kpi = compute_window_kpis(nc, [], checkpoint_min, horizon_min, dt_sec)
    # Recompute PFV with the frozen sentinel list.
    priority_file = project_root / "data" / "project6_v3_sentinel_nodes.txt"
    priority = [
        line.strip()
        for line in priority_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    di_kpi = compute_window_kpis(di, priority, checkpoint_min, horizon_min, dt_sec)
    nc_kpi = compute_window_kpis(nc, priority, checkpoint_min, horizon_min, dt_sec)

    rows: list[dict] = []
    first_candidate: pd.DataFrame | None = None
    for candidate_path in candidate_files:
        candidate = read_window(candidate_path)
        authority_vs_di = classify_action_authority(di, candidate, facility_ids)
        if first_candidate is None:
            first_candidate = candidate
            pairwise = None
        else:
            pairwise = classify_action_authority(
                first_candidate, candidate, facility_ids
            )
        kpi = compute_window_kpis(
            candidate, priority, checkpoint_min, horizon_min, dt_sec
        )
        actual_columns = [
            f"a:{facility_id}"
            for facility_id in facility_ids
            if f"a:{facility_id}" in candidate.columns
        ]
        actual_hash = (
            hashlib.sha256(
                np.round(candidate[actual_columns].to_numpy(float), 8).tobytes()
            ).hexdigest()
            if actual_columns
            else ""
        )
        rows.append(
            {
                "candidate_id": candidate_path.parent.name,
                "detail_path": str(candidate_path),
                "detail_sha256": _sha256_file(candidate_path),
                "actual_schedule_sha256": actual_hash,
                "PFV_m3": kpi["PFV"],
                "TFV_m3": kpi["TFV"],
                "peak_TFV_rate_m3s": kpi["peak_TFV_rate"],
                "delta_PFV_vs_NC_m3": kpi["PFV"] - nc_kpi["PFV"],
                "delta_TFV_vs_DI_m3": kpi["TFV"] - di_kpi["TFV"],
                "delta_peak_vs_DI_m3s": kpi["peak_TFV_rate"]
                - di_kpi["peak_TFV_rate"],
                **{
                    f"vs_DI_{key}": value
                    for key, value in authority_vs_di.to_dict().items()
                },
                "pairwise_flatness_class": (
                    pairwise.authority_class
                    if pairwise is not None
                    else "reference_candidate"
                ),
                "pairwise_actual_action_distance": (
                    pairwise.actual_action_distance
                    if pairwise is not None
                    else 0.0
                ),
                "pairwise_local_hydraulic_distance": (
                    pairwise.local_hydraulic_distance
                    if pairwise is not None
                    else 0.0
                ),
                "pairwise_kpi_distance": (
                    pairwise.kpi_distance if pairwise is not None else 0.0
                ),
            }
        )
    result = pd.DataFrame(rows)
    result.to_csv(output_root / "existing_gate5_reaudit.csv", index=False)
    audit = {
        "status": "pass",
        "legacy_evidence_status": "superseded_invalid_kpi_units",
        "n_candidates": len(result),
        "authority_vs_di_class_counts": result[
            "vs_DI_authority_class"
        ].value_counts().to_dict(),
        "pairwise_flatness_class_counts": result[
            "pairwise_flatness_class"
        ].value_counts().to_dict(),
        "n_unique_actual_schedules": int(
            result["actual_schedule_sha256"].nunique()
        ),
        "kpi_contract": {
            "volume": "sum flooding_rate * 300 s",
            "peak": "max summed flooding_rate in m3/s",
        },
    }
    (output_root / "reaudit_existing_gate5.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    return audit


def build_event_inventory(
    catalog_path: Path,
    revealed_event_ids: Iterable[str] = (),
) -> pd.DataFrame:
    catalog = pd.read_csv(catalog_path)
    revealed = {str(event_id) for event_id in revealed_event_ids}
    series_hash = (
        "rainfall_series_sha256"
        if "rainfall_series_sha256" in catalog.columns
        else "rainfall_file_sha256"
    )
    inventory = catalog.copy()
    inventory["revealed"] = inventory["event_id"].astype(str).isin(revealed)
    inventory["eligible"] = (
        inventory.get("split", "").astype(str).eq("development_fit")
        & ~inventory["revealed"]
    )
    inventory["duplicate_rainfall_series"] = inventory.duplicated(
        series_hash, keep="first"
    )
    missing_series_hash = (
        inventory[series_hash].isna()
        | inventory[series_hash].astype(str).str.strip().eq("")
    )
    inventory["missing_rainfall_series_hash"] = missing_series_hash
    inventory.loc[
        inventory["duplicate_rainfall_series"] | missing_series_hash, "eligible"
    ] = False
    return inventory


def scan_existing_dynamic_internal(
    detail_paths: Iterable[Path],
    facility_ids: list[str],
    facility_nodes: dict[str, tuple[str, str]] | None = None,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in detail_paths:
        header = pd.read_csv(path, nrows=0).columns.tolist()
        usecols = [
            column
            for column in header
            if column in {"event_id", "elapsed_min", "rainfall_mm_h"}
            or column in {
                "excess_fullness_p95",
                "system_inflow_m3s",
                "total_outfall_flow_m3s",
            }
            or column.startswith(
                ("flow:", "a:", "flood:", "storage_volume:", "h:", "head:")
            )
        ]
        detail = pd.read_csv(path, usecols=usecols)
        opportunities = scan_control_opportunities(
            detail, facility_ids, facility_nodes=facility_nodes
        )
        event_id = (
            str(detail["event_id"].iloc[0])
            if "event_id" in detail.columns and len(detail)
            else path.stem
        )
        opportunities.insert(0, "event_id", event_id)
        opportunities["source_detail"] = str(path)
        rows.append(opportunities)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
