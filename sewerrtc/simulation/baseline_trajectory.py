from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from sewerrtc.contracts.prompt3a import INP_PATH, OUT_ROOT, PROJECT_ROOT, read_csv, sha256_file, utc_now, write_csv, write_json


PLAN_SCHEMA_VERSION = "project6_baseline_trajectory_plan_v1"
POLICIES = ("no_control", "internal_rules", "executable_passive")
DEVELOPMENT_SPLITS = {"development_fit", "action_effect_fit"}
EXCLUDED_SPLITS = {"gat_independent_holdout", "calibration", "formal", "formal_blind"}

PLAN_COLUMNS = [
    "plan_schema_version",
    "trajectory_id",
    "event_id",
    "canonical_event_id",
    "storm_family_id",
    "split",
    "policy_id",
    "policy_mode",
    "network_policy",
    "rainfall_path",
    "rainfall_file_sha256",
    "rainfall_series_sha256",
    "network_path",
    "network_sha256",
    "event_catalog_path",
    "event_catalog_sha256",
    "event_split_manifest_sha256",
    "prompt2_import_lock_sha256",
    "native_rule_audit_sha256",
    "passive_fallback_contract_sha256",
    "internal_fallback_contract_sha256",
    "fallback_selection_contract_sha256",
    "truth_controller_separation_required",
    "controller_visible_state_contract_version",
    "controller_memory_required",
    "hotstart_required",
    "start_time",
    "end_time",
    "tail_min",
    "tail_policy",
    "output_root",
    "status",
    "exclusion_reason",
]


class BaselinePlanError(RuntimeError):
    def __init__(self, exit_code: int, reason: str, diagnostics: dict[str, Any] | None = None) -> None:
        super().__init__(reason)
        self.exit_code = exit_code
        self.reason = reason
        self.diagnostics = diagnostics or {}


def _sha_or_missing(path: Path) -> tuple[str, str]:
    if not path.exists() or not path.is_file():
        return "", str(path)
    return sha256_file(path), ""


def _contract_hashes() -> tuple[dict[str, str], list[str]]:
    paths = {
        "prompt2_import_lock_sha256": OUT_ROOT / "completion_markers" / "ImportPrompt2Artifacts_COMPLETED.json",
        "native_rule_audit_sha256": OUT_ROOT / "native_rules" / "native_rule_audit_report.json",
        "passive_fallback_contract_sha256": OUT_ROOT / "fallbacks" / "passive_fallback_contract.json",
        "internal_fallback_contract_sha256": OUT_ROOT / "fallbacks" / "internal_fallback_contract.json",
        "fallback_selection_contract_sha256": OUT_ROOT / "fallbacks" / "fallback_selection_contract.json",
    }
    hashes: dict[str, str] = {}
    missing: list[str] = []
    for key, path in paths.items():
        value, miss = _sha_or_missing(path)
        hashes[key] = value
        if miss:
            missing.append(miss)
    return hashes, missing


def _indexed_split_rows(split_manifest_path: Path) -> dict[str, dict[str, str]]:
    rows = read_csv(split_manifest_path)
    indexed: dict[str, dict[str, str]] = {}
    for row in rows:
        event_id = row.get("event_id", "")
        if event_id:
            indexed[event_id] = row
    return indexed


def _eligible_event_rows(catalog_rows: list[dict[str, str]], split_rows: dict[str, dict[str, str]]) -> tuple[list[dict[str, str]], list[dict[str, str]], list[str]]:
    eligible: list[dict[str, str]] = []
    excluded: list[dict[str, str]] = []
    problems: list[str] = []
    seen_canonical: set[str] = set()
    for row in catalog_rows:
        event_id = row.get("event_id", "")
        canonical = row.get("canonical_event_id") or event_id
        if not event_id:
            problems.append("event_catalog_row_missing_event_id")
            continue
        if canonical in seen_canonical:
            problems.append(f"duplicate_canonical_event_id:{canonical}")
        seen_canonical.add(canonical)
        split_record = split_rows.get(event_id)
        if not split_record:
            problems.append(f"missing_event_split_manifest_row:{event_id}")
            continue
        split = split_record.get("split") or row.get("split", "")
        joined = dict(row)
        joined["split"] = split
        if row.get("gat_independent_holdout") == "true":
            joined["exclusion_reason"] = "gat_independent_holdout"
            excluded.append(joined)
            continue
        if split in EXCLUDED_SPLITS or row.get("calibration_eligible") == "true" or row.get("formal_eligible") == "true":
            joined["exclusion_reason"] = "calibration_or_formal_or_holdout"
            excluded.append(joined)
            continue
        if split not in DEVELOPMENT_SPLITS:
            joined["exclusion_reason"] = f"split_not_development:{split}"
            excluded.append(joined)
            continue
        if split_record.get("round0_eligible", row.get("round0_eligible", "")).lower() != "true":
            joined["exclusion_reason"] = "round0_not_eligible"
            excluded.append(joined)
            continue
        eligible.append(joined)
    return eligible, excluded, problems


def _validate_plan_rows(rows: list[dict[str, Any]], report: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    if not rows:
        problems.append("plan_file_empty")
    trajectory_ids = [str(row.get("trajectory_id", "")) for row in rows]
    duplicate_trajectory_ids = [key for key, count in Counter(trajectory_ids).items() if count > 1]
    if duplicate_trajectory_ids:
        problems.append(f"duplicate_trajectory_id:{'|'.join(duplicate_trajectory_ids[:10])}")
    event_policy = [(str(row.get("event_id", "")), str(row.get("policy_id", ""))) for row in rows]
    duplicate_event_policy = [f"{event}:{policy}" for (event, policy), count in Counter(event_policy).items() if count > 1]
    if duplicate_event_policy:
        problems.append(f"duplicate_event_policy:{'|'.join(duplicate_event_policy[:10])}")
    policies_by_event: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        policies_by_event[str(row.get("event_id", ""))].add(str(row.get("policy_id", "")))
    incomplete_events = [event for event, policies in policies_by_event.items() if policies != set(POLICIES)]
    if incomplete_events:
        problems.append(f"missing_three_policies:{'|'.join(incomplete_events[:10])}")
    for row in rows:
        for col in PLAN_COLUMNS:
            if col not in row:
                problems.append(f"missing_plan_column:{col}")
                break
        if not row.get("split"):
            problems.append(f"missing_split:{row.get('event_id','')}")
        rainfall_value = str(row.get("rainfall_path", "")).strip()
        rainfall_path = Path(rainfall_value) if rainfall_value else None
        if rainfall_path is None or not rainfall_path.is_file():
            problems.append(f"rainfall_path_missing:{row.get('event_id','')}:{rainfall_value or '<blank>'}")
        if not row.get("rainfall_file_sha256") or not row.get("rainfall_series_sha256"):
            problems.append(f"rainfall_hash_missing:{row.get('event_id','')}")
        if row.get("policy_id") != row.get("policy_mode"):
            problems.append(f"policy_id_policy_mode_mismatch:{row.get('trajectory_id','')}")
        if row.get("split") in EXCLUDED_SPLITS:
            problems.append(f"excluded_split_in_plan:{row.get('event_id','')}:{row.get('split')}")
    if len(set(row.get("network_sha256", "") for row in rows)) > 1:
        problems.append("network_hash_not_uniform")
    if report.get("planned_trajectory_count") != len(rows):
        problems.append("plan_row_count_report_mismatch")
    return problems


def plan_baseline_trajectories(
    event_catalog_path: str | Path,
    out_dir: str | Path = OUT_ROOT / "baseline_trajectories",
    split_manifest_path: str | Path | None = None,
    split_leakage_path: str | Path | None = None,
) -> tuple[int, dict[str, Any], list[Path]]:
    out_dir = Path(out_dir)
    event_catalog_path = Path(event_catalog_path)
    split_manifest_path = Path(split_manifest_path) if split_manifest_path else event_catalog_path.with_name("event_split_manifest.csv")
    split_leakage_path = Path(split_leakage_path) if split_leakage_path else event_catalog_path.with_name("event_split_leakage_audit.csv")
    missing_assets = [str(p) for p in [event_catalog_path, split_manifest_path, split_leakage_path] if not p.exists()]
    contract_hashes, missing_contracts = _contract_hashes()
    missing_assets.extend(missing_contracts)
    if missing_assets:
        report = {"status": "blocked", "failure_reason": "missing_required_plan_assets", "missing_assets": missing_assets, "created_at": utc_now()}
        report_path = write_json(out_dir / "baseline_trajectory_plan_report.json", report)
        return 3, report, [report_path]

    catalog_rows = read_csv(event_catalog_path)
    split_rows = _indexed_split_rows(split_manifest_path)
    leakage_rows = read_csv(split_leakage_path)
    if leakage_rows:
        report = {"status": "failed_gate", "failure_reason": "event_split_leakage_audit_not_empty", "leakage_count": len(leakage_rows), "created_at": utc_now()}
        report_path = write_json(out_dir / "baseline_trajectory_plan_report.json", report)
        return 5, report, [report_path]

    eligible_events, excluded_events, catalog_problems = _eligible_event_rows(catalog_rows, split_rows)
    network_sha256 = sha256_file(INP_PATH)
    event_catalog_sha256 = sha256_file(event_catalog_path)
    event_split_sha256 = sha256_file(split_manifest_path)
    rows: list[dict[str, Any]] = []
    output_root = OUT_ROOT / "baseline_trajectories" / "trajectories"
    for event in eligible_events:
        rainfall_path = Path(event.get("rainfall_path", ""))
        rainfall_file_sha = sha256_file(rainfall_path) if rainfall_path.is_file() else ""
        rainfall_series_sha = event.get("rainfall_series_sha256") or event.get("rainfall_series_hash") or rainfall_file_sha
        for policy in POLICIES:
            rows.append(
                {
                    "plan_schema_version": PLAN_SCHEMA_VERSION,
                    "trajectory_id": f"{event.get('canonical_event_id')}_{policy}",
                    "event_id": event.get("event_id", ""),
                    "canonical_event_id": event.get("canonical_event_id", event.get("event_id", "")),
                    "storm_family_id": event.get("storm_family_id", ""),
                    "split": event.get("split", ""),
                    "policy_id": policy,
                    "policy_mode": policy,
                    "network_policy": "single_retrofit_inp",
                    "rainfall_path": str(rainfall_path),
                    "rainfall_file_sha256": event.get("rainfall_file_sha256") or event.get("rainfall_file_hash") or rainfall_file_sha,
                    "rainfall_series_sha256": rainfall_series_sha,
                    "network_path": str(INP_PATH),
                    "network_sha256": network_sha256,
                    "event_catalog_path": str(event_catalog_path),
                    "event_catalog_sha256": event_catalog_sha256,
                    "event_split_manifest_sha256": event_split_sha256,
                    **contract_hashes,
                    "truth_controller_separation_required": True,
                    "controller_visible_state_contract_version": "project6_controller_visible_state_v1",
                    "controller_memory_required": True,
                    "hotstart_required": True,
                    "start_time": event.get("start_time", ""),
                    "end_time": event.get("end_time", ""),
                    "tail_min": 180,
                    "tail_policy": "rain_end_plus_180min_or_recovery",
                    "output_root": str(output_root / event.get("canonical_event_id", event.get("event_id", "")) / policy),
                    "status": "planned",
                    "exclusion_reason": "",
                }
            )

    report = {
        "status": "completed",
        "created_at": utc_now(),
        "plan_schema_version": PLAN_SCHEMA_VERSION,
        "planned_event_count": len(eligible_events),
        "planned_trajectory_count": len(rows),
        "excluded_event_count": len(excluded_events),
        "excluded_gat_holdout_count": sum(1 for row in excluded_events if row.get("exclusion_reason") == "gat_independent_holdout"),
        "policies": list(POLICIES),
        "event_catalog_path": str(event_catalog_path),
        "event_catalog_sha256": event_catalog_sha256,
        "event_split_manifest_sha256": event_split_sha256,
        "network_path": str(INP_PATH),
        "network_sha256": network_sha256,
        "outputs": {
            "plan": str(out_dir / "baseline_trajectory_plan.csv"),
            "report": str(out_dir / "baseline_trajectory_plan_report.json"),
            "schema": str(out_dir / "trajectory_schema.json"),
            "exclusions": str(out_dir / "baseline_trajectory_exclusion_audit.csv"),
        },
    }
    problems = catalog_problems + _validate_plan_rows(rows, report)
    if problems:
        report["status"] = "contract_mismatch"
        report["failure_reason"] = "baseline_plan_contract_violations"
        report["violations"] = problems
        report_path = write_json(out_dir / "baseline_trajectory_plan_report.json", report)
        write_csv(out_dir / "baseline_trajectory_plan.csv", rows, PLAN_COLUMNS)
        write_csv(out_dir / "baseline_trajectory_exclusion_audit.csv", excluded_events)
        return 6, report, [report_path]

    files = [
        write_csv(out_dir / "baseline_trajectory_plan.csv", rows, PLAN_COLUMNS),
        write_json(out_dir / "baseline_trajectory_plan_report.json", report),
        write_json(out_dir / "trajectory_schema.json", {"truth_controller_separation_required": True, "same_retrofit_inp_required": True, "plan_schema_version": PLAN_SCHEMA_VERSION}),
        write_csv(out_dir / "baseline_trajectory_exclusion_audit.csv", excluded_events),
    ]
    return 0, report, files


def validate_frozen_baseline_plan(plan_path: str | Path) -> tuple[bool, dict[str, Any]]:
    path = Path(plan_path)
    rows = read_csv(path)
    report = {"planned_trajectory_count": len(rows)}
    if not path.exists():
        return False, {"status": "blocked", "failure_reason": "plan_missing", "plan_path": str(path)}
    missing_columns = [col for col in PLAN_COLUMNS if rows and col not in rows[0]]
    if missing_columns:
        return False, {"status": "contract_mismatch", "failure_reason": "old_or_invalid_baseline_plan_schema", "missing_columns": missing_columns}
    problems = _validate_plan_rows(rows, report)
    if problems:
        return False, {"status": "contract_mismatch", "failure_reason": "baseline_plan_contract_violations", "violations": problems}
    return True, {"status": "valid", "plan_sha256": sha256_file(path), "planned_trajectory_count": len(rows)}
