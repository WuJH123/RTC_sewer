"""Zero-SWMM restamp of the frozen Peak Boundary evidence.

RestampPeakBoundaryEvidence re-derives the Peak Boundary Dataset and Audit
from the 240 cached branch outputs only, compares every scientific output
against the frozen archive under ``audits/frozen_evidence/peak_boundary/
<old_code_sha>/`` with the frozen strict numerical tolerance, and — only on
exact agreement — writes a superseding stamp and atomically refreshes the
canonical ``AuditPeakBoundary`` stage status under the new working code SHA.

This module must never invoke a SWMM runner: it does not import
``sewerrtc.v4.simulation``, ``run_prepared_case`` or any pyswmm entry point,
and the stamp it emits always records ``new_swmm_run_count = 0``.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from .peak_boundary import audit_peak_boundary, build_peak_boundary_dataset
from .runtime import atomic_write_json, working_code_sha


RESTAMP_STAGE = "RestampPeakBoundaryEvidence"
CANONICAL_STAGE = "AuditPeakBoundary"
EXPECTED_COMPLETIONS = 240

DELTA_TOLERANCE_KEYS = {
    "delta_pfv_h120_vs_no_control": "pfv_m3",
    "delta_tfv_h120_vs_dynamic_internal": "tfv_m3",
    "delta_peak_h120_vs_dynamic_internal": "peak_m3s",
}

LABEL_COLUMNS = (
    "pfv_safe",
    "tfv_improved",
    "peak_noninferior",
    "joint_noninferior",
    "materially_beneficial",
    "neutral",
    "hard_negative_type",
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: dict) -> str:
    canonical = json.dumps(
        value, sort_keys=True, separators=(",", ":")
    ).encode()
    return hashlib.sha256(canonical).hexdigest()


def _read_id_list(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith(("#", ";"))
    ]


def locate_frozen_archive(output_root: Path) -> Path | None:
    """Return the newest frozen Peak Boundary archive directory."""
    root = Path(output_root) / "audits" / "frozen_evidence" / "peak_boundary"
    candidates = [
        item
        for item in (root.iterdir() if root.exists() else [])
        if item.is_dir() and (item / "archive_manifest.json").exists()
    ]
    if not candidates:
        return None

    def archived_at(item: Path) -> str:
        try:
            payload = json.loads(
                (item / "archive_manifest.json").read_text(encoding="utf-8")
            )
        except (OSError, ValueError):
            return ""
        return str(payload.get("archived_at") or payload.get("created_at") or "")

    return sorted(candidates, key=archived_at)[-1]


def _expected_completion_input_sha(plan_path: Path, config: dict) -> str:
    return _sha256_json(
        {
            "stage": "RunPeakBoundary",
            "plan_sha": _sha256_file(plan_path),
            "network_variant": config.get("runtime", {}).get(
                "network_variant"
            ),
        }
    )


def _network_matches_contract_anchor(
    project_root: Path,
    config: dict,
    plan: pd.DataFrame,
    network_path: Path,
) -> bool:
    """Fail-closed network identity check for plans without network_sha256.

    The frozen historical plan records the network only through
    runner_kwargs.inp_path; anchor the live file hash to the authoritative
    contract network_sha256 and require every plan row to point at that
    single network file.
    """
    contract_rel = str(config.get("project", {}).get("contract", ""))
    if not contract_rel:
        return False
    contract_path = project_root / contract_rel
    if not contract_path.exists():
        return False
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    anchor = str(contract.get("network_sha256", ""))
    if not anchor or _sha256_file(network_path) != anchor:
        return False
    inp_paths: set[str] = set()
    for raw in plan.get("runner_kwargs", pd.Series(dtype=str)).astype(str):
        try:
            kwargs = json.loads(raw)
        except ValueError:
            return False
        inp_paths.add(str(kwargs.get("inp_path", "")))
    if len(inp_paths) != 1:
        return False
    sole = Path(inp_paths.pop())
    if not sole.is_absolute():
        sole = project_root / sole
    try:
        return sole.resolve() == network_path.resolve()
    except OSError:
        return False


def verify_cached_inputs(
    project_root: Path,
    output_root: Path,
    config: dict,
    archive_dir: Path,
) -> tuple[bool, dict]:
    """Verify network/rainfall/input SHA, completions, no-hotstart, raw SHA.

    Read-only fail-closed verification of the cached branch outputs; never
    launches a simulation.
    """
    peak_root = Path(output_root) / "peak_boundary"
    plan_path = peak_root / "peak_boundary_plan.csv"
    manifest_path = peak_root / "run_manifest.csv"
    checks: dict[str, object] = {}
    failures: list[str] = []

    for name, path in (
        ("plan_present", plan_path),
        ("run_manifest_present", manifest_path),
    ):
        checks[name] = path.exists()
        if not path.exists():
            failures.append(name)
    if failures:
        return False, {"checks": checks, "failures": failures}

    # The live plan must be byte-identical to the archived plan, otherwise
    # the 240 completion markers can no longer be attributed to it.
    archived_plan = archive_dir / "peak_boundary" / "peak_boundary_plan.csv"
    plan_frozen_ok = (
        archived_plan.exists()
        and _sha256_file(plan_path) == _sha256_file(archived_plan)
    )
    checks["plan_matches_archive"] = plan_frozen_ok
    if not plan_frozen_ok:
        failures.append("plan_matches_archive")

    plan = pd.read_csv(plan_path)

    # Network SHA: the physical network file must still hash to the value
    # recorded in every plan row.  Historical frozen plans have no
    # network_sha256 column; for those, fall back to the authoritative
    # contract anchor plus the unique runner_kwargs inp_path.
    network_path = project_root / str(
        config.get("project", {}).get("network", "")
    )
    if "network_sha256" in plan.columns:
        plan_network = set(plan["network_sha256"].astype(str))
        network_ok = (
            network_path.exists()
            and len(plan_network) == 1
            and _sha256_file(network_path) in plan_network
        )
    else:
        network_ok = network_path.exists() and _network_matches_contract_anchor(
            project_root, config, plan, network_path
        )
    checks["network_sha_ok"] = network_ok
    if not network_ok:
        failures.append("network_sha_ok")

    # Rainfall SHA + no-hotstart from the frozen runner kwargs.
    rainfall_ok = True
    hotstart_free = not bool(config.get("runtime", {}).get("use_hotstart"))
    seen_rainfall: dict[str, str] = {}
    for _, row in plan.iterrows():
        try:
            kwargs = json.loads(str(row.get("runner_kwargs", "{}")))
        except ValueError:
            rainfall_ok = False
            break
        if any("hotstart" in key.lower() and kwargs[key] for key in kwargs):
            hotstart_free = False
        expected = str(row.get("rainfall_sha256", ""))
        raw_path = str(kwargs.get("rainfall_path", ""))
        if not expected or not raw_path:
            rainfall_ok = False
            continue
        if expected in seen_rainfall:
            rainfall_ok = rainfall_ok and seen_rainfall[expected] == raw_path
            continue
        rain_file = Path(raw_path)
        if not rain_file.is_absolute():
            rain_file = project_root / rain_file
        if not rain_file.exists() or _sha256_file(rain_file) != expected:
            rainfall_ok = False
        seen_rainfall[expected] = raw_path
    checks["rainfall_sha_ok"] = rainfall_ok
    checks["no_hotstart"] = hotstart_free
    if not rainfall_ok:
        failures.append("rainfall_sha_ok")
    if not hotstart_free:
        failures.append("no_hotstart")

    # Case completions: exactly 240 pass markers bound to the frozen plan.
    expected_input_sha = _expected_completion_input_sha(plan_path, config)
    completions = sorted((peak_root / "runs").glob("*/completion.json"))
    completion_ok = len(completions) == EXPECTED_COMPLETIONS
    bad_completions: list[str] = []
    for marker in completions:
        try:
            payload = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            payload = {}
        if (
            payload.get("status") != "pass"
            or payload.get("input_sha") != expected_input_sha
            or str(payload.get("case_id", "")) != marker.parent.name
        ):
            bad_completions.append(marker.parent.name)
    completion_ok = completion_ok and not bad_completions
    checks["completion_count"] = len(completions)
    checks["completion_ok"] = completion_ok
    checks["invalid_completions"] = bad_completions[:10]
    if not completion_ok:
        failures.append("completion_ok")

    # Raw branch file SHA against the frozen archive listing.
    raw_listing = archive_dir / "raw_branch_file_sha256.csv"
    raw_ok = raw_listing.exists()
    raw_mismatches: list[str] = []
    raw_count = 0
    if raw_ok:
        listing = pd.read_csv(raw_listing)
        raw_count = int(len(listing))
        for _, row in listing.iterrows():
            target = peak_root / str(row["relative_path"])
            if not target.exists() or _sha256_file(target) != str(
                row["sha256"]
            ):
                raw_mismatches.append(str(row["relative_path"]))
        raw_ok = not raw_mismatches
    checks["raw_branch_file_count"] = raw_count
    checks["raw_branch_sha_ok"] = raw_ok
    checks["raw_branch_mismatches"] = raw_mismatches[:10]
    if not raw_ok:
        failures.append("raw_branch_sha_ok")

    return not failures, {"checks": checks, "failures": failures}


def rebuild_scientific_outputs(
    project_root: Path, output_root: Path, config: dict
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Rebuild the Peak dataset + audit purely from cached branch outputs."""
    project = config.get("project", {})
    run_manifest = pd.read_csv(
        Path(output_root) / "peak_boundary" / "run_manifest.csv"
    )
    samples, rejected = build_peak_boundary_dataset(
        run_manifest,
        priority_nodes=_read_id_list(
            project_root / str(project.get("priority_nodes", ""))
        ),
        facility_ids=_read_id_list(
            project_root / str(project.get("canonical_ids", ""))
        ),
        scientific_margin=config["thresholds"]["scientific_margin"],
        dead_zone=config["thresholds"]["dead_zone"],
    )
    audit = audit_peak_boundary(samples)
    return samples, rejected, audit


def compare_with_frozen(
    samples: pd.DataFrame,
    audit: dict,
    output_root: Path,
    archive_dir: Path,
    *,
    tolerance: dict[str, float],
) -> tuple[bool, dict]:
    """Compare rebuilt scientific outputs against the frozen archive."""
    frozen_dir = archive_dir / "peak_boundary"
    # round_trip parsing: the default lossy C float parser injects 1-ULP
    # read errors that would spuriously break the all-zero tolerance.
    old_samples = pd.read_csv(
        frozen_dir / "sample_manifest.csv", float_precision="round_trip"
    )
    old_manifest = pd.read_csv(frozen_dir / "run_manifest.csv")
    old_audit = json.loads(
        (frozen_dir / "peak_boundary_audit.json").read_text(encoding="utf-8")
    )
    live_manifest = pd.read_csv(
        Path(output_root) / "peak_boundary" / "run_manifest.csv"
    )
    mismatches: list[str] = []
    checks: dict[str, object] = {}

    new_ids = sorted(samples["sample_id"].astype(str))
    old_ids = sorted(old_samples["sample_id"].astype(str))
    checks["sample_count"] = len(new_ids)
    if new_ids != old_ids:
        mismatches.append("sample_ids_differ")

    def _pairs(frame: pd.DataFrame) -> set[tuple[str, str]]:
        passed = frame[frame["status"].astype(str).eq("pass")]
        return {
            (str(row["sample_id"]), str(row["branch"]))
            for _, row in passed.iterrows()
        }

    new_pairs = _pairs(live_manifest)
    checks["branch_association_count"] = len(new_pairs)
    if new_pairs != _pairs(old_manifest):
        mismatches.append("branch_associations_differ")
    if len(new_pairs) != EXPECTED_COMPLETIONS:
        mismatches.append("branch_association_count_not_240")

    new = samples.set_index(samples["sample_id"].astype(str)).sort_index()
    old = old_samples.set_index(
        old_samples["sample_id"].astype(str)
    ).sort_index()
    if new_ids == old_ids:
        actual_equal = (
            new["actual_schedule_sha256"].astype(str)
            == old["actual_schedule_sha256"].astype(str)
        ).all()
        checks["actual_schedule_sha_equal"] = bool(actual_equal)
        if not actual_equal:
            mismatches.append("actual_schedule_sha_differ")
        for column, key in DELTA_TOLERANCE_KEYS.items():
            atol = float(tolerance.get(key, 0.0))
            diff = np.abs(
                new[column].astype(float).to_numpy()
                - old[column].astype(float).to_numpy()
            )
            within = bool(np.all(diff <= atol))
            checks[f"{column}_max_abs_diff"] = float(diff.max()) if len(diff) else 0.0
            if not within:
                mismatches.append(f"{column}_exceeds_tolerance")
        for column in LABEL_COLUMNS:
            new_values = new[column].fillna("").astype(str)
            old_values = old[column].fillna("").astype(str)
            if not (new_values == old_values).all():
                mismatches.append(f"label_{column}_differ")

    same_state_all = bool(samples["state_hash_match"].astype(bool).all())
    readback_all = bool(samples["readback_ok"].astype(bool).all())
    duplicates = int(len(samples)) - int(
        samples["actual_schedule_sha256"].astype(str).nunique()
    )
    checks["same_state_all"] = same_state_all
    checks["readback_all"] = readback_all
    checks["actual_schedule_duplicates"] = duplicates
    if not same_state_all:
        mismatches.append("same_state_not_100pct")
    if not readback_all:
        mismatches.append("readback_not_100pct")
    if duplicates:
        mismatches.append("actual_schedule_duplicates_nonzero")

    checks["peak_degraded"] = int(audit.get("peak_degraded", -1))
    checks["pfv_safe_peak_hard_negative"] = int(
        audit.get("pfv_safe_peak_hard_negative", -1)
    )
    if audit.get("status") != "pass":
        mismatches.append("rebuilt_audit_not_pass")
    if int(audit.get("peak_degraded", -1)) != int(
        old_audit.get("peak_degraded", -2)
    ):
        mismatches.append("peak_degraded_count_differ")
    if int(audit.get("pfv_safe_peak_hard_negative", -1)) != int(
        old_audit.get("pfv_safe_peak_hard_negative", -2)
    ):
        mismatches.append("pfv_safe_peak_hard_negative_count_differ")

    return not mismatches, {"checks": checks, "mismatches": mismatches}


def restamp_peak_boundary_evidence(
    project_root: str | Path,
    output_root: str | Path,
    config: dict,
    *,
    config_path: str | Path | None = None,
) -> dict:
    """Full restamp flow. Returns a payload with status pass/blocked/scientific_fail."""
    project_root = Path(project_root)
    output_root = Path(output_root)
    started_at = time.time()
    archive_dir = locate_frozen_archive(output_root)
    if archive_dir is None:
        return {
            "status": "blocked",
            "reason": "frozen_archive_missing",
        }
    archive_manifest = json.loads(
        (archive_dir / "archive_manifest.json").read_text(encoding="utf-8")
    )
    old_code_sha = str(archive_manifest.get("old_code_sha", archive_dir.name))
    new_code_sha = working_code_sha(project_root)

    inputs_ok, input_report = verify_cached_inputs(
        project_root, output_root, config, archive_dir
    )
    if not inputs_ok:
        return {
            "status": "blocked",
            "reason": "cached_input_verification_failed",
            "input_report": input_report,
            "old_code_sha": old_code_sha,
            "new_code_sha": new_code_sha,
        }

    try:
        samples, rejected, audit = rebuild_scientific_outputs(
            project_root, output_root, config
        )
    except (KeyError, OSError, ValueError) as exc:
        return {
            "status": "blocked",
            "reason": f"rebuild_failed: {type(exc).__name__}: {exc}",
            "old_code_sha": old_code_sha,
            "new_code_sha": new_code_sha,
        }

    tolerance = dict(
        config.get("thresholds", {}).get("numerical_repeat_tolerance", {})
    )
    equal, comparison = compare_with_frozen(
        samples,
        audit,
        output_root,
        archive_dir,
        tolerance=tolerance,
    )

    restamp_dir = output_root / "peak_boundary" / "restamp"
    restamp_dir.mkdir(parents=True, exist_ok=True)
    if not equal:
        # Fail Closed: record the failure, never touch the canonical status.
        failure_path = restamp_dir / (
            f"restamp_failure_{time.strftime('%Y%m%d_%H%M%S')}.json"
        )
        atomic_write_json(
            failure_path,
            {
                "status": "scientific_fail",
                "old_code_sha": old_code_sha,
                "new_code_sha": new_code_sha,
                "comparison": comparison,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            },
        )
        return {
            "status": "scientific_fail",
            "reason": "scientific_outputs_not_equal",
            "comparison": comparison,
            "failure_report": str(failure_path),
            "old_code_sha": old_code_sha,
            "new_code_sha": new_code_sha,
        }

    old_status_path = archive_dir / "stage_status" / f"{CANONICAL_STAGE}.json"
    old_status = json.loads(old_status_path.read_text(encoding="utf-8"))
    stamp_path = restamp_dir / f"restamp_stamp_{new_code_sha[:16]}.json"
    latest_path = restamp_dir / "restamp_stamp.json"
    stamp = {
        "stage": RESTAMP_STAGE,
        "old_code_sha": old_code_sha,
        "new_code_sha": new_code_sha,
        "old_stamp_path": str(old_status_path),
        "new_stamp_path": str(stamp_path),
        "restamp_reason": (
            "pilot_subsystem_code_change_requires_revalidation_of_frozen_"
            "peak_boundary_evidence_from_cached_branch_outputs"
        ),
        "cached_branch_count": EXPECTED_COMPLETIONS,
        "new_swmm_run_count": 0,
        "scientific_outputs_equal": True,
        "supersedes": old_status.get("run_uuid", ""),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "input_report": input_report,
        "comparison": comparison,
        "rebuilt_audit": audit,
        "rejected_count": int(len(rejected)),
    }
    atomic_write_json(stamp_path, stamp)
    atomic_write_json(latest_path, stamp)

    # Atomically refresh the canonical AuditPeakBoundary status under the
    # new code SHA; the archived old stamp remains untouched forever.
    config_bytes = (
        Path(config_path).read_bytes() if config_path else b""
    )
    record = dict(old_status)
    record.update(
        {
            "run_uuid": str(uuid.uuid4()),
            "config_sha": hashlib.sha256(config_bytes).hexdigest()
            if config_bytes
            else old_status.get("config_sha", ""),
            "code_git_sha": new_code_sha,
            "started_at": float(started_at),
            "finished_at": float(time.time()),
            "evidence": audit,
            "restamped": True,
            "restamp_stamp": str(stamp_path),
            "supersedes_run_uuid": old_status.get("run_uuid", ""),
        }
    )
    status_path = (
        output_root / "audits" / "stage_status" / f"{CANONICAL_STAGE}.json"
    )
    atomic_write_json(status_path, record)
    atomic_write_json(
        status_path.with_name(f"{CANONICAL_STAGE}.completion.json"),
        {
            "stage": CANONICAL_STAGE,
            "run_uuid": record["run_uuid"],
            "input_sha": record.get("input_sha", ""),
            "status_sha256": hashlib.sha256(
                status_path.read_bytes()
            ).hexdigest(),
        },
    )
    return {
        "status": "pass",
        "stamp": str(stamp_path),
        "canonical_status": str(status_path),
        "old_code_sha": old_code_sha,
        "new_code_sha": new_code_sha,
        "comparison": comparison,
        "input_report": input_report,
        "peak_degraded": int(audit.get("peak_degraded", -1)),
        "pfv_safe_peak_hard_negative": int(
            audit.get("pfv_safe_peak_hard_negative", -1)
        ),
        "new_swmm_run_count": 0,
    }
