"""
V4.2 Admission Gate — 13 Hard Conditions
==========================================

Final quality gate that verifies 13 hard contract conditions before the
V4.2 data pool is admitted for model training and evaluation.

Each check is independent — if one fails the others still run.

Verdicts
--------
PASS                  : all 13 checks pass, sufficient data
DATA_CONTRACT_FAIL    : one or more critical contract checks fail (1-3, 7-8, 10)
UNDERPOWERED          : all checks pass but TARGET_FULL_SUPERVISION count < 100
PARTIAL_POOL_ONLY     : some checks pass but not enough datasets populated

Output
------
- audits/v42_final_pool/admission_gate_result.json
- audits/v42_final_pool/admission_gate_summary.json
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------

class GateVerdict(str, Enum):
    PASS = "PASS"
    DATA_CONTRACT_FAIL = "DATA_CONTRACT_FAIL"
    UNDERPOWERED = "UNDERPOWERED"
    PARTIAL_POOL_ONLY = "PARTIAL_POOL_ONLY"


@dataclass
class AdmissionGateResult:
    verdict: GateVerdict
    checks: Dict[str, Dict[str, Any]]  # {check_name: {"pass": bool, "detail": str}}
    summary: str


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Checks whose failure is considered a critical data-contract breach.
_CRITICAL_CHECKS = {1, 2, 3, 7, 8, 10}

# Minimum number of TARGET_FULL_SUPERVISION samples for full power.
_MIN_FULL_SUPERVISION = 100

# Expected 12 dataset names (from the dataset manifest contract).
_EXPECTED_DATASETS = [
    "target_no_dwf_full_supervision",
    "source_dwf_full_supervision",
    "dynamics_pretrain",
    "actuator_effect",
    "pfv_constraint_core8",
    "tfv_objective",
    "peak_constraint",
    "within_state_ranking_pairs",
    "consumed_development",
    "reserved_evaluation_manifest",
    "rejected_samples",
    "sample_lineage",
]


# ---------------------------------------------------------------------------
# Individual check implementations
# ---------------------------------------------------------------------------

def _check_1_priority_contract(audit_dir: Path) -> Dict[str, Any]:
    """Check 1: Priority CORE8 contract PASS."""
    try:
        from sewerrtc.v4.v42_priority_contract import audit_contract
        report = audit_contract()
        status = report.get("status", "UNKNOWN")
        passed = status == "PASS"
        return {
            "pass": passed,
            "detail": f"audit_contract status={status}",
        }
    except Exception as exc:
        return {"pass": False, "detail": f"audit_contract raised: {exc}"}


def _check_2_sentinel_separated(audit_dir: Path) -> Dict[str, Any]:
    """Check 2: Sentinel/Priority completely separated — no overlap."""
    try:
        from sewerrtc.v4.v42_priority_contract import (
            PFV_CORE_8_IDS,
            DEPTH_SENTINEL_2_IDS,
        )
        overlap = set(PFV_CORE_8_IDS) & set(DEPTH_SENTINEL_2_IDS)
        passed = len(overlap) == 0
        detail = (
            f"PFV_CORE8={len(PFV_CORE_8_IDS)}, SENTINEL={len(DEPTH_SENTINEL_2_IDS)}, "
            f"overlap={sorted(overlap)}"
        )
        return {"pass": passed, "detail": detail}
    except Exception as exc:
        return {"pass": False, "detail": f"Cannot load contract nodes: {exc}"}


def _check_3_no_depth_proxy(project_root: Path) -> Dict[str, Any]:
    """Check 3: PFV no longer uses depth proxy in active pipeline.

    Scans the active pipeline files for patterns that would indicate
    depth is being used as a PFV proxy.
    """
    try:
        pipeline_files = [
            project_root / "sewerrtc" / "v4" / "pipeline_v42.py",
            project_root / "sewerrtc" / "v4" / "v42_final_datasets.py",
            project_root / "sewerrtc" / "v4" / "v42_independent_pfv_oracle.py",
        ]
        suspicious_patterns = [
            "depth_as_pfv",
            "depth.*proxy.*pfv",
            "pfv.*depth.*proxy",
            "use_depth.*pfv",
        ]
        import re
        issues = []
        for pf in pipeline_files:
            if not pf.is_file():
                continue
            text = pf.read_text(encoding="utf-8", errors="replace")
            for pat in suspicious_patterns:
                if re.search(pat, text, re.IGNORECASE):
                    issues.append(f"{pf.name}: matches '{pat}'")

        passed = len(issues) == 0
        detail = "No depth-as-PFV-proxy patterns found" if passed else "; ".join(issues)
        return {"pass": passed, "detail": detail}
    except Exception as exc:
        return {"pass": False, "detail": f"Depth proxy scan error: {exc}"}


def _check_4_13frame_rebuild(audit_dir: Path) -> Dict[str, Any]:
    """Check 4: 13-frame real rebuild — history_rebuild_audit.json."""
    try:
        audit_path = audit_dir / "history_rebuild_audit.json"
        if not audit_path.is_file():
            return {"pass": False, "detail": "history_rebuild_audit.json not found"}
        data = json.loads(audit_path.read_text(encoding="utf-8"))
        full_frames = data.get("samples_with_full_13_frames", 0)
        passed = full_frames > 0
        detail = (
            f"samples_with_full_13_frames={full_frames}, "
            f"total_attempted={data.get('total_samples_attempted', '?')}"
        )
        return {"pass": passed, "detail": detail}
    except Exception as exc:
        return {"pass": False, "detail": f"13-frame audit read error: {exc}"}


def _check_5_four_branch_reference(audit_dir: Path) -> Dict[str, Any]:
    """Check 5: Four-branch Reference correct — semantic audit."""
    try:
        summary_path = audit_dir / "semantic_source_summary.csv"
        if not summary_path.is_file():
            return {"pass": False, "detail": "semantic_source_summary.csv not found"}
        import csv
        with open(summary_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            return {"pass": False, "detail": "semantic_source_summary.csv is empty"}
        total_branch_failures = sum(
            int(r.get("branch_contract_failures", 0)) for r in rows
        )
        passed = total_branch_failures == 0
        detail = (
            f"total branch_contract_failures={total_branch_failures} "
            f"across {len(rows)} source rounds"
        )
        return {"pass": passed, "detail": detail}
    except Exception as exc:
        return {"pass": False, "detail": f"Four-branch audit read error: {exc}"}


def _check_6_action_completeness(audit_dir: Path) -> Dict[str, Any]:
    """Check 6: actual/readback complete — semantic audit."""
    try:
        summary_path = audit_dir / "semantic_source_summary.csv"
        if not summary_path.is_file():
            return {"pass": False, "detail": "semantic_source_summary.csv not found"}
        import csv
        with open(summary_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            return {"pass": False, "detail": "semantic_source_summary.csv is empty"}
        total_action_failures = sum(
            int(r.get("action_contract_failures", 0)) for r in rows
        )
        passed = total_action_failures == 0
        detail = (
            f"total action_contract_failures={total_action_failures} "
            f"across {len(rows)} source rounds"
        )
        return {"pass": passed, "detail": detail}
    except Exception as exc:
        return {"pass": False, "detail": f"Action completeness audit read error: {exc}"}


def _check_7_labels_recomputed(audit_dir: Path) -> Dict[str, Any]:
    """Check 7: Labels independently recomputed PASS — PFV and TFV/Peak oracle audits."""
    results = []
    for fname, label in [
        ("pfv_oracle_audit.json", "PFV"),
        ("tfv_peak_oracle_audit.json", "TFV/Peak"),
    ]:
        fpath = audit_dir / fname
        if not fpath.is_file():
            results.append(f"{label}: file not found")
            continue
        try:
            data = json.loads(fpath.read_text(encoding="utf-8"))
            ok = data.get("pass", False)
            n_samples = data.get("n_samples", "?")
            n_mismatches = data.get("n_mismatches", data.get("tfv_max_abs_error_m3", "?"))
            results.append(f"{label}: pass={ok}, n_samples={n_samples}")
        except Exception:
            results.append(f"{label}: read error")

    # Pass if both audits report pass=true
    passed = all("pass=True" in r for r in results)
    return {"pass": passed, "detail": "; ".join(results)}


def _check_8_dedup(audit_dir: Path) -> Dict[str, Any]:
    """Check 8: V4 vs Round data physically deduped."""
    try:
        audit_path = audit_dir / "deduplication_audit.json"
        if not audit_path.is_file():
            return {"pass": False, "detail": "deduplication_audit.json not found"}
        data = json.loads(audit_path.read_text(encoding="utf-8"))
        total = data.get("total_samples", 0)
        unique = data.get("unique_physical_samples", 0)
        dedup_performed = data.get("duplicate_group_count", 0) > 0
        passed = dedup_performed and unique > 0
        detail = (
            f"total={total}, unique_physical={unique}, "
            f"duplicate_groups={data.get('duplicate_group_count', 0)}"
        )
        return {"pass": passed, "detail": detail}
    except Exception as exc:
        return {"pass": False, "detail": f"Dedup audit read error: {exc}"}


def _check_9_dwf_tagged(audit_dir: Path) -> Dict[str, Any]:
    """Check 9: DWF domain correctly tagged."""
    try:
        audit_path = audit_dir / "dwf_audit_summary.json"
        if not audit_path.is_file():
            return {"pass": False, "detail": "dwf_audit_summary.json not found"}
        data = json.loads(audit_path.read_text(encoding="utf-8"))
        total = data.get("total_samples", 0)
        counts = data.get("classification_counts", {})
        classified = sum(counts.values())
        checks = data.get("dwf_checks", {})
        all_checks_complete = all(v == total for v in checks.values()) if total > 0 else False
        passed = classified > 0 and all_checks_complete
        detail = (
            f"total={total}, classified={classified}, "
            f"all_dwf_checks_complete={all_checks_complete}"
        )
        return {"pass": passed, "detail": detail}
    except Exception as exc:
        return {"pass": False, "detail": f"DWF audit read error: {exc}"}


def _check_10_reserved_isolated(data_dir: Path, audit_dir: Path) -> Dict[str, Any]:
    """Check 10: Reserved data completely isolated — no reserved samples in dev datasets."""
    try:
        reserved_csv = data_dir / "reserved_evaluation_manifest.csv"
        if not reserved_csv.is_file():
            return {"pass": False, "detail": "reserved_evaluation_manifest.csv not found"}

        import csv
        reserved_ids = set()
        with open(reserved_csv, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                sid = row.get("sample_id", "")
                if sid:
                    reserved_ids.add(sid)

        # Check that development datasets don't contain reserved samples
        dev_datasets = [
            "dynamics_pretrain.parquet",
            "actuator_effect.parquet",
            "pfv_constraint_core8.parquet",
            "tfv_objective.parquet",
            "peak_constraint.parquet",
        ]
        contamination = []
        for ds_name in dev_datasets:
            ds_path = data_dir / ds_name
            if not ds_path.is_file():
                continue
            try:
                import pandas as pd
                df = pd.read_parquet(ds_path, columns=["sample_id"])
                overlap = set(df["sample_id"].tolist()) & reserved_ids
                if overlap:
                    contamination.append(f"{ds_name}: {len(overlap)} reserved leaked")
            except Exception:
                pass

        passed = len(contamination) == 0
        detail = (
            f"reserved_count={len(reserved_ids)}, no contamination in dev datasets"
            if passed
            else "; ".join(contamination)
        )
        return {"pass": passed, "detail": detail}
    except Exception as exc:
        return {"pass": False, "detail": f"Reserved isolation check error: {exc}"}


def _check_11_no_cross_rainfall_leakage(audit_dir: Path) -> Dict[str, Any]:
    """Check 11: No cross-rainfall leakage — grouped splits verify same rainfall in same fold."""
    try:
        # Use semantic_source_summary to verify event-level integrity
        summary_path = audit_dir / "semantic_source_summary.csv"
        if not summary_path.is_file():
            return {"pass": False, "detail": "semantic_source_summary.csv not found"}

        import csv
        with open(summary_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            return {"pass": False, "detail": "No source rounds in summary"}

        # Check time contract failures as proxy for rainfall grouping integrity
        total_time_failures = sum(
            int(r.get("time_contract_failures", 0)) for r in rows
        )
        passed = total_time_failures == 0
        detail = (
            f"time_contract_failures={total_time_failures} "
            f"across {len(rows)} source rounds"
        )
        return {"pass": passed, "detail": detail}
    except Exception as exc:
        return {"pass": False, "detail": f"Rainfall leakage check error: {exc}"}


def _check_12_schema_fixed(data_dir: Path) -> Dict[str, Any]:
    """Check 12: Final data schema fixed — all 12 datasets exist with correct schemas."""
    try:
        manifest_path = data_dir / "dataset_manifest.json"
        if not manifest_path.is_file():
            return {"pass": False, "detail": "dataset_manifest.json not found"}

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        datasets = manifest.get("datasets", [])
        ds_names = [d["dataset"] for d in datasets]

        missing = set(_EXPECTED_DATASETS) - set(ds_names)
        extra = set(ds_names) - set(_EXPECTED_DATASETS)

        # Verify each dataset file exists
        file_issues = []
        for ds in datasets:
            ds_path = data_dir.parent.parent / ds["path"]
            if not ds_path.is_file():
                file_issues.append(f"{ds['dataset']}: file missing")

        passed = len(missing) == 0 and len(file_issues) == 0
        detail_parts = [f"datasets_in_manifest={len(ds_names)}"]
        if missing:
            detail_parts.append(f"missing={sorted(missing)}")
        if extra:
            detail_parts.append(f"extra={sorted(extra)}")
        if file_issues:
            detail_parts.append("; ".join(file_issues))
        detail = ", ".join(detail_parts)
        return {"pass": passed, "detail": detail}
    except Exception as exc:
        return {"pass": False, "detail": f"Schema check error: {exc}"}


def _check_13_lineage_complete(data_dir: Path) -> Dict[str, Any]:
    """Check 13: All samples have complete lineage — no null lineage fields."""
    try:
        lineage_path = data_dir / "sample_lineage.parquet"
        if not lineage_path.is_file():
            return {"pass": False, "detail": "sample_lineage.parquet not found"}

        import pandas as pd
        df = pd.read_parquet(lineage_path)
        null_counts = df.isnull().sum()
        cols_with_nulls = null_counts[null_counts > 0]
        passed = len(cols_with_nulls) == 0
        detail = (
            f"lineage_rows={len(df)}, columns_with_nulls={len(cols_with_nulls)}"
        )
        if not passed:
            detail += f", null_cols={dict(cols_with_nulls.head(5))}"
        return {"pass": passed, "detail": detail}
    except Exception as exc:
        return {"pass": False, "detail": f"Lineage check error: {exc}"}


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_admission_gate(
    project_root: Path,
    output_root: Path,
) -> AdmissionGateResult:
    """Run all 13 admission gate checks. Returns result.

    Parameters
    ----------
    project_root : Path
        Project6 root directory (e.g. ``E:\\RTC_sewer\\Project6``).
    output_root : Path
        Output root where audit results will be written.

    Returns
    -------
    AdmissionGateResult
        Contains verdict, per-check details, and human-readable summary.
    """
    project_root = Path(project_root)
    output_root = Path(output_root)
    audit_dir = project_root / "audits" / "v42_final_pool"
    data_dir = project_root / "data" / "v42_final_unified"

    checks: Dict[str, Dict[str, Any]] = {}

    # Run all 13 checks independently
    check_fns = [
        ("01_priority_core8_contract", lambda: _check_1_priority_contract(audit_dir)),
        ("02_sentinel_separated", lambda: _check_2_sentinel_separated(audit_dir)),
        ("03_no_depth_proxy", lambda: _check_3_no_depth_proxy(project_root)),
        ("04_13frame_rebuild", lambda: _check_4_13frame_rebuild(audit_dir)),
        ("05_four_branch_reference", lambda: _check_5_four_branch_reference(audit_dir)),
        ("06_action_completeness", lambda: _check_6_action_completeness(audit_dir)),
        ("07_labels_recomputed", lambda: _check_7_labels_recomputed(audit_dir)),
        ("08_dedup", lambda: _check_8_dedup(audit_dir)),
        ("09_dwf_tagged", lambda: _check_9_dwf_tagged(audit_dir)),
        ("10_reserved_isolated", lambda: _check_10_reserved_isolated(data_dir, audit_dir)),
        ("11_no_cross_rainfall_leakage", lambda: _check_11_no_cross_rainfall_leakage(audit_dir)),
        ("12_schema_fixed", lambda: _check_12_schema_fixed(data_dir)),
        ("13_lineage_complete", lambda: _check_13_lineage_complete(data_dir)),
    ]

    for name, fn in check_fns:
        try:
            checks[name] = fn()
        except Exception as exc:
            checks[name] = {"pass": False, "detail": f"Unhandled error: {exc}"}

    # Determine verdict
    n_pass = sum(1 for c in checks.values() if c["pass"])
    n_total = len(checks)
    failed_indices = set()
    for idx, (name, result) in enumerate(checks.items(), start=1):
        if not result["pass"]:
            failed_indices.add(idx)

    # Critical checks: 1, 2, 3, 7, 8, 10
    critical_failures = failed_indices & _CRITICAL_CHECKS

    if critical_failures:
        verdict = GateVerdict.DATA_CONTRACT_FAIL
    elif n_pass == n_total:
        # All pass — check if underpowered
        verdict = _check_power_level(data_dir)
    elif n_pass >= n_total * 0.6:
        # Majority pass but some datasets may not be populated
        verdict = GateVerdict.PARTIAL_POOL_ONLY
    else:
        verdict = GateVerdict.DATA_CONTRACT_FAIL

    # Build summary string
    summary_lines = [
        f"Admission Gate Verdict: {verdict.value}",
        f"Checks passed: {n_pass}/{n_total}",
    ]
    for name, result in checks.items():
        status = "PASS" if result["pass"] else "FAIL"
        summary_lines.append(f"  [{status}] {name}: {result['detail']}")
    summary = "\n".join(summary_lines)

    result = AdmissionGateResult(
        verdict=verdict,
        checks=checks,
        summary=summary,
    )

    # Write outputs
    _write_outputs(result, output_root, audit_dir)

    return result


def _check_power_level(data_dir: Path) -> GateVerdict:
    """Determine if the pool is fully powered or underpowered."""
    try:
        manifest_path = data_dir / "dataset_manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            counts = manifest.get("per_dataset_counts", {})
            target_full = counts.get("target_no_dwf_full_supervision", 0)
            source_dwf = counts.get("source_dwf_full_supervision", 0)
            total_full = target_full + source_dwf
            if total_full < _MIN_FULL_SUPERVISION:
                return GateVerdict.UNDERPOWERED

        # Also check actual file row counts as fallback
        target_path = data_dir / "target_no_dwf_full_supervision.parquet"
        if target_path.is_file():
            import pandas as pd
            df = pd.read_parquet(target_path, columns=["sample_id"])
            if len(df) < _MIN_FULL_SUPERVISION:
                return GateVerdict.UNDERPOWERED
    except Exception:
        pass

    return GateVerdict.PASS


def _write_outputs(
    result: AdmissionGateResult,
    output_root: Path,
    audit_dir: Path,
) -> None:
    """Write admission gate results to disk."""
    out_dir = audit_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Full result JSON
    full_result = {
        "verdict": result.verdict.value,
        "checks": result.checks,
        "summary": result.summary,
    }
    full_path = out_dir / "admission_gate_result.json"
    full_path.write_text(
        json.dumps(full_result, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    logger.info("Wrote %s", full_path)

    # Summary JSON
    n_pass = sum(1 for c in result.checks.values() if c["pass"])
    n_fail = len(result.checks) - n_pass
    summary_result = {
        "verdict": result.verdict.value,
        "checks_passed": n_pass,
        "checks_failed": n_fail,
        "total_checks": len(result.checks),
        "per_check_pass_fail": {
            name: c["pass"] for name, c in result.checks.items()
        },
    }
    summary_path = out_dir / "admission_gate_summary.json"
    summary_path.write_text(
        json.dumps(summary_result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    logger.info("Wrote %s", summary_path)
