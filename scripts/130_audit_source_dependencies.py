from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def path_status(paths: Iterable[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for raw in paths:
        path = Path(raw)
        if not path.is_absolute():
            path = ROOT / path
        rows.append(
            {
                "path": str(path),
                "exists": path.exists(),
                "sha256": sha256_file(path),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit V3 source dependencies without running SWMM or training.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    cfg = load_yaml(config_path)

    contracts = [
        "AGENTS.md",
        "docs/contracts/PROJECT6_PFVFIRST_DUALFALLBACK_10MIN_V3.md",
        "docs/contracts/kpi_contract.json",
        "docs/contracts/forecast_contract.json",
        "docs/contracts/execution_status.schema.json",
        "docs/contracts/sentinel_nodes_provenance.json",
        "docs/contracts/facility_semantics_contract.json",
        "docs/plans/PROJECT6_PFVFIRST_DUALFALLBACK_10MIN_V3.md",
        "docs/runbooks/PROJECT6_PFVFIRST_DUALFALLBACK_V3_RUNBOOK.md",
    ]
    scripts = [
        "scripts/125_audit_pfvfirst_dualfallback_assets.py",
        "scripts/126_plan_information_coverage_cases.py",
        "scripts/127_build_same_state_dualfallback_dataset.py",
        "scripts/128_train_pfvfirst_effect_model.py",
        "scripts/129_gate_pfvfirst_model.py",
        "scripts/130_audit_source_dependencies.py",
        "scripts/131_build_event_catalog.py",
        "scripts/132_build_checkpoint_catalog.py",
        "scripts/133_run_internal_pfv_opportunity_scan.py",
        "scripts/project6_runs/RUN_PROJECT6_PFVFIRST_DUALFALLBACK_10MIN_V3.ps1",
    ]
    modules = [
        "sewerrtc/evaluation/kpi_contract.py",
        "sewerrtc/data/coverage_contract.py",
        "sewerrtc/data/forecast_contract.py",
        "sewerrtc/execution/status_contract.py",
        "sewerrtc/control/native_rule_audit.py",
        "sewerrtc/data/event_catalog_contract.py",
        "sewerrtc/data/checkpoint_catalog_contract.py",
        "sewerrtc/data/opportunity_scan_contract.py",
        "sewerrtc/control/pfvfirst_dualfallback.py",
    ]

    enabled_stages = [
        "Status",
        "Audit",
        "InitCoverageSchema",
    ]
    disabled = [
        "FatalAudit",
        "AuditNativeRules",
        "AuditFallbacks",
        "RegisterGAT",
        "AuditGAT",
        "BuildStateFeatures",
        "BuildEventCatalog",
        "BuildCheckpointCatalog",
        "StateCloneTest",
        "RunInternalPFVOpportunityScan",
        "DryRunRound0",
        "GenerateRound0",
        "BuildDataset",
        "TrainPilot",
        "RunPolicyShiftAudit",
        "PlanRound1",
        "GenerateRound1",
        "PlanRound2",
        "GenerateRound2",
        "TrainFinal",
        "MinimalGate",
        "OptimizerExploitationAudit",
        "DecisionShadowGate",
        "BuildMPC",
        "RunMPCDryRun",
        "RunSmoke",
        "CalibrationA",
        "LockedValidationB",
        "PolicyLock",
        "FormalBlind",
    ]

    report = {
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "contracts": path_status(contracts),
        "scripts": path_status(scripts),
        "modules": path_status(modules),
        "reusable_modules": [
            "INP parsing and static asset table generation",
            "KPI public contract implementation",
            "coverage schema and manifest validation",
            "forecast schema validation",
            "PFV-first dual-fallback controller dataclasses",
        ],
        "modules_requiring_real_implementation_before_closed_loop": [
            "native-rule audit",
            "executable passive fallback simulator",
            "GAT metadata compatibility verifier",
            "state clone and hot-start equivalence test",
            "same-state SWMM case generator",
            "OOD and optimizer exploitation audits",
            "receding-horizon MPC execution",
            "Smoke/Calibration/Formal runners",
        ],
        "enabled_runner_stages": enabled_stages,
        "disabled_or_unimplemented_stage_groups": disabled,
        "status": "source_dependency_audit_only_no_runtime_validation",
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
