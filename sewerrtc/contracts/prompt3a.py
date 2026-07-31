from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from sewerrtc._project_root import PROJECT_ROOT
from sewerrtc.io.inp_parser import parse_links, parse_nodes, read_sections

RUN_TAG = "project6_pfvfirst_dualfallback_10min_v3"
OUT_ROOT = PROJECT_ROOT / "outputs" / RUN_TAG
INP_PATH = PROJECT_ROOT / "data" / "wuhan_v8_storage_retrofit.inp"
MANAGED_IDS_PATH = PROJECT_ROOT / "data" / "project6_v8_storage_retrofit_control_enabled_ids.txt"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str] | None = None) -> Path:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    return path


def managed_facility_ids() -> list[str]:
    ids: list[str] = []
    for raw in MANAGED_IDS_PATH.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if line:
            ids.append(line)
    return ids


def config_hash(config: str | Path) -> str:
    path = Path(config)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return sha256_file(path)


def script_hash(script_name: str) -> str:
    return sha256_file(PROJECT_ROOT / script_name)


def prompt2_paths() -> dict[str, Path]:
    return {
        "primary_gat_lock": OUT_ROOT / "gat" / "gat_primary_selection_lock.json",
        "independent_holdout_lock": OUT_ROOT / "gat" / "gat_independent_validation_lock.json",
        "independent_robustness_gate": OUT_ROOT / "gat" / "independent_holdout" / "sr0p15" / "gat_sr0p15_independent_robustness_gate.json",
        "prompt2_readiness_gate": OUT_ROOT / "gates" / "project6_prompt2_gat_readiness_gate.json",
        "state_manifest": OUT_ROOT / "state" / "augmented_state_sample_manifest.csv",
        "shape_audit": OUT_ROOT / "state" / "augmented_state_shape_audit.json",
        "causality_audit": OUT_ROOT / "state" / "augmented_state_causality_audit.csv",
        "missingness_audit": OUT_ROOT / "state" / "augmented_state_missingness_audit.csv",
        "holdout_manifest": OUT_ROOT / "gat" / "gat_independent_validation_manifest.csv",
        "holdout_cache_report": OUT_ROOT / "gat" / "independent_holdout" / "generated_trajectories" / "gat_independent_holdout_cache_report.json",
    }


def import_prompt2_artifacts(config: str | Path) -> tuple[int, dict[str, Any], list[Path]]:
    paths = prompt2_paths()
    checks: dict[str, Any] = {}
    for key, path in paths.items():
        checks[key] = {
            "path": str(path),
            "exists": path.exists(),
            "sha256": sha256_file(path) if path.exists() and path.is_file() else None,
        }
    readiness = read_json(paths["prompt2_readiness_gate"])
    robustness = read_json(paths["independent_robustness_gate"])
    shape = read_json(paths["shape_audit"])
    causality_rows = read_csv(paths["causality_audit"])
    checks["prompt2_readiness_status"] = readiness.get("status")
    checks["allowed_to_enter_prompt3a"] = readiness.get("allowed_to_enter_prompt3a")
    checks["independent_robustness_status"] = robustness.get("status")
    checks["shape_status"] = shape.get("status")
    checks["causality_rows"] = len(causality_rows)
    pass_ready = (
        readiness.get("status") == "pass"
        and readiness.get("allowed_to_enter_prompt3a") is True
        and robustness.get("status") == "pass"
        and bool(paths["primary_gat_lock"].exists())
        and bool(paths["independent_holdout_lock"].exists())
        and bool(paths["state_manifest"].exists())
        and bool(causality_rows)
    )
    manifest = {
        "status": "completed" if pass_ready else "blocked",
        "created_at": utc_now(),
        "config_hash": config_hash(config),
        "network_path": str(INP_PATH),
        "network_sha256": sha256_file(INP_PATH) if INP_PATH.exists() else None,
        "checks": checks,
        "round0_unlock_allowed": False,
    }
    out_manifest = OUT_ROOT / "contracts" / "prompt2_import_manifest.json"
    out_gate = OUT_ROOT / "gates" / "prompt3a_entry_gate.json"
    write_json(PROJECT_ROOT / "docs" / "contracts" / "project6_prompt2_import_contract.json", {
        "contract_version": "project6_prompt2_import_contract_v1",
        "requires_prompt2_readiness_pass": True,
        "requires_independent_holdout_robustness_pass": True,
        "requires_node_level_7frame_state_validation": True,
        "round0_unlock_allowed": False,
        "gat_holdout_use": "state_estimation_validation_only",
    })
    write_json(out_manifest, manifest)
    write_json(out_gate, {
        "status": "pass" if pass_ready else "blocked",
        "blocking_reasons": [] if pass_ready else [k for k, v in checks.items() if isinstance(v, dict) and not v.get("exists", False)],
        "allowed_to_continue_prompt3a": pass_ready,
        "round0_unlock_allowed": False,
        "created_at": utc_now(),
    })
    return (0 if pass_ready else 3), manifest, [out_manifest, out_gate, PROJECT_ROOT / "docs" / "contracts" / "project6_prompt2_import_contract.json"]


def static_physical_audit(config: str | Path) -> tuple[int, dict[str, Any], list[Path]]:
    sections = read_sections(INP_PATH)
    links = parse_links(sections)
    nodes = parse_nodes(sections)
    ids = managed_facility_ids()
    link_ids = set(links["link_id"].astype(str)) if not links.empty else set()
    node_ids = set(nodes["node_id"].astype(str)) if not nodes.empty else set()
    missing_facilities = [fid for fid in ids if fid not in link_ids]
    duplicate_ids = sorted({fid for fid in ids if ids.count(fid) > 1})
    sentinels = [line.strip() for line in (PROJECT_ROOT / "data" / "project6_v3_sentinel_nodes.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    sentinel_presence = {node: node in node_ids for node in sentinels}
    gate_pass = INP_PATH.exists() and len(ids) == 36 and not missing_facilities and not duplicate_ids and all(sentinel_presence.values())
    fatal_dir = OUT_ROOT / "fatal_audit"
    contracts_dir = OUT_ROOT / "contracts"
    report = {
        "status": "pass" if gate_pass else "blocked",
        "created_at": utc_now(),
        "config_hash": config_hash(config),
        "network_path": str(INP_PATH),
        "network_sha256": sha256_file(INP_PATH),
        "managed_facility_count": len(ids),
        "missing_facilities": missing_facilities,
        "duplicate_facility_ids": duplicate_ids,
        "node_count": len(node_ids),
        "sentinel_node_presence": sentinel_presence,
        "sentinel_contract_status": read_json(PROJECT_ROOT / "docs" / "contracts" / "sentinel_nodes_provenance.json").get("sentinel_contract_status"),
        "formal_safety_readiness": "blocked_pending_sentinel_thresholds_and_add350_bounds",
        "engineering_development_allowed": gate_pass,
        "round0_unlock_allowed": False,
    }
    files = [
        write_json(fatal_dir / "fatal_audit_report.json", report),
        write_json(fatal_dir / "engineering_development_gate.json", {"status": "pass" if gate_pass else "blocked", "checks": report}),
        write_json(fatal_dir / "formal_safety_readiness_gate.json", {"status": "blocked", "reason": report["formal_safety_readiness"]}),
        write_json(contracts_dir / "network_contract.json", {"inp_path": str(INP_PATH), "sha256": sha256_file(INP_PATH), "single_network_required": True}),
        write_json(contracts_dir / "contract_manifest.json", {"created_at": utc_now(), "network_contract": "network_contract.json", "kpi_contract": "docs/contracts/kpi_contract.json", "forecast_contract": "docs/contracts/forecast_contract.json"}),
    ]
    return (0 if gate_pass else 3), report, files

