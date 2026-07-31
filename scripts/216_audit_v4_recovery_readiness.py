"""Gate 0 — Read-only audit of assets, processes, and existing outputs.

Produces 6 files under outputs/project6_dual_reference_v4/recovery_audit/:
  - gate0_asset_inventory.json
  - gate0_execution_chain.json
  - gate0_reference_semantics_audit.json
  - gate0_manifest_provenance.json
  - gate0_label_computation_graph.json
  - gate0_known_defects.md

This script is STRICTLY read-only: it does not run SWMM, does not modify
any file on disk (other than the audit outputs), and does not alter any
existing manifest or model.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(path: Path) -> str:
    if not path.exists() or not path.is_file():
        return "MISSING"
    h = hashlib.sha256()
    try:
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return "UNREADABLE"


def _sha256_csv(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "sha256": "MISSING", "rows": 0, "columns": []}
    try:
        df = pd.read_csv(path, nrows=0)
        df_full = pd.read_csv(path, low_memory=False)
        return {
            "exists": True,
            "path": str(path),
            "sha256": _sha256(path),
            "rows": int(len(df_full)),
            "columns": list(df.columns),
            "size_bytes": int(path.stat().st_size),
        }
    except Exception as exc:  # noqa: BLE001
        return {"exists": True, "path": str(path), "sha256": _sha256(path),
                "rows": 0, "columns": [], "read_error": repr(exc)}


def _write_json(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Asset inventory
# ---------------------------------------------------------------------------

def _asset_inventory() -> dict[str, Any]:
    network = ROOT / "data" / "wuhan_v8_storage_retrofit.inp"
    ids = ROOT / "data" / "project6_v8_storage_retrofit_control_enabled_ids.txt"
    # Configs
    cfg_dir = ROOT / "configs"
    cfgs = sorted(cfg_dir.glob("*.yaml")) if cfg_dir.exists() else []
    # sewerrtc/prompt3/*.py
    p3_dir = ROOT / "sewerrtc" / "prompt3"
    p3_pys = sorted(p3_dir.glob("*.py")) if p3_dir.exists() else []
    # Key scripts
    key_scripts = [
        ROOT / "scripts" / "205_prompt3_v4.py",
        ROOT / "scripts" / "103_repair_peak_label_semantics.py",
    ]
    key_scripts += sorted(ROOT.glob("scripts/160_*.py"))
    key_scripts += sorted(ROOT.glob("scripts/161_*.py"))
    key_scripts += sorted(ROOT.glob("scripts/162_*.py"))
    key_scripts += sorted(ROOT.glob("scripts/163_*.py"))
    key_scripts += sorted(ROOT.glob("scripts/164_*.py"))
    key_scripts += sorted(ROOT.glob("scripts/165_*.py"))

    inventory: dict[str, Any] = {
        "network": {"path": str(network), "sha256": _sha256(network),
                    "exists": network.exists()},
        "managed_facility_ids": {"path": str(ids), "sha256": _sha256(ids),
                                 "exists": ids.exists()},
        "configs": {p.name: {"path": str(p), "sha256": _sha256(p)} for p in cfgs},
        "sewerrtc_prompt3": {p.name: {"path": str(p), "sha256": _sha256(p)} for p in p3_pys},
        "key_scripts": {p.name: {"path": str(p), "sha256": _sha256(p)} for p in key_scripts},
    }
    if ids.exists():
        try:
            lines = [ln.strip() for ln in ids.read_text(encoding="utf-8").splitlines() if ln.strip() and not ln.startswith("#")]
            inventory["managed_facility_ids"]["count"] = len(lines)
            inventory["managed_facility_ids"]["ids"] = lines
        except OSError:
            pass
    return inventory


# ---------------------------------------------------------------------------
# Execution chain (static source tracing, no runtime)
# ---------------------------------------------------------------------------

def _execution_chain() -> dict[str, Any]:
    """Static description of the Build/Train/Gate/Smoke call paths.

    The audit does not execute any of these paths; it records the code
    references that define them so a reviewer can verify the chain.
    """
    orchestrator = ROOT / "scripts" / "205_prompt3_v4.py"
    aug1 = ROOT / "sewerrtc" / "prompt3" / "action_effect_v4_aug1.py"
    base = ROOT / "sewerrtc" / "prompt3" / "action_effect_v4.py"
    runner = ROOT / "sewerrtc" / "simulation" / "pyswmm_runner.py"
    return {
        "BuildV4AugmentedDataset": {
            "entry": "scripts/205_prompt3_v4.py:main() -> stage='BuildV4AugmentedDataset'",
            "library": "sewerrtc/prompt3/action_effect_v4_aug1.py:build_v4_augmented_dataset()",
            "worker": "sewerrtc/prompt3/action_effect_v4_aug1.py:_run_group() -> _run_branch()",
            "swmm": "sewerrtc/simulation/pyswmm_runner.py:run_swmm_no_control_action_ablation()",
            "inp_mutation": "sewerrtc/prompt3/action_effect_v4_aug1.py:mutate_inp_for_event(..., strip_controls=True)",
            "note": "strip_controls=True disables native [CONTROLS] in the case INP. This is the root cause of 'Internal is not dynamic' in the aug1 generation.",
        },
        "TrainV4Aug1": {
            "entry": "scripts/205_prompt3_v4.py:main() -> stage='TrainV4Aug1'",
            "library": "sewerrtc/prompt3/action_effect_v4_aug1.py:train_v4_aug1()",
            "note": "Residual head trained on aug1-layer rows only (post-fix); reference head trained on all rows.",
        },
        "EvaluateV4Aug1ModelGate": {
            "entry": "scripts/205_prompt3_v4.py:main() -> stage='EvaluateV4Aug1ModelGate'",
            "library": "sewerrtc/prompt3/action_effect_v4_aug1.py:evaluate_v4_aug1_model_gate()",
            "note": "Advisory heads (delta_TFV_H120_vs_internal, delta_peak_H120_vs_internal) recorded but not blocking.",
        },
        "RunClosedLoopSmokeV4": {
            "entry": "scripts/205_prompt3_v4.py:run_smoke()",
            "subprocess": "scripts/08_run_closed_loop.py with --proposed-controller proposed_dual_reference_v4 --action-effect-model <aug1 model>",
            "note": "Heavy SWMM subprocess; not executed by this audit.",
        },
        "EvaluateClosedLoopSmokeV4": {
            "entry": "scripts/205_prompt3_v4.py:evaluate_smoke()",
            "note": "Reads proposed_results.csv + baseline_results.csv + controller_history.csv.",
        },
    }


# ---------------------------------------------------------------------------
# Reference semantics audit (static code tracing)
# ---------------------------------------------------------------------------

def _reference_semantics() -> dict[str, Any]:
    """Per-branch code path that builds the override sequence.

    Source of truth: sewerrtc/prompt3/action_effect_v4_aug1.py, function
    _reference_sequences() at lines 417-428.
    """
    return {
        "no_control": {
            "sequence_builder": "_constant_sequence(actuator_ids, {aid: 1.0 for aid in actuator_ids}, n_steps)",
            "policy_id_passed_to_runner": "no_control",
            "override_target_sequence": "all actuators = 1.0 for every step",
            "native_controls_enabled": False,
            "case_INP_strip_controls": True,
            "contract_note": "Name suggests 'no control' but actual semantics is 'all managed facilities set to 1.0 (open/bypassed)'. Contract must verify this matches the intended No-control definition.",
        },
        "passive_anchor": {
            "sequence_builder": "_sequence_from_detail(passive_detail, actuator_ids, checkpoint_min, n_steps)",
            "policy_id_passed_to_runner": "passive_anchor",
            "override_target_sequence": "passive baseline trajectory's a: columns at checkpoint_min, repeated for n_steps",
            "native_controls_enabled": False,
            "case_INP_strip_controls": True,
            "contract_note": "If the passive baseline has converged to all-1.0 by checkpoint_min (>= 40 min), this sequence equals the no_control sequence. Empirically confirmed in aug1 generation.",
        },
        "internal_current_action": {
            "sequence_builder": "_sequence_from_detail(internal_detail, actuator_ids, checkpoint_min, n_steps)",
            "policy_id_passed_to_runner": "internal_current_action",
            "override_target_sequence": "internal_rules baseline trajectory's a: columns at checkpoint_min, repeated for n_steps",
            "native_controls_enabled": False,
            "case_INP_strip_controls": True,
            "contract_note": "This is a FROZEN snapshot of the internal rules' past actions, NOT the dynamic native rules running. Must be renamed to hold_internal_snapshot.",
        },
        "hold_previous": {
            "sequence_builder": "_constant_sequence(actuator_ids, hold_map, n_steps) where hold_map = _settings_at(internal_detail, checkpoint_min, actuator_ids)",
            "policy_id_passed_to_runner": "hold_previous",
            "override_target_sequence": "internal_rules trajectory's a: columns at checkpoint_min, held constant for n_steps",
            "native_controls_enabled": False,
            "case_INP_strip_controls": True,
            "contract_note": "Semantically identical to internal_current_action in the current code (both read from internal_detail at checkpoint_min). Contract must verify this is intentional.",
        },
        "candidate": {
            "sequence_builder": "_candidate_sequence(plan_row, internal_detail, ...) + _ensure_candidate_differs(...)",
            "policy_id_passed_to_runner": "candidate",
            "override_target_sequence": "perturbation of internal checkpoint action, per plan_row (action_type/magnitude/direction/k_value)",
            "native_controls_enabled": False,
            "case_INP_strip_controls": True,
            "contract_note": "Candidate overrides the entire post-checkpoint window with the planned perturbation + fallback.",
        },
    }


# ---------------------------------------------------------------------------
# Manifest provenance
# ---------------------------------------------------------------------------

def _manifest_provenance() -> dict[str, Any]:
    out = ROOT / "outputs" / "project6_dual_reference_v4"
    manifests = {
        "base_v4": out / "action_effect_dataset_v4" / "v4_dataset_manifest.csv",
        "aug1_generation": out / "dual_reference_aug1" / "v4_aug1_generation_manifest.csv",
        "aug1_dataset": out / "dual_reference_aug1" / "v4_aug1_dataset_manifest.csv",
        "aug1_failed": out / "dual_reference_aug1" / "v4_aug1_generation_failed.csv",
        "aug1_case_plan": out / "dual_reference_aug1" / "v4_aug1_case_plan.csv",
    }
    result = {}
    for key, path in manifests.items():
        info = _sha256_csv(path)
        result[key] = info
    # Cross-hash: aug1_dataset should be derived from base_v4 + aug1_generation
    result["derivation_chain"] = [
        "base_v4 + aug1_generation -> aug1_dataset (via build_v4_augmented_dataset)",
    ]
    return result


# ---------------------------------------------------------------------------
# Label computation graph
# ---------------------------------------------------------------------------

def _label_computation_graph() -> dict[str, Any]:
    """Symbolic expression for each of the 14 labels, traced to branch KPIs."""
    return {
        "REFERENCE_LABELS": {
            "no_control_PFV_H120":      "PFV(no_control_branch, window=H120)",
            "passive_PFV_H120":         "PFV(passive_anchor_branch, window=H120)",
            "internal_PFV_H120":        "PFV(internal_current_action_branch, window=H120)",
            "internal_TFV_H120":        "TFV(internal_current_action_branch, window=H120)",
            "internal_peak_H120":       "peak_TFV_rate(internal_current_action_branch, window=H120)",
            "no_control_PFV_full":      "PFV(no_control_branch, window=full_event)",
            "passive_PFV_full":         "PFV(passive_anchor_branch, window=full_event)",
            "internal_PFV_full":        "PFV(internal_current_action_branch, window=full_event)",
        },
        "RESIDUAL_LABELS": {
            "delta_PFV_H120_vs_no_control": "candidate_PFV_H120 - no_control_PFV_H120",
            "delta_PFV_H120_vs_passive":    "candidate_PFV_H120 - passive_PFV_H120",
            "delta_TFV_H120_vs_internal":   "candidate_TFV_H120 - internal_TFV_H120",
            "delta_peak_H120_vs_internal":  "candidate_peak_H120 - internal_peak_H120",
            "delta_PFV_full_vs_no_control": "candidate_PFV_full - no_control_PFV_full",
            "delta_PFV_full_vs_passive":    "candidate_PFV_full - passive_PFV_full",
        },
        "source_branch_detail_columns": {
            "PFV": "sum(flood:{priority_node} * dt) over window — computed by sewerrtc/simulation/pyswmm_runner.py:compute_kpis()",
            "TFV": "sum(flood:{any_node} * dt) over window — computed by sewerrtc/simulation/pyswmm_runner.py:compute_kpis()",
            "peak_TFV_rate": "max over window of TFV_rate (m3/h) — computed by sewerrtc/simulation/pyswmm_runner.py:compute_kpis()",
        },
        "delta_convention": "Candidate - Reference (positive = candidate worse than reference for PFV/TFV/peak)",
    }


# ---------------------------------------------------------------------------
# Known defects
# ---------------------------------------------------------------------------

def _known_defects() -> str:
    return """# Gate 0 — Known defects in the current aug1 generation

This file enumerates confirmed defects. Each defect is independently
verifiable from the code references in `gate0_execution_chain.json` and
`gate0_reference_semantics_audit.json`.

## D1. Passive-anchor degeneracy

- Symptom: `passive_PFV_H120` equals `no_control_PFV_H120` for all 1011 aug1 rows.
- Root cause: the passive baseline trajectory's `a:` columns have converged
  to all-1.0 by checkpoint_min >= 40 min. The passive_anchor override
  sequence therefore equals the no_control constant sequence.
- Evidence: `outputs/project6_dual_reference_v4/dual_reference_aug1/cases/`
  case CSVs `__passiv.csv` and `__no_con.csv` have identical `a:` columns
  post-checkpoint.
- Impact: `delta_PFV_H120_vs_passive` is a literal duplicate of
  `delta_PFV_H120_vs_no_control`. Same for full-event pair. The 4 PFV
  heads reduce to 2 independent heads.

## D2. Internal branch is frozen snapshot, not dynamic

- Symptom: `internal_current_action` branch replays the internal_rules
  baseline's `a:` columns at checkpoint_min as a fixed override. Native
  SWMM `[CONTROLS]` are disabled (case INP built with `strip_controls=True`).
- Root cause: `mutate_inp_for_event(..., strip_controls=True)` in
  `_run_group()` removes the `[CONTROLS]` section from the case INP.
- Impact: `delta_TFV_H120_vs_internal` and `delta_peak_H120_vs_internal`
  are NOT relative to dynamic Internal rules. They are relative to a
  frozen snapshot of the internal rules' past actions.

## D3. Peak label near-zero

- Symptom: 98.5% of `delta_peak_H120_vs_internal` values lie within +/- 25.
- Root cause: candidate and internal branches produce similar peak TFV
  rates because both are forced-override trajectories starting from the
  same checkpoint state.
- Impact: Peak direction classification is dominated by numerical noise.

## D4. No-control definition not verified against contract

- Symptom: code uses all-1.0 for `no_control` sequence. Contract must
  confirm this matches the intended No-control semantics (all managed
  facilities open/bypassed). If contract defines No-control as all-off,
  this is a defect.
- Status: NOT YET VERIFIED — Gate 1 must resolve.

## D5. Hold-previous vs internal-current-action collision

- Symptom: `hold_previous` and `internal_current_action` both read from
  `internal_detail` at `checkpoint_min`. Semantically identical in the
  current code.
- Impact: potential double-counting if both are treated as independent
  references.

## D6. Dedup key history

- Symptom: the aug1 dedup key was changed during development
  (`action_type` -> `actual_schedule_sha256`). Older base V4 rows may
  use the legacy key.
- Impact: cross-manifest joins must be key-version aware.
"""


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _run_gate0_audit() -> int:
    out_root = ROOT / "outputs" / "project6_dual_reference_v4" / "recovery_audit"
    out_root.mkdir(parents=True, exist_ok=True)

    _write_json(out_root / "gate0_asset_inventory.json", _asset_inventory())
    _write_json(out_root / "gate0_execution_chain.json", _execution_chain())
    _write_json(out_root / "gate0_reference_semantics_audit.json", _reference_semantics())
    _write_json(out_root / "gate0_manifest_provenance.json", _manifest_provenance())
    _write_json(out_root / "gate0_label_computation_graph.json", _label_computation_graph())
    (out_root / "gate0_known_defects.md").write_text(_known_defects(), encoding="utf-8")

    # Summary to stdout
    produced = sorted(p.name for p in out_root.iterdir() if p.is_file())
    print(json.dumps({"status": "pass", "output_root": str(out_root),
                      "produced_files": produced}, indent=2, ensure_ascii=False))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["gate0_audit"])
    args = ap.parse_args()
    if args.stage == "gate0_audit":
        return _run_gate0_audit()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
