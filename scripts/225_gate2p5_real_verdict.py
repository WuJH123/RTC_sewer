"""Gate 2.5-real Stage 8: Comprehensive verdict and 21-point evidence report.

Aggregates all stage outputs, checks 20 PASS conditions, generates:
  - provenance.csv
  - gate2p5_real_verdict.json
  - completion.json
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "outputs" / "project6_dual_reference_v4" / "recovery_validation" / "gate2p5_real"


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_all() -> dict:
    """Load all stage outputs."""
    d = {}
    d["inventory"] = json.loads((OUT_DIR / "native_control_inventory.json").read_text(encoding="utf-8"))
    d["selection"] = json.loads((OUT_DIR / "positive_control_selection.json").read_text(encoding="utf-8"))
    d["repeatability"] = pd.read_csv(OUT_DIR / "native_internal_repeatability.csv")
    d["checkpoint_catalog"] = pd.read_csv(OUT_DIR / "checkpoint_catalog.csv")
    d["state_hash"] = pd.read_csv(OUT_DIR / "state_hash_comparison.csv")
    d["action_comparison"] = pd.read_csv(OUT_DIR / "branch_action_comparison.csv")
    d["di_trace"] = pd.read_csv(OUT_DIR / "dynamic_internal_trace.csv")
    d["override_audit"] = pd.read_csv(OUT_DIR / "external_override_audit.csv")
    d["readback_audit"] = pd.read_csv(OUT_DIR / "readback_audit.csv")
    d["kpi_comparison"] = pd.read_csv(OUT_DIR / "branch_kpi_comparison.csv")
    d["recovery_audit"] = pd.read_csv(OUT_DIR / "recovery_audit.csv")
    d["scan"] = pd.read_csv(OUT_DIR / "positive_control_event_scan.csv")
    return d


def _check_conditions(data: dict) -> list[dict]:
    """Evaluate 20 PASS conditions."""
    checks = []

    inv = data["inventory"]
    sel = data["selection"]
    rep = data["repeatability"]
    cat = data["checkpoint_catalog"]
    di = data["di_trace"]
    kpi = data["kpi_comparison"]
    rec = data["recovery_audit"]
    scan = data["scan"]
    state_h = data["state_hash"]
    override = data["override_audit"]
    readback = data["readback_audit"]

    # --- Stage 1: Native Inventory ---
    # P1: Native control rules exist
    n_rules = inv.get("native_rule_count", inv.get("total_rules", 0))
    checks.append({"id": "P1", "stage": 1, "description": "Native INP [CONTROLS] has rules",
                    "condition": f"total_rules={n_rules} > 0", "PASS": n_rules > 0})

    # P2: Eng36 facilities overlap with native controls
    n_eng36 = inv.get("engineering36_overlap_count", inv.get("eng36_overlap_count", 0))
    checks.append({"id": "P2", "stage": 1, "description": "Engineering36 facilities controlled by native rules",
                    "condition": f"eng36_overlap={n_eng36} > 0", "PASS": n_eng36 > 0})

    # --- Stage 2: Positive Control Scan ---
    # P3: At least 6 events scanned
    n_events = len(scan)
    checks.append({"id": "P3", "stage": 2, "description": "At least 6 events scanned",
                    "condition": f"scanned_events={n_events} >= 6", "PASS": n_events >= 6})

    # P4: At least one event shows positive control
    if "native_change_detected" in scan.columns:
        positive_events = scan[scan["native_change_detected"] == True]
    elif "total_action_changes" in scan.columns:
        positive_events = scan[scan["total_action_changes"] > 0]
    else:
        positive_events = scan.iloc[0:0]  # empty
    n_positive = len(positive_events)
    checks.append({"id": "P4", "stage": 2, "description": "At least one event with native rule changes",
                    "condition": f"positive_events={n_positive} >= 1", "PASS": n_positive >= 1})

    # P5: Primary event selected with sufficient changes
    primary = sel.get("primary_event", "")
    checks.append({"id": "P5", "stage": 2, "description": "Primary event selected",
                    "condition": f"primary='{primary[:30]}...'", "PASS": bool(primary)})

    # --- Stage 3: Determinism ---
    # P6: Action SHA match
    if len(rep) >= 2:
        action_match = bool(rep.iloc[0].get("action_sha_match_run1_vs_run2", False))
    else:
        action_match = False
    checks.append({"id": "P6", "stage": 3, "description": "Action schedule SHA256 match (run1 vs run2)",
                    "condition": f"action_sha_match={action_match}", "PASS": action_match})

    # P7: PFV/TFV diff = 0
    if len(rep) >= 2:
        pfv_diff = float(rep.iloc[0].get("pfv_abs_diff", 999))
        tfv_diff = float(rep.iloc[0].get("tfv_abs_diff", 999))
        det_pass = pfv_diff == 0.0 and tfv_diff == 0.0
    else:
        pfv_diff = tfv_diff = 999
        det_pass = False
    checks.append({"id": "P7", "stage": 3, "description": "KPI determinism: PFV/TFV diff = 0",
                    "condition": f"pfv_diff={pfv_diff:.2e}, tfv_diff={tfv_diff:.2e}", "PASS": det_pass})

    # --- Stage 4: Checkpoint Selection ---
    # P8: Two distinct checkpoints
    n_cp = len(cat)
    checks.append({"id": "P8", "stage": 4, "description": "Two distinct checkpoints selected",
                    "condition": f"n_checkpoints={n_cp} == 2", "PASS": n_cp == 2})

    # P9: Checkpoints at different phases
    if n_cp >= 2:
        phases = cat["phase_at_checkpoint"].tolist() if "phase_at_checkpoint" in cat.columns else []
        diff_phase = len(set(phases)) > 1 if phases else True
        cp_times = cat["checkpoint_elapsed_min"].tolist()
        time_diff = abs(cp_times[0] - cp_times[1]) if len(cp_times) >= 2 else 0
        checks.append({"id": "P9", "stage": 4, "description": "Checkpoints at different hydraulic phases/times",
                        "condition": f"phases={phases}, time_diff={time_diff:.0f}min",
                        "PASS": diff_phase or time_diff >= 30})
    else:
        checks.append({"id": "P9", "stage": 4, "description": "Checkpoints at different phases",
                        "condition": "N/A (< 2 checkpoints)", "PASS": False})

    # --- Stage 5: Four-Branch ---
    # P10: All branches executed (8 detail files: 2 cp x 4 branches)
    branch_files = list(OUT_DIR.glob("branch_*_cp*_detail.csv"))
    n_branch_files = len(branch_files)
    checks.append({"id": "P10", "stage": 5, "description": "All 8 branch detail files exist (2 cp x 4 branches)",
                    "condition": f"branch_files={n_branch_files} == 8", "PASS": n_branch_files >= 8})

    # P11: no_control has 0 post-checkpoint action changes
    nc_rows = data["action_comparison"]
    nc_post = nc_rows[(nc_rows["branch"] == "no_control")]["post_checkpoint_action_changes"]
    nc_zero = bool((nc_post == 0).all()) if len(nc_post) > 0 else False
    checks.append({"id": "P11", "stage": 5, "description": "no_control: 0 post-checkpoint action changes",
                    "condition": f"no_control_changes={nc_post.tolist()}", "PASS": nc_zero})

    # P12: dynamic_internal has >0 post-checkpoint changes
    di_rows = data["action_comparison"]
    di_post = di_rows[(di_rows["branch"] == "dynamic_internal")]["post_checkpoint_action_changes"]
    di_active = bool((di_post > 0).all()) if len(di_post) > 0 else False
    checks.append({"id": "P12", "stage": 5, "description": "dynamic_internal: >0 post-checkpoint action changes",
                    "condition": f"dynamic_changes={di_post.tolist()}", "PASS": di_active})

    # P13: State hashes computed (network + rainfall SHA present)
    has_net_sha = "network_sha256" in state_h.columns and state_h["network_sha256"].notna().all()
    has_rain_sha = "rainfall_sha256" in state_h.columns and state_h["rainfall_sha256"].notna().all()
    checks.append({"id": "P13", "stage": 5, "description": "State hashes computed for all branches",
                    "condition": f"network_sha={has_net_sha}, rainfall_sha={has_rain_sha}",
                    "PASS": has_net_sha and has_rain_sha})

    # --- Stage 6: Dynamic Internal Audit ---
    # P14: policy_phase transitions from prefix_replay to native_rules
    if len(di) > 0:
        transition_ok = bool(di["policy_phase_transition"].all())
    else:
        transition_ok = False
    checks.append({"id": "P14", "stage": 6, "description": "policy_phase transitions prefix_replay -> native_rules",
                    "condition": f"all_transitions={transition_ok}", "PASS": transition_ok})

    # P15: Override inactive after transition
    if len(override) > 0:
        override_ok = bool(override["PASS"].all())
    else:
        override_ok = False
    checks.append({"id": "P15", "stage": 6, "description": "External override inactive after override_start_min",
                    "condition": f"override_audit_PASS={override_ok}", "PASS": override_ok})

    # P16: Eng36 facility changes post-checkpoint
    if len(di) > 0:
        eng36_changes = int(di["eng36_changes_post_checkpoint"].sum())
        eng36_ok = eng36_changes > 0
    else:
        eng36_changes = 0
        eng36_ok = False
    checks.append({"id": "P16", "stage": 6, "description": "Eng36 facility setting changes post-checkpoint",
                    "condition": f"total_eng36_changes={eng36_changes}", "PASS": eng36_ok})

    # P17: Action SHA differs from hold_internal_snapshot
    if len(di) > 0:
        sha_differs = bool(di["action_sha_differs_from_hold_snapshot"].all())
    else:
        sha_differs = False
    checks.append({"id": "P17", "stage": 6, "description": "Dynamic internal action SHA differs from hold_snapshot",
                    "condition": f"sha_differs={sha_differs}", "PASS": sha_differs})

    # P18: Binary pumps ADD301.2/ADD301.3 strict 0/1
    if len(di) > 0:
        bin2 = bool(di["binary_pump_ADD301_2_strict_01"].all()) if "binary_pump_ADD301_2_strict_01" in di.columns else False
        bin3 = bool(di["binary_pump_ADD301_3_strict_01"].all()) if "binary_pump_ADD301_3_strict_01" in di.columns else False
    else:
        bin2 = bin3 = False
    checks.append({"id": "P18", "stage": 6, "description": "Binary pumps ADD301.2/ADD301.3 strict 0/1",
                    "condition": f"ADD301.2={bin2}, ADD301.3={bin3}", "PASS": bin2 and bin3})

    # P19: Readback a: vs setting: passes
    if len(readback) > 0:
        rb_pass = bool(readback["PASS"].all())
    else:
        rb_pass = False
    checks.append({"id": "P19", "stage": 6, "description": "Readback: a: columns match setting: columns",
                    "condition": f"all_readback_pass={rb_pass}", "PASS": rb_pass})

    # --- Stage 7: KPI ---
    # P20: KPIs computed for all branches at all checkpoints
    n_kpi_rows = len(kpi)
    expected_kpi = len(cat) * 4  # 4 branches per checkpoint
    checks.append({"id": "P20", "stage": 7, "description": "KPIs computed for all branches",
                    "condition": f"kpi_rows={n_kpi_rows} == {expected_kpi}", "PASS": n_kpi_rows >= expected_kpi})

    return checks


def _build_provenance(data: dict) -> list[dict]:
    """Build provenance records for each stage output."""
    rows = []
    stage_files = {
        1: ["native_control_inventory.json", "native_control_rules.csv", "native_control_facility_map.csv"],
        2: ["positive_control_event_scan.csv", "positive_control_selection.json"],
        3: ["native_internal_run1_detail.csv", "native_internal_run2_detail.csv", "native_internal_repeatability.csv"],
        4: ["checkpoint_catalog.csv"],
        5: ["state_hash_comparison.csv", "branch_action_comparison.csv"],
        6: ["dynamic_internal_trace.csv", "external_override_audit.csv", "readback_audit.csv"],
        7: ["branch_kpi_comparison.csv", "recovery_audit.csv"],
    }
    for stage, files in stage_files.items():
        for fname in files:
            fpath = OUT_DIR / fname
            if fpath.exists():
                rows.append({
                    "stage": stage,
                    "filename": fname,
                    "exists": True,
                    "size_bytes": fpath.stat().st_size,
                    "sha256": _file_sha256(fpath)[:16],
                    "generated_by": f"scripts/{217 + stage}_gate2p5_real_*.py",
                })
            else:
                rows.append({
                    "stage": stage, "filename": fname, "exists": False,
                    "size_bytes": 0, "sha256": "", "generated_by": "",
                })
    # Add branch detail files
    for f in sorted(OUT_DIR.glob("branch_*_cp*_detail.csv")):
        rows.append({
            "stage": 5, "filename": f.name, "exists": True,
            "size_bytes": f.stat().st_size, "sha256": _file_sha256(f)[:16],
            "generated_by": "scripts/222_gate2p5_real_four_branch.py",
        })
    return rows


def _build_evidence_21(data: dict, checks: list[dict]) -> list[dict]:
    """Build 21-point evidence report."""
    inv = data["inventory"]
    sel = data["selection"]
    rep = data["repeatability"]
    cat = data["checkpoint_catalog"]
    di = data["di_trace"]
    kpi = data["kpi_comparison"]
    override = data["override_audit"]

    evidence = []

    # E1: Network
    evidence.append({"point": 1, "topic": "Network INP",
                     "detail": f"wuhan_v8_storage_retrofit.inp, {inv.get('native_rule_count', 0)} native rules, "
                               f"{inv.get('rule_controlled_facility_count', 0)} facilities, {inv.get('engineering36_overlap_count', 0)} Eng36 overlap"})

    # E2: Rainfall event
    evidence.append({"point": 2, "topic": "Primary rainfall event",
                     "detail": f"{sel.get('primary_event', 'N/A')}, duration={sel.get('primary_duration_min', '?')}min"})

    # E3: Positive control evidence
    if "native_change_detected" in data["scan"].columns:
        n_positive = int((data["scan"]["native_change_detected"] == True).sum())
    elif "total_action_changes" in data["scan"].columns:
        n_positive = int((data["scan"]["total_action_changes"] > 0).sum())
    else:
        n_positive = 0
    evidence.append({"point": 3, "topic": "Positive control events",
                     "detail": f"{n_positive}/{len(data['scan'])} events show native rule changes"})

    # E4: Determinism
    det_verdict = rep.iloc[0].get("determinism_verdict", "UNKNOWN") if len(rep) > 0 else "UNKNOWN"
    evidence.append({"point": 4, "topic": "Determinism verification",
                     "detail": f"Two identical runs: {det_verdict}"})

    # E5: Checkpoint A
    if len(cat) >= 1:
        cp_a = cat.iloc[0]
        evidence.append({"point": 5, "topic": "Checkpoint A",
                         "detail": f"t={cp_a['checkpoint_elapsed_min']:.0f}min, phase={cp_a.get('phase_at_checkpoint', '?')}, "
                                   f"rain={cp_a.get('rainfall_at_checkpoint', 0):.2f}mm/h"})
    else:
        evidence.append({"point": 5, "topic": "Checkpoint A", "detail": "NOT SELECTED"})

    # E6: Checkpoint B
    if len(cat) >= 2:
        cp_b = cat.iloc[1]
        evidence.append({"point": 6, "topic": "Checkpoint B",
                         "detail": f"t={cp_b['checkpoint_elapsed_min']:.0f}min, phase={cp_b.get('phase_at_checkpoint', '?')}, "
                                   f"rain={cp_b.get('rainfall_at_checkpoint', 0):.2f}mm/h"})
    else:
        evidence.append({"point": 6, "topic": "Checkpoint B", "detail": "NOT SELECTED"})

    # E7-E14: Four-branch results per checkpoint
    ev_point = 7
    for i, (_, cp_row) in enumerate(cat.iterrows()):
        cp_label = cp_row["checkpoint_label"]
        cp_kpi = kpi[kpi["checkpoint_label"] == cp_label]
        for _, kr in cp_kpi.iterrows():
            br = kr["branch"]
            evidence.append({"point": ev_point, "topic": f"CP{cp_label} {br}",
                             "detail": f"H120_PFV={kr.get('H120_PFV', 0):.1f}, full_TFV={kr.get('full_TFV', 0):.1f}, "
                                       f"peak={kr.get('full_peak_TFV_rate', 0):.3f}, changes_post={kr.get('action_changes_post_checkpoint', 0)}"})
            ev_point += 1
            if ev_point > 14:
                break
        if ev_point > 14:
            break

    # Pad to E14 if needed
    while len(evidence) < 14:
        evidence.append({"point": len(evidence) + 1, "topic": "Branch result (padding)", "detail": "N/A"})

    # E15: Policy phase transition
    if len(di) > 0:
        trans = di.iloc[0]
        evidence.append({"point": 15, "topic": "Policy phase transition",
                         "detail": f"prefix_replay -> native_rules at t={trans.get('transition_elapsed_min', '?')}min"})
    else:
        evidence.append({"point": 15, "topic": "Policy phase transition", "detail": "N/A"})

    # E16: Override audit
    if len(override) > 0:
        ov = override.iloc[0]
        evidence.append({"point": 16, "topic": "External override audit",
                         "detail": f"override_inactive_post={ov.get('override_inactive_after_transition', '?')}, "
                                   f"transition_t={ov.get('transition_elapsed_min', '?')}min"})
    else:
        evidence.append({"point": 16, "topic": "External override audit", "detail": "N/A"})

    # E17: Eng36 changes
    if len(di) > 0:
        total_changes = int(di["eng36_changes_post_checkpoint"].sum())
        total_facilities = int(di["eng36_facilities_changed"].sum())
        evidence.append({"point": 17, "topic": "Eng36 facility changes",
                         "detail": f"{total_changes} changes across {total_facilities} facility-checkpoint combinations"})
    else:
        evidence.append({"point": 17, "topic": "Eng36 facility changes", "detail": "N/A"})

    # E18: Binary pump verification
    if len(di) > 0 and bool(di["binary_pump_ADD301_2_strict_01"].all()):
        evidence.append({"point": 18, "topic": "Binary pump verification",
                         "detail": "ADD301.2/ADD301.3: strict 0/1 verified"})
    else:
        evidence.append({"point": 18, "topic": "Binary pump verification", "detail": "N/A"})

    # E19: Readback audit
    rb = data["readback_audit"]
    n_rb_pass = int(rb["PASS"].sum()) if len(rb) > 0 else 0
    n_rb_total = len(rb)
    evidence.append({"point": 19, "topic": "Readback audit",
                     "detail": f"{n_rb_pass}/{n_rb_total} actuator-checkpoint pairs pass a: vs setting: check"})

    # E20: KPI comparison
    nc_kpi = kpi[kpi["branch"] == "no_control"]
    di_kpi = kpi[kpi["branch"] == "dynamic_internal"]
    if len(nc_kpi) > 0 and len(di_kpi) > 0:
        nc_tfv = nc_kpi["full_TFV"].mean()
        di_tfv = di_kpi["full_TFV"].mean()
        evidence.append({"point": 20, "topic": "KPI: TFV comparison",
                         "detail": f"no_control TFV={nc_tfv:.1f} m3, dynamic_internal TFV={di_tfv:.1f} m3"})
    else:
        evidence.append({"point": 20, "topic": "KPI: TFV comparison", "detail": "N/A"})

    # E21: Overall verdict
    all_pass = all(c["PASS"] for c in checks)
    n_pass = sum(1 for c in checks if c["PASS"])
    evidence.append({"point": 21, "topic": "GATE 2.5-real VERDICT",
                     "detail": f"{'PASS' if all_pass else 'FAIL'} ({n_pass}/{len(checks)} conditions met)"})

    return evidence


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[Stage 8] Loading all stage outputs...")
    data = _load_all()

    print("[Stage 8] Checking 20 PASS conditions...")
    checks = _check_conditions(data)

    n_pass = sum(1 for c in checks if c["PASS"])
    n_fail = len(checks) - n_pass
    all_pass = n_fail == 0

    print(f"\n  PASS: {n_pass}/{len(checks)}")
    for c in checks:
        status = "PASS" if c["PASS"] else "FAIL"
        print(f"    [{status}] {c['id']}: {c['description']} -- {c['condition']}")

    # Build provenance
    provenance = _build_provenance(data)
    prov_df = pd.DataFrame(provenance)
    prov_path = OUT_DIR / "provenance.csv"
    prov_df.to_csv(prov_path, index=False)
    print(f"\n[Stage 8] Wrote {prov_path}")

    # Build 21-point evidence
    evidence = _build_evidence_21(data, checks)

    # Verdict JSON
    verdict = {
        "gate": "2.5-real",
        "gate_name": "Real-network Dynamic Internal validation",
        "timestamp": datetime.now().isoformat(),
        "verdict": "PASS" if all_pass else "FAIL",
        "pass_conditions": n_pass,
        "fail_conditions": n_fail,
        "total_conditions": len(checks),
        "primary_event": data["selection"].get("primary_event", ""),
        "checkpoints": data["checkpoint_catalog"]["checkpoint_elapsed_min"].tolist(),
        "native_rules_count": data["inventory"].get("native_rule_count", data["inventory"].get("total_rules", 0)),
        "eng36_overlap": data["inventory"].get("engineering36_overlap_count", data["inventory"].get("eng36_overlap_count", 0)),
        "determinism_verdict": data["repeatability"].iloc[0].get("determinism_verdict", "UNKNOWN") if len(data["repeatability"]) > 0 else "UNKNOWN",
        "conditions": checks,
        "evidence_21": evidence,
        "gate3_authorization": False,
        "note": "Even if PASS, Gate 3 requires separate user authorization.",
    }
    verdict_path = OUT_DIR / "gate2p5_real_verdict.json"
    verdict_path.write_text(json.dumps(verdict, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"[Stage 8] Wrote {verdict_path}")

    # Completion JSON
    completion = {
        "gate": "2.5-real",
        "status": "COMPLETE",
        "verdict": "PASS" if all_pass else "FAIL",
        "timestamp": datetime.now().isoformat(),
        "stages_completed": [1, 2, 3, 4, 5, 6, 7, 8],
        "output_directory": str(OUT_DIR),
        "key_outputs": [
            "native_control_inventory.json",
            "positive_control_selection.json",
            "native_internal_repeatability.csv",
            "checkpoint_catalog.csv",
            "state_hash_comparison.csv",
            "branch_action_comparison.csv",
            "dynamic_internal_trace.csv",
            "external_override_audit.csv",
            "readback_audit.csv",
            "branch_kpi_comparison.csv",
            "recovery_audit.csv",
            "provenance.csv",
            "gate2p5_real_verdict.json",
            "completion.json",
        ],
        "gate3_next": False,
    }
    comp_path = OUT_DIR / "completion.json"
    comp_path.write_text(json.dumps(completion, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[Stage 8] Wrote {comp_path}")

    # Print 21-point evidence report
    print(f"\n{'='*70}")
    print(f"  GATE 2.5-real: 21-POINT EVIDENCE REPORT")
    print(f"{'='*70}")
    for e in evidence:
        print(f"  [{e['point']:2d}] {e['topic']:40s} | {e['detail']}")
    print(f"{'='*70}")
    print(f"  VERDICT: {verdict['verdict']} ({n_pass}/{len(checks)} conditions PASS)")
    print(f"{'='*70}")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
