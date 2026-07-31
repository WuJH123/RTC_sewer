"""Gate 3.5 v2 Phase 7: Golden event marking + Gate 3 verdict."""
import json, time
from pathlib import Path

OUT_DIR = Path(r"outputs/project6_dual_reference_v4/recovery_capability_v2")
golden_events = [
    "V31_RP10_D2H_P65_v31_independent_gamma_084",
    "V31_RP10_D3H_P20_v31_independent_gamma_090",
    "V31_RP10_D2H_P80_v31_independent_gamma_087",
    "V31_RP10_D2H_P65_v31_s_curve_085",
    "V31_RP10_D2H_P65_v31_front_back_split_086",
    "V31_RP10_D2H_P80_v31_s_curve_088",
    "V31_RP10_D5H_P20_v31_independent_gamma_105",
    "V31_RP10_D5H_P35_v31_independent_gamma_108",
]
stress_candidates = {"V31_RP10_D5H_P20_v31_independent_gamma_105", "V31_RP10_D5H_P35_v31_independent_gamma_108"}

marking = {
    "gate": 3,
    "gate_verdict": "PARTIAL",
    "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "reason": "0 recovery-qualified events (need >= 6). All 8 events R0 under both no_control and dynamic_internal.",
    "golden_events": [],
    "summary": {
        "total_golden": 8,
        "recovery_qualified": 0,
        "censored_stress": 0,
        "full_event_eligible": 0,
    },
}
for eid in golden_events:
    is_stress = eid in stress_candidates
    marking["golden_events"].append({
        "event_id": eid,
        "development_golden": True,
        "formal_eligible": False,
        "formal_blacklisted": True,
        "role": "stress_candidate" if is_stress else "recovery_candidate",
        "recovery_class_no_control": "R0_not_recovered",
        "recovery_class_dynamic_internal": "R0_not_recovered",
    })

p = OUT_DIR / "gate3_golden_event_marking.json"
p.write_text(json.dumps(marking, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"Wrote: {p}")
v = marking["gate_verdict"]
print(f"Verdict: {v}")
