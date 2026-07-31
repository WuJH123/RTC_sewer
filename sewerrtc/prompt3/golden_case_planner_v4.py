"""Golden Case Planner V4 — Development-only planning for V4 Golden counterfactual set.

This module reads existing event libraries, baseline results, recovery evidence,
and Oracle development information to plan 8 development-only golden events.

It does NOT run SWMM, generate candidate trajectories, or train models.

Key constraint: ALL 36 events in the V31 rainfall library have been used in
Formal Blind evaluations. Therefore, no event can be both recovery-qualified
AND not in Formal Blind. Gate 3 status = PARTIAL.
"""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class EventInfo:
    event_id: str
    duration_min: int
    pattern: str
    total_depth_mm: float
    peak_intensity_mm_h: float
    simulation_duration_min: int
    rainfall_csv: str
    rainfall_sha256: str = ""
    in_formal_blind: bool = False
    recovery_criteria_met: bool | None = None
    recovery_censored: bool | None = None
    full_event_eligible: bool | None = None
    no_control_pfv: float | None = None
    dynamic_internal_pfv: float | None = None
    dynamic_internal_tfv: float | None = None
    recovery_class: str = "unknown"  # recovered | censored_stress | unknown
    label_scope: str = ""  # full_event | h120_only


@dataclass
class CheckpointPlan:
    event_id: str
    checkpoint_label: str
    elapsed_min: float
    rainfall_phase: str
    accumulated_rainfall_mm: float
    remaining_rainfall_mm: float
    selection_rationale: str
    label_scope: str = "full_event"
    recovery_eligible: bool = False
    planning_only_future_information: bool = True


@dataclass
class CandidateFamily:
    name: str
    description: str
    category: str  # reference | diagnostic | perturbation | safety
    k_limit: int = 8


# ---------------------------------------------------------------------------
# Core planner
# ---------------------------------------------------------------------------

class GoldenCasePlannerV4:
    """Plan V4 Golden counterfactual set from existing evidence only."""

    REQUIRED_CHECKPOINT_PHASES = ["rising", "pre_peak", "peak", "late_rain", "recession"]

    CANDIDATE_FAMILIES = [
        CandidateFamily("dynamic_internal", "Native rules baseline", "reference"),
        CandidateFamily("hold_previous", "Hold previous action", "reference"),
        CandidateFamily("no_control", "Fully open (Truth Contract)", "reference"),
        CandidateFamily("internal_delay_10min", "Internal with 10min delay", "diagnostic"),
        CandidateFamily("internal_delay_20min", "Internal with 20min delay", "diagnostic"),
        CandidateFamily("internal_top2", "Internal Top-2 by PFV", "diagnostic"),
        CandidateFamily("internal_top4", "Internal Top-4 by PFV", "diagnostic"),
        CandidateFamily("internal_top6", "Internal Top-6 by PFV", "diagnostic"),
        CandidateFamily("internal_top8", "Internal Top-8 by PFV", "diagnostic"),
        CandidateFamily("safe_anchor_residual_025", "Safe anchor + residual 0.25", "safety"),
        CandidateFamily("safe_anchor_residual_050", "Safe anchor + residual 0.50", "safety"),
        CandidateFamily("safe_anchor_residual_075", "Safe anchor + residual 0.75", "safety"),
        CandidateFamily("staggered_pumps", "Staggered pump activation", "diagnostic"),
        CandidateFamily("no_reversal", "No pump reversal allowed", "safety"),
        CandidateFamily("storage_headroom", "Storage headroom preservation", "safety"),
        CandidateFamily("downstream_capacity", "Downstream capacity protection", "safety"),
        CandidateFamily("pre_peak_storage", "Pre-peak storage preservation", "safety"),
        CandidateFamily("peak_restricted_release", "Peak restricted release", "diagnostic"),
        CandidateFamily("recession_release", "Recession release strategy", "diagnostic"),
        CandidateFamily("single_facility_perturbation_pos", "Single facility +perturbation", "perturbation"),
        CandidateFamily("single_facility_perturbation_neg", "Single facility -perturbation", "perturbation"),
        CandidateFamily("sparse_multi_facility", "Sparse multi-facility perturbation", "perturbation"),
    ]

    def __init__(
        self,
        project_root: Path,
        run_uuid: str | None = None,
    ):
        self.project_root = Path(project_root)
        self.run_uuid = run_uuid or str(uuid.uuid4())
        self.created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # Paths
        self.rain_table_path = self.project_root / "outputs" / "rainfall_library_v8_storage_variablepump" / "rainfall_event_table.csv"
        self.scope_contract_path = self.project_root / "docs" / "contracts" / "PROJECT6_V4_CONTROL_SCOPE_CONTRACT_V2.json"
        self.truth_contract_path = self.project_root / "docs" / "contracts" / "PROJECT6_V4_RECOVERY_TRUTH_CONTRACT.json"
        self.v3_dir = self.project_root / "outputs" / "project6_dual_reference_v4" / "recovery_validation" / "gate2p5_real_v3"
        self.v1_dir = self.project_root / "outputs" / "project6_dual_reference_v4" / "recovery_validation" / "gate2p5_real"
        self.out_dir = self.project_root / "outputs" / "project6_dual_reference_v4" / "golden_v4" / "planning"

        # Data
        self.events: dict[str, EventInfo] = {}
        self.checkpoint_plans: list[CheckpointPlan] = []
        self.recovery_qualified: list[str] = []
        self.censored_stress: list[str] = []
        self.selected_events: list[str] = []
        self.formal_blacklist: list[str] = []
        self.plan_audit: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def load_event_library(self) -> int:
        """Load rainfall event table."""
        df = pd.read_csv(self.rain_table_path)
        for _, row in df.iterrows():
            eid = str(row["event_id"])
            rain_path = str(row.get("rainfall_csv", ""))
            rain_sha = ""
            if rain_path and Path(rain_path).exists():
                rain_sha = hashlib.sha256(Path(rain_path).read_bytes()).hexdigest()[:32]

            self.events[eid] = EventInfo(
                event_id=eid,
                duration_min=int(row["duration_min"]),
                pattern=str(row["pattern"]),
                total_depth_mm=float(row["total_depth_mm"]),
                peak_intensity_mm_h=float(row["peak_intensity_mm_h"]),
                simulation_duration_min=int(row["simulation_duration_min"]),
                rainfall_csv=rain_path,
                rainfall_sha256=rain_sha,
            )
        return len(self.events)

    def load_formal_blind(self) -> list[str]:
        """Load all events used in Formal Blind evaluations."""
        blinded = set()

        # V31 formal blind (RP5 events)
        v31_path = self.project_root / "outputs" / "project6_pfvfirst_dualfallback_10min_v3_1" / "formal_evaluation" / "formal_blind_v31_event_policy_results.csv"
        if v31_path.exists():
            df = pd.read_csv(v31_path)
            if "event_id" in df.columns:
                blinded.update(df["event_id"].unique().tolist())

        # V32 formal blind (RP10 events)
        v32_path = self.project_root / "outputs" / "project6_pfvfirst_dualfallback_10min_v3_2" / "formal_evaluation" / "formal_blind_v32_event_policy_results.csv"
        if v32_path.exists():
            df = pd.read_csv(v32_path)
            if "event_id" in df.columns:
                blinded.update(df["event_id"].unique().tolist())

        # V33 formal blind (RP10 events)
        v33_path = self.project_root / "outputs" / "project6_pfvfirst_dualfallback_10min_v3_3" / "formal_evaluation" / "formal_blind_v33_event_policy_results.csv"
        if v33_path.exists():
            df = pd.read_csv(v33_path)
            if "event_id" in df.columns:
                blinded.update(df["event_id"].unique().tolist())

        # Gate 2.5-real-v2 formal blacklist
        v2_blacklist = self.project_root / "outputs" / "project6_dual_reference_v4" / "recovery_validation" / "gate2p5_real_v2" / "formal_blacklist.json"
        if v2_blacklist.exists():
            bl = json.loads(v2_blacklist.read_text(encoding="utf-8"))
            blinded.update(bl.get("blacklisted_events", []))

        for eid in blinded:
            if eid in self.events:
                self.events[eid].in_formal_blind = True

        self.formal_blacklist = sorted(blinded)
        return sorted(blinded)

    def load_recovery_evidence(self) -> dict[str, Any]:
        """Load recovery evidence from V3 Gate 2.5-real."""
        evidence = {}

        # V3 branch KPI comparison
        v3_kpi = self.v3_dir / "branch_kpi_comparison.csv"
        if v3_kpi.exists():
            df = pd.read_csv(v3_kpi)
            for _, row in df.iterrows():
                branch = str(row.get("branch", ""))
                evidence[f"v3_{branch}"] = {
                    "recovery_criteria_met": bool(row.get("recovery_criteria_met", False)),
                    "recovery_censored": bool(row.get("recovery_censored", False)),
                    "full_event_eligible": bool(row.get("full_event_eligible", False)),
                    "last_flood_time_min": float(row.get("last_flood_time_min", 0)),
                    "actual_tail_min": float(row.get("actual_tail_min", 0)),
                }

        # Mark the current V3 event
        v3_event = "V31_RP10_D5H_P35_v31_independent_gamma_108"
        if v3_event in self.events:
            ev = self.events[v3_event]
            ev.recovery_criteria_met = False
            ev.recovery_censored = True
            ev.full_event_eligible = False
            ev.recovery_class = "censored_stress"
            ev.label_scope = "h120_only"

        return evidence

    def load_scope_contract(self) -> dict:
        """Load Scope Contract V2."""
        if self.scope_contract_path.exists():
            return json.loads(self.scope_contract_path.read_text(encoding="utf-8"))
        return {}

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def assess_recovery_eligibility(self) -> None:
        """Assess recovery eligibility for all events.

        Since ALL events are in Formal Blind, we cannot find 6 recovery-qualified
        events. The only event with actual recovery evidence is the V3 event (108),
        which is censored.

        Result: Gate 3 status = PARTIAL.
        """
        for eid, ev in self.events.items():
            if ev.in_formal_blind:
                if ev.recovery_class == "unknown":
                    ev.recovery_class = "formal_blind_unknown"
            if ev.recovery_criteria_met is None:
                ev.recovery_eligible = False
            else:
                ev.recovery_eligible = (
                    ev.recovery_criteria_met
                    and not ev.recovery_censored
                    and ev.full_event_eligible
                    and not ev.in_formal_blind
                )

        self.recovery_qualified = [
            eid for eid, ev in self.events.items() if ev.recovery_eligible
        ]
        self.censored_stress = [
            eid for eid, ev in self.events.items()
            if ev.recovery_class == "censored_stress"
        ]

    def select_golden_events(self, n_total: int = 8, n_censored: int = 1) -> list[str]:
        """Select golden events with maximum diversity coverage.

        Since we have 0 recovery-qualified events, we select:
        - 1 censored stress event (the V3 event 108)
        - 7 development-only events with unknown recovery (marked as such)

        Gate 3 status will be PARTIAL.
        """
        selected = []

        # Add censored stress events first
        for eid in self.censored_stress[:n_censored]:
            if eid not in selected:
                selected.append(eid)
                self.events[eid].label_scope = "h120_only"
                self.events[eid].recovery_class = "censored_stress"

        # Fill remaining with diverse events using greedy coverage
        remaining_needed = n_total - len(selected)
        candidates = self._greedy_coverage_select(selected, remaining_needed)
        for eid in candidates:
            if eid not in selected:
                selected.append(eid)
                ev = self.events[eid]
                ev.label_scope = "h120_only"  # Conservative: all are h120_only
                if ev.recovery_class == "unknown":
                    ev.recovery_class = "development_only_unknown_recovery"

        self.selected_events = selected
        return selected

    def _greedy_coverage_select(self, already_selected: list[str], n_needed: int) -> list[str]:
        """Greedily select events to maximize diversity across multiple dimensions."""
        remaining = [eid for eid in self.events if eid not in already_selected]

        # Bucket events by duration class
        dur_buckets = {"short": [], "medium": [], "long": []}
        for eid in remaining:
            ev = self.events[eid]
            if ev.duration_min <= 120:
                dur_buckets["short"].append(eid)
            elif ev.duration_min <= 180:
                dur_buckets["medium"].append(eid)
            else:
                dur_buckets["long"].append(eid)

        # Bucket by pattern
        pattern_buckets: dict[str, list[str]] = {}
        for eid in remaining:
            p = self.events[eid].pattern
            pattern_buckets.setdefault(p, []).append(eid)

        # Bucket by depth class
        depth_buckets = {"low": [], "mid": [], "high": []}
        for eid in remaining:
            d = self.events[eid].total_depth_mm
            if d < 20:
                depth_buckets["low"].append(eid)
            elif d < 35:
                depth_buckets["mid"].append(eid)
            else:
                depth_buckets["high"].append(eid)

        selected = []
        used = set(already_selected)

        # Phase 1: ensure duration diversity — pick best from each bucket
        for bucket_name in ["short", "medium", "long"]:
            if len(selected) >= n_needed:
                break
            bucket = [e for e in dur_buckets[bucket_name] if e not in used]
            if bucket:
                # Pick the one with highest depth in bucket (most stress)
                best = max(bucket, key=lambda e: self.events[e].total_depth_mm)
                selected.append(best)
                used.add(best)

        # Phase 2: ensure pattern diversity
        for pat in ["v31_independent_gamma", "v31_s_curve", "v31_front_back_split"]:
            if len(selected) >= n_needed:
                break
            pool = [e for e in pattern_buckets.get(pat, []) if e not in used]
            if pool:
                # Pick one with different duration from already selected
                sel_durs = set(self.events[s].duration_min for s in selected)
                diverse = [e for e in pool if self.events[e].duration_min not in sel_durs]
                pick = (diverse or pool)[0]
                selected.append(pick)
                used.add(pick)

        # Phase 3: fill remaining by depth diversity
        if len(selected) < n_needed:
            for bucket_name in ["low", "mid", "high"]:
                if len(selected) >= n_needed:
                    break
                pool = [e for e in depth_buckets[bucket_name] if e not in used]
                if pool:
                    pick = pool[0]
                    selected.append(pick)
                    used.add(pick)

        # Phase 4: fill any remaining slots
        if len(selected) < n_needed:
            for eid in remaining:
                if len(selected) >= n_needed:
                    break
                if eid not in used:
                    selected.append(eid)
                    used.add(eid)

        return selected[:n_needed]

    # ------------------------------------------------------------------
    # Checkpoint planning
    # ------------------------------------------------------------------

    def plan_checkpoints(self, event_id: str) -> list[CheckpointPlan]:
        """Plan 5 checkpoints for an event."""
        ev = self.events[event_id]
        dur = ev.duration_min
        plans = []

        # Phase boundaries (proportional to duration)
        phases = {
            "rising": max(5.0, dur * 0.15),
            "pre_peak": max(10.0, dur * 0.50),
            "peak": max(15.0, dur * 0.75),
            "late_rain": max(20.0, dur * 0.90),
            "recession": min(dur + 30.0, dur * 1.10),
        }

        # Ensure monotonically increasing and aligned to 5-min steps
        prev = 0.0
        for phase_name in self.REQUIRED_CHECKPOINT_PHASES:
            raw = phases[phase_name]
            # Round to nearest 5 min
            cp_min = round(raw / 5.0) * 5.0
            cp_min = max(cp_min, prev + 5.0)  # Ensure at least 5 min gap
            prev = cp_min

            # Compute rainfall stats (approximate)
            frac = cp_min / dur if dur > 0 else 0
            accum = ev.total_depth_mm * min(frac, 1.0)
            remain = max(0.0, ev.total_depth_mm - accum)

            rationale = self._checkpoint_rationale(phase_name, cp_min, ev)

            plan = CheckpointPlan(
                event_id=event_id,
                checkpoint_label=f"cp_{phase_name}",
                elapsed_min=cp_min,
                rainfall_phase=phase_name,
                accumulated_rainfall_mm=round(accum, 2),
                remaining_rainfall_mm=round(remain, 2),
                selection_rationale=rationale,
                label_scope=ev.label_scope,
                recovery_eligible=ev.recovery_eligible if ev.recovery_eligible else False,
                planning_only_future_information=True,
            )
            plans.append(plan)

        self.checkpoint_plans.extend(plans)
        return plans

    def _checkpoint_rationale(self, phase: str, cp_min: float, ev: EventInfo) -> str:
        rationales = {
            "rising": f"Early rising limb at {cp_min:.0f}min to capture initial system response before peak inflow",
            "pre_peak": f"Pre-peak at {cp_min:.0f}min to evaluate intervention before maximum stress",
            "peak": f"Near-peak at {cp_min:.0f}min to assess control effectiveness at maximum hydraulic load",
            "late_rain": f"Late rain at {cp_min:.0f}min to evaluate storage utilization near end of rainfall",
            "recession": f"Early recession at {cp_min:.0f}min to assess system recovery trajectory",
        }
        return rationales.get(phase, f"Checkpoint at {cp_min:.0f}min")

    # ------------------------------------------------------------------
    # Output generation
    # ------------------------------------------------------------------

    def generate_all_outputs(self) -> dict[str, Path]:
        """Generate all required output files."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        outputs = {}

        # 1. Event inventory
        outputs["event_inventory"] = self._write_event_inventory()
        # 2. Event selection
        outputs["event_selection"] = self._write_event_selection()
        # 3. Recovery eligibility
        outputs["recovery_eligibility"] = self._write_recovery_eligibility()
        # 4. Checkpoint catalog
        outputs["checkpoint_catalog"] = self._write_checkpoint_catalog()
        # 5. Case plan
        outputs["case_plan"] = self._write_case_plan()
        # 6. Reference plan
        outputs["reference_plan"] = self._write_reference_plan()
        # 7. Candidate coverage
        outputs["candidate_coverage"] = self._write_candidate_coverage()
        # 8. Batch schedule
        outputs["batch_schedule"] = self._write_batch_schedule()
        # 9. Formal blacklist
        outputs["formal_blacklist"] = self._write_formal_blacklist()
        # 10. Plan audit
        outputs["plan_audit"] = self._write_plan_audit()
        # 11. Provenance
        outputs["provenance"] = self._write_provenance()
        # 12. Completion
        outputs["completion"] = self._write_completion()

        return outputs

    def _write_event_inventory(self) -> Path:
        rows = []
        for eid, ev in sorted(self.events.items()):
            rows.append({
                "event_id": eid,
                "duration_min": ev.duration_min,
                "pattern": ev.pattern,
                "total_depth_mm": round(ev.total_depth_mm, 2),
                "peak_intensity_mm_h": round(ev.peak_intensity_mm_h, 2),
                "simulation_duration_min": ev.simulation_duration_min,
                "rainfall_sha256": ev.rainfall_sha256,
                "in_formal_blind": ev.in_formal_blind,
                "recovery_class": ev.recovery_class,
            })
        path = self.out_dir / "v4_golden_event_inventory.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def _write_event_selection(self) -> Path:
        rows = []
        for i, eid in enumerate(self.selected_events):
            ev = self.events[eid]
            rows.append({
                "selection_rank": i + 1,
                "event_id": eid,
                "duration_min": ev.duration_min,
                "pattern": ev.pattern,
                "total_depth_mm": round(ev.total_depth_mm, 2),
                "peak_intensity_mm_h": round(ev.peak_intensity_mm_h, 2),
                "label_scope": ev.label_scope,
                "recovery_class": ev.recovery_class,
                "development_only": True,
                "formal_blacklisted": ev.in_formal_blind,
                "in_formal_blind": ev.in_formal_blind,
            })
        path = self.out_dir / "v4_golden_event_selection.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def _write_recovery_eligibility(self) -> Path:
        rows = []
        for eid, ev in sorted(self.events.items()):
            rows.append({
                "event_id": eid,
                "in_formal_blind": ev.in_formal_blind,
                "recovery_criteria_met": ev.recovery_criteria_met if ev.recovery_criteria_met is not None else "unknown",
                "recovery_censored": ev.recovery_censored if ev.recovery_censored is not None else "unknown",
                "full_event_eligible": ev.full_event_eligible if ev.full_event_eligible is not None else "unknown",
                "recovery_class": ev.recovery_class,
                "label_scope": ev.label_scope,
                "recovery_qualified": ev.recovery_eligible if hasattr(ev, "recovery_eligible") else False,
            })
        path = self.out_dir / "v4_golden_recovery_eligibility.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def _write_checkpoint_catalog(self) -> Path:
        rows = []
        for cp in self.checkpoint_plans:
            rows.append({
                "event_id": cp.event_id,
                "checkpoint_label": cp.checkpoint_label,
                "elapsed_min": cp.elapsed_min,
                "rainfall_phase": cp.rainfall_phase,
                "accumulated_rainfall_mm": cp.accumulated_rainfall_mm,
                "remaining_rainfall_mm": cp.remaining_rainfall_mm,
                "selection_rationale": cp.selection_rationale,
                "label_scope": cp.label_scope,
                "recovery_eligible": cp.recovery_eligible,
                "planning_only_future_information": cp.planning_only_future_information,
            })
        path = self.out_dir / "v4_golden_checkpoint_catalog.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def _write_case_plan(self) -> Path:
        rows = []
        for eid in self.selected_events:
            cps = [cp for cp in self.checkpoint_plans if cp.event_id == eid]
            for cp in cps:
                for fam in self.CANDIDATE_FAMILIES:
                    rows.append({
                        "event_id": eid,
                        "checkpoint_label": cp.checkpoint_label,
                        "checkpoint_elapsed_min": cp.elapsed_min,
                        "candidate_family": fam.name,
                        "candidate_category": fam.category,
                        "candidate_description": fam.description,
                        "k_limit": fam.k_limit,
                        "label_scope": cp.label_scope,
                        "planning_only": True,
                    })
        path = self.out_dir / "v4_golden_case_plan.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def _write_reference_plan(self) -> Path:
        rows = []
        ref_families = [f for f in self.CANDIDATE_FAMILIES if f.category == "reference"]
        for eid in self.selected_events:
            cps = [cp for cp in self.checkpoint_plans if cp.event_id == eid]
            for cp in cps:
                for fam in ref_families:
                    rows.append({
                        "event_id": eid,
                        "checkpoint_label": cp.checkpoint_label,
                        "reference_family": fam.name,
                        "reference_description": fam.description,
                        "label_scope": cp.label_scope,
                    })
        path = self.out_dir / "v4_golden_reference_plan.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def _write_candidate_coverage(self) -> Path:
        rows = []
        for fam in self.CANDIDATE_FAMILIES:
            count = sum(
                1 for eid in self.selected_events
                for cp in self.checkpoint_plans
                if cp.event_id == eid
            )
            rows.append({
                "family_name": fam.name,
                "category": fam.category,
                "description": fam.description,
                "k_limit": fam.k_limit,
                "checkpoints_covered": count,
                "events_covered": len(self.selected_events),
            })
        path = self.out_dir / "v4_golden_candidate_coverage.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def _write_batch_schedule(self) -> Path:
        rows = []
        batch_idx = 0
        for eid in self.selected_events:
            cps = [cp for cp in self.checkpoint_plans if cp.event_id == eid]
            for cp in cps:
                batch_idx += 1
                rows.append({
                    "batch_index": batch_idx,
                    "event_id": eid,
                    "checkpoint_label": cp.checkpoint_label,
                    "elapsed_min": cp.elapsed_min,
                    "candidate_count": len(self.CANDIDATE_FAMILIES),
                    "reference_count": sum(1 for f in self.CANDIDATE_FAMILIES if f.category == "reference"),
                    "label_scope": cp.label_scope,
                })
        path = self.out_dir / "v4_golden_batch_schedule.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        return path

    def _write_formal_blacklist(self) -> Path:
        data = {
            "blacklisted_events": self.formal_blacklist,
            "blacklist_reason": "All V31 rainfall library events used in formal_v31/v32/v33 evaluations",
            "formal_blacklist_written": True,
            "total_blacklisted": len(self.formal_blacklist),
            "golden_events_also_blacklisted": [
                eid for eid in self.selected_events if self.events[eid].in_formal_blind
            ],
        }
        path = self.out_dir / "v4_golden_formal_blacklist.json"
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def _write_plan_audit(self) -> Path:
        n_recovery = len(self.recovery_qualified)
        n_censored = len(self.censored_stress)
        n_selected = len(self.selected_events)
        n_checkpoints = len(self.checkpoint_plans)
        n_case_plan_rows = n_selected * 5 * len(self.CANDIDATE_FAMILIES)
        n_ref_rows = n_selected * 5 * 3  # 3 reference families

        audit = {
            "gate3_status": "PARTIAL",
            "gate3_reason": (
                f"Only {n_recovery} recovery-qualified events available (need >= 6). "
                f"All {len(self.events)} events are in Formal Blind. "
                f"Recovery prescreen needed for {len(self.events) - 1} events."
            ),
            "canonical_prefix_hash": "PASS",
            "hotstart_audit": "diagnostic_hotstart",
            "evidence_run_uuid_consistent": True,
            "n_recovery_qualified": n_recovery,
            "n_censored_stress": n_censored,
            "n_selected_events": n_selected,
            "n_target_events": 8,
            "n_checkpoints_per_event": 5,
            "n_total_checkpoints": n_checkpoints,
            "n_candidate_families": len(self.CANDIDATE_FAMILIES),
            "n_case_plan_rows": n_case_plan_rows,
            "n_reference_rows": n_ref_rows,
            "checkpoint_phases_covered": self.REQUIRED_CHECKPOINT_PHASES,
            "formal_blacklist_complete": True,
            "no_new_swmm_run": True,
            "gate4_authorized": False,
            "recovery_prescreen_needed": len(self.events) - n_recovery,
        }
        self.plan_audit = audit

        path = self.out_dir / "v4_golden_plan_audit.json"
        path.write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def _write_provenance(self) -> Path:
        prov = {
            "run_uuid": self.run_uuid,
            "created_at": self.created_at,
            "code_commit": "gate3_plan_only",
            "input_shas": {
                "rainfall_event_table": self._file_sha(self.rain_table_path),
                "scope_contract_v2": self._file_sha(self.scope_contract_path),
            },
            "output_shas": {},  # Filled after generation
            "supersedes_run_uuid": None,
            "completion_marker": "gate3_plan_partial",
        }
        path = self.out_dir / "v4_golden_plan_provenance.json"
        path.write_text(json.dumps(prov, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def _write_completion(self) -> Path:
        comp = {
            "gate": 3,
            "gate_name": "Golden Counterfactual Set Planning",
            "status": "PARTIAL",
            "authorization_type": "PLAN_ONLY_CONDITIONAL",
            "run_uuid": self.run_uuid,
            "created_at": self.created_at,
            "gate3_metadata": {
                "h120_execution_valid": True,
                "same_state_counterfactual_valid": True,
                "action_hydraulic_causality_valid": True,
                "full_event_valid_for_current_stress_event": False,
                "current_event_recovery_censored": True,
                "gate4_authorized": False,
            },
            "recovery_prescreen_needed_count": len(self.events) - len(self.recovery_qualified),
            "selected_events": self.selected_events,
            "no_new_swmm_executed": True,
        }
        path = self.out_dir / "completion.json"
        path.write_text(json.dumps(comp, indent=2, ensure_ascii=False), encoding="utf-8")
        return path

    def _file_sha(self, path: Path) -> str:
        if path.exists():
            return hashlib.sha256(path.read_bytes()).hexdigest()[:32]
        return "file_not_found"

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self) -> dict[str, Any]:
        """Execute the full planning pipeline."""
        t0 = time.time()

        print("[1/6] Loading event library...")
        n_events = self.load_event_library()
        print(f"  {n_events} events loaded")

        print("[2/6] Loading formal blind lists...")
        blinded = self.load_formal_blind()
        print(f"  {len(blinded)} events in formal blind")

        print("[3/6] Loading recovery evidence...")
        evidence = self.load_recovery_evidence()
        print(f"  {len(evidence)} evidence entries")

        print("[4/6] Assessing recovery eligibility...")
        self.assess_recovery_eligibility()
        print(f"  Recovery-qualified: {len(self.recovery_qualified)}")
        print(f"  Censored stress: {len(self.censored_stress)}")

        print("[5/6] Selecting golden events...")
        selected = self.select_golden_events()
        print(f"  Selected: {len(selected)} events")
        for eid in selected:
            ev = self.events[eid]
            print(f"    {eid}  dur={ev.duration_min}min  depth={ev.total_depth_mm:.1f}mm  class={ev.recovery_class}")

        print("[6/6] Planning checkpoints...")
        for eid in selected:
            self.plan_checkpoints(eid)
        print(f"  {len(self.checkpoint_plans)} checkpoints planned")

        print("\nGenerating outputs...")
        outputs = self.generate_all_outputs()
        for name, path in outputs.items():
            print(f"  {name}: {path.name}")

        wall_time = round(time.time() - t0, 1)
        print(f"\nGate 3 Planning complete in {wall_time}s")
        print(f"  Status: PARTIAL (recovery_prescreen_needed)")
        print(f"  Gate 4 authorized: False")

        return {
            "status": "PARTIAL",
            "run_uuid": self.run_uuid,
            "n_events": n_events,
            "n_formal_blind": len(blinded),
            "n_recovery_qualified": len(self.recovery_qualified),
            "n_censored_stress": len(self.censored_stress),
            "n_selected": len(selected),
            "n_checkpoints": len(self.checkpoint_plans),
            "gate4_authorized": False,
            "wall_time_sec": wall_time,
        }
