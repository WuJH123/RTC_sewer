# Project6 V4 Final Opportunity-to-Peak chain repair

Date: 2026-07-27

This repair closes the three producer/consumer gaps reported in
`PROJECT6_V4_FINAL_DEFECT_REPORT_OPPORTUNITY_CHAIN.md`.

## Repaired stage chain

```text
BuildEventInventory
  -> PlanOpportunityPool
  -> ScanOpportunityPool
  -> BuildOpportunityPool
  -> AuditOpportunityCoverage
  -> BuildPeakCandidateCatalog
  -> PlanPeakBoundary
  -> RunPeakBoundary
  -> BuildPeakBoundaryDataset
  -> AuditPeakBoundary
```

The stages have distinct artifacts:

- `PlanOpportunityPool` writes the executable
  `opportunities/opportunity_scan_plan.csv`.
- `ScanOpportunityPool` writes only the case-completion/run manifest
  `opportunities/opportunity_scan_run_manifest.csv`.
- `BuildOpportunityPool` reads successful detail trajectories, calculates
  opportunity scores, selects checkpoint roles, and writes the scored
  `opportunities/opportunity_pool.csv`.
- `BuildPeakCandidateCatalog` writes targeted Peak-stress candidates to
  `opportunities/peak_candidate_catalog.csv`.
- `PlanPeakBoundary` expands each selected candidate into the four authoritative
  branches: candidate, no-control, dynamic internal rules, and hold previous.
- `BuildPeakBoundaryDataset` accepts only complete, same-state four-branch
  samples and independently calculates H120 PFV, TFV, and Peak labels.

## Peak-failure sample protection

Peak failure examples are not inferred from candidate family names. A
candidate becomes a Peak-degraded sample only after authoritative SWMM
execution and exact label construction:

```text
delta_peak_h120_vs_dynamic_internal
  = Peak(candidate) - Peak(dynamic_internal_rules)
```

`AuditPeakBoundary` fails closed unless the exact dataset contains:

- Peak-degraded samples in at least 3 independent events;
- Peak-degraded samples in at least 6 checkpoints;
- 30 to 60 actual-schedule-unique Peak-degraded samples;
- at least 10 PFV-safe Peak hard negatives;
- at least 2 targeted candidate families.

If the frozen search does not produce these samples,
`peak_constraint_binding_audit.json` is written and the Peak constraint is
retained. The pipeline does not relabel neutral samples as failures and does
not lower the Peak margin.

The later Pilot dataset gate independently repeats the requirements for at
least 30 Peak-degraded samples, at least 10 PFV-safe Peak hard negatives, and
both Peak classes across at least 3 events. Therefore Pilot and Train1600
cannot be authorised with a one-class Peak target.

## Verification boundary

The executable Opportunity plan and all static/unit contracts were verified.
No long Wuhan-network Opportunity or Peak Boundary task was run during this
repair. Exact Peak sample counts remain scientific runtime evidence and must
be established by the future fail-fast command sequence in
`docs/runbooks/PROJECT6_V4_FINAL_RUNBOOK.md`.
