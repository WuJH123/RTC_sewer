# Project6 Engineering36 Runbook

This is the new Project6 Engineering36 branch. Old Project4/Project5 controllers, old thresholds, and old action models are reference material only.

## Frozen Contract

Run Stage 0 first:

```powershell
cd E:\RTC_sewer\Project6
$Py = "E:\RTC_sewer\Project6\.venv\Scripts\python.exe"
$Pipe = ".\scripts\project6_runs\RUN_PROJECT6_ENGINEERING36.ps1"
& $Pipe -InitContract -Python $Py
& $Pipe -AuditContract -Python $Py
```

Stage 0 writes:

- `outputs/project6_engineering36/frozen/facilities_36_semantics.csv`
- `outputs/project6_engineering36/frozen/priority_core_nodes.csv`
- `outputs/project6_engineering36/frozen/sentinel_nodes.csv`
- `outputs/project6_engineering36/frozen/event_split.csv`
- `outputs/project6_engineering36/frozen/gat_model_registry.csv`
- `outputs/project6_engineering36/frozen/contract_manifest.json`

## Control Contract

- Tier 0: No-control fallback.
- Tier 1: frozen Core26 engineering candidates.
- Tier 2: rejectable Residual10 increments only.
- Action form: `u36 = u_core26 + delta_u_new10`.
- `ADD301.2` and `ADD301.3` are binary pumps.
- PFV event budget: `max(200 m3, 0.02 * predicted_event_no_control_PFV)`.
- PFV budget is cumulative across the event and must use UCB, not mean prediction.
- Priority PFV nodes and sentinel nodes are frozen before training and FormalBlind.

## Required Stage Order

1. Freeze contract and event split.
2. Establish same retrofit INP baselines: No-control and Core26.
3. Run SWMM Pareto/oracle audit for Residual10 physical benefit.
4. Generate same-state counterfactual data.
5. Train or reuse state model; train action effect and uncertainty models.
6. Build fit-only empirical reliability.
7. Pass offline model gate.
8. Run 4-5 event smoke.
9. Run calibration.
10. Freeze model, thresholds, and event split.
11. Lock and run FormalBlind once.

Any failed stage stops later stages.

## FormalBlind Lock

Only lock FormalBlind after calibration passes:

```powershell
& $Pipe -FormalBlindLock -Python $Py
```

The lock prevents silent reruns against the same blind split.
