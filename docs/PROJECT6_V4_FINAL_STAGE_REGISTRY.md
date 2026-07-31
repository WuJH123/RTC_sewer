# Project6 V4 Final stage registry

All stages are registered by `sewerrtc.v4.pipeline.ALL_STAGES` and invoked only
through `scripts/project6_v4_final.py` or
`scripts/project6_runs/RUN_PROJECT6_V4_FINAL.ps1`.

Unknown stages fail with exit code 2. Every invocation writes a status record
under:

```text
outputs/project6_dual_reference_v4/final_v4/audits/stage_status/
```

The record contains `run_uuid`, config/code/input SHA values, timestamps,
`exit_code`, `batch_complete`, `scope_complete`, `completed`, `remaining`, and
completion-marker state. A separate completion marker is emitted only when
`scope_complete=true`.

## Ordered stages

1. `AuditContracts`
2. `BuildEventInventory`
3. `PlanOpportunityPool`
4. `ScanOpportunityPool`
5. `BuildOpportunityPool`
6. `AuditOpportunityCoverage`
7. `BuildPeakCandidateCatalog`
8. `PlanPeakBoundary`
9. `RunPeakBoundary`
10. `BuildPeakBoundaryDataset`
11. `AuditPeakBoundary`
12. `ClassifyExistingGate5R`
13. `PlanPilot400`
14. `AuditPilotPlan`
15. `RunPilot400`
16. `BuildPilotDataset`
17. `AuditPilotDataset`
18. `TrainPilotBaselines`
19. `EvaluatePilotGate`
20. `PlanTrain1600`
21. `AuditTrain1600Plan`
22. `RunTrainRound0`
23. `AuditTrainRound0`
24. `TrainActiveLearner0`
25. `SelectTrainRound1`
26. `RunTrainRound1`
27. `AuditTrainRound1`
28. `TrainActiveLearner1`
29. `SelectTrainRound2`
30. `RunTrainRound2`
31. `AuditTrainRound2`
32. `TrainActiveLearner2`
33. `SelectTrainRound3`
34. `RunTrainRound3`
35. `AuditTrainRound3`
36. `BuildTrain1600Dataset`
37. `AuditTrain1600Dataset`
38. `TrainV4`
39. `CalibrateV4`
40. `EvaluateV4Locked`
41. `PlanExactClosedLoop`
42. `RunExactClosedLoop`
43. `AuditExactClosedLoop`
44. `PlanSurrogateClosedLoop`
45. `RunSurrogateClosedLoop`
46. `AuditSurrogateClosedLoop`
47. `LockPolicy`
48. `RunChallenge`
49. `AuditChallenge`
50. `BuildFormalBlindInventory`
51. `RunFormalBlind`
52. `AuditFormalBlind`
53. `BuildPaperResults`
54. `BuildPaperFigures`
55. `BuildPaperTables`
56. `BuildReproducibilityBundle`

## Exit codes

| Code | Meaning |
|---:|---|
| 0 | Contract or scientific gate passed |
| 2 | Contract/input/prerequisite blocked |
| 3 | Incomplete plan, batch, or evidence |
| 4 | Runtime/process failure |
| 5 | Scientific gate failure |

No downstream stage is authorised after a non-zero result.
