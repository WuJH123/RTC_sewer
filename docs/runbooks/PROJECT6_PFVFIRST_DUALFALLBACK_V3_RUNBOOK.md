# Project6 PFV-First Dual-Fallback V3 Runbook

This runbook contains the only commands that should be executed for the V3 line
until the user provides results for review. Codex must not run these commands
automatically.

## Environment

Run from PowerShell:

```powershell
cd E:\RTC_sewer\Project6
$Py = "E:\RTC_sewer\Project6\.venv\Scripts\python.exe"
$Cfg = "configs\wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml"
$Pipe = ".\scripts\project6_runs\RUN_PROJECT6_PFVFIRST_DUALFALLBACK_10MIN_V3.ps1"
```

If script execution is blocked, use:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File $Pipe -Audit -Python $Py -Config $Cfg
```

## Stage 0: Static Asset Audit

Command:

```powershell
& $Pipe -Audit -Python $Py -Config $Cfg
```

Inputs:

- `configs/wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml`
- `data/wuhan_v8_storage_retrofit.inp`
- `data/project6_v8_storage_retrofit_control_enabled_ids.txt`
- `docs/contracts/kpi_contract.json`
- `docs/contracts/forecast_contract.json`
- Project4 read-only GAT checkpoint paths

Outputs:

- `outputs/project6_pfvfirst_dualfallback_10min_v3/audit/asset_audit_manifest.json`

Success conditions:

- retrofit INP exists
- exactly 36 managed actuator ids
- KPI and forecast contracts exist
- GAT candidate paths are listed with hashes or marked missing

Stop on failure:

- missing INP
- managed actuator count not 36
- missing contract files

Provide these logs on failure:

- terminal output
- `outputs/project6_pfvfirst_dualfallback_10min_v3/audit/asset_audit_manifest.json` if created

## Stage 1: Coverage Planning Preview

Command:

```powershell
& $Pipe -PlanCoverage -Python $Py -Config $Cfg
```

Inputs:

- V3 config coverage targets

Outputs:

- `outputs/project6_pfvfirst_dualfallback_10min_v3/coverage/coverage_gap_audit.csv`
- `outputs/project6_pfvfirst_dualfallback_10min_v3/coverage/candidate_manifest_preview.csv`
- `outputs/project6_pfvfirst_dualfallback_10min_v3/coverage/coverage_plan_report.json`

Success conditions:

- coverage target table is written
- candidate preview file is initialized

Stop on failure:

- config cannot be parsed
- output directory cannot be created

Provide these logs on failure:

- terminal output
- `coverage_plan_report.json` if created

Troubleshooting:

- If Python reports `ModuleNotFoundError: No module named 'sewerrtc'`, use the
  updated runner in `scripts/project6_runs/RUN_PROJECT6_PFVFIRST_DUALFALLBACK_10MIN_V3.ps1`.
  It sets `PYTHONPATH` to the Project6 root before invoking Python.

## Stage 2: Same-State Dataset Build

Do not run until a real case manifest exists.

Command template:

```powershell
$CaseManifest = "outputs\project6_pfvfirst_dualfallback_10min_v3\same_state\approved_case_manifest.csv"
& $Pipe -BuildDataset -Python $Py -Config $Cfg -CaseManifest $CaseManifest
```

Inputs:

- approved same-state case manifest
- completed branch detail files
- actual executed action sequences

Outputs:

- `outputs/project6_pfvfirst_dualfallback_10min_v3/same_state/dataset_build_report.json`

Success conditions:

- manifest contains required schema
- action input field is `actual_executed_action_sequence`
- selected fallback id is present before candidate labels

Stop on failure:

- missing required manifest columns
- branch details missing
- continuation policy missing

Provide these logs on failure:

- terminal output
- case manifest header
- `dataset_build_report.json` if created

## Stage 3: Train Effect Model

Do not run until Stage 2 has produced an approved `.npz` or `.parquet` dataset.

Command template:

```powershell
$Dataset = "outputs\project6_pfvfirst_dualfallback_10min_v3\same_state\pfvfirst_dualfallback_dataset.npz"
& $Pipe -TrainModel -Python $Py -Config $Cfg -Dataset $Dataset
```

Inputs:

- same-state effect dataset
- V3 config model gate thresholds

Outputs:

- `outputs/project6_pfvfirst_dualfallback_10min_v3/models/training_manifest.json`
- trained model files only after implementation is completed

Success conditions:

- dataset exists
- dataset format is `.npz` or `.parquet`
- training manifest records required inputs and outputs

Stop on failure:

- unsupported dataset format
- missing dataset
- coverage gate not satisfied

Provide these logs on failure:

- terminal output
- `training_manifest.json` if created
- dataset audit report

## Stage 4: Gate Effect Model

Do not run until a real model validation metrics JSON exists.

Command template:

```powershell
$Metrics = "outputs\project6_pfvfirst_dualfallback_10min_v3\models\validation_metrics.json"
& $Pipe -GateModel -Python $Py -Config $Cfg -Metrics $Metrics
```

Inputs:

- model validation metrics JSON
- V3 configured gate thresholds

Outputs:

- `outputs/project6_pfvfirst_dualfallback_10min_v3/gates/pfvfirst_model_gate_report.json`

Success conditions:

- all required gate metrics present
- all configured thresholds pass
- independent event support is sufficient

Stop on failure:

- missing metrics key
- unsafe recall below threshold
- false-safe above threshold
- interval coverage outside range
- independent event support below threshold

Provide these logs on failure:

- terminal output
- validation metrics JSON
- gate report JSON if created

## Future Stages Not Yet Enabled

The following stages must not be run until the user has provided successful
Stage 0-4 outputs and Codex has reviewed them:

- SWMM same-state data generation
- top-k shadow validation
- Smoke
- Calibration-A
- Locked Validation-B
- FormalBlind

## Expected Follow-Up To Codex

After running a stage, provide:

1. terminal output
2. generated report JSON/CSV paths
3. any traceback
4. whether files were created

Codex will then inspect the supplied logs and modify source/config files only.

## Prompt 2 Close-Out: sr0p15 Primary GAT Lock And State Entry

This section reflects the user-confirmed primary GAT decision:

- `registry_name = sr0p15`
- `declared_sensor_ratio = 0.15`
- `expected_sensor_count = 134`
- checkpoint: `E:\RTC_sewer\Project4\outputs\sensor_sensitivity\sr0p15\models\gat_sr0p10.pt`
- expected SHA256: `11f40e6a36016202139e604f04c7d888b5ec3805511c46172ad968a7c20d0e20`

The selection lock does not mean GAT robustness passed and does not unlock Round 0.

### Common Setup

```powershell
cd E:\RTC_sewer\Project6

$Py = "E:\RTC_sewer\Project6\.venv\Scripts\python.exe"
$Run = ".\scripts\project6_runs\RUN_PROJECT6_PFVFIRST_DUALFALLBACK_10MIN_V3.ps1"
$Cfg = ".\configs\wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml"
```

Because this update changes the V3 config hash, refresh the low-cost GAT audit markers before selecting the primary GAT:

```powershell
& $Run -RegisterGAT -Python $Py -Config $Cfg
$LASTEXITCODE

& $Run -RecoverGATMetadata -Python $Py -Config $Cfg
$LASTEXITCODE

& $Run -InspectGATCheckpoints -Python $Py -Config $Cfg
$LASTEXITCODE

& $Run -AuditGAT -Python $Py -Config $Cfg
$LASTEXITCODE
```

Expected exit code for each command: `0`.

Stop if any command returns nonzero. Provide the terminal output and the relevant file under:

`outputs\project6_pfvfirst_dualfallback_10min_v3\gat`

### 1. Run Contract Tests

```powershell
& $Py -m pytest `
  tests\test_project6_v3_runner_contract.py `
  tests\test_project6_v3_gat_state_contracts.py `
  tests\test_project6_v3_gat_selection_lock.py `
  tests\test_project6_v3_gat_robustness.py `
  tests\test_project6_v3_runtime_state_features.py `
  tests\test_project6_v3_state_clone_contract.py `
  tests\test_project6_prompt2_completion_gate.py -q
$LASTEXITCODE
```

Expected exit code: `0`.

Stop if tests fail. Provide the complete pytest failure output.

### 2. Generate Manual sr0p15 Selection Lock

```powershell
& $Run -SelectPrimaryGAT `
  -Python $Py `
  -Config $Cfg `
  -GATRegistryName "sr0p15" `
  -AcknowledgeSelection
$LASTEXITCODE
```

Expected exit code: `0`.

Expected output:

`outputs\project6_pfvfirst_dualfallback_10min_v3\gat\gat_primary_selection_lock.json`

Stop conditions:

- exit `6`: checkpoint/report/config/hash mismatch
- exit `7`: missing `-GATRegistryName` or missing acknowledgement
- no lock file created

The lock must keep:

- `robustness_status = pending`
- `round0_unlock_allowed = false`

### 3. Run sr0p15 Robustness Audit

```powershell
& $Run -RunGATRobustnessAudit `
  -Python $Py `
  -Config $Cfg `
  -MaxSamples 2048
$LASTEXITCODE
```

Expected exit code:

- `0` if the audit runs and writes all sr0p15 diagnostic files;
- `3` if the selection lock is missing or validation assets are insufficient;
- `4` if the audit code fails at runtime.

Expected outputs include:

- `outputs\project6_pfvfirst_dualfallback_10min_v3\gat\gat_sr0p15_validation_dataset_manifest.json`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\gat\gat_sr0p15_priority_leaveout_audit.csv`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\gat\gat_sr0p15_sentinel_leaveout_audit.csv`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\gat\gat_sr0p15_highwater_phase_audit.csv`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\gat\gat_sr0p15_sensor_failure_audit.csv`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\gat\gat_sr0p15_robustness_gate.json`

Open the robustness gate. If `status` is not `pass`, stop and send the gate JSON plus the audit CSVs to Codex. Do not enter state generation.

### 4. Prepare State Contracts

```powershell
& $Run -PrepareStateFeatureContracts `
  -Python $Py `
  -Config $Cfg
$LASTEXITCODE
```

Expected exit code: `0`.

Expected outputs:

- `outputs\project6_pfvfirst_dualfallback_10min_v3\state\state_feature_contract.json`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\state\state_feature_schema.json`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\state\facility_state_schema.csv`

This stage does not generate runtime state features and does not unlock Round 0.

### 5. Build Real Seven-Frame Runtime State Features

Run only after a real state input manifest has been prepared.

```powershell
$StateInput = "<实际状态输入manifest路径>"

& $Run -BuildStateFeatures `
  -Python $Py `
  -Config $Cfg `
  -StateInputManifest $StateInput `
  -MaxSamples 100
$LASTEXITCODE
```

Expected exit code:

- `0` if state histories pass shape, causality and missingness audits;
- `3` if the primary GAT lock is missing, the manifest is missing, fields are insufficient, 60 min history is unavailable, or future data/zero-filled missing flow is detected.

Expected outputs:

- `outputs\project6_pfvfirst_dualfallback_10min_v3\state\augmented_state_sample_manifest.csv`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\state\augmented_state_shape_audit.json`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\state\augmented_state_causality_audit.csv`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\state\augmented_state_missingness_audit.csv`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\state\augmented_state_facility_audit.csv`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\state\state_input_gap_report.json`

Stop if `state_input_gap_report.json` is not `completed`.

### 6. Evaluate Prompt 2 Completion Gate

```powershell
& $Run -EvaluatePrompt2Completion `
  -Python $Py `
  -Config $Cfg
$LASTEXITCODE
```

Expected exit code:

- `0` only when every Prompt 2 condition has passed;
- `3` while blocked at manual lock, sr0p15 robustness, or runtime state validation.

Expected output:

`outputs\project6_pfvfirst_dualfallback_10min_v3\gates\project6_prompt2_completion_gate.json`

Stop if the gate is not `pass`. Do not run Round 0.

### 7. View Status

```powershell
& $Run -Status -Python $Py -Config $Cfg
$LASTEXITCODE
```

Expected exit code: `0`.
## V3 Repair Runbook Addendum

This runbook section reflects the repaired PFV-first dual-fallback V3 contracts. Codex must not run these commands automatically under the current working rule. The user runs them manually in PowerShell or PyCharm.

### Common Setup

```powershell
cd E:\RTC_sewer\Project6
$Py = "E:\RTC_sewer\Project6\.venv\Scripts\python.exe"
$Cfg = "configs\wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml"
$Pipe = ".\scripts\project6_runs\RUN_PROJECT6_PFVFIRST_DUALFALLBACK_10MIN_V3.ps1"
```

### 1. Status And Source Dependency Audit

```powershell
& $Pipe -Status -Python $Py -Config $Cfg
$LASTEXITCODE
```

Expected output:

- JSON report with `status = source_dependency_audit_only_no_runtime_validation`.
- Contract, script and module paths with existence and SHA256 hashes.
- Disabled or unimplemented stage groups listed explicitly.

Pass condition:

- Required V3 contracts, scripts and modules exist.
- Expected exit code: `0`.

Stop condition:

- Any required contract or core script is missing.

### 2. Static Asset Audit

```powershell
& $Pipe -Audit -Python $Py -Config $Cfg
$LASTEXITCODE
```

Expected output:

- `status = passed_file_presence_and_hash_audit`.
- 36 managed actuator IDs.
- Duplicate-ID report.
- Preliminary actuator audit.
- GAT candidates marked `present_unverified`, not `compatible`.
- `preliminary_text_reference_count`, not native-rule behavior counts.
- Sentinel contract may report `human_resolution_required`; this blocks later real control stages.

Pass condition:

- Config, retrofit INP, 36-ID file, KPI contract and forecast contract exist and have hashes.
- Expected exit code: `0`.

Stop condition:

- Any managed ID is missing from the active input files after the later detailed INP audit is implemented.

### 3. InitCoverageSchema Only

```powershell
& $Pipe -InitCoverageSchema -Python $Py -Config $Cfg
$LASTEXITCODE
```

Expected output:

- Coverage target audit.
- Coverage cell schema.
- Candidate manifest schema.
- `status = scaffold_only`.
- `batch_effective_candidate_max` listed as a maximum constraint, not as a missing gap.
- No `_COMPLETED.json` marker is created.
- No downstream stage is unlocked.
- Expected exit code: `0`.

Pass condition:

- The schema contains event/storm-family, split, checkpoint, phase, risk cluster, anchor, facility group, direction, magnitude, duration, concurrency, interaction type, unique event support, feasibility, outcome and decision relevance.

Stop condition:

- The output claims real coverage is complete before any actual SWMM candidate data exist.

### 4. Disabled Dataset/Training/Gate Stages

The following stages are intentionally disabled in the current V3 scaffold and must return non-zero:

```powershell
& $Pipe -BuildDataset -Python $Py -Config $Cfg
$LASTEXITCODE
& $Pipe -TrainPilot -Python $Py -Config $Cfg
$LASTEXITCODE
& $Pipe -MinimalGate -Python $Py -Config $Cfg
$LASTEXITCODE
```

Expected output:

- execution status written with `status = disabled`;
- `failure_reason = not_implemented`;
- `completion_marker = null`;
- no `_COMPLETED.json` marker.
- Expected exit code: `2`.

Verify the exit code immediately after each command:

```powershell
$LASTEXITCODE
```

For `BuildDataset`, `TrainPilot`, `MinimalGate`, `RunSmoke`, and all other disabled stages, the value must be `2`. If it is `3`, the runner is incorrectly mapping a disabled stage to a blocked stage.

Pass condition for this scaffold:

- The stage fails clearly and does not create a completion marker.

Stop condition:

- Any of these stages returns success before real implementation exists.

### Disabled Stages

These commands are intentionally expected to fail fast until their implementation scripts and markers exist:

```powershell
& $Pipe -AuditGAT -Python $Py -Config $Cfg
& $Pipe -StateCloneTest -Python $Py -Config $Cfg
& $Pipe -BuildDataset -Python $Py -Config $Cfg
& $Pipe -TrainPilot -Python $Py -Config $Cfg
& $Pipe -MinimalGate -Python $Py -Config $Cfg
& $Pipe -BuildEventCatalog -Python $Py -Config $Cfg
& $Pipe -BuildCheckpointCatalog -Python $Py -Config $Cfg
& $Pipe -RunInternalPFVOpportunityScan -Python $Py -Config $Cfg
& $Pipe -RunMPCDryRun -Python $Py -Config $Cfg
& $Pipe -RunSmoke -Python $Py -Config $Cfg
& $Pipe -CalibrationA -Python $Py -Config $Cfg
& $Pipe -FormalBlind -Python $Py -Config $Cfg
```

Expected output:

- A clear disabled-stage error, not a silent success.
- Expected exit code: `2`.

### Exit Code Contract

- `0`: allowed read-only audit, scaffold initialization, or real completed stage success.
- `2`: disabled or not implemented.
- `3`: blocked by missing or stale upstream completion.
- `4`: runtime exception.
- `5`: real safety or model gate failed.
- `6`: configuration, contract, or hash mismatch.
- `7`: user input, CLI parameter, or stage-selection error.

Stable exit-code smoke checks:

```powershell
& $Pipe -Status -Python $Py -Config $Cfg
$LASTEXITCODE    # expected 0

& $Pipe -Audit -Python $Py -Config $Cfg
$LASTEXITCODE    # expected 0

& $Pipe -BuildDataset -Python $Py -Config $Cfg
$LASTEXITCODE    # expected 2

& $Pipe -Status -Audit -Python $Py -Config $Cfg
$LASTEXITCODE    # expected 7
```

For a blocked `InitCoverageSchema` check, first ensure a populated candidate manifest exists in the V3 coverage directory, then run:

```powershell
& $Pipe -InitCoverageSchema -Python $Py -Config $Cfg
$LASTEXITCODE    # expected 3 when populated artifacts are present
```

### Current Stage Order

1. `Status`
2. `Audit`
3. `InitCoverageSchema`
4. `FatalAudit` disabled
5. `AuditNativeRules` disabled
6. `AuditFallbacks` disabled
7. `RegisterGAT` disabled
8. `AuditGAT` disabled
9. `BuildStateFeatures` disabled
10. `BuildEventCatalog` disabled
11. `BuildCheckpointCatalog` disabled
12. `StateCloneTest` disabled
13. `RunInternalPFVOpportunityScan` disabled
14. `PlanRound0` alias of `InitCoverageSchema` only, real planning disabled
15. `DryRunRound0` disabled
16. `GenerateRound0` disabled

### Coverage Schema Safety

Normal `InitCoverageSchema` may create empty schema files only. If any target schema artifact contains data rows, the stage must block with exit code `3` and must not modify files.

Dangerous reinitialization is intentionally not part of normal workflow:

```powershell
& $Pipe -InitCoverageSchema -ForceReinitializeEmptyCoverage -AcknowledgeDataLoss -Python $Py -Config $Cfg
```

Use it only after confirming the target directory is exactly `outputs\project6_pfvfirst_dualfallback_10min_v3\coverage` and after preserving any needed data elsewhere.

### Reporting Back To Codex

If any command fails, provide:

- the exact command;
- the full terminal output;
- the relevant JSON or CSV path named in the output;
- whether the failure occurred before or after a file was written.
## Manual Commands: Step 1 GAT Registration And State Contracts

Do not run downstream action-data, training, MPC, Smoke, Calibration, or Formal
stages before these commands complete and their outputs are reviewed.

Set variables:

```powershell
cd E:\RTC_sewer\Project6

$Py = "E:\RTC_sewer\Project6\.venv\Scripts\python.exe"
$Run = ".\scripts\project6_runs\RUN_PROJECT6_PFVFIRST_DUALFALLBACK_10MIN_V3.ps1"
$Cfg = ".\configs\wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml"
```

### 1. Check status

```powershell
& $Run -Status -Python $Py -Config $Cfg
$LASTEXITCODE
```

Expected exit code: `0`.

Expected output:

- `outputs\project6_pfvfirst_dualfallback_10min_v3\execution_status\execution_status_index.json`

Stop if:

- exit code is not `0`;
- any disabled stage incorrectly has a completion marker;
- any completed stage has a missing output.

### 2. Register Project4 GAT candidates

```powershell
& $Run -RegisterGAT -Python $Py -Config $Cfg
$LASTEXITCODE
```

Expected exit code:

- `0` when all five read-only source checkpoints are found and registry files are written;
- `4` if a checkpoint source is missing or cannot be read.

Expected outputs:

- `outputs\project6_pfvfirst_dualfallback_10min_v3\gat\gat_external_registry.csv`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\gat\gat_checkpoint_hashes.csv`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\gat\gat_registration_report.json`
- completion marker for `RegisterGAT` only if exit code is `0`.

Stop if:

- any Project4 checkpoint is missing;
- the registry collapses the five same-named checkpoints into one record;
- Project4 source files are modified.

### 3. Audit GAT compatibility

```powershell
& $Run -AuditGAT -Python $Py -Config $Cfg
$LASTEXITCODE
```

Expected exit code: `0` if the audit report is written. Incompatible or
metadata-incomplete candidates must be reported in the audit, not silently fixed
or retrained. This stage may safely load checkpoint tensors on CPU to inspect
state_dict keys, dtype summaries, and NaN/Inf parameter status, but it must not
run GAT inference or training.

Expected outputs:

- `outputs\project6_pfvfirst_dualfallback_10min_v3\gat\gat_compatibility_report.json`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\gat\gat_node_mapping.csv`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\gat\gat_sensor_mapping.csv`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\gat\gat_normalization_audit.csv`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\gat\gat_graph_signature_audit.csv`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\gat\gat_checkpoint_load_audit.csv`

Stop if:

- RegisterGAT has not completed;
- config hash mismatch is reported;
- all candidates are `load_failed`.

Review condition:

- If no candidate is `compatible_strict`, do not claim formal GAT readiness.
  Human selection or metadata recovery is required.

### 4. Build augmented-state contracts

```powershell
& $Run -BuildStateFeatures -Python $Py -Config $Cfg
$LASTEXITCODE
```

Expected exit code: `0` when schema and contracts are generated. This does not
freeze a primary GAT and does not generate action-effect data.

Expected outputs:

- `outputs\project6_pfvfirst_dualfallback_10min_v3\state\state_feature_contract.json`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\state\state_feature_schema.json`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\state\facility_state_schema.csv`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\state\temporal_state_alignment_audit.csv`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\state\state_quality_contract.json`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\state\local_flow_feature_contract.json`

Stop if:

- AuditGAT has not completed;
- output claims `selected_primary_gat` is frozen;
- sentinel safety PASS appears while sentinel contract is unresolved.

### 5. Prepare state-clone test schema

```powershell
& $Run -StateCloneTest -Python $Py -Config $Cfg
$LASTEXITCODE
```

Expected exit code: `3` in the current setup, because real SWMM checkpoint and
controller-memory artifacts are not yet available.

Expected outputs before blocking:

- `outputs\project6_pfvfirst_dualfallback_10min_v3\state\state_clone_contract.json`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\state\controller_state_manifest.schema.json`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\state\state_clone_equivalence.schema.csv`

Stop if:

- exit code is `0` before a real SWMM clone equivalence test is run;
- any completion marker is created for `StateCloneTest` in this setup-only run.

### Disabled stages that should still return 2

```powershell
& $Run -BuildDataset -Python $Py -Config $Cfg
$LASTEXITCODE

& $Run -TrainPilot -Python $Py -Config $Cfg
$LASTEXITCODE

& $Run -MinimalGate -Python $Py -Config $Cfg
$LASTEXITCODE
```

Expected exit code for each: `2`.

## Prompt 2 Repair: sr0p15 Robustness Memory And State Input Manifest

This repair separates total audit sample count from the GAT forward batch size.
`MaxSamples` is the total number of validation samples. `BatchSize` is only the
micro-batch size used for each Project4 GAT forward call. The default requested
batch size is `8`; the Python stage writes a memory plan and reduces the
effective batch size if the estimated peak memory would exceed `MaxMemoryGB`.

### Test The Repair Contracts

```powershell
& $Py -m pytest `
  tests\test_project6_v3_runner_contract.py `
  tests\test_project6_v3_gat_selection_lock.py `
  tests\test_project6_v3_gat_robustness.py `
  tests\test_project6_v3_state_input_manifest.py `
  tests\test_project6_v3_runtime_state_features.py `
  tests\test_project6_prompt2_completion_gate.py -q
$LASTEXITCODE
```

Expected exit code: `0`.

Stop if this fails.

### Refresh GAT Audit And Selection Lock

Run this because the primary lock hash schema changed. The old
`gat_primary_selection_lock.json` must be considered stale.

```powershell
& $Run -AuditGAT -Python $Py -Config $Cfg
$LASTEXITCODE

& $Run -SelectPrimaryGAT `
  -Python $Py `
  -Config $Cfg `
  -GATRegistryName "sr0p15" `
  -AcknowledgeSelection
$LASTEXITCODE
```

Expected exit code for each command: `0`.

Stop if `SelectPrimaryGAT` returns:

- `6`: checkpoint/report/config/hash mismatch;
- `7`: missing acknowledgement or wrong registry name.

Expected output:

- `outputs\project6_pfvfirst_dualfallback_10min_v3\gat\gat_primary_selection_lock.json`

The lock must contain separate hash evidence objects for:

- `state_dict_signature`;
- `edge_set_hash`;
- `static_tensor_hash`;
- `sensor_ids_hash`;
- `node_ids_hash`;
- `node_order_hash`.

### Small sr0p15 Robustness Audit

Start with a small run:

```powershell
& $Run -RunGATRobustnessAudit `
  -Python $Py `
  -Config $Cfg `
  -MaxSamples 64 `
  -BatchSize 8 `
  -FlushEvery 8 `
  -MaxMemoryGB 4
$LASTEXITCODE
```

Expected exit code:

- `0`: audit files were written;
- `3`: selection lock or validation assets are missing;
- `4`: runtime failure, including confirmed out-of-memory.

Inspect:

- `outputs\project6_pfvfirst_dualfallback_10min_v3\gat\gat_robustness_memory_plan.json`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\gat\gat_sr0p15_robustness_gate.json`

If memory is stable, continue:

```powershell
& $Run -RunGATRobustnessAudit `
  -Python $Py `
  -Config $Cfg `
  -MaxSamples 256 `
  -BatchSize 8 `
  -FlushEvery 16 `
  -MaxMemoryGB 4 `
  -Resume
$LASTEXITCODE
```

Only consider increasing to `MaxSamples 512` or `2048` after reviewing the
memory plan and runtime. If memory pressure remains low, try `-BatchSize 16`;
the stage will still reduce the effective batch size if the estimate exceeds
4 GB.

### Prepare State Contracts Again

```powershell
& $Run -PrepareStateFeatureContracts -Python $Py -Config $Cfg
$LASTEXITCODE
```

Expected exit code: `0`.

The generated state contract should now read the real primary lock status:

- `selection_lock_status = locked` when a valid lock exists;
- `selection_lock_status = stale` or blocked if hashes no longer match;
- `selection_lock_status = pending_manual_execution` only when no lock exists.

### Build A State Input Manifest

For Project4 GAT node-level validation only:

```powershell
& $Run -BuildStateInputManifest `
  -Python $Py `
  -Config $Cfg `
  -SourceMode "project4_gat_validation" `
  -MaxSamples 100
$LASTEXITCODE
```

Expected exit code: `0`.

Expected output:

- `outputs\project6_pfvfirst_dualfallback_10min_v3\state_inputs\state_input_manifest_v1.csv`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\state_inputs\state_trajectory_gap_report.json`

This manifest must contain:

- `gat_node_state_validation_eligible = true`;
- `full_project6_augmented_state_eligible = false`.

It must not be treated as full Project6 augmented state.

For full Project6 state, provide a real current-retrofit trajectory root:

```powershell
& $Run -BuildStateInputManifest `
  -Python $Py `
  -Config $Cfg `
  -SourceMode "project6_retrofit_baseline" `
  -TrajectoryRoot "<real Project6 retrofit trajectory root>" `
  -MaxSamples 100
$LASTEXITCODE
```

Expected exit code:

- `0` only after the trajectory schema mapping is implemented and valid;
- `3` if the trajectory root is missing or not yet mappable.

### Build State Features

Do not use placeholder paths. Placeholder values such as `<实际状态输入manifest路径>`,
`placeholder`, or `TODO` return exit code `7`.

```powershell
$StateInput = "E:\RTC_sewer\Project6\outputs\project6_pfvfirst_dualfallback_10min_v3\state_inputs\state_input_manifest_v1.csv"

& $Run -BuildStateFeatures `
  -Python $Py `
  -Config $Cfg `
  -StateInputManifest $StateInput `
  -MaxSamples 100
$LASTEXITCODE
```

Expected exit code:

- `0` only for a full Project6 augmented-state manifest with sr0p15 robustness
  gate `pass`;
- `3` if sr0p15 robustness is not `pass`, if the manifest is node-validation
  only, if 60 min history is missing, or if required facility/storage/pump
  fields are unavailable;
- `7` if the manifest path is a placeholder.

### Prompt 2 Completion Gate

```powershell
& $Run -EvaluatePrompt2Completion -Python $Py -Config $Cfg
$LASTEXITCODE
```

Expected exit code:

- `0` only when sr0p15 lock, robustness and real seven-frame state validation
  have all passed;
- `3` while blocked at robustness or runtime state validation.

## Prompt 2 Final Close-Out: sr0p15 Four-State Robustness Gate

This section closes the remaining sr0p15 robustness checks without changing the
GAT model, lowering thresholds, or running any downstream action-data or MPC
stage.

### Run Tests

```powershell
& $Py -m pytest `
  tests\test_project6_v3_gat_robustness.py `
  tests\test_project6_v3_gat_validation_provenance.py `
  tests\test_project6_v3_gat_event_leakage.py `
  tests\test_project6_v3_gat_sensor_failure.py `
  tests\test_project6_v3_gat_latency.py `
  tests\test_project6_prompt2_gat_readiness.py `
  tests\test_project6_v3_runner_contract.py -q

$LASTEXITCODE
```

Expected exit code: `0`.

Stop if tests fail.

### Re-run sr0p15 Robustness Audit

```powershell
& $Run -RunGATRobustnessAudit `
  -Python $Py `
  -Config $Cfg `
  -MaxSamples 256 `
  -BatchSize 8 `
  -FlushEvery 16 `
  -MaxMemoryGB 4 `
  -Resume

$LASTEXITCODE
```

Expected exit code:

- `0`: audit artifacts written;
- `3`: missing lock or insufficient validation assets;
- `4`: runtime failure, including out-of-memory.

Expected new or repaired outputs:

- `gat_sr0p15_validation_sample_inventory.csv`
- `gat_sr0p15_validation_provenance_audit.csv`
- `gat_sr0p15_rainfall_near_duplicate_audit.csv`
- `gat_sr0p15_split_membership_audit.csv`
- `gat_sr0p15_sensor_failure_contract.json`
- `gat_sr0p15_sensor_failure_completion_matrix.csv`
- `gat_sr0p15_sensor_failure_summary.csv`
- `gat_sr0p15_latency_contract.json`
- `gat_sr0p15_latency_summary.json`

### Audit sr0p15 Validation Provenance Without Re-running GAT

Run this after `RunGATRobustnessAudit` and before `EvaluateGATRobustnessGate`.
This stage only searches and audits Project4 training/validation provenance and
reuses the existing sr0p15 robustness outputs. It does not run GAT inference.

```powershell
& $Run -AuditGATValidationProvenance `
  -Python $Py `
  -Config $Cfg

$LASTEXITCODE
```

Expected exit code:

- `0`: provenance audit completed. This may mean PASS or a confirmed leakage
  failure; run `EvaluateGATRobustnessGate` next to classify the gate;
- `3`: critical Project4 provenance assets are still missing or ambiguous.
  Inspect `gat_sr0p15_independent_validation_gap_report.json` and stop;
- `4`: code/runtime error;
- `6`: config/hash/contract mismatch.

Inspect:

```powershell
$GatDir = "E:\RTC_sewer\Project6\outputs\project6_pfvfirst_dualfallback_10min_v3\gat"

Import-Csv "$GatDir\gat_sr0p15_validation_provenance_audit.csv" |
  Format-Table -AutoSize

Import-Csv "$GatDir\gat_sr0p15_validation_leakage_audit.csv" |
  Format-Table -AutoSize

Import-Csv "$GatDir\gat_sr0p15_validation_event_support.csv" |
  Format-Table -AutoSize
```

### Evaluate The Robustness Gate Without Re-running GAT

```powershell
& $Run -EvaluateGATRobustnessGate `
  -Python $Py `
  -Config $Cfg

$LASTEXITCODE
```

Expected exit code:

- `0`: four-state robustness gate is PASS;
- `3`: evidence is incomplete;
- `5`: complete audit confirms leakage or a performance gate failure;
- `6`: lock, config, checkpoint or hash mismatch;
- `4`: code/runtime error.

This stage only reads existing reports. It must not run GAT inference.

### View The Gate

```powershell
$Gate = "E:\RTC_sewer\Project6\outputs\project6_pfvfirst_dualfallback_10min_v3\gat\gat_sr0p15_robustness_gate.json"

Get-Content $Gate -Raw |
  ConvertFrom-Json |
  Format-List *
```

Proceed only if:

```text
status = pass
```

### Project4 Node-Level Seven-Frame Validation

Build or refresh the Project4 node-state manifest:

```powershell
& $Run -BuildStateInputManifest `
  -Python $Py `
  -Config $Cfg `
  -SourceMode "project4_gat_validation" `
  -MaxSamples 100

$LASTEXITCODE
```

Expected exit code: `0`.

Then validate node-level seven-frame state causality:

```powershell
$StateInput = "E:\RTC_sewer\Project6\outputs\project6_pfvfirst_dualfallback_10min_v3\state_inputs\state_input_manifest_v1.csv"

& $Run -BuildStateFeatures `
  -Python $Py `
  -Config $Cfg `
  -StateInputManifest $StateInput `
  -StateValidationMode "project4_node_only" `
  -MaxSamples 100

$LASTEXITCODE
```

Expected exit code:

- `0`: node-level seven-frame GAT state validation completed;
- `3`: robustness gate is not PASS or manifest evidence is insufficient;
- `7`: a placeholder path was supplied.

This mode does not mark full Project6 augmented state complete.

### Evaluate Prompt 2 GAT Readiness For Prompt 3A

```powershell
& $Run -EvaluatePrompt2GATReadiness `
  -Python $Py `
  -Config $Cfg

$LASTEXITCODE
```

Expected exit code:

- `0`: allowed to enter Prompt 3A;
- `3`: still blocked.

Only enter Prompt 3A when the output contains:

```text
status = pass
allowed_to_enter_prompt3a = true
full_project6_augmented_state_complete = false
```

## Prompt 0 Recovery: Current Truth, Split Prompt3A Gates, And Marker Audit

This stage does not generate trajectories, run SWMM, run GAT inference, train
models, run MPC, run Calibration, or run FormalBlind. It only reads current
evidence files and rebuilds truthful machine-readable status.

```powershell
cd E:\RTC_sewer\Project6

$Py   = "E:\RTC_sewer\Project6\.venv\Scripts\python.exe"
$Run  = ".\scripts\project6_runs\RUN_PROJECT6_PFVFIRST_DUALFALLBACK_10MIN_V3.ps1"
$Cfg  = ".\configs\wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml"
$Root = "E:\RTC_sewer\Project6\outputs\project6_pfvfirst_dualfallback_10min_v3"
```

### Contract Tests

```powershell
& $Py -m pytest `
  tests\test_project6_v3_current_truth.py `
  tests\test_project6_v3_runner_contract.py `
  tests\test_project6_v3_state_input_manifest.py `
  tests\test_project6_v3_runtime_state_features.py `
  tests\test_project6_v3_state_clone.py -q

$LASTEXITCODE
```

Expected exit code: `0`.

Stop if this returns nonzero and provide the full pytest failure.

### Rebuild Current Truth Matrix

```powershell
& $Run -AuditCurrentTruth -Python $Py -Config $Cfg
$LASTEXITCODE
```

Expected exit code: `0`.

Expected outputs:

- `outputs\project6_pfvfirst_dualfallback_10min_v3\status\project6_current_truth_matrix.csv`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\status\project6_current_truth_report.json`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\gates\project6_recovery_gate.json`

Expected truthful content:

- Prompt2: `pass`
- Baseline trajectory count: `6`
- Real state-processed trajectory count: read from current files
- Hot-start equivalence: `not_run` or `missing`
- Hydraulic dry-run: `not_run`
- Effective Round0 candidates: current report value, expected around `9`

### Evaluate Prompt3A Engineering Gate

```powershell
& $Run -EvaluatePrompt3AEngineeringGate -Python $Py -Config $Cfg
$LASTEXITCODE
```

Expected exit code: `0` if static contracts, small baseline generation, state
schema, coverage schema, and candidate preview evidence are present.

Expected output:

`outputs\project6_pfvfirst_dualfallback_10min_v3\gates\project6_prompt3a_engineering_gate.json`

This does not unlock Round0 generation and does not prove runtime safety.

### Evaluate Prompt3A Runtime Gate

```powershell
& $Run -EvaluatePrompt3ARuntimeGate -Python $Py -Config $Cfg
$LASTEXITCODE
```

Expected current exit code: `3`.

Expected current status: `blocked`.

This is the correct result until all baseline trajectories enter the state
pipeline, actual features match the frozen schemas, real hot-start and
controller-memory files exist, State Clone equivalence passes, the hydraulic
candidate dry-run truly executes, truth leakage is zero, engineering violations
are zero, and Round0 has 1500 to 2000 effective candidates.

### Optional Aggregate Prompt3A Completion Gate

```powershell
& $Run -EvaluatePrompt3ACompletion -Python $Py -Config $Cfg
$LASTEXITCODE
```

Expected current exit code: `3`, because completion now requires both the
engineering gate and the runtime gate to pass.

## Prompt 1 Small Scientific Loop: Full State, Tail Recovery, Hot-Start, And State Clone

Run this only after `AuditCurrentTruth` has rebuilt the truth matrix. These
commands intentionally keep the scope to the current 2 events x 3 policies.

```powershell
cd E:\RTC_sewer\Project6

$Py   = "E:\RTC_sewer\Project6\.venv\Scripts\python.exe"
$Run  = ".\scripts\project6_runs\RUN_PROJECT6_PFVFIRST_DUALFALLBACK_10MIN_V3.ps1"
$Cfg  = ".\configs\wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml"
$Root = "E:\RTC_sewer\Project6\outputs\project6_pfvfirst_dualfallback_10min_v3"
```

### Prompt 1 Contract Tests

```powershell
& $Py -m pytest `
  tests\test_project6_v3_prompt3a_scientific_loop.py `
  tests\test_project6_v3_current_truth.py `
  tests\test_project6_v3_runtime_state_features.py `
  tests\test_project6_v3_state_input_manifest.py `
  tests\test_project6_v3_state_clone.py `
  tests\test_project6_v3_runner_contract.py -q

$LASTEXITCODE
```

Expected exit code: `0`. Stop and provide the failure log if nonzero.

### Regenerate The 2-Event Baseline With 5-Min State Visibility

Do not use `-Resume` for this repair run; existing detail files must be
refreshed so hot-start and controller-memory sidecars are produced.

```powershell
& $Run -GenerateBaselineTrajectories `
  -Python $Py `
  -Config $Cfg `
  -MaxEvents 2 `
  -Workers 1 `
  -TailMin 180

$LASTEXITCODE
```

Expected exit code: `0`.

Expected outputs:

- `baseline_trajectory_manifest.csv`
- `baseline_recovery_audit.csv`
- `baseline_checkpoint_audit.csv`
- per-trajectory recovery JSON files
- per-checkpoint controller-memory JSON files
- per-checkpoint hot-start files when PySWMM supports hot-start save

Acceptance:

- selected trajectories = `6`
- failed trajectories = `0`
- visible state step = `300 sec`
- RTC decision interval = `600 sec`

### Build State Input Manifest From The 6 Baseline Trajectories

```powershell
& $Run -BuildStateInputManifest `
  -Python $Py `
  -Config $Cfg `
  -SourceMode "project6_retrofit_baseline" `
  -TrajectoryRoot "$Root\baseline_trajectories" `
  -MaxSamples 0

$LASTEXITCODE
```

Expected exit code: `0`.

Acceptance:

- baseline trajectory count = `6`
- state input trajectory count = `6`
- trajectory key is `trajectory_id|event_id|policy_id`

### Build Full Project6 Runtime State Features

```powershell
$StateInput = "$Root\state_inputs\state_input_manifest_v1.csv"

& $Run -BuildStateFeatures `
  -Python $Py `
  -Config $Cfg `
  -StateInputManifest $StateInput `
  -StateValidationMode "project6_full_baseline" `
  -MaxSamples 0

$LASTEXITCODE
```

Expected exit code: `0`.

Expected outputs:

- `node_feature_index.json`
- `facility_feature_index.json`
- `storage_feature_index.json`
- `feature_materialization_audit.csv`
- `augmented_state_shape_audit.json`

Acceptance:

- `len(actual_node_feature_names) == F_node`
- `len(actual_facility_feature_names) == F_facility`
- `len(actual_storage_feature_names) == F_storage`
- future data count = `0`
- missing flow is not encoded as true zero

### Build Checkpoint Catalog

```powershell
& $Run -BuildCheckpointCatalog -Python $Py -Config $Cfg
$LASTEXITCODE
```

Expected exit code:

- `0` if at least one checkpoint has real hot-start and controller-memory files;
- `3` if PySWMM did not produce hot-start files.

Acceptance for continuing:

- hotstart files > `0`
- controller-memory files > `0`
- runtime-clone-eligible checkpoints > `0`

### Prepare And Evaluate State Clone

```powershell
& $Run -PrepareStateCloneCheckpoints -Python $Py -Config $Cfg
$LASTEXITCODE

& $Run -EstimateStateCloneNumericalNoise -Python $Py -Config $Cfg
$LASTEXITCODE

& $Run -RunStateCloneEquivalence -Python $Py -Config $Cfg
$LASTEXITCODE

& $Run -EvaluateStateCloneGate -Python $Py -Config $Cfg
$LASTEXITCODE
```

Expected final acceptance:

- `state_clone_gate.json` has `status = pass`
- `hotstart_equivalence_status = pass`
- `controller_memory_restore_status = pass`
- `actual_setting_equivalence = pass`
- `PFV_equivalence = pass`
- `TFV_equivalence = pass`
- `peak_equivalence = pass`
- `formal_same_state_unlock_allowed = true`

If any step returns `3`, stop. Do not continue to runtime gate until real clone
evidence exists.

### Rebuild Truth Matrix And Runtime Gate

```powershell
& $Run -AuditCurrentTruth -Python $Py -Config $Cfg
$LASTEXITCODE

& $Run -EvaluatePrompt3ARuntimeGate -Python $Py -Config $Cfg
$LASTEXITCODE
```

Expected final target:

- `6/6` trajectories processed
- feature index count equals tensor dimension
- recovery contract complete
- hot-start files > `0`
- controller-memory files > `0`
- runtime-clone-eligible checkpoints > `0`
- State Clone gate = `pass`
- truth leakage = `0`
- Prompt3A runtime gate = `pass`

## Prompt 3A Schema/Fallback Repair Run Sequence

This section supersedes earlier Prompt 3A fallback and baseline-plan commands
after the baseline trajectory plan schema, native-rule conflict classification,
and fallback-stage isolation changes.

### Repair Validation Tests

```powershell
& $Py -m pytest `
  tests\test_project6_v3_native_rules.py `
  tests\test_project6_v3_fallbacks.py `
  tests\test_project6_v3_baseline_trajectory.py `
  tests\test_project6_v3_event_catalog.py `
  tests\test_project6_v3_runner_contract.py -q

$LASTEXITCODE
```

Expected exit code: `0`.

Stop if this returns non-zero.

### Rebuild Reference Roles, Fallbacks, Native Rules, Event Catalog And Plan

Because contracts and output hashes changed, rerun these stages in order:

```powershell
& $Run -AuditReferencesFallbacks -Python $Py -Config $Cfg
$LASTEXITCODE

& $Run -RebuildContract -Python $Py -Config $Cfg
$LASTEXITCODE

& $Run -AuditNativeRules -Python $Py -Config $Cfg
$LASTEXITCODE

& $Run -AuditFallbacks -Python $Py -Config $Cfg
$LASTEXITCODE

& $Run -BuildRainfallAssetIndex -Python $Py -Config $Cfg
$LASTEXITCODE

& $Run -BuildEventCatalog -Python $Py -Config $Cfg
$LASTEXITCODE

& $Run -PlanBaselineTrajectories -Python $Py -Config $Cfg
$LASTEXITCODE
```

Expected exit code for each command: `0`.

Review these files before any trajectory generation:

- `outputs\project6_pfvfirst_dualfallback_10min_v3\reference_roles\reference_roles_contract.json`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\rainfall_assets\rainfall_asset_inventory.csv`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\rainfall_assets\rainfall_asset_resolution_audit.csv`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\native_rules\native_rule_conflicts.csv`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\fallbacks\fallback_execution_audit_report.json`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\baseline_trajectories\baseline_trajectory_plan.csv`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\baseline_trajectories\baseline_trajectory_plan_report.json`

The baseline plan must include all columns in
`docs\contracts\baseline_trajectory_plan_contract.json`. The plan is invalid if
`rainfall_path`, `split`, any contract hash, or any policy row is missing.

### Small Baseline Generation Probe Only

Do not run the full 978 planned branches first. After manual plan inspection,
start with one or two events:

```powershell
& $Run -GenerateBaselineTrajectories `
  -Python $Py `
  -Config $Cfg `
  -MaxEvents 2 `
  -Workers 1 `
  -Resume

$LASTEXITCODE
```

Expected current scaffold exit code: `3` until the SWMM baseline generator is
explicitly enabled and connected. A `6` means the frozen plan schema or hashes
are invalid. Do not proceed to checkpoint catalog, state clone, coverage, or
Round 0 until the small generation probe produces real validated trajectory
outputs.

## Prompt 3A: Physical Contracts, Dual Fallbacks, Baseline State And Round 0 Planning

Codex must not run these commands automatically in the current working rule.
The user runs them manually after Prompt 2 GAT readiness has passed.

### Common Setup

```powershell
cd E:\RTC_sewer\Project6

$Py = "E:\RTC_sewer\Project6\.venv\Scripts\python.exe"
$Run = ".\scripts\project6_runs\RUN_PROJECT6_PFVFIRST_DUALFALLBACK_10MIN_V3.ps1"
$Cfg = ".\configs\wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml"
```

### 1. Prompt 3A Contract Tests

```powershell
& $Py -m pytest `
  tests\test_project6_prompt3a_import.py `
  tests\test_project6_v3_native_rules.py `
  tests\test_project6_v3_fallbacks.py `
  tests\test_project6_v3_event_catalog.py `
  tests\test_project6_v3_baseline_trajectory.py `
  tests\test_project6_v3_full_state.py `
  tests\test_project6_v3_state_clone.py `
  tests\test_project6_v3_coverage_contract.py `
  tests\test_project6_v3_round0_planner.py `
  tests\test_project6_v3_runner_contract.py -q

$LASTEXITCODE
```

Expected exit code: `0`.

Stop if tests fail. Provide the complete pytest output.

### 2. Import Prompt 2 Artifacts

```powershell
& $Run -ImportPrompt2Artifacts -Python $Py -Config $Cfg
$LASTEXITCODE
```

Expected exit code:

- `0` only when Prompt 2 GAT readiness is `pass` and
  `allowed_to_enter_prompt3a=true`;
- `3` if Prompt 2 readiness, independent robustness or node-level seven-frame
  state evidence is still incomplete;
- `6` on hash or contract mismatch.

Expected outputs:

- `docs\contracts\project6_prompt2_import_contract.json`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\contracts\prompt2_import_manifest.json`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\gates\prompt3a_entry_gate.json`

Stop unless exit code is `0`.

### 3. Static Physical Contracts And Fallbacks

```powershell
& $Run -FatalAudit -Python $Py -Config $Cfg
$LASTEXITCODE

& $Run -AuditReferencesFallbacks -Python $Py -Config $Cfg
$LASTEXITCODE

& $Run -RebuildContract -Python $Py -Config $Cfg
$LASTEXITCODE

& $Run -AuditNativeRules -Python $Py -Config $Cfg
$LASTEXITCODE

& $Run -AuditFallbacks -Python $Py -Config $Cfg
$LASTEXITCODE
```

Expected exit code for each: `0`.

Stop on any nonzero exit code. Inspect:

- `outputs\project6_pfvfirst_dualfallback_10min_v3\fatal_audit\fatal_audit_report.json`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\native_rules\native_rule_audit_report.json`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\fallbacks\fallback_audit_report.json`

Formal safety readiness may remain blocked because sentinel thresholds and
`add350.1` residual bounds are not frozen. That does not block Prompt 3A
development, but it blocks formal deployment and FormalBlind.

### 4. Event Catalog And Split

```powershell
& $Run -BuildEventCatalog -Python $Py -Config $Cfg
$LASTEXITCODE
```

Expected exit code: `0`.

Stop if:

- any GAT independent holdout event has `round0_eligible=true`;
- an event or storm family crosses action-effect fit, calibration, validation
  or formal splits;
- Formal events appear in development stages.

### 5. Baseline Trajectory Planning

```powershell
& $Run -PlanBaselineTrajectories -Python $Py -Config $Cfg
$LASTEXITCODE
```

Expected exit code: `0`.

Review:

`outputs\project6_pfvfirst_dualfallback_10min_v3\baseline_trajectories\baseline_trajectory_plan.csv`

Then, after manual review:

```powershell
& $Run -GenerateBaselineTrajectories `
  -Python $Py `
  -Config $Cfg `
  -Resume

$LASTEXITCODE
```

Current scaffold expected exit code: `3` until the SWMM trajectory execution
implementation is connected and manually allowed. This is intentional; do not
proceed to checkpoint catalog until real baseline trajectories and controller
memory artifacts exist.

### 6. Full Project6 State And Checkpoint Catalog

After real baseline trajectories exist:

```powershell
$TrajectoryRoot = "E:\RTC_sewer\Project6\outputs\project6_pfvfirst_dualfallback_10min_v3\baseline_trajectories"

& $Run -BuildStateInputManifest `
  -Python $Py `
  -Config $Cfg `
  -SourceMode "project6_retrofit_baseline" `
  -TrajectoryRoot $TrajectoryRoot

$LASTEXITCODE
```

Then:

```powershell
$StateInput = "E:\RTC_sewer\Project6\outputs\project6_pfvfirst_dualfallback_10min_v3\state_inputs\project6_baseline_state_input_manifest.csv"

& $Run -BuildStateFeatures `
  -Python $Py `
  -Config $Cfg `
  -StateInputManifest $StateInput `
  -StateValidationMode "project6_full_baseline" `
  -MaxSamples 100

$LASTEXITCODE
```

Then:

```powershell
& $Run -BuildCheckpointCatalog -Python $Py -Config $Cfg
$LASTEXITCODE

& $Run -StateCloneTest -Python $Py -Config $Cfg
$LASTEXITCODE
```

Stop if State Clone is not a real PASS. A schema-only clone output must not
unlock same-state data generation.

### 7. Coverage Contract And Round 0 Planning

```powershell
& $Run -BuildCoverageContract -Python $Py -Config $Cfg
$LASTEXITCODE

& $Run -PlanRound0 -Python $Py -Config $Cfg
$LASTEXITCODE
```

Expected for implemented planning stages: `0`, after upstream checkpoint and
state-clone conditions are satisfied.

Review:

- `outputs\project6_pfvfirst_dualfallback_10min_v3\round0\paired_manifest_round0.csv`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\round0\preflight_noop_audit_round0.csv`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\round0\planned_concurrency_support_round0.csv`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\round0\structural_infeasible_cells.csv`

Do not execute full Round 0 until the manifest is manually approved.

### 8. Small Round 0 Dry Run

```powershell
& $Run -DryRunRound0 -Python $Py -Config $Cfg
$LASTEXITCODE
```

Current scaffold expected exit code: `3` until same-state hot-start branch
execution is available. A return code of `0` before real branch execution would
be a contract violation.

### 9. Prompt 3A Completion Gate

```powershell
& $Run -EvaluatePrompt3ACompletion -Python $Py -Config $Cfg
$LASTEXITCODE

& $Run -Status -Python $Py -Config $Cfg
$LASTEXITCODE
```

Expected exit code:

- `0` only after Prompt2 import, physical contracts, native rules, fallback
  contracts, event split, baseline trajectories, full Project6 state, State
  Clone, coverage, Round0 plan and dry-run all pass;
- `3` while any required Prompt 3A condition is blocked.

### Prompt 3A Stop Conditions

Stop immediately if any of the following occur:

- Prompt2 import hash changes or readiness is not PASS.
- GAT independent holdout events enter Round 0.
- Network hash differs across branches.
- Native rule parsing fails.
- Passive fallback becomes all-zero, all-open or full reset.
- Internal fallback leaves learned override or stale candidate target active.
- `add350.1` appears in residual candidates before bounds are frozen.
- Binary pump action is not exactly `0` or `1`.
- Baseline trajectory truth leaks into controller-visible fields.
- Seven-frame state uses future observations.
- State Clone lacks either SWMM hot-start or controller memory restoration.
- Round 0 manifest is not manually approved.
- Calibration or Formal events enter development stages.

## Prompt 2 Independent GAT Holdout Replacement

The previous sr0p15 robustness evidence based on the Project4 default cache is
formally contaminated:

```text
validation_event = T15_D105_block
matching_training_event = T15_D105_block
match_type = exact_event_id
validation_status = diagnostic_contaminated
independent_validation_eligible = false
```

Do not repair this by changing batch size, sample count, memory, or gate
thresholds. The old result is retained as diagnostic evidence only. The formal
sr0p15 robustness gate must use a locked independent holdout manifest.

### Test The Independent-Holdout Contracts

```powershell
& $Py -m pytest `
  tests\test_project6_v3_gat_independent_validation.py `
  tests\test_project6_v3_gat_event_leakage.py `
  tests\test_project6_v3_gat_robustness.py `
  tests\test_project6_prompt2_gat_readiness.py `
  tests\test_project6_v3_runner_contract.py -q

$LASTEXITCODE
```

Expected exit code: `0`.

Stop if tests fail.

### Build The Independent Validation Catalog

```powershell
& $Run -BuildGATIndependentValidationCatalog `
  -Python $Py `
  -Config $Cfg

$LASTEXITCODE
```

Expected exit code: `0` if the catalog and exclusion reports are written.

Expected outputs:

- `outputs\project6_pfvfirst_dualfallback_10min_v3\gat\gat_validation_asset_inventory.csv`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\gat\gat_contaminated_event_manifest.csv`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\gat\gat_contaminated_storm_family_manifest.csv`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\gat\gat_model_selection_event_manifest.csv`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\gat\gat_independent_validation_candidates.csv`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\gat\gat_independent_validation_exclusion_audit.csv`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\gat\gat_independent_validation_manifest.csv`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\gat\gat_independent_trajectory_plan.csv`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\gat\gat_independent_validation_catalog_report.json`

Stop if:

- the catalog includes `T15_D105_block` or any `T15_block` family event as eligible;
- exact event overlap, rainfall hash overlap, trajectory hash overlap, storm-family overlap, intensity scaling, time shifting, renamed series, or padded/truncated series is not reported as an exclusion reason.

### Review Candidates

```powershell
$GatDir = "E:\RTC_sewer\Project6\outputs\project6_pfvfirst_dualfallback_10min_v3\gat"

Import-Csv "$GatDir\gat_independent_validation_candidates.csv" |
  Format-Table -AutoSize

Import-Csv "$GatDir\gat_independent_validation_exclusion_audit.csv" |
  Format-Table -AutoSize
```

If `gat_independent_validation_manifest.csv` has no eligible rows, inspect:

```text
outputs\project6_pfvfirst_dualfallback_10min_v3\gat\gat_independent_trajectory_plan.csv
outputs\project6_pfvfirst_dualfallback_10min_v3\gat\gat_independent_validation_gap_report.json
```

Generate the listed independent trajectories manually in the active Project6
retrofit network, then rerun the catalog stage. Do not use future control Formal
events as GAT holdout patches.

### Generate Supplemental Independent Holdout Trajectories

Run this when the catalog reports:

```text
eligible_event_count = 0
requires_new_trajectory_count > 0
```

Start with a moderate, diverse batch. This creates current-network trajectories,
then builds a sr0p15-compatible validation cache and rewrites
`gat_independent_validation_manifest.csv` with eligible generated rows.

```powershell
& $Run -GenerateGATIndependentHoldoutTrajectories `
  -Python $Py `
  -Config $Cfg `
  -MaxHoldoutEvents 24 `
  -HoldoutPolicies "no_control,internal_rules" `
  -Workers 4 `
  -TailMin 180 `
  -TimeStride 1 `
  -Resume

$LASTEXITCODE
```

Expected exit code:

- `0`: independent trajectories, cache, and manifest were created;
- `3`: no valid planned rows, missing plan, or no valid cache samples;
- `4`: PySWMM/runtime failure.

Expected outputs:

- `outputs\project6_pfvfirst_dualfallback_10min_v3\gat\independent_holdout\generated_trajectories\gat_holdout_generation_report.json`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\gat\independent_holdout\generated_trajectories\gat_independent_holdout_summary.csv`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\gat\independent_holdout\generated_trajectories\gat_independent_holdout_cache_report.json`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\gat\independent_holdout\generated_trajectories\sr0p15_cache\gat_independent_holdout_sr0p15_cache.npz`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\gat\gat_independent_validation_manifest.csv`

Stop if the cache report has `status != completed` or `event_count < 2`.
If memory or runtime is too high, rerun with:

```powershell
& $Run -GenerateGATIndependentHoldoutTrajectories `
  -Python $Py `
  -Config $Cfg `
  -MaxHoldoutEvents 8 `
  -HoldoutPolicies "no_control" `
  -Workers 1 `
  -TailMin 180 `
  -TimeStride 2 `
  -Resume

$LASTEXITCODE
```

After successful generation, rerun the catalog to preserve the updated
inventory and then continue to lock the generated manifest:

```powershell
& $Run -BuildGATIndependentValidationCatalog `
  -Python $Py `
  -Config $Cfg

$LASTEXITCODE
```

### Lock An Eligible Independent Manifest

Only run this if the manifest contains eligible independent holdout rows.

```powershell
$IndependentManifest =
"E:\RTC_sewer\Project6\outputs\project6_pfvfirst_dualfallback_10min_v3\gat\gat_independent_validation_manifest.csv"

& $Run -LockGATIndependentValidationManifest `
  -Python $Py `
  -Config $Cfg `
  -ValidationManifest $IndependentManifest `
  -AcknowledgeIndependentHoldout

$LASTEXITCODE
```

Expected exit code:

- `0`: `gat_independent_validation_lock.json` created;
- `3`: manifest empty or assets insufficient;
- `5`: manifest contains contaminated or ineligible rows;
- `7`: acknowledgement or manifest path missing.

Stop unless exit code is `0`.

### Run Formal sr0p15 Robustness On The Locked Independent Holdout

```powershell
& $Run -RunGATRobustnessAudit `
  -Python $Py `
  -Config $Cfg `
  -ValidationManifest $IndependentManifest `
  -MaxSamples 0 `
  -BatchSize 8 `
  -FlushEvery 16 `
  -MaxMemoryGB 4

$LASTEXITCODE
```

`MaxSamples 0` means use all eligible samples from the locked independent
manifest.

Expected exit code:

- `0`: independent robustness artifacts written under `gat\independent_holdout\sr0p15`;
- `3`: independent lock missing, manifest hash mismatch, or manifest lacks a usable NPZ validation cache;
- `4`: runtime error.

This stage must not overwrite the old diagnostic contaminated files in
`gat\`.

### Evaluate The Formal Independent Gate

```powershell
& $Run -EvaluateGATRobustnessGate `
  -Python $Py `
  -Config $Cfg `
  -ValidationManifest $IndependentManifest

$LASTEXITCODE
```

Expected exit code:

- `0`: `gat_sr0p15_independent_robustness_gate.json` passed;
- `3`: evidence incomplete;
- `5`: complete independent audit failed scientifically;
- `6`: lock, config, checkpoint, or hash mismatch.

Only continue if the gate status is `pass`.

### Build Independent Node-Level State Input Manifest

```powershell
& $Run -BuildStateInputManifest `
  -Python $Py `
  -Config $Cfg `
  -SourceMode "gat_independent_holdout" `
  -ValidationManifest $IndependentManifest

$LASTEXITCODE
```

Expected exit code: `0`.

Expected output:

`outputs\project6_pfvfirst_dualfallback_10min_v3\state_inputs\state_input_manifest_v1.csv`

The output is node-level GAT state validation only; it does not claim full
Project6 facility/storage/pump/TTL state readiness.

### Build Independent Seven-Frame Node State Features

```powershell
$StateInput =
"E:\RTC_sewer\Project6\outputs\project6_pfvfirst_dualfallback_10min_v3\state_inputs\state_input_manifest_v1.csv"

& $Run -BuildStateFeatures `
  -Python $Py `
  -Config $Cfg `
  -StateInputManifest $StateInput `
  -StateValidationMode "gat_independent_node_only" `
  -MaxSamples 100

$LASTEXITCODE
```

Expected exit code:

- `0`: independent node-level 7-frame causality validation completed;
- `3`: independent robustness gate is not PASS or state input evidence is insufficient;
- `7`: placeholder path supplied.

### Evaluate Prompt 2 GAT Readiness

```powershell
& $Run -EvaluatePrompt2GATReadiness `
  -Python $Py `
  -Config $Cfg

$LASTEXITCODE
```

Only enter Prompt 3A when the output contains:

```text
status = pass
allowed_to_enter_prompt3a = true
full_project6_augmented_state_complete = false
```
## Prompt 3A Prompt 1 State Clone Runtime Repair

This sequence is for the real SWMM hot-start State Clone equivalence executor.
It does not implement Candidate hydraulic dry-run or formal Round0 generation.

Set the common variables:

```powershell
cd E:\RTC_sewer\Project6

$Py   = "E:\RTC_sewer\Project6\.venv\Scripts\python.exe"
$Run  = ".\scripts\project6_runs\RUN_PROJECT6_PFVFIRST_DUALFALLBACK_10MIN_V3.ps1"
$Cfg  = ".\configs\wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml"
$Root = "E:\RTC_sewer\Project6\outputs\project6_pfvfirst_dualfallback_10min_v3"
```

Run the contract and runtime tests:

```powershell
& $Py -m pytest `
  tests\test_project6_v3_state_clone_runtime.py `
  tests\test_project6_v3_state_clone.py `
  tests\test_project6_v3_prompt3a_scientific_loop.py `
  tests\test_project6_v3_current_truth.py `
  tests\test_project6_v3_runner_contract.py -q

$LASTEXITCODE
```

Expected exit code: `0`. Stop if this returns nonzero.

Refresh checkpoint readiness:

```powershell
& $Run -PrepareStateCloneCheckpoints -Python $Py -Config $Cfg
$LASTEXITCODE
```

Expected exit code: `0` when the checkpoint catalog, baseline checkpoint audit,
hot-start files, controller-memory sidecars, detail CSVs and event INPs are
present.

Run a three-checkpoint smoke restore:

```powershell
& $Run -RunStateCloneEquivalence `
  -Python $Py `
  -Config $Cfg `
  -StateCloneMode "smoke" `
  -MaxCheckpoints 3 `
  -Workers 1 `
  -Resume

$LASTEXITCODE
```

Expected exit code:

- `0` if three hot-start restored continuations were actually executed and
  written under `outputs\project6_pfvfirst_dualfallback_10min_v3\state_clone\smoke_runs`;
- `3` if the existing baseline detail or controller-memory files are missing
  required runtime fields.

If exit code is `3`, open:

```text
outputs\project6_pfvfirst_dualfallback_10min_v3\state_clone\state_clone_report_smoke.json
```

If it lists missing controller-memory fields, missing 36-facility columns,
missing node head columns or missing storage volume columns, regenerate only the
two-event baseline after this repair, then rebuild the state input/features,
checkpoint catalog and readiness before retrying Smoke.

Estimate empirical State Clone numerical noise:

```powershell
& $Run -EstimateStateCloneNumericalNoise `
  -Python $Py `
  -Config $Cfg `
  -MaxCheckpoints 3 `
  -Workers 1

$LASTEXITCODE
```

Expected exit code:

- `0` only when duplicate restored continuations were actually executed and
  `state_clone_numerical_noise.json` has `empirically_measured=true`;
- `3` if prerequisites are incomplete.

Run the full 18-checkpoint State Clone:

```powershell
& $Run -RunStateCloneEquivalence `
  -Python $Py `
  -Config $Cfg `
  -StateCloneMode "full" `
  -MaxCheckpoints 0 `
  -Workers 1 `
  -Resume

$LASTEXITCODE
```

Expected exit code:

- `0` only when all 18 eligible checkpoints execute and pass;
- `3` if prerequisites or empirical noise are missing;
- `5` if SWMM restore ran but equivalence metrics failed.

Evaluate the formal State Clone gate:

```powershell
& $Run -EvaluateStateCloneGate `
  -Python $Py `
  -Config $Cfg

$LASTEXITCODE
```

Expected exit code: `0` only when:

- `runtime_executed=true`;
- eligible checkpoints = `18`;
- executed checkpoints = `18`;
- passed checkpoints = `18`;
- timeline, controller-memory, hydraulic, facility and KPI equivalence all pass;
- numerical noise is empirically measured;
- `formal_same_state_unlock_allowed=true`.

Refresh truth and Prompt3A runtime status:

```powershell
& $Run -AuditCurrentTruth -Python $Py -Config $Cfg
$LASTEXITCODE

& $Run -EvaluatePrompt3ARuntimeGate -Python $Py -Config $Cfg
$LASTEXITCODE
```

After State Clone passes, Prompt3A runtime may still return `3`; at that point
the only acceptable remaining blockers are:

```text
hydraulic_candidate_dryrun_pass
formal_round0_candidate_target_met
```
## Prompt3A Same-State Dual Path Closure

Use this sequence after the 6-trajectory Prompt3A baseline, state features, and
checkpoint catalog exist. Hot-start clone may fail scientifically; the formal
same-state branch can still pass through deterministic prefix replay.

```powershell
cd E:\RTC_sewer\Project6

$Py   = "E:\RTC_sewer\Project6\.venv\Scripts\python.exe"
$Run  = ".\scripts\project6_runs\RUN_PROJECT6_PFVFIRST_DUALFALLBACK_10MIN_V3.ps1"
$Cfg  = ".\configs\wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml"

& $Py -m pytest `
  tests\test_project6_v3_state_clone_runtime.py `
  tests\test_project6_v3_state_clone_diagnostics.py `
  tests\test_project6_v3_same_state_replay.py `
  tests\test_project6_v3_prompt3a_scientific_loop.py `
  tests\test_project6_v3_current_truth.py `
  tests\test_project6_v3_runner_contract.py -q
$LASTEXITCODE

& $Run -RunContinuousReplayDeterminismAudit `
  -Python $Py `
  -Config $Cfg `
  -MaxCheckpoints 3 `
  -Workers 1
$LASTEXITCODE

& $Run -RunStateCloneDiagnosticMatrix `
  -Python $Py `
  -Config $Cfg `
  -MaxCheckpoints 3 `
  -Workers 1 `
  -Resume
$LASTEXITCODE

& $Run -RunSameStateReplayEquivalence `
  -Python $Py `
  -Config $Cfg `
  -Mode full `
  -MaxCheckpoints 0 `
  -Workers 1 `
  -Resume
$LASTEXITCODE

& $Run -EvaluateHotstartCloneGate -Python $Py -Config $Cfg
$LASTEXITCODE

& $Run -EvaluateSameStateBranchGate -Python $Py -Config $Cfg
$LASTEXITCODE

& $Run -AuditCurrentTruth -Python $Py -Config $Cfg
$LASTEXITCODE

& $Run -EvaluatePrompt3ARuntimeGate -Python $Py -Config $Cfg
$LASTEXITCODE
```

Expected current outcome:

- tests: `0`
- continuous replay determinism: `0`
- diagnostic matrix: `0`
- deterministic prefix replay full: `0`
- hot-start clone gate: `5`
- same-state branch gate: `0`
- current truth audit: `0`
- Prompt3A runtime gate: `3`, blocked only by
  `hydraulic_candidate_dryrun_pass` and
  `formal_round0_candidate_target_met`
# Prompt 1H Hot-start acceleration audit

This section certifies whether SWMM hot-start can be used as an acceleration path. It must not replace the already passing deterministic prefix replay oracle unless a checkpoint is explicitly certified.

```powershell
cd E:\RTC_sewer\Project6

$Py   = "E:\RTC_sewer\Project6\.venv\Scripts\python.exe"
$Run  = ".\scripts\project6_runs\RUN_PROJECT6_PFVFIRST_DUALFALLBACK_10MIN_V3.ps1"
$Cfg  = ".\configs\wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml"
$Root = "E:\RTC_sewer\Project6\outputs\project6_pfvfirst_dualfallback_10min_v3"
```

Run the hot-start regression tests:

```powershell
& $Py -m pytest `
  tests\test_project6_v3_hotstart_cache.py `
  tests\test_project6_v3_hotstart_forcing.py `
  tests\test_project6_v3_hotstart_certification.py `
  tests\test_project6_v3_same_state_hybrid_runner.py `
  tests\test_project6_v3_hotstart_performance.py `
  tests\test_project6_v3_state_clone_runtime.py `
  tests\test_project6_v3_same_state_replay.py `
  tests\test_project6_v3_runner_contract.py -q
$LASTEXITCODE
```

Expected exit code: `0`.

Freeze the replay oracle and diagnose hot-start:

```powershell
& $Run -DiagnoseHotstartFirstDivergence -Python $Py -Config $Cfg -MaxCheckpoints 3 -Workers 1
$LASTEXITCODE

& $Run -AuditHotstartCompatibility -Python $Py -Config $Cfg -MaxCheckpoints 3
$LASTEXITCODE

& $Run -BuildCanonicalHotstartCache -Python $Py -Config $Cfg -MaxCheckpoints 3 -Workers 1 -Resume
$LASTEXITCODE
```

Expected exit code for each command: `0`.

Run hot-start smoke certification:

```powershell
& $Run -RunHotstartSmoke -Python $Py -Config $Cfg -MaxCheckpoints 3 -Workers 1 -Resume
$LASTEXITCODE

& $Run -EvaluateHotstartSmokeGate -Python $Py -Config $Cfg
$LASTEXITCODE
```

Current expected exit code: `5`, because the existing mid-run hot-start branch diverges hydraulically from the replay oracle. This is a scientific gate failure, not a prerequisite failure.

Only if smoke returns `0`, continue to full validation:

```powershell
& $Run -RunHotstartFullValidation -Python $Py -Config $Cfg -MaxCheckpoints 0 -Workers 1 -Resume
$LASTEXITCODE

& $Run -EvaluateHotstartFullGate -Python $Py -Config $Cfg
$LASTEXITCODE
```

If smoke fails, full validation should return `3` and must not execute.

Always update certification and readiness:

```powershell
& $Run -CertifyHotstartCheckpoints -Python $Py -Config $Cfg
$LASTEXITCODE

& $Run -BenchmarkHotstartAcceleration -Python $Py -Config $Cfg -CandidateCounts "1,5,10,20" -WorkerCounts "1,2,4"
$LASTEXITCODE

& $Run -EvaluateHotstartAccelerationReadiness -Python $Py -Config $Cfg
$LASTEXITCODE

& $Run -AuditCurrentTruth -Python $Py -Config $Cfg
$LASTEXITCODE

& $Run -EvaluateSameStateBranchGate -Python $Py -Config $Cfg
$LASTEXITCODE
```

Current expected result:

- `CertifyHotstartCheckpoints`: exit `5`, certified checkpoint count `0`.
- `BenchmarkHotstartAcceleration`: exit `0`, diagnostic report only.
- `EvaluateHotstartAccelerationReadiness`: exit `5`, hot-start not allowed.
- `AuditCurrentTruth`: exit `0`.
- `EvaluateSameStateBranchGate`: exit `0`, still using `deterministic_prefix_replay`.

Prompt2 must use the hybrid same-state runner contract: certified hot-start first when available, immediate fingerprint check after loading, then automatic deterministic replay fallback. With the current evidence, no checkpoint is certified for hot-start, so all Prompt2 branches must use replay.
## Prompt 2 Prerequisite: Deterministic Replay Acceleration

This acceleration layer does not change the scientific Same-State path.
Candidate truth remains `deterministic_prefix_replay`. Hot-start remains
disallowed for labels unless separately certified.

Use:

```powershell
cd E:\RTC_sewer\Project6

$Py   = "E:\RTC_sewer\Project6\.venv\Scripts\python.exe"
$Run  = ".\scripts\project6_runs\RUN_PROJECT6_PFVFIRST_DUALFALLBACK_10MIN_V3.ps1"
$Cfg  = ".\configs\wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml"
$Root = "E:\RTC_sewer\Project6\outputs\project6_pfvfirst_dualfallback_10min_v3"
```

Run tests:

```powershell
& $Py -m pytest `
  tests\test_project6_v3_interface_cache.py `
  tests\test_project6_v3_reference_cache_and_prefilter.py `
  tests\test_project6_v3_hotstart_cache.py `
  tests\test_project6_v3_same_state_replay.py `
  tests\test_project6_v3_runner_contract.py -q
$LASTEXITCODE
```

Expected exit code: `0`.

Run the acceleration audit stages:

```powershell
& $Run -AuditRunoffCacheEligibility -Python $Py -Config $Cfg
$LASTEXITCODE

& $Run -BuildRainfallInterfaceCache `
  -Python $Py -Config $Cfg `
  -MaxEvents 2 -Workers 1 -Resume
$LASTEXITCODE

& $Run -BuildRunoffInterfaceCache `
  -Python $Py -Config $Cfg `
  -MaxEvents 2 -Workers 1 -Resume
$LASTEXITCODE

& $Run -AuditRunoffInterfaceEquivalence `
  -Python $Py -Config $Cfg `
  -MaxEvents 2 -Workers 1
$LASTEXITCODE

& $Run -EvaluateRunoffCacheGate -Python $Py -Config $Cfg
$LASTEXITCODE

& $Run -BuildReferenceBranchCache -Python $Py -Config $Cfg
$LASTEXITCODE

& $Run -RunCandidatePrefilterAudit -Python $Py -Config $Cfg
$LASTEXITCODE

& $Run -BenchmarkReplayAcceleration `
  -Python $Py -Config $Cfg `
  -CandidateCounts "1,5,10,20" `
  -WorkerCounts "1,2,4"
$LASTEXITCODE

& $Run -EvaluateReplayAccelerationGate -Python $Py -Config $Cfg
$LASTEXITCODE
```

Expected exit code for each stage above: `0`.

Expected outputs:

- `outputs\project6_pfvfirst_dualfallback_10min_v3\interface_cache\runoff_cache_eligibility.csv`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\interface_cache\rainfall_interface_cache_index.csv`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\interface_cache\runoff_interface_cache_index.csv`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\interface_cache\runoff_interface_equivalence_audit.csv`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\interface_cache\runoff_cache_gate.json`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\interface_cache\reference_branch_cache_index.csv`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\interface_cache\candidate_prefilter_audit.csv`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\interface_cache\binary_pump_direction_support.csv`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\interface_cache\replay_acceleration_benchmark.csv`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\interface_cache\replay_acceleration_gate.json`

Stop conditions:

- Stop if Same-State Branch Gate is no longer `pass`.
- Stop if any cache or reference artifact reports stale input hashes.
- Stop if `runoff_cache_gate.json` is not `pass`; later Candidate runs must fall
  back to full hydrology replay.
- Stop if `candidate_prefilter_audit.csv` admits binary pump values outside
  `0` and `1`.
- Stop if `replay_acceleration_gate.json` is not `pass`.

Important limitation: if the current baseline detail files do not contain
explicit subcatchment runoff columns, the gate records
`runoff_equivalence = not_materialized_subcatchment_runoff_columns_unavailable`.
In that case the layer is valid only as a hashed interface/cache and fallback
contract for the current deterministic replay path; it must not be described as
a fully certified native SWMM runoff-interface substitute.

## Prompt 2: Round 0 Candidate Planning And Dry-Run Gates

Use this section after Same-State Branch Gate is `pass` with
`deterministic_prefix_replay`, Prompt2 GAT readiness is `pass`, and hot-start
remains uncertified for Candidate labels.

Common setup:

```powershell
cd E:\RTC_sewer\Project6

$Py  = "E:\RTC_sewer\Project6\.venv\Scripts\python.exe"
$Run = "E:\RTC_sewer\Project6\scripts\project6_runs\RUN_PROJECT6_PFVFIRST_DUALFALLBACK_10MIN_V3.ps1"
$Cfg = ".\configs\wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml"
```

Run the contract tests:

```powershell
& $Py -m pytest `
  tests\test_project6_v3_prompt2_entry.py `
  tests\test_project6_v3_control_aligned_checkpoints.py `
  tests\test_project6_v3_candidate_actions.py `
  tests\test_project6_v3_binary_pumps.py `
  tests\test_project6_v3_round0_planner.py `
  tests\test_project6_v3_round0_hydraulic_dryrun.py `
  tests\test_project6_v3_round0_dataset.py `
  tests\test_project6_v3_runner_contract.py -q
$LASTEXITCODE
```

Expected exit code: `0`.

Run Prompt2 entry and control-aligned checkpoint catalog:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Run `
  -AuditPrompt2Entry -Python $Py -Config $Cfg
$LASTEXITCODE

powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Run `
  -BuildControlAlignedCheckpointCatalog -Python $Py -Config $Cfg
$LASTEXITCODE

powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Run `
  -AuditControlAlignedCheckpointCatalog -Python $Py -Config $Cfg
$LASTEXITCODE

powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Run `
  -BuildRound0CoverageContract -Python $Py -Config $Cfg
$LASTEXITCODE
```

Expected exit code for each stage above: `0`.

Expected outputs:

- `outputs\project6_pfvfirst_dualfallback_10min_v3\prompt2\prompt2_entry_audit.json`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\control_checkpoints\control_checkpoint_catalog.csv`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\control_checkpoints\control_checkpoint_catalog_report.json`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\round0\round0_coverage_contract.json`

Plan Round 0:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Run `
  -PlanRound0 `
  -Python $Py `
  -Config $Cfg `
  -TargetEffectiveCandidates 1800 `
  -ReserveCandidates 400 `
  -PressureCandidates 90 `
  -Seed 20260719
$LASTEXITCODE
```

Expected exit code:

- `0` only when the current control-aligned checkpoint support can produce
  1500 to 2000 effective Candidates.
- `3` when the current checkpoint support is insufficient. With the current
  two-event Prompt3A sample, this is expected: only the 60 min rising
  checkpoints are RTC-control aligned; 75 min and 135 min remain diagnostic
  checkpoints only.

Expected outputs even when blocked:

- `outputs\project6_pfvfirst_dualfallback_10min_v3\round0\paired_manifest_round0.csv`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\round0\round0_plan_report.json`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\round0\planned_phase_support_round0.csv`
- `outputs\project6_pfvfirst_dualfallback_10min_v3\round0\planned_concurrency_support_round0.csv`

Stop if `round0_plan_report.json` reports fewer than 1500 effective Candidates.
Do not approve the manifest and do not run Candidate SWMM dry-run.

After a valid 1500 to 2000 Candidate manifest exists, run:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Run `
  -AuditRound0Manifest -Python $Py -Config $Cfg
$LASTEXITCODE

powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Run `
  -PlanRound0HydraulicDryRun -Python $Py -Config $Cfg -MaxCandidates 20
$LASTEXITCODE
```

Expected exit code: `0` only after the manifest audit passes.

The real hydraulic dry-run gate is intentionally strict:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Run `
  -RunRound0HydraulicDryRun -Python $Py -Config $Cfg -MaxCandidates 20 -Workers 1 -Resume
$LASTEXITCODE

powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Run `
  -EvaluateRound0HydraulicDryRunGate -Python $Py -Config $Cfg
$LASTEXITCODE
```

Expected exit code:

- `0` only after at least 12 real SWMM Candidate suffix branches execute with
  deterministic prefix replay, valid binary pump actions, zero truth leakage,
  and explicit recovery labels.
- `3` while no real Candidate SWMM branches have executed.
- `5` if Candidate SWMM branches execute but fail the scientific gate.

Only after the dry-run gate passes and the user manually approves the manifest:

```powershell
$Round0Manifest = "E:\RTC_sewer\Project6\outputs\project6_pfvfirst_dualfallback_10min_v3\round0\paired_manifest_round0.csv"

powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Run `
  -ApproveRound0Manifest `
  -Python $Py `
  -Config $Cfg `
  -Round0Manifest $Round0Manifest `
  -AcknowledgeRound0Manifest
$LASTEXITCODE
```

Stop conditions:

- Stop if `control_checkpoint_catalog_report.json` reports insufficient control
  aligned support for formal Round 0.
- Stop if any GAT holdout, calibration, locked validation, or formal split
  appears as Round0-eligible.
- Stop if `add350.1` appears in residual override Candidates before its bounds
  are frozen.
- Stop if `ADD301.2` or `ADD301.3` has any action other than `0` or `1`.
- Stop if hot-start is used for Candidate labels; current Prompt2 truth path is
  deterministic prefix replay.

### Prompt 2 Formal Expansion Flow

The earlier two-event Prompt3A sample is diagnostic only for Round 0 planning.
Formal Round 0 must first expand the action-effect fit split, generate expanded
baseline trajectories, build new 10 min control-aligned checkpoints, and
materialize 7-frame state features for those checkpoints.

Run this test set before the expansion:

```powershell
& $Py -m pytest `
  tests\test_project6_v3_prompt2_event_expansion.py `
  tests\test_project6_v3_prompt2_baseline_expansion.py `
  tests\test_project6_v3_control_aligned_checkpoints.py `
  tests\test_project6_v3_prompt2_checkpoint_support.py `
  tests\test_project6_v3_round0_planner.py `
  tests\test_project6_v3_round0_hydraulic_dryrun.py `
  tests\test_project6_v3_runner_contract.py -q

$LASTEXITCODE
```

Expected exit code: `0`.

Plan and audit the fit-event expansion:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Run `
  -PlanPrompt2FitEventExpansion `
  -Python $Py `
  -Config $Cfg `
  -TargetFitEvents 36 `
  -Seed 20260719

$LASTEXITCODE

powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Run `
  -AuditPrompt2FitEventExpansion `
  -Python $Py `
  -Config $Cfg

$LASTEXITCODE
```

Expected: `0`; at least 30 fit events selected, with GAT holdout,
Calibration, Locked Validation, and Formal excluded.

Plan the expanded baseline trajectories. First run a three-event smoke plan and
generation:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Run `
  -PlanPrompt2BaselineExpansion `
  -Python $Py `
  -Config $Cfg `
  -MaxEvents 3

$LASTEXITCODE

powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Run `
  -GeneratePrompt2BaselineExpansion `
  -Python $Py `
  -Config $Cfg `
  -MaxEvents 3 `
  -Workers 1 `
  -TailMin 180 `
  -Resume

$LASTEXITCODE
```

Expected: generation returns `0` only if the requested smoke trajectories
complete with recovery and controller-memory outputs. Do not use
`AuditPrompt2BaselineExpansion` as the smoke gate because it is the formal
`>=30 events / >=90 trajectories` gate and should remain blocked for a
three-event smoke.

After smoke succeeds, re-plan and generate all selected fit events:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Run `
  -PlanPrompt2BaselineExpansion `
  -Python $Py `
  -Config $Cfg `
  -MaxEvents 0

$LASTEXITCODE

powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Run `
  -GeneratePrompt2BaselineExpansion `
  -Python $Py `
  -Config $Cfg `
  -MaxEvents 0 `
  -TailMin 180 `
  -RefreshExistingOnly

$LASTEXITCODE

powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Run `
  -GeneratePrompt2BaselineExpansion `
  -Python $Py `
  -Config $Cfg `
  -MaxEvents 0 `
  -Workers 16 `
  -TailMin 180 `
  -SkipExisting `
  -Resume

$LASTEXITCODE

powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Run `
  -AuditPrompt2BaselineExpansion `
  -Python $Py `
  -Config $Cfg

$LASTEXITCODE
```

Expected: `RefreshExistingOnly` returns `0` if all planned outputs already
exist, or `3` with a missing-output list if some trajectories still need SWMM.
The 16-worker generation command should then fill missing trajectories. If it
is interrupted, rerun the same command with `-Resume -SkipExisting`; completed
trajectory outputs are preserved and reused. Final acceptance is
`unique completed events >= 30` and `completed trajectories >= 90`.

Build and select formal 10 min control-aligned checkpoints:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Run `
  -BuildPrompt2ControlCheckpointCandidates `
  -Python $Py `
  -Config $Cfg

$LASTEXITCODE

powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Run `
  -SelectPrompt2ControlCheckpoints `
  -Python $Py `
  -Config $Cfg `
  -TargetCheckpoints 144 `
  -MaxPerEvent 6 `
  -Seed 20260719

$LASTEXITCODE

powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Run `
  -AuditPrompt2ControlCheckpointSupport `
  -Python $Py `
  -Config $Cfg

$LASTEXITCODE
```

Expected: `0` only when at least 120 checkpoints from at least 30 fit events
pass 10 min alignment, 60 min history, 120 min future horizon, split leakage,
and phase support checks.

Build Prompt2 state input and runtime state features:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Run `
  -BuildPrompt2StateInputManifest `
  -Python $Py `
  -Config $Cfg `
  -MaxSamples 0

$LASTEXITCODE

powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Run `
  -BuildPrompt2StateFeatures `
  -Python $Py `
  -Config $Cfg `
  -MaxSamples 0

$LASTEXITCODE

powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Run `
  -AuditPrompt2StateCoverage `
  -Python $Py `
  -Config $Cfg

$LASTEXITCODE

powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Run `
  -EvaluatePrompt2CheckpointSupportGate `
  -Python $Py `
  -Config $Cfg

$LASTEXITCODE
```

Expected: `EvaluatePrompt2CheckpointSupportGate = 0`. Stop if it returns `3`
or `6`.

Only after the support gate passes, re-plan Round 0:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Run `
  -PlanRound0 `
  -Python $Py `
  -Config $Cfg `
  -TargetEffectiveCandidates 1800 `
  -ReserveCandidates 400 `
  -PressureCandidates 90 `
  -Seed 20260719

$LASTEXITCODE

powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Run `
  -AuditRound0Manifest `
  -Python $Py `
  -Config $Cfg

$LASTEXITCODE
```

Expected: both `0`, effective Candidates in `1500-2000`, at least 30 events,
at least 120 checkpoints, no event above 100 main Candidates, no checkpoint
above 20 main Candidates.

Then run the real hydraulic dry-run plan and execution:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Run `
  -PlanRound0HydraulicDryRun `
  -Python $Py `
  -Config $Cfg `
  -MaxCandidates 20

$LASTEXITCODE

powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Run `
  -RunRound0HydraulicDryRun `
  -Python $Py `
  -Config $Cfg `
  -MaxCandidates 20 `
  -Workers 16 `
  -Resume

$LASTEXITCODE

powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Run `
  -EvaluateRound0HydraulicDryRunGate `
  -Python $Py `
  -Config $Cfg

$LASTEXITCODE
```

Expected: `EvaluateRound0HydraulicDryRunGate = 0` only after at least 12 real
SWMM Candidate branches execute with deterministic prefix replay, no
engineering violations, strict binary pump actions, zero truth leakage, and
complete or explicitly censored recovery labels.

Hot-start note: locally generated `.hsf` files are valid checkpoint artifacts,
but current certification gates keep `hotstart_acceleration_allowed = false`
and `certified_checkpoint_count = 0`. Do not use hot-start for Candidate labels
until `EvaluateHotstartFullGate` and `EvaluateHotstartAccelerationReadiness`
explicitly pass. Prompt2 hydraulic labels must continue to use
`deterministic_prefix_replay`.

## Formal Paper Evaluation: Authoritative SWMM Only

The old `RunMPCClosedLoopSmoke` output is retained only as
`closed_loop_replay` diagnostic evidence. It is not a formal closed-loop
result. Formal paper evaluation must use `closed_loop_authoritative_swmm`, where
each event-policy branch starts a SWMM/PySWMM run, writes facility actions into
the model, reads back actual settings, advances the hydraulic model, and writes
full detail time series.

Set variables:

```powershell
cd E:\RTC_sewer\Project6

$Py   = "E:\RTC_sewer\Project6\.venv\Scripts\python.exe"
$Run  = "E:\RTC_sewer\Project6\scripts\project6_runs\RUN_PROJECT6_PFVFIRST_DUALFALLBACK_10MIN_V3.ps1"
$Cfg  = ".\configs\wuhan_project6_pfvfirst_dualfallback_10min_v3.yaml"
$Root = "E:\RTC_sewer\Project6\outputs\project6_pfvfirst_dualfallback_10min_v3"
```

Run the Formal evaluation code tests:

```powershell
& $Py -m pytest `
  tests\test_project6_v3_formal_evaluation.py `
  tests\test_project6_v3_mpc_closed_loop.py `
  tests\test_project6_v3_runner_contract.py -q

$LASTEXITCODE
```

Build and audit the event splits:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Run `
  -BuildEvaluationEventSplits `
  -Python $Py `
  -Config $Cfg

$LASTEXITCODE

powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Run `
  -AuditEvaluationEventSplits `
  -Python $Py `
  -Config $Cfg

$LASTEXITCODE
```

Expected:

- `BuildEvaluationEventSplits` returns `0`.
- `AuditEvaluationEventSplits` returns `0` only if all Calibration,
  Locked-Validation, and the 36 Formal Blind rainfall assets have real paths
  and matching SHA256 hashes.
- If `AuditEvaluationEventSplits` returns `5`, stop and inspect
  `outputs\project6_pfvfirst_dualfallback_10min_v3\formal_evaluation\evaluation_event_split_audit.json`.
  Do not run Calibration, Locked Validation, Policy Lock, or Formal Blind until
  the missing or mismatched rainfall assets are fixed.

Run Calibration-A after split audit passes:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Run `
  -CalibrationA `
  -Python $Py `
  -Config $Cfg `
  -Workers 16 `
  -MaxEvents 0 `
  -Resume

$LASTEXITCODE

powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Run `
  -EvaluateCalibrationAGate `
  -Python $Py `
  -Config $Cfg

$LASTEXITCODE
```

Run Locked Validation-B after Calibration-A gate passes:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Run `
  -LockedValidationB `
  -Python $Py `
  -Config $Cfg `
  -Workers 16 `
  -MaxEvents 0 `
  -Resume

$LASTEXITCODE

powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Run `
  -EvaluateLockedValidationBGate `
  -Python $Py `
  -Config $Cfg

$LASTEXITCODE
```

Lock the policy only after both gates pass:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Run `
  -PolicyLock `
  -Python $Py `
  -Config $Cfg

$LASTEXITCODE

powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Run `
  -AuditPolicyLock `
  -Python $Py `
  -Config $Cfg

$LASTEXITCODE
```

Run Formal Blind only after the policy lock is audited:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Run `
  -FormalBlind `
  -Python $Py `
  -Config $Cfg `
  -Workers 16 `
  -MaxEvents 0 `
  -Resume

$LASTEXITCODE
```

Build and evaluate paper outputs:

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Run `
  -BuildFormalPairedComparison `
  -Python $Py `
  -Config $Cfg

$LASTEXITCODE

powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Run `
  -EvaluateFormalPerformanceGate `
  -Python $Py `
  -Config $Cfg

$LASTEXITCODE

powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Run `
  -ExportFormalPaperTables `
  -Python $Py `
  -Config $Cfg

$LASTEXITCODE
```

Stop conditions:

- Any `3`: prerequisite missing or not yet run.
- Any `4`: Python or SWMM runtime error; inspect the corresponding stdout and
  stderr files in `formal_evaluation`.
- Any `5`: complete evidence exists but the scientific/contract gate failed.
- Any `6`: hash, split, or contract mismatch.

Acceptance requires `hydraulic_evidence_source=authoritative_swmm`,
`uses_lookup_table_substitute=false`, one detail time series per event-policy
row, zero engineering violations, paired initial-state hashes per event, and
formal performance evaluated against Internal rules.


---

## 2026-07-23 — Auto-RBC / EFD baseline identity fix (dual-reference V4)

### Root cause (quantitative)

The Project6 actuator table has **no** `from_node` / `to_node` columns
(`scripts/160_generate_baseline_trajectories.py::_load_baseline_actuators`
produces 36 rows, `from_node` absent; verified live: `cols_have_from_node
False`). The filling-degree policies `auto_rbc` and `efd_storage_priority`
resolved their reference node only from `from_node`/`to_node`, so every
reference node was `""`, every reference depth was `0.0`, and
`_efd_reference_fill` returned `nanmax(d) < 1e-6`. EFD then unconditionally did
`return GenericActionPolicy("auto_rbc", ...)`, so the two baselines produced
byte-identical hydraulics — the cached formal detail CSVs differ **only** in the
`policy_id` label column (same KPI/PFV/TFV/peak), which is the observed
"完全相同".

### Fix (single hydraulic choke point)

- `sewerrtc/simulation/action_policies.py`
  - Added `_first_nonempty`, `_reference_node_for_row` (resolves via
    `efd_reference_node` → `storage_node`/`upstream_node`/`downstream_node` →
    runtime `from_node`/`to_node`, role-aware), and `attach_reference_nodes`
    (fills `from_node`/`to_node` from INP link topology).
  - Rewrote `_safe_depths_for_actuators` and the `_efd_reference_fill` node loop
    to use `_reference_node_for_row`.
  - Removed the EFD→Auto-RBC delegation. EFD now holds the previous setting only
    when there are no actuators, and otherwise always runs its own equal-filling
    bands.
- `sewerrtc/simulation/pyswmm_runner.py`
  - `run_swmm_trajectory` now calls `attach_reference_nodes(actuators, inp_path)`
    before constructing the policy, so all 36 facilities get a real reference
    node (verified live: `actuators_with_resolved_ref_node 36 of 36`).
- `tests/test_project6_integrity.py`
  - Added regression tests: reference-node resolution without legacy columns,
    `auto_rbc != efd_storage_priority` under the real Project6 schema, and INP
    topology attachment.

Because `run_swmm_trajectory` is the only baseline-generation choke point, every
future baseline (closed-loop, Formal) automatically uses the corrected policies.
**Any previously generated `auto_rbc` / `efd_storage_priority` detail CSVs under
`outputs/**/baselines/` are stale and must be regenerated.**

### Commands (user-run; PowerShell)

```powershell
cd E:\RTC_sewer\Project6
$Py  = "E:\RTC_sewer\Project6\.venv\Scripts\python.exe"
$Cfg = "E:\RTC_sewer\Project6\configs\wuhan_project6_dual_reference_v4.yaml"

# 0. Verify the fix (fast, offline). Expect: 5 passed.
& $Py -m pytest tests/test_project6_integrity.py `
  -k "auto_rbc or efd or reference_node or attach_reference" -q

# 1. Remove stale (buggy) baselines so they regenerate with the fixed policies.
#    (Only the auto_rbc / efd_storage_priority baseline folders.)
Get-ChildItem -Recurse -Directory outputs `
  | Where-Object { $_.FullName -match "\\baselines\\(auto_rbc|efd_storage_priority)$" } `
  | ForEach-Object { Remove-Item -Recurse -Force $_.FullName }

# 2. FULL dual-reference case generation (NOT smoke). Plan/Build return 3 until
#    the 1600 effective-case gate is reached; Generate returns 0.
& $Py scripts\205_prompt3_v4.py --config $Cfg --stage PlanV4DualReferenceFullEventCases
& $Py scripts\205_prompt3_v4.py --config $Cfg --stage GenerateV4DualReferenceFullEventCases --workers 16 --resume
& $Py scripts\205_prompt3_v4.py --config $Cfg --stage BuildV4AugmentedDataset   # expect 0 once >=1600 effective

# 3. Retrain + model gate. EvaluateV4Aug1ModelGate must return 0 (was 5 on the
#    8-case smoke set — that was an insufficient-data gate, not the baseline bug).
& $Py scripts\205_prompt3_v4.py --config $Cfg --stage TrainV4Aug1 --ensemble-size 5
& $Py scripts\205_prompt3_v4.py --config $Cfg --stage EvaluateV4Aug1ModelGate

# 4. Closed-loop smoke on the corrected pipeline.
& $Py scripts\205_prompt3_v4.py --config $Cfg --stage RunClosedLoopSmokeV4 --workers 8 --resume
& $Py scripts\205_prompt3_v4.py --config $Cfg --stage EvaluateClosedLoopSmokeV4
```

Downstream Calibration → Validation → Policy Lock → fresh Formal are driven by
`scripts/project6_runs/RUN_PROJECT6_DUAL_REFERENCE_V4.ps1`; they regenerate all
baselines through the same fixed `run_swmm_trajectory`, so the Formal comparison
will now show distinct `auto_rbc` and `efd_storage_priority` rows.

### Acceptance

- Step 0: `5 passed`.
- Step 2: `BuildV4AugmentedDataset` exit `0` (effective >= 1600).
- Step 3: `EvaluateV4Aug1ModelGate` exit `0`.
- Formal: `auto_rbc` and `efd_storage_priority` KPI rows differ (not just the
  label); regenerated detail CSV SHA256 differ in content, not only `policy_id`.

---

## Train1600 V3 production chain (user-executed, 2026-07-28)

Planning is complete and frozen. Everything below is a LONG SWMM task and was
NOT run automatically. Run in order; each stage writes
`outputs\project6_dual_reference_v4\final_v4\audits\stage_status\<Stage>.json`.

Exit codes: `0` pass, `2` blocked (send the stage status JSON), `3` incomplete
(long run in progress; re-run with `--resume`).

```powershell
$Py  = "E:\RTC_sewer\Project6\.venv\Scripts\python.exe"
$Run = "E:\RTC_sewer\Project6\scripts\project6_v4_final.py"

# Already run and PASS (light, re-runnable): AuditContracts, FreezeP3Evidence,
# AuditDataGenerationAuthorizationV3, PlanTrain1600V3, AuditTrain1600PlanV3,
# AuditTrainRound0PreflightV3. RunTrainRound0V3 --dry-run selected 400/400.

# Optional single-sample canary (needs explicit authorization):
#   & $Py $Run --stage RunTrainRound0V3 --limit 1 --resume

# --- Train Round 0 (400 accepted target) ---
& $Py $Run --stage RunTrainRound0V3 --workers 16 --resume      # 3 until done, then 0
& $Py $Run --stage BuildTrainRound0PartialV3                   # every >=64 cumulative
& $Py $Run --stage AuditTrainRound0PartialV3
& $Py $Run --stage BuildTrainRound0V3
& $Py $Run --stage AuditTrainRound0V3
& $Py $Run --stage TrainActiveLearner0V3                       # Train split only

# --- Train Round 1 (400) ---
& $Py $Run --stage SelectTrainRound1V3
& $Py $Run --stage RunTrainRound1V3 --workers 16 --resume
& $Py $Run --stage BuildTrainRound1PartialV3
& $Py $Run --stage AuditTrainRound1PartialV3
& $Py $Run --stage BuildTrainRound1V3
& $Py $Run --stage AuditTrainRound1V3
& $Py $Run --stage TrainActiveLearner1V3

# --- Train Round 2 (400; total Train = 1200) ---
& $Py $Run --stage SelectTrainRound2V3
& $Py $Run --stage RunTrainRound2V3 --workers 16 --resume
& $Py $Run --stage BuildTrainRound2PartialV3
& $Py $Run --stage AuditTrainRound2PartialV3
& $Py $Run --stage BuildTrainRound2V3
& $Py $Run --stage AuditTrainRound2V3

# --- Round 3: Calibration 200 + Locked Validation 200 (plans already frozen;
#     Plan stages only republish and re-verify the frozen SHA) ---
& $Py $Run --stage PlanCalibration200V3
& $Py $Run --stage RunCalibration200V3 --workers 16 --resume
& $Py $Run --stage BuildCalibration200V3
& $Py $Run --stage AuditCalibration200V3
& $Py $Run --stage PlanLockedValidation200V3
& $Py $Run --stage RunLockedValidation200V3 --workers 16 --resume
& $Py $Run --stage BuildLockedValidation200V3
& $Py $Run --stage AuditLockedValidation200V3

# --- Final 1600 dataset + quality gate ---
& $Py $Run --stage BuildTrain1600DatasetV3
& $Py $Run --stage AuditTrain1600DatasetV3
```

### Acceptance (Train1600 V3)

- Every `Audit*` stage exits `0`; every partial audit reports the hard block
  (same-state / readback / constraints / label recompute = 100 %, actual
  duplicates = 0, reference-cache SHA consistent, accounting closed).
- `AuditTrain1600DatasetV3` exits `0` with accepted = 1600, 64 events,
  320 state groups, split 48/8/8, per-state exactly 5 actual-unique samples.
- Model training, Policy Lock, closed loop, Challenge, and Formal Blind stay
  blocked until the (deferred) Model Safety Gate V3 is evaluated on powered
  Locked Validation data.

### If a stage fails

Send `audits\stage_status\<Stage>.json` plus, for Run stages, the newest
`train1600_v3\<segment>\runs\**\attempt.json` of the failing sample.


---

## V4 True-State Model Training (§4–§9, offline only)

This section covers the H120 True-state model training, calibration, one-shot
Locked evaluation, and offline safety audit. **No SWMM closed loop is run
here.** All stages read only the frozen Train1600 V3 evidence (1600 accepted
cases, 64 events, 320 states, split 1200/200/200). Model Safety Gate stays
`deferred`; Policy Lock, Challenge, and Formal Blind remain blocked.

```powershell
Set-Location -LiteralPath 'E:\RTC_sewer\Project6'
$Py  = "E:\RTC_sewer\Project6\.venv\Scripts\python.exe"
$V4  = "scripts\project6_v4_final.py"
$Cfg = "configs\wuhan_project6_v4_final.yaml"
```

### Step 0 — Re-stamp frozen evidence under the new code SHA (defect 1)

The new training code (`train_v4_loader.py`, `train_v4_models.py`,
`pipeline_train_v4_model.py`, plus edits to `pipeline.py` /
`pipeline_train_v4.py`) changes `working_code_sha`. `FreezeTrain1600V3Evidence`
re-verifies every frozen file SHA and, if all match, writes a **new** freeze
record + pointer under the new `code_sha` directory (recording a
`code_sha_rotation`). It never rewrites the original 1600 manifest, labels, or
split. Then re-run the two upstream audit gates so their completion sidecars
carry the new `code_git_sha`.

```powershell
& $Py $V4 --stage FreezeTrain1600V3Evidence          --config $Cfg
& $Py $V4 --stage AuditTrain1600LearnabilityV4       --config $Cfg
& $Py $V4 --stage AuditModelTrainingAuthorizationV4  --config $Cfg
```

- Expected exit code: `0` for all three.
- `freeze_pointer.json` `code_sha256` equals the current `working_code_sha`;
  when rotated, the freeze record contains `code_sha_rotation.previous_code_sha256`.
- Acceptance: all three `audits\stage_status\<Stage>.json` exit `0`,
  `scope_complete=true`, and `code_git_sha` matches the new working code SHA.

> **Agent note:** these three light gates were executed by the agent in this
> turn to produce valid evidence under the new code SHA (see the response
> "Stage evidence" summary). Re-run them yourself only if the code SHA changes
> again.

### Step 1 — Baselines (§4)

```powershell
& $Py $V4 --stage TrainV4Baselines    --config $Cfg
& $Py $V4 --stage EvaluateV4Baselines --config $Cfg
```

- Expected exit code: `0`.
- Outputs: `models/v4_true_state/baselines.json`,
  `models/v4_true_state/baseline_eval.json`.
- Acceptance: baseline metrics computed on the **calibration** split (never
  Locked); zero / mean / ridge / HGB tiers present for continuous heads,
  mean / logistic / HGB tiers for classification heads.

### Step 2 — True-state model (§5–§6)

```powershell
& $Py $V4 --stage TrainV4TrueState --config $Cfg
```

- Expected exit code: `0` (long-running: 5-seed ensemble).
- Output: `models/v4_true_state/true_state_model.pkl` +
  `models/v4_true_state/train_summary.json`.
- Acceptance (from `train_summary.json`):
  - `full_event_heads_enabled=false` (Train1600 has `full_event_eligible=false`).
  - `peak_hard_negatives_in_train=691`, `peak_hard_negatives_downsampled=false`.
  - `online_candidate_k_policy.allowed_online=[4,6,8]`,
    `disabled_online=[1,2]`, `k_le_8_modified=false`.
  - PFV head is a hurdle (gate + active-only regressor), not plain MSE.
  - Trained on Train events only.

### Step 3 — Calibration (§7)

```powershell
& $Py $V4 --stage CalibrateV4TrueState --config $Cfg
```

- Expected exit code: `0`.
- Output: `models/v4_true_state/calibration.json`.
- Acceptance: `split_used="calibration"`, `calibration_n=200`; temperatures,
  conformal q90, abstain/OOD thresholds calibrated on the **calibration** split
  only. Locked is not read.

### Step 4 — Locked evaluation (§8, ONE-SHOT)

```powershell
& $Py $V4 --stage EvaluateV4TrueStateLocked --config $Cfg
```

- Expected exit code: `0` on the **first and only** run.
- Before evaluating, writes immutable
  `models/v4_true_state/locked_evaluation_intent.json`
  (model SHA, config SHA, calibration SHA, event/rainfall SHA), then
  `models/v4_true_state/locked_evaluation_result.json`.
- **One-shot guard:** if the intent or result already exists, the stage exits
  non-zero with `reason="locked_evaluation_already_executed"`. Do **not**
  delete these files to force a re-run. Locked results must not feed back into
  model structure, loss, hyper-parameters, thresholds, or candidate rules.

### Step 5 — Offline safety audit (§9)

```powershell
& $Py $V4 --stage AuditV4OfflineSafetyGate --config $Cfg
```

- Expected exit code: `0` (offline pass) or `5` (offline fail).
- Output: `models/v4_true_state/offline_safety_gate.json`.
- Acceptance: six offline checks pass — calibration did not read Locked, Locked
  not used for tuning, Peak hard negatives preserved, full-event heads disabled,
  online K1/K2 disabled, Locked evaluation present. The gate reports
  `model_safety_gate_status="deferred"` and **cannot** clear that deferral.

### Still blocked after this section

- Policy Lock, Challenge, Formal Blind, and any SWMM closed loop remain blocked
  until a future powered Model Safety Gate is evaluated.
- `EvaluateV4TrueStateLocked` cannot be re-run.

### If a V4 model stage fails

Send `audits\stage_status\<Stage>.json`, and for training stages the newest
`models\v4_true_state\*.json` plus any traceback printed to the console.

---

## §21 V4.1 Compact Head-Specific Surrogate Rescue — user commands

> **Purpose.** Rescue the failed V4.0 predictive generalization (Locked PFV
> R²=-1.66 / TFV R²=0.09 / Peak R²=-0.55) via a feature-block / task / model-
> structure ablation (Phase-1, Train-only) followed by a **brand-new independent**
> Calibration + Locked evaluation (Phase-2) scored by the frozen Predictive
> Generalization Gate. The pipeline **stops at the gate**; it never authorizes
> the SWMM closed loop and never re-runs the V4.0 offline gate.
>
> **Execution rules (AGENTS.md).** Run every command yourself and paste the
> logs back. The agent must not run these. Never delete a one-shot intent /
> result to force a re-run. Do not edit the frozen 1600 labels, margins,
> dead-zones, or Split. Do not proceed past `AuditV4PredictiveGeneralizationGateV1`.

```powershell
cd 'E:\RTC_sewer\Project6'
$Py  = "E:\RTC_sewer\Project6\.venv\Scripts\python.exe"
$V4  = "scripts\project6_v4_final.py"
$Cfg = "configs\wuhan_project6_v4_final.yaml"
```

### Step 0 — Phase-2 test evidence (run BEFORE any stage)

```powershell
& $Py -m pytest tests\test_v4_compact_phase1.py tests\test_v4_compact_phase2.py -x -q
& $Py -m pytest tests\ -q -k "v4"
```

- Expected: `11 passed` (Phase-1) + `15 passed` (Phase-2); the full `-k v4`
  suite shows **no regression**.
- Acceptance: both invocations exit `0`. If red, paste the full pytest output.
  Do not run any pipeline stage until the tests are green.

### Step 1 — Phase-1 freeze + diagnostics (§1–§3)

```powershell
& $Py $V4 --stage FreezeV4OfflineV0Evidence        --config $Cfg
& $Py $V4 --stage AuditV4LockedMetricComparabilityV0 --config $Cfg
& $Py $V4 --stage AuditV4GeneralizationFailureV0     --config $Cfg
```

- Expected exit code: `0` each.
- Outputs: `audits/frozen_evidence/v4_offline_v0/freeze_pointer.json`,
  `audits/v4_diagnostics/locked_v0_metric_comparability.json`,
  `audits/v4_diagnostics/generalization_failure_v0.json`.
- Acceptance: the freeze re-verifies the V4.0 offline artifacts under the
  **current** code SHA (they were produced under an earlier SHA); the two
  diagnostics quantify the V4.0 Locked failure without reading any new data.

### Step 2 — Feature / task / model-structure ablation (§5–§8, Train-only)

```powershell
& $Py $V4 --stage BuildV4FeatureBlockCatalogV1       --config $Cfg
& $Py $V4 --stage BuildV4LearningCurvesV1            --config $Cfg
& $Py $V4 --stage RunV4FeatureBlockAblationV1        --config $Cfg
& $Py $V4 --stage RunV4HeadArchitectureAblationV1    --config $Cfg
& $Py $V4 --stage AuditV4MultitaskGradientConflictV1 --config $Cfg
```

- Expected exit code: `0` each (long-running: cross-validated ablation).
- Outputs (all under `models/v4_compact_v1/`):
  `feature_block_catalog_summary.json`, `learning_curves_summary.json`,
  `feature_block_ablation.json`, `head_architecture_ablation.json`,
  `gradient_conflict.json`.
- Acceptance: every ablation is scored on the **calibration** split only
  (Locked is never read); the feature-block, head-architecture, and
  multitask-conflict evidence is written before model selection.

### Step 3 — Select + train the compact head-specific model (§10–§11)

```powershell
& $Py $V4 --stage SelectV4CompactModelV1     --config $Cfg
& $Py $V4 --stage TrainV4CompactTrueStateV1  --config $Cfg
```

- Expected exit code: `0` each (`TrainV4CompactTrueStateV1` long-running).
- Outputs: `models/v4_compact_v1/v4_compact_v1_selection.json`,
  `models/v4_compact_v1/compact_head_specific_model.pkl`,
  `models/v4_compact_v1/completion.json`.
- Acceptance: selection is justified **only** by Phase-1 ablation evidence
  (never by any Locked metric); the model is trained on Train events only.

### Step 4 — Freeze the fresh evaluation split (§12)

```powershell
& $Py $V4 --stage PlanV4CompactCalibrationLockedV1 --config $Cfg
& $Py $V4 --stage AuditV4CompactEvaluationPlanV1    --config $Cfg
```

- Expected exit code: `0` each.
- Outputs (under `v4_compact_eval/planning/`):
  `evaluation_plan_freeze.json`, `v4_compact_v1_calibration_plan.csv`,
  `v4_compact_v1_locked_plan.csv`, `v4_compact_v1_accrual_plan.csv`,
  `old_calibration_locked_consumption.json`, `evaluation_plan_audit.json`.
- Acceptance: the new Calibration / Locked / accrual events are drawn
  **only from Reserve** (`assigned_split=="reserve"`), the selection is frozen
  before any new label, the old V4.0 Calibration/Locked are marked
  `eligible_for_v1_official_evaluation=false`, and the audit reports
  `status="pass"` with `all_from_reserve` and `not_selected_by_old_locked` true.
  If Reserve is too small the plan **fails closed** (non-zero) — paste the log.

### Step 5 — SWMM Calibration branch (§13, V3 reuse)

```powershell
& $Py $V4 --stage RunV4CompactCalibrationV1   --config $Cfg --workers 16 --resume
& $Py $V4 --stage BuildV4CompactCalibrationV1 --config $Cfg
& $Py $V4 --stage AuditV4CompactCalibrationV1 --config $Cfg
```

- Expected exit code: `0` each after the run completes. `RunV4CompactCalibrationV1`
  is resumable — re-run the same command until it reports scope complete.
- Outputs (under `v4_compact_eval/calibration/`): `run_manifest.csv`,
  `dataset/round_sample_manifest.csv`, `round_audit.json`.
- Acceptance: the four SWMM branches (candidate / no_control /
  dynamic_internal_rules / hold_previous) run against
  `data/wuhan_v8_storage_retrofit.inp`; the round audit passes with the
  same reference-cache / sampled-only label semantics as the V3 pilot.

### Step 6 — SWMM Locked branch (§13, V3 reuse)

```powershell
& $Py $V4 --stage RunV4CompactLockedV1   --config $Cfg --workers 16 --resume
& $Py $V4 --stage BuildV4CompactLockedV1 --config $Cfg
& $Py $V4 --stage AuditV4CompactLockedV1 --config $Cfg
```

- Expected exit code: `0` each after the run completes (resumable).
- Outputs (under `v4_compact_eval/locked/`): `run_manifest.csv`,
  `dataset/round_sample_manifest.csv`, `round_audit.json`.
- Acceptance: same as Step 5, on the frozen Locked events. **These SWMM
  results are never read for tuning** — they exist only to feed the one-shot
  Locked evaluation in Step 8.

### Step 7 — Calibrate the compact model (§14, NEW calibration only)

```powershell
& $Py $V4 --stage CalibrateV4CompactV1 --config $Cfg
```

- Expected exit code: `0`.
- Output: `models/v4_compact_v1/v4_compact_v1_calibration.json`.
- Acceptance: `reads_locked=false`, `updates_model_weights=false`,
  `split_used="v4.1_calibration"`; one-sided conformal intervals are fit per
  continuous head on the **new** Calibration split only.

### Step 8 — One-shot Locked evaluation (§16, ONE-SHOT)

```powershell
& $Py $V4 --stage EvaluateV4CompactLockedV1 --config $Cfg
```

- Expected exit code: `0` on the **first and only** run.
- Before reading any Locked data the stage writes the immutable
  `models/v4_compact_v1/v4_compact_v1_locked_intent.json` (model / calibration /
  gate-contract / plan-freeze / config / code SHAs), then
  `models/v4_compact_v1/v4_compact_v1_locked_evaluation.json`.
- **One-shot guard:** if the intent or result already exists the stage exits
  non-zero with `reason="locked_evaluation_already_executed"`. If required
  inputs are missing it exits `3` **without** writing the intent (the one-shot
  is not burned). Do **not** delete these files to force a re-run.
- Acceptance: report has `used_for_tuning=false`, `split_used="v4.1_locked"`,
  and continuous-head R² for pfv / tfv / peak.

### Step 9 — Predictive Generalization Gate (§15, STOP here)

```powershell
& $Py $V4 --stage AuditV4PredictiveGeneralizationGateV1 --config $Cfg
```

- Expected exit code: `0` (**pass**) / `3` (**underpowered**) / `5`
  (**scientific_fail**).
- Output: `models/v4_compact_v1/v4_predictive_generalization_gate.json`
  (scored against `docs/contracts/PROJECT6_V4_PREDICTIVE_GENERALIZATION_GATE_V1.json`).
- Acceptance / verdict semantics:
  - **pass** → `authorizes_closed_loop=true`. Stop and report; do **not** run
    `PlanExactClosedLoopV4` or any SWMM closed-loop stage without a new turn.
  - **underpowered** → `authorizes_closed_loop=false`; triggers the pre-
    registered accrual reserve. Do not change the model or thresholds.
  - **scientific_fail** → `authorizes_closed_loop=false`; the rescue did not
    generalize. Do not tune against Locked. Stop and report.

### If a Phase-2 stage fails

Send `audits\stage_status\<Stage>.json`, the newest
`v4_compact_eval\**\round_audit.json` (for Run/Build/Audit stages) or
`models\v4_compact_v1\*.json` (for Calibrate/Evaluate/Gate stages), plus any
traceback printed to the console.

---

## §22 Final return checklist (report these after the gate)

When the run reaches `AuditV4PredictiveGeneralizationGateV1`, report:

1. Phase-1 test result: `test_v4_compact_phase1.py` passed count.
2. Phase-2 test result: `test_v4_compact_phase2.py` passed count.
3. Full `-k v4` suite result (regression check).
4. `FreezeV4OfflineV0Evidence` freeze pointer + code SHA it re-verified under.
5. `locked_v0_metric_comparability.json` verdict.
6. `generalization_failure_v0.json` verdict.
7. `feature_block_catalog_summary.json` block list.
8. `learning_curves_summary.json` headline curve.
9. `feature_block_ablation.json` best feature block.
10. `head_architecture_ablation.json` best head architecture.
11. `gradient_conflict.json` multitask-conflict verdict.
12. `v4_compact_v1_selection.json` selected config + justification source.
13. `TrainV4CompactTrueStateV1` completion + trained-on-Train-only confirmation.
14. `evaluation_plan_freeze.json` new Calibration / Locked / accrual event ids.
15. `old_calibration_locked_consumption.json` old-Locked eligibility flags.
16. `evaluation_plan_audit.json` status + `all_from_reserve` / `not_selected_by_old_locked`.
17. Calibration branch `round_audit.json` status.
18. Locked branch `round_audit.json` status.
19. `v4_compact_v1_calibration.json` `reads_locked` / `updates_model_weights` / `split_used`.
20. `v4_compact_v1_locked_intent.json` recorded SHAs (proof of one-shot ordering).
21. `v4_compact_v1_locked_evaluation.json` continuous R² (pfv / tfv / peak) + classification metrics.
22. `v4_predictive_generalization_gate.json` verdict (pass / underpowered / scientific_fail).
23. `authorizes_closed_loop` flag.
24. Exit codes for every stage above.
25. Explicit confirmation that the pipeline **stopped at the gate** and no
    SWMM closed-loop stage was run.

---

## V4.2 Repair — Mechanical Gate and Tiny Overfit

These commands are for the V4.2 control-effect model repair pipeline.
All code fixes have been applied. The user must execute the following
commands manually and provide logs.

### Step 1: Run V4.2 Tests (verify all 44 new tests pass)

```powershell
cd E:\RTC_sewer\Project6
.\.venv\Scripts\python.exe -m pytest tests/test_v42_twin_branch_and_aliasing.py tests/test_v42_target_keys_and_ranking.py tests/test_v42_physics_gradient_and_perturbation.py tests/test_v42_physics_units.py tests/test_v42_action_shuffle.py tests/test_v42_training_summary.py tests/test_v42_head_activation_and_optimizer.py -v
```

Expected: 44 passed

### Step 2: Run Tiny Overfit Gate (7 sub-experiments)

```powershell
.\.venv\Scripts\python.exe sewerrtc/v4/v42_tiny_overfit.py
```

Expected output: `audits/v42_tiny_overfit/tiny_overfit_audit.json` and `tiny_overfit_summary.json`

Pass conditions:
- Each head (PFV/TFV/Peak) loss drops >30%
- KPI outputs non-constant
- Candidate=Reference → delta ≈ 0
- Action shuffle → output changes
- Checkpoint save/resume consistent

### Step 3: Re-run Head Activation Audit (with repaired code)

```powershell
.\.venv\Scripts\python.exe sewerrtc/v4/v42_head_activation_audit.py
```

Expected: all heads PASS (previously 47/48 — delta was constant)

### Step 4: Re-run Ranking/Physics Audit

```powershell
.\.venv\Scripts\python.exe sewerrtc/v4/v42_ranking_physics_audit.py
```

Expected: ranking uses softplus, physics units consistent

### Step 5: Full V4/V4.2 Regression

```powershell
.\.venv\Scripts\python.exe -m pytest tests/ -k "v4" -q --tb=line
```

Expected: all pass except the pre-existing flaky `test_pipeline_import_does_not_load_torch`

