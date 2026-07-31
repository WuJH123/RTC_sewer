[CmdletBinding()]
param(
  [switch]$PlanReuse,
  [switch]$BuildGatCache,
  [switch]$WriteSensorConfigs,
  [switch]$TrainGatSensorSweep,
  [switch]$BuildActionPretrain,
  [switch]$TrainActionPretrain,
  [switch]$MpcGate,
  [switch]$AuditGateHistory,
  [switch]$Review,
  [switch]$Resume,
  [string]$Python = "E:\RTC_sewer\Project6\.venv\Scripts\python.exe",
  [string]$Config = "configs\wuhan_project6_36_temporal_joint.yaml",
  [ValidateSet("cpu", "cuda")][string]$Device = "cuda",
  [string]$SensorRatios = "0.05,0.10,0.15,0.20,0.30",
  [int]$GatEpochs = 120,
  [int]$GatEvalEvery = 5,
  [int]$GatPatience = 20,
  [ValidateSet("risk_local", "full_state")][string]$ActionTargetMode = "risk_local",
  [int]$ActionChunkSize = 10000,
  [int]$ActionPretrainEpochs = 20,
  [int]$ActionPretrainBatchSize = 8,
  [int]$ActionPretrainSamplesPerEpoch = 5000,
  [int]$ActionPretrainValidationSamples = 4096,
  [string]$ActionPretrainOutDir = "outputs\models_temporal_action_pretrain_36_actionaware_v2",
  [double]$ActionRichWeight = 2.0,
  [double]$MinimumActionExcitation = 0.02,
  [double]$ActuatorNeighbourLossWeight = 1.0,
  [double]$RiskChangeLossWeight = 0.5,
  [int]$ActionPrefetchDepth = 4,
  [int]$CpuThreads = 12,
  [bool]$ActionAmp = $true,
  [bool]$ActionTf32 = $true,
  [int]$TimeStride = 1,
  [int]$MaxFiles = 0,
  [int]$MaxSamples = 0
)

$ErrorActionPreference = "Stop"
$Root = "E:\RTC_sewer\Project6"
Set-Location $Root

$env:OMP_NUM_THREADS = "$CpuThreads"
$env:MKL_NUM_THREADS = "$CpuThreads"
if ($Device -eq "cuda") {
  $env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
}

if (-not (Test-Path $Python)) {
  throw "Python executable not found: $Python"
}

function Invoke-Python([string]$Label, [string[]]$Arguments) {
  Write-Host "[Project6 three-step research] step=$Label"
  & $Python @Arguments
  if ($LASTEXITCODE -ne 0) {
    throw "Python step failed [$Label] with exit code $LASTEXITCODE"
  }
}

$LimitedRun = ($MaxFiles -gt 0 -or $MaxSamples -gt 0)
$GatCacheDir = if ($LimitedRun) { "outputs\cache_research_mixed_gat_smoke" } else { "outputs\cache_research_mixed_gat" }
$SensorOutRoot = if ($LimitedRun) { "outputs\research_sensor_sweep_smoke" } else { "outputs\research_sensor_sweep" }
$SensorConfigDir = if ($LimitedRun) { "configs\research_sensor_sweep_smoke" } else { "configs\research_sensor_sweep" }
$ActionPretrainPath = if ($LimitedRun) { "outputs\cache_temporal_action_pretrain_36_smoke\temporal_action_pretrain_36.npz" } else { "outputs\cache_temporal_action_pretrain_36\temporal_action_pretrain_36.npz" }

Invoke-Python "environment_preflight" @(
  "-c",
  "import sys,torch,numpy,pandas,yaml; print(sys.executable); print('torch',torch.__version__,'cuda',torch.cuda.is_available()); assert '$Device' != 'cuda' or torch.cuda.is_available()"
)

if ($PlanReuse) {
  Invoke-Python "plan_reuse" @(
    "scripts\96_plan_research_data_reuse.py",
    "--config", $Config,
    "--out-dir", "outputs\research_reuse_plan",
    "--sensor-ratios", $SensorRatios
  )
}

if ($BuildGatCache) {
  $args = @(
    "scripts\97_build_mixed_gat_cache.py",
    "--config", $Config,
    "--manifest", "outputs\research_reuse_plan\gat_mixed_trajectory_manifest.csv",
    "--base-cache", "outputs\cache_v8_storage_variablepump\transition_cache.npz",
    "--out-npz", "$GatCacheDir\transition_cache.npz",
    "--time-stride", "$TimeStride"
  )
  if ($MaxFiles -gt 0) { $args += @("--max-files", "$MaxFiles") }
  if ($MaxSamples -gt 0) { $args += @("--max-samples", "$MaxSamples") }
  Invoke-Python "build_mixed_gat_cache" $args
}

if ($WriteSensorConfigs) {
  Invoke-Python "write_sensor_sweep_configs" @(
    "scripts\100_write_sensor_sweep_configs.py",
    "--base-config", $Config,
    "--ratios", $SensorRatios,
    "--cache-dir", $GatCacheDir,
    "--out-config-dir", $SensorConfigDir,
    "--out-root", $SensorOutRoot
  )
}

if ($TrainGatSensorSweep) {
  if ($LimitedRun) {
    throw "Refusing to train GAT from a limited smoke cache. Re-run without -MaxFiles/-MaxSamples for formal training."
  }
  $cacheMeta = Join-Path $Root "$GatCacheDir\transition_cache.meta.json"
  if (-not (Test-Path $cacheMeta)) {
    throw "Missing mixed GAT cache metadata. Run -BuildGatCache first."
  }
  $cacheReport = Get-Content $cacheMeta | ConvertFrom-Json
  if (($cacheReport.max_files -gt 0) -or ($cacheReport.max_samples -gt 0)) {
    throw "Refusing to train GAT from a limited cache: max_files=$($cacheReport.max_files), max_samples=$($cacheReport.max_samples). Rebuild without limits first."
  }
  $manifestPath = Join-Path $Root "$SensorConfigDir\sensor_sweep_config_manifest.json"
  if (-not (Test-Path $manifestPath)) {
    throw "Missing sensor sweep config manifest. Run -WriteSensorConfigs first."
  }
  $manifest = Get-Content $manifestPath | ConvertFrom-Json
  foreach ($item in $manifest.configs) {
    $cfgPath = $item.config
    Invoke-Python "select_sensors_$($item.label)" @(
      "scripts\02_select_priority_and_sensors.py",
      "--config", $cfgPath
    )
    Invoke-Python "train_gat_$($item.label)" @(
      "scripts\05_train_gat.py",
      "--config", $cfgPath,
      "--epochs", "$GatEpochs",
      "--device", $Device,
      "--eval-every", "$GatEvalEvery",
      "--patience", "$GatPatience",
      "--score-full-weight", "0.50",
      "--score-priority-weight", "0.50"
    )
  }
}

if ($BuildActionPretrain) {
  $args = @(
    "scripts\98_build_temporal_action_pretrain_dataset.py",
    "--config", $Config,
    "--manifest", "outputs\research_reuse_plan\temporal_action_learning_manifest.csv",
    "--base-cache", "outputs\cache_v8_storage_variablepump\transition_cache.npz",
    "--canonical-action-order", "outputs\project6_36_fulltrain_v1\canonical_action_order\canonical_36_actuator_order.csv",
    "--out-npz", $ActionPretrainPath,
    "--horizon-steps", "6",
    "--target-mode", $ActionTargetMode,
    "--time-stride", "$TimeStride",
    "--chunk-size-samples", "$ActionChunkSize"
  )
  if ($MaxFiles -gt 0) { $args += @("--max-files", "$MaxFiles") }
  if ($MaxSamples -gt 0) { $args += @("--max-samples", "$MaxSamples") }
  Invoke-Python "build_temporal_action_pretrain_dataset" $args
}

if ($TrainActionPretrain) {
  $actionMeta = $ActionPretrainPath.Replace(".npz", ".meta.json")
  if (-not (Test-Path $ActionPretrainPath) -or -not (Test-Path $actionMeta)) {
    throw "Missing formal temporal action pretraining dataset. Run -BuildActionPretrain first."
  }
  $actionReport = Get-Content $actionMeta | ConvertFrom-Json
  if (($actionReport.files_requested -lt 10000) -or ($actionReport.actions -ne 36) -or ($actionReport.horizon_steps -ne 6)) {
    throw "Refusing non-formal action pretraining data: files=$($actionReport.files_requested), actions=$($actionReport.actions), horizon=$($actionReport.horizon_steps)"
  }
  $trainActionArgs = @(
    "scripts\102_train_temporal_action_dynamics_pretrain.py",
    "--config", $Config,
    "--dataset-index", $ActionPretrainPath,
    "--out-dir", $ActionPretrainOutDir,
    "--model-name", "raw_joint_36_actionaware_observational_dynamics.pt",
    "--epochs", "$ActionPretrainEpochs",
    "--batch-size", "$ActionPretrainBatchSize",
    "--max-train-samples-per-epoch", "$ActionPretrainSamplesPerEpoch",
    "--max-validation-samples", "$ActionPretrainValidationSamples",
    "--scale-samples", "5000",
    "--device", $Device,
    "--action-rich-weight", "$ActionRichWeight",
    "--minimum-action-excitation", "$MinimumActionExcitation",
    "--actuator-neighbour-loss-weight", "$ActuatorNeighbourLossWeight",
    "--risk-change-loss-weight", "$RiskChangeLossWeight",
    "--prefetch-depth", "$ActionPrefetchDepth",
    "--cpu-threads", "$CpuThreads"
  )
  if ($ActionAmp) { $trainActionArgs += "--amp" }
  if ($ActionTf32) { $trainActionArgs += "--tf32" }
  if ($Resume) { $trainActionArgs += "--resume" }
  Invoke-Python "train_temporal_action_dynamics" $trainActionArgs
}

if ($MpcGate) {
  Invoke-Python "mpc_gate_preflight" @(
    "scripts\99_mpc_gate_preflight.py",
    "--config", $Config,
    "--model-report", "outputs\models_temporal_joint_36_v3\raw_joint_36_same_state_v3_train_report.json",
    "--out-json", "outputs\research_reuse_plan\mpc_gate_preflight.json",
    "--enforce"
  )
}

if ($AuditGateHistory) {
  Invoke-Python "audit_26_vs_36_gate_history" @(
    "scripts\101_audit_26_vs_36_gate_history.py",
    "--config", $Config,
    "--gate26", "outputs\evaluation_project6_no_control_repair_formal_30_v8\no_control_repair_gate.json",
    "--gate36", "outputs\evaluation_project6_v8_storage_T5_T100_v1\no_control_repair_gate.json",
    "--out-json", "outputs\research_reuse_plan\gate_26_vs_36_audit.json"
  )
}

if ($Review) {
  $ActionPretrainReportRel = ($ActionPretrainOutDir.Replace("\", "/").TrimEnd("/")) + "/temporal_action_dynamics_pretrain_report.json"
  Invoke-Python "review_outputs" @(
    "-c",
    @"
import json, pathlib
root=pathlib.Path(r'$Root')
for rel in [
 'outputs/research_reuse_plan/research_reuse_summary.json',
 r'$($GatCacheDir.Replace("\","/"))/transition_cache.meta.json',
 r'$($ActionPretrainPath.Replace("\","/").Replace(".npz",".meta.json"))',
 'outputs/research_reuse_plan/mpc_gate_preflight.json',
 'outputs/research_reuse_plan/gate_26_vs_36_audit.json',
 r'$ActionPretrainReportRel',
]:
 p=root/rel
 print('\\n##', rel)
 if p.exists():
  data=json.loads(p.read_text(encoding='utf-8'))
  for k in ['inventory_rows','gat_manifest_rows','action_learning_rows','samples','events','policies','actions','action_tensor_shape','target_mode','local_nodes','passed','blocking_reasons','main_explanation']:
   if k in data: print(f'{k}: {data[k]}')
 else:
  print('missing')
"@
  )
}

if (-not ($PlanReuse -or $BuildGatCache -or $WriteSensorConfigs -or $TrainGatSensorSweep -or $BuildActionPretrain -or $TrainActionPretrain -or $MpcGate -or $AuditGateHistory -or $Review)) {
  Write-Host "Select stages: -PlanReuse -BuildGatCache -WriteSensorConfigs -TrainGatSensorSweep -BuildActionPretrain -TrainActionPretrain -MpcGate -AuditGateHistory -Review"
}
