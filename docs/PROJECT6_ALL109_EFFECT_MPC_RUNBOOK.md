# Project6 109-facility No-control Effect MPC

## Corrected contract

The controller executes absolute 109-dimensional settings. Every candidate also carries an explicit residual sequence relative to the same-time No-control action sequence. The effect head predicts only candidate minus No-control horizon effects; the No-control digital-twin horizon supplies the absolute PFV, TFV, and peak reference.

The formal pipeline uses:

1. All 8,100 compatible Project5 trajectories (270 events x 30 policies) for the GAT transition cache.
2. The 4,500 T20-T100 trajectories (150 events x 30 policies) for formal horizon-state and rainfall coverage.
3. Exact single-actuator No-control replay counterfactuals for action-effect learning, direction reliability, and uncertainty calibration.
4. A multiscale response surface using residual amplitudes and absolute targets in `[0, 1]`.
5. Direction-safe joint action candidates for simultaneous multi-facility rolling-horizon control.

## Smoke verification

```powershell
cd E:\RTC_sewer\Project6

.\scripts\project6_runs\RUN_PROJECT6_ALL109_EFFECT_MPC_PIPELINE.ps1 `
  -RunLevel smoke `
  -Device cuda `
  -Workers 4 `
  -ProposedWorkers 2 `
  -GatEpochs 5 `
  -SurrogateEpochs 5 `
  -AblationEvents 1 `
  -AblationMaxActuators 8 `
  -RepresentativeEvents 2 `
  -RunTag project6_all109_effect_mpc_smoke_v2 `
  -AllowGateFail
```

## Formal run

```powershell
cd E:\RTC_sewer\Project6

.\scripts\project6_runs\RUN_PROJECT6_ALL109_EFFECT_MPC_PIPELINE.ps1 `
  -RunLevel formal `
  -Device cuda `
  -Workers 16 `
  -ProposedWorkers 4 `
  -GatEpochs 100 `
  -SurrogateEpochs 180 `
  -AblationEvents 10 `
  -AblationSamplesPerPhase 1 `
  -AblationMaxActuators 0 `
  -RepresentativeEvents 30 `
  -RunTag project6_all109_effect_mpc_formal_v2
```

`AblationMaxActuators 0` means all 109 facilities. The ablation stage is resumable by case id. Generic trajectory import, GAT feature generation, horizon chunks, and closed-loop event details are also resumable.

The formal response scan first performs the `+/-0.05` local screen, then adds relative amplitudes `0.05, 0.10, 0.20, 0.40` and absolute targets `0, 0.25, 0.50, 0.75, 1.00`. The online controller still obeys its per-step ramp limit; the larger scans identify nonlinear response rather than authorizing an instantaneous 0-to-1 command.

To resume after an interruption, rerun the same command. To retain an already trained GAT, add `-SkipGatTraining`. To retain a completed ablation bank, add `-SkipAblation`.

## Evidence gates

The formal pipeline stops before closed-loop publication claims unless held-out exact counterfactual validation passes PFV/TFV/peak direction and joint-safe precision thresholds. The final claim is based only on the 30-event paired closed-loop gate, not on offline R2.
