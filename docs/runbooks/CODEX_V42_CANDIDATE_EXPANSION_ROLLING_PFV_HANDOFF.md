# Project6 V4.2 — targeted candidate expansion and rolling PFV handoff

## Frozen evidence

The authoritative PFV–TFV Pareto audit used 81 same-state rainfall groups and
518 recorded candidate trajectories.  Under the current 5% + 100 m3 contract,
only 5/81 states had an oracle TFV reduction of at least 20%.  Across the full
PFV margin grid, the best count was 11/81.  The current candidate population is
therefore insufficient to establish a 20% control ceiling.

The online selector also compares only candidate/No-control H120 predictions.
It does not include the realised prefix and reuses the full allowance at every
10-minute replan.  This is not equivalent to an event-level PFV
non-inferiority contract.

## New development components

* `sewerrtc/control/targeted_candidate_expansion_v42.py`
  * H3-only global singles, binary toggles, pair/quad coordination and
    positive-control neighbourhoods.
  * deterministic deduplication and family-balanced cap.
* `sewerrtc/control/rolling_pfv_budget_v42.py`
  * causal event-level prefix accounting.
* `sewerrtc/v4/v42_candidate_expansion_runtime_patch.py`
  * opt-in development adapter; never authorises Formal execution.
* `scripts/plan_v42_targeted_candidate_expansion.py`
  * read-only selection of high-information states.

## Mandatory local execution order

1. Synchronise this branch and preserve unrelated local changes.
2. Run focused tests and compile checks.
3. Generate the targeted state plan from the existing Pareto CSV files.
4. Freeze the selected state list and candidate recipe.
5. Reuse the exact rainfall/checkpoint/prefix and existing
   No-control/Internal/Hold references.
6. Run only new candidate branches in two bounded rounds.
7. Rebuild the same-state authoritative manifest and rerun the Pareto audit.
8. Only if the expanded true-state oracle supports the target, redesign Step2
   or run a closed-loop micro test.
9. Wire realised-prefix rolling PFV accounting before any Formal closed loop.

## Candidate expansion rounds

### Round 1 — coverage

For every selected state, generate the deterministic population and choose a
family-balanced subset of at most 128 candidates.  Preserve:

* Hold;
* both binary toggles when executable;
* global continuous singles at 0.05/0.10/0.20;
* H3 constant, pulse, ramp and release profiles;
* coordinated pairs;
* a small number of coordinated quads;
* neighbours of authoritative positive-control actions.

Run only candidate branches.  References are shared and must not be rerun.

### Round 2 — local refinement

Only for LOW/MODERATE states that still lack a 20% safe oracle opportunity,
expand to at most 384 candidates around the best Round-1 actions.  Do not expand
NEAR/SEVERE states merely to force the paper target if their oracle opportunity
is already physically small.

## Required identity and validity fields

Every branch row must include:

* rainfall fingerprint;
* event ID and checkpoint minute;
* network/INP hash;
* prefix-state hash;
* candidate action signature in Engineering36 order;
* H3 and H12 action arrays;
* changed-facility count;
* binary validity;
* bounds validity;
* target-setting write/readback evidence;
* authoritative PFV, TFV and global peak;
* shared reference trajectory identities.

A candidate ID alone never proves a distinct action.

## Rolling PFV event contract

For relative margin `delta` and absolute margin `B`, admission must use

```
realised_candidate_prefix
- (1 + delta) * realised_no_control_prefix
+ UCB(candidate_future - (1 + delta) * no_control_future)
<= B
```

The plant runtime must update the prefix state after every executed 10-minute
interval.  The No-control prefix must come from a causal parallel shadow using
only rainfall observed up to the current time.  The full allowance must not be
reinitialised at each decision.

## Go/no-go after the expanded oracle

* **PATH A:** LOW/MODERATE event-balanced oracle reduction is close to or above
  20% under an engineering-acceptable PFV contract.  Proceed to a direct PFV
  budget model plus within-state TFV ranker.
* **PATH B:** improvement exists but remains well below 20%.  Inspect actuator
  controllability and candidate structure once more; do not retrain Step2 yet.
* **PATH C:** expanded true-state oracle remains below the target.  Freeze and
  report the physically supported ceiling under Engineering36 and the Internal
  baseline; do not manufacture 20% by relaxing PFV excessively.

## Prohibited actions

* no R0 restart;
* no Step1 retraining;
* no new full Calibration campaign;
* no full 12-event closed loop before the expanded oracle passes;
* no silent use of the old 72-entry action map;
* no old-model/new-calibration mixing;
* no allowance reset at every MPC step;
* no merge to `main` while the development objective is unverified.
