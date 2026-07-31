# Project6 Migration Boundary

Project6 is the standalone remediation branch for the no-control
PFV-preserving system-risk repair study. It intentionally does not read
Project2, Project3, Project4, or Project5 outputs during its formal workflow.

## Copied as reusable inputs

- `data/wuhan_with_controls.inp` and Project5 priority-node definitions;
- public benchmark inputs under `data/open_benchmarks`;
- Project5 source code, configurations, tests, and documentation, renamed and
  corrected in Project6 as the remediation proceeds.

## Explicitly invalidated

- Project5 horizon datasets and chunk files;
- Project5 horizon surrogates, uncertainty heads, tensor cache, and formal
  closed-loop results;
- Project5 actuator reliability summaries and any gate verdict derived from
  them.

## Reusable but not yet imported

Project5's raw SWMM trajectory bank is potentially useful for GAT pretraining
and for audited trajectory reuse. It is intentionally not copied here yet:
the bank is about 25 GB and must pass Project6 source-fingerprint, network,
control-step, actuator-schema, and rainfall-manifest checks before import.
It must never be treated as a Project6 formal horizon dataset without rebuild.

## Runtime boundary

Use `configs/wuhan_project6.yaml` and `Project6/.venv`. The Project6 pipeline
is `scripts/project6_runs/RUN_PROJECT6_NO_CONTROL_REPAIR_PIPELINE.ps1`.
