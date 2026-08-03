# V4.2 Formal F2 precompute readiness audit

## Objective

Audit existing Formal F2 artifacts without training or SWMM execution, and
fail closed on identity, causal-history, target-coverage, evaluation, and R0
lineage problems.

## Scope

1. Reuse the existing Formal F2 manifests, event ledger, evaluation plan,
   R0 adapter audit, GAT history window extraction, and hydraulic target
   contract. Do not scan unrelated historical outputs.
2. Add one bounded-memory audit script that writes a single readiness JSON and
   supporting CSV/JSON diagnostics under `formal_f2/precompute_readiness`.
3. Add one visible VS Code process task for running the full audit.
4. Add focused regression coverage for identity conflicts, group-weighted
   effective sample size, candidate action deduplication, and fail-closed
   target readiness.
5. Add missing R0 lineage hashes without changing any scientific threshold.

## Verification

- `py_compile` for the new audit and touched R0 adapter.
- focused Formal F2 tests plus new readiness tests.
- full readiness audit only from the visible VS Code Integrated Terminal.
- inspect the generated readiness JSON; do not claim Step1 readiness from an
  exit code alone.

## Stop rules

- Never start Step1/Step2 training or SWMM generation.
- Keep `READY_FOR_STEP2` false unless all required hydraulic target families
  are present, finite, and trainer supervision is enabled.
- Preserve existing artifacts and use bounded, per-file reads for detail CSVs.
