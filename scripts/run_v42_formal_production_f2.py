"""Production entrypoint for the V4.2 Formal paper campaign.

The orchestration implementation is kept separate from the low-level safety
wrapper so tests can prove that no development/qualification controller is
promoted.  This entrypoint injects the rule-free-plant/native-Internal-shadow
runtime and expands Policy-Lock hashing to cover every executable Formal module.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.run_v42_formal_paper_f2 as orchestrator
from sewerrtc.v4 import v42_formal_runtime as base_runtime
from sewerrtc.v4.v42_formal_runtime_safe import (
    run_baseline_event,
    run_proposed_event,
)


def _production_policy_sha(project_root: Path) -> str:
    files = (
        project_root / "sewerrtc/control/pfvfirst_mpc_v42.py",
        project_root / "sewerrtc/v4/v42_formal_runtime.py",
        project_root / "sewerrtc/v4/v42_formal_runtime_safe.py",
        project_root / "scripts/run_v42_formal_paper_f2.py",
        project_root / "scripts/run_v42_formal_production_f2.py",
        project_root / "configs/v42_formal_fallback_contract.json",
        project_root / "docs/contracts/PROJECT6_V42_PAPER_WORKFLOW_CONTRACT.json",
    )
    return hashlib.sha256(
        "\n".join(base_runtime.sha256_file(path) for path in files).encode("utf-8")
    ).hexdigest()


def _production_policy_lock_payload(project_root: str | Path):
    payload = base_runtime.policy_lock_payload(project_root)
    payload["policy_sha256"] = _production_policy_sha(Path(project_root))
    payload["production_runtime"] = "scripts/run_v42_formal_production_f2.py"
    payload["rule_free_proposed_plant"] = True
    payload["native_internal_causal_shadow"] = True
    return payload


# Replace only low-level execution and policy-hash functions. All split,
# one-shot, evidence and stage-order logic remains in the Formal orchestrator.
orchestrator.run_baseline_event = run_baseline_event
orchestrator.run_proposed_event = run_proposed_event
orchestrator._policy_sha = _production_policy_sha
orchestrator.policy_lock_payload = _production_policy_lock_payload


if __name__ == "__main__":
    raise SystemExit(orchestrator.main())
