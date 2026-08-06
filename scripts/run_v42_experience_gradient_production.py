"""Production-compatible entrypoint for V4.2 experience-guided gradient MPC.

This wrapper reuses the existing Formal orchestrator and plant runtime but
replaces the candidate-search function with the experience/gradient selector.
The online PFV gate, Engineering36 projection, write/readback verification and
rolling execution remain owned by the existing production path.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import scripts.run_v42_formal_production_f2 as production
from sewerrtc.v4 import v42_formal_runtime as base_runtime
from sewerrtc.v4 import v42_formal_runtime_safe as safe_runtime
from sewerrtc.v4 import v42_formal_surrogate_closed_loop as surrogate_runtime
from sewerrtc.v4.v42_experience_gradient_runtime import (
    DEFAULT_BANK_RELATIVE_PATH,
    predict_and_decide as experience_gradient_predict_and_decide,
)


ENTRYPOINT_CONTRACT = "PROJECT6_V42_EXPERIENCE_GRADIENT_PRODUCTION_ENTRYPOINT_V1"


def _patch_runtime_selector() -> None:
    base_runtime.predict_and_decide = experience_gradient_predict_and_decide
    safe_runtime.predict_and_decide = experience_gradient_predict_and_decide
    surrogate_runtime.predict_and_decide = experience_gradient_predict_and_decide


def _extended_policy_sha(project_root: Path) -> str:
    base_sha = production._production_policy_sha(project_root)
    files = [
        project_root / "sewerrtc/control/experience_bank_v42.py",
        project_root / "sewerrtc/control/differentiable_hybrid_search_v42.py",
        project_root / "sewerrtc/control/pfv_tfv_tradeoff_v42.py",
        project_root / "sewerrtc/v4/v42_experience_gradient_runtime.py",
        project_root / "scripts/run_v42_experience_gradient_production.py",
        project_root / "docs/contracts/PROJECT6_V42_EXPERIENCE_GRADIENT_MPC_V1.json",
    ]
    bank = project_root / DEFAULT_BANK_RELATIVE_PATH
    if bank.exists():
        files.append(bank)
    digest = hashlib.sha256()
    digest.update(base_sha.encode("utf-8"))
    for path in files:
        digest.update(b"\n")
        digest.update(base_runtime.sha256_file(path).encode("utf-8"))
    return digest.hexdigest()


def _experience_policy_lock_payload(project_root: str | Path):
    root = Path(project_root)
    payload = production._production_policy_lock_payload(root)
    payload["policy_sha256"] = _extended_policy_sha(root)
    payload["production_runtime"] = "scripts/run_v42_experience_gradient_production.py"
    payload["candidate_search_scope"] = "global_coverage_plus_authoritative_experience_plus_gradient_refinement"
    payload["experience_gradient_contract"] = ENTRYPOINT_CONTRACT
    payload["experience_bank_relative_path"] = str(DEFAULT_BANK_RELATIVE_PATH)
    payload["experience_bank_required_for_formal"] = True
    payload["gradient_search_is_safety_authority"] = False
    payload["pfv_admission_authority"] = "rolling_calibrated_PFV_UCB"
    return payload


def install_experience_gradient_production_contract(project_root: str | Path = PROJECT_ROOT) -> None:
    root = Path(project_root)
    bank = root / DEFAULT_BANK_RELATIVE_PATH
    if not bank.exists():
        raise FileNotFoundError(
            f"authoritative experience bank missing: {bank}; build and freeze it before this production entrypoint"
        )
    _patch_runtime_selector()
    production._production_policy_sha = _extended_policy_sha
    production._production_policy_lock_payload = _experience_policy_lock_payload
    production.orchestrator.policy_lock_payload = _experience_policy_lock_payload


def main() -> int:
    install_experience_gradient_production_contract(PROJECT_ROOT)
    return int(production.main())


if __name__ == "__main__":
    raise SystemExit(main())
