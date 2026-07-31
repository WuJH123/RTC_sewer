"""Train1600 V4 model-training gate handlers (spec sections 1-3).

Wired lazily from ``pipeline.build_registry`` alongside the V3 handlers.  Three
stages only:

* ``FreezeTrain1600V3Evidence``      -- immutable archive of the accepted 1600
* ``AuditTrain1600LearnabilityV4``   -- training readiness audit
* ``AuditModelTrainingAuthorizationV4`` -- minimum training conditions

``FreezeTrain1600V3Evidence`` has no prerequisite chain: adding the V4 code
changes ``working_code_sha`` which would otherwise block any stage gated on the
V3 dataset audit (stamped under the old code sha).  The freeze instead verifies
the passing Dataset Gate audit internally and re-stamps the frozen evidence
under the current code sha, so the readiness / authorization chain gates
cleanly on the fresh freeze.

Nothing here mutates the original 1600-sample manifest, labels, splits,
margins, dead-zones or Locked data; the freeze copies small evidence files and
skips the multi-GB regenerable ``runs/`` trajectories (their SHAs already live
in the copied run/branch manifests).
"""
from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Callable

import pandas as pd

from .runtime import (
    EXIT_BLOCKED,
    EXIT_INCOMPLETE,
    EXIT_PASS,
    EXIT_SCIENTIFIC_FAIL,
    RuntimeOptions,
    StageResult,
    atomic_write_json,
    working_code_sha,
)
from .train1600_v4 import (
    ONLINE_CANDIDATE_K_VALUES,
    audit_train1600_learnability_v4,
    build_train1600_v3_freeze_payload,
    evaluate_model_training_authorization_v4,
)

T16V3_ROOT = "train1600_v3"
DATASET_MANIFEST_REL = (
    f"{T16V3_ROOT}/dataset/train1600_v3_sample_manifest.csv"
)
DATASET_AUDIT_REL = f"{T16V3_ROOT}/dataset/train1600_v3_dataset_audit.json"
FREEZE_ROOT_REL = "audits/frozen_evidence/train1600_v3"
FREEZE_POINTER_REL = f"{FREEZE_ROOT_REL}/freeze_pointer.json"
FREEZE_NAME = "train1600_v3_freeze.json"
READINESS_DIR_REL = f"{T16V3_ROOT}/training_readiness_v4"
READINESS_VERDICT_REL = (
    f"{READINESS_DIR_REL}/training_readiness_verdict.json"
)
AUTH_V4_REL = (
    f"{T16V3_ROOT}/authorization/model_training_authorization_v4.json"
)
MSG_V4_REL = (
    f"{T16V3_ROOT}/authorization/model_safety_gate_v4_status.json"
)
AUTH_V4_CONTRACT_REL = (
    "docs/contracts/PROJECT6_V4_MODEL_TRAINING_AUTHORIZATION_V4.json"
)
MSG_CONTRACT_REL = "docs/contracts/PROJECT6_V4_MODEL_SAFETY_GATE_V3.json"

# Evidence segments archived by the freeze (small text files copied; any file
# under a ``runs/`` subtree is skipped -- regenerable multi-GB SWMM output).
FREEZE_SEGMENTS = (
    "dataset",
    "planning",
    "round0",
    "round1",
    "round2",
    "calibration",
    "locked_validation",
    "authorization",
)


# ---------------------------------------------------------------------------
# Local helpers (kept independent of pipeline_train_v3 privates)
# ---------------------------------------------------------------------------

def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _aggregate_sha(manifest: dict[str, str]) -> str:
    lines = "\n".join(
        f"{name}:{sha}" for name, sha in sorted(manifest.items())
    )
    return hashlib.sha256(lines.encode("utf-8")).hexdigest()


def _missing_result(stage: str, missing: list[str]) -> StageResult:
    return StageResult(
        stage,
        "incomplete",
        EXIT_INCOMPLETE,
        remaining=1,
        evidence={"reason": "inputs_missing", "missing_inputs": missing},
    )


def _blocked(stage: str, reason: str, **extra) -> StageResult:
    return StageResult(
        stage,
        "blocked",
        EXIT_BLOCKED,
        evidence={"reason": reason, **extra},
    )


def _frozen_dir(output_root: Path) -> Path | None:
    pointer_path = output_root / FREEZE_POINTER_REL
    if not pointer_path.exists():
        return None
    pointer = _read_json(pointer_path)
    frozen = output_root / str(pointer.get("frozen_dir_rel", ""))
    return frozen if frozen.exists() else None


def _thresholds(config: dict) -> tuple[dict[str, float], dict[str, float]]:
    thresholds = config.get("thresholds", {})
    margin = thresholds.get("scientific_margin", {})
    dead = thresholds.get("dead_zone", {})
    margins = {
        "pfv": float(margin.get("pfv_m3", 0.0)),
        "tfv": float(margin.get("tfv_m3", 0.0)),
        "peak": float(margin.get("peak_m3s", 0.0)),
    }
    dead_zones = {
        "pfv": float(dead.get("pfv_m3", 0.0)),
        "tfv": float(dead.get("tfv_m3", 0.0)),
        "peak": float(dead.get("peak_m3s", 0.0)),
    }
    return margins, dead_zones


def _state_count(df: pd.DataFrame) -> int:
    return int(
        (
            df["event_id"].astype(str)
            + "::"
            + df["checkpoint_id"].astype(str)
        ).nunique()
    )


# ---------------------------------------------------------------------------
# Section 1: FreezeTrain1600V3Evidence
# ---------------------------------------------------------------------------

def _freeze_train1600_v3_handler(
    project_root: Path, output_root: Path, config: dict
) -> Callable[[RuntimeOptions], StageResult]:
    """Immutable archive of the accepted Train1600 V3 evidence."""

    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "FreezeTrain1600V3Evidence"
        code_sha = working_code_sha(project_root)
        pointer_path = output_root / FREEZE_POINTER_REL
        # Defect 1: a freeze pointer may only be reused when it was stamped
        # under the *current* working code sha AND every previously frozen file
        # still re-hashes to its recorded SHA.  When new training code changes
        # the code sha we must re-verify the prior frozen evidence integrity
        # and create a fresh freeze record + pointer under the new code sha
        # directory.  The original 1600 manifest / labels / splits are never
        # mutated -- the source tree is only read.
        code_sha_rotation: dict | None = None
        if pointer_path.exists():
            pointer = _read_json(pointer_path)
            prev_dir_rel = str(pointer.get("frozen_dir_rel", ""))
            prev_dir = output_root / prev_dir_rel
            prev_freeze_file = prev_dir / FREEZE_NAME
            prev_code_sha = str(pointer.get("code_sha256", ""))
            if prev_freeze_file.exists():
                prev_freeze = _read_json(prev_freeze_file)
                prev_unresolved = [
                    rel
                    for rel, sha in prev_freeze.get("file_sha256", {}).items()
                    if not (prev_dir / rel).exists()
                    or _sha_file(prev_dir / rel) != sha
                ]
                if prev_code_sha == code_sha:
                    # Genuine idempotent reuse: same code sha and intact files.
                    if not prev_unresolved:
                        return StageResult(
                            stage,
                            "pass",
                            EXIT_PASS,
                            completed=1,
                            batch_complete=True,
                            scope_complete=True,
                            evidence={
                                "already_frozen": True,
                                "immutable": True,
                                "frozen_dir": str(prev_dir),
                                "code_sha256": code_sha,
                            },
                        )
                    # Same sha but tampered/incomplete: re-archive below.
                else:
                    # Code sha changed: the prior frozen evidence must still
                    # verify before we re-stamp under the new code sha.
                    if prev_unresolved:
                        return _blocked(
                            stage,
                            "prior_frozen_evidence_sha_mismatch",
                            previous_code_sha256=prev_code_sha,
                            unresolved=prev_unresolved[:20],
                        )
                    code_sha_rotation = {
                        "previous_code_sha256": prev_code_sha,
                        "previous_frozen_dir_rel": prev_dir_rel,
                        "previous_files_reverified": len(
                            prev_freeze.get("file_sha256", {})
                        ),
                        "previous_evidence_intact": True,
                    }
        audit_path = output_root / DATASET_AUDIT_REL
        manifest_path = output_root / DATASET_MANIFEST_REL
        missing = [
            str(path)
            for path in (audit_path, manifest_path)
            if not path.exists()
        ]
        if missing:
            return _missing_result(stage, missing)
        dataset_audit = _read_json(audit_path)
        if str(dataset_audit.get("status")) != "pass":
            return _blocked(
                stage,
                "dataset_gate_not_pass",
                found=str(dataset_audit.get("status")),
            )
        t16_root = output_root / T16V3_ROOT
        target = output_root / FREEZE_ROOT_REL / code_sha
        target.mkdir(parents=True, exist_ok=True)
        copied: dict[str, str] = {}
        skipped_runs: set[str] = set()
        for segment in FREEZE_SEGMENTS:
            segment_dir = t16_root / segment
            if not segment_dir.exists():
                continue
            for source in sorted(segment_dir.rglob("*")):
                if not source.is_file():
                    continue
                rel_parts = source.relative_to(t16_root).parts
                if "runs" in rel_parts:
                    skipped_runs.add("/".join(rel_parts[:2]))
                    continue
                rel = source.relative_to(t16_root).as_posix()
                destination = target / rel
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    shutil.copy2(source, destination)
                copied[rel] = _sha_file(source)
        reference_root = output_root / "pilot" / "references"
        if reference_root.exists():
            reference_manifest = {
                source.relative_to(reference_root).as_posix(): _sha_file(
                    source
                )
                for source in sorted(reference_root.rglob("*"))
                if source.is_file()
            }
            reference_cache_sha = _aggregate_sha(reference_manifest)
        else:
            reference_cache_sha = ""
        manifest = pd.read_csv(manifest_path)
        split_counts = {
            str(key): int(value)
            for key, value in manifest["split"]
            .astype(str)
            .value_counts()
            .items()
        }
        payload = build_train1600_v3_freeze_payload(
            dataset_audit=dataset_audit,
            split_counts=split_counts,
            event_count=int(manifest["event_id"].nunique()),
            state_count=_state_count(manifest),
            file_manifest=copied,
            skipped_runs_dirs=len(skipped_runs),
            reference_cache_sha256=reference_cache_sha,
            code_sha256=code_sha,
        )
        if code_sha_rotation is not None:
            payload["code_sha_rotation"] = code_sha_rotation
        atomic_write_json(target / FREEZE_NAME, payload)
        pointer_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(
            pointer_path,
            {
                "stage": stage,
                "frozen_dir_rel": f"{FREEZE_ROOT_REL}/{code_sha}",
                "code_sha256": code_sha,
                "freeze_sha256": _sha_file(target / FREEZE_NAME),
                "immutable": True,
                "code_sha_rotation": code_sha_rotation,
            },
        )
        return StageResult(
            stage,
            "pass",
            EXIT_PASS,
            completed=len(copied),
            batch_complete=True,
            scope_complete=True,
            evidence={
                "frozen_files": len(copied),
                "skipped_runs_dirs": len(skipped_runs),
                "accepted": int(payload["accepted"]),
                "events": int(payload["events"]),
                "states": int(payload["states"]),
                "immutable": True,
                "code_sha256": code_sha,
                "code_sha_rotated": code_sha_rotation is not None,
            },
        )

    return handler


# ---------------------------------------------------------------------------
# Section 2: AuditTrain1600LearnabilityV4
# ---------------------------------------------------------------------------

def _audit_learnability_v4_handler(
    project_root: Path, output_root: Path, config: dict
) -> Callable[[RuntimeOptions], StageResult]:
    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "AuditTrain1600LearnabilityV4"
        frozen = _frozen_dir(output_root)
        if frozen is None:
            return _missing_result(
                stage, [str(output_root / FREEZE_POINTER_REL)]
            )
        manifest_path = (
            frozen / "dataset" / "train1600_v3_sample_manifest.csv"
        )
        audit_path = frozen / "dataset" / "train1600_v3_dataset_audit.json"
        missing = [
            str(path)
            for path in (manifest_path, audit_path)
            if not path.exists()
        ]
        if missing:
            return _missing_result(stage, missing)
        manifest = pd.read_csv(manifest_path)
        margins, dead_zones = _thresholds(config)
        result = audit_train1600_learnability_v4(
            manifest,
            margins=margins,
            dead_zones=dead_zones,
            online_k_values=ONLINE_CANDIDATE_K_VALUES,
        )
        out_dir = output_root / READINESS_DIR_REL
        out_dir.mkdir(parents=True, exist_ok=True)
        table_files = {
            "split_label_distribution": "split_label_distribution.csv",
            "event_label_distribution": "event_label_distribution.csv",
            "state_candidate_variance": "state_candidate_variance.csv",
            "action_coverage": "action_coverage.csv",
            "k_semantics_audit": "k_semantics_audit.csv",
            "boundary_coverage": "boundary_coverage.csv",
        }
        tables = result["tables"]
        for key, filename in table_files.items():
            pd.DataFrame(tables[key]).to_csv(
                out_dir / filename, index=False
            )
        atomic_write_json(
            out_dir / "residual_schema_audit.json",
            result["residual_schema_audit"],
        )
        atomic_write_json(
            out_dir / "feature_leakage_audit.json",
            result["feature_leakage_audit"],
        )
        atomic_write_json(
            out_dir / "covariate_shift_report.json",
            result["covariate_shift_report"],
        )
        persisted = {
            "verdict": result["verdict"],
            "feature_leakage_audit": result["feature_leakage_audit"],
            "residual_schema_audit": result["residual_schema_audit"],
            "train_core_two_sided": result["train_core_two_sided"],
            "locked_power_report": result["locked_power_report"],
            "k_semantics_summary": result["k_semantics_summary"],
            "pfv_zero_inflation": result["pfv_zero_inflation"],
            "split_leakage": result["split_leakage"],
            "code_sha256": working_code_sha(project_root),
        }
        atomic_write_json(
            out_dir / "training_readiness_verdict.json", persisted
        )
        readiness = str(result["verdict"]["training_readiness"])
        permitted = readiness in {"pass", "conditional_pass"}
        return StageResult(
            stage,
            "pass" if permitted else "scientific_fail",
            EXIT_PASS if permitted else EXIT_SCIENTIFIC_FAIL,
            completed=1,
            batch_complete=True,
            scope_complete=permitted,
            evidence={
                "training_readiness": readiness,
                "hard_failures": result["verdict"]["hard_failures"],
                "conditional_reasons": result["verdict"][
                    "conditional_reasons"
                ],
                "k1_k2_supplement_required": result["verdict"][
                    "k1_k2_supplement_required"
                ],
                "calibration_risk_heads": result["verdict"][
                    "calibration_risk_heads"
                ],
                "underpowered_locked": result["verdict"][
                    "underpowered_locked"
                ],
                "feature_leakage_count": result["verdict"][
                    "feature_leakage_count"
                ],
            },
        )

    return handler


# ---------------------------------------------------------------------------
# Section 3: AuditModelTrainingAuthorizationV4
# ---------------------------------------------------------------------------

def _audit_model_training_authorization_v4_handler(
    project_root: Path, output_root: Path, config: dict
) -> Callable[[RuntimeOptions], StageResult]:
    def handler(_options: RuntimeOptions) -> StageResult:
        stage = "AuditModelTrainingAuthorizationV4"
        frozen = _frozen_dir(output_root)
        if frozen is None:
            return _missing_result(
                stage, [str(output_root / FREEZE_POINTER_REL)]
            )
        verdict_path = output_root / READINESS_VERDICT_REL
        freeze_path = frozen / FREEZE_NAME
        audit_path = frozen / "dataset" / "train1600_v3_dataset_audit.json"
        contract_path = project_root / AUTH_V4_CONTRACT_REL
        missing = [
            str(path)
            for path in (
                verdict_path,
                freeze_path,
                audit_path,
                contract_path,
            )
            if not path.exists()
        ]
        if missing:
            return _missing_result(stage, missing)
        learnability = _read_json(verdict_path)
        freeze = _read_json(freeze_path)
        dataset_audit = _read_json(audit_path)
        unresolved: list[str] = []
        for rel, sha in freeze.get("file_sha256", {}).items():
            candidate = frozen / rel
            if not candidate.exists() or _sha_file(candidate) != sha:
                unresolved.append(rel)
        result = evaluate_model_training_authorization_v4(
            dataset_audit=dataset_audit,
            learnability=learnability,
            freeze=freeze,
            unresolved_files=unresolved,
        )
        result["contract"] = AUTH_V4_CONTRACT_REL
        result["contract_sha256"] = _sha_file(contract_path)
        result["frozen_evidence_dir"] = str(frozen)
        result["code_sha256"] = working_code_sha(project_root)
        auth_path = output_root / AUTH_V4_REL
        auth_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(auth_path, result)
        atomic_write_json(
            output_root / MSG_V4_REL,
            {
                "gate": "PROJECT6_V4_MODEL_SAFETY_GATE_V3",
                "status": "deferred",
                "reason": (
                    "training_authorized_but_safety_gate_requires_locked"
                    "_evaluation"
                ),
                "controls": [
                    "policy_lock",
                    "surrogate_closed_loop",
                    "challenge",
                    "formal_blind",
                ],
                "model_training_authorized": bool(
                    result["model_training_authorized"]
                ),
                "evaluated_with_code_sha256": result["code_sha256"],
            },
        )
        passed = bool(result["model_training_authorized"])
        return StageResult(
            stage,
            "pass" if passed else "blocked",
            EXIT_PASS if passed else EXIT_BLOCKED,
            completed=1,
            batch_complete=True,
            scope_complete=passed,
            evidence={
                "model_training_authorized": passed,
                "training_readiness": result["training_readiness"],
                "conditions": result["conditions"],
                "pending_obligations": result["pending_obligations"],
                "unresolved_files": len(unresolved),
                "model_safety_gate_status": "deferred",
            },
        )

    return handler


# ---------------------------------------------------------------------------
# Registry factory
# ---------------------------------------------------------------------------

def build_train_v4_handlers(
    *, project_root: Path, output_root: Path, config: dict
) -> dict[str, Callable[[RuntimeOptions], StageResult]]:
    """Handlers for the V4 model-training gate chain (sections 1-3)."""
    return {
        "FreezeTrain1600V3Evidence": _freeze_train1600_v3_handler(
            project_root, output_root, config
        ),
        "AuditTrain1600LearnabilityV4": _audit_learnability_v4_handler(
            project_root, output_root, config
        ),
        "AuditModelTrainingAuthorizationV4": (
            _audit_model_training_authorization_v4_handler(
                project_root, output_root, config
            )
        ),
    }
