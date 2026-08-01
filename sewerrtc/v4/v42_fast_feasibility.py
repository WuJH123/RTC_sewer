"""Development-only fast feasibility helpers for the V4.2 paper line.

This module is intentionally *not* part of the formal evidence chain.  It exists
so the project can answer an early go/no-go question with a small, diverse,
auditable subset before spending days on formal multi-seed/calibration/blind
runs.

Fast-pilot semantics
--------------------
* Step 1 may use a small, deterministic set of ``auxiliary_pretrain`` rainfall
  groups for representation pretraining, followed by target-domain fine tuning.
* Step 2 may use ``eligible_source_domain_counterfactual_aux`` cases for a
  development-only control-core surrogate.  These cases are never relabelled as
  formal target-domain evidence.
* Outfall supervision is intentionally omitted in the quick control-core path;
  PFV/TFV/Peak are derived from node flooding-rate trajectories.
* Every output is stamped ``development_only=True`` and cannot authorize the
  formal V4.2 mainline.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .v42_r0_paper_dataset import FOUR_ROLES, _cols, _exact_times, _kpis, _load_graph_topology, _rain, _select
from .v42_reusable_pool_strict import _bool as _strict_bool

FAST_CONTRACT_ID = "PROJECT6_V42_FAST_FEASIBILITY_V1"


@dataclass(frozen=True)
class FastStep2DatasetResult:
    manifest_path: Path
    audit_path: Path
    candidate_cases: int
    accepted_cases: int
    rejected_cases: int
    rainfall_groups: int
    lineage_sha256: str


def _read(path: str | Path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    return pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p)


def _write_table(frame: pd.DataFrame, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.suffix.lower() == ".parquet":
        frame.to_parquet(p, index=False)
    else:
        frame.to_csv(p, index=False)
    return p


def _json_ids(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [str(x) for x in json.loads(str(value))]


def _flag(row: Any, name: str) -> bool:
    value = getattr(row, name, False)
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return False
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value) != 0.0
    text = str(value).strip().casefold()
    if text in {"true", "1", "yes", "y", "t"}:
        return True
    if text in {"false", "0", "no", "n", "f", "", "none", "nan"}:
        return False
    raise ValueError(f"unsupported boolean {name}={value!r}")


def _hash_key(text: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{text}".encode("utf-8")).hexdigest()


def build_fast_step1_aux_allowlist(
    *,
    manifest_path: str | Path,
    output_path: str | Path,
    max_groups: int = 64,
    seed: int = 42,
) -> dict[str, Any]:
    """Select a small deterministic auxiliary rainfall population.

    The selection broadens rainfall/hydraulic exposure cheaply.  It is *not* a
    provenance upgrade: selected groups remain auxiliary and may only be used
    before target-domain fine tuning.
    """
    frame = _read(manifest_path)
    if "step1_domain_role" not in frame.columns or "split_group_key" not in frame.columns:
        raise KeyError("Step1 manifest lacks step1_domain_role/split_group_key")
    aux = frame[frame["step1_domain_role"].astype(str) == "auxiliary_pretrain"].copy()
    if aux.empty:
        raise ValueError("no auxiliary_pretrain groups available")
    groups = sorted(aux["split_group_key"].astype(str).unique())
    ranked = sorted(groups, key=lambda g: (_hash_key(g, seed), g))
    selected = ranked[: min(int(max_groups), len(ranked))]
    payload = {
        "contract_id": FAST_CONTRACT_ID,
        "development_only": True,
        "purpose": "step1_representation_pretraining_only",
        "formal_target_upgrade": False,
        "seed": int(seed),
        "available_aux_groups": int(len(groups)),
        "selected_aux_groups": int(len(selected)),
        "groups": selected,
        "selection_sha256": hashlib.sha256("\n".join(selected).encode("utf-8")).hexdigest(),
    }
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    return payload


def _role_rows_core(case: Any, physical_by_id: dict[str, Any]) -> dict[str, Any]:
    role_rows: dict[str, list[Any]] = {r: [] for r in FOUR_ROLES}
    for pid in _json_ids(getattr(case, "branch_physical_ids", "[]")):
        row = physical_by_id.get(pid)
        if row is None:
            continue
        role = str(getattr(row, "branch_role", ""))
        if role in role_rows:
            role_rows[role].append(row)
    out: dict[str, Any] = {}
    required_flags = (
        "mask_depth",
        "mask_flood",
        "mask_readback",
        "mask_rainfall",
        "mask_finite",
        "windowable_13x12",
    )
    for role in FOUR_ROLES:
        usable = [r for r in role_rows[role] if all(_flag(r, f) for f in required_flags)]
        if not usable:
            raise ValueError(f"no control-core physical branch for role={role}")
        out[role] = sorted(
            usable,
            key=lambda r: (
                str(getattr(r, "physical_identity_sha256", "")),
                str(getattr(r, "detail_path", "")),
            ),
        )[0]
    return out


def _read_core_detail(path: str | Path, node_ids: list[str], facility_ids: list[str]) -> pd.DataFrame:
    p = Path(path)
    required = [
        "elapsed_min",
        "rainfall_mm_h",
        *[f"h:{x}" for x in node_ids],
        *[f"flood:{x}" for x in node_ids],
        *[f"setting:{x}" for x in facility_ids],
    ]
    header = pd.read_csv(p, nrows=0)
    available = set(map(str, header.columns))
    missing = [c for c in required if c not in available]
    if missing:
        raise KeyError(f"core detail missing required columns: {missing[:10]}")
    df = pd.read_csv(p, usecols=required, low_memory=False)
    return df.loc[:, required]


def _select_balanced_cases(meta: list[tuple[Any, str]], *, max_cases: int, seed: int) -> list[tuple[Any, str]]:
    by_group: dict[str, list[Any]] = {}
    for case, group in meta:
        by_group.setdefault(group, []).append(case)
    for group, rows in by_group.items():
        rows.sort(key=lambda r: _hash_key(str(getattr(r, "case_uid", "")), seed))
    groups = sorted(by_group, key=lambda g: (_hash_key(g, seed), g))
    selected: list[tuple[Any, str]] = []
    cursor = 0
    while groups and len(selected) < int(max_cases):
        next_groups: list[str] = []
        for group in groups:
            rows = by_group[group]
            if cursor < len(rows):
                selected.append((rows[cursor], group))
                if len(selected) >= int(max_cases):
                    break
            if cursor + 1 < len(rows):
                next_groups.append(group)
        cursor += 1
        groups = next_groups
    return selected


def build_fast_step2_core_dataset(
    *,
    project_root: str | Path,
    physical_manifest: str | Path,
    case_manifest: str | Path,
    split_manifest: str | Path,
    output_manifest: str | Path,
    audit_output: str | Path,
    max_cases: int = 96,
    seed: int = 42,
) -> FastStep2DatasetResult:
    """Materialize a small source-domain four-reference control-core dataset.

    This is deliberately development-only.  It uses the strict reusable pool's
    ``eligible_source_domain_counterfactual_aux`` label and never changes any
    domain/provenance field.
    """
    root = Path(project_root)
    physical = _read(physical_manifest)
    cases = _read(case_manifest)
    split = _read(split_manifest)
    if physical.empty or cases.empty or split.empty:
        raise ValueError("R0 manifests cannot be empty")
    if "eligible_source_domain_counterfactual_aux" not in cases.columns:
        raise KeyError("case manifest missing eligible_source_domain_counterfactual_aux")

    admitted = cases[_strict_bool(cases, "eligible_source_domain_counterfactual_aux")].copy()
    if "source_role" in admitted.columns:
        admitted = admitted[admitted["source_role"].astype(str) != "reserved_evaluation"].copy()
    if admitted.empty:
        raise ValueError("no source-domain counterfactual auxiliary cases available")

    graph = _load_graph_topology(root)
    node_ids = [str(x) for x in graph["node_ids"]]
    facility_ids = [str(x) for x in graph["facility_ids"]]
    priority_idx = [int(x) for x in __import__(
        "sewerrtc.v4.v42_priority_contract", fromlist=["get_pfv_core_node_indices"]
    ).get_pfv_core_node_indices(node_ids)]

    physical_by_id = {
        str(r.physical_identity_sha256): r for r in physical.itertuples(index=False)
    }
    split_by_id = {
        str(r.physical_identity_sha256): str(r.split_group_key)
        for r in split.itertuples(index=False)
    }

    meta: list[tuple[Any, str]] = []
    preblocked: list[dict[str, str]] = []
    for case in admitted.itertuples(index=False):
        try:
            roles = _role_rows_core(case, physical_by_id)
            group_keys = {
                split_by_id.get(str(getattr(r, "physical_identity_sha256", "")), "")
                for r in roles.values()
            }
            group_keys.discard("")
            if len(group_keys) != 1:
                raise ValueError("four branches do not share one split group")
            meta.append((case, next(iter(group_keys))))
        except Exception as exc:
            preblocked.append({"case_uid": str(getattr(case, "case_uid", "")), "error": f"{type(exc).__name__}: {exc}"})

    selected = _select_balanced_cases(meta, max_cases=max_cases, seed=seed)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    detail_cache: dict[str, pd.DataFrame] = {}
    for case, group in selected:
        case_uid = str(getattr(case, "case_uid", ""))
        try:
            checkpoint = float(getattr(case, "checkpoint_min"))
            history_times, future_times = _exact_times(checkpoint)
            roles = _role_rows_core(case, physical_by_id)
            details: dict[str, pd.DataFrame] = {}
            for role, row in roles.items():
                path = str(getattr(row, "detail_path"))
                if path not in detail_cache:
                    detail_cache[path] = _read_core_detail(path, node_ids, facility_ids)
                details[role] = detail_cache[path]
            history = _select(details["candidate"], history_times)
            history_depth = _cols(history, "h:", node_ids)
            history_actions = _cols(history, "setting:", facility_ids)

            branches: dict[str, dict[str, np.ndarray]] = {}
            for role, detail in details.items():
                fut = _select(detail, future_times)
                branches[role] = {
                    "depth": _cols(fut, "h:", node_ids),
                    "flood": _cols(fut, "flood:", node_ids),
                    "action": _cols(fut, "setting:", facility_ids),
                    "rainfall": _rain(fut),
                }
            rainfall = branches["candidate"]["rainfall"]
            for role in FOUR_ROLES[1:]:
                if not np.allclose(rainfall, branches[role]["rainfall"], atol=1e-7, rtol=0.0):
                    raise ValueError(f"future rainfall differs for {role}")
            pfv, tfv, peak = _kpis(branches, priority_idx)
            event_id = str(getattr(case, "event_id", ""))
            rec: dict[str, Any] = {
                "contract_id": FAST_CONTRACT_ID,
                "development_only": True,
                "formal_target_domain": False,
                "case_uid": case_uid,
                "case_id": str(getattr(case, "case_id", "")),
                "event_id": event_id,
                "checkpoint_min": checkpoint,
                "state_key": f"{event_id}::{checkpoint:.6f}",
                "domain_id": str(getattr(case, "domain_id", "")),
                "split_group_key": group,
                "history_depth": json.dumps(history_depth.tolist(), allow_nan=False),
                "history_actions_readback": json.dumps(history_actions.tolist(), allow_nan=False),
                "rainfall_forecast": json.dumps(rainfall.tolist(), allow_nan=False),
                "pfv_delta": float(pfv),
                "tfv_delta": float(tfv),
                "peak_delta": float(peak),
            }
            for role, arrays in branches.items():
                rec[f"action_{role}_readback"] = json.dumps(arrays["action"].tolist(), allow_nan=False)
                rec[f"trajectory_depth_{role}"] = json.dumps(arrays["depth"].tolist(), allow_nan=False)
                rec[f"trajectory_flood_{role}"] = json.dumps(arrays["flood"].tolist(), allow_nan=False)
                rec[f"source_detail_path_{role}"] = str(getattr(roles[role], "detail_path"))
            records.append(rec)
        except Exception as exc:
            failures.append({"case_uid": case_uid, "error": f"{type(exc).__name__}: {exc}"})

    frame = pd.DataFrame(records)
    out = _write_table(frame, output_manifest)
    lineage_payload = "\n".join(sorted(frame.get("case_uid", pd.Series(dtype=str)).astype(str)))
    lineage = hashlib.sha256(lineage_payload.encode("utf-8")).hexdigest() if records else ""
    audit = {
        "contract_id": FAST_CONTRACT_ID,
        "development_only": True,
        "formal_mainline_authorized": False,
        "data_role": "source_domain_four_reference_control_core_pilot",
        "source_domain_relabelled": False,
        "outfall_required": False,
        "candidate_cases": int(len(admitted)),
        "preblocked_cases": int(len(preblocked)),
        "selected_cases": int(len(selected)),
        "accepted_cases": int(len(records)),
        "rejected_cases": int(len(failures)),
        "rainfall_groups": int(frame["split_group_key"].nunique()) if not frame.empty else 0,
        "state_groups": int(frame["state_key"].nunique()) if not frame.empty else 0,
        "sample_lineage_sha256": lineage,
        "selection_seed": int(seed),
        "max_cases": int(max_cases),
        "preblocked_examples": preblocked[:20],
        "failure_examples": failures[:20],
    }
    audit_path = Path(audit_output)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, indent=2, allow_nan=False), encoding="utf-8")
    return FastStep2DatasetResult(
        manifest_path=out,
        audit_path=audit_path,
        candidate_cases=int(len(admitted)),
        accepted_cases=int(len(records)),
        rejected_cases=int(len(failures)),
        rainfall_groups=int(audit["rainfall_groups"]),
        lineage_sha256=lineage,
    )
