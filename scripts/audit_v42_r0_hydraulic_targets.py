from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sewerrtc.v4.v42_hydraulic_target_audit import audit_detail_pool, write_audit
from sewerrtc.v4.v42_trajectory_builder import _load_graph_topology, _parse_inp_topology


ROLES = ("candidate", "no_control", "dynamic_internal", "hold_previous")


def _read(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path) if path.suffix.lower() == ".parquet" else pd.read_csv(path)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit raw all-target coverage for the exact R0-derived Step-2 population."
    )
    default_dataset = (
        PROJECT_ROOT / "outputs" / "project6_dual_reference_v4" / "final_v4"
        / "v42_paper" / "step2_surrogate" / "dataset"
    )
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=default_dataset / "trajectory_manifest.parquet",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_dataset / "hydraulic_target_audit.json",
    )
    args = parser.parse_args()

    manifest = _read(args.manifest)
    if manifest.empty:
        raise ValueError("Step-2 trajectory manifest is empty")
    required = {"sample_lineage_sha256"} | {
        f"source_detail_path_{role}" for role in ROLES
    }
    missing = required - set(manifest.columns)
    if missing:
        raise KeyError(f"Step-2 manifest missing target-audit lineage fields: {sorted(missing)}")
    lineages = [
        str(x) for x in manifest["sample_lineage_sha256"].dropna().unique() if str(x)
    ]
    if len(lineages) != 1:
        raise RuntimeError(f"Step-2 manifest must have exactly one sample lineage, got {lineages}")
    lineage = lineages[0]

    detail_paths = sorted(
        {
            str(row[f"source_detail_path_{role}"])
            for _, row in manifest.iterrows()
            for role in ROLES
        }
    )
    graph = _load_graph_topology(args.project_root)
    node_ids = [str(x) for x in graph["node_ids"]]
    facility_ids = [str(x) for x in graph["facility_ids"]]
    nodes, _ = _parse_inp_topology(
        args.project_root / "data" / "wuhan_v8_storage_retrofit.inp"
    )
    storage_ids = [
        str(x) for x in nodes.loc[nodes["node_type"] == "storage", "node_id"].tolist()
    ]
    outfall_ids = [
        str(x) for x in nodes.loc[nodes["node_type"] == "outfall", "node_id"].tolist()
    ]
    payload = audit_detail_pool(
        detail_paths,
        node_ids=node_ids,
        storage_node_ids=storage_ids,
        facility_ids=facility_ids,
        outfall_node_ids=outfall_ids,
        sample_lineage_sha256=lineage,
    )
    payload["trajectory_manifest"] = str(args.manifest)
    payload["case_count"] = int(len(manifest))
    payload["unique_raw_detail_count"] = int(len(detail_paths))
    write_audit(args.output, payload)
    print(json.dumps({k: v for k, v in payload.items() if k != "rows"}, indent=2))
    return 0 if payload["formal_complete"] else 5


if __name__ == "__main__":
    raise SystemExit(main())
