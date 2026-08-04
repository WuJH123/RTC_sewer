"""Fixed-window Step1 input-pipeline benchmark; no model training."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sewerrtc.v4.v42_step1_streaming import Step1StreamingDataset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_parquet(args.manifest)
    frame = frame[frame["step1_domain_role"].astype(str).eq("target_formal")].copy()
    frame = frame.sort_values(
        ["detail_path", "anchor_min", "physical_identity_sha256"], kind="mergesort"
    ).head(int(args.limit)).reset_index(drop=True)
    if frame.empty:
        raise RuntimeError("no target_formal windows available")
    expected_records = [
        f"{path}|{float(anchor):.6f}|{group}|{physical}"
        for path, anchor, group, physical in frame[
            ["detail_path", "anchor_min", "split_group_key", "physical_identity_sha256"]
        ].itertuples(index=False, name=None)
    ]
    expected_hash = hashlib.sha256("\n".join(sorted(expected_records)).encode()).hexdigest()
    results = []
    for workers in args.workers:
        dataset = Step1StreamingDataset(
            project_root=args.project_root,
            manifest_frame=frame,
            sensor_ratio=0.10,
            sensor_layout_seed=42,
            domain_roles=("target_formal",),
            cache_dir=args.cache_dir,
            shuffle_files=False,
            iteration_seed=42,
        )
        loader_kwargs = {
            "dataset": dataset,
            "batch_size": int(args.batch_size),
            "shuffle": False,
            "num_workers": int(workers),
            "pin_memory": torch.cuda.is_available(),
            "drop_last": False,
        }
        if workers:
            loader_kwargs.update(
                {
                    "persistent_workers": bool(args.persistent_workers),
                    "prefetch_factor": max(1, int(args.prefetch_factor)),
                }
            )
        loader = DataLoader(**loader_kwargs)
        seen = []
        started = time.perf_counter()
        count = 0
        for batch in loader:
            size = int(batch["target_depth"].shape[0])
            count += size
            for path, anchor, group, physical in zip(
                batch["detail_path"],
                batch["anchor_min"],
                batch["split_group_key"],
                batch["physical_identity_sha256"],
            ):
                seen.append(f"{path}|{float(anchor):.6f}|{group}|{physical}")
        elapsed = time.perf_counter() - started
        identity_hash = hashlib.sha256("\n".join(sorted(seen)).encode()).hexdigest()
        results.append(
            {
                "workers": int(workers),
                "expected_windows": int(len(dataset)),
                "actual_windows": int(count),
                "elapsed_sec": float(elapsed),
                "windows_per_sec": float(count / max(elapsed, 1e-9)),
                "identity_sha256": identity_hash,
                "identity_matches_workers0": identity_hash == expected_hash,
                "rss_mb": float(__import__("psutil").Process(os.getpid()).memory_info().rss / 1e6),
                "persistent_workers": bool(args.persistent_workers),
                "prefetch_factor": int(args.prefetch_factor),
            }
        )
        print(json.dumps(results[-1]), flush=True)
        del loader, dataset
    if not all(item["identity_matches_workers0"] for item in results):
        raise RuntimeError("DataLoader worker benchmark changed the window identity set")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"limit": int(args.limit), "results": results}, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
