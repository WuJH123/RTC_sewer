"""Prewarm the lossless projected-detail cache used by Formal Step1.

This is metadata-driven: it touches only unique detail files referenced by the
frozen Step1 manifest and never constructs training tensors or runs a model.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sewerrtc.v4.v42_step1_dataset import _build_usecols, load_graph_assets
from sewerrtc.v4.v42_step1_streaming import (
    _read_manifest,
    _projected_cache_location,
    _read_projected_detail,
)


def _atomic_json(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be >= 1")
    frame = _read_manifest(args.manifest)
    frame = frame[frame["step1_domain_role"].astype(str).eq("target_formal")].copy()
    if frame.empty:
        raise RuntimeError("frozen Formal Step1 manifest has no target_formal rows")
    groups = []
    for detail_path, rows in frame.groupby("detail_path", sort=True):
        source_identity = "|".join(
            sorted(rows["physical_identity_sha256"].astype(str).unique())
        )
        groups.append((str(detail_path), source_identity))
    if args.limit is not None:
        groups = groups[: max(0, int(args.limit))]

    graph = load_graph_assets(args.project_root)
    required = _build_usecols(graph.node_ids, graph.facility_ids)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    progress_path = args.cache_dir / "CACHE_PROGRESS.json"
    started = time.time()
    totals = {
        "files_total": len(groups),
        "files_done": 0,
        "cache_hits": 0,
        "cache_written": 0,
        "bytes_written": 0,
        "elapsed": 0.0,
        "files_per_second": 0.0,
        "last_path": "",
    }

    def prepare(item: tuple[str, str]) -> tuple[str, bool, int]:
        path, source_identity = item
        cache_path, _ = _projected_cache_location(
            path, required, cache_dir=args.cache_dir, source_identity=source_identity
        )
        existed = cache_path.exists()
        _read_projected_detail(
            path,
            required,
            cache_dir=args.cache_dir,
            source_identity=source_identity,
        )
        size = cache_path.stat().st_size if cache_path.exists() else 0
        return path, existed, int(size)

    print(f"[CACHE] files_total={len(groups)} workers={args.workers}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(prepare, item) for item in groups]
        for future in as_completed(futures):
            path, existed, size = future.result()
            totals["files_done"] += 1
            totals["cache_hits"] += int(existed)
            totals["cache_written"] += int(not existed)
            totals["bytes_written"] += int(0 if existed else size)
            totals["last_path"] = path
            elapsed = time.time() - started
            totals["elapsed"] = elapsed
            totals["files_per_second"] = totals["files_done"] / max(elapsed, 1e-9)
            if (
                totals["files_done"] % max(1, args.progress_every) == 0
                or totals["files_done"] == totals["files_total"]
            ):
                _atomic_json(progress_path, totals)
                print(
                    f"[CACHE] {totals['files_done']}/{totals['files_total']} "
                    f"rate={totals['files_per_second']:.2f}/s "
                    f"hits={totals['cache_hits']} written={totals['cache_written']}",
                    flush=True,
                )
    _atomic_json(progress_path, totals)
    print(json.dumps(totals, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
