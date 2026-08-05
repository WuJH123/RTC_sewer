"""Durable sequential retraining for the corrected surrogate action map."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def _write_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seeds", default="17,42,73")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--min-train-groups", type=int, default=65)
    args = parser.parse_args()

    seeds = [int(x.strip()) for x in str(args.seeds).split(",") if x.strip()]
    status_path = args.output_root / "ACTION_MAP_RETRAIN_STATUS.json"
    status = {
        "status": "running",
        "started_at": time.time(),
        "current_seed": None,
        "completed_seeds": [],
        "output_root": str(args.output_root),
        "manifest": str(args.manifest),
        "action_map_contract": "undirected_khop_inverse_distance_v1_radius10",
    }
    _write_status(status_path, status)
    try:
        for seed in seeds:
            status["current_seed"] = seed
            _write_status(status_path, status)
            output_dir = args.output_root / f"seed_{seed}"
            command = [
                sys.executable,
                "-u",
                str(args.project_root / "scripts/train_v42_step2_formal_f2.py"),
                "--project-root",
                str(args.project_root),
                "--manifest",
                str(args.manifest),
                "--output-dir",
                str(output_dir),
                "--seed",
                str(seed),
                "--split-seed",
                "42",
                "--epochs",
                str(args.epochs),
                "--patience",
                str(args.patience),
                "--batch-size",
                str(args.batch_size),
                "--min-train-groups",
                str(args.min_train_groups),
                "--target-contract",
                "CONTROL_CORE",
            ]
            print("RUN: " + " ".join(command), flush=True)
            subprocess.run(command, cwd=str(args.project_root), check=True)
            status["completed_seeds"].append(seed)
            _write_status(status_path, status)
        status["status"] = "pass"
        status["current_seed"] = None
        status["finished_at"] = time.time()
        _write_status(status_path, status)
        return 0
    except Exception as exc:
        status["status"] = "fail"
        status["last_failure"] = f"{type(exc).__name__}: {exc}"
        status["finished_at"] = time.time()
        _write_status(status_path, status)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
