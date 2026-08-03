"""Run one Temporal Sparse GAT qualification seed without creating Formal evidence."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUALIFICATION_CONTRACT = "PROJECT6_V42_QUALIFICATION_FIRST_PASS_V1"
EIGHT_GIB = 8 * 1024**3
QUALIFICATION_BATCH_CAP_8GB = 64


def _effective_batch_size(requested: int, total_memory_bytes: int | None) -> int:
    requested = max(1, int(requested))
    if total_memory_bytes is not None and int(total_memory_bytes) <= EIGHT_GIB:
        return min(requested, QUALIFICATION_BATCH_CAP_8GB)
    return requested


def _cuda_total_memory_bytes() -> int | None:
    try:
        import torch

        if torch.cuda.is_available():
            return int(torch.cuda.get_device_properties(0).total_memory)
    except Exception:
        pass
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-seed", type=int, required=True)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--sensor-layout-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument("--min-train-groups", type=int, default=65)
    args = parser.parse_args()

    cuda_memory = _cuda_total_memory_bytes()
    effective_batch_size = _effective_batch_size(args.batch_size, cuda_memory)
    if effective_batch_size != int(args.batch_size):
        print(
            "QUALIFICATION_STEP1_BATCH_FALLBACK: "
            f"requested={args.batch_size} effective={effective_batch_size} "
            f"cuda_memory_gib={cuda_memory / 1024**3:.3f}",
            flush=True,
        )

    command = [
        sys.executable,
        "-u",
        str(args.project_root / "scripts/train_v42_step1_formal_f2.py"),
        "--project-root",
        str(args.project_root),
        "--manifest",
        str(args.manifest),
        "--output-dir",
        str(args.output_dir),
        "--model-seed",
        str(args.model_seed),
        "--split-seed",
        str(args.split_seed),
        "--sensor-layout-seed",
        str(args.sensor_layout_seed),
        "--epochs",
        str(args.epochs),
        "--aux-epochs",
        "0",
        "--batch-size",
        str(effective_batch_size),
        "--patience",
        str(args.patience),
        "--min-train-groups",
        str(args.min_train_groups),
    ]
    print("RUN:", " ".join(command), flush=True)
    child_env = os.environ.copy()
    child_env["RTC_V42_STEP1_AMP"] = "1" if cuda_memory is not None else "0"
    subprocess.run(command, cwd=str(args.project_root), check=True, env=child_env)

    report_path = args.output_dir / "formal_step1_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["source_formal_generation_id"] = report.get("formal_generation_id")
    report["formal_generation_id"] = QUALIFICATION_CONTRACT
    report["contract_id"] = QUALIFICATION_CONTRACT
    report["stage"] = "qualification_step1_single_seed"
    report["qualification_only"] = True
    report["development_only"] = True
    report["formal_mainline_authorized"] = False
    report["formal_evidence_eligible"] = False
    report["requested_batch_size"] = int(args.batch_size)
    report["effective_batch_size"] = int(effective_batch_size)
    report["cuda_total_memory_bytes"] = cuda_memory
    report["mixed_precision_amp"] = bool(cuda_memory is not None)
    report["qualification_note"] = (
        "One-epoch qualification model used only to exercise downstream interfaces. "
        "It cannot substitute for Formal production training."
    )
    qualification_report = args.output_dir / "qualification_step1_report.json"
    qualification_report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    # Remove the misleading formal-named report from the isolated qualification root.
    report_path.unlink()
    print(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
