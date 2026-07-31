#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sewerrtc.state.gat_robustness import run_sr0p15_robustness_audit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run sr0p15-only GAT robustness diagnostics.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--gat-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--flush-every", type=int, default=32)
    parser.add_argument("--max-memory-gb", type=float, default=4.0)
    parser.add_argument("--scenario-filter", default="")
    parser.add_argument("--seed", type=int, default=150)
    parser.add_argument("--validation-manifest", default="")
    return parser.parse_args()


def _cache_path_from_manifest(path: Path) -> Path | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key in ("cache_path", "state_cache_path", "node_truth_path", "sensor_input_path"):
            value = (row.get(key) or "").strip()
            if value and value.lower().endswith(".npz") and Path(value).exists():
                return Path(value)
    return None


def main() -> int:
    args = parse_args()
    config = Path(args.config)
    gat_dir = Path(args.gat_dir)
    out_dir = Path(args.out_dir)
    if not config.is_absolute():
        config = ROOT / config
    if not gat_dir.is_absolute():
        gat_dir = ROOT / gat_dir
    if not out_dir.is_absolute():
        out_dir = ROOT / out_dir
    validation_cache = None
    if args.validation_manifest:
        manifest = Path(args.validation_manifest)
        if not manifest.is_absolute():
            manifest = ROOT / manifest
        validation_cache = _cache_path_from_manifest(manifest)
        if validation_cache is None:
            out_dir.mkdir(parents=True, exist_ok=True)
            gate = out_dir / "gat_sr0p15_independent_robustness_gate.json"
            gate.write_text(
                json.dumps(
                    {
                        "status": "blocked",
                        "blocking_reason": "No usable NPZ validation cache was found in the independent manifest",
                        "validation_manifest": str(manifest),
                        "allowed_to_build_node_state": False,
                        "allowed_to_enter_prompt3a": False,
                        "round0_unlock_allowed": False,
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(json.dumps({"status": "blocked", "reason": "independent validation manifest lacks usable NPZ cache", "gate": str(gate)}, indent=2))
            return 3
    outputs = run_sr0p15_robustness_audit(
        config,
        gat_dir,
        out_dir,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        resume=args.resume,
        flush_every=args.flush_every,
        max_memory_gb=args.max_memory_gb,
        scenario_filter=args.scenario_filter,
        seed=args.seed,
        validation_cache_path=validation_cache,
    )
    gate_path = outputs.get("gate") or outputs.get("gat_sr0p15_robustness_gate.json") or (out_dir / "gat_sr0p15_robustness_gate.json")
    status = "completed"
    if gate_path.exists():
        gate = json.loads(gate_path.read_text(encoding="utf-8-sig"))
        if gate.get("status") == "failed" and gate.get("failure_reason") == "out_of_memory":
            status = "failed"
            print(json.dumps({"status": status, "reason": "out_of_memory", "outputs": {k: str(v) for k, v in outputs.items()}}, indent=2))
            return 4
        if gate.get("status") == "blocked_pending_manual_selection_lock":
            status = "blocked"
            print(json.dumps({"status": status, "reason": gate.get("status"), "outputs": {k: str(v) for k, v in outputs.items()}}, indent=2))
            return 3
    print(json.dumps({"status": status, "outputs": {k: str(v) for k, v in outputs.items()}}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
