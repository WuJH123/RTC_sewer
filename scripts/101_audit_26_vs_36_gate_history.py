from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sewerrtc.data.three_step_research_builders import summarize_gate_comparison
from sewerrtc.io.project_paths import cfg_path, load_config


def _path(root: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else root / p


def main() -> None:
    parser = argparse.ArgumentParser(description="Explain why the 26-asset v8 line appeared successful and why the 36-asset line is blocked.")
    parser.add_argument("--config", default="configs/wuhan_project6_36_temporal_joint.yaml")
    parser.add_argument("--gate26", default="outputs/evaluation_project6_no_control_repair_formal_30_v8/no_control_repair_gate.json")
    parser.add_argument("--gate36", default="outputs/evaluation_project6_v8_storage_T5_T100_v1/no_control_repair_gate.json")
    parser.add_argument("--out-json", default="outputs/research_reuse_plan/gate_26_vs_36_audit.json")
    args = parser.parse_args()
    cfg = load_config(args.config)
    root = cfg_path(cfg, "project_root")
    gate26_path = _path(root, args.gate26)
    gate36_path = _path(root, args.gate36)
    gate26 = json.loads(gate26_path.read_text(encoding="utf-8"))
    gate36 = json.loads(gate36_path.read_text(encoding="utf-8"))
    summary = summarize_gate_comparison(gate26, gate36)
    summary["inputs"] = {"gate26": str(gate26_path), "gate36": str(gate36_path)}
    out = _path(root, args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
