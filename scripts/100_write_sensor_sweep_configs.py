from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from sewerrtc.io.project_paths import cfg_path, ensure_dir, load_config


def _label(ratio: float) -> str:
    return f"sr{ratio:.2f}".replace(".", "p")


def main() -> None:
    parser = argparse.ArgumentParser(description="Write sensor-ratio configs for mixed-trajectory GAT training/evaluation.")
    parser.add_argument("--base-config", default="configs/wuhan_project6_36_temporal_joint.yaml")
    parser.add_argument("--ratios", default="0.05,0.10,0.15,0.20,0.30")
    parser.add_argument("--cache-dir", default="outputs/cache_research_mixed_gat")
    parser.add_argument("--out-config-dir", default="configs/research_sensor_sweep")
    parser.add_argument("--out-root", default="outputs/research_sensor_sweep")
    args = parser.parse_args()
    cfg = load_config(args.base_config)
    root = cfg_path(cfg, "project_root")
    base_config_path = Path(str(cfg["_config_path"])).resolve()
    out_config_dir = ensure_dir(root / args.out_config_dir)
    ratios = [float(item.strip()) for item in args.ratios.replace(";", ",").split(",") if item.strip()]
    rows = []
    for ratio in ratios:
        label = _label(ratio)
        config_path = out_config_dir / f"wuhan_{label}.yaml"
        payload = {
            "_inherits": base_config_path.as_posix(),
            "experiment": {"sensor_ratio": float(ratio)},
            "outputs": {
                "cache": args.cache_dir,
                "design": f"{args.out_root}/{label}/design",
                "models": f"{args.out_root}/{label}/models",
                "diagnostics": f"{args.out_root}/{label}/diagnostics",
                "gat_features": f"{args.out_root}/{label}/gat_reconstructed_features",
            },
        }
        config_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")
        rows.append({"sensor_ratio": ratio, "label": label, "config": str(config_path)})
    summary = {"configs": rows, "cache_dir": str(root / args.cache_dir), "out_config_dir": str(out_config_dir)}
    (out_config_dir / "sensor_sweep_config_manifest.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
