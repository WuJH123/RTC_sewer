from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sewerrtc.io.project_paths import cfg_path, load_config
from sewerrtc.io.rainfall_injection import build_rainfall_library


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/wuhan.yaml")
    ap.add_argument("--mode", choices=["debug", "train", "formal"], default="debug")
    args = ap.parse_args()
    cfg = load_config(args.config)
    rain = cfg["rainfall"]
    if args.mode == "debug":
        ids, durations = rain["debug_rain_ids"], rain["debug_durations"]
        patterns = rain.get("debug_patterns", rain["temporal_patterns"][:1])
    elif args.mode == "train":
        ids, durations = rain["train_rain_ids"], rain["train_durations"]
        patterns = rain.get("train_patterns", rain["temporal_patterns"])
    else:
        ids, durations = rain["formal_rain_ids"], rain["formal_durations"]
        patterns = rain.get("formal_patterns", rain["temporal_patterns"])
    table = build_rainfall_library(
        ids,
        [int(x) for x in durations],
        rain["design_depth_mm"],
        patterns,
        cfg_path(cfg, "outputs.rainfall"),
        int(cfg["experiment"]["recession_min"]),
    )
    print(json.dumps({"mode": args.mode, "events": len(table), "out": str(cfg_path(cfg, "outputs.rainfall"))}, indent=2))


if __name__ == "__main__":
    main()
