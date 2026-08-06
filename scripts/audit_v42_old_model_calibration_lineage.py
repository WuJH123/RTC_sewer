"""Record old Step-2 model/calibration compatibility before repair."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-root", type=Path, required=True)
    ap.add_argument("--calibration", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    models = {}
    for seed in (17, 42, 73):
        directory = args.model_root / f"seed_{seed}"
        model = directory / "best_model.pt"
        report = json.loads((directory / "formal_step2_report.json").read_text(encoding="utf-8"))
        models[str(seed)] = {
            "path": str(model.resolve()),
            "file_sha256": sha256(model),
            "reported_model_sha256": report.get("surrogate_model_sha256"),
            "calibration_model_sha256": calibration.get("model_hashes", {}).get(str(seed)),
            "reported_model_matches_calibration": report.get("surrogate_model_sha256") == calibration.get("model_hashes", {}).get(str(seed)),
        }
    payload = {
        "audit_id": "OLD_MODEL_CALIBRATION_LINEAGE_V1",
        "development_only": True,
        "calibration_path": str(args.calibration.resolve()),
        "calibration_file_sha256": sha256(args.calibration),
        "calibration_status": calibration.get("status"),
        "calibration_development_only": calibration.get("development_only"),
        "models": models,
        "old_calibration_reusable_for_new_model": False,
        "reason": "any head-only or partial-finetuned checkpoint changes the model lineage; recalibration is mandatory before online safety use",
    }
    out = args.output.resolve(); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps({"output": str(out), "all_reported_model_hashes_match": all(v["reported_model_matches_calibration"] for v in models.values())}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
