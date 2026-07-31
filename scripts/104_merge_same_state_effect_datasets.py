from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from sewerrtc.data.effect_dataset_merge import merge_effect_payloads


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dataset", required=True)
    parser.add_argument("--supplement-dataset", required=True)
    parser.add_argument("--out-npz", required=True)
    parser.add_argument("--base-split-policy", choices=("preserve", "all_train"), default="preserve")
    parser.add_argument("--locked-validation-events-file")
    args = parser.parse_args()

    base_path = Path(args.base_dataset)
    supplement_path = Path(args.supplement_dataset)
    base = np.load(base_path, allow_pickle=True)
    supplement = np.load(supplement_path, allow_pickle=True)
    locked_validation_events: set[str] | None = None
    if args.locked_validation_events_file:
        lock_path = Path(args.locked_validation_events_file)
        locked_validation_events = {
            line.strip() for line in lock_path.read_text(encoding="utf-8").splitlines() if line.strip()
        }
    payload, report = merge_effect_payloads(
        base,
        supplement,
        base_split_policy=str(args.base_split_policy),
        locked_validation_events=locked_validation_events,
    )
    out = Path(args.out_npz)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, **payload)
    report = {
        **report,
        "base_dataset": str(base_path.resolve()),
        "supplement_dataset": str(supplement_path.resolve()),
        "out_npz": str(out.resolve()),
    }
    out.with_suffix(".merge_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
