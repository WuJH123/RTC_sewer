from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from sewerrtc.data.peak_label_semantics import (
    RISK_LABEL_CHANNELS,
    peak_label_semantics_valid,
    repair_paired_risk_rate_sequences,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write a non-destructive same-state dataset with physically consistent peak labels."
    )
    parser.add_argument(
        "--dataset",
        default="outputs/project6_36_temporal_joint_v4/effect_dataset/same_state_raw_joint_36_v3.npz",
    )
    parser.add_argument(
        "--out-dataset",
        default="outputs/project6_36_temporal_joint_peakfixed_v1/effect_dataset/same_state_raw_joint_36_peakfixed_v1.npz",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    source = Path(args.dataset)
    source = source if source.is_absolute() else root / source
    destination = Path(args.out_dataset)
    destination = destination if destination.is_absolute() else root / destination
    if source.resolve() == destination.resolve():
        raise ValueError("Peak-label repair must not overwrite the source dataset")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite repaired dataset: {destination}")

    with np.load(source, allow_pickle=True) as data:
        payload = {key: data[key] for key in data.files}
    old_reference = np.asarray(payload["reference_risk_rate_seq"], dtype=np.float32)
    old_delta = np.asarray(payload["delta_risk_rate_seq"], dtype=np.float32)
    reference, delta = repair_paired_risk_rate_sequences(old_reference, old_delta)
    payload["reference_risk_rate_seq"] = reference.astype(np.float32)
    payload["delta_risk_rate_seq"] = delta.astype(np.float32)
    payload["risk_label_channels"] = np.asarray(RISK_LABEL_CHANNELS)
    payload["peak_label_definition"] = np.asarray(
        "running peak TFV rate; aggregate delta peak=max(candidate TFV rate)-max(reference TFV rate)"
    )

    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **payload)
    candidate = reference + delta
    report = {
        "source_dataset": str(source.resolve()),
        "source_sha256": _sha256(source),
        "out_dataset": str(destination.resolve()),
        "out_sha256": _sha256(destination),
        "samples": int(reference.shape[0]),
        "horizon_steps": int(reference.shape[1]),
        "risk_label_channels": list(RISK_LABEL_CHANNELS),
        "old_reference_channel_2_duplicated_TFV_rate": bool(
            np.allclose(old_reference[:, :, 2], old_reference[:, :, 1])
        ),
        "old_delta_channel_2_duplicated_TFV_rate": bool(
            np.allclose(old_delta[:, :, 2], old_delta[:, :, 1])
        ),
        "reference_peak_semantics_valid": peak_label_semantics_valid(reference),
        "candidate_peak_semantics_valid": peak_label_semantics_valid(candidate),
        "source_overwritten": False,
    }
    report_path = destination.with_suffix(".peak_label_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
