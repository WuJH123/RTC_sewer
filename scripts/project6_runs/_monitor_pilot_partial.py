"""Read-only Pilot partial distribution monitor (no SWMM, no writes)."""
import sys
from pathlib import Path

import pandas as pd
import yaml

PROJECT = Path(r"e:\RTC_sewer\Project6")
sys.path.insert(0, str(PROJECT))

from sewerrtc.v4.pipeline import _pilot_partial_bundle  # noqa: E402


def main() -> None:
    root = PROJECT / "outputs" / "project6_dual_reference_v4" / "final_v4" / "pilot"
    cfg = yaml.safe_load(
        (PROJECT / "configs" / "wuhan_project6_v4_final.yaml").read_text(
            encoding="utf-8"
        )
    )
    _plan, bundle = _pilot_partial_bundle(PROJECT, root.parent, cfg)
    m = bundle["sample_manifest"]
    print("accepted=", len(m), "rejected=", len(bundle["rejected"]),
          "dups=", len(bundle["actual_duplicates"]),
          "pending=", len(bundle["pending"]))
    if not len(m):
        return
    print("events:", m.groupby("event_id").size().to_dict())
    print("states=", m["checkpoint_id"].nunique())
    print("families:", m.groupby("candidate_family").size().to_dict())
    if "k_target" in m:
        print("K:", m.groupby("k_target").size().to_dict())
    for c in (
        "delta_pfv_h120_vs_no_control",
        "delta_tfv_h120_vs_dynamic_internal",
        "delta_peak_h120_vs_dynamic_internal",
    ):
        print(
            c,
            "min=%.5g med=%.5g max=%.5g nuniq=%d"
            % (m[c].min(), m[c].median(), m[c].max(), m[c].nunique()),
        )
    labels = [
        c
        for c in (
            "pfv_safe", "pfv_unsafe", "tfv_improved", "tfv_degraded",
            "peak_noninferior", "peak_degraded", "joint_noninferior",
            "joint_unsafe", "materially_beneficial", "neutral",
            "hard_negative", "pfv_safe_peak_hard_negative", "is_noop",
        )
        if c in m
    ]
    print("labels:", {c: int(m[c].astype(bool).sum()) for c in labels})
    if "local_response_magnitude" in m:
        nz = (m["local_response_magnitude"].abs() > 0).mean()
        print("local_response_nonzero_rate=%.3f" % nz)


if __name__ == "__main__":
    main()
