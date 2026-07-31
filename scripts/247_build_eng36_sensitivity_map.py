"""Gate 5 Phase 4: Engineering36 Sensitivity Map.

Builds sensitivity map from ablation results:
  - PFV sensitivity
  - TFV sensitivity
  - Peak sensitivity
  - Priority-node influence
  - Response lag
  - Action direction
  - Confidence
  - Data support count

Classifies facilities:
  - pfv_protective
  - tfv_improving
  - peak_reducing
  - conflict
  - low_response
  - unsupported

Output:
  - eng36_sensitivity_map.csv
  - eng36_sensitivity_audit.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

OUT_DIR = PROJECT_ROOT / "outputs" / "project6_dual_reference_v4" / "recovery_capability_v2" / "gate4_h120_batch0"
ABLATION_DIR = OUT_DIR / "ablation_uniform90"
GATE5_DIR = OUT_DIR / "gate5_exact_diagnosis"
ACTUATOR_CSV = PROJECT_ROOT / "data" / "project6_v3_facility_semantics_36.csv"


def main():
    print("=" * 70)
    print("  Gate 5 Phase 4: Engineering36 Sensitivity Map")
    print("=" * 70)

    # Load ablation results
    loo_csv = ABLATION_DIR / "facility_marginal_effects.csv"
    group_csv = ABLATION_DIR / "facility_group_marginal_effects.csv"
    pert_csv = ABLATION_DIR / "facility_perturbation_effects.csv"

    if not loo_csv.exists():
        print(f"  ERROR: {loo_csv} not found. Run script 243 first.")
        return 1

    loo_df = pd.read_csv(loo_csv)
    print(f"  LOO data: {len(loo_df)} facilities")

    # Load facility semantics
    semantics = pd.read_csv(ACTUATOR_CSV)

    # Compute sensitivity metrics
    sensitivity_rows = []
    for _, row in loo_df.iterrows():
        fid = row["facility_id"]
        error = row.get("error")
        if error is not None and str(error) != "nan":
            sensitivity_rows.append({
                "facility_id": fid,
                "pfv_sensitivity": 0.0,
                "tfv_sensitivity": 0.0,
                "peak_sensitivity": 0.0,
                "pfv_direction": "unknown",
                "tfv_direction": "unknown",
                "classification": "unsupported",
                "confidence": 0.0,
                "data_support_count": 0,
                "error": str(error),
            })
            continue

        delta_pfv = row.get("delta_pfv_vs_nc", 0)
        delta_tfv = row.get("delta_tfv_vs_di", 0)
        delta_peak = row.get("delta_peak_vs_di", 0)

        # Handle NaN
        if pd.isna(delta_pfv):
            delta_pfv = 0
        if pd.isna(delta_tfv):
            delta_tfv = 0
        if pd.isna(delta_peak):
            delta_peak = 0

        # Sensitivity = absolute change when facility is restored
        pfv_sens = abs(delta_pfv)
        tfv_sens = abs(delta_tfv)
        peak_sens = abs(delta_peak)

        # Direction: if restoring facility to 0.5 REDUCES the metric, then
        # the uniform_90pct action was INCREASING the metric
        pfv_dir = "pfv_reducing" if delta_pfv < 0 else "pfv_increasing"
        tfv_dir = "tfv_reducing" if delta_tfv < 0 else "tfv_increasing"

        # Classification
        pfv_protective = delta_pfv < -0.1  # Restoring reduces PFV -> 90pct was bad for PFV
        tfv_improving = delta_tfv < -0.1  # Restoring reduces TFV -> 90pct was bad for TFV
        peak_reducing = delta_peak < -0.0005

        if pfv_protective and tfv_improving:
            classification = "conflict"  # Both PFV and TFV sensitive
        elif pfv_protective:
            classification = "pfv_protective"
        elif tfv_improving:
            classification = "tfv_improving"
        elif peak_reducing:
            classification = "peak_reducing"
        elif pfv_sens < 0.01 and tfv_sens < 0.01 and peak_sens < 0.0001:
            classification = "low_response"
        else:
            classification = "unsupported"

        # Confidence: based on magnitude of effect
        confidence = min(1.0, (pfv_sens + tfv_sens / 10 + peak_sens * 100) / 3)

        # Get semantics info
        sem_row = semantics[semantics["facility_id"] == fid]
        actuator_type = sem_row["actuator_type"].iloc[0] if not sem_row.empty else "unknown"
        storage_role = sem_row.get("storage_role", pd.Series(["none"])).iloc[0] if not sem_row.empty else "none"

        sensitivity_rows.append({
            "facility_id": fid,
            "actuator_type": actuator_type,
            "storage_role": storage_role,
            "pfv_sensitivity": round(pfv_sens, 4),
            "tfv_sensitivity": round(tfv_sens, 4),
            "peak_sensitivity": round(peak_sens, 6),
            "delta_pfv_when_restored": round(delta_pfv, 4),
            "delta_tfv_when_restored": round(delta_tfv, 4),
            "delta_peak_when_restored": round(delta_peak, 6),
            "pfv_direction": pfv_dir,
            "tfv_direction": tfv_dir,
            "classification": classification,
            "confidence": round(confidence, 3),
            "data_support_count": 1,
        })

    sens_df = pd.DataFrame(sensitivity_rows)
    sens_df.to_csv(GATE5_DIR / "eng36_sensitivity_map.csv", index=False)

    # Summary
    print(f"\n  Classification summary:")
    for cls, count in sens_df["classification"].value_counts().items():
        print(f"    {cls}: {count}")

    # Top PFV-protective facilities
    pfv_prot = sens_df[sens_df["classification"] == "pfv_protective"].sort_values("pfv_sensitivity", ascending=False)
    print(f"\n  Top PFV-protective facilities:")
    for _, r in pfv_prot.head(5).iterrows():
        print(f"    {r['facility_id']}: PFV sens={r['pfv_sensitivity']:.4f}")

    # Top TFV-improving facilities
    tfv_imp = sens_df[sens_df["classification"] == "tfv_improving"].sort_values("tfv_sensitivity", ascending=False)
    print(f"\n  Top TFV-improving facilities:")
    for _, r in tfv_imp.head(5).iterrows():
        print(f"    {r['facility_id']}: TFV sens={r['tfv_sensitivity']:.4f}")

    # Audit JSON
    audit = {
        "n_facilities_analyzed": len(sens_df),
        "n_with_error": int(sens_df["classification"].eq("unsupported").sum()),
        "classification_counts": sens_df["classification"].value_counts().to_dict(),
        "top_pfv_protective": pfv_prot["facility_id"].head(5).tolist() if not pfv_prot.empty else [],
        "top_tfv_improving": tfv_imp["facility_id"].head(5).tolist() if not tfv_imp.empty else [],
        "low_response_count": int((sens_df["classification"] == "low_response").sum()),
    }

    (GATE5_DIR / "eng36_sensitivity_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"\n  Outputs saved to {GATE5_DIR}")
    print("=" * 70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
