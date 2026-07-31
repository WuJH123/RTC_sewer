from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def paired_stats(comp: pd.DataFrame, metrics=("PFV", "TFV", "peak_TFV_rate"), n_boot: int = 5000, seed: int = 2026) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for m in metrics:
        p, b = f"{m}_proposed", f"{m}_baseline"
        if p not in comp or b not in comp:
            continue
        baseline = comp[b].to_numpy(float)
        proposed = comp[p].to_numpy(float)
        mask = np.isfinite(baseline) & np.isfinite(proposed) & (np.abs(baseline) > 1e-6)
        reduction = (baseline[mask] - proposed[mask]) / baseline[mask] * 100
        if len(reduction) == 0:
            rows.append(
                {
                    "metric": m,
                    "n": 0,
                    "n_total": int(len(comp)),
                    "mean_reduction_pct": np.nan,
                    "median_reduction_pct": np.nan,
                    "ci95_low": np.nan,
                    "ci95_high": np.nan,
                    "wilcoxon_p_greater": np.nan,
                    "note": "all baseline values were zero/undefined for percent reduction",
                }
            )
            continue
        boots = []
        for _ in range(n_boot):
            idx = rng.integers(0, len(reduction), len(reduction))
            boots.append(np.mean(reduction[idx]))
        try:
            stat_p = wilcoxon(reduction, alternative="greater").pvalue
        except Exception:
            stat_p = np.nan
        rows.append(
            {
                "metric": m,
                "n": len(reduction),
                "n_total": int(len(comp)),
                "mean_reduction_pct": float(np.mean(reduction)),
                "median_reduction_pct": float(np.median(reduction)),
                "ci95_low": float(np.percentile(boots, 2.5)),
                "ci95_high": float(np.percentile(boots, 97.5)),
                "wilcoxon_p_greater": float(stat_p),
                "note": "",
            }
        )
    return pd.DataFrame(rows)
