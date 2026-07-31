from __future__ import annotations

import numpy as np
import pandas as pd


def nominal_safe_action(actuators: pd.DataFrame, phase: str, rainfall_mm_h: float) -> np.ndarray:
    n = len(actuators)
    u = np.ones(n, dtype=np.float32)
    if n == 0:
        return u
    typ = actuators["link_type"].to_numpy(str)
    storage = actuators.get("near_storage", pd.Series(False, index=actuators.index)).to_numpy(bool)
    risk = min(1.0, max(0.0, rainfall_mm_h / 80.0))
    if phase in ("pre_peak", "peak"):
        u[storage & (typ != "pump")] = np.clip(0.80 - 0.65 * risk, 0.10, 0.90)
        u[typ == "pump"] = np.clip(0.55 - 0.25 * risk, 0.15, 0.75)
    else:
        u[storage & (typ != "pump")] = 0.85
        u[typ == "pump"] = 1.0
    return u.astype(np.float32)

