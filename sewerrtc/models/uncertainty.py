from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


class ResidualQuantileUncertainty:
    def __init__(self, target_columns: list[str], q50: np.ndarray, q90: np.ndarray):
        self.target_columns = list(target_columns)
        self.q50 = np.asarray(q50, dtype=float)
        self.q90 = np.asarray(q90, dtype=float)

    @classmethod
    def fit(cls, y_true: pd.DataFrame, y_pred: pd.DataFrame, target_columns: list[str]) -> "ResidualQuantileUncertainty":
        residuals = []
        for col in target_columns:
            r = pd.to_numeric(y_true[col], errors="coerce").fillna(0.0).to_numpy(float) - pd.to_numeric(
                y_pred[f"pred_{col}"], errors="coerce"
            ).fillna(0.0).to_numpy(float)
            residuals.append(r)
        mat = np.vstack(residuals).T
        return cls(target_columns, np.nanquantile(mat, 0.50, axis=0), np.nanquantile(mat, 0.90, axis=0))

    def predict_quantiles(self, pred: pd.DataFrame, *, clip_lower: bool = True) -> pd.DataFrame:
        """Return residual-adjusted quantiles for absolute targets or signed effects.

        Absolute hydraulic risks cannot be negative, whereas a candidate-minus-
        reference action effect must retain its sign.  Callers calibrating an
        effect model therefore pass ``clip_lower=False``.
        """
        out = pd.DataFrame(index=pred.index)
        for j, col in enumerate(self.target_columns):
            base = pd.to_numeric(pred[f"pred_{col}"], errors="coerce").fillna(0.0).to_numpy(float)
            p50 = base + self.q50[j]
            p90 = base + self.q90[j]
            if clip_lower:
                p50 = np.maximum(0.0, p50)
                p90 = np.maximum(0.0, p90)
            out[f"{col}_p50"] = p50
            out[f"{col}_p90"] = p90
        spread = np.maximum(0.0, self.q90 - self.q50)
        denom = np.maximum(1.0, np.abs(self.q50) + np.abs(self.q90))
        out["uncertainty_score"] = float(np.mean(spread / denom))
        return out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, target_columns=np.array(self.target_columns, dtype=object), q50=self.q50, q90=self.q90)

    @classmethod
    def load(cls, path: str | Path) -> "ResidualQuantileUncertainty":
        data = np.load(path, allow_pickle=True)
        return cls([str(x) for x in data["target_columns"].tolist()], data["q50"], data["q90"])
