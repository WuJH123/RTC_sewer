from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


TARGET_COLUMNS = [
    "PFV_H",
    "TFV_H",
    "peak_TFV_rate_H",
    "priority_peak_depth_H",
    "high_risk_exposure_time_H",
    "full_depth_mean_H",
    "full_depth_p95_H",
    "full_depth_max_H",
    "priority_depth_mean_H",
    "priority_depth_p95_H",
]
NON_FEATURE_COLUMNS = {
    "event_id",
    "policy_id",
    "detail_file",
    "phase",
    "row_index",
    "elapsed_min",
    *TARGET_COLUMNS,
}


def select_horizon_feature_columns(df: pd.DataFrame) -> list[str]:
    """Select numeric surrogate features while excluding labels and identifiers."""
    cols: list[str] = []
    for col in df.columns:
        if col in NON_FEATURE_COLUMNS or str(col).startswith("reference_"):
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        if pd.api.types.is_numeric_dtype(numeric):
            cols.append(str(col))
    return cols


def _feature_matrix(df: pd.DataFrame, feature_columns: list[str]) -> np.ndarray:
    # Construct once to avoid one DataFrame block insertion per feature. The
    # action/path signature expansion can exceed 200 columns in formal runs.
    frame = pd.DataFrame(
        {
            col: pd.to_numeric(df[col], errors="coerce")
            if col in df
            else pd.Series(0.0, index=df.index)
            for col in feature_columns
        },
        index=df.index,
    )
    return frame.fillna(0.0).to_numpy(np.float32)


def _target_matrix(df: pd.DataFrame, target_columns: list[str]) -> np.ndarray:
    frame = pd.DataFrame(
        {
            col: pd.to_numeric(df[col], errors="coerce")
            if col in df
            else pd.Series(0.0, index=df.index)
            for col in target_columns
        },
        index=df.index,
    )
    return frame.fillna(0.0).to_numpy(np.float32)


class HorizonRidgeSurrogate:
    """Lightweight horizon surrogate baseline.

    This is intentionally small and dependency-light: it provides the formal
    action-sequence -> horizon-risk interface before heavier Temporal GNN
    variants are trained.
    """

    def __init__(self, alpha: float = 1e-3):
        self.alpha = float(alpha)
        self.feature_columns: list[str] = []
        self.target_columns: list[str] = TARGET_COLUMNS.copy()
        self.x_mean: np.ndarray | None = None
        self.x_std: np.ndarray | None = None
        self.y_mean: np.ndarray | None = None
        self.y_std: np.ndarray | None = None
        self.coef: np.ndarray | None = None
        self.calibration_margins: dict[str, float] = {}

    def fit(self, df: pd.DataFrame, feature_columns: list[str], target_columns: list[str] | None = None) -> "HorizonRidgeSurrogate":
        target_columns = target_columns or TARGET_COLUMNS
        x = _feature_matrix(df, feature_columns).astype(float)
        y = _target_matrix(df, target_columns).astype(float)
        self.feature_columns = list(feature_columns)
        self.target_columns = list(target_columns)
        self.x_mean = x.mean(axis=0)
        self.x_std = x.std(axis=0) + 1e-6
        self.y_mean = y.mean(axis=0)
        self.y_std = y.std(axis=0) + 1e-6
        xs = (x - self.x_mean) / self.x_std
        ys = (y - self.y_mean) / self.y_std
        xb = np.concatenate([np.ones((xs.shape[0], 1)), xs], axis=1)
        eye = np.eye(xb.shape[1])
        eye[0, 0] = 0.0
        self.coef = np.linalg.solve(xb.T @ xb + self.alpha * eye, xb.T @ ys)
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.coef is None or self.x_mean is None or self.x_std is None or self.y_mean is None or self.y_std is None:
            raise RuntimeError("HorizonRidgeSurrogate is not fitted")
        x = _feature_matrix(df, self.feature_columns).astype(float)
        xs = (x - self.x_mean) / self.x_std
        xb = np.concatenate([np.ones((xs.shape[0], 1)), xs], axis=1)
        y = xb @ self.coef
        y = y * self.y_std + self.y_mean
        out = pd.DataFrame(y, columns=[f"pred_{c}" for c in self.target_columns], index=df.index)
        return out.clip(lower=0.0)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            alpha=self.alpha,
            feature_columns=np.array(self.feature_columns, dtype=object),
            target_columns=np.array(self.target_columns, dtype=object),
            x_mean=self.x_mean,
            x_std=self.x_std,
            y_mean=self.y_mean,
            y_std=self.y_std,
            coef=self.coef,
            calibration_margin_keys=np.array(list(self.calibration_margins), dtype=object),
            calibration_margin_values=np.array(list(self.calibration_margins.values()), dtype=float),
        )

    @classmethod
    def load(cls, path: str | Path) -> "HorizonRidgeSurrogate":
        data = np.load(path, allow_pickle=True)
        obj = cls(float(data["alpha"]))
        obj.feature_columns = [str(x) for x in data["feature_columns"].tolist()]
        obj.target_columns = [str(x) for x in data["target_columns"].tolist()]
        obj.x_mean = data["x_mean"]
        obj.x_std = data["x_std"]
        obj.y_mean = data["y_mean"]
        obj.y_std = data["y_std"]
        obj.coef = data["coef"]
        if "calibration_margin_keys" in data and "calibration_margin_values" in data:
            obj.calibration_margins = {
                str(k): float(v)
                for k, v in zip(data["calibration_margin_keys"].tolist(), data["calibration_margin_values"].tolist())
            }
        return obj


class _TemporalMLP:
    """Lazy wrapper so importing this module does not require torch setup."""

    @staticmethod
    def build(input_dim: int, output_dim: int, hidden_dim: int, layers: int, dropout: float):
        import torch
        from torch import nn

        blocks: list[nn.Module] = []
        dim = int(input_dim)
        for _ in range(max(1, int(layers))):
            blocks.append(nn.Linear(dim, int(hidden_dim)))
            blocks.append(nn.LayerNorm(int(hidden_dim)))
            blocks.append(nn.GELU())
            if float(dropout) > 0:
                blocks.append(nn.Dropout(float(dropout)))
            dim = int(hidden_dim)
        blocks.append(nn.Linear(dim, int(output_dim)))
        return nn.Sequential(*blocks)


class TemporalGraphHorizonSurrogate:
    """Temporal neural horizon surrogate over graph-derived action/state features.

    The current Project5 horizon dataset is stored as graph-derived tabular
    features: reconstructed state summaries, rainfall windows, action sequence
    features, and influence-domain features. This class provides the formal
    neural horizon-surrogate interface used by GAT-MPC while keeping ridge as a
    smoke baseline. It can later be replaced by a full tensor GNN without
    changing the controller contract.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 128,
        layers: int = 3,
        dropout: float = 0.10,
        seed: int = 2026,
        target_transform: str = "log1p",
    ) -> None:
        self.hidden_dim = int(hidden_dim)
        self.layers = int(layers)
        self.dropout = float(dropout)
        self.seed = int(seed)
        self.target_transform = str(target_transform)
        self.feature_columns: list[str] = []
        self.target_columns: list[str] = TARGET_COLUMNS.copy()
        self.x_mean: np.ndarray | None = None
        self.x_std: np.ndarray | None = None
        self.y_mean: np.ndarray | None = None
        self.y_std: np.ndarray | None = None
        self.state_dict: dict | None = None
        self.training_history: list[dict] = []
        self.best_val_loss: float | None = None
        self.calibration_margins: dict[str, float] = {}
        self.effect_state_dict: dict | None = None
        self.effect_y_mean: np.ndarray | None = None
        self.effect_y_std: np.ndarray | None = None
        self.effect_calibration_margins: dict[str, float] = {}
        self.effect_calibration_margins_by_quantile: dict[str, dict[str, float]] = {}

    def _transform_y(self, y: np.ndarray) -> np.ndarray:
        if self.target_transform == "log1p":
            return np.log1p(np.maximum(0.0, y))
        return y

    def _inverse_y(self, y: np.ndarray) -> np.ndarray:
        if self.target_transform == "log1p":
            return np.expm1(y)
        return y

    def fit(
        self,
        df: pd.DataFrame,
        feature_columns: list[str],
        target_columns: list[str] | None = None,
        *,
        epochs: int = 80,
        batch_size: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        val_df: pd.DataFrame | None = None,
        device: str = "cpu",
        patience: int = 12,
    ) -> "TemporalGraphHorizonSurrogate":
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        target_columns = list(target_columns or TARGET_COLUMNS)
        self.feature_columns = list(feature_columns)
        self.target_columns = target_columns
        torch.manual_seed(self.seed)
        if torch.cuda.is_available() and str(device).lower() == "cuda":
            dev = torch.device("cuda")
        else:
            dev = torch.device("cpu")

        x = _feature_matrix(df, self.feature_columns)
        y = self._transform_y(_target_matrix(df, self.target_columns))
        self.x_mean = x.mean(axis=0)
        self.x_std = x.std(axis=0) + 1e-6
        self.y_mean = y.mean(axis=0)
        self.y_std = y.std(axis=0) + 1e-6
        xs = (x - self.x_mean) / self.x_std
        ys = (y - self.y_mean) / self.y_std

        if val_df is None or val_df.empty:
            val_df = df
        vx = (_feature_matrix(val_df, self.feature_columns) - self.x_mean) / self.x_std
        vy = (self._transform_y(_target_matrix(val_df, self.target_columns)) - self.y_mean) / self.y_std

        model = _TemporalMLP.build(
            input_dim=len(self.feature_columns),
            output_dim=len(self.target_columns),
            hidden_dim=self.hidden_dim,
            layers=self.layers,
            dropout=self.dropout,
        ).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
        loss_fn = torch.nn.MSELoss()
        ds = TensorDataset(torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32))
        dl = DataLoader(ds, batch_size=max(1, int(batch_size)), shuffle=True)
        val_x = torch.tensor(vx, dtype=torch.float32, device=dev)
        val_y = torch.tensor(vy, dtype=torch.float32, device=dev)
        best_state: dict | None = None
        best_loss = float("inf")
        stale = 0
        self.training_history = []
        for ep in range(1, max(1, int(epochs)) + 1):
            model.train()
            losses = []
            for xb, yb in dl:
                xb = xb.to(dev)
                yb = yb.to(dev)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                opt.zero_grad()
                loss.backward()
                opt.step()
                losses.append(float(loss.detach().cpu()))
            model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(model(val_x), val_y).detach().cpu())
            train_loss = float(np.mean(losses)) if losses else float("nan")
            self.training_history.append({"epoch": int(ep), "train_loss": train_loss, "val_loss": val_loss})
            if val_loss < best_loss - 1e-8:
                best_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if int(patience) > 0 and stale >= int(patience):
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        self.state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        self.best_val_loss = float(best_loss)
        self._fit_effect_head(
            df,
            val_df,
            target_columns,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            weight_decay=weight_decay,
            device=str(device),
            patience=patience,
        )
        return self

    def _fit_effect_head(
        self,
        df: pd.DataFrame,
        val_df: pd.DataFrame,
        target_columns: list[str],
        *,
        epochs: int,
        batch_size: int,
        lr: float,
        weight_decay: float,
        device: str,
        patience: int,
    ) -> None:
        reference_cols = [f"reference_{c}" for c in target_columns]
        if not all(c in df.columns for c in reference_cols):
            self.effect_state_dict = None
            return
        effect_cols = [f"effect_{c}" for c in target_columns]
        if "effect_label_mode" in df.columns:
            exact_mask = df["effect_label_mode"].astype(str).str.startswith("exact_no_control_")
            if exact_mask.any():
                df = df.loc[exact_mask].copy()
                if "effect_label_mode" in val_df.columns:
                    exact_val = val_df["effect_label_mode"].astype(str).str.startswith("exact_no_control_")
                    if exact_val.any():
                        val_df = val_df.loc[exact_val].copy()
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        dev = torch.device("cuda" if str(device).lower() == "cuda" and torch.cuda.is_available() else "cpu")
        x = _feature_matrix(df, self.feature_columns)
        y = (
            _target_matrix(df, effect_cols)
            if all(c in df.columns for c in effect_cols)
            else _target_matrix(df, target_columns) - _target_matrix(df, reference_cols)
        )
        self.effect_y_mean = y.mean(axis=0)
        self.effect_y_std = y.std(axis=0) + 1e-6
        xs = (x - self.x_mean) / self.x_std
        ys = (y - self.effect_y_mean) / self.effect_y_std
        vx = (_feature_matrix(val_df, self.feature_columns) - self.x_mean) / self.x_std
        val_effect = (
            _target_matrix(val_df, effect_cols)
            if all(c in val_df.columns for c in effect_cols)
            else _target_matrix(val_df, target_columns) - _target_matrix(val_df, reference_cols)
        )
        vy = (val_effect - self.effect_y_mean) / self.effect_y_std
        torch.manual_seed(self.seed + 101)
        model = _TemporalMLP.build(len(self.feature_columns), len(target_columns), self.hidden_dim, self.layers, self.dropout).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))
        loss_fn = torch.nn.SmoothL1Loss()
        dl = DataLoader(TensorDataset(torch.tensor(xs, dtype=torch.float32), torch.tensor(ys, dtype=torch.float32)), batch_size=max(1, int(batch_size)), shuffle=True)
        vx_t = torch.tensor(vx, dtype=torch.float32, device=dev)
        vy_t = torch.tensor(vy, dtype=torch.float32, device=dev)
        best_state = None
        best_loss = float("inf")
        stale = 0
        for _ in range(max(1, int(epochs))):
            model.train()
            for xb, yb in dl:
                pred = model(xb.to(dev))
                loss = loss_fn(pred, yb.to(dev))
                opt.zero_grad()
                loss.backward()
                opt.step()
            model.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(model(vx_t), vy_t).detach().cpu())
            if val_loss < best_loss - 1e-8:
                best_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if int(patience) > 0 and stale >= int(patience):
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        self.effect_state_dict = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    def _build_model(self, device: str = "cpu"):
        import torch

        if self.state_dict is None or self.x_mean is None:
            raise RuntimeError("TemporalGraphHorizonSurrogate is not fitted")
        dev = torch.device("cuda" if str(device).lower() == "cuda" and torch.cuda.is_available() else "cpu")
        model = _TemporalMLP.build(
            input_dim=len(self.feature_columns),
            output_dim=len(self.target_columns),
            hidden_dim=self.hidden_dim,
            layers=self.layers,
            dropout=self.dropout,
        ).to(dev)
        model.load_state_dict(self.state_dict)
        model.eval()
        return model, dev

    def predict(self, df: pd.DataFrame, *, batch_size: int = 4096, device: str = "cpu") -> pd.DataFrame:
        import torch

        if self.x_mean is None or self.x_std is None or self.y_mean is None or self.y_std is None:
            raise RuntimeError("TemporalGraphHorizonSurrogate is not fitted")
        x = _feature_matrix(df, self.feature_columns)
        xs = (x - self.x_mean) / self.x_std
        model, dev = self._build_model(device)
        outs = []
        with torch.no_grad():
            for start in range(0, len(xs), max(1, int(batch_size))):
                batch = torch.tensor(xs[start : start + int(batch_size)], dtype=torch.float32, device=dev)
                y = model(batch).detach().cpu().numpy()
                outs.append(y)
        ys = np.concatenate(outs, axis=0) if outs else np.zeros((0, len(self.target_columns)), dtype=float)
        y = ys * self.y_std + self.y_mean
        y = self._inverse_y(y)
        out = pd.DataFrame(y, columns=[f"pred_{c}" for c in self.target_columns], index=df.index)
        return out.clip(lower=0.0)

    def predict_effect(self, df: pd.DataFrame, *, batch_size: int = 4096, device: str = "cpu") -> pd.DataFrame:
        import torch
        if self.effect_state_dict is None or self.effect_y_mean is None or self.effect_y_std is None:
            return pd.DataFrame(0.0, index=df.index, columns=[f"pred_{c}" for c in self.target_columns])
        x = _feature_matrix(df, self.feature_columns)
        xs = (x - self.x_mean) / self.x_std
        model, dev = self._build_model(device)
        model.load_state_dict(self.effect_state_dict)
        outs = []
        with torch.no_grad():
            for start in range(0, len(xs), max(1, int(batch_size))):
                batch = torch.tensor(xs[start : start + int(batch_size)], dtype=torch.float32, device=dev)
                outs.append(model(batch).detach().cpu().numpy())
        ys = np.concatenate(outs, axis=0) if outs else np.zeros((0, len(self.target_columns)), dtype=float)
        y = ys * self.effect_y_std + self.effect_y_mean
        return pd.DataFrame(y, columns=[f"pred_{c}" for c in self.target_columns], index=df.index)

    def save(self, path: str | Path) -> None:
        import torch

        if self.state_dict is None:
            raise RuntimeError("TemporalGraphHorizonSurrogate is not fitted")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_kind": "temporal_gnn",
                "architecture": "temporal_graph_feature_mlp",
                "hidden_dim": self.hidden_dim,
                "layers": self.layers,
                "dropout": self.dropout,
                "seed": self.seed,
                "target_transform": self.target_transform,
                "feature_columns": self.feature_columns,
                "target_columns": self.target_columns,
                "x_mean": self.x_mean,
                "x_std": self.x_std,
                "y_mean": self.y_mean,
                "y_std": self.y_std,
                "state_dict": self.state_dict,
                "training_history": self.training_history,
                "best_val_loss": self.best_val_loss,
                "calibration_margins": self.calibration_margins,
                "effect_state_dict": self.effect_state_dict,
                "effect_y_mean": self.effect_y_mean,
                "effect_y_std": self.effect_y_std,
                "effect_calibration_margins": self.effect_calibration_margins,
                "effect_calibration_margins_by_quantile": self.effect_calibration_margins_by_quantile,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "TemporalGraphHorizonSurrogate":
        import torch

        data = torch.load(Path(path), map_location="cpu", weights_only=False)
        obj = cls(
            hidden_dim=int(data.get("hidden_dim", 128)),
            layers=int(data.get("layers", 3)),
            dropout=float(data.get("dropout", 0.10)),
            seed=int(data.get("seed", 2026)),
            target_transform=str(data.get("target_transform", "log1p")),
        )
        obj.feature_columns = [str(x) for x in data.get("feature_columns", [])]
        obj.target_columns = [str(x) for x in data.get("target_columns", TARGET_COLUMNS)]
        obj.x_mean = np.asarray(data.get("x_mean"), dtype=np.float32)
        obj.x_std = np.asarray(data.get("x_std"), dtype=np.float32)
        obj.y_mean = np.asarray(data.get("y_mean"), dtype=np.float32)
        obj.y_std = np.asarray(data.get("y_std"), dtype=np.float32)
        obj.state_dict = data.get("state_dict")
        obj.training_history = list(data.get("training_history", []))
        best = data.get("best_val_loss")
        obj.best_val_loss = float(best) if best is not None else None
        obj.calibration_margins = {
            str(k): float(v) for k, v in dict(data.get("calibration_margins", {})).items()
        }
        obj.effect_state_dict = data.get("effect_state_dict")
        obj.effect_y_mean = np.asarray(data.get("effect_y_mean"), dtype=np.float32) if data.get("effect_y_mean") is not None else None
        obj.effect_y_std = np.asarray(data.get("effect_y_std"), dtype=np.float32) if data.get("effect_y_std") is not None else None
        obj.effect_calibration_margins = {
            str(k): float(v) for k, v in (data.get("effect_calibration_margins", {}) or {}).items()
        }
        obj.effect_calibration_margins_by_quantile = {
            str(q): {str(k): float(v) for k, v in (vals or {}).items()}
            for q, vals in (data.get("effect_calibration_margins_by_quantile", {}) or {}).items()
        }
        return obj


def load_horizon_surrogate(path: str | Path):
    path = Path(path)
    if path.suffix.lower() == ".npz":
        return HorizonRidgeSurrogate.load(path)
    return TemporalGraphHorizonSurrogate.load(path)


def regression_report(y_true: pd.DataFrame, y_pred: pd.DataFrame, target_columns: list[str]) -> pd.DataFrame:
    rows = []
    for col in target_columns:
        t = pd.to_numeric(y_true[col], errors="coerce").fillna(0.0).to_numpy(float)
        p = pd.to_numeric(y_pred[f"pred_{col}"], errors="coerce").fillna(0.0).to_numpy(float)
        mae = float(np.mean(np.abs(p - t)))
        rmse = float(np.sqrt(np.mean((p - t) ** 2)))
        denom = float(np.sum((t - t.mean()) ** 2))
        r2 = float(1.0 - np.sum((p - t) ** 2) / denom) if denom > 1e-9 else float("nan")
        rows.append({"target": col, "MAE": mae, "RMSE": rmse, "R2": r2})
    return pd.DataFrame(rows)
