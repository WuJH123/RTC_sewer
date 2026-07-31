from __future__ import annotations

import torch
from torch import nn


MODEL_TASKS = {
    "graph_state_encoding",
    "action_sequence_encoding",
    "trajectory_residual",
    "kpi_delta",
    "joint_feasibility",
    "ranking",
    "aleatoric_uncertainty",
    "epistemic_uncertainty",
    "ood_abstain",
}

# Train1600 planning roles live in the torch-free ``training_plan`` module;
# this module only hosts learning code that genuinely needs torch.


class V4MultiTaskModel(nn.Module):
    """Compact Final-V4 multi-task head over frozen graph-state features."""

    def __init__(
        self,
        *,
        state_features: int,
        facilities: int = 36,
        process_targets: int = 7,
        hidden: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(int(state_features), int(hidden)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )
        self.action_encoder = nn.GRU(
            input_size=int(facilities),
            hidden_size=int(hidden),
            batch_first=True,
        )
        self.shared = nn.Sequential(
            nn.Linear(int(hidden) * 2, int(hidden)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
        )
        self.trajectory_head = nn.Linear(
            int(hidden), 12 * int(process_targets)
        )
        self.kpi_head = nn.Linear(int(hidden), 3)
        self.joint_head = nn.Linear(int(hidden), 1)
        self.ranking_head = nn.Linear(int(hidden), 1)
        self.aleatoric_head = nn.Linear(int(hidden), 3)
        self.ood_head = nn.Linear(int(hidden), 1)
        self.process_targets = int(process_targets)

    def forward(self, state, action_sequence):
        state_embedding = self.state_encoder(state)
        _, action_hidden = self.action_encoder(action_sequence)
        shared = self.shared(
            torch.cat([state_embedding, action_hidden[-1]], dim=-1)
        )
        trajectory = self.trajectory_head(shared).reshape(
            -1, 12, self.process_targets
        )
        return {
            "trajectory_residual": trajectory,
            "kpi_delta": self.kpi_head(shared),
            "joint_logit": self.joint_head(shared),
            "ranking_score": self.ranking_head(shared),
            "aleatoric_log_variance": self.aleatoric_head(shared),
            "ood_logit": self.ood_head(shared),
        }


def validate_training_partitions(
    *,
    train: set[str],
    calibration: set[str],
    locked_validation: set[str],
    tuning: set[str],
) -> dict:
    disjoint = (
        not train & calibration
        and not train & locked_validation
        and not calibration & locked_validation
    )
    tuning_safe = not (tuning & locked_validation)
    return {
        "status": "pass" if disjoint and tuning_safe else "blocked",
        "checks": {
            "event_splits_disjoint": bool(disjoint),
            "locked_validation_not_tuned": bool(tuning_safe),
        },
    }


def build_baseline_models() -> dict:
    from sklearn.dummy import DummyClassifier, DummyRegressor
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression, Ridge

    return {
        "zero_predictor": DummyRegressor(strategy="constant", constant=0.0),
        "majority_classifier": DummyClassifier(strategy="most_frequent"),
        "ridge": Ridge(alpha=1.0),
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced"
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=100
        ),
    }
