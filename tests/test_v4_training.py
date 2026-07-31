from sewerrtc.v4.training import (
    MODEL_TASKS,
    V4MultiTaskModel,
    build_baseline_models,
    validate_training_partitions,
)


def test_training_tasks_include_process_kpi_ranking_uncertainty_and_ood() -> None:
    assert {
        "trajectory_residual",
        "kpi_delta",
        "joint_feasibility",
        "ranking",
        "aleatoric_uncertainty",
        "epistemic_uncertainty",
        "ood_abstain",
    }.issubset(MODEL_TASKS)


def test_locked_validation_cannot_be_used_for_tuning() -> None:
    audit = validate_training_partitions(
        train={"e1"}, calibration={"e2"}, locked_validation={"e3"}, tuning={"e1", "e2"}
    )
    assert audit["status"] == "pass"


def test_pilot_baselines_include_zero_majority_linear_and_tree_models() -> None:
    models = build_baseline_models()
    assert set(models) == {
        "zero_predictor",
        "majority_classifier",
        "ridge",
        "logistic_regression",
        "hist_gradient_boosting",
    }


def test_v4_multitask_model_outputs_process_kpi_ranking_uncertainty_and_ood() -> None:
    import torch

    model = V4MultiTaskModel(
        state_features=10,
        facilities=36,
        process_targets=7,
        hidden=16,
    )
    output = model(
        torch.zeros((2, 10)),
        torch.zeros((2, 12, 36)),
    )

    assert output["trajectory_residual"].shape == (2, 12, 7)
    assert output["kpi_delta"].shape == (2, 3)
    assert output["joint_logit"].shape == (2, 1)
    assert output["ranking_score"].shape == (2, 1)
    assert output["aleatoric_log_variance"].shape == (2, 3)
    assert output["ood_logit"].shape == (2, 1)
