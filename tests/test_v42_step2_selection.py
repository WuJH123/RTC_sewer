from scripts.train_v42_step2_formal_f2 import _selection_key


def test_control_selection_prefers_control_direction_over_total_loss() -> None:
    epoch1 = {
        "loss": 10.0,
        "pfv_delta_sign_accuracy": 0.02,
        "tfv_delta_sign_accuracy": 0.90,
        "pfv_delta_mae": 100.0,
        "tfv_delta_mae": 10.0,
    }
    epoch2 = {
        "loss": 11.0,
        "pfv_delta_sign_accuracy": 0.98,
        "tfv_delta_sign_accuracy": 0.90,
        "pfv_delta_mae": 120.0,
        "tfv_delta_mae": 10.0,
    }
    assert _selection_key(epoch2, "control") < _selection_key(epoch1, "control")


def test_default_selection_remains_total_validation_loss() -> None:
    low_loss = {
        "loss": 10.0,
        "pfv_delta_sign_accuracy": 0.02,
        "tfv_delta_sign_accuracy": 0.90,
        "pfv_delta_mae": 100.0,
        "tfv_delta_mae": 10.0,
    }
    high_loss = dict(low_loss, loss=11.0, pfv_delta_sign_accuracy=0.98)
    assert _selection_key(low_loss, "loss") < _selection_key(high_loss, "loss")
