from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ROBUSTNESS = ROOT / "sewerrtc" / "state" / "gat_robustness.py"


def test_latency_contract_requires_warmup_repeats_p95_and_seven_frame() -> None:
    text = ROBUSTNESS.read_text(encoding="utf-8")
    for token in [
        "gat_sr0p15_latency_contract.json",
        '"warmup_runs": 5',
        '"measured_runs": 30',
        "single_sample",
        "batch_size_8",
        "seven_frame",
        "p95_ms",
        "latency_measurement_complete",
    ]:
        assert token in text


def test_latency_uses_eval_model_and_inference_mode_path() -> None:
    text = ROBUSTNESS.read_text(encoding="utf-8")
    assert "model.eval()" in text
    assert "with torch.inference_mode()" in text
    assert "checkpoint loading, and file I/O" in text
