from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CLONE = ROOT / "sewerrtc" / "state" / "state_clone_equivalence.py"


def test_state_clone_outputs_require_swmm_and_controller_memory() -> None:
    text = CLONE.read_text(encoding="utf-8")
    assert "requires_real_swmm_hotstart_equivalence_run" in text
    assert "SWMM hot-start and controller memory" in text
    assert "completion_marker" in text

