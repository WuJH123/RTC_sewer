from __future__ import annotations

from pathlib import Path


RESULT_FILES = (
    "final_event_metrics.csv",
    "paired_strategy_metrics.csv",
    "statistical_tests.csv",
    "bootstrap_intervals.csv",
    "failure_case_catalog.csv",
    "ablation_results.csv",
)


def paper_artifact_paths(root: str | Path) -> dict[str, Path]:
    base = Path(root)
    return {
        "results": base / "results",
        "figures": base / "figures",
        "tables": base / "tables",
        "reproducibility": base / "reproducibility",
    }
