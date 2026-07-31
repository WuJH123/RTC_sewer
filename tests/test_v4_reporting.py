from pathlib import Path

from sewerrtc.v4.reporting import paper_artifact_paths


def test_reporting_paths_cover_results_figures_tables_and_reproducibility(
    tmp_path: Path,
) -> None:
    paths = paper_artifact_paths(tmp_path)

    assert {"results", "figures", "tables", "reproducibility"} == set(paths)
    assert all(path.is_relative_to(tmp_path) for path in paths.values())
