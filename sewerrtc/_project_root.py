"""Project root resolution utility."""
import os
from pathlib import Path


def get_project_root() -> Path:
    """Get project root from environment variable or auto-detect.

    Checks PROJECT6_ROOT env var first, then falls back to
    navigating up from this file to find the project root.
    """
    env_root = os.environ.get("PROJECT6_ROOT")
    if env_root:
        return Path(env_root)
    # Auto-detect: sewerrtc/ is one level below project root
    return Path(__file__).parent.parent.resolve()


PROJECT_ROOT = get_project_root()
