from __future__ import annotations

from __future__ import annotations

import runpy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).with_name("06_train_surrogate.py")), run_name="__main__")
