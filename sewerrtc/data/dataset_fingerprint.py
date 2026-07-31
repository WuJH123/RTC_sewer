from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable


def source_file_fingerprint(paths: Iterable[str | Path]) -> str:
    """Return a deterministic content fingerprint for an ordered source set."""
    digest = hashlib.sha256()
    for raw_path in paths:
        path = Path(raw_path).resolve()
        digest.update(str(path).encode("utf-8"))
        digest.update(b"\0")
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return digest.hexdigest()
