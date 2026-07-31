from __future__ import annotations

import hashlib
import re
from pathlib import Path


def canonical_event_id(event_id: str) -> str:
    return event_id.strip()


def storm_family_id(event_id: str) -> str:
    m = re.match(r"^(T[^_]+)_D\d+_(.+)$", event_id)
    if not m:
        return event_id
    return f"{m.group(1)}_{m.group(2)}"


def file_sha256(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

