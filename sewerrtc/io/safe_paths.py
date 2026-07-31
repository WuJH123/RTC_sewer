from __future__ import annotations

"""Windows-safe output path and single-writer helpers for long SWMM runs.

The preferred mitigation is to keep run tags and leaf names short.  The Win32
``\\\\?\\`` prefix is only applied for Python file-system operations and is not
passed to SWMM/PySWMM because some native libraries do not accept it.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Iterator

DEFAULT_PATH_BUDGET = 235
_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def stable_token(value: str, length: int = 10) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()[: max(4, int(length))]


def short_run_tag(value: str, *, max_length: int = 36) -> str:
    clean = _SAFE.sub("_", str(value or "run")).strip("._-") or "run"
    if len(clean) <= max_length:
        return clean
    suffix = stable_token(clean, 10)
    head = clean[: max(1, max_length - len(suffix) - 1)].rstrip("._-")
    return f"{head}_{suffix}"


def compact_component(value: str, *, max_length: int = 52) -> str:
    clean = _SAFE.sub("_", str(value or "item")).strip("._-") or "item"
    if len(clean) <= max_length:
        return clean
    return f"{clean[: max_length - 11]}_{stable_token(clean, 10)}"


def path_length(path: str | Path) -> int:
    return len(str(Path(path).absolute()))


def path_budget_check(path: str | Path, *, budget: int = DEFAULT_PATH_BUDGET) -> dict[str, object]:
    absolute = str(Path(path).absolute())
    return {
        "path": absolute,
        "length": len(absolute),
        "budget": int(budget),
        "within_budget": len(absolute) <= int(budget),
        "remaining": int(budget) - len(absolute),
    }


def ensure_within_budget(path: str | Path, *, budget: int = DEFAULT_PATH_BUDGET) -> Path:
    p = Path(path)
    audit = path_budget_check(p, budget=budget)
    if not audit["within_budget"]:
        raise OSError(
            f"output path exceeds configured Windows path budget: "
            f"length={audit['length']} budget={audit['budget']} path={audit['path']}"
        )
    return p


def extended_path(path: str | Path) -> str:
    """Return a Windows extended path for Python I/O only."""
    absolute = str(Path(path).absolute())
    if os.name != "nt" or absolute.startswith("\\\\?\\"):
        return absolute
    if absolute.startswith("\\\\"):
        return "\\\\?\\UNC\\" + absolute.lstrip("\\")
    return "\\\\?\\" + absolute


def mkdir_parent(path: str | Path, *, budget: int = DEFAULT_PATH_BUDGET) -> Path:
    p = ensure_within_budget(path, budget=budget)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def atomic_write_text(path: str | Path, text: str, *, encoding: str = "utf-8", budget: int = DEFAULT_PATH_BUDGET) -> Path:
    p = mkdir_parent(path, budget=budget)
    tmp = p.with_name(compact_component(p.name + ".tmp", max_length=min(60, len(p.name) + 12)))
    tmp.write_text(text, encoding=encoding)
    os.replace(tmp, p)
    return p


def atomic_write_json(path: str | Path, payload: object, *, budget: int = DEFAULT_PATH_BUDGET) -> Path:
    return atomic_write_text(path, json.dumps(payload, indent=2, ensure_ascii=False), budget=budget)


@dataclass(frozen=True)
class OutputLease:
    lock_path: Path
    owner: str


@contextmanager
def single_writer_lease(
    output_root: str | Path,
    *,
    owner: str,
    stale_after_sec: float = 24 * 3600,
) -> Iterator[OutputLease]:
    """Prevent two orchestration processes from writing the same output root."""
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".writer.lock"
    now = time.time()
    if lock.exists():
        try:
            payload = json.loads(lock.read_text(encoding="utf-8"))
            created = float(payload.get("created_epoch", 0.0))
        except Exception:
            created = 0.0
            payload = {}
        if now - created <= float(stale_after_sec):
            raise RuntimeError(
                "output root already has an active writer lease: "
                f"{lock}; owner={payload.get('owner', 'unknown')}"
            )
        lock.unlink(missing_ok=True)
    token = {"owner": str(owner), "pid": os.getpid(), "created_epoch": now}
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    fd = os.open(str(lock), flags)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as stream:
            json.dump(token, stream, ensure_ascii=False, indent=2)
        yield OutputLease(lock_path=lock, owner=str(owner))
    finally:
        try:
            current = json.loads(lock.read_text(encoding="utf-8")) if lock.exists() else {}
            if int(current.get("pid", -1)) == os.getpid():
                lock.unlink(missing_ok=True)
        except Exception:
            pass
