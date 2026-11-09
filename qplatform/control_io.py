"""Durable small control records with bounded Windows sharing-conflict retries."""
from __future__ import annotations

import errno
import json
import os
import time
import uuid
from pathlib import Path
from typing import Any


def windows_access_conflict(exc: OSError) -> bool:
    return getattr(exc, "winerror", None) in {5, 32, 33} or (
        os.name == "nt" and isinstance(exc, PermissionError) and exc.errno == errno.EACCES
    )


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        for delay in (.01, .02, .04, .08, .1, None):
            try:
                os.replace(temporary, path)
                break
            except OSError as exc:
                if delay is None or not windows_access_conflict(exc):
                    raise
                time.sleep(delay)
    finally:
        temporary.unlink(missing_ok=True)
