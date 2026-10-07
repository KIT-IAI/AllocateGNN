"""Atomic writers used by every evolving stage."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np


def _replace_bytes(path: Path | str, payload: bytes) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.", suffix=".part", dir=target.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def atomic_bytes(path: Path | str, payload: bytes) -> Path:
    return _replace_bytes(path, payload)


def atomic_text(path: Path | str, payload: str, *, encoding: str = "utf-8") -> Path:
    return _replace_bytes(path, payload.encode(encoding))


def atomic_json(document: Any, path: Path | str, *, exclusive: bool = False) -> Path:
    """Write UTF-8 JSON, optionally refusing existing or interrupted output.

    Exclusive writers claim the deterministic partial path with ``O_EXCL``.
    Publishing via a hard link also refuses a target created by another writer;
    no check-then-replace window can overwrite an existing receipt. A partial
    left by an interrupted process is evidence and is never removed by a caller
    that did not create it.

    Ordinary writes retain the core writer's LF bytes. Exclusive writes retain
    the learned workers' historical ``Path.write_text`` native newline format.
    """
    payload = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if not exclusive:
        return atomic_text(path, payload)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.part")
    # Opening is deliberately outside cleanup: failure means this partial
    # belongs to an earlier process or a concurrent writer.
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline=None) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)
    return target


def atomic_csv(table: Any, path: Path | str) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.part")
    try:
        table.to_csv(temporary, index=False)
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target


def atomic_npz(path: Path | str, **arrays: np.ndarray) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.part")
    try:
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, target)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return target
