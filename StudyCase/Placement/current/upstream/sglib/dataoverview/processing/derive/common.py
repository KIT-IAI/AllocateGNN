from __future__ import annotations

import os
from pathlib import Path


def atomic_geofile(frame, path: Path, *, layer: str | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.stem}.part{path.suffix}")
    partial.unlink(missing_ok=True)
    kwargs = {"driver": "GPKG", "index": False, "layer": layer or path.stem}
    try:
        frame.to_file(partial, **kwargs)
        os.replace(partial, path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return path
