from __future__ import annotations

import json
from pathlib import Path

import requests

from sglib.core.infra.artifacts import atomic_text


def run(query: dict, target_root: Path, *, refresh: bool = False) -> list[Path]:
    target = target_root / str(query["filename"])
    if target.is_file() and target.stat().st_size > 0 and not refresh:
        return [target]
    response = requests.get(str(query["url"]), timeout=120)
    response.raise_for_status()
    document = response.json()
    if not document.get("features"):
        raise RuntimeError("ArcGIS response contains no features")
    atomic_text(target, json.dumps(document, ensure_ascii=False))
    return [target]
