from __future__ import annotations

import json
from pathlib import Path

import requests

from sglib.core.infra.artifacts import atomic_text


def run(query: dict, target_root: Path, *, refresh: bool = False) -> list[Path]:
    outputs: list[Path] = []
    for layer in query["layers"]:
        target = target_root / "geoserver" / f"{str(layer).replace(':', '_')}.geojson"
        if target.is_file() and target.stat().st_size > 0 and not refresh:
            outputs.append(target)
            continue
        url = (
            f"{str(query['endpoint']).rstrip('/')}/wfs?service=WFS&version=2.0.0"
            f"&request=GetFeature&typeNames={layer}&outputFormat=application/json"
        )
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        document = response.json()
        if "features" not in document:
            raise RuntimeError(f"GeoServer response contains no features: {layer}")
        atomic_text(target, json.dumps(document, ensure_ascii=False))
        outputs.append(target)
    return outputs
