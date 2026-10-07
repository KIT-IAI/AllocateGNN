"""OGC API Features adapter with explicit pagination completeness."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urljoin

import requests

from sglib.core.infra.artifacts import atomic_json


def run(query: dict, target_root: Path, *, refresh: bool = False) -> list[Path]:
    target = target_root / str(query["filename"])
    audit = target.with_suffix(target.suffix + ".pagination.json")
    if target.is_file() and audit.is_file() and not refresh:
        try:
            import json

            recorded = json.loads(audit.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            recorded = {}
        if (
            recorded.get("schema_version") == "sg_ogc_pagination_audit_v1"
            and recorded.get("complete") is True
            and int(recorded.get("total", -1)) >= 0
        ):
            return [target, audit]
    url = str(query["url"])
    features: list[dict] = []
    pages: list[dict] = []
    visited: set[str] = set()
    expected_total: int | None = None
    while url:
        if url in visited:
            raise RuntimeError(f"OGC API pagination cycle detected at {url}")
        visited.add(url)
        response = requests.get(url, timeout=120)
        response.raise_for_status()
        document = response.json()
        page_features = document.get("features")
        if not isinstance(page_features, list):
            raise RuntimeError("OGC API response has no features array")
        matched = document.get("numberMatched")
        if matched not in (None, "unknown"):
            current_expected = int(matched)
            if expected_total is not None and current_expected != expected_total:
                raise RuntimeError("OGC API numberMatched changed across pages")
            expected_total = current_expected
        returned = document.get("numberReturned")
        if returned is not None and int(returned) != len(page_features):
            raise RuntimeError("OGC API numberReturned differs from features length")
        features.extend(page_features)
        pages.append(
            {
                "url": url,
                "count": len(page_features),
                "cumulative_count": len(features),
            }
        )
        next_links = [link for link in document.get("links", []) if link.get("rel") == "next"]
        if len(next_links) > 1:
            raise RuntimeError("OGC API page exposes multiple next links")
        url = urljoin(url, str(next_links[0]["href"])) if next_links else ""
    if expected_total is not None and len(features) != expected_total:
        raise RuntimeError(
            f"OGC API pagination incomplete: fetched={len(features)}, "
            f"numberMatched={expected_total}"
        )
    feature_ids = [str(item["id"]) for item in features if "id" in item]
    if len(feature_ids) == len(features) and len(feature_ids) != len(set(feature_ids)):
        raise RuntimeError("OGC API pagination returned duplicate feature ids")
    collection = {"type": "FeatureCollection", "features": features}
    atomic_json(collection, target)
    atomic_json(
        {
            "schema_version": "sg_ogc_pagination_audit_v1",
            "complete": True,
            "expected_total": expected_total,
            "feature_ids_complete": len(feature_ids) == len(features),
            "pages": pages,
            "total": len(features),
        },
        audit,
    )
    return [target, audit]
