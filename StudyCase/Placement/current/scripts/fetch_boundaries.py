"""Download the official boundary layers used only for Figures 1-3.

UK: ONS Open Geography Portal, ITL2 January 2021 full-resolution clipped (BFC V3),
matching the January 2021 LAD/ITL lookup used by the upstream DataOverview.
AU: ABS ASGS Edition 3 SA4 2021 (GDA2020) shapefile.

Files are written once under ``results/_backup/inputs/boundaries`` with a
provenance record (URL, retrieval time, bytes, SHA-256, licence). An existing
file is never overwritten; rerunning only re-verifies the recorded hash.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import urllib.parse
import urllib.request

ONS_SERVICE = ("https://services1.arcgis.com/ESMARspQHYMw9BZ9/arcgis/rest/services/"
               "ITL2_JAN_2021_UK_BFC_V3/FeatureServer/0")
ABS_SA4 = ("https://www.abs.gov.au/statistics/standards/australian-statistical-geography-standard-asgs/"
           "edition-3-july-2021-june-2026/access-and-downloads/digital-boundary-files/SA4_2021_AUST_SHP_GDA2020.zip")

SOURCES = {
    "uk_itl2_jan2021_bfc_v3.geojson": {
        "publisher": "Office for National Statistics, Open Geography Portal",
        "dataset": "International Territorial Level 2 (January 2021) Boundaries UK BFC V3",
        "service": ONS_SERVICE,
        "version": "ITL2_JAN_2021_UK_BFC_V3",
        "licence": "Open Government Licence v3.0",
        "attribution": ("Source: Office for National Statistics licensed under the Open Government Licence v.3.0. "
                        "Contains OS data (c) Crown copyright and database right 2021."),
    },
    "au_sa4_2021_gda2020_shp.zip": {
        "publisher": "Australian Bureau of Statistics",
        "dataset": "ASGS Edition 3 Statistical Area Level 4 (SA4) 2021, GDA2020 shapefile",
        "url": ABS_SA4,
        "version": "ASGS Edition 3 (July 2021 - June 2026), SA4_2021",
        "licence": "Creative Commons Attribution 4.0 International",
        "attribution": "Source: Australian Bureau of Statistics, ASGS Edition 3, CC BY 4.0.",
    },
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def fetch(url: str) -> bytes:
    request = urllib.request.Request(url, headers={"User-Agent": "margin-criterion-revision/1.0"})
    with urllib.request.urlopen(request, timeout=300) as response:
        return response.read()


def ons_geojson() -> tuple[bytes, list[str]]:
    """Page through the feature service one feature at a time (full-resolution polygons)."""
    ids = json.loads(fetch(f"{ONS_SERVICE}/query?where=1%3D1&returnIdsOnly=true&f=json"))["objectIds"]
    features, urls = [], []
    for oid in sorted(ids):
        query = urllib.parse.urlencode({"objectIds": oid, "outFields": "*", "outSR": 4326, "f": "geojson"})
        url = f"{ONS_SERVICE}/query?{query}"
        page = json.loads(fetch(url))
        if len(page["features"]) != 1:
            raise RuntimeError(f"ONS object {oid}: expected one feature")
        features.extend(page["features"])
        urls.append(url)
    document = {"type": "FeatureCollection", "features": features}
    return json.dumps(document, separators=(",", ":"), sort_keys=True).encode("utf-8"), urls


def main() -> None:
    here = Path(__file__).resolve()
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backup", type=Path, default=here.parents[2] / "results" / "_backup")
    args = ap.parse_args()
    target = args.backup / "inputs" / "boundaries"
    target.mkdir(parents=True, exist_ok=True)
    record_path = target / "provenance.json"
    records = json.loads(record_path.read_text(encoding="utf-8")) if record_path.exists() else {}
    for name, meta in SOURCES.items():
        path = target / name
        if path.exists():
            if name not in records or records[name]["sha256"] != sha256(path):
                raise RuntimeError(f"{path} exists without a matching provenance record")
            print(f"verified {name}")
            continue
        if name.startswith("uk_"):
            payload, urls = ons_geojson()
            extra = {"request_urls": urls, "feature_count": len(json.loads(payload)["features"])}
        else:
            payload = fetch(meta["url"])
            extra = {}
        path.write_bytes(payload)
        records[name] = {**meta, **extra, "retrieved_utc": datetime.now(timezone.utc).isoformat(),
                         "bytes": path.stat().st_size, "sha256": sha256(path)}
        print(f"downloaded {name}: {path.stat().st_size} bytes")
    record_path.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
