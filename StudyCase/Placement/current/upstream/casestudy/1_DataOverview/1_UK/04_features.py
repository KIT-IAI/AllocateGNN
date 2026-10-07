"""UK 04 · NTL 落地与特征提取（ntl download + features）。

Two units, in DAG order:

1. ``uk.ntl.download`` — queries the VIIRS composite from Google Earth Engine,
   clipped to the bounds of the canonical regions (``02_regions_stations.ipynb``
   must be DONE). Needs ``GEE_SERVICE_ACCOUNT_KEY_PATH``; BLOCKED without it
   unless the raster is already present.
2. ``uk.features.derive`` — fetches the OSM land-use and GHSL Built-S caches
   for every analysis region, extracts land-use proportions, Built-S fraction,
   the C/U/Z support partition, and NTL onto the B+ grid (``03_grid.ipynb`` must
   be DONE), and writes ``data/datasets/2_derived/uk/features_bplus/extracted``
   plus the receipt.

This is the long-running step of the chain, so it is a script, not a notebook.

    python casestudy/1_DataOverview/1_UK/04_features.py
    python casestudy/1_DataOverview/1_UK/04_features.py --refresh        # rebuild feature arrays
    python casestudy/1_DataOverview/1_UK/04_features.py --refresh-ntl    # re-fetch NTL from GEE

Next step: ``05_features_overview.ipynb``.
"""

from __future__ import annotations

import argparse
import sys

from sglib.core.infra.paths import find_repo_root
from sglib.dataoverview import stage

COUNTRY = "uk"


def _load_dotenv(repo_root) -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:  # pragma: no cover - optional convenience only
        return
    load_dotenv(repo_root / ".env")


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--refresh", action="store_true", help="rebuild all feature arrays")
    parser.add_argument("--refresh-ntl", action="store_true", help="re-fetch the NTL composite from GEE")
    args = parser.parse_args(argv)
    root = find_repo_root(__file__)
    _load_dotenv(root)
    ntl = stage.run_step(root, COUNTRY, ["ntl"], refresh=args.refresh_ntl)
    features = stage.run_step(root, COUNTRY, ["features"], refresh=args.refresh)
    ran = len(ntl.ran) + len(features.ran)
    skipped = len(ntl.skipped) + len(features.skipped)
    print(f"selected={len(ntl.selected) + len(features.selected)} ran={ran} skipped={skipped}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
