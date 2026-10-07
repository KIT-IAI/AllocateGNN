"""NZ 01 · 数据落地（download）。

Runs the ``nz.<dataset>.download`` units that have no predecessor in the DAG:
Commerce Commission, Stats NZ and Census objects land under `data/datasets/1_raw/nz` (see also `acquire_fresh_sources.py` for the fresh-source manifest). Download units that depend on a derived product are reported here
and run later in the chain; the VIIRS NTL composite needs the canonical
regions for its bounds, so it is fetched by ``04_features.py`` (GEE credential
``GEE_SERVICE_ACCOUNT_KEY_PATH`` required). Units whose outputs already exist
are skipped with the reason printed; ``--refresh`` forces a re-download.

    python casestudy/1_DataOverview/5_NZ/01_download.py
    python casestudy/1_DataOverview/5_NZ/01_download.py --dataset <dataset_id> --refresh

Next step: ``02_regions_stations.ipynb``.
"""

from __future__ import annotations

import argparse
import sys

from sglib.core.infra.paths import find_repo_root
from sglib.dataoverview import stage

COUNTRY = "nz"


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
    parser.add_argument("--dataset", action="append", help="restrict to one dataset id (repeatable)")
    parser.add_argument("--refresh", action="store_true", help="re-download even if outputs exist")
    args = parser.parse_args(argv)
    root = find_repo_root(__file__)
    _load_dotenv(root)
    units, _ = stage.build_units(root)
    downloads = stage.select_units(units, COUNTRY, ["download"])
    if args.dataset:
        members = list(args.dataset)
    else:
        members = [unit_id for unit_id in downloads if not units[unit_id].depends_on]
        for unit_id in downloads:
            if units[unit_id].depends_on:
                print(
                    f"NOTE {unit_id} depends on {', '.join(units[unit_id].depends_on)}; "
                    "it runs in 04_features.py"
                )
    report = stage.run_step(root, COUNTRY, members, refresh=args.refresh)
    print(f"selected={len(report.selected)} ran={len(report.ran)} skipped={len(report.skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
