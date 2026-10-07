"""Land the formal NZ official-source bundle outside Slurm compute nodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sglib.dataoverview.processing.derive.nz.fresh_sources import (  # noqa: E402
    FreshSourcePaths,
    acquire_nz_fresh_sources,
    verify_fresh_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download Commerce Commission and Stats NZ inputs with byte and "
            "ArcGIS pagination identity receipts."
        )
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="redownload every official object instead of reusing a verified landing",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify the existing fresh landing without network access",
    )
    args = parser.parse_args(argv)
    if args.verify_only:
        paths = FreshSourcePaths.from_repo(REPO_ROOT)
        document = verify_fresh_manifest(paths)
    else:
        paths = acquire_nz_fresh_sources(REPO_ROOT, refresh=args.refresh)
        document = verify_fresh_manifest(paths)
    print(json.dumps(document, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
