"""Run the local NZ structural smoke in an isolated output directory."""

from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

from sglib.core.infra.paths import find_repo_root
from sglib.examples.nz import run_smoke


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args(argv)
    repo_root = find_repo_root(__file__)
    output_root = args.output_root or Path(tempfile.mkdtemp(prefix="sg-nz-smoke-"))
    print(run_smoke(repo_root, output_root, refresh=args.refresh))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
