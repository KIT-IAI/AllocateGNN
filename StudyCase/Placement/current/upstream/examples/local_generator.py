"""Run synthetic or existing-data local smoke; outputs default to a new temp root."""
from __future__ import annotations

import argparse
from pathlib import Path
import tempfile

from sglib.core.infra.paths import find_repo_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country", choices=("synthetic", "nl", "nz"), default="synthetic")
    parser.add_argument("--output-root", type=Path)
    args = parser.parse_args(argv)
    repository = find_repo_root(__file__)
    output = args.output_root or Path(tempfile.mkdtemp(prefix=f"sg-{args.country}-example-"))
    if args.country == "synthetic":
        from sglib.examples.synthetic import run
    elif args.country == "nl":
        from sglib.examples.nl import run
    else:
        from sglib.examples.nz import run_smoke as run
    print(run(repository, output))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
