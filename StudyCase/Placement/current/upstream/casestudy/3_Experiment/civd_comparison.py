"""CLI for the package's descriptive CIVD comparison."""
import argparse
import json
from pathlib import Path

from sglib.analysis.civd_comparison import build_comparison


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build_comparison(args.repo, args.results), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
