"""Recompute current margin-criterion statistics and claims from frozen CSVs.

Use --verify to check the public subset manifest and frozen statistical tables.
This entry point does not retrain models or regenerate the spatial inputs.
"""
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from SpatialPlacement.reproduce_paper_numbers import main


if __name__ == "__main__":
    raise SystemExit(main())
