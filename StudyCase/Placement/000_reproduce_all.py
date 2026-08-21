"""Reproduce and verify every numerical result in the placement manuscript."""
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from SpatialPlacement.reproduce_paper_numbers import main


if __name__ == "__main__":
    raise SystemExit(main())
