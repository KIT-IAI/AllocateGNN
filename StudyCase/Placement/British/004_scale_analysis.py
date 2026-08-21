"""Show the British station-centred geographic-support sweep."""
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from SpatialPlacement.reproduce_paper_numbers import reproduce_scale


if __name__ == "__main__":
    print(json.dumps(reproduce_scale()["GB"], indent=2, sort_keys=True))
