"""Show the Australian PF=1 connection-requirement comparison."""
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from SpatialPlacement.reproduce_paper_numbers import reproduce_tasks


if __name__ == "__main__":
    result = reproduce_tasks()["AU"]["connection"]
    print(json.dumps(result, indent=2, sort_keys=True))
