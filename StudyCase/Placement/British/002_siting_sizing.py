"""Show British p-median siting and both sizing-matching protocols."""
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from SpatialPlacement.reproduce_paper_numbers import reproduce_tasks


if __name__ == "__main__":
    tasks = reproduce_tasks()["GB"]
    result = {
        "siting": tasks["siting"],
        "sizing_many_to_one": tasks["sizing"],
        "sizing_one_to_one": tasks["sizing_1to1"],
    }
    print(json.dumps(result, indent=2, sort_keys=True))
