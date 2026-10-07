"""Reproduce the current margin-criterion paper from its frozen CSV exports.

The historical task-scale-lu5 package remains available explicitly through
``python -m SpatialPlacement.reproduce_legacy_paper_numbers --verify``.
"""
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from SpatialPlacement.current_paper import (  # noqa: F401
    DEFAULT_OUTPUT,
    RELEASE_ROOT,
    main,
    reproduce,
    reproduce_adequacy,
    reproduce_controls,
    reproduce_scale,
    reproduce_tasks,
    verify_headlines,
    verify_manifest,
    verify_source,
)


if __name__ == "__main__":
    raise SystemExit(main())
