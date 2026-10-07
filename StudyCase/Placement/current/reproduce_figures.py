"""Regenerate the three non-map manuscript figures from published frozen CSVs.

This command redraws the evaluation design, scale curves and conditional-bound
curves. It does not rerun allocation models or planning experiments. Map figures
need external GIS inputs and are outside this lightweight entry point.

    python StudyCase/Placement/current/reproduce_figures.py --output outputs/figures
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


HERE = Path(__file__).resolve().parent
REPOSITORY = HERE.parents[2]
DEFAULT_ROOT = REPOSITORY / "study_materials" / "placement" / "current" / "frozen"
INPUTS = (
    "config.json",
    "ledger/claim_ledger.csv",
    "stats/scale_curves.csv",
    "uk_adequacy.csv",
    "au_adequacy.csv",
)
FIGURES = ("fig0_pipeline", "fig4_scale", "fig5_adequacy")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=DEFAULT_ROOT,
        help="Frozen CSV directory (default: bundled study_materials/placement/current/frozen)",
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Output directory for PDF, PNG, captions and reproduction provenance",
    )
    args = parser.parse_args()
    root, output = args.root.resolve(), args.output.resolve()
    if output == root or root in output.parents:
        parser.error("--output must be outside the frozen input directory")
    missing = [name for name in INPUTS if not (root / name).is_file()]
    if missing:
        parser.error(f"Missing frozen inputs in {root}: {', '.join(missing)}")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    run_id = config["run_id"]
    before = {name: sha256(root / name) for name in INPUTS}
    acronym_mapping = HERE / "acronyms.json"
    acronym_sha256 = sha256(acronym_mapping)
    scripts = HERE / "scripts"
    output.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["MPLBACKEND"] = "Agg"
    for stem in FIGURES:
        command = [
            sys.executable, str(scripts / f"make_{stem}.py"),
            "--run-id", run_id, "--root", str(root), "--out", str(output),
        ]
        subprocess.run(command, cwd=REPOSITORY, env=environment, check=True)
    after = {name: sha256(root / name) for name in INPUTS}
    if before != after:
        raise RuntimeError("Frozen inputs changed while figures were generated")
    if sha256(acronym_mapping) != acronym_sha256:
        raise RuntimeError("Acronym mapping changed while figures were generated")
    expected = [
        name
        for stem in FIGURES
        for name in (f"{stem}.pdf", f"{stem}.png", f"captions/{stem}.tex", f"captions/{stem}.md")
    ] + ["fig5_sources.json"]
    for name in expected:
        if not (output / name).is_file() or (output / name).stat().st_size == 0:
            raise RuntimeError(f"Expected nonempty figure output missing: {output / name}")
    provenance = {
        "run_id": run_id,
        "scope": "Redraw three non-map figures from frozen results, without rerunning experiments",
        "frozen_root": str(root),
        "input_sha256": before,
        "acronym_sha256": acronym_sha256,
        "script_sha256": {
            name: sha256(scripts / name)
            for name in [*(f"make_{stem}.py" for stem in FIGURES), "paper2_style.py"]
        },
        "output_sha256": {name: sha256(output / name) for name in expected},
    }
    (output / "figure_reproduction.json").write_text(
        json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Generated {len(FIGURES)} non-map figures and provenance in {output}")


if __name__ == "__main__":
    main()
