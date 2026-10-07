"""Regional observations for uk: numbered Experiment step 2 (plan 006a).

Run from any directory. DONE units are skipped with a reason; predecessors must
be DONE. ``--backend hpc`` only serializes the unit for the private HPC shell.
"""
import argparse
import os
from pathlib import Path

from sglib.core.infra.paths import find_repo_root
from sglib.dataoverview.handoff import load_bundle as load_dataoverview
from sglib.generator import downstream as generator_downstream
from sglib.experiment import stage

UPSTREAM = {"data": load_dataoverview, "generator": generator_downstream.load_handoff}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "preflight", "formal"), default=os.environ.get("SG_PROFILE", "formal"))
    parser.add_argument("--results-root", default=os.environ.get("SG_RESULTS_ROOT"))
    parser.add_argument("--backend", choices=("local", "hpc"), default="local")
    parser.add_argument("--region", action="append", help="restrict to one or more regions (default: all configured)")
    args = parser.parse_args()
    report = stage.run_step(find_repo_root(Path(__file__)), "uk", ['observe'], profile=args.profile,
                            results_root=args.results_root, upstream=UPSTREAM, backend=args.backend, regions=args.region)
    print(f"selected={len(report.selected)} ran={len(report.ran)} skipped={len(report.skipped)} prepared={len(report.prepared)}")


if __name__ == "__main__":
    main()
