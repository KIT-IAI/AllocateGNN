"""AU Generator: train."""
import argparse
import os
from sglib.core.infra.paths import find_repo_root
from sglib.generator import stage

COUNTRY = "au"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("smoke", "preflight", "formal"), default=os.environ.get("SG_PROFILE", "formal"))
    parser.add_argument("--results-root", default=os.environ.get("SG_RESULTS_ROOT"))
    parser.add_argument("--backend", choices=("local", "hpc"), default="local")
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    stage.run_step(find_repo_root(__file__), COUNTRY, ['train'], profile=args.profile,
        results_root=args.results_root, backend=args.backend, refresh=args.refresh)


if __name__ == "__main__":
    main()
