"""UK Analysis 01: six registered core units, consuming sealed Experiment tables."""
import argparse
import os
from pathlib import Path

from sglib.core.infra.paths import find_repo_root
from sglib.experiment.downstream import country_status, load_country_tables
from sglib.analysis import stage

UPSTREAM = {'status': country_status, 'tables': load_country_tables}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('formal', 'smoke', 'preflight'), default=os.environ.get('SG_PROFILE', 'formal'))
    parser.add_argument('--results-root', default=os.environ.get('SG_RESULTS_ROOT'))
    parser.add_argument('--claim', nargs='+', choices=tuple(f'C{i}' for i in range(1, 7)))
    args = parser.parse_args()
    report = stage.run_step(find_repo_root(Path(__file__)), 'uk', args.claim or ['core'],
                            profile=args.profile, results_root=args.results_root, upstream=UPSTREAM)
    print(f'selected={len(report.selected)} ran={len(report.ran)} skipped={len(report.skipped)}')


if __name__ == '__main__':
    main()
