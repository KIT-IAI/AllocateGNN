"""Report 01: seven mechanical units consuming sealed Analysis tables."""
import argparse
import os
from pathlib import Path

from sglib.core.infra.paths import find_repo_root
from sglib.analysis.downstream import status, load
from sglib.report import stage
from sglib.report.config import load_report_config

UPSTREAM = {'status': status, 'tables': load}


def main():
    repo = find_repo_root(Path(__file__))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--profile', choices=('formal', 'smoke', 'preflight'), default=os.environ.get('SG_PROFILE', 'formal'))
    parser.add_argument('--results-root', default=os.environ.get('SG_RESULTS_ROOT'))
    parser.add_argument('--claim', nargs='+', choices=tuple(load_report_config(repo).registrations))
    args = parser.parse_args()
    report = stage.run_step(repo, 'cross', args.claim or ['render'], profile=args.profile,
                            results_root=args.results_root, upstream=UPSTREAM)
    print(f'selected={len(report.selected)} ran={len(report.ran)} skipped={len(report.skipped)}')


if __name__ == '__main__':
    main()
