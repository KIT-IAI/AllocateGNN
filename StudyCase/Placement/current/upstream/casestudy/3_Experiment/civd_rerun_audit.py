"""Wire the independent package audit to the case's frozen-input readers."""
import argparse
import json
from pathlib import Path

from sglib.analysis.civd_downstream import scientific_invariants
from sglib.dataoverview.handoff import load_bundle
from sglib.experiment.config import load_experiment_config
from sglib.experiment.civd_observations import observation_code
from sglib.generator.civd_extension import extend_country
from sglib.report.civd_audit import audit_release as produce_audit
from civd_rerun_downstream import check_links


def country_inputs(repo, country):
    return load_bundle(repo, country), load_experiment_config(repo, country)


def audit_release(repo, results, *, correction_id="refactor_20260922_r2"):
    return produce_audit(repo, results, country_inputs=country_inputs,
        observation_code=observation_code(extend_country),
        scientific_invariants=scientific_invariants, verify_links=check_links, correction_id=correction_id)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit_release(args.repo, args.results), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
