"""Reproduce the four-country CIVD correction from frozen fields and assignments.

Commands: prepare, observe --country uk, analyze, validate, snapshot, seal. Each country can run in
its own process. Unrelated observations are inherited with their original
receipt, while allocator/T1 tables are recomputed and independently checked.
The formal historical results are never overwritten.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


from sglib.core.infra.artifacts import atomic_json
from sglib.core.infra.paths import portable_path
from sglib.core.infra.content_chain import (
    derive_chain_closure, verify_chain,
)
from sglib.core.infra.hashing import sha256_file
from sglib.core.infra.paths import find_repo_root
from sglib.experiment import stage
from sglib.experiment.civd_observations import audit_country, checked_receipt, read_json


COUNTRIES = {"uk": "1_UK", "au": "2_AU", "nl": "4_NL", "nz": "5_NZ"}
CORRECTION = "refactor_20260922_r2"


def prepare(repo, results):
    if results.exists():
        raise ValueError(f"correction root must be new: {results}")
    results.mkdir(parents=True)
    for name in ("3_Experiment", "4_Analysis", "5_Report"):
        source = repo / "results" / name
        for path in source.rglob("*"):
            if not path.is_file():
                continue
            parts = path.relative_to(source).parts
            if "_closures" in parts or "setup" in parts:
                continue
            if name == "3_Experiment":
                if len(parts) == 2 and parts[1] in {"audit.json", "manifest.json"}:
                    continue
                if len(parts) >= 4 and parts[1] == "observations" and parts[3] in {"allocator", "T1", "receipt.json"}:
                    continue
            elif name == "4_Analysis":
                if len(parts) >= 2 and parts[1] in {"C3", "support", "synthesis", "audit.json", "manifest.json"}:
                    continue
            elif parts[0] in {"C2", "C3", "C4", "C5", "C6", "SYN", "audit.json", "index.md"}:
                continue
            target = results / name / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        print(f"PASS inherited {name}", flush=True)
    atomic_json({"correction_id": CORRECTION, "countries": list(COUNTRIES),
        "scope": "structural_refactor_regeneration",
        "scientific_scope": "four_country_posthoc_bugfix_comparison",
        "production_event": "recompute_the_already_corrected_formal_implementation",
        "historical_results": portable_path(repo / "results", repo),
        "recomputed": ["allocator", "T1", "Analysis.C3", "Analysis.support", "Analysis.synthesis", "Report"],
        "inherited": "unrelated observations, planning, bounds, defenses and C1/C2/C4/C5/C6 core analyses",
        "core_contrasts": "retain original five C3 contrasts; new CIVD comparisons are explicitly post hoc",
        "created_at": datetime.now(timezone.utc).isoformat()}, results / "correction_scope.json")


def observe_country(repo, results, country):
    from civd_rerun_inputs import load_country
    from sglib.experiment.civd_observations import observe_country as produce
    from sglib.generator.civd_extension import extend_country
    ctx, _, _ = load_country(repo, country, results)
    return produce(ctx, input_loader=extend_country, correction_id=CORRECTION)

def seal(repo, results):
    from civd_rerun_audit import audit_release

    from sglib.core.infra.setup_snapshot import git_state, read_setup, verify_setup
    state = git_state(repo)
    if state["commit"] is None or state["dirty"]:
        raise ValueError("formal sealing requires a committed, clean working tree")
    required_setups = [results / name / directory
        for name in ("2_Generator", "3_Experiment", "4_Analysis") for directory in COUNTRIES.values()]
    required_setups += [results / "4_Analysis/9_CrossCountry", results / "5_Report"]
    setup_roots = set(required_setups) | {path.parent.parent for path in results.rglob("setup/setup.json")}
    setup_hashes = {}
    for root in sorted(setup_roots):
        record = read_setup(root)
        if record is None or verify_setup(root)["status"] != "PASS":
            raise ValueError(f"missing or invalid setup before sealing: {root}")
        if record["profile"] != "formal" or record["git"]["dirty"] or record["git"]["commit"] != state["commit"]:
            raise ValueError(f"setup must record the current clean formal commit: {root}")
        path = root / "setup/setup.json"
        setup_hashes[path.relative_to(results).as_posix()] = sha256_file(path)

    from sglib.core.infra.leaf_manifest import verify
    for name in ("2_Generator", "3_Experiment", "4_Analysis"):
        for path in (results / name).glob("*/manifest.json"):
            manifest = read_json(path)
            leaves = {key: [record["path"] for record in leaf["artifacts"]]
                      for key, leaf in manifest["leaves"].items()}
            actual = {p.relative_to(path.parent).as_posix() for p in path.parent.rglob("*")
                      if p.is_file() and p != path}
            if actual != {value for paths in leaves.values() for value in paths}:
                raise ValueError(f"manifest file inventory changed before sealing: {path}")
            if verify(manifest, leaves, path.parent)["status"] != "PASS":
                raise ValueError(f"manifest content changed before sealing: {path}")

    supporting = {}
    for relative in ("inherited_model_verification.json", "refactor_validation.json"):
        if read_json(results / relative)["status"] != "PASS":
            raise ValueError(f"required refactor validation did not pass: {relative}")
        supporting[relative] = sha256_file(results / relative)
    for path in sorted((results / "refactor_evidence").rglob("*")):
        if path.is_file():
            supporting[path.relative_to(results).as_posix()] = sha256_file(path)

    if read_json(results / "5_Report/audit.json")["status"] != "PASS":
        raise ValueError("cannot seal an incomplete correction")
    if not (results / "civd_release_audit.receipt.json").exists():
        audit_release(repo, results)
    audit_receipt = verify_chain(read_json(results / "civd_release_audit.receipt.json"))
    for relative, digest in audit_receipt["outputs"].items():
        path = (results / relative).resolve()
        if not path.is_relative_to(results) or sha256_file(path) != digest:
            raise ValueError(f"completed release audit output changed: {relative}")
    if read_json(results / "civd_release_audit.json")["status"] != "PASS":
        raise ValueError("completed release audit did not pass")
    comparison = checked_receipt(results / "civd_comparison")
    for key, relative in (("downstream_validation", "civd_downstream_validation.json"),
                          ("correction_scope", "correction_scope.json"), ("report_audit", "5_Report/audit.json")):
        if audit_receipt["commitment"]["inputs"][key] != sha256_file(results / relative):
            raise ValueError(f"completed release audit input changed: {key}")
    if audit_receipt["commitment"]["inputs"]["comparison"] != comparison["receipt_sha256"]:
        raise ValueError("completed release audit references a different comparison")
    for country, directory in COUNTRIES.items():
        for path in (results / "3_Experiment" / directory / "observations").glob("*/receipt.json"):
            receipt = checked_receipt(path.parent)
            if audit_receipt["commitment"]["inputs"][f"observation:{country}:{path.parent.name}"] != receipt["receipt_sha256"]:
                raise ValueError(f"completed release audit observation changed: {path}")
    generator_receipts = [verify_chain(read_json(path))
        for directory in COUNTRIES.values()
        for path in (results / "2_Generator" / directory / "civd").glob("*receipt.json")]
    generator_closure = derive_chain_closure(generator_receipts)
    if generator_closure["closure_sha256"] != read_json(results / "civd_release_audit.json")["generator_closure_sha256"]:
        raise ValueError("Generator receipts changed since the independent audit")
    atomic_json(generator_closure, results / "2_Generator/_closures" / f"{CORRECTION}.json")
    for name in ("3_Experiment", "4_Analysis", "5_Report"):
        receipts = [checked_receipt(p.parent) for p in (results / name).rglob("receipt.json")]
        closure = derive_chain_closure(receipts)
        atomic_json(closure, results / name / "_closures" / f"{CORRECTION}.json")
        print(f"PASS sealed {name}: {len(receipts)} receipts", flush=True)
    pointer = {"correction_id": CORRECTION, "scope": "structural_refactor_regeneration",
        "code_commit": state["commit"], "setup_snapshots": setup_hashes, "validation_artifacts": supporting,
        "results_root": portable_path(results, repo), "comparison": portable_path(results / "civd_comparison", repo),
        "report_audit_sha256": sha256_file(results / "5_Report/audit.json"),
        "release_audit_receipt_sha256": audit_receipt["receipt_sha256"],
        "comparison_receipt_sha256": comparison["receipt_sha256"],
        "generator_closure_sha256": sha256_file(results / "2_Generator/_closures" / f"{CORRECTION}.json"),
        "production_event": "new_package_implementation_of_the_already_corrected_CIVD_results",
        "inherited_artifacts": "original trained models and predictions retain their original production identities",
        "historical_release": "preserved_without_rewriting_receipts_or_closures",
        "unaffected": "historical preregistered five C3 contrasts retained and verified"}
    atomic_json(pointer, results / "correction_release.json")


def main():
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "observe", "audit", "analyze", "validate", "snapshot", "seal"))
    parser.add_argument("--country", choices=COUNTRIES)
    parser.add_argument("--results-root", default=f"results/_releases/{CORRECTION}")
    args = parser.parse_args()
    repo = find_repo_root(Path(__file__))
    results = (repo / args.results_root).resolve()
    if not results.is_relative_to((repo / "results").resolve()) or results == repo / "results":
        raise ValueError("correction root must be a separate child of results")
    if args.command == "prepare":
        prepare(repo, results)
    elif args.command == "observe":
        if args.country is None:
            parser.error("observe requires --country")
        observe_country(repo, results, args.country)
    elif args.command == "audit":
        for country in ([args.country] if args.country else COUNTRIES):
            ctx = stage.country_context(repo, country, profile="smoke", results_root=results)
            audit_country(ctx)
    elif args.command == "analyze":
        from civd_rerun_downstream import run_downstream
        from civd_comparison import build_comparison
        run_downstream(repo, results)
        build_comparison(repo, results)
    elif args.command == "validate":
        from civd_rerun_downstream import run_downstream
        from civd_comparison import build_comparison
        from civd_rerun_audit import audit_release
        run_downstream(repo, results, verify_only=True)
        if not (results / "civd_comparison/receipt.json").is_file():
            build_comparison(repo, results)
        audit_release(repo, results)
    elif args.command == "snapshot":
        from civd_snapshot import snapshot
        snapshot(repo, results)
    else:
        seal(repo, results)


if __name__ == "__main__":
    main()
