"""Record current setup and extend existing leaf manifests after local validation."""
import importlib
from pathlib import Path

from sglib.core.infra.artifacts import atomic_json
from sglib.core.infra.content_chain import code_projection
from sglib.core.infra.hashing import sha256_file
from sglib.core.infra.leaf_manifest import build, read_current, verify
from sglib.core.infra.setup_snapshot import (
    git_state, receipt_code_checks, verify_setup, write_setup,
)
from sglib.experiment import manifest as experiment_manifest, stage as experiment_stage
from sglib.experiment.civd_observations import observation_code, read_json
from sglib.generator import stage as generator_stage
from sglib.generator.civd_extension import extend_country
from sglib.analysis import manifest as analysis_manifest, stage as analysis_stage
from sglib.report import stage as report_stage


COUNTRIES = {"uk": "1_UK", "au": "2_AU", "nl": "4_NL", "nz": "5_NZ"}


def reproject(document):
    """Re-read explicit locator symbols; their original role labels stay explicit."""
    symbols = {}
    for role, locator in document["locators"].items():
        value = importlib.import_module(locator["module"])
        for attribute in locator["qualname"].split("."):
            value = getattr(value, attribute)
        symbols[role] = value
    return code_projection(symbols)["code_sha256"]


def snapshot(repo, results):
    repo, results = Path(repo).resolve(), Path(results).resolve()
    state = git_state(repo)
    if not state["commit"] or state["dirty"]:
        raise ValueError("formal snapshots require the current committed clean worktree")
    if list(results.glob("*/_closures/*.json")):
        raise ValueError("cannot snapshot an already closed result root")
    if read_json(results / "civd_release_audit.json")["status"] != "PASS":
        raise ValueError("independent local validation must pass before setup snapshots")
    inherited = read_json(results / "inherited_model_verification.json")
    if inherited["status"] != "PASS":
        raise ValueError("inherited model and inference verification must pass")

    notes = {
        "release_kind": "structural_refactor_regeneration",
        "inherited_results_root": "results",
        "inherited_identity": "models, predictions and unchanged downstream products retain their original receipts",
        "local_identity": "regenerated CIVD, observations, analysis and report use current code projections",
        "model_verification_sha256": sha256_file(results / "inherited_model_verification.json"),
        "validation_receipt_file_sha256": sha256_file(results / "civd_release_audit.receipt.json"),
        "compute_scope": "local CPU only; no HPC training or inference submitted",
    }
    numerical_observation = observation_code(extend_country)
    roots = []
    release_checks = []
    for country, directory in COUNTRIES.items():
        root = results / "2_Generator" / directory
        region = next(p for p in sorted((root / "civd").glob("*.receipt.json")))
        numerical = reproject(read_json(region)["observations"]["code_projection"])
        loaded = generator_stage.country_config(repo, country)
        files = generator_stage.setup_config_files(loaded, repo)
        checks = receipt_code_checks(root, lambda *args: numerical)
        roots.append((root, "2_Generator", country, files, checks, None))

        ctx = experiment_stage.country_context(repo, country, profile="smoke", results_root=results)
        checks = experiment_stage.code_checks(ctx)
        for row in checks:
            record = read_json(root.parent.parent / "3_Experiment" / directory / row["receipt"])
            if record.get("observations", {}).get("correction_id"):
                row["current_code_sha256"] = numerical_observation
                row["match"] = numerical_observation == row["receipt_code_sha256"]
        roots.append((ctx.root, "3_Experiment", country,
                      experiment_stage.setup_config_files(ctx), checks,
                      lambda root=ctx.root, regions=ctx.loaded.values["regions"]: experiment_manifest.enumerate_leaves(root, regions)))

    for country in (*COUNTRIES, "cross"):
        ctx = analysis_stage.country_context(repo, country, profile="smoke", results_root=results)
        roots.append((ctx.root, "4_Analysis", country, analysis_stage.setup_config_files(ctx),
                      analysis_stage.code_checks(ctx),
                      lambda root=ctx.root, country=country: analysis_manifest.enumerate_leaves(root, country)))
    ctx = report_stage.country_context(repo, profile="smoke", results_root=results)
    roots.append((ctx.root, "5_Report", "cross", report_stage.setup_config_files(ctx), report_stage.code_checks(ctx), None))

    for root, stage, country, configs, checks, enumerate_leaves in roots:
        release_checks.extend({**row, "receipt": root.relative_to(results).as_posix() + "/" + row["receipt"]}
                              for row in checks)
        write_setup(root, repo_root=repo, stage=stage, country=country, profile="formal",
                    config_files=configs, allow_dirty=False, code_checks=checks, notes=notes)
        if verify_setup(root)["status"] != "PASS":
            raise ValueError(f"new setup did not verify: {root}")
        if enumerate_leaves is not None:
            leaves, unclaimed = enumerate_leaves()
            baseline = read_json(root / "manifest.json")
            previous = {key: value for key, value in leaves.items() if key != "setup"}
            if unclaimed or verify(baseline, previous, root)["status"] != "PASS":
                raise ValueError(f"existing leaves changed before setup: {root}")
            atomic_json(build(stage, country, read_current(leaves, root)), root / "manifest.json")
            if verify(read_json(root / "manifest.json"), leaves, root)["status"] != "PASS":
                raise ValueError(f"setup leaf did not verify: {root}")
        elif stage == "2_Generator":
            leaves = {name: sorted(path.relative_to(root).as_posix() for path in (root / name).rglob("*")
                                   if path.is_file()) for name in ("civd", "setup")}
            manifest = build(stage, country, read_current(leaves, root))
            atomic_json(manifest, root / "manifest.json")
            if verify(manifest, leaves, root)["status"] != "PASS":
                raise ValueError(f"CIVD extension leaves did not verify: {root}")
        print(f"PASS formal setup {stage}/{country}", flush=True)

    scripts = {path.name: path for path in Path(__file__).parent.glob("civd*.py")}
    scripts["rerun_civd.py"] = Path(__file__).with_name("rerun_civd.py")
    for relative in ("civd_comparison/receipt.json", "civd_release_audit.receipt.json"):
        receipt = read_json(results / relative)
        current = reproject(receipt["observations"]["code_projection"])
        recorded = receipt["commitment"]["code_sha256"]
        if current != recorded:
            raise ValueError(f"new release receipt uses stale code: {relative}")
        release_checks.append({"node_id": receipt["node_id"], "receipt": relative,
                               "receipt_code_sha256": recorded, "current_code_sha256": current, "match": True})
    write_setup(results, repo_root=repo, stage="release", country="cross", profile="formal",
                config_files=["README.md", "docs/contracts.md", "requirements.txt", "pyproject.toml", "pytest.ini"],
                allow_dirty=False, orchestration=scripts, code_checks=release_checks, notes=notes)
