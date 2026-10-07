"""Shared execution for the CLI and numbered Generator notebooks."""

from __future__ import annotations

from functools import partial
from sglib.core.chain import stage as chain_stage
from sglib.core.chain.stage import StageOrderError, StepReport, resolve_results_root

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable

from sglib.core.infra.content_chain import verify_chain
from sglib.core.infra.hashing import sha256_json
from sglib.core.infra.terms import load_country_profile
from . import execution
from .config import load_generator_config, scientific_config_fingerprint, training_params
from .registry import build_registry, topological_order, unit_output_path, unit_status
from .weighter.candidates import load_candidate_registry
from .weighter.learned.training.matrix import load_task_matrix




@dataclass(frozen=True)
class StageContext:
    repo: Path
    results: Path
    root: Path
    country: str
    profile: str
    backend: str
    loaded: object
    candidates: dict
    units: dict
    read_only: bool
    input_loader: object = None




def country_config(repo_root: Path | str, country: str, *, config_root: Path | str | None = None):
    """Load a country's Generator configuration from the checkout or from another configuration tree.

    ``config_root`` (a closed root's ``setup/config``) holds the repository layout
    of every file the Generator reads, authorities included.
    """

    repo = Path(config_root if config_root is not None else repo_root).resolve()
    profile = load_country_profile(repo / "casestudy/config/countries" / f"{country}.toml")
    return load_generator_config(repo, repo / "casestudy/2_Generator/general/generator.toml",
        profile.source_path, repo / "casestudy/2_Generator" / profile.directory / f"{country}.toml")


def setup_config_files(loaded, repo_root: Path | str) -> list[str]:
    """Repository-relative configuration files a Generator country actually reads."""

    from sglib.core.infra.engineering_admission import (
        INDEXED_EQUIVALENCE_EVIDENCE_PATH, INDEXED_EQUIVALENCE_IMMUTABILITY_MARKER,
    )

    repo = Path(repo_root).resolve()
    files = {Path(path).resolve() for path in loaded.sources.values()}
    authorities = loaded.values["authorities"]
    files.update(repo / str(authorities[name]) for name in
                 ("candidate_registry", "training_task_matrix", "idr_contract", "engineering_admission"))
    files.add(repo / str(loaded.values["checks"]["admission_contract"]))
    files.update((repo / INDEXED_EQUIVALENCE_EVIDENCE_PATH, repo / INDEXED_EQUIVALENCE_IMMUTABILITY_MARKER))
    return sorted(path.relative_to(repo).as_posix() for path in files if path.is_relative_to(repo))


def code_checks(ctx) -> list[dict]:
    """Compare every content-chain receipt of the root with the current Generator code projection."""

    from sglib.core.infra.setup_snapshot import receipt_code_checks
    from .chain import numerical_code

    postprocess = numerical_code()
    return receipt_code_checks(ctx.root, lambda node_id, receipt: None if ".civd_extension." in node_id else postprocess)




def _closed_results(results: Path) -> bool:
    paths = sorted((results / "2_Generator/_closures").glob("*"))
    for path in paths:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            schema = value.get("schema_version")
            if schema == "sg_content_chain_closure_v1":
                verify_chain(value)
            elif schema == "sg_generator_hpc_training_closure_v2":
                if value.get("status") != "PASS" or sha256_json(value["scientific_view"]) != value["scientific_fingerprint"]:
                    raise ValueError("invalid training closure")
            elif schema == "sg_generator_hpc_inference_closure_v2" or (
                schema is None and set(value) == {"status", "countries", "chain_closure"}
            ):
                if value.get("status") != "PASS":
                    raise ValueError("invalid closure status")
                verify_chain(value["chain_closure"])
            else:
                raise ValueError(f"unrecognized closure schema: {schema}")
        except (OSError, ValueError, KeyError, AttributeError, TypeError) as exc:
            raise StageOrderError(f"unrecognized or invalid closure: {path}") from exc
    return bool(paths)


def _reject_stale_inputs(root: Path, loaded, repo: Path) -> None:
    """Refuse a writable root whose input bundle was built under another scientific configuration.

    Every later unit reads ``inputs/bundle.pkl``; a receipt from an older
    configuration (v1 schema, or a v2/v3 receipt with a different config
    fingerprint) would otherwise be reported DONE and silently feed stale
    authorities into training, materialization and audit.  Closed roots are
    never checked here: they are judged by their own recorded setup, not by
    the evolving configuration in Git.
    """

    receipt_path = root / "inputs/receipt.json"
    if not receipt_path.is_file():
        return
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, ValueError):
        return  # unit_status reports the unreadable marker as INVALID
    current = scientific_config_fingerprint(loaded, repo_root=repo)
    schema = receipt.get("schema_version")
    recorded = receipt.get("config_fingerprint")
    if schema in {"sg_generator_inputs_receipt_v2", "sg_generator_inputs_receipt_v3"} and recorded == current:
        return
    raise StageOrderError(
        f"results root {root} holds an input bundle from another scientific configuration "
        f"(receipt {schema}, config fingerprint {str(recorded)[:12]} vs current {current[:12]}). "
        "Its downstream artifacts are not comparable with the current authorities: "
        "use a fresh --results-root / SG_RESULTS_ROOT, or delete this root before rerunning 01."
    )


def country_context(repo_root, country, *, profile="formal", results_root=None, backend="local", input_loader=None) -> StageContext:
    if backend not in {"local", "hpc"}:
        raise ValueError(f"unknown backend: {backend}")
    from sglib.core.infra.setup_snapshot import config_root as setup_config_root

    repo = Path(repo_root).resolve()
    loaded = country_config(repo, country)
    results = resolve_results_root(repo, profile, results_root)
    root = Path(str(loaded.values["paths"][f"{profile}_root"]).format(
        results_root=results.as_posix(), directory=loaded.country_profile.directory)).resolve()
    if root != results / "2_Generator" / loaded.country_profile.directory:
        raise ValueError("country output root must be inside the selected results root")
    read_only = _closed_results(results)
    authority_root = repo
    if read_only and setup_config_root(root) is not None:
        # A closed root is judged by the configuration it recorded, not by the evolving checkout.
        authority_root = setup_config_root(root)
        loaded = country_config(repo, country, config_root=authority_root)
    candidates = load_candidate_registry(authority_root / loaded.values["authorities"]["candidate_registry"])
    matrix = load_task_matrix(authority_root / loaded.values["authorities"]["training_task_matrix"])
    families = {item.task_group: item.family for item in matrix if item.country == country}
    units = build_registry(country, loaded.values, candidates, families)
    if not read_only:
        _reject_stale_inputs(root, loaded, repo)
    return StageContext(repo, results, root, country, profile, backend, loaded, candidates, units, read_only, input_loader)


select_units = partial(chain_stage.select_units, label='Generator', sort=True)


def skip_reason(unit, root: Path) -> str:
    marker = unit_output_path(unit, root)
    return f"marker contract is DONE: {marker}"


def require_done(repo_root, country, members, **kwargs) -> list[str]:
    ctx = country_context(repo_root, country, **kwargs)
    selected = select_units(ctx.units, members)
    missing = [f"{key} ({unit_status(ctx.units[key], ctx.root)})" for key in selected
               if unit_status(ctx.units[key], ctx.root) != "DONE"]
    if missing:
        raise StageOrderError("required units are not DONE: " + ", ".join(missing))
    return selected


def figures_root(ctx: StageContext) -> Path:
    """Views mirror the results root under results/_views (formal: results/_views/2_Generator/<dir>)."""

    base = ctx.repo / "results"
    try:
        relative = ctx.results.relative_to(base)
    except ValueError:
        relative = Path(ctx.results.name)
    path = base / "_views" / relative / "2_Generator" / ctx.loaded.country_profile.directory
    path.mkdir(parents=True, exist_ok=True)
    return path


def _execute(ctx: StageContext, unit) -> bool:
    root = ctx.root
    if unit.step == "inputs":
        if ctx.input_loader is None:
            raise ValueError("the orchestration entrypoint must provide the DataOverview handoff loader")
        worker = training_params(ctx.loaded)
        if ctx.profile == "smoke":
            worker["regions"] = worker["regions"][:4]
            for spec in worker["config_map"].values():
                spec["epochs"] = int(ctx.loaded.values["training"]["smoke_epochs"])
        execution.prepare_inputs(ctx.input_loader(ctx.repo, ctx.country), ctx.loaded.values, root,
            selected_regions=worker["regions"] if ctx.profile == "smoke" else None,
            worker_params=worker, require_formal_evidence=ctx.profile != "smoke")
    elif unit.step in {"static", "public_activity"}:
        execution.generate_static_component(root, unit.member)
    elif unit.step == "train":
        execution.prepare_training_group(root, repo_root=ctx.repo, results_root=ctx.results,
            group=unit.member, execution_backend=ctx.backend)
        if ctx.backend == "hpc":
            return False
        # Full authority coordinates are retained; smoke changes epochs and
        # regions only, so materialization and sweeps can finish without fixtures.
        execution.run_training_group_local(root, unit.member, profile=ctx.profile)
    elif unit.step == "verify":
        execution.verify_training_group(root, unit.member, profile=ctx.profile)
    elif unit.step == "infer":
        execution.prepare_inference_group(root, repo_root=ctx.repo, results_root=ctx.results,
            group=unit.member, execution_backend=ctx.backend)
        if ctx.backend == "hpc":
            return False
        execution.run_inference_group_local(root, unit.member)
        execution.verify_inference_group(root, unit.member)
    elif unit.step == "materialize":
        execution.materialize_family(root, unit.member, ctx.candidates)
    elif unit.step == "finalize":
        execution.finalize_candidate_indexes(root)
    elif unit.step == "sweeps":
        execution.run_sweep(root, unit.member)
    elif unit.step == "audit":
        execution.run_audit(root, profile=ctx.profile, full=True)
    else:
        {"civd": execution.run_civd, "idr_fixed": execution.run_idr_fixed,
         "idr_matched": execution.run_idr_matched}[unit.step](root)
    return True


def run_units(ctx, selected, *, expand=False, refresh=False):
    selected = tuple(selected)
    return chain_stage.run_units(ctx, selected, _hooks(ctx),
        ordered=topological_order(ctx.units, set(selected)), expand=expand, refresh=refresh)


def run_step(repo_root, country, members, *, profile="formal", results_root=None, backend="local", refresh=False, input_loader=None) -> StepReport:
    ctx = country_context(repo_root, country, profile=profile, results_root=results_root, backend=backend, input_loader=input_loader)
    return run_units(ctx, select_units(ctx.units, members), refresh=refresh)


def _hooks(ctx):
    return chain_stage.RunHooks(
        status=lambda unit: unit_status(unit, ctx.root),
        execute=lambda unit: _execute(ctx, unit),
        skip_reason=lambda unit: skip_reason(unit, ctx.root))
