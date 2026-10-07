"""What downstream stages consume from the Generator without importing it.

The orchestration entrypoints of later stages inject these callables (the same
pattern as ``input_loader`` for the Generator's own inputs unit), so
``sglib.experiment`` never imports ``sglib.generator``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


from sglib.core.chain.stage import reject_archived_root
from sglib.core.infra.hashing import sha256_file
from sglib.core.infra.terms import load_country_profile

from .config import load_generator_config
from .generation import derive_run_identity
from .handoff import GeneratorBundle, load_bundle


@dataclass(frozen=True)
class GeneratorHandoff:
    bundle: GeneratorBundle
    root: Path
    config: Mapping[str, Any]
    inputs_receipt_sha256: str
    run_fingerprint: str


def load_handoff(repo_root: Path | str, country: str, *, results_root: Path | str | None = None) -> GeneratorHandoff:
    """Typed Generator handoff for one country from the formal (or given) results root."""

    from sglib.core.infra.setup_snapshot import config_root as setup_config_root

    repo = Path(repo_root).resolve()
    profile = load_country_profile(repo / "casestudy/config/countries" / f"{country}.toml")
    results = Path(results_root).resolve() if results_root is not None else repo / "results"
    reject_archived_root(results, repo)
    root = results / "2_Generator" / profile.directory
    # A root that recorded its setup is read with the configuration and authorities it recorded.
    authority_root = setup_config_root(root) or repo
    config = load_generator_config(authority_root, authority_root / "casestudy/2_Generator/general/generator.toml",
                                   authority_root / "casestudy/config/countries" / f"{country}.toml",
                                   authority_root / "casestudy/2_Generator" / profile.directory / f"{country}.toml").values
    bundle = load_bundle(root, config, schemas_root=repo / "casestudy/2_Generator/schemas", repo_root=authority_root)
    identity = derive_run_identity(root, config, repo_root=authority_root)
    return GeneratorHandoff(bundle, root, config, sha256_file(root / "inputs/receipt.json"), identity["run_fingerprint"])
