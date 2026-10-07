"""Shared Generator context and content-chain projections."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from sglib.core.infra.artifacts import atomic_json
from sglib.core.infra.content_chain import code_projection
from sglib.core.infra.hashing import sha256_file, sha256_json
from .config import load_generator_config
from .delivery import checked_path, read_json
from .registry import build_registry, unit_output_path

COUNTRIES = ("uk", "au", "nl", "nz")
STEPS = frozenset({"static", "public_activity", "materialize", "finalize", "sweeps", "civd", "idr_fixed", "idr_matched", "audit"})


def context(repo: Path, results: Path, country: str):
    from sglib.core.infra.terms import load_country_profile

    profile = load_country_profile(repo / "casestudy/config/countries" / f"{country}.toml")
    config = load_generator_config(repo, repo / "casestudy/2_Generator/general/generator.toml",
                                   profile.source_path, repo / "casestudy/2_Generator" / profile.directory / f"{country}.toml").values
    candidates = read_json(repo / config["authorities"]["candidate_registry"])
    return config, candidates, build_registry(country, config, candidates), results / "2_Generator" / profile.directory


def immutable(document: dict, path: Path) -> Path:
    if path.exists():
        if read_json(path) != document:
            raise ValueError(f"禁止改写既有登记或 receipt：{path}")
        return path
    return atomic_json(document, path)


def scientific_json(value):
    """索引中的定位与运行观测不参与输出身份。"""
    excluded = {"path", "bytes", "sha256", "candidate_path", "runtime", "observations",
                "execution_backend", "repo_root", "results_root", "started_at", "completed_at"}
    if isinstance(value, dict):
        return {k: scientific_json(v) for k, v in value.items() if k not in excluded}
    if isinstance(value, list):
        return [scientific_json(v) for v in value]
    return value


def output_view(root: Path, unit):
    """从单元索引派生逻辑输出及物理定位，既不扫描其他族也不读调度 registry。"""
    marker = unit_output_path(unit, root)
    docs = [("index", marker)]
    if unit.step == "finalize":
        docs.append(("qa_index", root / "candidates/candidate_qa_index.json"))
    outputs, records = {}, []
    for role, path in docs:
        document = read_json(path)
        outputs[role] = sha256_json(scientific_json(document))
        records.append({"role": role, "path": path.relative_to(root).as_posix(), "kind": "json_projection"})
        for entry in document.get("entries", document.get("regions", [])):
            if not isinstance(entry, dict) or "path" not in entry:
                continue
            artifact = checked_path(root, entry)
            identity = {k: entry[k] for k in ("label", "candidate", "region", "seed", "fold", "signal", "parameter", "value") if k in entry}
            name = role + ":" + json.dumps(identity, sort_keys=True, ensure_ascii=False)
            if name in outputs:
                raise ValueError(f"重复逻辑输出：{unit.id}/{name}")
            outputs[name] = sha256_file(artifact)
            records.append({"role": name, "path": artifact.relative_to(root).as_posix(), "kind": "file"})
    return outputs, records


def numerical_code():
    from .materialize import core, idr, civd
    from .allocator.idr import gate
    from .allocator.idr import idr as idr_math
    from .weighter import correction
    from .weighter.native import GPMWeighter, UniformWeighter, materialize_equal_grid
    symbols = {"GPM": GPMWeighter, "Uniform": UniformWeighter, "EqualGrid": materialize_equal_grid}
    for module in (core, idr, civd, gate, idr_math, correction):
        for name, value in inspect.getmembers(module, inspect.isfunction):
            if value.__module__.startswith("sglib.generator"):
                symbols[f"{module.__name__}.{name}"] = value
    return code_projection(symbols)["code_sha256"]


