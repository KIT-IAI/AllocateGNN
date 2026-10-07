"""Case configuration and upstream wiring for the package CIVD extension."""
from functools import partial
from pathlib import Path

from sglib.dataoverview.handoff import load_bundle
from sglib.experiment import stage
from sglib.generator.civd_extension import extend_country
from sglib.generator.downstream import load_handoff


def load_country(repo, country, results):
    repo, results = Path(repo).resolve(), Path(results).resolve()
    ctx = stage.country_context(repo, country, profile="smoke", results_root=results,
        upstream={"data": load_bundle, "generator": partial(load_handoff, results_root=repo / "results")})
    data = stage.load_upstream(ctx, "data")
    generator = extend_country(repo, country, results, data=data,
        generator=stage.load_upstream(ctx, "generator"), regions=ctx.loaded.values["regions"])
    ctx = stage.with_upstream(ctx, data=data, generator=generator)
    return ctx, data, generator
