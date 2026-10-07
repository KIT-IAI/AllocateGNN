from __future__ import annotations

from pathlib import Path

from ...config import CountryPipelineContext

from . import au, au_ledger


def run(context: CountryPipelineContext, *, force: bool = False) -> list[Path]:
    outputs = list(au.derive_au(context, force=force))
    outputs.extend(au_ledger.build_ledger(context, force=force))
    return outputs
