from __future__ import annotations

from ...config import CountryPipelineContext

from pathlib import Path

from . import au_ledger


def run(context: CountryPipelineContext, *, force: bool = False) -> list[Path]:
    return list(au_ledger.build_ledger(context, force=force))
