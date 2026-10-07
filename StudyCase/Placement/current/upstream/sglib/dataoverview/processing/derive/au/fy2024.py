from __future__ import annotations

from ...config import CountryPipelineContext
from .au_fy2024 import publish_au_fy2024_demand
from .regions import canonical_station_table, sync_region_tables


def run(context: CountryPipelineContext) -> dict:
    result = publish_au_fy2024_demand(context=context)
    sync_region_tables(context)
    canonical_station_table(context)
    return result
