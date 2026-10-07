from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sglib.core.infra.artifacts import atomic_json
from sglib.core.infra.hashing import sha256_file, sha256_json

from ..processing.config import CountryPipelineContext
from ..processing.features.cuz_support import load_cuz_support
from ..processing.features.grid_bundle import grid_paths, load_grid_bundle
from ..processing.pipeline import expected_feature_artifacts
from ..evidence import load_handoff_evidence
from ..reader import canonical_paths, read_artifact


def _file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.resolve().relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_inventory(context: CountryPipelineContext, *, output_path: Path | str | None = None) -> Path:
    canonical = canonical_paths(
        context.repo_root,
        context.country_code,
        configuration=context.merged,
    )
    regions = read_artifact(canonical["regions"])
    stations = read_artifact(canonical["stations"])
    station_contract = context.merged["station_contract"]
    demand_column = str(station_contract["demand_column"])
    region_demand_column = str(station_contract.get("region_demand_column", ""))
    region_column = str(station_contract["region_column"])
    prediction_total_matches_observed_total = False
    mass_alignment_basis = "not_applicable"
    if demand_column in stations and region_demand_column and region_demand_column in regions:
        assigned = stations.loc[stations[region_column].notna()] if region_column in stations else stations
        observed_total = float(pd.to_numeric(assigned[demand_column], errors="raise").sum())
        region_total = float(pd.to_numeric(regions[region_demand_column], errors="coerce").sum())
        prediction_total_matches_observed_total = bool(
            np.isclose(observed_total, region_total, rtol=1e-10, atol=1e-8)
        )
        mass_alignment_basis = (
            "assigned_station_demand_equals_region_demand"
            if prediction_total_matches_observed_total
            else "mass_mismatch"
        )
    region_records: list[dict[str, Any]] = []
    for item in context.region_items:
        name = str(item["id"])
        grid, _, metadata = load_grid_bundle(name, context.grid_root)
        support_path = context.artifact_root / f"{name}_cuz_support.npz"
        support = load_cuz_support(support_path)
        if len(grid) != len(support.features):
            raise RuntimeError(f"{context.country_code}/{name}: grid/support row mismatch")
        if np.count_nonzero(support.features[support.zero_mask]):
            raise RuntimeError(f"{context.country_code}/{name}: Z exact-zero invariant failed")
        region_records.append(
            {
                "region": name,
                "n_cells": len(grid),
                "n_sources": int(metadata["n_sources"]),
                "covered": int(support.covered_mask.sum()),
                "unknown": int(support.unknown_mask.sum()),
                "zero": int(support.zero_mask.sum()),
                "grid_metadata": metadata,
            }
        )
    schemas = context.repo_root / "casestudy" / "1_DataOverview" / "schemas"
    evidence = load_handoff_evidence(
        context.repo_root,
        context.merged,
        schemas_root=schemas,
    )
    artifacts = [
        canonical["regions"],
        canonical["stations"],
        *expected_feature_artifacts(context),
        *(item.path for item in evidence.values()),
        *(path for item in context.region_items
          for path in reversed(grid_paths(str(item["id"]), context.grid_root))),
        context.derived_root / "features_bplus" / "features_receipt.json",
    ]
    missing = [path for path in artifacts if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"inventory inputs missing: {missing}")
    directory = str(context.merged["country"]["directory"])
    output = (Path(output_path) if output_path is not None else
              context.repo_root / "results" / "1_DataOverview" / directory / "data_inventory.json")
    document = {
        "schema_version": "sg_dataoverview_inventory_v1",
        "country": context.country_code,
        "country_directory": directory,
        "evaluation_scope": context.merged["country"]["evaluation_scope"],
        "temporal_protocol": context.merged["temporal"]["protocol"],
        "capacity_basis": context.merged["station_contract"]["capacity_basis"],
        "prediction_total_matches_observed_total": prediction_total_matches_observed_total,
        "mass_alignment_basis": mass_alignment_basis,
        "n_region_rows": len(regions),
        "n_station_rows": len(stations),
        "regions": region_records,
        "handoff_artifacts": {
            name: {
                "path": item.path.relative_to(context.repo_root).as_posix(),
                "sha256": item.sha256,
                "bytes": item.bytes,
                "schema": item.schema,
                "formal_required": item.formal_required,
                "required_status": item.required_status,
                "observed_status": (
                    item.document.get("status") if item.document is not None else None
                ),
            }
            for name, item in evidence.items()
        },
        "artifacts": [_file_record(path, context.repo_root) for path in artifacts],
    }
    document["fingerprint"] = sha256_json(document["artifacts"])
    return atomic_json(document, output)
