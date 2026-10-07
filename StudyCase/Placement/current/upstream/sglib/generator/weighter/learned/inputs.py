"""Convert torch-free Generator bundles into graph caches for learned weighters."""

from __future__ import annotations

from pathlib import Path
import pickle
from typing import Any, Mapping

import numpy as np
from shapely.geometry import Point
from sklearn.preprocessing import StandardScaler
import torch

from ...inputs import GeneratorDataBundle, GeneratorDataError, region_zscore_log1p
from .models.gnn.graph import (
    INDEXED_LANDUSE_REPRESENTATION,
    LEGACY_DENSE_LANDUSE_REPRESENTATION,
    prepare_hetero_graph_from_processed,
    preprocess_features,
)

GRAPH_CACHE_SCHEMA_V2 = "sg_generator_graph_cache_v2"
GRAPH_CACHE_SCHEMA_V3 = "sg_generator_graph_cache_v3"
# Historical callers import this name; leaving it on v2 also makes the default
# dense-cache contract explicit.
GRAPH_CACHE_SCHEMA = GRAPH_CACHE_SCHEMA_V3
GRAPH_CACHE_SCHEMAS = (GRAPH_CACHE_SCHEMA_V2, GRAPH_CACHE_SCHEMA_V3)
FEATURE_SETS = ("lu5", "fusionN", "fusionP", "fusionNP")


def build_graphs(
    bundle: GeneratorDataBundle,
    *,
    feature_set: str = "lu5",
    inject_priors: bool = True,
    representation: str = INDEXED_LANDUSE_REPRESENTATION,
) -> dict[str, Any]:
    if feature_set not in FEATURE_SETS:
        raise GeneratorDataError(f"feature_set must be one of {FEATURE_SETS}")
    feature_config = bundle.params["features"]
    mapping = feature_config["landuse_mapping"]
    agent_columns = list(feature_config["agent_columns"])
    source_columns = list(feature_config["source_columns"])
    if feature_set in {"fusionN", "fusionNP"}:
        agent_columns.append("ntl_feat")
    if feature_set in {"fusionP", "fusionNP"}:
        agent_columns.append("prox_feat")
    categories = list(mapping.values())
    graphs: dict[str, Any] = {}
    for region in bundle.regions:
        frame = bundle.grids[region][0].copy().reset_index(drop=True)
        sources = bundle.source_regions[region].copy().to_crs("EPSG:3857")
        targets = bundle.stations[region].copy().to_crs("EPSG:3857")
        if "landuse" not in frame:
            dominant = frame[list(mapping)].to_numpy(float).argmax(axis=1)
            frame["landuse"] = [categories[index] for index in dominant]
            frame.loc[~frame["covered_mask"], "landuse"] = None
        active_positions = np.flatnonzero(frame["covered_mask"].to_numpy(bool))
        agents = frame.iloc[active_positions].copy().to_crs("EPSG:3857")
        sources["geometry"] = sources.geometry.centroid
        if agents.empty or sources.empty or targets.empty:
            raise GeneratorDataError(f"{region}: graph nodes must be non-empty")
        agent_xy = np.column_stack((agents.geometry.x, agents.geometry.y))
        target_xy = np.column_stack((targets.geometry.x, targets.geometry.y))
        source_xy = np.column_stack((sources.geometry.x, sources.geometry.y))
        scaler = StandardScaler().fit(np.vstack((agent_xy, target_xy, source_xy)))
        agent_scaled = scaler.transform(agent_xy)
        source_scaled = scaler.transform(source_xy)
        agents["geometry"] = [Point(x, y) for x, y in agent_scaled]
        sources["geometry"] = [Point(x, y) for x, y in source_scaled]
        if feature_set in {"fusionN", "fusionNP"}:
            agents["ntl_feat"] = region_zscore_log1p(bundle.ntl[region][active_positions], f"{region}/ntl")
        if feature_set in {"fusionP", "fusionNP"}:
            agents["prox_feat"] = region_zscore_log1p(bundle.proximity[region][active_positions], f"{region}/proximity")
        missing_agent = [column for column in agent_columns if column not in agents]
        missing_source = [column for column in source_columns if column not in sources]
        if missing_agent or missing_source:
            raise GeneratorDataError(f"{region}: missing graph features agent={missing_agent} source={missing_source}")
        processed_agents = preprocess_features(agents[agent_columns + ["geometry"]], numerical_col_names_all=agent_columns)
        processed_sources = preprocess_features(sources[source_columns + ["geometry"]], numerical_col_names_all=source_columns)
        graph = prepare_hetero_graph_from_processed(
            sources,
            agents,
            processed_features_s=processed_sources,
            processed_features_a=processed_agents,
            relation_column=bundle.relation_column,
            agent_connectivity=None,
            representation=representation,
        )
        if inject_priors:
            graph["agent"].ntl_values = torch.tensor(bundle.ntl[region][active_positions], dtype=torch.float32)
            graph["agent"].proximity_scores = torch.tensor(bundle.proximity[region][active_positions], dtype=torch.float32)
            graph["agent"].rci_mask = torch.tensor(bundle.rci[region][active_positions], dtype=torch.bool)
        graph.full_n_agents = len(frame)
        graph.generator_country = bundle.country
        graph.generator_region = region
        graph.generator_feature_set = feature_set
        graphs[region] = graph
    return graphs


def save_graph_cache(
    path: Path | str,
    bundle: GeneratorDataBundle,
    graphs: Mapping[str, Any],
    *,
    feature_set: str,
    input_fingerprints: Mapping[str, str],
) -> Path:
    if tuple(graphs) != bundle.regions:
        raise GeneratorDataError("graph order differs from configured region order")
    destination = Path(path).resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f".{destination.name}.part")
    graph_representations = {
        getattr(graph, "landuse_supervision_representation")
        for graph in graphs.values()
        if "landuse_flat_index" in graph
    }
    if graph_representations:
        if graph_representations != {INDEXED_LANDUSE_REPRESENTATION}:
            raise GeneratorDataError(
                "indexed graph cache has inconsistent supervision representation"
            )
        if any("landuse_mapping_matrix" in graph for graph in graphs.values()):
            raise GeneratorDataError("graph cache mixes dense and indexed supervision")
        if not all("landuse_flat_index" in graph for graph in graphs.values()):
            raise GeneratorDataError("graph cache mixes indexed and legacy graphs")
        cache_schema = GRAPH_CACHE_SCHEMA_V3
    else:
        if any("landuse_flat_index" in graph for graph in graphs.values()):
            raise GeneratorDataError(
                "indexed graph is missing landuse_supervision_representation"
            )
        cache_schema = GRAPH_CACHE_SCHEMA_V2

    document = {
        "schema": cache_schema,
        "country": bundle.country,
        "feature_set": feature_set,
        "regions": bundle.regions,
        "source_column": bundle.source_column,
        "demand_column": bundle.demand_column,
        "relation_column": bundle.relation_column,
        "input_fingerprints": dict(input_fingerprints),
        "graphs": dict(graphs),
        "grids": bundle.grids,
        "ntl_dict": bundle.ntl,
        "region_dict": bundle.source_regions,
        "subs_dict": bundle.stations,
    }
    if cache_schema == GRAPH_CACHE_SCHEMA_V3:
        document["landuse_supervision_representation"] = (
            INDEXED_LANDUSE_REPRESENTATION
        )
    try:
        with partial.open("wb") as stream:
            pickle.dump(document, stream, protocol=pickle.HIGHEST_PROTOCOL)
        partial.replace(destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return destination


def load_graph_cache(
    path: Path | str,
    *,
    expected_country: str | None = None,
    expected_feature_set: str | None = None,
    required_schema: str | None = None,
) -> dict[str, Any]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with source.open("rb") as stream:
        document = pickle.load(stream)
    if not isinstance(document, dict) or document.get("schema") not in GRAPH_CACHE_SCHEMAS:
        raise GeneratorDataError("unknown graph-cache schema")
    if required_schema is not None and document.get("schema") != required_schema:
        raise GeneratorDataError(
            f"graph-cache schema {document.get('schema')!r} is not permitted; "
            f"required={required_schema!r}"
        )
    if expected_country and document.get("country") != expected_country:
        raise GeneratorDataError("graph-cache country differs from task")
    if expected_feature_set and document.get("feature_set") != expected_feature_set:
        raise GeneratorDataError("graph-cache feature set differs from task")
    regions = tuple(document.get("regions", ()))
    if not regions or tuple(document.get("graphs", {})) != regions:
        raise GeneratorDataError("graph-cache region order is invalid")
    if document["schema"] == GRAPH_CACHE_SCHEMA_V2:
        for region, graph in document["graphs"].items():
            if "landuse_flat_index" in graph:
                raise GeneratorDataError(
                    f"v2 graph cache region {region!r} contains indexed supervision"
                )
    else:
        if (
            document.get("landuse_supervision_representation")
            != INDEXED_LANDUSE_REPRESENTATION
        ):
            raise GeneratorDataError("v3 graph cache has invalid indexed representation")
        for region, graph in document["graphs"].items():
            if "landuse_flat_index" not in graph:
                raise GeneratorDataError(
                    f"v3 graph cache region {region!r} lacks landuse_flat_index"
                )
            if "landuse_mapping_matrix" in graph:
                raise GeneratorDataError(
                    f"v3 graph cache region {region!r} contains dense supervision"
                )
            flat_index = graph.landuse_flat_index
            if not torch.is_tensor(flat_index) or flat_index.dtype != torch.int64:
                raise GeneratorDataError(
                    f"v3 graph cache region {region!r} has invalid flat-index dtype"
                )
            if "landuse_ratio" not in graph:
                raise GeneratorDataError(
                    f"v3 graph cache region {region!r} lacks landuse_ratio"
                )
            if (
                getattr(graph, "landuse_supervision_representation", None)
                != INDEXED_LANDUSE_REPRESENTATION
            ):
                raise GeneratorDataError(
                    f"v3 graph cache region {region!r} has invalid representation"
                )
    return document


__all__ = [
    "FEATURE_SETS",
    "GRAPH_CACHE_SCHEMA",
    "GRAPH_CACHE_SCHEMA_V2",
    "GRAPH_CACHE_SCHEMA_V3",
    "build_graphs",
    "load_graph_cache",
    "save_graph_cache",
]
