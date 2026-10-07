from __future__ import annotations

from pathlib import Path
from typing import Any

from .tasks import InferenceTaskError

import geopandas as gpd
import numpy as np

from ...support import compose_support_field
from ..models.gnn.inference import assert_inference_graph, sanitize_for_inference
from ..models.gnn.training import predict_gnn_scores
from ..models.mlp import predict_mlp_scores


class InferenceFieldError(ValueError):
    pass


def _frame(grid: gpd.GeoDataFrame, source_column: str) -> gpd.GeoDataFrame:
    result = grid.copy().reset_index(drop=True)
    required = {source_column, "covered_mask", "unknown_mask", "zero_mask", "built_fraction"}
    missing = required - set(result)
    if missing:
        raise InferenceFieldError(f"grid lacks {sorted(missing)}")
    return result


def infer_demand_field(
    family: str,
    model: Any,
    graph: Any,
    grid: gpd.GeoDataFrame,
    source_regions: gpd.GeoDataFrame,
    *,
    source_column: str,
    demand_column: str,
    device: str = "cpu",
) -> np.ndarray:
    if device != "cpu":
        raise InferenceFieldError("formal inference is CPU-only")
    frame = _frame(grid, source_column)
    sanitized = sanitize_for_inference(graph)
    assert_inference_graph(sanitized)
    if family == "gnn":
        scores = predict_gnn_scores(model, sanitized, len(frame))
    elif family == "mlp":
        if not isinstance(model, tuple) or len(model) != 2:
            raise InferenceFieldError("MLP model must be (encoder, edge_layer)")
        scores = predict_mlp_scores(model[0], model[1], sanitized, len(frame), device="cpu")
    else:
        raise InferenceFieldError("family must be gnn or mlp")
    keys = source_regions[source_column].astype(str)
    if keys.duplicated().any():
        raise InferenceFieldError("source demand keys are not unique")
    demand = dict(zip(keys, source_regions[demand_column].astype(float), strict=True))
    field = compose_support_field(
        frame[source_column].astype(str).to_numpy(),
        demand,
        frame["covered_mask"].to_numpy(bool),
        frame["unknown_mask"].to_numpy(bool),
        frame["built_fraction"].to_numpy(float),
        np.asarray(scores, dtype=float).reshape(-1),
    )
    if field.shape != (len(frame),) or not np.isfinite(field).all() or (field < 0).any():
        raise InferenceFieldError("inference field is invalid")
    if np.count_nonzero(field[frame["zero_mask"].to_numpy(bool)]):
        raise InferenceFieldError("inference field violates Z exact-zero")
    observed = {
        key: float(field[frame[source_column].astype(str).eq(key)].sum())
        for key in demand
    }
    if any(not np.isclose(observed[key], demand[key], rtol=1e-8, atol=1e-8) for key in demand):
        raise InferenceFieldError("inference field violates source conservation")
    return field


def write_field(path: Path, field: np.ndarray) -> None:
    partial = path.with_name(f".{path.name}.part")
    if partial.exists():
        raise InferenceTaskError(f"interrupted field write requires manual resolution: {partial}")
    with partial.open("wb") as stream:
        np.savez_compressed(stream, data=np.asarray(field, dtype=np.float64))
    partial.replace(path)
