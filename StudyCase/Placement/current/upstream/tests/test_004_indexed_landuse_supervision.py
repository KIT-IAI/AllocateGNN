"""Dormant indexed land-use supervision compatibility contract."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader

from sglib.generator.inputs import GeneratorDataError
from sglib.generator.weighter.learned.inputs import (
    GRAPH_CACHE_SCHEMA_V2,
    GRAPH_CACHE_SCHEMA_V3,
    load_graph_cache,
    save_graph_cache,
)
from sglib.generator.weighter.learned.models.gnn.graph import (
    _build_landuse_matrices,
)
from sglib.generator.weighter.learned.models.gnn.inference import (
    assert_inference_graph,
    sanitize_for_inference,
)
from sglib.generator.weighter.learned.models.gnn.landuse import (
    INDEXED_LANDUSE_REPRESENTATION,
    LEGACY_DENSE_LANDUSE_REPRESENTATION,
)
from sglib.generator.weighter.learned.models.gnn.losses.base import (
    LandusePredictionLoss,
    landuse_prediction_ratio,
)
from sglib.generator.weighter.learned.models.gnn.solver import (
    _add_landuse_loss_metadata,
)
from sglib.generator.weighter.learned.models.mlp import loss_metadata

pytestmark = pytest.mark.consume


def _supervision():
    edge_index = torch.tensor(
        [[0, 0, 0, 1, 1], [0, 1, 2, 3, 4]], dtype=torch.int64
    )
    flat_index = torch.tensor([0, 1, -1, 3, 5], dtype=torch.int64)
    dense = torch.zeros((5, 6), dtype=torch.float64)
    dense[0, 0] = 1
    dense[1, 1] = 1
    dense[3, 3] = 1
    dense[4, 5] = 1
    ratio = torch.tensor(
        [[0.25, 0.75, 0.0], [0.5, 0.0, 0.5]], dtype=torch.float64
    )
    dense_metadata = {
        "num_s": 2,
        "landuse_mapping_matrix": dense,
        "landuse_ratio": ratio,
    }
    indexed_metadata = {
        "num_s": 2,
        "landuse_flat_index": flat_index,
        "landuse_ratio": ratio,
        "landuse_supervision_representation": INDEXED_LANDUSE_REPRESENTATION,
    }
    return edge_index, dense_metadata, indexed_metadata


def _graph(*, indexed: bool) -> HeteroData:
    edge_index, dense, sparse = _supervision()
    graph = HeteroData()
    graph["source"].x = torch.ones((2, 1))
    graph["agent"].x = torch.ones((5, 1))
    graph["source", "connects_to", "agent"].edge_index = edge_index
    graph.landuse_ratio = dense["landuse_ratio"].to(torch.float32)
    if indexed:
        graph.landuse_flat_index = sparse["landuse_flat_index"]
        graph.landuse_supervision_representation = INDEXED_LANDUSE_REPRESENTATION
    else:
        graph.landuse_mapping_matrix = dense["landuse_mapping_matrix"].to(
            torch.float32
        )
    return graph


def _bundle():
    return SimpleNamespace(
        country="uk",
        regions=("R",),
        source_column="source_id",
        demand_column="demand",
        relation_column="source_id",
        grids={"R": None},
        ntl={"R": None},
        source_regions={"R": None},
        stations={"R": None},
    )


def test_graph_builder_defaults_dense_and_indexed_is_explicit() -> None:
    sources = pd.DataFrame(
        {"a_percent": [2.0, 0.0], "b_percent": [1.0, 4.0]}
    )
    agents = pd.DataFrame({"landuse": ["a", "b", None, "b"]})
    edges = torch.tensor([[0, 0, 0, 1], [0, 1, 2, 3]], dtype=torch.int64)

    dense = _build_landuse_matrices(sources, agents, edges)
    indexed = _build_landuse_matrices(
        sources,
        agents,
        edges,
        representation=INDEXED_LANDUSE_REPRESENTATION,
    )

    assert set(dense) == {"landuse_mapping_matrix", "landuse_ratio"}
    assert dense["landuse_mapping_matrix"].shape == (4, 4)
    assert set(indexed) == {
        "landuse_flat_index",
        "landuse_ratio",
        "landuse_supervision_representation",
    }
    assert indexed["landuse_flat_index"].dtype == torch.int64
    assert indexed["landuse_flat_index"].tolist() == [0, 1, -1, 3]
    assert indexed["landuse_supervision_representation"] == (
        INDEXED_LANDUSE_REPRESENTATION
    )
    assert torch.equal(dense["landuse_ratio"], indexed["landuse_ratio"])


def test_dense_and_indexed_forward_ratio_loss_and_gradient_are_equivalent() -> None:
    edge_index, dense, indexed = _supervision()
    dense_weights = torch.tensor(
        [0.2, 0.3, 0.1, 0.4, 0.5], dtype=torch.float64, requires_grad=True
    )
    indexed_weights = dense_weights.detach().clone().requires_grad_(True)

    dense_ratio, _ = landuse_prediction_ratio(dense_weights, edge_index, dense)
    indexed_ratio, _ = landuse_prediction_ratio(
        indexed_weights, edge_index, indexed
    )
    expected = torch.tensor(
        [[0.4, 0.6, 0.0], [4 / 9, 0.0, 5 / 9]], dtype=torch.float64
    )
    assert torch.allclose(dense_ratio, expected, atol=2e-8, rtol=0)
    assert torch.equal(dense_ratio, indexed_ratio)

    objective = LandusePredictionLoss()
    dense_loss = objective(dense_weights, edge_index, dense)
    indexed_loss = objective(indexed_weights, edge_index, indexed)
    assert torch.equal(dense_loss, indexed_loss)
    dense_loss.backward()
    indexed_loss.backward()
    assert torch.equal(dense_weights.grad, indexed_weights.grad)


def test_indexed_path_obeys_determinism_contract() -> None:
    edge_index, _dense, indexed = _supervision()
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    try:
        observed = []
        for _ in range(3):
            weights = torch.tensor(
                [0.2, 0.3, 0.1, 0.4, 0.5],
                dtype=torch.float64,
                requires_grad=True,
            )
            loss = LandusePredictionLoss()(weights, edge_index, indexed)
            loss.backward()
            observed.append((loss.detach().clone(), weights.grad.detach().clone()))
        assert all(torch.equal(observed[0][0], item[0]) for item in observed[1:])
        assert all(torch.equal(observed[0][1], item[1]) for item in observed[1:])
    finally:
        torch.use_deterministic_algorithms(previous)


def test_indexed_metadata_is_strict_and_never_silently_zero() -> None:
    edge_index, dense, indexed = _supervision()
    weights = torch.ones(5, dtype=torch.float64)

    invalid = []
    missing_ratio = deepcopy(indexed)
    del missing_ratio["landuse_ratio"]
    invalid.append((missing_ratio, "missing landuse_ratio"))
    missing_representation = deepcopy(indexed)
    del missing_representation["landuse_supervision_representation"]
    invalid.append((missing_representation, "missing representation"))
    wrong_dtype = deepcopy(indexed)
    wrong_dtype["landuse_flat_index"] = wrong_dtype[
        "landuse_flat_index"
    ].to(torch.int32)
    invalid.append((wrong_dtype, "int64"))
    wrong_length = deepcopy(indexed)
    wrong_length["landuse_flat_index"] = torch.tensor([0], dtype=torch.int64)
    invalid.append((wrong_length, "length"))
    below_sentinel = deepcopy(indexed)
    below_sentinel["landuse_flat_index"][0] = -2
    invalid.append((below_sentinel, "-1 or within"))
    above_range = deepcopy(indexed)
    above_range["landuse_flat_index"][0] = 6
    invalid.append((above_range, "-1 or within"))
    wrong_source = deepcopy(indexed)
    wrong_source["landuse_flat_index"][0] = 3
    invalid.append((wrong_source, "source component"))
    both = deepcopy(indexed)
    both["landuse_mapping_matrix"] = dense["landuse_mapping_matrix"]
    invalid.append((both, "exactly one"))
    missing_both = {"num_s": 2, "landuse_ratio": indexed["landuse_ratio"]}
    invalid.append((missing_both, "exactly one"))
    unknown = deepcopy(indexed)
    unknown["landuse_supervision_representation"] = "future_unknown"
    invalid.append((unknown, "unknown"))

    for metadata, message in invalid:
        with pytest.raises((TypeError, ValueError), match=message):
            LandusePredictionLoss()(weights, edge_index, metadata)


def test_dense_v2_metadata_remains_valid_but_is_also_validated() -> None:
    edge_index, dense, _indexed = _supervision()
    weights = torch.ones(5, dtype=torch.float64)
    assert torch.isfinite(LandusePredictionLoss()(weights, edge_index, dense))

    malformed = deepcopy(dense)
    malformed["landuse_mapping_matrix"] = torch.ones((4, 6))
    with pytest.raises(ValueError, match="shape"):
        LandusePredictionLoss()(weights, edge_index, malformed)
    malformed = deepcopy(dense)
    malformed["landuse_mapping_matrix"][0, 1] = 1
    with pytest.raises(ValueError, match="rows must sum"):
        LandusePredictionLoss()(weights, edge_index, malformed)


def test_gnn_and_mlp_pass_indexed_loss_metadata() -> None:
    graph = _graph(indexed=True)
    mlp_metadata = loss_metadata(graph)
    assert mlp_metadata["landuse_flat_index"].dtype == torch.int64
    assert mlp_metadata["landuse_supervision_representation"] == (
        INDEXED_LANDUSE_REPRESENTATION
    )

    batch = next(iter(DataLoader([graph], batch_size=1, shuffle=False)))
    gnn_metadata = {
        "num_s": int(batch["source"].num_nodes),
        "num_a": int(batch["agent"].num_nodes),
    }
    _add_landuse_loss_metadata(batch, gnn_metadata)
    assert gnn_metadata["landuse_supervision_representation"] == [
        INDEXED_LANDUSE_REPRESENTATION
    ]
    weights = torch.ones(5, dtype=torch.float32)
    assert torch.isfinite(
        LandusePredictionLoss()(
            weights,
            batch["source", "connects_to", "agent"].edge_index,
            gnn_metadata,
        )
    )


def test_inference_sanitizer_removes_dense_and_indexed_supervision() -> None:
    for indexed in (False, True):
        graph = _graph(indexed=indexed)
        graph["source"].y = torch.ones(2)
        cleaned = sanitize_for_inference(graph)
        assert_inference_graph(cleaned)
        assert "landuse_ratio" not in cleaned
        assert "landuse_mapping_matrix" not in cleaned
        assert "landuse_flat_index" not in cleaned
        assert "landuse_supervision_representation" not in cleaned
        assert "landuse_ratio" in graph


def test_cache_writes_dense_v2_indexed_v3_and_reads_both(tmp_path) -> None:
    bundle = _bundle()
    dense_path = save_graph_cache(
        tmp_path / "dense.pkl",
        bundle,
        {"R": _graph(indexed=False)},
        feature_set="lu5",
        input_fingerprints={"inputs_receipt": "dense"},
    )
    indexed_path = save_graph_cache(
        tmp_path / "indexed.pkl",
        bundle,
        {"R": _graph(indexed=True)},
        feature_set="lu5",
        input_fingerprints={"inputs_receipt": "indexed"},
    )
    dense = load_graph_cache(
        dense_path, expected_country="uk", expected_feature_set="lu5"
    )
    indexed = load_graph_cache(
        indexed_path, expected_country="uk", expected_feature_set="lu5"
    )
    assert dense["schema"] == GRAPH_CACHE_SCHEMA_V2
    assert "landuse_supervision_representation" not in dense
    assert indexed["schema"] == GRAPH_CACHE_SCHEMA_V3
    assert indexed["landuse_supervision_representation"] == (
        INDEXED_LANDUSE_REPRESENTATION
    )
    assert "landuse_mapping_matrix" in dense["graphs"]["R"]
    assert "landuse_flat_index" in indexed["graphs"]["R"]


def test_cache_rejects_ambiguous_supervision(tmp_path) -> None:
    graph = _graph(indexed=True)
    graph.landuse_mapping_matrix = torch.zeros((5, 6))
    with pytest.raises(GeneratorDataError, match="mixes dense and indexed"):
        save_graph_cache(
            tmp_path / "ambiguous.pkl",
            _bundle(),
            {"R": graph},
            feature_set="lu5",
            input_fingerprints={"inputs_receipt": "bad"},
        )


def test_default_representation_identifier_is_not_the_indexed_opt_in() -> None:
    assert LEGACY_DENSE_LANDUSE_REPRESENTATION != INDEXED_LANDUSE_REPRESENTATION
