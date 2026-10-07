"""正式 TEST 推理图的监督张量移除。"""

from __future__ import annotations

from torch_geometric.data import HeteroData


GLOBAL_SUPERVISION = (
    "landuse_mapping_matrix",
    "landuse_flat_index",
    "landuse_ratio",
    "landuse_supervision_representation",
    "macro_attributes",
    "attribute_stds",
)
AGENT_SUPERVISION = (
    "demand",
    "ntl_values",
    "proximity_scores",
    "rci_mask",
)


def sanitize_for_inference(graph: HeteroData) -> HeteroData:
    """克隆图并物理删除训练目标；节点输入与边结构保持不变。"""

    data = graph.clone()
    source_store = data["source"]
    if "y" in source_store:
        del source_store["y"]
    agent_store = data["agent"]
    for name in AGENT_SUPERVISION:
        if name in agent_store:
            del agent_store[name]
    for name in GLOBAL_SUPERVISION:
        if name in data:
            del data[name]
    return data


def assert_inference_graph(graph: HeteroData) -> None:
    """供 driver/test 证明传给模型的图不再携带监督张量。"""

    if "y" in graph["source"]:
        raise ValueError("TEST inference graph still contains source.y")
    leaked_agent = sorted(set(AGENT_SUPERVISION) & set(graph["agent"].keys()))
    leaked_global = sorted(name for name in GLOBAL_SUPERVISION if name in graph)
    if leaked_agent or leaked_global:
        raise ValueError(
            f"TEST inference supervision leak: agent={leaked_agent}, global={leaked_global}"
        )

