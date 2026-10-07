"""The single content-addressed identity and lineage primitive.

Callers pass an explicit scientific projection; this module never hashes an
ambient file/config container or a runtime locator by implication.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import re
import textwrap
from typing import Any, Iterable, Mapping

from .hashing import canonical_json, sha256_json


COMMITMENT = "sg_content_chain_commitment_v1"
RECEIPT = "sg_content_chain_receipt_v1"
CLOSURE = "sg_content_chain_closure_v1"
_SHA = re.compile(r"^[0-9a-f]{64}$")


class ContentChainError(RuntimeError):
    pass


def _canonical(value: Any, label: str) -> Any:
    try:
        return json.loads(canonical_json(value))
    except (TypeError, ValueError) as exc:
        raise ContentChainError(f"{label} is not canonical JSON") from exc


def _digest(value: Any, label: str) -> str:
    digest = str(value).lower()
    if _SHA.fullmatch(digest) is None:
        raise ContentChainError(f"{label} must be a SHA-256 digest")
    return digest


def _hashes(values: Mapping[str, Any], label: str, *, empty: bool = False) -> dict[str, str]:
    if not isinstance(values, Mapping) or (not values and not empty):
        raise ContentChainError(f"{label} must be a mapping")
    if any(not str(name) for name in values):
        raise ContentChainError(f"{label} contains an empty name")
    return dict(
        sorted(
            (
                str(name),
                _digest(
                    value.get("sha256") if isinstance(value, Mapping) else value,
                    f"{label}.{name}",
                ),
            )
            for name, value in values.items()
            if str(name)
        )
    )


def code_projection(symbols: Mapping[str, Any]) -> dict[str, Any]:
    """Hash normalized ASTs of explicitly selected computational symbols."""

    if not symbols or any(not str(role) for role in symbols):
        raise ContentChainError("code projection symbols must be named")
    identity: dict[str, str] = {}
    locators: dict[str, dict[str, str]] = {}
    for role, symbol in sorted(symbols.items()):
        try:
            tree = ast.parse(textwrap.dedent(inspect.getsource(symbol)))
        except (OSError, TypeError, SyntaxError) as exc:
            raise ContentChainError(f"cannot project code symbol: {role}") from exc
        node = tree.body[0]
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            node.name = "__scientific_symbol__"
            if node.body and isinstance(node.body[0], ast.Expr) and isinstance(
                node.body[0].value, ast.Constant
            ) and isinstance(node.body[0].value.value, str):
                node.body.pop(0)
        normalized = ast.dump(tree, annotate_fields=True, include_attributes=False)
        identity[str(role)] = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
        locators[str(role)] = {
            "module": symbol.__module__,
            "qualname": symbol.__qualname__,
        }
    projection = {"schema_version": "sg_code_projection_v1", "symbols": identity}
    return {
        **projection,
        "code_sha256": sha256_json(projection),
        "locators": locators,
        "locator_identity_binding": "excluded",
    }


def derive_chain_commitment(
    node_id: str,
    *,
    inputs: Mapping[str, Any],
    scientific_parameters: Any,
    code_sha256: str,
) -> dict[str, Any]:
    parameters = _canonical(scientific_parameters, "scientific_parameters")
    identity = {
        "node_id": str(node_id),
        "inputs": _hashes(inputs, "inputs", empty=True),
        "scientific_parameters_sha256": sha256_json(parameters),
        "code_sha256": _digest(code_sha256, "code_sha256"),
    }
    if not identity["node_id"]:
        raise ContentChainError("node_id is empty")
    return {
        "schema_version": COMMITMENT,
        **identity,
        "scientific_parameters": parameters,
        "commitment_sha256": sha256_json(identity),
    }


def derive_chain_receipt(
    commitment: Mapping[str, Any],
    *,
    outputs: Mapping[str, Any],
    observations: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    committed = verify_chain(commitment)
    if committed["schema_version"] != COMMITMENT:
        raise ContentChainError("receipt requires a commitment")
    output_hashes = _hashes(outputs, "outputs")
    identity = {
        "commitment_sha256": committed["commitment_sha256"],
        "outputs": output_hashes,
    }
    receipt = {
        "schema_version": RECEIPT,
        "node_id": committed["node_id"],
        "commitment": committed,
        "outputs": output_hashes,
        "receipt_sha256": sha256_json(identity),
    }
    if observations is not None:
        receipt["observations"] = _canonical(observations, "observations")
    return receipt


def derive_chain_closure(receipts: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    nodes = []
    for raw in receipts:
        receipt = verify_chain(raw)
        if receipt["schema_version"] != RECEIPT:
            raise ContentChainError("closure requires receipts")
        nodes.append(
            {
                "node_id": receipt["node_id"],
                "commitment_sha256": receipt["commitment"]["commitment_sha256"],
                "receipt_sha256": receipt["receipt_sha256"],
                "outputs": receipt["outputs"],
            }
        )
    nodes.sort(key=lambda item: item["node_id"])
    if not nodes or len({item["node_id"] for item in nodes}) != len(nodes):
        raise ContentChainError("closure nodes must be unique and non-empty")
    projection = {"node_count": len(nodes), "nodes": nodes}
    return {"schema_version": CLOSURE, **projection, "closure_sha256": sha256_json(projection)}


def verify_chain(document: Mapping[str, Any]) -> dict[str, Any]:
    """The only verifier: canonicalize and rederive any chain document."""

    if not isinstance(document, Mapping):
        raise ContentChainError("chain document must be a mapping")
    schema = document.get("schema_version")
    if schema == COMMITMENT:
        expected = derive_chain_commitment(
            str(document.get("node_id", "")),
            inputs=document.get("inputs", {}),
            scientific_parameters=document.get("scientific_parameters"),
            code_sha256=str(document.get("code_sha256", "")),
        )
    elif schema == RECEIPT:
        expected = derive_chain_receipt(
            document.get("commitment", {}),
            outputs=document.get("outputs", {}),
            observations=document.get("observations") if "observations" in document else None,
        )
    elif schema == CLOSURE:
        nodes = document.get("nodes")
        if not isinstance(nodes, list) or not nodes:
            raise ContentChainError("closure nodes are missing")
        normalized = [
            {
                "node_id": str(item["node_id"]),
                "commitment_sha256": _digest(item["commitment_sha256"], "commitment_sha256"),
                "receipt_sha256": _digest(item["receipt_sha256"], "receipt_sha256"),
                "outputs": _hashes(item["outputs"], "outputs"),
            }
            for item in nodes
        ]
        normalized.sort(key=lambda item: item["node_id"])
        if any(not item["node_id"] for item in normalized) or len(
            {item["node_id"] for item in normalized}
        ) != len(normalized):
            raise ContentChainError("closure nodes must be unique and non-empty")
        projection = {"node_count": len(normalized), "nodes": normalized}
        expected = {"schema_version": CLOSURE, **projection, "closure_sha256": sha256_json(projection)}
    else:
        raise ContentChainError("unknown content-chain schema")
    if dict(document) != expected:
        raise ContentChainError("content-chain projection differs")
    return expected


def verify_chain_link(
    parent: Mapping[str, Any], child: Mapping[str, Any], links: Mapping[str, str]
) -> None:
    """The sole lineage rule: child input hash equals parent output hash."""

    parent = verify_chain(parent)
    child = verify_chain(child)
    if not isinstance(links, Mapping) or not links:
        raise ContentChainError("chain links must be non-empty")
    parent_outputs = parent.get("outputs")
    if parent["schema_version"] != RECEIPT or not isinstance(parent_outputs, Mapping):
        raise ContentChainError("parent must be a receipt")
    commitment = child["commitment"] if child["schema_version"] == RECEIPT else child
    for child_name, parent_name in links.items():
        if child_name not in commitment["inputs"] or parent_name not in parent_outputs:
            raise ContentChainError(f"chain link endpoint is missing: {child_name} <- {parent_name}")
        if commitment["inputs"][child_name] != parent_outputs[parent_name]:
            raise ContentChainError(f"chain link hash differs: {child_name} <- {parent_name}")


__all__ = [
    "CLOSURE",
    "COMMITMENT",
    "RECEIPT",
    "ContentChainError",
    "code_projection",
    "derive_chain_closure",
    "derive_chain_commitment",
    "derive_chain_receipt",
    "verify_chain",
    "verify_chain_link",
]
