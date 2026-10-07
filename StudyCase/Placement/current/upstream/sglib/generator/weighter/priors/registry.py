from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np


@dataclass(frozen=True)
class PriorSpec:
    id: str
    source_artifact: str
    normalization: str
    support_rule: str
    correction_factor: Callable[[np.ndarray], np.ndarray]
    loss_terms: tuple[str, ...]
    feature_channels: tuple[str, ...]


class PriorRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, PriorSpec] = {}

    def register(self, spec: PriorSpec) -> PriorSpec:
        if spec.id in self._specs:
            raise ValueError(f"prior is already registered: {spec.id}")
        self._specs[spec.id] = spec
        return spec

    def get(self, prior_id: str) -> PriorSpec:
        try:
            return self._specs[prior_id]
        except KeyError as exc:
            raise KeyError(f"unknown prior {prior_id!r}; available={sorted(self._specs)}") from exc

    def ids(self) -> tuple[str, ...]:
        return tuple(self._specs)


prior_registry = PriorRegistry()

