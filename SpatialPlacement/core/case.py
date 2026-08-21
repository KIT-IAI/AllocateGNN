                       
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
   
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Protocol, runtime_checkable

import numpy as np


@dataclass
class RegionBundle:
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
\
       
    region: str
    grid_lonlat: np.ndarray
    source_key: np.ndarray
    lu_prop: np.ndarray
    source_demand: Dict[str, float]
    source_pct: Dict[str, np.ndarray]
    station_lonlat: np.ndarray
    station_demand: np.ndarray
    station_firm: np.ndarray
    d_region: float
    k: int
    buildable: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        n = len(self.grid_lonlat)
        if len(self.source_key) != n or len(self.lu_prop) != n:
            raise ValueError(f"{self.region}: grid arrays have inconsistent lengths")
        m = len(self.station_lonlat)
        if len(self.station_demand) != m or len(self.station_firm) != m:
            raise ValueError(f"{self.region}: station arrays have inconsistent lengths")
        if self.k != m:
            raise ValueError(f"{self.region}: k={self.k} differs from station count {m}")


@dataclass
class CostSpec:
\
\
\
\
\
\
\
\
       
    currency: Optional[str] = None
    unit_cost_per_kva: Optional[float] = None
    horizon_years: int = 20
    discount: float = 0.035


@runtime_checkable
class CaseSpec(Protocol):
\
\
\
\
       

    name: str
    cost: CostSpec

    def regions(self) -> List[str]:
                           
        ...

    def bundle(self, region: str) -> RegionBundle:
                                  
        ...

    def gnn_fields(self, region: str) -> Dict[str, np.ndarray]:
\
\
\
\
           
        ...

    def tariff(self, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
                                            
        ...
