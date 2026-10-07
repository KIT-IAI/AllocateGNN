\
\
\
\
\
   
import numpy as np
from typing import Dict, List, Sequence


def _haversine_vector(lon1: np.ndarray, lat1: np.ndarray,
                      lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
                                     
    R = 6371.0
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return R * 2 * np.arcsin(np.sqrt(a))


def compute_wsd(demand_coords: np.ndarray,
                facility_coords: np.ndarray,
                assignment: np.ndarray,
                weights: np.ndarray) -> float:
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
       
    assigned_facility = facility_coords[assignment]
    dist = _haversine_vector(
        demand_coords[:, 0], demand_coords[:, 1],
        assigned_facility[:, 0], assigned_facility[:, 1],
    )
    w_sum = weights.sum()
    if w_sum == 0:
        return 0.0
    return float(np.dot(weights, dist) / w_sum)


def compute_lwfl(demand_coords: np.ndarray,
                 facility_coords: np.ndarray,
                 assignment: np.ndarray,
                 weights: np.ndarray) -> float:
\
\
\
\
\
       
    assigned_facility = facility_coords[assignment]
    dist = _haversine_vector(
        demand_coords[:, 0], demand_coords[:, 1],
        assigned_facility[:, 0], assigned_facility[:, 1],
    )
    return float(np.dot(weights, dist))


def compute_lbi(assignment: np.ndarray,
                weights: np.ndarray,
                k: int) -> float:
\
\
\
\
\
       
    loads = np.zeros(k)
    np.add.at(loads, assignment, weights)
    mean_load = loads.mean()
    if mean_load == 0:
        return 0.0
    return float(loads.std() / mean_load)


def compute_dcr(demand_coords: np.ndarray,
                facility_coords: np.ndarray,
                assignment: np.ndarray,
                weights: np.ndarray,
                radius_km: float) -> float:
\
\
\
\
\
       
    assigned_facility = facility_coords[assignment]
    dist = _haversine_vector(
        demand_coords[:, 0], demand_coords[:, 1],
        assigned_facility[:, 0], assigned_facility[:, 1],
    )
    w_sum = weights.sum()
    if w_sum == 0:
        return 0.0
    covered = weights[dist <= radius_km].sum()
    return float(covered / w_sum)


def compute_all_siting_metrics(demand_coords: np.ndarray,
                               facility_coords: np.ndarray,
                               assignment: np.ndarray,
                               weights: np.ndarray,
                               dcr_radii: Sequence[float]) -> Dict[str, float]:
\
\
\
\
\
\
\
\
       
    k = int(facility_coords.shape[0])
    result = {
        "WSD": compute_wsd(demand_coords, facility_coords, assignment, weights),
        "LWFL": compute_lwfl(demand_coords, facility_coords, assignment, weights),
        "LBI": compute_lbi(assignment, weights, k),
    }
    for r in dcr_radii:
        key = f"DCR_{int(r)}km" if r == int(r) else f"DCR_{r}km"
        result[key] = compute_dcr(demand_coords, facility_coords, assignment, weights, r)
    return result


              


def _round_up_to_standard(value: float, standard_sizes: Sequence[float]) -> float:
                                                  
    sizes = sorted(standard_sizes)
    for s in sizes:
        if s >= value:
            return s
                     
    max_size = sizes[-1]
    return float(np.ceil(value / max_size) * max_size)


def compute_tur(predicted_demand_mva: np.ndarray,
                actual_demand_mva: np.ndarray,
                sizing_config: dict) -> tuple:
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
       
    safety = sizing_config["safety_margin"]
    sizes = sizing_config["standard_sizes_mva"]

    q_rec = np.array([
        _round_up_to_standard(p * safety, sizes)
        for p in predicted_demand_mva
    ])
    tur = np.where(q_rec > 0, actual_demand_mva / q_rec * 100.0, 0.0)
    return tur, q_rec


def compute_all_sizing_metrics(predicted_demand_mva: np.ndarray,
                               actual_demand_mva: np.ndarray,
                               sizing_config: dict,
                               firm_capacity_mva: np.ndarray = None) -> Dict[str, float]:
\
\
\
\
\
\
\
\
       
    tur, q_rec = compute_tur(predicted_demand_mva, actual_demand_mva, sizing_config)

                 
    ce = np.where(
        actual_demand_mva > 0,
        np.abs(q_rec - actual_demand_mva) / actual_demand_mva * 100.0,
        0.0,
    )

                                     
    opr = float(np.mean(q_rec > 2 * actual_demand_mva))

                                
    upr = float(np.mean(q_rec < actual_demand_mva))

    result = {
        "TUR_mean": float(tur.mean()),
        "TUR_std": float(tur.std()),
        "CE_mean": float(ce.mean()),
        "OPR": opr,
        "UPR": upr,
    }

                        
    if firm_capacity_mva is not None:
        valid = (firm_capacity_mva > 0) & np.isfinite(firm_capacity_mva)
        if valid.sum() > 0:
            fc_valid = firm_capacity_mva[valid]
            qr_valid = q_rec[valid]
            fce = np.abs(qr_valid - fc_valid) / fc_valid * 100.0
            result["FCE_mean"] = float(fce.mean())
            result["FC_match_rate"] = float(np.mean(np.isclose(qr_valid, fc_valid, rtol=0.01)))

    return result
