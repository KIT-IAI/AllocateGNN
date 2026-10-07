\
\
\
\
\
   
import warnings
from collections import defaultdict
from typing import Dict, Optional, Tuple

import numpy as np
import pyomo.environ as pyo

from .pmedian_solver import haversine_distance_matrix, solve_pmedian_greedy


def solve_pmedian_milp(
    demand_coords: np.ndarray,
    facility_coords: np.ndarray,
    weights: np.ndarray,
    k: int,
    milp_config: Dict,
    warm_start_selected: Optional[np.ndarray] = None,
    warm_start_assignment: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, str]:
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
       
    solver_name = milp_config.get("solver", "scip")
    time_limit = milp_config.get("time_limit_seconds", 300)
    k_nearest = milp_config.get("k_nearest", 5)

    N = len(demand_coords)
    M = len(facility_coords)
    k = min(k, M)

    dist_matrix = haversine_distance_matrix(demand_coords, facility_coords)

                                          
    k_nn = min(max(k_nearest, k), M)
    if k_nn >= M:
        sparse_pairs = [(i, j) for i in range(N) for j in range(M)]
    else:
        nn_indices = np.argpartition(dist_matrix, k_nn, axis=1)[:, :k_nn]
        sparse_pairs = []
        for i in range(N):
            for j in nn_indices[i]:
                sparse_pairs.append((i, int(j)))

                                                  
    neighbors_of = defaultdict(list)
    for (i, j) in sparse_pairs:
        neighbors_of[i].append(j)

              
    model = pyo.ConcreteModel()
    model.I = pyo.RangeSet(0, N - 1)
    model.J = pyo.RangeSet(0, M - 1)
    model.S = pyo.Set(initialize=sparse_pairs)

                                           
    model.y = pyo.Var(model.J, domain=pyo.Binary)
    model.x = pyo.Var(model.S, domain=pyo.NonNegativeReals, bounds=(0, 1))

          
    model.obj = pyo.Objective(
        expr=pyo.quicksum(
            weights[i] * dist_matrix[i, j] * model.x[i, j]
            for (i, j) in sparse_pairs
        ),
        sense=pyo.minimize,
    )

                   
    model.facility_count = pyo.Constraint(
        expr=pyo.quicksum(model.y[j] for j in model.J) == k
    )

                         
    def demand_assign_rule(m, i):
        nbrs = neighbors_of[i]
        if not nbrs:
            return pyo.Constraint.Skip
        return pyo.quicksum(m.x[i, j] for j in nbrs) == 1
    model.demand_assign = pyo.Constraint(model.I, rule=demand_assign_rule)

                      
    def link_rule(m, i, j):
        return m.x[i, j] <= m.y[j]
    model.link = pyo.Constraint(model.S, rule=link_rule)

                
    if warm_start_selected is not None:
        ws_set = set(int(j) for j in warm_start_selected)
        for j in range(M):
            model.y[j].value = 1 if j in ws_set else 0
        if warm_start_assignment is not None:
            for (i, j) in sparse_pairs:
                model.x[i, j].value = (
                    1.0 if int(warm_start_assignment[i]) == j else 0.0
                )

        
    solver = pyo.SolverFactory(solver_name)
    if solver_name == "scip":
        solver.options["limits/time"] = time_limit
    elif solver_name == "gurobi":
        solver.options["TimeLimit"] = time_limit

    try:
        result = solver.solve(model, tee=False)
        status = str(result.solver.termination_condition)
        print(f"  MILP termination: {status}")
    except Exception as e:
        warnings.warn(f"MILP solver error: {e}")
        return np.array([], dtype=int), np.array([], dtype=int), "error"

          
    if status in ("optimal", "feasible"):
        solver_status = "optimal" if status == "optimal" else "time_limit"
    elif "time" in status.lower() or "limit" in status.lower():
        solver_status = "time_limit"
    else:
        solver_status = "error"

                                           
    if solver_status in ("optimal", "time_limit"):
        try:
            selected = np.array([
                j for j in range(M)
                if pyo.value(model.y[j], exception=False) is not None
                and pyo.value(model.y[j], exception=False) > 0.5
            ])
        except Exception:
            warnings.warn("MILP variables are uninitialised; treating the solve as failed")
            return np.array([], dtype=int), np.array([], dtype=int), "error"

        if len(selected) == 0:
            warnings.warn("MILP selected no sites; treating the solve as failed")
            return np.array([], dtype=int), np.array([], dtype=int), "error"

                               
        assignment = np.full(N, -1, dtype=int)
        for i in range(N):
            best_j = -1
            best_val = -1.0
            for j in neighbors_of[i]:
                val = pyo.value(model.x[i, j], exception=False)
                if val is not None and val > best_val:
                    best_val = val
                    best_j = j
            assignment[i] = best_j

                
        unassigned = np.where(assignment < 0)[0]
        if len(unassigned) > 0:
            sub_dist = dist_matrix[np.ix_(unassigned, selected)]
            local_nearest = sub_dist.argmin(axis=1)
            assignment[unassigned] = selected[local_nearest]

        return selected, assignment, solver_status
    else:
        return np.array([], dtype=int), np.array([], dtype=int), solver_status


def solve_with_fallback(
    demand_coords: np.ndarray,
    facility_coords: np.ndarray,
    weights: np.ndarray,
    k: int,
    config: Dict,
) -> Tuple[np.ndarray, np.ndarray, str]:
\
\
\
\
\
       
                       
    greedy_sel, greedy_assign = solve_pmedian_greedy(
        demand_coords, facility_coords, weights, k,
        config.get("pmedian", {"max_iter": 50, "random_restarts": 1}),
    )

    selected, assignment, status = solve_pmedian_milp(
        demand_coords, facility_coords, weights, k, config["milp"],
        warm_start_selected=greedy_sel,
        warm_start_assignment=greedy_assign,
    )

    if status in ("optimal", "time_limit") and len(selected) > 0:
        return selected, assignment, "milp"

    if config["milp"].get("fallback_to_greedy", True):
        warnings.warn(f"MILP status is '{status}'; using the greedy fallback")
        return greedy_sel, greedy_assign, "greedy_fallback"

    return selected, assignment, "milp"
