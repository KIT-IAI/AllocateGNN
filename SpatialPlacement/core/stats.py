                       
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

from itertools import product
from typing import Dict

import numpy as np

                                    
MAX_EXHAUSTIVE = 20
MC_DRAWS = 200_000
MC_SEED = 42


def min_attainable_p(n: int) -> float:
                                          
    if n < 1:
        return 1.0
    return min(1.0, 2.0 ** (1 - n))


def exact_sign_flip_p(d: np.ndarray) -> Dict[str, float]:
\
\
\
\
\
\
\
       
    d = np.asarray(d, dtype=float)
    n = len(d)
    if n == 0 or np.allclose(d, 0):
        return {"p": 1.0, "exact": 1.0, "n": float(n)}

    obs = abs(d.sum())
    if n <= MAX_EXHAUSTIVE:
        signs = np.array(list(product([1.0, -1.0], repeat=n)))             
        stats = np.abs(signs @ d)
                                                                                
                                                                        
                                                                                
                                                                               
                                                                         
                                                                    
        tol = max(1e-12, np.spacing(max(1.0, obs)) * 64.0)
        return {"p": float((stats >= obs - tol).mean()),
                "exact": 1.0, "n": float(n)}

    rng = np.random.RandomState(MC_SEED)
    signs = rng.choice([1.0, -1.0], size=(MC_DRAWS, n))
    stats = np.abs(signs @ d)
    return {"p": float((stats >= obs - 1e-12).mean()),
            "exact": 0.0, "n": float(n)}


def paired_bootstrap_ci(d: np.ndarray, b: int = 10_000,
                        seed: int = 42) -> tuple[float, float]:
                                                             
    rng = np.random.RandomState(seed)
    d = np.asarray(d, dtype=float)
    n = len(d)
    means = np.array([d[rng.randint(0, n, n)].mean() for _ in range(b)])
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def seed_mean_report(base: np.ndarray, per_seed: np.ndarray,
                     lower_is_better: bool = True) -> Dict[str, object]:
\
\
\
\
\
\
\
\
       
    b = np.asarray(base, float)
    mat = np.asarray(per_seed, float)
    sign = 1.0 if lower_is_better else -1.0

    def one(d: np.ndarray, with_ci: bool) -> Dict[str, object]:
        t = exact_sign_flip_p(d)
                                                         
                                                    
        ratio = np.divide(d, b, out=np.full_like(d, np.nan), where=b != 0)
        out: Dict[str, object] = {
            "median_impr_pct": float(np.median(ratio * 100.0)),
            "wins": int((d > 0).sum()),
            "n": int(len(d)),
            "p": t["p"],
            "exact": bool(t["exact"]),
        }
        if with_ci:
            out["ci"] = paired_bootstrap_ci(d)
        return out

    d_mean = sign * (b - mat.mean(axis=1))
    return {
        "main": one(d_mean, with_ci=True),
        "per_seed": [one(sign * (b - mat[:, j]), with_ci=False)
                     for j in range(mat.shape[1])],
    }
