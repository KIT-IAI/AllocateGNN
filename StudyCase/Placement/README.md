# Task- and scale-matched placement evaluation

This package contains the country-independent methods and derived-result
reproduction workflow for the forthcoming manuscript "Task- and scale-matched
evaluation of spatial allocation proxies for network planning".

## What is included

- Peak reconstruction on existing-substation Voronoi cells.
- Demand-weighted p-median siting with greedy/local-exchange and optional MILP implementations.
- Rule-based sizing and rule-based sizing deviation (RSD).
- A headroom-based connection proxy, fixed-site error, top-set regret, and rectangular uncertainty bounds.
- PERM/SMOOTH diagnostic fields and regional statistical utilities.
- A manuscript-scoped numerical package under `study_materials/placement`.

## Reproduce the reported numbers

From the repository root:

```powershell
python -m StudyCase.Placement.scripts.reproduce_paper_numbers --verify
```

The output is written to `results/placement_reproduction/paper_numbers.json`.

## Reproducibility boundary

The command reproduces the reported tables, statistics, and quantitative
figure inputs from twelve frozen derived CSV files. Raw source data, the lu5
held-out fields, and training checkpoints are not distributed and were not
retained in the source workspace. Consequently, this release does not claim
end-to-end GNN retraining or held-out-field regeneration.

`model_contract.json` records the missing-artifact boundary. The LU-sharp
alpha range is explicitly marked `frozen_summary_only` because its per-region
lu5 artifact is unavailable.

## Package map

- `core/`: country-independent mathematical and optimisation methods.
- `pipeline/`: candidate generation, metrics, p-median, sizing, and statistics.
- `British/` and `Australia/`: case roles and data contracts.
- `scripts/`: frozen-result numerical reproduction.
- `tests/`: formula, manifest, and headline regression gates.
