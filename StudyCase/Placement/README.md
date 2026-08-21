# Task- and scale-matched placement evaluation

This case-study directory documents the British and Australian experiments for
the forthcoming manuscript "Task- and scale-matched evaluation of spatial
allocation proxies for network planning". Reusable methods live in the
top-level `SpatialPlacement/` package.

The numbered scripts are the paper-facing demonstration layer. They select the
case, assemble the relevant frozen evidence, call reusable `SpatialPlacement`
functions, and print the reported JSON summaries. No mathematical or
statistical implementation is duplicated here.

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
python StudyCase/Placement/000_reproduce_all.py --verify
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

## Repository map

- `SpatialPlacement/`: reusable mathematical, optimisation, and evaluation code.
- `StudyCase/Placement/British/` and `Australia/`: case roles and data contracts.
- `StudyCase/Placement/000_reproduce_all.py`: complete paper-facing entry point.
- `study_materials/placement/`: manuscript-scoped frozen numerical evidence.
- `tests/SpatialPlacement/`: formula, manifest, and headline regression gates.
