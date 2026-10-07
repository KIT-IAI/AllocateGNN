# Current placement-paper experiments

This directory supports **Task- and Scale-Matched Evaluation of Spatial Demand
Allocation for Power Grid Planning**. Its numerical release is
`fixedload_20260925_r2`, based on `refactor_20260922_r2` and the paper's
fixed-load connection calculations.

From the repository root:

```bash
pip install -r StudyCase/Placement/current/requirements-current.txt
python StudyCase/Placement/000_reproduce_all.py --verify
```

The command recomputes the reported statistics from frozen derived result
tables and checks the current release. Outputs go to
`results/placement_current_reproduction/`.

The [current reproduction guide](current/README.md) describes the source
versions, figure commands, data requirements and verification scope.

## Experiments

- [Britain](British/README.md): 16 regions, including monetary connection-cost error.
- [Australia](Australia/README.md): 12 regions, with PF=1 reinforcement requirements.
- Peak-demand reconstruction, substation siting, rule-based sizing and connection
  are evaluated separately. Siting determines the locations used for sizing.
  The new data-center load enters the connection task only.

The numbered case scripts display the current results. They do not imply that
the four evaluations form a sequential construction project.

## Source layout

- `current/upstream/`: exported public `sglib` source, case configurations and tests.
- `current/scripts/`: paper-specific calculations and figure generators.
- `current/SOURCE.json`: exact source versions and file hashes.
- [Current numerical evidence](../../study_materials/placement/current/README.md):
  the public subset and its manifest.
- `SpatialPlacement/current_paper.py` at the repository root: public reproduction adapter.

## Historical lu5 release

The original lu5 evidence remains under `study_materials/placement/` outside
its `current/` subdirectory. It describes an earlier manuscript and must not be
used to verify the current paper. Its command is:

```bash
python -m SpatialPlacement.reproduce_legacy_paper_numbers --verify
```
