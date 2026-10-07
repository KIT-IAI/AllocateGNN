# Britain: current paper

The current study covers 16 British regions, 1,891 substations and 799,346 grid
cells. The four task comparisons include all 16 regions. Reference-dependent
analyses use the 13 regions that pass the reference-allocation screen.

At a 10 km radius and a 300 MW incoming load, mean fixed-location cost error
decreases from GBP 11.59 million for LU to GBP 8.63 million for GNN-based
allocation. All 16 regions improve. Siting and sizing have mixed directions.

The numbered scripts display reconstruction, siting and sizing, connection,
scale, conditional bounds and the diagnostic panel, respectively. They can
be run independently from the repository root:

```bash
python StudyCase/Placement/British/003_connection.py
```

Protocol and region counts are recorded in
`study_materials/placement/current/frozen/config.json` and `uk_region_support.csv`.
See the [current reproduction guide](../current/README.md).
