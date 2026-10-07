# Australia: current paper

The current study covers 12 Australian regions and 173 substations. All four
task comparisons, including connection, use all 12 regions. Reference-dependent
diagnostics use the 10 regions that pass the reference-allocation screen.

The four main task comparisons favor LU over GNN-based allocation. Connection
outputs are reinforcement requirements in MVA-equivalent units under PF=1.
They are not monetary costs. This supersedes the earlier lu5 comparison with
only 10 connection regions and the opposite aggregate connection direction.

The numbered scripts display reconstruction, siting and sizing, connection,
scale and conditional bounds. Run them independently from the repository root:

```bash
python StudyCase/Placement/Australia/003_connection.py
```

Protocol and region counts are recorded in
`study_materials/placement/current/frozen/config.json` and `au_region_support.csv`.
See the [current reproduction guide](../current/README.md).
