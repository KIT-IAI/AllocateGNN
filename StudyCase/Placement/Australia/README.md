# Australia case

Australia is a 12-region composite-register stress test; the comparative
connection task uses the 10 regions that pass the reference-support check.
Capacity-coupled outputs are PF=1 MVA-equivalent requirements, not monetary
costs. Raw Ausgrid inputs and lu5 held-out fields are not distributed.

See `study_materials/placement/case_facts/register_summary.json` for the
reported register facts and `protocol.json` for fixed settings.

## Execution order

1. `001_reconstruction.py` - peak reconstruction stress test.
2. `002_siting_sizing.py` - siting and both sizing-matching protocols.
3. `003_connection.py` - PF=1 connection-requirement comparison.
4. `004_scale_analysis.py` - station-centred geographic-support sweep.
5. `005_insensitivity_audit.py` - coverage and zero-bound audit.

Run a script from the repository root, for example:

```powershell
python StudyCase/Placement/Australia/003_connection.py
```
