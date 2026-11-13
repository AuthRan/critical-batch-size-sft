# Raw results

Copies of the run ledgers from the GPU machine, taken on 2026-11-09 after the last run finished (2026-11-07 17:47). One JSON object per finished run: its arguments, held-out loss, cross-corpus losses, loss curve, gradient-norm statistics and, for noise runs, the noise-scale measurements.

- `ledger.jsonl`: main grids, seeds, extensions and controls (1,059 rows)
- `branch_ledger.jsonl`: branched-CBS base runs and branches (365 rows)
- `noise2_ledger.jsonl`: noise-scale runs with the v2 estimator (PLAN.md section 10, item 2)
- `noise_ledger_v1.jsonl`: the v1 noise-scale runs, kept as the record of the estimator failure; used for no hypothesis
- `smoke_ledger.jsonl`: the smoke runs
- `confirm_ledger.jsonl`: the grid runs of the confirmatory study (PLAN.md section 11; 276 rows), copied on 2026-11-13 after its last run (2026-11-13 00:45). Appended to `ledger.jsonl` it gives the GPU machine's ledger.
- `confirm_noise_ledger.jsonl`: the noise runs of the confirmatory study (12 rows: one per new corpus and one repeat per original setting), the rows that follow `noise2_ledger.jsonl` in the GPU machine's file

`src/analyze.py` keeps the first row per run_id (see `read_ledger`). Reproduce the tables with
`python src/analyze.py data/ledger.jsonl --branch_ledger data/branch_ledger.jsonl --noise_ledger data/noise2_ledger.jsonl --out runs/analysis`.

`python src/confirm.py` scores the confirmatory predictions C1 to C4 (also run by `python src/paper.py`).
