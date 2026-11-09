# Raw results

Copies of the run ledgers from the GPU machine, taken on 2026-11-09 after the last run finished (2026-11-07 17:47). One JSON object per finished run: its arguments, held-out loss, cross-corpus losses, loss curve, gradient-norm statistics and, for noise runs, the noise-scale measurements.

- `ledger.jsonl`: main grids, seeds, extensions and controls (1,059 rows)
- `branch_ledger.jsonl`: branched-CBS base runs and branches (365 rows)
- `noise2_ledger.jsonl`: noise-scale runs with the v2 estimator (PLAN.md section 10, item 2)
- `noise_ledger_v1.jsonl`: the v1 noise-scale runs, kept as the record of the estimator failure; used for no hypothesis
- `smoke_ledger.jsonl`: the smoke runs

`src/analyze.py` keeps the first row per run_id (see `read_ledger`). Reproduce the tables with
`python src/analyze.py data/ledger.jsonl --branch_ledger data/branch_ledger.jsonl --noise_ledger data/noise2_ledger.jsonl --out runs/analysis`.
