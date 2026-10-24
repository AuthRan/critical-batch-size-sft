# The Critical Batch Size of Supervised Fine-Tuning

See `PLAN.md` for the question, the pre-registered hypotheses, the design and the timeline.

## Setup on the GPU machine (Windows, 2x RTX 2080 Ti, over SSH)

From the laptop, copy the project (type the account password when asked):

```
scp -r "D:\Research paper\Paper 2" <user>@<gpu-host>:<home>/paper2_sft_cbs
```

Then on the GPU machine (`ssh <user>@<gpu-host>`):

```
cd <home>\paper2_sft_cbs
setup_remote.cmd
```

`setup_remote.cmd` creates a venv that inherits the system CUDA torch, installs the rest, runs the tests, and runs the 30-step smoke config on GPU 0. Then launch the pilot on both GPUs as detached processes that survive logging out:

```
powershell -ExecutionPolicy Bypass -File launch.ps1 -Config "configs/pilot.json,configs/pilot_lowlr.json,configs/noise_pilot.json,configs/control_beta2.json,configs/control_clip.json" -Gpu 0 -Shard 0/2
powershell -ExecutionPolicy Bypass -File launch.ps1 -Config "configs/pilot.json,configs/pilot_lowlr.json,configs/noise_pilot.json,configs/control_beta2.json,configs/control_clip.json" -Gpu 1 -Shard 1/2
powershell -Command "Get-Content runs\pilot_gpu0.log -Tail 20 -Wait"
```

Configs separated by commas run back to back (here the pilot, then its low-lr extension). If a check-in finds the GPUs idle with the logs stale, the processes were ended from the console: re-run the same two launch lines, the sweep skips every run already in the ledger.

`push.ps1` on the laptop re-syncs code and configs. The sweep starts each job as a fresh `train.py` process, so code pushed while a sweep is running applies to its later jobs: push only changes that do not alter what is measured (logging, memory) while a grid is in progress.

Copy results back to the laptop with:

```
scp <user>@<gpu-host>:<home>/paper2_sft_cbs/runs/ledger.jsonl "D:\Research paper\Paper 2\runs\ledger_remote_snapshot.jsonl"
scp <user>@<gpu-host>:<home>/paper2_sft_cbs/runs/noise_ledger.jsonl "D:\Research paper\Paper 2\runs\noise_ledger_remote_snapshot.jsonl"
```

## Setup on any other machine

```
python -m venv .venv && .venv\Scripts\activate        # Windows; on Linux: source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
python -m pytest tests -q                              # CPU-only, no downloads, ~1 min
```

## Smoke run (downloads SmolLM2-135M and dolly-15k, 30 steps)

```
python src/sweep.py configs/smoke.json
```

## Pilot grid (SmolLM2-360M x dolly, 7 batch sizes x 7 learning rates x 3 seeds)

```
python src/sweep.py configs/pilot.json configs/pilot_lowlr.json   # resumable; re-run to continue
python src/analyze.py runs/ledger.jsonl --out runs/analysis
```

## One run by hand

```
python src/train.py --model HuggingFaceTB/SmolLM2-135M --dataset dolly --tokens_per_step 4096 --lr 1e-4 --seed 1 --noise_scale_at 0,0.5,1
```

## Layout

- `src/data.py` — corpora, tokenisation, exact token-count step planning (no packing)
- `src/noise_scale.py` — gradient noise scale estimator (McCandlish App. A, least-squares form)
- `src/train.py` — one run; writes one JSON line to `runs/ledger.jsonl`
- `src/sweep.py` — resumable grid runner
- `src/analyze.py` — lr*(B) fits, exponent, fixed-data CBS, plot
- `tests/` — CPU tests with a tiny random model
- `notes/` — reading notes, one file per paper
