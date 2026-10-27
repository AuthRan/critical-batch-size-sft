# The Critical Batch Size of Supervised Fine-Tuning

See `PLAN.md` for the question, the pre-registered hypotheses, the design and the timeline.

## Setup on the GPU machine (Windows, 2x RTX 2080 Ti, over SSH)

The project lives in `G:\paper2_sft_cbs` and the Hugging Face cache in `G:\hf_cache` (`HF_HOME`), both moved off C: on 2026-10-27. The laptop logs in with the key `~/.ssh/id_ed25519_<gpu-host>` (a `Host <gpu-host>` entry in `~/.ssh/config`), so `ssh` and `scp` ask no password. From the laptop, copy the project:

```
scp -r "D:\Research paper\Paper 2" <user>@<gpu-host>:G:/paper2_sft_cbs
```

Then on the GPU machine (`ssh <user>@<gpu-host>`):

```
cd /d G:\paper2_sft_cbs
setup_remote.cmd
```

`setup_remote.cmd` creates a venv that inherits the system CUDA torch, installs the rest, runs the tests, and runs the 30-step smoke config on GPU 0. Then launch the pilot on both GPUs as detached processes that survive logging out:

```
powershell -ExecutionPolicy Bypass -File launch.ps1 -Config "configs/pilot.json,configs/pilot_lowlr.json,configs/noise_pilot.json,configs/control_beta2.json,configs/control_clip.json" -Gpu 0 -Shard 0/2
powershell -ExecutionPolicy Bypass -File launch.ps1 -Config "configs/pilot.json,configs/pilot_lowlr.json,configs/noise_pilot.json,configs/control_beta2.json,configs/control_clip.json" -Gpu 1 -Shard 1/2
powershell -Command "Get-Content runs\pilot_gpu0.log -Tail 20 -Wait"
```

Check on both queues with `powershell -ExecutionPolicy Bypass -File status.ps1`. A log tail can be hours old, so a queue is alive only if its venv python is listed and its log was written minutes ago (on 2026-10-25 both queues had died at 00:40 while their tails still looked like runs in progress).

Launches go through `src/detach.py`: Windows OpenSSH ends every process a session started when the session closes, and before 2026-10-27 that ended the sweeps whenever the SSH window that started them disconnected. If another lab user's job is holding a GPU, `launch_when_free.ps1` (same arguments as `launch.ps1`) waits until that GPU's memory use stays under 1.5 GB, then launches.

Configs separated by commas run back to back (here the pilot, then its low-lr extension). If a check-in finds the GPUs idle with the logs stale, the processes were ended from the console: re-run the same two launch lines, the sweep skips every run already in the ledger.

To stop both GPUs (for instance to re-queue them): stop any `launch_after` waiters by the pid it printed (`Stop-Process -Id <pid>`), then every python process started from this folder's venv, which also ends the interpreters it started: `powershell -Command "Get-Process python | Where-Object { $_.Path -like '*paper2_sft_cbs*' } | Stop-Process -Force"`. Run that twice, a few seconds apart: a sweep whose run is stopped before the sweep itself can start its next run in between (seen 2026-10-24). A stopped run writes no ledger row, so the next launch re-runs it from the start.

`push.ps1` on the laptop re-syncs code and configs. The sweep starts each job as a fresh `train.py` process, so code pushed while a sweep is running applies to its later jobs: push only changes that do not alter what is measured (logging, memory) while a grid is in progress.

Copy results back to the laptop with:

```
scp <user>@<gpu-host>:G:/paper2_sft_cbs/runs/ledger.jsonl "D:\Research paper\Paper 2\runs\ledger_remote_snapshot.jsonl"
scp <user>@<gpu-host>:G:/paper2_sft_cbs/runs/noise2_ledger.jsonl "D:\Research paper\Paper 2\runs\noise2_ledger_remote_snapshot.jsonl"
```

`noise_ledger.jsonl` holds the v1 noise-scale estimator's runs, which are not used (PLAN.md section 10, deviation 2); `noise2_ledger.jsonl` holds the corrected estimator's. Analyse with `python src/analyze.py runs/ledger_remote_snapshot.jsonl --noise_ledger runs/noise2_ledger_remote_snapshot.jsonl`.

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

## Main grids (8 settings, from the scout columns)

When `configs/scout.json` has finished, pull the ledger, then on the laptop:

```
python src/grid_configs.py runs/ledger_remote_snapshot.jsonl
```

That writes, per setting, `configs/grid_<model>_<dataset>.json` (7 B x 7 lrs centred on lr*(4096) x (B / 4096)^0.5, seed 1) and `configs/noise_<model>_<dataset>.json` (B = 4,096 at lr*(4096)), or `configs/scout_ext_<model>_<dataset>.json` if the scout minimum sits at the edge of its column. Push, then launch both GPUs on the same config list with `-Shard queue`: each sweep takes the next job that no other sweep has claimed (claim files in `runs\claims`), so the faster GPU does more and neither idles at a config boundary. A claim left by a stopped sweep is cleared when the next sweep starts, so relaunching retries that job. When a setting's grid is complete, `python src/grid_configs.py runs/ledger_remote_snapshot.jsonl --seeds` writes seeds 2 and 3 at the three lrs nearest each column's minimum, plus a `_ext` config for any column whose minimum is at its edge.

## One run by hand

```
python src/train.py --model HuggingFaceTB/SmolLM2-135M --dataset dolly --tokens_per_step 4096 --lr 1e-4 --seed 1 --noise_scale_at 0,0.5,1
```

## Branched CBS (Merrill et al. 2025 protocol, PLAN.md section 4)

```
python src/branch_configs.py --model HuggingFaceTB/SmolLM2-360M --dataset dolly --lr0 2.34e-5   # lr0 = lr*(1024) of that setting
```

That writes `configs/branch_<model>_<dataset>_base.json`, four checkpoint configs (`f0.00` to `f0.75`, 25% windows) and one Delta-sensitivity config (`f0.50_d0.125`). Queue the base config well ahead of the branch configs (base, then other work, then branches) on both GPUs. The base run saves its checkpoints to `runs/ckpt/<model>_<dataset>/`. A branch waits at most 1 h for a missing checkpoint and then fails, so a base run that could not save (full disk, crash) costs an hour, not a night. A checkpoint holds the weights plus AdamW state, about 4.3 GB at 360M. When `runs/branch_ledger.jsonl` has all 90 branch rows for a setting, delete its folder (`rmdir /s /q runs\ckpt\SmolLM2-360M_dolly`) before starting the next setting. `analyze.py` reads the branch ledger and adds the branched-CBS table.

## Layout

- `src/data.py` — corpora, tokenisation, exact token-count step planning (no packing)
- `src/noise_scale.py` — gradient noise scale estimator (McCandlish App. A, least-squares form)
- `src/train.py` — one run; writes one JSON line to `runs/ledger.jsonl`
- `src/sweep.py` — resumable grid runner (`--shard i/n`, or `--shard queue` for a shared queue)
- `src/analyze.py` — lr*(B) fits, exponent, fixed-data CBS, branched CBS, plot
- `src/branch_configs.py` — writes the base and branch configs for one setting
- `src/grid_configs.py` — writes the main-grid, noise and seed-repeat configs from the scout columns
- `tests/` — CPU tests with a tiny random model
- `notes/` — reading notes, one file per paper
