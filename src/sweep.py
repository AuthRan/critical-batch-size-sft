"""Run a grid of train.py jobs, skipping any run_id already in the ledger.

  python src/sweep.py configs/pilot.json
  python src/sweep.py configs/pilot.json --shard 0/2     # on GPU 0, with --shard 1/2 on GPU 1
  python src/sweep.py configs/pilot.json configs/pilot_lowlr.json --shard 0/2   # configs run back to back

Each job is a subprocess so that one crash (e.g. OOM) does not take the sweep down.
Re-running the same command resumes where it left off.
"""
from __future__ import annotations

import itertools
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train import make_run_id  # noqa: E402


def done_ids(ledger: str) -> set[str]:
    if not Path(ledger).exists():
        return set()
    ids = set()
    for line in open(ledger, encoding="utf-8"):
        try:
            ids.add(json.loads(line)["run_id"])
        except Exception:
            pass
    return ids


def flags(d: dict) -> list[str]:
    out: list[str] = []
    for k, v in d.items():
        if isinstance(v, bool):
            if v:
                out.append(f"--{k}")
        else:
            out += [f"--{k}", str(v)]
    return out


def main(cfg_path: str, dry_run: bool = False, shard: str = "0/1"):
    me, n_shards = (int(x) for x in shard.split("/"))
    cfg = json.load(open(cfg_path, encoding="utf-8"))
    ledger = cfg.get("ledger", "runs/ledger.jsonl")
    common = cfg.get("common", {})
    per_model = cfg.get("per_model", {})
    done = done_ids(ledger)
    jobs = []
    lr_by_B = cfg.get("lr_by_B", {})  # optional {"B": [lrs]}: a grid that follows lr*(B) instead of one lr list
    points = [(m, d, B, lr, s) for m, d, B in itertools.product(cfg["models"], cfg["datasets"], cfg["tokens_per_step"])
              for lr in lr_by_B.get(str(B), cfg.get("lr", [])) for s in cfg["seeds"]]
    for model, dataset, B, lr, seed in points:
        extra = {**common, **per_model.get(model, {})}
        jobs.append((make_run_id(model, dataset, B, lr, seed, extra), model, dataset, B, lr, seed, extra))
    # smallest B first: same FLOPs, worst wall-clock, so get them going
    jobs.sort(key=lambda j: (j[3], j[1], j[2], j[4], j[5]))
    # shard the full grid, then skip finished runs: a job's shard never depends on when a shard was (re)launched,
    # so shards started at different times cannot run the same job (changed 2026-10-24; before, it sharded the remainder)
    jobs = [j for k, j in enumerate(jobs) if k % n_shards == me and j[0] not in done]
    print(f"{len(jobs)} jobs to run on shard {me}/{n_shards}, {len(done)} already in {ledger}", flush=True)
    for k, (run_id, model, dataset, B, lr, seed, extra) in enumerate(jobs, 1):
        cmd = [sys.executable, str(Path(__file__).resolve().parent / "train.py"),
               "--model", model, "--dataset", dataset, "--tokens_per_step", str(B), "--lr", str(lr),
               "--seed", str(seed), "--ledger", ledger, "--run_id", run_id] + flags(extra)
        print(f"[{k}/{len(jobs)}] {run_id}", flush=True)
        if dry_run:
            continue
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"  FAILED (exit {r.returncode}); continuing", flush=True)


if __name__ == "__main__":
    argv = sys.argv[1:]
    shard = "0/1"
    if "--shard" in argv:
        i = argv.index("--shard")
        shard = argv[i + 1]
        del argv[i : i + 2]
    dry = "--dry-run" in argv
    cfgs = [a for a in argv if not a.startswith("--")]
    if not cfgs:
        sys.exit("usage: sweep.py CONFIG [CONFIG ...] [--shard i/n] [--dry-run]")
    for cfg in cfgs:  # several configs run back to back against their own ledgers
        main(cfg, dry_run=dry, shard=shard)
