"""Keep the queues fed while nobody can reach the GPU machine (added 2026-10-28, before a 2-day absence).

  python src/detach.py runs/autoplan.log runs/autoplan.err -- .venv/Scripts/python src/autoplan.py --every 1200

Every --every seconds, on the live ledger:
- runs grid_configs.py (pass 1, then the seeds pass) and queues every config it writes whose content is new. The copy
  goes to configs/auto/<name>__<n>.json, because a running sweep never re-runs a list entry it has finished, and
  grid_configs rewrites the same file names (an extension still at its edge, a seeds config that gains columns);
- once SmolLM2-135M x dolly has seeds 1 to 3 at the argmin of its B = 1,024 column and both neighbours, writes its
  branched-CBS configs (branch_configs.py, lr0 = that cell's lr*, the same seed-averaged 3-point fit as analyze.py)
  and queues them for GPU 0 only, base first, so that no branch waits on a base run another GPU has not reached.

New lines go just above the marker line of a list (MARKER), or at its end without one. SmolLM2-135M work goes to
both lists (GPU 0 can only hold 135M while something else keeps 2.2 GB on it), the rest to configs/queue.txt only.
Lists and configs are written to a temporary file and renamed, so a sweep never reads half a file.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path

import numpy as np

import branch_configs
import grid_configs
from analyze import cell_fit, read_ledger, seed_mean

MARKER = "# auto: new configs go above this line"
QUEUE, QUEUE_135M = "configs/queue.txt", "configs/queue_135m.txt"
BRANCH = ("HuggingFaceTB/SmolLM2-135M", "dolly")
MAX_EXT = 3  # extensions per grid (2 lrs each), i.e. 2 decades past the original column


def write_atomic(path: Path, text: str):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def enqueue(list_path: str, lines: list[str]) -> list[str]:
    """Add the lines that the list does not already hold, above its marker (or at its end); returns those added."""
    p = Path(list_path)
    old = p.read_text(encoding="utf-8").splitlines()
    new = [s for s in lines if s not in old]
    if new:
        k = old.index(MARKER) if MARKER in old else len(old)
        write_atomic(p, "\n".join(old[:k] + new + old[k:]) + "\n")
    return new


def copies_of(cfg_path: Path, auto_dir: Path) -> list[Path]:
    return sorted(auto_dir.glob(f"{cfg_path.stem}__*.json"), key=lambda q: int(q.stem.rsplit("__", 1)[1]))


def snapshot_differs(cfg_path: Path, auto_dir: Path) -> bool:
    copies = copies_of(cfg_path, auto_dir)
    return not copies or json.loads(copies[-1].read_text(encoding="utf-8")) != json.loads(cfg_path.read_text(encoding="utf-8"))


def snapshot(cfg_path: Path, auto_dir: Path) -> str:
    """The newest configs/auto/<stem>__<n>.json, after adding a copy of cfg_path if its content differs from it."""
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    copies = copies_of(cfg_path, auto_dir)
    if copies and json.loads(copies[-1].read_text(encoding="utf-8")) == cfg:
        return copies[-1].as_posix()
    auto_dir.mkdir(parents=True, exist_ok=True)
    out = auto_dir / f"{cfg_path.stem}__{len(copies) + 1}.json"
    write_atomic(out, json.dumps(cfg, indent=2) + "\n")
    return out.as_posix()


def branch_lr0(recs: list[dict], model: str, dataset: str, extra: dict) -> float | None:
    """lr*(1024) once its argmin and both neighbours have 3 seeds and the fit is ok; else None."""
    by_lr: dict[float, list[float]] = {}
    for s in (1, 2, 3):
        for lr, loss in grid_configs.column(recs, model, dataset, 1024, extra, seed=s).items():
            by_lr.setdefault(lr, []).append(loss)
    lrs = sorted(by_lr)
    if len(lrs) < 7:
        return None
    k = int(np.argmin([seed_mean(by_lr[lr]) for lr in lrs]))
    if k in (0, len(lrs) - 1) or any(len(by_lr[lr]) < 3 for lr in lrs[k - 1 : k + 2]):
        return None
    fit = cell_fit(by_lr, np.random.default_rng(0))
    return float(f"{fit['lr_star']:.3e}") if fit["flag"] == "ok" else None


def once(ledger: str, configs: str = "configs") -> list[str]:
    log, auto = [], Path(configs) / "auto"
    written = grid_configs.write(ledger, out_dir=configs) + grid_configs.write(ledger, out_dir=configs, seeds_pass=True)
    # every pass re-offers every newest copy: enqueue skips what a list holds, and a pass that failed half-way is redone
    for line in written:
        if line.endswith(".json"):
            stem = Path(line).stem
            if "_ext" in stem and len(list(auto.glob(f"{stem}__*.json"))) >= MAX_EXT \
                    and snapshot_differs(Path(line), auto):  # a flat loss curve would extend forever: leave it to me
                log.append(f"{stem}: still at its edge after {MAX_EXT} extensions, not extended further")
                continue
            copy = snapshot(Path(line), auto)
            for q in [QUEUE_135M, QUEUE] if "SmolLM2-135M" in copy else [QUEUE]:
                log += [f"queued {s} in {q}" for s in enqueue(q, [copy])]
    model, dataset = BRANCH
    short = f"{model.split('/')[-1]}_{dataset}"
    if not (Path(configs) / f"branch_{short}_f0.50_d0.125.json").exists():  # the last of the six written
        scout = json.load(open(Path(configs) / "scout.json", encoding="utf-8"))
        extra = {**scout.get("common", {}), **scout.get("per_model", {}).get(model, {})}
        recs = [r for r in read_ledger(ledger) if r.get("holdout_loss") is not None]
        if (lr0 := branch_lr0(recs, model, dataset, extra)) is None:
            return log
        branch_configs.write(model, dataset, lr0, out_dir=configs)
        log.append(f"branch {short}: lr0 = {lr0:.3e}")
    branches = sorted(p.as_posix() for p in Path(configs).glob(f"branch_{short}_f*.json"))
    base = (Path(configs) / f"branch_{short}_base.json").as_posix()
    log += [f"queued {s} in {QUEUE_135M}" for s in enqueue(QUEUE_135M, [base] + branches)]
    return log


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default="runs/ledger.jsonl")
    ap.add_argument("--every", type=int, default=0, help="seconds between passes; 0 runs one pass")
    a = ap.parse_args()
    while True:
        try:  # a ledger line caught mid-write, say: log it and try again next pass
            for line in once(a.ledger):
                print(time.strftime("%m-%d %H:%M"), line, flush=True)
        except Exception:
            print(time.strftime("%m-%d %H:%M"), "pass failed:", traceback.format_exc(), flush=True)
        if not a.every:
            break
        time.sleep(a.every)
