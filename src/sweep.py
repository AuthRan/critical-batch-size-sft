"""Run a grid of train.py jobs, skipping any run_id already in the ledger.

  python src/sweep.py configs/pilot.json
  python src/sweep.py configs/pilot.json --shard 0/2     # on GPU 0, with --shard 1/2 on GPU 1
  python src/sweep.py configs/pilot.json configs/pilot_lowlr.json --shard 0/2   # configs run back to back
  python src/sweep.py configs/grid_a.json --shard queue     # on both GPUs: each takes the next unclaimed job
  python src/sweep.py configs/queue.txt --shard queue       # configs listed in a file that can grow while it runs

Each job is a subprocess so that one crash (e.g. OOM) does not take the sweep down.
Re-running the same command resumes where it left off.
"""
from __future__ import annotations

import itertools
import json
import os
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


def pid_alive(pid: int) -> bool:
    if os.name == "nt":  # os.kill(pid, 0) would terminate the process on Windows
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32")
        k32.OpenProcess.restype = wintypes.HANDLE
        k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        k32.CloseHandle.argtypes = [wintypes.HANDLE]
        h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return False
        code = wintypes.DWORD()
        ok = k32.GetExitCodeProcess(h, ctypes.byref(code))
        k32.CloseHandle(h)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        pass
    return True


def claim(run_id: str, claims: Path) -> bool:
    """Take a job for this sweep (queue mode); False if another sweep holds it. The claim file holds this pid."""
    claims.mkdir(parents=True, exist_ok=True)
    try:
        fd = os.open(claims / f"{run_id}.claim", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    return True


def clear_stale_claims(claims: Path):
    """A claim left by a sweep that is no longer running frees its job (a stopped or failed run is retried)."""
    for p in claims.glob("*.claim"):
        try:
            if not pid_alive(int(p.read_text())):
                p.unlink()
        except (OSError, ValueError):  # ValueError: another sweep is writing its claim right now
            pass


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
    queue = shard == "queue"  # shared queue: every sweep walks the same list and claims jobs, so a faster GPU takes more
    me, n_shards = (0, 1) if queue else (int(x) for x in shard.split("/"))
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
    claims = Path(ledger).parent / "claims"
    print(f"{len(jobs)} jobs to run {'from the shared queue' if queue else f'on shard {me}/{n_shards}'}, "
          f"{len(done)} already in {ledger}", flush=True)
    tried: set[str] = set()
    while True:
        if queue and not dry_run:
            # every pick re-reads the ledger and frees the claims of stopped sweeps, so another sweep's finished,
            # failed or abandoned jobs are seen; a job that failed here is left to another sweep, not retried by this one
            clear_stale_claims(claims)
            done = done_ids(ledger)
            job = next((j for j in jobs if j[0] not in done and j[0] not in tried and claim(j[0], claims)), None)
        else:
            job = next((j for j in jobs if j[0] not in tried), None)
        if job is None:
            break
        run_id, model, dataset, B, lr, seed, extra = job
        tried.add(run_id)
        cmd = [sys.executable, str(Path(__file__).resolve().parent / "train.py"),
               "--model", model, "--dataset", dataset, "--tokens_per_step", str(B), "--lr", str(lr),
               "--seed", str(seed), "--ledger", ledger, "--run_id", run_id] + flags(extra)
        print(f"[{jobs.index(job) + 1}/{len(jobs)}] {run_id}", flush=True)
        if dry_run:
            continue
        r = subprocess.run(cmd)
        if r.returncode != 0:
            print(f"  FAILED (exit {r.returncode}); continuing", flush=True)
            if queue:  # e.g. this GPU is out of memory because someone else is using it: another GPU may run it
                (claims / f"{run_id}.claim").unlink(missing_ok=True)


def expand(args: list[str]) -> list[str]:
    """Config paths; a .txt argument lists configs, one per line (# for comments)."""
    out: list[str] = []
    for a in args:
        if a.endswith(".txt"):
            out += [s for s in (line.split("#")[0].strip() for line in open(a, encoding="utf-8")) if s]
        else:
            out.append(a)
    return out


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
    # several configs run back to back against their own ledgers; a .txt list is re-read each time a config finishes,
    # so configs can be added to it (or reordered) while the sweep runs
    finished: set[str] = set()
    while (cfg := next((c for c in expand(cfgs) if c not in finished), None)) is not None:
        finished.add(cfg)
        try:
            main(cfg, dry_run=dry, shard=shard)
        except (OSError, json.JSONDecodeError) as e:  # a config not pushed yet, or caught mid-copy: skip, keep going
            print(f"{cfg}: skipped ({e})", flush=True)
