"""Write the branched-CBS configs for one setting (PLAN.md section 4, protocol note of 2026-10-24).

  python src/branch_configs.py --model HuggingFaceTB/SmolLM2-360M --dataset dolly --lr0 2.34e-5

Writes configs/branch_<model>_<dataset>_base.json (B0 = 1,024 at lr0, checkpoints at 0/25/50/75%) and one config per
checkpoint: B = 1,024 to 32,768, lr = sqrt(B / B0) * lr0 * {1/2.15, 1, 2.15}, the next 25% of the epoch; plus the
Delta-sensitivity config (12.5% window at the 50% checkpoint). Queue the base config well before the branch configs
(e.g. base, then other work, then branches), so the checkpoints exist when the branches start. A branch waits at most
1 h for a missing checkpoint and then fails, so a base run that could not save costs an hour, not a night.

  python src/branch_configs.py --model HuggingFaceTB/SmolLM2-360M --dataset dolly --lr0 2.34e-5 --ext_from runs/branch_ledger.jsonl

Instead writes <window>_ext configs: the next 2 lrs past the sweep's edge, for each cell whose best lr is at that edge
(PLAN.md section 10, item 5). Run it again on the new results for another round; it rewrites the same file names.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from analyze import branch_table, read_ledger

B0 = 1024
BS = [1024, 2048, 4096, 8192, 16384, 32768]
MULTS = [1 / 2.15, 1.0, 2.15]
FRACS = [0.0, 0.25, 0.5, 0.75]


def write(model: str, dataset: str, lr0: float, out_dir: str = "configs", extra: dict | None = None) -> list[Path]:
    short = model.split("/")[-1]
    ck = f"runs/ckpt/{short}_{dataset}"
    common = {"epochs": 1.0, "clip": 0, "micro_token_budget": 1024, **(extra or {})}
    out = []

    def dump(name: str, cfg: dict):
        p = Path(out_dir) / f"branch_{short}_{dataset}_{name}.json"
        p.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        out.append(p)

    dump("base", {"ledger": "runs/branch_ledger.jsonl", "models": [model], "datasets": [dataset],
                  "tokens_per_step": [B0], "lr": [lr0], "seeds": [1],
                  "common": {**common, "save_at": ",".join(f"{f:g}" for f in FRACS), "save_dir": ck}})
    lr_by_B = {str(B): [float(f"{(B / B0) ** 0.5 * lr0 * m:.3e}") for m in MULTS] for B in BS}
    for f, delta in [(f, 0.25) for f in FRACS] + [(0.5, 0.125)]:
        dump(f"f{f:.2f}_d{delta:g}", {"ledger": "runs/branch_ledger.jsonl", "models": [model], "datasets": [dataset],
                                       "tokens_per_step": BS, "lr_by_B": lr_by_B, "seeds": [1],
                                       "common": {**common, "init_from": f"{ck}/f{f:.2f}.pt", "delta_frac": delta,
                                                  "wait_for_init_h": 1}})
    return out


def write_ext(model: str, dataset: str, lr0: float, rows: list[dict], out_dir: str = "configs", n: int = 2) -> list[Path]:
    """rows: analyze.branch_table. One config per window with a cell at its sweep's edge: the next n lrs past that edge."""
    short = model.split("/")[-1]
    ck = f"runs/ckpt/{short}_{dataset}"
    out = []
    for r in rows:
        if (r["model"], r["dataset"]) != (short, dataset) or not r["edge"]:
            continue
        lr_by_B = {}
        for B in r["edge"]:
            k = round(math.log(r["best_mult"][B]) / math.log(MULTS[-1]))  # the best lr is sqrt rule * 2.15^k
            step = 1 if k > 0 else -1
            lr_by_B[str(B)] = [float(f"{(B / B0) ** 0.5 * lr0 * MULTS[-1] ** (k + step * i):.3e}") for i in range(1, n + 1)]
        f, delta = r["base_frac"], r["delta_frac"]
        p = Path(out_dir) / f"branch_{short}_{dataset}_f{f:.2f}_d{delta:g}_ext.json"
        p.write_text(json.dumps({"ledger": "runs/branch_ledger.jsonl", "models": [model], "datasets": [dataset],
                                 "tokens_per_step": r["edge"], "lr_by_B": lr_by_B, "seeds": [1],
                                 "common": {"epochs": 1.0, "clip": 0, "micro_token_budget": 1024,
                                            "init_from": f"{ck}/f{f:.2f}.pt", "delta_frac": delta, "wait_for_init_h": 1}},
                                indent=2) + "\n", encoding="utf-8")
        out.append(p)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--lr0", type=float, required=True, help="lr*(1024) of this setting from the main grid")
    ap.add_argument("--extra", default="{}", help='JSON merged into "common", e.g. {"optim": "adamw8bit"}')
    ap.add_argument("--ext_from", help="a branch ledger: write the bracketing extensions of this setting instead")
    a = ap.parse_args()
    if a.ext_from:
        rows = branch_table([r for r in read_ledger(a.ext_from) if r.get("branch") and r.get("holdout_loss") is not None])
        paths = write_ext(a.model, a.dataset, a.lr0, rows)
    else:
        paths = write(a.model, a.dataset, a.lr0, extra=json.loads(a.extra))
    for p in paths:
        print(p)
