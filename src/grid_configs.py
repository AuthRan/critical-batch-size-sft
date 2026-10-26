"""Write the main-grid configs from the scout columns (PLAN.md section 4, centring rule of 2026-10-24).

  python src/grid_configs.py runs/ledger_remote_snapshot.jsonl            # pass 1 + noise runs, every scouted setting
  python src/grid_configs.py runs/ledger_remote_snapshot.jsonl --seeds    # pass 2, once a setting's pass 1 is complete

Pass 1, per setting whose B = 4,096 scout column is complete: lr*(4096) from that column (one seed, the same 3-point
vertex as analyze.py). If its minimum sits at the bottom (top) of the column, only the scout extension is written, the
next two 1/3-decade points below (above), and pass 1 waits for it. Otherwise grid_<model>_<dataset>.json: 7 B x 7 lrs
at the 1/3-decade points, centred on the point nearest lr*(4096) x (B / 4096)^0.5, seed 1 (the B = 4,096 column reuses
the scout runs where the lrs coincide), and noise_<model>_<dataset>.json: B = 4,096 at lr = lr*(4096), 1 seed.
Pass 2: seeds 2 and 3 at the argmin and its two neighbours of each B column. A column whose minimum is at its edge gets
two more lrs beyond that edge first (grid_..._ext.json, seed 1) and its seeds on the next pass.
The pilot setting (SmolLM2-360M x dolly) has its own grid with 3 seeds everywhere and is skipped.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from analyze import fit_lr_star, read_ledger
from train import make_run_id

BS = [512, 1024, 2048, 4096, 8192, 16384, 32768]
B_SCOUT = 4096
PILOT = ("HuggingFaceTB/SmolLM2-360M", "dolly")
MANT = [1.0, 2.2, 4.6]  # the 1/3-decade points 1, 2.2, 4.6 x 10^k
NOISE = {"noise_scale_at": "0,0.25,0.5,0.75,1", "noise_scale_tokens": 65536, "noise_scale_big_batches": 4,
         "noise_scale_micro_per_big": 8}


def point(i: int) -> float:
    """The i-th 1/3-decade point: point(-15) = 1e-5, point(-14) = 2.2e-5, point(-13) = 4.6e-5."""
    return float(f"{MANT[i % 3]}e{i // 3}")


def index_of(lr: float) -> int:
    """Index of the 1/3-decade point nearest lr in log space."""
    i = round(3 * math.log10(lr))
    return min((i - 1, i, i + 1), key=lambda j: abs(math.log10(point(j) / lr)))


def column(recs: list[dict], model: str, dataset: str, B: int, extra: dict, seed: int = 1) -> dict[float, float]:
    """{lr: held-out loss} of the main-variant runs at one B and seed: run_id rebuilt from the config's flags, so the
    clip, beta2, branch and fixed-step controls never enter."""
    out = {}
    for r in recs:
        a = r["args"]
        if (a["model"], a["dataset"], int(a["tokens_per_step"]), int(a["seed"])) == (model, dataset, B, seed) \
                and r["run_id"] == make_run_id(model, dataset, B, a["lr"], seed, extra):
            out[float(a["lr"])] = float(r["holdout_loss"])
    return out


def minimum(col: dict[float, float]) -> tuple[float, str, int]:
    """(lr*, flag, side): side -1 / +1 when the argmin is the lowest / highest lr of the column, else 0."""
    lrs = sorted(col)
    losses = [col[lr] if math.isfinite(col[lr]) else float("inf") for lr in lrs]  # diverged ranks last (PLAN 2026-10-21)
    k = int(np.argmin(losses))
    side = -1 if k == 0 else (1 if k == len(lrs) - 1 else 0)
    lr_star, _, flag = fit_lr_star(lrs, losses)
    return lr_star, flag, side


def beyond(col: dict[float, float], side: int) -> list[float]:
    """The next two 1/3-decade points past the column's lowest (side -1) or highest (side +1) lr."""
    i = index_of(min(col) if side < 0 else max(col))
    return [point(i + side), point(i + 2 * side)]


def write(ledger: str, scout_cfg: str = "configs/scout.json", out_dir: str = "configs", seeds_pass: bool = False) -> list[str]:
    scout = json.load(open(scout_cfg, encoding="utf-8"))
    recs = [r for r in read_ledger(ledger) if r.get("holdout_loss") is not None]
    log = []

    def dump(name: str, cfg: dict):
        p = Path(out_dir) / f"{name}.json"
        p.write_text(json.dumps(cfg, indent=2) + "\n", encoding="utf-8")
        log.append(str(p))

    for model in scout["models"]:
        extra = {**scout.get("common", {}), **scout.get("per_model", {}).get(model, {})}
        for dataset in scout["datasets"]:
            if (model, dataset) == PILOT:
                continue
            short = f"{model.split('/')[-1]}_{dataset}"
            base = {"ledger": "runs/ledger.jsonl", "models": [model], "datasets": [dataset]}
            sc = column(recs, model, dataset, B_SCOUT, extra)
            if not all(lr in sc for lr in scout["lr"]):
                log.append(f"{short}: scout column {len(sc)}/{len(scout['lr'])}, skipped")
                continue
            lr4, flag, side = minimum(sc)
            if side:
                dump(f"scout_ext_{short}", {**base, "tokens_per_step": [B_SCOUT], "lr": beyond(sc, side), "seeds": [1],
                                            "common": extra})
                continue
            if not seeds_pass:
                if (Path(out_dir) / f"grid_{short}.json").exists():  # written once: later B = 4,096 runs would move lr*(4096)
                    log.append(f"{short}: grid config exists, kept")
                    continue
                lr_by_B = {str(B): [point(index_of(lr4 * (B / B_SCOUT) ** 0.5) + d) for d in range(-3, 4)] for B in BS}
                dump(f"grid_{short}", {**base, "tokens_per_step": BS, "lr_by_B": lr_by_B, "seeds": [1], "common": extra,
                                       "lr_star_4096": float(f"{lr4:.3e}"), "lr_star_4096_flag": flag})
                noise = {k: v for k, v in extra.items() if k != "cross_eval"}
                dump(f"noise_{short}", {**base, "ledger": "runs/noise2_ledger.jsonl", "tokens_per_step": [B_SCOUT],
                                        "lr": [float(f"{lr4:.3e}")], "seeds": [1], "common": {**noise, **NOISE}})
                continue
            grid = json.load(open(Path(out_dir) / f"grid_{short}.json", encoding="utf-8"))["lr_by_B"]
            seeds_by_B, ext_by_B = {}, {}
            for B in BS:
                col = column(recs, model, dataset, B, extra)
                if not all(lr in col for lr in grid[str(B)]):
                    break
                _, _, side = minimum(col)
                if side:
                    ext_by_B[str(B)] = beyond(col, side)
                    continue
                lrs = sorted(col)
                k = int(np.argmin([col[lr] if math.isfinite(col[lr]) else float("inf") for lr in lrs]))
                seeds_by_B[str(B)] = lrs[k - 1 : k + 2]
            else:
                if ext_by_B:
                    dump(f"grid_{short}_ext", {**base, "tokens_per_step": [int(b) for b in ext_by_B], "lr_by_B": ext_by_B,
                                               "seeds": [1], "common": extra})
                dump(f"grid_{short}_seeds", {**base, "tokens_per_step": [int(b) for b in seeds_by_B],
                                             "lr_by_B": seeds_by_B, "seeds": [2, 3], "common": extra})
                continue
            log.append(f"{short}: pass 1 incomplete at B = {B}, no seeds config")
    return log


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ledger")
    ap.add_argument("--scout", default="configs/scout.json")
    ap.add_argument("--out", default="configs")
    ap.add_argument("--seeds", action="store_true", help="pass 2: seeds 2 and 3 near each column's minimum")
    a = ap.parse_args()
    print("\n".join(write(a.ledger, a.scout, a.out, a.seeds)))
