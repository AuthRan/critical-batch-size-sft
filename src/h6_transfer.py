"""H6 as fixed in PLAN.md section 10, item 8: the gain from retuning lr for B, on the same corpus and the other two.

  python src/h6_transfer.py runs/ledger.jsonl runs/analysis/cells.json
"""
import json
import math
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from analyze import load


def main_optim(model: str) -> str:
    """Each grid's own stack (Qwen3 runs 8-bit AdamW); the SmolLM2-135M 8-bit control is not a main grid."""
    return "adamw8bit" if model.startswith("Qwen") else "adamw"


def gains(recs: list[dict], cells: dict) -> list[dict]:
    """One row per (model, dataset, B != 4096): reference lr, retuned lr, and the gain on every corpus."""
    runs = defaultdict(dict)  # (model, dataset) -> {(B, lr): {corpus: loss}}
    for r in recs:
        a = r["args"]
        m = a["model"].split("/")[-1]
        if a.get("optim", "adamw") != main_optim(m) or a.get("beta2_ref_tokens") or float(a.get("clip", 1.0)) != 0 or a["seed"] != 1:
            continue
        runs[(m, a["dataset"])][(int(a["tokens_per_step"]), float(a["lr"]))] = {a["dataset"]: r["holdout_loss"], **(r.get("cross") or {})}
    out = []
    for (m, d), cell in sorted(runs.items()):
        lr4096 = cells[f"{m} | {d} | 4096 | {main_optim(m)} | 0 | 0.0"]["lr_star"]
        others = sorted({c for v in cell.values() for c in v} - {d})
        for B in sorted({B for B, _ in cell}):
            if B == 4096:
                continue
            col = {lr: v for (b, lr), v in cell.items() if b == B}
            ref = min(col, key=lambda lr: abs(math.log(lr / lr4096)))
            tuned = min(col, key=lambda lr: col[lr][d])
            out.append({"model": m, "dataset": d, "B": B, "lr4096": lr4096, "ref": ref, "tuned": tuned, "others": others,
                        "gain": {c: col[ref][c] - col[tuned][c] for c in [d] + others}})
    return out


if __name__ == "__main__":
    rows = gains(load(sys.argv[1]), json.load(open(sys.argv[2], encoding="utf-8")))
    ratios_all, neg, n_cross, last = [], 0, 0, None
    for g in rows:
        m, d, others = g["model"], g["dataset"], g["others"]
        if (m, d) != last:
            print(f"\n{m} x {d}: lr*(4096) {g['lr4096']:.2e}; cross corpora {others}")
            last = (m, d)
        gn = g["gain"]
        line = (f"  B {g['B']:>5}: ref {g['ref']:.1e} tuned {g['tuned']:.1e} | gain {d} {gn[d]:+.4f} | "
                + " ".join(f"{c} {gn[c]:+.4f}" for c in others))
        if gn[d] >= 0.005:
            rs = [gn[c] / gn[d] for c in others]
            ratios_all += rs
            neg += sum(x < 0 for x in rs)
            n_cross += len(rs)
            line += " | ratio " + " ".join(f"{x:+.2f}" for x in rs)
        print(line)
    print(f"\nB with same-corpus gain >= 0.005: {n_cross // 2}; cross ratios median {st.median(ratios_all):+.2f}, "
          f"range [{min(ratios_all):+.2f}, {max(ratios_all):+.2f}], negative {neg} of {n_cross}")
