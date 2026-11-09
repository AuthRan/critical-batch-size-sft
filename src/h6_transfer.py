"""H6 as fixed in PLAN.md section 10, item 8: the gain from retuning lr for B, on the same corpus and the other two.

  python src/h6_transfer.py runs/ledger.jsonl runs/analysis/cells.json
"""
import json
import statistics as st
import sys
from collections import defaultdict

sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
from analyze import load

recs = load(sys.argv[1])
cells = json.load(open(sys.argv[2], encoding="utf-8"))
runs = defaultdict(dict)  # (model, dataset) -> {(B, lr): {corpus: loss}}
for r in recs:
    a = r["args"]
    main_optim = "adamw8bit" if a["model"].startswith("Qwen") else "adamw"  # each grid's own stack; not the 8-bit control
    if a.get("optim", "adamw") != main_optim or a.get("beta2_ref_tokens") or float(a.get("clip", 1.0)) != 0 or a["seed"] != 1:
        continue
    m, d = a["model"].split("/")[-1], a["dataset"]
    runs[(m, d)][(int(a["tokens_per_step"]), float(a["lr"]))] = {d: r["holdout_loss"], **(r.get("cross") or {})}

ratios_all, neg, n_cross = [], 0, 0
for (m, d), cell in sorted(runs.items()):
    lr4096 = cells[f"{m} | {d} | 4096 | {'adamw8bit' if m.startswith('Qwen') else 'adamw'} | 0 | 0.0"]["lr_star"]
    others = sorted({c for v in cell.values() for c in v} - {d})
    print(f"\n{m} x {d}: lr*(4096) {lr4096:.2e}; cross corpora {others}")
    for B in sorted({B for B, _ in cell}):
        if B == 4096:
            continue
        col = {lr: v for (b, lr), v in cell.items() if b == B}
        ref = min(col, key=lambda lr: abs(__import__("math").log(lr / lr4096)))
        tuned = min(col, key=lambda lr: col[lr][d])
        g = {c: col[ref][c] - col[tuned][c] for c in [d] + others}
        line = f"  B {B:>5}: ref {ref:.1e} tuned {tuned:.1e} | gain {d} {g[d]:+.4f} | " + " ".join(f"{c} {g[c]:+.4f}" for c in others)
        if g[d] >= 0.005:
            rs = [g[c] / g[d] for c in others]
            ratios_all += rs
            neg += sum(x < 0 for x in rs)
            n_cross += len(rs)
            line += " | ratio " + " ".join(f"{x:+.2f}" for x in rs)
        print(line)
print(f"\nB with same-corpus gain >= 0.005: {n_cross // 2}; cross ratios median {st.median(ratios_all):+.2f}, "
      f"range [{min(ratios_all):+.2f}, {max(ratios_all):+.2f}], negative {neg} of {n_cross}")
