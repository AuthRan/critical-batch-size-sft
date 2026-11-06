"""Fit lr*(B) per (model, dataset) from the ledger, the exponent alpha, and the fixed-data CBS.

  python src/analyze.py runs/ledger.jsonl --out runs/analysis

Analysis choices are fixed in PLAN.md section 4 and must not be tuned to the data:
  lr*  = vertex of the quadratic in log10(lr) through the argmin and its two neighbours
  CI   = bootstrap over seeds (200 resamples), widened in quadrature by the 3-vs-5-point vertex difference
  alpha = weighted least squares of log lr* on log B
  CBS_fixed_data = largest B whose loss at lr*(B) is within eps of every smaller B's (Merrill et al. 2025)
  diverged = a non-finite held-out loss; an lr with any diverged seed ranks below every finite lr
             and is never fitted (rule added 2026-10-21, see PLAN.md)
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

EPS = 0.005  # nats, fixed in PLAN.md


def read_ledger(path: str) -> list[dict]:
    """First row per run_id: the sweep code before 2026-10-24 13:17 could run a job on both GPUs (PLAN.md section 10)."""
    seen, out = set(), []
    for l in open(path, encoding="utf-8"):
        if l.strip() and (r := json.loads(l))["run_id"] not in seen:
            seen.add(r["run_id"])
            out.append(r)
    return out


def load(ledger: str) -> list[dict]:
    recs = read_ledger(ledger)
    return [r for r in recs if r.get("holdout_loss") is not None and not r["args"].get("max_steps")
            and not r["args"].get("init_from")]


def fit_lr_star(lrs, losses, half_window: int = 1):
    """Returns (lr_star, loss_star, flag). flag in {'ok', 'edge', 'nonconvex'}.
    half_window 1 fits the argmin and its two neighbours (primary); 2 fits up to 5 points (systematic check)."""
    order = np.argsort(lrs)
    lrs = np.asarray(lrs, float)[order]
    losses = np.asarray(losses, float)[order]
    k = int(np.argmin(losses))
    if k == 0 or k == len(lrs) - 1 or not np.isfinite(losses[[k - 1, k + 1]]).all():
        return float(lrs[k]), float(losses[k]), "edge"
    lo, hi = max(0, k - half_window), min(len(lrs), k + half_window + 1)
    keep = np.isfinite(losses[lo:hi])
    x, y = np.log10(lrs[lo:hi])[keep], losses[lo:hi][keep]
    a, b, c = np.polyfit(x, y, 2)
    if a <= 0:
        return float(lrs[k]), float(losses[k]), "nonconvex"
    xs = float(np.clip(-b / (2 * a), np.log10(lrs[k - 1]), np.log10(lrs[k + 1])))
    return 10 ** xs, float(a * xs * xs + b * xs + c), "ok"


def seed_mean(v) -> float:
    """Mean over seeds, or inf if any seed diverged, so that lr ranks last and stays out of the fit."""
    v = np.asarray(v, float)
    return float(v.mean()) if np.isfinite(v).all() else float("inf")


def cell_fit(by_lr: dict[float, list[float]], rng, n_boot: int = 200):
    """by_lr: {lr: [loss per seed]}. Seed-averaged 3-point fit; the interval adds the seed bootstrap and the
    3-vs-5-point window difference (lr_sys, decades) in quadrature (rule added 2026-10-24, see PLAN.md)."""
    lrs = sorted(by_lr)
    mean = [seed_mean(by_lr[lr]) for lr in lrs]
    lr_star, loss_star, flag = fit_lr_star(lrs, mean)
    lr5, _, flag5 = fit_lr_star(lrs, mean, half_window=2)
    sys_dec = abs(np.log10(lr_star / lr5)) if flag == flag5 == "ok" else 0.0
    boots = []
    for _ in range(n_boot):
        samp = [seed_mean(rng.choice(by_lr[lr], size=len(by_lr[lr]), replace=True)) for lr in lrs]
        boots.append(fit_lr_star(lrs, samp)[0])
    b_lo, b_hi = np.log10(np.percentile(boots, [16, 84])) - np.log10(lr_star)
    lo = lr_star * 10 ** -np.hypot(min(b_lo, 0.0), sys_dec)
    hi = lr_star * 10 ** np.hypot(max(b_hi, 0.0), sys_dec)
    diverged = {f"{lr:.1e}": int((~np.isfinite(by_lr[lr])).sum()) for lr in lrs if not np.isfinite(by_lr[lr]).all()}
    return {"lr_star": lr_star, "loss_star": loss_star, "flag": flag, "lr_lo": float(lo), "lr_hi": float(hi),
            "lr_star_5pt": lr5, "lr_sys": float(sys_dec),
            "n_lr": len(lrs), "seeds": int(min(len(v) for v in by_lr.values())), "diverged": diverged}


def critical_batch(pts, eps: float):
    """Merrill et al. 2025: the largest B whose loss is within eps of every smaller B's (rule added 2026-10-24).
    Returns (cbs, the next B up as the interval's upper end, or None if cbs is the largest B)."""
    Bs = [p[0] for p in pts]
    losses = [p[1]["loss_star"] for p in pts]
    ok = [i for i in range(len(Bs)) if all(losses[i] <= losses[j] + eps for j in range(i))]
    k = max(ok)
    return Bs[k], (int(Bs[k + 1]) if k + 1 < len(Bs) else None)


def group_fit(pts, rng) -> dict:
    """pts: [(B, cell fit)] for one (model, dataset, variant). Exponent alpha and the fixed-data CBS.
    Only bracketed ('ok') cells enter the alpha fit: an edge cell's interval is zero-width by construction,
    so inverse-width weighting would hand it almost all the weight (PLAN.md: edge cells get re-run)."""
    pts = sorted(pts, key=lambda p: p[0])
    fit_pts = [p for p in pts if p[1]["flag"] == "ok"]
    alpha, alpha_lo, alpha_hi = float("nan"), float("nan"), float("nan")
    if len(fit_pts) >= 3:
        Bs = np.array([p[1]["B_real"] for p in fit_pts], float)  # exponent fit on realised tokens per step
        ls = np.array([p[1]["lr_star"] for p in fit_pts], float)
        lo_hi = np.log10([[p[1]["lr_lo"], p[1]["lr_hi"]] for p in fit_pts])
        w = 1.0 / np.maximum(1e-6, lo_hi[:, 1] - lo_hi[:, 0])
        X = np.stack([np.ones_like(Bs), np.log10(Bs)], 1)
        W = np.diag(w)
        alpha = float(np.linalg.solve(X.T @ W @ X, X.T @ W @ np.log10(ls))[1])
        boots = []
        for _ in range(200):
            ls_b = rng.uniform(lo_hi[:, 0], lo_hi[:, 1])
            boots.append(float(np.linalg.solve(X.T @ W @ X, X.T @ W @ ls_b)[1]))
        alpha_lo, alpha_hi = map(float, np.percentile(boots, [16, 84]))
    cbs, upper = critical_batch(pts, EPS)
    return {"alpha": alpha, "alpha_lo": alpha_lo, "alpha_hi": alpha_hi, "cbs_fixed_data": int(cbs), "cbs_upper": upper,
            "cbs_fixed_data_eps01": int(critical_batch(pts, 0.01)[0]),
            "n_fit": len(fit_pts), "excluded": [int(p[0]) for p in pts if p[1]["flag"] != "ok"],
            "lr_star_by_B": {int(B): p["lr_star"] for B, p in pts}}


def branch_table(recs: list[dict], core_only: bool = False) -> list[dict]:
    """Branched CBS (PLAN.md section 4, Merrill et al. 2025): per (model, dataset, checkpoint, window), the best
    held-out loss at each B over its lr mini-sweep, then the same CBS rule as the fixed-data grid. core_only keeps the
    pre-registered {1/2.15, 1, 2.15} sweep and leaves out the bracketing extensions (PLAN.md section 10, item 5)."""
    groups: dict[tuple, dict] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for r in recs:
        a, b = r["args"], r["branch"]
        if core_only and abs(np.log(a["lr"] / ((a["tokens_per_step"] / b["base_B"]) ** 0.5 * b["base_lr"]))) > 1.01 * np.log(2.15):
            continue
        key = (a["model"].split("/")[-1], a["dataset"], round(b["base_frac"], 2), b["delta_frac"], b["base_B"], b["base_lr"])
        groups[key][int(a["tokens_per_step"])][float(a["lr"])].append(float(r["holdout_loss"]))
    rows = []
    for key in sorted(groups):
        m, d, frac, delta, B0, lr0 = key
        best_loss, best_mult, edge = {}, {}, []
        for B, by_lr in sorted(groups[key].items()):
            lrs = sorted(by_lr)
            means = [seed_mean(by_lr[lr]) for lr in lrs]
            k = int(np.argmin(means))
            best_loss[B] = means[k]
            best_mult[B] = lrs[k] / ((B / B0) ** 0.5 * lr0)  # relative to Merrill's sqrt(k) rule
            if len(lrs) > 1 and k in (0, len(lrs) - 1):
                edge.append(B)
        pts = [(B, {"loss_star": v}) for B, v in sorted(best_loss.items())]
        cbs, upper = critical_batch(pts, EPS)
        rows.append({"model": m, "dataset": d, "base_frac": frac, "delta_frac": delta, "cbs": int(cbs), "cbs_upper": upper,
                     "cbs_eps01": int(critical_batch(pts, 0.01)[0]), "best_loss": best_loss, "best_mult": best_mult,
                     "edge": edge})
    return rows


def noise_table(recs: list[dict]) -> list[dict]:
    """Per run: B_simple at each measured fraction and the run average, mean tr(Sigma) / mean |G|^2 over the points
    (numerator and denominator averaged separately, McCandlish App. A). v1-estimator points (no n_groups) are
    skipped: they fit across length-sorted micro-batches (PLAN.md section 10, deviation 2)."""
    out = []
    for r in recs:
        pts = [m for m in r.get("noise_scale") or [] if m.get("n_groups")]
        if not pts:
            continue
        a = r["args"]
        G2, trS = float(np.mean([m["G2"] for m in pts])), float(np.mean([m["trSigma"] for m in pts]))
        out.append({"model": a["model"].split("/")[-1], "dataset": a["dataset"], "lr": a["lr"], "seed": a["seed"],
                    "by_frac": {round(m["frac"], 2): m["B_simple"] for m in pts}, "b_max": max(m["b_max"] for m in pts),
                    "avg": trS / G2 if G2 > 0 else float("inf")})
    return sorted(out, key=lambda o: (o["model"], o["dataset"], o["lr"], o["seed"]))


def main(ledger: str, out: str, branch_ledger: str = "", noise_ledger: str = ""):
    recs = load(ledger)
    rng = np.random.default_rng(0)
    cells: dict[tuple, dict[float, list[float]]] = defaultdict(lambda: defaultdict(list))
    real_b: dict[tuple, list[float]] = defaultdict(list)  # realised mean supervised tokens per step
    for r in recs:
        a = r["args"]
        key = (a["model"].split("/")[-1], a["dataset"], int(a["tokens_per_step"]), a.get("optim", "adamw"),
               int(a.get("beta2_ref_tokens", 0)), float(a.get("clip", 1.0)))
        cells[key][float(a["lr"])].append(float(r["holdout_loss"]))
        real_b[key].append(float(r.get("mean_tokens_per_step", a["tokens_per_step"])))

    results = {}
    lines = ["| model | dataset | B (tokens) | mean tokens/step | lr* | 68% CI | loss* | flag | seeds | diverged |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for key in sorted(cells):
        fit = cell_fit(cells[key], rng)
        fit["B_real"] = float(np.mean(real_b[key]))  # the step planner overshoots small B by about one example
        results[" | ".join(map(str, key))] = {"key": key, **fit}
        m, d, B, *_ = key
        lines.append(f"| {m} | {d} | {B} | {fit['B_real']:.0f} | {fit['lr_star']:.2e} | [{fit['lr_lo']:.1e}, {fit['lr_hi']:.1e}] | "
                     f"{fit['loss_star']:.4f} | {fit['flag']} | {fit['seeds']} | "
                     f"{', '.join(f'{lr}: {n}' for lr, n in fit['diverged'].items()) or '-'} |")

    # exponent and fixed-data CBS per (model, dataset, variant)
    lines += ["", "| model | dataset | variant | alpha | 68% CI | CBS_fixed_data (tokens) | CBS at eps 0.01 | n_B fitted | edge B (not fitted) |",
              "|---|---|---|---|---|---|---|---|---|"]
    groups: dict[tuple, list] = defaultdict(list)
    for key, fit in ((k, v) for k, v in results.items()):
        m, d, B, optim, b2, clip = fit["key"]
        groups[(m, d, optim, b2, clip)].append((B, fit))
    summary = {}
    for g, pts in sorted(groups.items()):
        pts.sort(key=lambda p: p[0])
        s = group_fit(pts, rng)
        m, d, optim, b2, clip = g
        variant = optim + (f"+b2ref{b2}" if b2 else "") + ("+noclip" if clip == 0 else "")
        summary["|".join(map(str, g))] = s
        lines.append(f"| {m} | {d} | {variant} | {s['alpha']:.2f} | [{s['alpha_lo']:.2f}, {s['alpha_hi']:.2f}] | "
                     f"{s['cbs_fixed_data']} (next {s['cbs_upper'] or '-'}) | {s['cbs_fixed_data_eps01']} | {s['n_fit']} | {', '.join(map(str, s['excluded'])) or '-'} |")

    if branch_ledger and Path(branch_ledger).exists():
        br_recs = [r for r in read_ledger(branch_ledger) if r.get("branch") and r.get("holdout_loss") is not None]
        brs, core = branch_table(br_recs), branch_table(br_recs, core_only=True)
        for title, table in [("", brs)] + ([("Pre-registered 3-lr sweep only:", core)] if core != brs else []):
            lines += ["", title] if title else [""]
            lines += ["| model | dataset | checkpoint | window | CBS_branched (tokens) | CBS at eps 0.01 | best loss by B | "
                      "best lr / sqrt rule by B | lr at sweep edge |", "|---|---|---|---|---|---|---|---|---|"]
            for r in table:
                lines.append(f"| {r['model']} | {r['dataset']} | {r['base_frac']:.2f} | {r['delta_frac']:g} | "
                             f"{r['cbs']} (next {r['cbs_upper'] or '-'}) | {r['cbs_eps01']} | "
                             f"{', '.join(f'{B}: {v:.4f}' for B, v in r['best_loss'].items())} | "
                             f"{', '.join(f'{B}: {v:.2f}' for B, v in r['best_mult'].items())} | "
                             f"{', '.join(map(str, r['edge'])) or '-'} |")
        summary["branches_core"] = core
        summary["branches"] = brs

    ns = noise_table(read_ledger(noise_ledger)) if noise_ledger and Path(noise_ledger).exists() else []
    if ns:
        def fmt(v, b_max):  # an unresolved fit (|G|^2 <= 0) is a lower bound, never a number (PLAN.md section 4)
            return f">{b_max / 1000:.0f}k" if not np.isfinite(v) else f"{v / 1000:.1f}k"

        fracs = sorted({f for o in ns for f in o["by_frac"]})
        lines += ["", "| model | dataset | lr | seed | " + " | ".join(f"B_simple at {f:g}" for f in fracs) + " | run average |",
                  "|---|---|---|---|" + "---|" * (len(fracs) + 1)]
        for o in ns:
            lines.append(f"| {o['model']} | {o['dataset']} | {o['lr']:.2e} | {o['seed']} | "
                         + " | ".join(fmt(o["by_frac"][f], o["b_max"]) if f in o["by_frac"] else "-" for f in fracs)
                         + f" | {fmt(o['avg'], o['b_max'])} |")
        summary["noise"] = ns
    Path(out).mkdir(parents=True, exist_ok=True)
    (Path(out) / "cells.json").write_text(json.dumps({k: {**v, "key": list(v["key"])} for k, v in results.items()}, indent=1))
    (Path(out) / "summary.json").write_text(json.dumps(summary, indent=1))
    (Path(out) / "tables.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for g, pts in sorted(groups.items()):
            pts.sort()
            Bs = [p[0] for p in pts]
            axes[0].errorbar(Bs, [p[1]["lr_star"] for p in pts],
                             yerr=[[p[1]["lr_star"] - p[1]["lr_lo"] for p in pts], [p[1]["lr_hi"] - p[1]["lr_star"] for p in pts]],
                             marker="o", label=f"{g[0]} / {g[1]}", capsize=3)
            axes[1].plot(Bs, [p[1]["loss_star"] for p in pts], marker="o", label=f"{g[0]} / {g[1]}")
        for ax, yl in zip(axes, ["lr* (held-out)", "held-out loss at lr*"]):
            ax.set_xscale("log", base=2)
            ax.set_xlabel("supervised tokens per step")
            ax.set_ylabel(yl)
            ax.grid(True, alpha=0.3)
        axes[0].set_yscale("log")
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(Path(out) / "lr_star_vs_B.png", dpi=150)
    except Exception as e:  # plotting is optional
        print(f"(no plot: {e})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ledger", nargs="?", default="runs/ledger.jsonl")
    ap.add_argument("--out", default="runs/analysis")
    ap.add_argument("--branch_ledger", default="runs/branch_ledger.jsonl")
    ap.add_argument("--noise_ledger", default="runs/noise2_ledger.jsonl")
    a = ap.parse_args()
    main(a.ledger, a.out, a.branch_ledger, a.noise_ledger)
