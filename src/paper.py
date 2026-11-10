"""Every number, table and figure in paper/main.tex, computed from the ledgers in data/.

  python src/paper.py

Writes paper/numbers.json and paper/numbers.md (the values quoted in the text), paper/tables/*.tex and
paper/figures/*.pdf. Cell fits, CBS and branch tables come from analyze.py, so they match its output exactly.
"""
from __future__ import annotations

import contextlib
import io
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import os

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).parent))
import analyze  # noqa: E402
import h6_transfer  # noqa: E402

DATA, OUT = Path("data"), Path("paper")
MODELS = ["SmolLM2-135M", "SmolLM2-360M", "Qwen3-0.6B-Base"]
CORPORA = ["dolly", "oasst1", "gsm8k"]
BS = [512, 1024, 2048, 4096, 8192, 16384, 32768]
NAME = {"SmolLM2-135M": "SmolLM2-135M", "SmolLM2-360M": "SmolLM2-360M", "Qwen3-0.6B-Base": "Qwen3-0.6B"}
CNAME = {"dolly": "Dolly", "oasst1": "OASST1", "gsm8k": "GSM8K"}
COLOR = {"SmolLM2-135M": "#2a78d6", "SmolLM2-360M": "#eb6834", "Qwen3-0.6B-Base": "#1baf7a"}  # validated, all pairs
MARK = {"SmolLM2-135M": "o", "SmolLM2-360M": "s", "Qwen3-0.6B-Base": "^"}
BLUE, ORANGE, GREY, INK, MUTED = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e"
RAMP = ["#86b6ef", "#6da7ec", "#3987e5", "#2a78d6", "#1c5cab", "#184f95", "#0d366b"]  # ordinal blue, B = 512 .. 32,768
PREVIEW = os.environ.get("PAPER_PREVIEW", "")  # also write PNGs there, to look at
NOISE_LR_360M_DOLLY = 4.6e-5  # PLAN.md section 10, item 2(b): the pilot setting is judged at the lr nearest lr*(4096)


def optim(m: str) -> str:
    return h6_transfer.main_optim(m)


def ckey(m, d, B, b2=0, clip=0.0, opt=None):
    return f"{m} | {d} | {B} | {opt or optim(m)} | {b2} | {clip}"


def gkey(m, d, b2=0, clip=0.0, opt=None):
    return f"{m}|{d}|{opt or optim(m)}|{b2}|{clip}"


def comma(x) -> str:
    return f"{int(round(x)):,}"


def kfmt(x) -> str:
    """Noise scales in thousands of tokens (k = 1,000), two significant figures."""
    if not np.isfinite(x):
        return "unres."
    return f"{x / 1000:.1f}k" if x < 10_000 else f"{x / 1000:.0f}k"


def sci(x) -> str:
    m, e = f"{x:.2e}".split("e")
    return f"{m}\\text{{e}}{int(e)}"


# ---------------------------------------------------------------- inputs
def run_analysis(tmp: Path):
    with contextlib.redirect_stdout(io.StringIO()):
        analyze.main(str(DATA / "ledger.jsonl"), str(tmp), str(DATA / "branch_ledger.jsonl"), str(DATA / "noise2_ledger.jsonl"))
    return json.load(open(tmp / "cells.json", encoding="utf-8")), json.load(open(tmp / "summary.json", encoding="utf-8"))


def noise_points(path: Path) -> dict:
    """(model, dataset) -> list of runs, each {frac: (G2, trSigma, B_simple)} for v2-estimator points."""
    out = defaultdict(list)
    for r in analyze.read_ledger(str(path)):
        a = r["args"]
        m, d = a["model"].split("/")[-1], a["dataset"]
        if m == "SmolLM2-360M" and d == "dolly" and abs(a["lr"] - NOISE_LR_360M_DOLLY) > 1e-9:
            continue
        pts = {round(p["frac"], 2): (p["G2"], p["trSigma"], p["B_simple"]) for p in r.get("noise_scale") or [] if p.get("n_groups")}
        if pts:
            out[(m, d)].append(pts)
    return out


def ratio_avg(pts: dict, fracs) -> float:
    """mean tr(Sigma) / mean |G|^2 over the given points (PLAN.md section 10, item 2(a))."""
    G2 = np.mean([pts[f][0] for f in fracs])
    trS = np.mean([pts[f][1] for f in fracs])
    return float(trS / G2) if G2 > 0 else float("inf")


# ---------------------------------------------------------------- main
def main():
    tmp = Path("runs/analysis_paper")
    cells, summ = run_analysis(tmp)
    recs = analyze.load(str(DATA / "ledger.jsonl"))
    raw = [json.loads(l) for l in open(DATA / "ledger.jsonl", encoding="utf-8") if l.strip()]
    rng = np.random.default_rng(0)
    N: dict = {"settings": {}}

    # corpus statistics, per tokenizer (SmolLM2 models share one)
    corp = {}
    for r in recs:
        a = r["args"]
        m, d = a["model"].split("/")[-1], a["dataset"]
        if a["tokens_per_step"] == 512 and a["seed"] == 1 and a.get("optim", "adamw") == optim(m) and not a.get("beta2_ref_tokens"):
            corp[(m, d)] = {"n_train": r["n_train"], "n_holdout": r["n_holdout"],
                            "tokens_epoch": r["n_steps"] * r["mean_tokens_per_step"]}
    for k, v in corp.items():
        v["tokens_per_example"] = v["tokens_epoch"] / v["n_train"]
    N["corpora"] = {f"{m}|{d}": v for (m, d), v in corp.items()}

    noise = noise_points(DATA / "noise2_ledger.jsonl")
    fracs_all = [0.0, 0.25, 0.5, 0.75, 1.0]

    for m in MODELS:
        for d in CORPORA:
            S: dict = {}
            pts = [(B, cells[ckey(m, d, B)]) for B in BS]
            g = summ[gkey(m, d)]
            S.update({k: g[k] for k in ("alpha", "alpha_lo", "alpha_hi", "cbs_fixed_data", "cbs_upper", "cbs_fixed_data_eps01", "excluded")})
            lr = {B: f["lr_star"] for B, f in pts}
            S["lr_star"] = lr
            S["lr_ci"] = {B: (f["lr_lo"], f["lr_hi"]) for B, f in pts}
            S["flag"] = {B: f["flag"] for B, f in pts}
            S["loss_star"] = {B: f["loss_star"] for B, f in pts}
            S["B_real"] = {B: f["B_real"] for B, f in pts}
            Bpk = max(BS, key=lambda B: lr[B])
            S["B_peak"] = Bpk if Bpk < BS[-1] else None
            sub = lambda keep: analyze.group_fit([(B, f) for B, f in pts if keep(B)], rng)
            if S["B_peak"]:
                a = sub(lambda B: B <= Bpk)
                S["alpha_below_peak"] = (a["alpha"], a["alpha_lo"], a["alpha_hi"])
                S["fall_after_peak"] = lr[BS[-1]] / lr[Bpk]
            for name, keep in (("alpha_low", lambda B: B <= 4096), ("alpha_high", lambda B: B >= 4096)):
                a = sub(keep)
                S[name] = (a["alpha"], a["alpha_lo"], a["alpha_hi"])
            S["lr_rise"] = max(lr.values()) / lr[512]
            losses = S["loss_star"]
            S["loss_pen_32k"] = losses[32768] - min(losses.values())
            c = corp[(m, d)]
            S["steps_at_cbs"] = c["tokens_epoch"] / S["cbs_fixed_data"]
            S["examples_at_cbs"] = S["cbs_fixed_data"] / c["tokens_per_example"]
            runs = noise[(m, d)]
            S["noise_n_runs"] = len(runs)
            S["G2"] = {f: float(np.mean([r[f][0] for r in runs])) for f in fracs_all}
            S["trSigma"] = {f: float(np.mean([r[f][1] for r in runs])) for f in fracs_all}
            S["B_simple"] = {f: float(np.mean([r[f][2] for r in runs])) for f in fracs_all}
            S["B_avg"] = float(np.mean([ratio_avg(r, fracs_all) for r in runs]))
            S["B_late"] = float(np.mean([ratio_avg(r, fracs_all[1:]) for r in runs]))  # exploratory
            N["settings"][f"{m}|{d}"] = S

    st = N["settings"]
    keys = [f"{m}|{d}" for m in MODELS for d in CORPORA]

    # H2: step-0 noise scale and its rise
    b0 = [st[k]["B_simple"][0.0] for k in keys]
    N["H2"] = {"step0_min": min(b0), "step0_max": max(b0), "n_above_16k": sum(x > 16000 for x in b0),
               "n_below_8k": sum(x < 8000 for x in b0),
               "rise_end_over_0": {k: st[k]["B_simple"][1.0] / st[k]["B_simple"][0.0] for k in keys},
               "rise_25_over_0": {k: st[k]["B_simple"][0.25] / st[k]["B_simple"][0.0] for k in keys},
               "G2_drop_0_to_25": {k: st[k]["G2"][0.0] / st[k]["G2"][0.25] for k in keys},
               "trS_ratio_25_over_0": {k: st[k]["trSigma"][0.25] / st[k]["trSigma"][0.0] for k in keys}}

    # H3 (pre-registered: run average) and the exploratory late average
    cbs = [st[k]["cbs_fixed_data"] for k in keys]
    for name, field in (("H3", "B_avg"), ("H3_late_exploratory", "B_late")):
        x = [st[k][field] for k in keys]
        ratios = [a / b for a, b in zip(x, cbs)]
        N[name] = {"rho": float(spearmanr(x, cbs).statistic), "ratios": dict(zip(keys, ratios)),
                   "median_ratio": float(np.median(ratios)), "n_within_2x": sum(0.5 <= q <= 2 for q in ratios)}

    # H1b: peaks against the noise scale
    N["H1b"] = {k: {"B_peak": st[k]["B_peak"], "over_avg": st[k]["B_peak"] / st[k]["B_avg"],
                    "over_late_exploratory": st[k]["B_peak"] / st[k]["B_late"]} for k in keys if st[k]["B_peak"]}

    # H5 and the clipping control (pilot setting, extreme B)
    ctl = {}
    for B in (512, 32768):
        base = cells[ckey("SmolLM2-360M", "dolly", B)]
        for name, k in (("beta2", ckey("SmolLM2-360M", "dolly", B, b2=4096)), ("clip", ckey("SmolLM2-360M", "dolly", B, clip=1.0))):
            c = cells[k]
            ctl[f"{name}|{B}"] = {"lr": c["lr_star"], "ci": (c["lr_lo"], c["lr_hi"]), "loss": c["loss_star"], "flag": c["flag"],
                                  "seeds": c["seeds"], "base_lr": base["lr_star"], "base_ci": (base["lr_lo"], base["lr_hi"]),
                                  "base_loss": base["loss_star"], "ratio": c["lr_star"] / base["lr_star"]}
    N["controls"] = ctl

    # section 7 optimiser-stack control
    eb = {}
    for B in BS:
        a, b = cells[ckey("SmolLM2-135M", "dolly", B, opt="adamw")], cells[ckey("SmolLM2-135M", "dolly", B, opt="adamw8bit")]
        overlap = a["lr_lo"] <= b["lr_hi"] and b["lr_lo"] <= a["lr_hi"]
        eb[B] = {"ratio": b["lr_star"] / a["lr_star"], "overlap": overlap, "loss_diff": b["loss_star"] - a["loss_star"],
                 "lr": b["lr_star"], "ci": (b["lr_lo"], b["lr_hi"])}
    g8 = summ[gkey("SmolLM2-135M", "dolly", opt="adamw8bit")]
    N["control_8bit"] = {"by_B": eb, "alpha": (g8["alpha"], g8["alpha_lo"], g8["alpha_hi"]), "cbs": g8["cbs_fixed_data"],
                         "cbs_eps01": g8["cbs_fixed_data_eps01"]}

    # branched CBS
    core = {(r["model"], r["dataset"], r["base_frac"], r["delta_frac"]): r for r in summ["branches_core"]}
    br = []
    for r in summ["branches"]:
        k = (r["model"], r["dataset"], r["base_frac"], r["delta_frac"])
        s = st[f"{r['model']}|{r['dataset']}"]
        br.append({"model": r["model"], "dataset": r["dataset"], "frac": r["base_frac"], "delta": r["delta_frac"],
                   "cbs": r["cbs"], "censored": r["cbs_upper"] is None, "cbs_eps01": r["cbs_eps01"], "cbs_core": core[k]["cbs"],
                   "core_censored": core[k]["cbs_upper"] is None, "core_edge": core[k]["edge"], "edge": r["edge"],
                   "fixed": s["cbs_fixed_data"], "fixed_eps01": s["cbs_fixed_data_eps01"],
                   "B_simple_t": s["B_simple"][r["base_frac"]], "best_loss": r["best_loss"], "best_mult": r["best_mult"]})
    N["branches"] = br
    braw = [json.loads(l) for l in open(DATA / "branch_ledger.jsonl", encoding="utf-8") if l.strip()]
    N["branch_runs"] = {"rows": len(braw), "branches": sum(1 for r in analyze.read_ledger(str(DATA / "branch_ledger.jsonl")) if r.get("branch")),
                        "ext": sum(1 for r in braw if r.get("branch") and abs(math.log(r["args"]["lr"] / ((r["args"]["tokens_per_step"] / r["branch"]["base_B"]) ** 0.5 * r["branch"]["base_lr"]))) > 1.01 * math.log(2.15))}

    # H6
    h6 = h6_transfer.gains(recs, cells)
    moved = [g for g in h6 if g["tuned"] != g["ref"]]
    down = [g for g in moved if g["tuned"] < g["ref"]]
    up = [g for g in moved if g["tuned"] > g["ref"]]
    cross = lambda gs: [g["gain"][c] for g in gs for c in g["others"]]
    big = [g for g in h6 if g["gain"][g["dataset"]] >= 0.005]
    rat = [g["gain"][c] / g["gain"][g["dataset"]] for g in big for c in g["others"]]
    N["H6"] = {"n_rows": len(h6), "n_moved": len(moved), "n_down": len(down), "n_up": len(up),
               "down_cross_n": len(cross(down)), "down_cross_neg": sum(x < 0 for x in cross(down)),
               "up_cross_n": len(cross(up)), "up_cross_neg": sum(x < 0 for x in cross(up)),
               "up_same_gain_max": max(g["gain"][g["dataset"]] for g in up), "up_cross_min": min(cross(up)),
               "down_cross_max": max(cross(down)),
               "n_big": len(big), "ratio_median": float(np.median(rat)), "ratio_n_neg": sum(x < 0 for x in rat), "ratio_n": len(rat),
               "down_ratio_median": float(np.median([g["gain"][c] / g["gain"][g["dataset"]] for g in big if g["tuned"] < g["ref"] for c in g["others"]]))}

    # seeds, divergence, skipped steps, compute
    by = defaultdict(list)
    for r in recs:
        a = r["args"]
        m = a["model"].split("/")[-1]
        if a.get("optim", "adamw") == optim(m) and not a.get("beta2_ref_tokens") and float(a.get("clip", 1.0)) == 0:
            by[(m, a["dataset"], a["tokens_per_step"], a["lr"])].append(r)
    spreads = [np.ptp([r["holdout_loss"] for r in v]) for v in by.values() if len(v) >= 3 and all(np.isfinite(r["holdout_loss"]) for r in v)]
    N["seed_spread"] = {"n_points": len(spreads), "median": float(np.median(spreads)), "p90": float(np.percentile(spreads, 90)), "max": float(np.max(spreads))}
    worst_skip = 0.0
    for m in MODELS:
        for d in CORPORA:
            for B in BS:
                lrs = sorted({k[3] for k in by if k[:3] == (m, d, B)})
                best = min(lrs, key=lambda x: analyze.seed_mean([r["holdout_loss"] for r in by[(m, d, B, x)]]))
                worst_skip = max(worst_skip, max(r["skipped_steps"] / r["n_steps"] for r in by[(m, d, B, best)]))
    N["skipped_at_argmin_max_frac"] = worst_skip
    main_runs = [r for v in by.values() for r in v]
    N["diverged_main"] = sum(1 for r in main_runs if not np.isfinite(r["holdout_loss"]))
    N["runs_main_grids"] = len(main_runs)
    noise_rows = [json.loads(l) for f in ("noise2_ledger.jsonl", "noise_ledger_v1.jsonl", "smoke_ledger.jsonl")
                  for l in open(DATA / f, encoding="utf-8") if l.strip()]
    N["runs"] = {"ledger_rows": len(raw), "ledger_unique": len(analyze.read_ledger(str(DATA / "ledger.jsonl"))),
                 "branch_rows": len(braw), "noise_and_smoke_rows": len(noise_rows)}
    N["gpu_hours"] = sum(r.get("wall_s", 0) for r in raw + braw + noise_rows) / 3600
    N["wall_min_median_135M_dolly"] = {B: float(np.median([r["wall_s"] / 60 for k, v in by.items() if k[:3] == ("SmolLM2-135M", "dolly", B) for r in v]))
                                       for B in BS}
    two = lambda lo, hi: math.log(hi / lo) / math.log(64)
    N["H5_two_point_alpha"] = {"main": two(ctl["beta2|512"]["base_lr"], ctl["beta2|32768"]["base_lr"]),
                               "beta2": two(ctl["beta2|512"]["lr"], ctl["beta2|32768"]["lr"]),
                               "clip": two(ctl["clip|512"]["lr"], ctl["clip|32768"]["lr"])}

    OUT.mkdir(exist_ok=True)
    (OUT / "numbers.json").write_text(json.dumps(N, indent=1, default=float), encoding="utf-8")
    write_numbers_md(N)
    write_tables(N, cells)
    figures(N, cells, recs, h6)


# ---------------------------------------------------------------- text
def write_numbers_md(N):
    st = N["settings"]
    L = ["# Numbers quoted in the paper (generated by src/paper.py)", ""]
    for k, s in st.items():
        L.append(f"## {k}")
        L.append(f"alpha {s['alpha']:.2f} [{s['alpha_lo']:.2f}, {s['alpha_hi']:.2f}]; low {s['alpha_low'][0]:.2f} high {s['alpha_high'][0]:.2f}"
                 + (f"; peak {s['B_peak']} alpha below peak {s['alpha_below_peak'][0]:.2f} [{s['alpha_below_peak'][1]:.2f}, {s['alpha_below_peak'][2]:.2f}], lr*(32k)/lr*(peak) {s['fall_after_peak']:.2f}" if s["B_peak"] else "; no peak"))
        L.append(f"lr* rise max/512 {s['lr_rise']:.2f}; CBS {s['cbs_fixed_data']} (next {s['cbs_upper']}) eps01 {s['cbs_fixed_data_eps01']}; "
                 f"steps/epoch at CBS {s['steps_at_cbs']:.0f}; examples/step at CBS {s['examples_at_cbs']:.0f}; loss penalty at 32k {s['loss_pen_32k']:.4f}")
        L.append("B_simple " + ", ".join(f"{f}: {v:.0f}" for f, v in s["B_simple"].items()) + f"; avg {s['B_avg']:.0f}; late {s['B_late']:.0f}; runs {s['noise_n_runs']}")
        L.append("G2 " + ", ".join(f"{f}: {v:.4f}" for f, v in s["G2"].items()) + "; trSigma " + ", ".join(f"{f}: {v:.0f}" for f, v in s["trSigma"].items()))
        L.append("")
    for k in ("corpora", "H2", "H3", "H3_late_exploratory", "H1b", "controls", "control_8bit", "H6", "branch_runs", "seed_spread", "wall_min_median_135M_dolly", "H5_two_point_alpha",
              "skipped_at_argmin_max_frac", "diverged_main", "runs_main_grids", "runs", "gpu_hours"):
        L.append(f"## {k}")
        L.append(json.dumps(N[k], indent=1, default=float))
        L.append("")
    L.append("## branches")
    for b in N["branches"]:
        L.append(f"{b['model']} {b['dataset']} f{b['frac']} d{b['delta']}: CBS {b['cbs']}{'+' if b['censored'] else ''} (3-lr {b['cbs_core']}{'+' if b['core_censored'] else ''}) "
                 f"eps01 {b['cbs_eps01']} | fixed {b['fixed']} ratio {b['cbs'] / b['fixed']:.2f} | B_simple(t) {b['B_simple_t']:.0f} ratio CBS/B_simple {b['cbs'] / b['B_simple_t']:.2f}")
    (OUT / "numbers.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def tsci(x, digits: int = 2) -> str:
    """A learning rate for a table: $1.58\\mathrm{e}{-5}$."""
    m, e = f"{x:.{digits}e}".split("e")
    return f"${m}\\mathrm{{e}}{{{int(e)}}}$"


def tci(lo, hi) -> str:
    d = 1 if f"{lo:.1e}" != f"{hi:.1e}" else 2  # one more digit when a narrow interval would print as [x, x]
    return f"[{tsci(lo, d)}, {tsci(hi, d)}]"


def tnum(x, fmt: str) -> str:
    """A signed number for a table, with a real minus sign and no negative zero."""
    s = format(x, fmt)
    if s.lstrip("+-").strip("0.") == "":
        s = s.lstrip("+-")
    return s.replace("-", "$-$")


BRANCH_SETTINGS = [("SmolLM2-135M", "dolly"), ("SmolLM2-360M", "dolly"), ("SmolLM2-135M", "gsm8k")]


def write_tables(N, cells):
    T = OUT / "tables"
    T.mkdir(parents=True, exist_ok=True)
    st = N["settings"]

    # corpora
    rows = []
    for d in CORPORA:
        a, q = N["corpora"][f"SmolLM2-135M|{d}"], N["corpora"][f"Qwen3-0.6B-Base|{d}"]
        rows.append(f"{CNAME[d]} & {comma(a['n_train'])} & {comma(a['n_holdout'])} & {a['tokens_epoch'] / 1e6:.2f}M & {q['tokens_epoch'] / 1e6:.2f}M & "
                    f"{a['tokens_per_example']:.0f} & {comma(a['tokens_epoch'] / 512)} & {a['tokens_epoch'] / 32768:.0f} \\\\")
    (T / "corpora.tex").write_text("\n".join(rows) + "\n\\bottomrule\n", encoding="utf-8")

    # main results
    rows = []
    for m in MODELS:
        for d in CORPORA:
            s = st[f"{m}|{d}"]
            cb = f"{comma(s['cbs_fixed_data'])}" + ("$^{\\ast}$" if s["cbs_upper"] is None else "")
            pk = comma(s["B_peak"]) if s["B_peak"] else "--"
            ab = f"{s['alpha_below_peak'][0]:.2f}" if s["B_peak"] else "--"
            rows.append(f"{NAME[m]} & {CNAME[d]} & {tnum(s['alpha'], '.2f')} [{tnum(s['alpha_lo'], '.2f')}, {tnum(s['alpha_hi'], '.2f')}] & "
                        f"{tnum(s['alpha_low'][0], '.2f')} & {tnum(s['alpha_high'][0], '.2f')} & {pk} & {ab} & {cb} & "
                        f"{comma(s['cbs_fixed_data_eps01'])} & {s['steps_at_cbs']:.0f} & {s['loss_pen_32k']:.3f} \\\\")
        if m != MODELS[-1]:
            rows.append("\\midrule")
    (T / "main.tex").write_text("\n".join(rows) + "\n\\bottomrule\n", encoding="utf-8")

    # noise
    rows = []
    for m in MODELS:
        for d in CORPORA:
            s = st[f"{m}|{d}"]
            rows.append(f"{NAME[m]} & {CNAME[d]} & " + " & ".join(kfmt(s["B_simple"][f]) for f in (0.0, 0.25, 0.5, 0.75, 1.0))
                        + f" & {kfmt(s['B_avg'])} & {kfmt(s['B_late'])} & {comma(s['cbs_fixed_data'])} \\\\")
        if m != MODELS[-1]:
            rows.append("\\midrule")
    (T / "noise.tex").write_text("\n".join(rows) + "\n\\bottomrule\n", encoding="utf-8")

    # branched, in the figure's order
    rows = []
    c = lambda v, cen: comma(v) + ("$^{\\ast}$" if cen else "")
    for i, setting in enumerate(BRANCH_SETTINGS):
        if i:
            rows.append("\\midrule")
        bs = sorted([b for b in N["branches"] if (b["model"], b["dataset"]) == setting], key=lambda b: (b["frac"], -b["delta"]))
        for j, b in enumerate(bs):
            name = f"{NAME[b['model']]}, {CNAME[b['dataset']]}" if j == 0 else ""
            rows.append(f"{name} & {b['frac'] * 100:.0f}\\% & {b['delta'] * 100:g}\\% & {c(b['cbs_core'], b['core_censored'])} & "
                        f"{c(b['cbs'], b['censored'])} & {c(b['cbs_eps01'], b['cbs_eps01'] == 32768)} & {comma(b['fixed'])} & {kfmt(b['B_simple_t'])} \\\\")
    (T / "branched.tex").write_text("\n".join(rows) + "\n\\bottomrule\n", encoding="utf-8")

    # controls
    ctl = N["controls"]
    rows = []
    for B in (512, 32768):
        base = ctl[f"beta2|{B}"]
        rows.append(f"{comma(B)} & main grid ($\\beta_2 = 0.95$, no clipping) & {tsci(base['base_lr'])} & {tci(*base['base_ci'])} & {base['base_loss']:.4f} & -- \\\\")
        for name, lab in (("beta2", "$\\beta_2$ half-life fixed in tokens"), ("clip", "gradient clipping at 1.0")):
            v = ctl[f"{name}|{B}"]
            rows.append(f" & {lab} & {tsci(v['lr'])} & {tci(*v['ci'])} & {v['loss']:.4f} & {v['ratio']:.2f} \\\\")
        if B == 512:
            rows.append("\\midrule")
    (T / "controls.tex").write_text("\n".join(rows) + "\n\\bottomrule\n", encoding="utf-8")

    rows = []
    for B in BS:
        a, b = cells[ckey("SmolLM2-135M", "dolly", B, opt="adamw")], N["control_8bit"]["by_B"][B]
        rows.append(f"{comma(B)} & {tsci(a['lr_star'])} & {tci(a['lr_lo'], a['lr_hi'])} & {tsci(b['lr'])} & {tci(*b['ci'])} & {b['ratio']:.2f} & "
                    f"{tnum(b['loss_diff'] * 1000, '+.1f')} \\\\")
    (T / "control_8bit.tex").write_text("\n".join(rows) + "\n\\bottomrule\n", encoding="utf-8")

    # appendix: every main-grid cell
    rows = []
    for m in MODELS:
        for d in CORPORA:
            for i, B in enumerate(BS):
                f = cells[ckey(m, d, B)]
                lab = f"{NAME[m]} & {CNAME[d]}" if i == 0 else " & "
                div = "; ".join(f"{tsci(float(k), 1)}: {v}" for k, v in f["diverged"].items()) or "--"
                star = tsci(f["lr_star"]) + ("$^{\\dagger}$" if f["flag"] != "ok" else "")
                rows.append(f"{lab} & {comma(B)} & {comma(f['B_real'])} & {star} & {tci(f['lr_lo'], f['lr_hi'])} & {f['loss_star']:.4f} & "
                            f"{f['seeds']} & {div} \\\\")
            rows.append("\\addlinespace")  # not \midrule: a rule before a page break doubles the longtable foot rule
    (T / "cells.tex").write_text("\n".join(rows[:-1]) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- figures
def style(plt):
    plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "axes.linewidth": 0.6, "axes.edgecolor": MUTED, "xtick.color": MUTED,
                         "ytick.color": MUTED, "axes.labelcolor": INK, "text.color": INK, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.5,
                         "lines.linewidth": 1.4, "lines.markersize": 4.5, "pdf.fonttype": 42, "font.family": "serif",
                         "mathtext.fontset": "cm", "legend.frameon": False})


def btick(ax):
    ax.set_xscale("log", base=2)
    ax.set_xticks(BS)
    ax.set_xticklabels(["0.5k", "1k", "2k", "4k", "8k", "16k", "32k"])
    ax.minorticks_off()


def save(fig, F: Path, name: str):
    fig.savefig(F / f"{name}.pdf")
    if PREVIEW:
        Path(PREVIEW).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(PREVIEW) / f"{name}.png", dpi=170)


def figures(N, cells, recs, h6):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    style(plt)
    F = OUT / "figures"
    F.mkdir(parents=True, exist_ok=True)
    st = N["settings"]

    # Figure: lr*(B) / lr*(4096)
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.25), sharey=True)
    for ax, d in zip(axes, CORPORA):
        x = np.array([256 * 1.6, 32768 * 1.6])
        ax.plot(x, (x / 4096) ** 0.5, color=GREY, lw=0.9, ls="--", zorder=1)
        for m in MODELS:
            s = st[f"{m}|{d}"]
            ref = s["lr_star"][4096]
            Bx = [s["B_real"][B] for B in BS]
            y = np.array([s["lr_star"][B] / ref for B in BS])
            lo = y - np.array([s["lr_ci"][B][0] / ref for B in BS])
            hi = np.array([s["lr_ci"][B][1] / ref for B in BS]) - y
            ax.errorbar(Bx, y, yerr=[lo, hi], color=COLOR[m], marker=MARK[m], ms=4.5, capsize=1.5, elinewidth=0.7, label=NAME[m], zorder=3,
                        markeredgecolor="white", markeredgewidth=0.5)
            for B, yy in zip(BS, y):
                if s["flag"][B] != "ok":
                    ax.plot(s["B_real"][B], yy, marker=MARK[m], ms=6.5, mfc="white", mec=COLOR[m], mew=1.0, zorder=4)
        btick(ax)
        ax.set_yscale("log")
        ax.set_yticks([0.25, 0.5, 1, 2, 4])
        ax.set_yticklabels(["0.25", "0.5", "1", "2", "4"])
        ax.minorticks_off()
        ax.set_ylim(0.17, 4.6)
        ax.set_title(CNAME[d])
        ax.set_xlabel("supervised tokens per step $B$")
        ax.set_xlim(400, 42000)
    axes[0].set_ylabel(r"$\eta^*(B)\,/\,\eta^*(4096)$")
    axes[0].text(2600, 0.2, r"dashed: $\propto B^{1/2}$", color=MUTED, fontsize=7)
    axes[0].legend(loc="upper left")
    fig.tight_layout(w_pad=0.6)
    save(fig, F, "lr_star")
    plt.close(fig)

    # Figure: loss at lr*(B) above the best B
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.25))
    for ax, d in zip(axes, CORPORA):
        ax.axhline(0.005, color=GREY, lw=0.9, ls="--", zorder=1)
        for m in MODELS:
            s = st[f"{m}|{d}"]
            best = min(s["loss_star"].values())
            ax.plot([s["B_real"][B] for B in BS], [s["loss_star"][B] - best for B in BS], color=COLOR[m], marker=MARK[m],
                    label=NAME[m], markeredgecolor="white", markeredgewidth=0.5, zorder=3)
        btick(ax)
        ax.set_title(CNAME[d])
        ax.set_xlabel("supervised tokens per step $B$")
        ax.set_xlim(400, 42000)
        ax.set_ylim(bottom=-0.0005 if d != "gsm8k" else -0.004)
    axes[0].text(560, 0.0054, r"$\epsilon = 0.005$", color=MUTED, fontsize=7, va="bottom")
    axes[0].set_ylabel("gap to the best $B$ (nats)")
    axes[0].legend(loc="upper left")
    fig.tight_layout(w_pad=0.6)
    save(fig, F, "loss_gap")
    plt.close(fig)

    # Figure: noise scale along the run
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.25), sharey=True)
    fr = [0.0, 0.25, 0.5, 0.75, 1.0]
    for ax, d in zip(axes, CORPORA):
        for m in MODELS:
            s = st[f"{m}|{d}"]
            ax.plot(fr, [s["B_simple"][f] for f in fr], color=COLOR[m], marker=MARK[m], label=NAME[m],
                    markeredgecolor="white", markeredgewidth=0.5, zorder=3)
            ax.axhline(s["B_avg"], color=COLOR[m], lw=0.8, ls=(0, (4, 2)), zorder=2)
        ax.set_yscale("log")
        ax.set_xticks(fr)
        ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
        ax.set_title(CNAME[d])
        ax.set_xlabel("fraction of the epoch")
    axes[0].set_ylabel(r"$B_{\mathrm{simple}}$ (supervised tokens)")
    axes[0].legend(loc="lower right")
    fig.tight_layout(w_pad=0.6)
    save(fig, F, "noise")
    plt.close(fig)

    # Figure: branched CBS against the checkpoint
    settings = [("SmolLM2-135M", "dolly"), ("SmolLM2-360M", "dolly"), ("SmolLM2-135M", "gsm8k")]
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.35), sharey=True)
    for ax, (m, d) in zip(axes, settings):
        bs = [b for b in N["branches"] if (b["model"], b["dataset"]) == (m, d)]
        main = sorted([b for b in bs if b["delta"] == 0.25], key=lambda b: b["frac"])
        x = [b["frac"] for b in main]
        ax.axhline(st[f"{m}|{d}"]["cbs_fixed_data"], color=INK, lw=0.9, ls="--", zorder=1, label="fixed-data CBS")
        ax.plot(x, [b["cbs"] for b in main], color=BLUE, marker="o", ms=5, zorder=3, label=r"branched CBS ($\Delta$ = 25%)",
                markeredgecolor="white", markeredgewidth=0.5)
        for b in main:
            if b["censored"]:
                ax.annotate("", xy=(b["frac"], b["cbs"] * 1.75), xytext=(b["frac"], b["cbs"] * 1.08),
                            arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=0.9, mutation_scale=6))
        h = [b for b in bs if b["delta"] == 0.125]
        for b in h:
            ax.plot(b["frac"] + 0.035, b["cbs"], marker="D", ms=4, mfc="white", mec=BLUE, mew=1.0, zorder=3, ls="none",
                    label=r"branched CBS ($\Delta$ = 12.5%)")
        ax.plot(x, [b["B_simple_t"] for b in main], color=ORANGE, marker="^", lw=1.0, ls=":", zorder=2,
                label=r"$B_{\mathrm{simple}}$ at the checkpoint", markeredgecolor="white", markeredgewidth=0.5)
        ax.set_yscale("log", base=2)
        ax.set_yticks([256, 1024, 4096, 16384, 65536, 262144])
        ax.set_yticklabels(["0.25k", "1k", "4k", "16k", "64k", "256k"])
        ax.minorticks_off()
        ax.set_ylim(180, 4.2e5)
        ax.set_xticks([0, 0.25, 0.5, 0.75])
        ax.set_xticklabels(["0%", "25%", "50%", "75%"])
        ax.set_xlim(-0.08, 0.83)
        ax.set_title(f"{NAME[m]}, {CNAME[d]}")
        ax.set_xlabel("checkpoint")
    axes[0].set_ylabel("supervised tokens per step")
    hs, ls = axes[0].get_legend_handles_labels()
    fig.legend(hs, ls, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=0.6)
    save(fig, F, "branched")
    plt.close(fig)

    # Figure: H6, gain from retuning against the lr change
    fig, axes = plt.subplots(1, 2, figsize=(6.3, 2.3), sharey=True)
    moved = [g for g in h6 if g["tuned"] != g["ref"]]
    jit = np.linspace(-0.05, 0.05, 9)
    for ax, (lab, same) in zip(axes, (("same corpus", True), ("the other two corpora", False))):
        ax.axhline(0, color=GREY, lw=0.8, zorder=1)
        ax.axvline(0, color=GREY, lw=0.8, zorder=1)
        xs, ys = [], []
        for i, g in enumerate(moved):
            dx = math.log10(g["tuned"] / g["ref"])
            for j, c in enumerate([g["dataset"]] if same else g["others"]):
                xs.append(dx + jit[(i * 2 + j) % len(jit)])
                ys.append(g["gain"][c])
        ax.scatter(xs, ys, s=14, color=BLUE if same else ORANGE, marker="o" if same else "^", edgecolor="white", linewidth=0.4, zorder=3)
        ax.set_yscale("symlog", linthresh=0.01)
        ax.set_yticks([-0.1, -0.01, 0, 0.01, 0.1, 1])
        ax.set_yticklabels(["$-0.1$", "$-0.01$", "0", "0.01", "0.1", "1"])
        ax.set_title(f"gain on {lab}")
        ax.set_xlabel(r"$\log_{10}$ (retuned lr / reference lr)")
        ax.set_xticks([-2 / 3, -1 / 3, 0, 1 / 3, 2 / 3])
        ax.set_xticklabels([r"$-\frac{2}{3}$", r"$-\frac{1}{3}$", "0", r"$\frac{1}{3}$", r"$\frac{2}{3}$"])
        ax.set_xlim(-0.85, 0.5)
    axes[0].set_ylabel("held-out loss reduction (nats)")
    fig.tight_layout(w_pad=0.8)
    save(fig, F, "transfer")
    plt.close(fig)

    # Appendix figure: loss against lr per B, seed means, relative to each column's best
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.3), sharey=True)
    by = defaultdict(list)
    for r in recs:
        a = r["args"]
        m = a["model"].split("/")[-1]
        if a.get("optim", "adamw") == optim(m) and not a.get("beta2_ref_tokens") and float(a.get("clip", 1.0)) == 0:
            by[(m, a["dataset"], a["tokens_per_step"], a["lr"])].append(r["holdout_loss"])
    for ax, (m, d) in zip(axes, [("SmolLM2-135M", "dolly"), ("SmolLM2-135M", "oasst1"), ("SmolLM2-135M", "gsm8k")]):
        for c, B in zip(RAMP, BS):
            lrs = sorted(k[3] for k in by if k[:3] == (m, d, B))
            ys = np.array([analyze.seed_mean(by[(m, d, B, x)]) for x in lrs])
            ok = np.isfinite(ys)
            ax.plot(np.array(lrs)[ok], ys[ok] - ys[ok].min(), color=c, marker="o", ms=3, lw=1.1, label=f"{B:,}")
        ax.set_xscale("log")
        ax.set_ylim(-0.003, 0.06)
        ax.set_title(f"{NAME[m]}, {CNAME[d]}")
        ax.set_xlabel("peak learning rate")
    axes[0].set_ylabel("gap to the column best (nats)")
    hs, ls = axes[0].get_legend_handles_labels()
    fig.legend(hs, ls, loc="lower center", ncol=7, bbox_to_anchor=(0.55, -0.01))
    fig.text(0.04, 0.03, "$B$ (tokens):", fontsize=7)
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=0.6)
    save(fig, F, "lr_curves")
    plt.close(fig)


if __name__ == "__main__":
    main()
