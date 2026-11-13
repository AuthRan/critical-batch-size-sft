"""Scores the confirmatory predictions C1-C4 of PLAN.md section 11 (fixed at tag prereg-confirm-v1, before any of its runs).

  python src/confirm.py      (src/paper.py runs it too)

Inputs, all in data/: ledger.jsonl followed by confirm_ledger.jsonl is the GPU machine's runs/ledger.jsonl, so the cell
fits are the ones analyze.py gives on that ledger; noise2_ledger.jsonl holds the original noise runs and
confirm_noise_ledger.jsonl the noise runs of section 11 (one per new corpus, one replication per original setting).
Writes paper/confirm.json, paper/confirm.md, paper/tables/confirm_new.tex and paper/tables/confirm_rep.tex.
"""
from __future__ import annotations

import contextlib
import io
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import analyze  # noqa: E402
import h6_transfer  # noqa: E402

DATA, OUT = Path("data"), Path("paper")
BS = [512, 1024, 2048, 4096, 8192, 16384, 32768]
NEW_MODEL, NEW = "SmolLM2-135M", ["alpaca8k", "codealpaca", "dialogsum"]
MODELS = ["SmolLM2-135M", "SmolLM2-360M", "Qwen3-0.6B-Base"]
CORPORA = ["dolly", "oasst1", "gsm8k"]
NAME = {"SmolLM2-135M": "SmolLM2-135M", "SmolLM2-360M": "SmolLM2-360M", "Qwen3-0.6B-Base": "Qwen3-0.6B"}
CNAME = {"dolly": "Dolly", "oasst1": "OASST1", "gsm8k": "GSM8K", "alpaca8k": "Alpaca-8k", "codealpaca": "CodeAlpaca",
         "dialogsum": "DialogSum"}
LATE = [0.25, 0.5, 0.75, 1.0]  # B_late: the noise points after step 0 (PLAN.md section 11)
NOISE_LR_360M_DOLLY = 4.6e-5  # the original noise lr of SmolLM2-360M x dolly (section 10, item 2(b)), as in paper.py


def ck(m, d, B):
    return f"{m} | {d} | {B} | {h6_transfer.main_optim(m)} | 0 | 0.0"


def gk(m, d):
    return f"{m}|{d}|{h6_transfer.main_optim(m)}|0|0.0"


def kfmt(x) -> str:
    if not np.isfinite(x):
        return "unres."
    return f"{x / 1000:.1f}k" if x < 10_000 else f"{x / 1000:.0f}k"


def analysis(tmp: Path):
    """analyze.py on data/ledger.jsonl + data/confirm_ledger.jsonl, i.e. on the GPU machine's ledger."""
    tmp.mkdir(parents=True, exist_ok=True)
    led = tmp / "ledger.jsonl"
    with open(led, "wb") as f:
        for p in (DATA / "ledger.jsonl", DATA / "confirm_ledger.jsonl"):
            b = open(p, "rb").read()
            f.write(b if b.endswith(b"\n") else b + b"\n")
    with contextlib.redirect_stdout(io.StringIO()):
        analyze.main(str(led), str(tmp))
    return json.load(open(tmp / "cells.json", encoding="utf-8")), json.load(open(tmp / "summary.json", encoding="utf-8"))


def noise_runs(path: Path) -> list[dict]:
    out = []
    for r in analyze.read_ledger(str(path)):
        a = r["args"]
        pts = {round(p["frac"], 2): (p["G2"], p["trSigma"], p["B_simple"]) for p in r.get("noise_scale") or [] if p.get("n_groups")}
        flags = sorted({p.get("flag") for p in r.get("noise_scale") or []})
        out.append({"run_id": r["run_id"], "model": a["model"].split("/")[-1], "dataset": a["dataset"], "seed": a["seed"],
                    "lr": a["lr"], "pts": pts, "flags": flags, "wall_s": r["wall_s"]})
    return out


def b_late(pts: dict) -> float:
    """mean tr(Sigma) / mean |G|^2 over the 25, 50, 75 and 100% points (as paper.ratio_avg)."""
    G2 = np.mean([pts[f][0] for f in LATE])
    return float(np.mean([pts[f][1] for f in LATE]) / G2) if G2 > 0 else float("inf")


def c1_holds(cbs: int, top: bool, bl: float) -> bool:
    """C1: CBS in [B_late / 6, B_late / 1.5]; a CBS at the top of the grid (a lower bound) is consistent if B_late / 1.5 >= 32,768."""
    if top:
        return bl / 1.5 >= BS[-1]
    return bl / 6 <= cbs <= bl / 1.5


def main() -> dict:
    cells, groups = analysis(Path("runs/analysis_confirm"))
    C: dict = {"new": {}, "rep": {}}

    # ---- the three new settings: C1, C2, C3
    new_noise = [r for r in noise_runs(DATA / "confirm_noise_ledger.jsonl") if r["dataset"] in NEW]
    recs = analyze.load(str(DATA / "confirm_ledger.jsonl"))
    rng = np.random.default_rng(0)
    for d in NEW:
        m = NEW_MODEL
        pts = [(B, cells[ck(m, d, B)]) for B in BS]
        g = groups[gk(m, d)]
        lr = {B: f["lr_star"] for B, f in pts}
        Bpk = max(BS, key=lambda B: lr[B])
        peak = Bpk if Bpk < BS[-1] else None
        a = analyze.group_fit([(B, f) for B, f in pts if B <= 4096], rng)
        (run,) = [r for r in new_noise if r["dataset"] == d]
        bl = b_late(run["pts"])
        cbs, top = g["cbs_fixed_data"], g["cbs_upper"] is None
        if bl <= 16384:
            c2_applies, c2 = True, peak is not None and bl / 2 <= peak <= 2 * bl
        elif bl >= 65536:
            c2_applies, c2 = True, peak is None
        else:
            c2_applies, c2 = False, None
        r512 = next(r for r in recs if r["args"]["dataset"] == d and r["args"]["tokens_per_step"] == 512 and r["args"]["seed"] == 1)
        tok = r512["n_steps"] * r512["mean_tokens_per_step"]
        p = run["pts"]
        C["new"][d] = {
            "B_late": bl, "c1_interval": [bl / 6, bl / 1.5], "cbs": cbs, "cbs_upper": g["cbs_upper"], "cbs_eps01": g["cbs_fixed_data_eps01"],
            "ratio_late_cbs": bl / cbs, "c1": c1_holds(cbs, top, bl),
            "peak": peak, "c2_applies": c2_applies, "c2": c2,
            "alpha_le4k": (a["alpha"], a["alpha_lo"], a["alpha_hi"]), "c3": bool(0.40 <= a["alpha"] <= 0.70),
            "alpha_all": (g["alpha"], g["alpha_lo"], g["alpha_hi"]),
            "lr_star": lr, "lr_ci": {B: (f["lr_lo"], f["lr_hi"]) for B, f in pts}, "loss_star": {B: f["loss_star"] for B, f in pts},
            "B_real": {B: f["B_real"] for B, f in pts}, "flag": {B: f["flag"] for B, f in pts}, "seeds": {B: f["seeds"] for B, f in pts},
            "diverged": sum(sum(f["diverged"].values()) for _, f in pts),
            "loss_pen_32k": pts[-1][1]["loss_star"] - min(f["loss_star"] for _, f in pts),
            "B_simple": {f: v[2] for f, v in p.items()}, "G2_drop_0_25": p[0.0][0] / p[0.25][0], "trS_drop_0_25": p[0.0][1] / p[0.25][1],
            "noise_lr": run["lr"], "noise_flags": run["flags"],
            "n_train": r512["n_train"], "n_holdout": r512["n_holdout"], "tokens_epoch": tok, "steps_at_32k": tok / 32768,
        }
    n1 = sum(v["c1"] for v in C["new"].values())
    C["C1"] = "supported" if n1 == 3 else "falsified" if n1 <= 1 else "neither supported nor falsified"
    app = [v for v in C["new"].values() if v["c2_applies"]]
    C["C2"] = "no test" if not app else "falsified" if not all(v["c2"] for v in app) else "not falsified"
    C["C2_applies_in"] = len(app)
    n3 = sum(v["c3"] for v in C["new"].values())
    C["C3"] = "falsified" if n3 <= 1 else "not falsified"
    C["C1_holds_in"], C["C3_holds_in"] = n1, n3

    # ---- C4: replication noise runs for the nine original settings
    orig = noise_runs(DATA / "noise2_ledger.jsonl")
    rep = [r for r in noise_runs(DATA / "confirm_noise_ledger.jsonl") if r["dataset"] in CORPORA]
    for m in MODELS:
        for d in CORPORA:
            at_lr = [r for r in orig if r["model"] == m and r["dataset"] == d
                     and not (m == "SmolLM2-360M" and d == "dolly" and abs(r["lr"] - NOISE_LR_360M_DOLLY) > 1e-9)]
            (o1,) = [r for r in at_lr if r["seed"] == 1]
            (r,) = [x for x in rep if x["model"] == m and x["dataset"] == d]
            assert abs(r["lr"] - o1["lr"]) <= 1e-9 * max(1.0, o1["lr"]) + 1e-12, (m, d, r["lr"], o1["lr"])
            g = groups[gk(m, d)]
            cbs, top = g["cbs_fixed_data"], g["cbs_upper"] is None
            p = r["pts"]
            bl_old, bl_new = b_late(o1["pts"]), b_late(p)
            bl_paper = float(np.mean([b_late(x["pts"]) for x in at_lr]))  # the value in the paper (two seeds for 360M x dolly)
            C["rep"][f"{m}|{d}"] = {
                "seed": r["seed"], "lr": r["lr"], "B_late_seed1": bl_old, "B_late_paper": bl_paper, "B_late_rep": bl_new,
                "ratio": bl_new / bl_old, "a": bool(0.5 <= bl_new / bl_old <= 2),
                "Bsimple_0": p[0.0][2], "Bsimple_25": p[0.25][2], "rise_0_25": p[0.25][2] / p[0.0][2], "b": bool(p[0.25][2] >= 4 * p[0.0][2]),
                "G2": {f: v[0] for f, v in p.items()}, "trSigma": {f: v[1] for f, v in p.items()}, "B_simple": {f: v[2] for f, v in p.items()},
                "G2_drop_0_05": p[0.0][0] / p[0.05][0], "c": bool(p[0.05][0] <= p[0.0][0] / 5),
                "cbs": cbs, "cbs_top": top, "d": c1_holds(cbs, top, bl_new), "c1_seed1": c1_holds(cbs, top, bl_old),
                "flags": r["flags"],
            }
    R = C["rep"].values()
    need = {"a": 8, "b": 7, "c": 7, "d": 7}
    for k, n in need.items():
        got = sum(v[k] for v in R)
        C[f"C4{k}"] = {"holds_in": got, "needed": n, "verdict": "holds" if got >= n else "fails"}
    C["C4a_vs_paper_value"] = sum(0.5 <= v["B_late_rep"] / v["B_late_paper"] <= 2 for v in R)

    # ---- bookkeeping
    nrow = [json.loads(l) for l in open(DATA / "confirm_ledger.jsonl", encoding="utf-8") if l.strip()]
    nn = noise_runs(DATA / "confirm_noise_ledger.jsonl")
    C["runs_grid"], C["runs_noise"] = len(nrow), len(nn)
    C["gpu_hours"] = (sum(r["wall_s"] for r in nrow) + sum(r["wall_s"] for r in nn)) / 3600
    by: dict = {}
    for r in recs:  # as paper.py: the skipped share at each column's best grid lr (lowest seed-mean loss)
        a = r["args"]
        by.setdefault((a["dataset"], a["tokens_per_step"], a["lr"]), []).append(r)
    worst = 0.0
    for d in NEW:
        for B in BS:
            lrs = sorted({k[2] for k in by if k[:2] == (d, B)})
            best = min(lrs, key=lambda x: analyze.seed_mean([r["holdout_loss"] for r in by[(d, B, x)]]))
            worst = max(worst, max(r["skipped_steps"] / r["n_steps"] for r in by[(d, B, best)]))
    C["skipped_at_argmin_max_frac"] = worst
    C["diverged"] = sum(1 for r in recs if not np.isfinite(r["holdout_loss"]))
    ratios = [v["B_late"] / v["cbs"] for v in C["new"].values()]
    C["ratio_late_cbs_new"] = (min(ratios), max(ratios))
    C["noise_flags_all_ok"] = all(r["flags"] == ["ok"] for r in nn)

    # consistency with the paper's main-study numbers, which come from data/ledger.jsonl alone
    if (OUT / "numbers.json").exists():
        st = json.load(open(OUT / "numbers.json", encoding="utf-8"))["settings"]
        for m in MODELS:
            for d in CORPORA:
                s = st[f"{m}|{d}"]
                assert s["cbs_fixed_data"] == C["rep"][f"{m}|{d}"]["cbs"], (m, d)
                assert abs(s["B_late"] - C["rep"][f"{m}|{d}"]["B_late_paper"]) <= 1e-6 * s["B_late"], (m, d)

    OUT.mkdir(exist_ok=True)
    (OUT / "confirm.json").write_text(json.dumps(C, indent=1, default=float), encoding="utf-8")
    write_md(C)
    write_tables(C)
    return C


def yn(x) -> str:
    return "yes" if x else "no"


def write_md(C):
    L = ["# Confirmatory study (PLAN.md section 11)", ""]
    for d, v in C["new"].items():
        a = v["alpha_le4k"]
        L.append(f"- {d}: B_late {v['B_late']:.0f}; C1 interval [{v['c1_interval'][0]:.0f}, {v['c1_interval'][1]:.0f}], CBS {v['cbs']} "
                 f"(next {v['cbs_upper']}) -> {yn(v['c1'])}; peak {v['peak']}, C2 {'n/a' if not v['c2_applies'] else yn(v['c2'])}; "
                 f"alpha_le4k {a[0]:.3f} [{a[1]:.2f}, {a[2]:.2f}] -> {yn(v['c3'])}; alpha all {v['alpha_all'][0]:.3f}; "
                 f"B_late/CBS {v['ratio_late_cbs']:.2f}; |G|^2 0->25% /{v['G2_drop_0_25']:.1f}, trS /{v['trS_drop_0_25']:.2f}; "
                 f"seeds {v['seeds']}; diverged {v['diverged']}; loss pen 32k {v['loss_pen_32k']:.4f}; tokens/epoch {v['tokens_epoch']:.0f}")
        L.append(f"  lr* {', '.join(f'{B}: {x:.3e}' for B, x in v['lr_star'].items())}")
        L.append(f"  loss* {', '.join(f'{B}: {x:.4f}' for B, x in v['loss_star'].items())}")
        L.append(f"  B_simple {', '.join(f'{f}: {x:.0f}' for f, x in v['B_simple'].items())}")
    L.append(f"- C1 {C['C1']} ({C['C1_holds_in']}/3); C2 {C['C2']} (applies in {C['C2_applies_in']}); C3 {C['C3']} ({C['C3_holds_in']}/3)")
    L.append("")
    for k, v in C["rep"].items():
        L.append(f"- {k} (seed {v['seed']}): B_late seed1 {v['B_late_seed1']:.0f}, paper {v['B_late_paper']:.0f}, rep {v['B_late_rep']:.0f}, "
                 f"ratio {v['ratio']:.2f} (a {yn(v['a'])}); B_simple 0 {v['Bsimple_0']:.0f} -> 25% {v['Bsimple_25']:.0f}, x{v['rise_0_25']:.1f} "
                 f"(b {yn(v['b'])}); |G|^2 0/5% x{v['G2_drop_0_05']:.1f} (c {yn(v['c'])}); CBS {v['cbs']}{'*' if v['cbs_top'] else ''} "
                 f"(d {yn(v['d'])}, seed-1 C1 {yn(v['c1_seed1'])})")
        L.append(f"  B_simple {', '.join(f'{f}: {x:.0f}' for f, x in v['B_simple'].items())}; "
                 f"|G|^2 {', '.join(f'{f}: {x:.3g}' for f, x in v['G2'].items())}")
    for k in "abcd":
        v = C[f"C4{k}"]
        L.append(f"- C4({k}) {v['verdict']}: {v['holds_in']}/9 (needs {v['needed']})")
    L.append(f"- C4(a) against the paper's value (two seeds for 360M x dolly): {C['C4a_vs_paper_value']}/9")
    L.append(f"- runs: {C['runs_grid']} grid + {C['runs_noise']} noise, {C['gpu_hours']:.1f} GPU-h; noise flags all ok: {C['noise_flags_all_ok']}; "
             f"max skipped frac at argmin {C['skipped_at_argmin_max_frac']:.4f}; B_late/CBS new {C['ratio_late_cbs_new'][0]:.2f}-{C['ratio_late_cbs_new'][1]:.2f}")
    (OUT / "confirm.md").write_text("\n".join(L) + "\n", encoding="utf-8")


def write_tables(C):
    T = OUT / "tables"
    T.mkdir(parents=True, exist_ok=True)
    rows = []
    for d, v in C["new"].items():
        lo, hi = v["c1_interval"]
        a = v["alpha_le4k"]
        c2 = "--" if not v["c2_applies"] else yn(v["c2"])
        peak = "none" if v["peak"] is None else f"{v['peak']:,}"
        rows.append(f"{CNAME[d]} & {kfmt(v['B_late'])} & [{kfmt(lo)}, {kfmt(hi)}] & {v['cbs']:,} & {yn(v['c1'])} & {peak} & {c2} & "
                    f"{a[0]:.2f} [{a[1]:.2f}, {a[2]:.2f}] & {yn(v['c3'])} \\\\")
    (T / "confirm_new.tex").write_text("\n".join(rows) + "\n\\bottomrule\n", encoding="utf-8")

    rows = []
    for m in MODELS:
        for i, d in enumerate(CORPORA):
            v = C["rep"][f"{m}|{d}"]
            lab = NAME[m] if i == 0 else ""
            x = lambda ok: "" if ok else "$^{\\circ}$"  # fails its criterion
            rows.append(f"{lab} & {CNAME[d]} & {kfmt(v['B_late_seed1'])} & {kfmt(v['B_late_rep'])} & {v['ratio']:.2f}{x(v['a'])} & "
                        f"{v['rise_0_25']:.1f}{x(v['b'])} & {v['G2_drop_0_05']:.1f}{x(v['c'])} & "
                        f"{v['cbs']:,}{'$^{*}$' if v['cbs_top'] else ''} & {yn(v['d'])} \\\\")
        if m != MODELS[-1]:
            rows.append("\\midrule")
    (T / "confirm_rep.tex").write_text("\n".join(rows) + "\n\\bottomrule\n", encoding="utf-8")


if __name__ == "__main__":
    C = main()
    print((OUT / "confirm.md").read_text(encoding="utf-8"))
