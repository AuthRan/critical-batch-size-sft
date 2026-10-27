"""CPU-only tests: step planning, the noise-scale estimator on a synthetic problem, and an
end-to-end smoke run of train.py on a tiny randomly initialised Llama. No downloads."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import train as train_mod  # noqa: E402
from data import encode_pairs, plan_steps, split_holdout  # noqa: E402
from noise_scale import collect_rows, fit_rows  # noqa: E402
from analyze import branch_table, cell_fit, group_fit, noise_table, read_ledger  # noqa: E402
import grid_configs  # noqa: E402
import sweep  # noqa: E402


class ByteTok:
    """utf-8 bytes -> ids 1..256; pad 0; eos 257. Mimics the HF tokenizer call signature."""

    eos_token_id = 257
    pad_token_id = 0
    eos_token = "<eos>"
    pad_token = "<pad>"
    vocab_size = 258

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [b + 1 for b in text.encode("utf-8")]}


def synthetic_pairs(n, seed=0):
    rng = np.random.default_rng(seed)
    pairs = []
    for _ in range(n):
        L = int(rng.integers(4, 30))
        s = "".join(chr(97 + int(c)) for c in rng.integers(0, 6, size=L))
        pairs.append({"prompt": f"Q: {s}\nA: ", "response": s[::-1]})
    return pairs


def tiny_model():
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(vocab_size=258, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=256,
                      pad_token_id=0, eos_token_id=257)
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg)


def test_encode_supervises_only_response():
    enc = encode_pairs(synthetic_pairs(5), ByteTok(), max_len=128, max_response=64)
    for e in enc:
        n_prompt = sum(1 for l in e.labels if l == -100)
        assert n_prompt > 0 and e.sup_tokens > 1
        assert e.labels[-1] == 257 and e.ids[-1] == 257  # eos supervised
        assert e.ids[n_prompt:] == e.labels[n_prompt:]


def test_step_plan_hits_token_target():
    enc = encode_pairs(synthetic_pairs(500), ByteTok(), max_len=128, max_response=64)
    for B in (64, 256, 1024):
        steps = plan_steps(enc, B, micro_token_budget=512, seed=1)
        assert steps, B
        sup = np.array([s.sup_tokens for s in steps])
        assert (sup >= B).all()
        assert sup.mean() < B + 64  # overshoot bounded by one example's response
        used = [i for s in steps for mb in s.micro for i in mb]
        assert len(used) == len(set(used))  # each example at most once per epoch
        for s in steps:
            for mb in s.micro:
                assert max(enc[i].length for i in mb) * len(mb) <= 512


def test_holdout_split_is_fixed():
    a = split_holdout(1000, 100, seed=0)
    b = split_holdout(1000, 100, seed=0)
    assert a == b and len(a[1]) == 100 and not set(a[0]) & set(a[1])


def test_diverged_lr_ranks_last():
    # one NaN seed at the top lr: np.argmin would pick the NaN and report lr* = 1e-3
    by_lr = {1e-5: [1.92, 1.92], 2.2e-5: [1.91, 1.91], 4.6e-5: [1.93, 1.93], 1e-4: [1.96, 1.96], 1e-3: [2.75, float("nan")]}
    fit = cell_fit(by_lr, np.random.default_rng(0))
    assert fit["flag"] == "ok" and 1e-5 < fit["lr_star"] < 4.6e-5 and 1e-5 <= fit["lr_lo"] <= fit["lr_hi"] <= 4.6e-5
    assert fit["diverged"] == {"1.0e-03": 1}


def test_lr_star_is_three_point_vertex_with_window_band():
    # measured B=4096 pilot curve, steep above the optimum: the 5-point fit drags the vertex low (4.0e-5 vs 5.2e-5)
    lrs = [1e-5, 2.2e-5, 4.6e-5, 1e-4, 2.2e-4, 4.6e-4]
    losses = [1.9239, 1.9154, 1.9107, 1.9136, 1.9342, 1.9898]
    fit = cell_fit({lr: [v, v] for lr, v in zip(lrs, losses)}, np.random.default_rng(0))
    a, b, _ = np.polyfit(np.log10(lrs[1:4]), losses[1:4], 2)
    assert fit["flag"] == "ok" and abs(np.log10(fit["lr_star"]) + b / (2 * a)) < 1e-9
    assert fit["lr_lo"] <= fit["lr_star_5pt"] < fit["lr_star"] < fit["lr_hi"]  # identical seeds: the band is all window systematic
    assert abs(fit["lr_sys"] - np.log10(fit["lr_star"] / fit["lr_star_5pt"])) < 1e-9


def test_alpha_ignores_edge_cells():
    # an unbracketed cell has a zero-width interval, so inverse-width weighting gave it ~3000x the weight of real cells
    ok = [(B, {"B_real": B, "lr_star": 1e-5 * (B / 512) ** 0.5, "lr_lo": 0.99e-5 * (B / 512) ** 0.5,
               "lr_hi": 1.01e-5 * (B / 512) ** 0.5, "loss_star": 1.91, "flag": "ok"}) for B in (512, 1024, 2048)]
    edge = (8192, {"B_real": 8192, "lr_star": 1e-3, "lr_lo": 1e-3, "lr_hi": 1e-3, "loss_star": 1.91, "flag": "edge"})
    g = group_fit(ok + [edge], np.random.default_rng(0))
    assert abs(g["alpha"] - 0.5) < 1e-6 and g["n_fit"] == 3 and g["excluded"] == [8192]


def test_cbs_compares_against_every_smaller_batch():
    # Merrill et al. 2025: largest B with L_B <= L_B' + eps for every smaller B'. A poor smallest-B cell (1.920)
    # must not set the bar: against it, B=4096 (1.917) passes, but it is 0.007 worse than B=1024.
    loss = {512: 1.920, 1024: 1.910, 2048: 1.916, 4096: 1.917, 8192: 1.930}
    pts = [(B, {"B_real": B, "lr_star": 1e-5, "lr_lo": 1e-5, "lr_hi": 1e-5, "loss_star": v, "flag": "ok"}) for B, v in loss.items()]
    g = group_fit(pts, np.random.default_rng(0))
    assert g["cbs_fixed_data"] == 1024 and g["cbs_upper"] == 2048 and g["cbs_fixed_data_eps01"] == 4096


def test_branch_cbs_per_checkpoint():
    # one checkpoint, B0 = 1024; best-of-3-lr loss per B; B = 8192 is 0.006 worse than B = 2048, so CBS = 4096
    recs = []
    for B, best in {1024: 1.910, 2048: 1.908, 4096: 1.911, 8192: 1.914}.items():
        for m, extra in ((1 / 2.15, 0.004), (1.0, 0.0), (2.15, 0.003)):
            lr = (B / 1024) ** 0.5 * 2e-5 * m
            recs.append({"holdout_loss": best + extra,
                         "args": {"model": "x/M", "dataset": "d", "tokens_per_step": B, "lr": lr, "seed": 1},
                         "branch": {"base_frac": 0.5, "delta_frac": 0.25, "base_B": 1024, "base_lr": 2e-5}})
    rows = branch_table(recs)
    assert len(rows) == 1
    r = rows[0]
    assert r["cbs"] == 4096 and r["cbs_upper"] == 8192 and r["cbs_eps01"] == 8192
    assert r["best_loss"][2048] == 1.908 and all(abs(v - 1.0) < 1e-9 for v in r["best_mult"].values()) and not r["edge"]


def test_ledger_keeps_first_row_per_run(tmp_path):
    # the pre-13:17 sweep could run a job on both GPUs; the second row must not count as an extra seed
    p = tmp_path / "l.jsonl"
    p.write_text("".join(json.dumps({"run_id": rid, "holdout_loss": v}) + "\n"
                         for rid, v in (("a", 1.91), ("b", 1.92), ("a", 1.93))) + "\n", encoding="utf-8")
    assert [(r["run_id"], r["holdout_loss"]) for r in read_ledger(str(p))] == [("a", 1.91), ("b", 1.92)]


def test_noise_scale_recovers_synthetic():
    rng = np.random.default_rng(0)
    d, N = 20, 40000
    G = rng.normal(size=d)
    G *= 0.5 / np.linalg.norm(G)
    per_sample = G + rng.normal(size=(N, d)) * 2.0  # tr(Sigma)=80, |G|^2=0.25 -> B_simple ~ 320
    true_B = per_sample.var(axis=0, ddof=1).sum() / (per_sample.mean(axis=0) ** 2).sum()
    rows, pos = [], 0
    for _ in range(200):
        sizes = [int(x) for x in rng.integers(4, 40, size=8)]
        Btot = sum(sizes)
        acc = np.zeros(d)
        for b in sizes:
            g = per_sample[pos : pos + b].mean(axis=0)
            pos = (pos + b) % (N - 50)
            rows.append((b, float(g @ g)))
            acc += g * (b / Btot)
        rows.append((Btot, float(acc @ acc)))
    fit = fit_rows(rows)
    assert fit["flag"] == "ok"
    assert abs(fit["trSigma"] - 80.0) / 80.0 < 0.05
    assert abs(fit["B_simple"] - true_B) / true_B < 0.3, (fit, true_B)


def test_noise_rows_come_from_round_robin_sub_batches():
    # micro-batches arrive length-sorted, so a row is a sub-batch dealt every n_groups-th micro-batch, not one micro-batch
    torch.manual_seed(0)
    X, y = torch.randn(12, 3), torch.randn(12)
    model = torch.nn.Linear(3, 1, bias=False)

    def loss_fn(m, idx):
        return ((m(X[idx]).squeeze(-1) - y[idx]) ** 2).sum()

    micro = [torch.arange(0, 2), torch.arange(2, 5), torch.arange(5, 6), torch.arange(6, 9), torch.arange(9, 12)]
    rows = collect_rows(model, [[(idx, len(idx)) for idx in micro]], loss_fn, n_groups=2, loss_scale=8.0)

    def g2(idx):
        model.zero_grad()
        (loss_fn(model, idx) / len(idx)).backward()
        return float((model.weight.grad ** 2).sum())

    want = [(6, g2(torch.cat(micro[0::2]))), (6, g2(torch.cat(micro[1::2]))), (12, g2(torch.arange(12)))]
    assert [b for b, _ in rows] == [b for b, _ in want]
    assert np.allclose([v for _, v in rows], [v for _, v in want], rtol=1e-5)


def test_noise_table_skips_v1_points_and_averages_parts():
    pt = lambda f, G2, trS: {"frac": f, "G2": G2, "trSigma": trS, "B_simple": trS / G2 if G2 > 0 else float("inf"),
                             "b_max": 65536, "n_groups": 8}
    v1 = {"frac": 0.5, "G2": -17.9, "trSigma": 9549.0, "B_simple": float("inf"), "b_max": 65735}
    rec = {"args": {"model": "x/M", "dataset": "d", "lr": 1e-4, "seed": 1},
           "noise_scale": [pt(0.0, 0.1, 1e4), v1, pt(1.0, -0.01, 9e3)]}
    (o,) = noise_table([rec])
    assert o["by_frac"] == {0.0: 1e5, 1.0: float("inf")}
    assert abs(o["avg"] - 9.5e3 / 0.045) < 1e-6  # mean tr(Sigma) / mean |G|^2, not a mean of ratios


def test_train_smoke(tmp_path):
    ledger = tmp_path / "ledger.jsonl"
    args = train_mod.parse_args([
        "--model", "tiny", "--dataset", "synthetic", "--tokens_per_step", "256", "--lr", "3e-3",
        "--seed", "1", "--epochs", "2", "--max_len", "128", "--max_response", "64",
        "--micro_token_budget", "512", "--n_holdout", "50", "--ledger", str(ledger),
        "--noise_scale_at", "0,1", "--noise_scale_tokens", "512", "--noise_scale_big_batches", "4",
        "--noise_scale_micro_per_big", "4", "--device", "cpu", "--precision", "fp32", "--log_every", "0",
    ])
    rec = train_mod.run(args, model=tiny_model(), tok=ByteTok(), pairs=synthetic_pairs(400))
    assert rec["n_steps"] > 10
    assert rec["train_loss_last"] < rec["train_loss_first"]
    assert math.isfinite(rec["holdout_loss"])
    assert len(rec["noise_scale"]) == 2
    first, last = rec["noise_scale"]
    assert first["flag"] == "ok" and math.isfinite(first["B_simple"]) and first["B_simple"] > 0
    assert last["flag"] in ("ok", "unresolved")  # near an optimum the signal may be buried; that is flagged, not hidden
    last = json.loads(ledger.read_text(encoding="utf-8").splitlines()[-1])
    assert last["run_id"] == rec["run_id"] and last["holdout_loss"] == rec["holdout_loss"]


def test_plan_keep_partial_uses_every_example():
    enc = encode_pairs(synthetic_pairs(100), ByteTok(), 128, 64)
    full = plan_steps(enc, 300, 512, seed=1, keep_partial=True)
    assert sum(s.sup_tokens for s in full) == sum(e.sup_tokens for e in enc)
    assert sum(s.sup_tokens for s in plan_steps(enc, 300, 512, seed=1)) < sum(e.sup_tokens for e in enc)


def _branch_args(tmp_path, extra):
    return train_mod.parse_args([
        "--model", "tiny", "--dataset", "synthetic", "--seed", "1", "--max_len", "128", "--max_response", "64",
        "--micro_token_budget", "512", "--n_holdout", "50", "--device", "cpu", "--precision", "fp32", "--log_every", "0",
        "--clip", "0", "--ledger", str(tmp_path / "l.jsonl")] + extra)


def test_branch_from_checkpoint(tmp_path):
    from data import split_holdout

    pairs, ck = synthetic_pairs(400), tmp_path / "ck"
    base = train_mod.run(_branch_args(tmp_path, ["--tokens_per_step", "256", "--lr", "3e-3", "--save_at", "0,0.5",
                                                 "--save_dir", str(ck)]), model=tiny_model(), tok=ByteTok(), pairs=pairs)
    assert sorted(p.name for p in ck.iterdir()) == ["f0.00.pt", "f0.50.pt"]
    meta = torch.load(ck / "f0.50.pt", map_location="cpu", weights_only=False)["meta"]
    assert meta["step"] == round(0.5 * base["n_steps"]) and len(set(meta["remaining"])) == len(meta["remaining"])

    # k = 2 branch: the window is the next 25% of the base's tokens, in base order; lr follows the base schedule in tokens
    br = train_mod.run(_branch_args(tmp_path, ["--tokens_per_step", "512", "--lr", str(2 ** 0.5 * 3e-3),
                                               "--init_from", str(ck / "f0.50.pt"), "--delta_frac", "0.25"]),
                       model=tiny_model(), tok=ByteTok(), pairs=pairs)
    b = br["branch"]
    assert "__brf0.50__d0.25" in br["run_id"] and b["base_run_id"] == base["run_id"]
    assert 0.25 * meta["tokens_total"] <= b["window_tokens"] < 0.25 * meta["tokens_total"] + 64
    assert sum(br["curve"]["lr"]) > 0 and br["mean_tokens_per_step"] * br["n_steps"] == b["window_tokens"]
    step_eq = meta["tokens_done"] / meta["tokens_total"] * meta["total"]
    expect = 2 ** 0.5 * 3e-3 * max(0.0, 1 - (step_eq - meta["warmup"]) / (meta["total"] - meta["warmup"]))
    assert abs(br["curve"]["lr"][0] - expect) < 1e-12

    # weights really load: an lr-0 branch started from a perturbed model must score exactly as the checkpoint does
    other = tiny_model()
    with torch.no_grad():
        for p in other.parameters():
            p.add_(0.1)
    still = train_mod.run(_branch_args(tmp_path, ["--tokens_per_step", "512", "--lr", "0", "--init_from",
                                                  str(ck / "f0.50.pt"), "--delta_frac", "0.1"]),
                          model=other, tok=ByteTok(), pairs=pairs)
    ref = tiny_model()
    ref.load_state_dict(torch.load(ck / "f0.50.pt", map_location="cpu", weights_only=False)["model"])
    enc = encode_pairs(pairs, ByteTok(), 128, 64)
    _, hold = split_holdout(len(enc), 50, seed=0)
    want = train_mod.evaluate(ref, [enc[i] for i in hold], range(len(hold)), 0, torch.device("cpu"), 512,
                              __import__("contextlib").nullcontext)
    assert abs(still["holdout_loss"] - want) < 1e-6


def test_fixed_steps_control_truncates(tmp_path):
    args = train_mod.parse_args([
        "--model", "tiny", "--dataset", "synthetic", "--tokens_per_step", "128", "--lr", "1e-3",
        "--max_steps", "7", "--max_len", "128", "--max_response", "64", "--micro_token_budget", "512",
        "--n_holdout", "20", "--ledger", str(tmp_path / "l.jsonl"), "--device", "cpu", "--precision", "fp32",
        "--log_every", "0",
    ])
    rec = train_mod.run(args, model=tiny_model(), tok=ByteTok(), pairs=synthetic_pairs(200))
    assert rec["n_steps"] == 7 and "steps7" in rec["run_id"]


def test_grid_configs_centre_each_column_on_the_sqrt_rule(tmp_path):
    scout = {"models": ["x/M"], "datasets": ["d", "e"], "lr": [grid_configs.point(i) for i in range(-15, -8)],
             "seeds": [1], "common": {"clip": 0}}
    (tmp_path / "scout.json").write_text(json.dumps(scout))
    rows = []

    def add(dataset, B, lr, lr_star, seed=1):
        rid = train_mod.make_run_id("x/M", dataset, B, lr, seed, {"clip": 0})
        rows.append({"run_id": rid, "holdout_loss": 1.9 + (math.log10(lr / lr_star)) ** 2,
                     "args": {"model": "x/M", "dataset": dataset, "tokens_per_step": B, "lr": lr, "seed": seed}})

    for lr in scout["lr"]:
        add("d", 4096, lr, 5e-5)
        add("e", 4096, lr, 3e-6)  # minimum below the scout column: extension only
    rows.append({**rows[0], "run_id": rows[0]["run_id"] + "__b2ref4096", "holdout_loss": 0.0})  # a control never counts
    ledger = tmp_path / "ledger.jsonl"
    ledger.write_text("".join(json.dumps(r) + "\n" for r in rows))
    grid_configs.write(str(ledger), str(tmp_path / "scout.json"), str(tmp_path))

    g = json.loads((tmp_path / "grid_M_d.json").read_text())
    assert abs(g["lr_star_4096"] - 5e-5) < 1e-9
    assert g["lr_by_B"]["4096"] == [4.6e-6, 1e-5, 2.2e-5, 4.6e-5, 1e-4, 2.2e-4, 4.6e-4]
    assert g["lr_by_B"]["512"][3] == 2.2e-5  # 5e-5 / sqrt(8) = 1.8e-5, nearest point 2.2e-5
    assert g["lr_by_B"]["32768"][3] == 1e-4  # 5e-5 * sqrt(8) = 1.4e-4, nearest point 1e-4
    assert json.loads((tmp_path / "noise_M_d.json").read_text())["lr"] == [5e-5]
    assert json.loads((tmp_path / "scout_ext_M_e.json").read_text())["lr"] == [4.6e-6, 2.2e-6]
    assert not (tmp_path / "grid_M_e.json").exists()

    for B, lrs in g["lr_by_B"].items():
        for lr in lrs:
            add("d", int(B), lr, 1e-6 if B == "512" else 5e-5 * (int(B) / 4096) ** 0.5)
    ledger.write_text("".join(json.dumps(r) + "\n" for r in rows))
    grid_configs.write(str(ledger), str(tmp_path / "scout.json"), str(tmp_path), seeds_pass=True)
    s = json.loads((tmp_path / "grid_M_d_seeds.json").read_text())
    assert s["seeds"] == [2, 3] and "512" not in s["lr_by_B"]
    assert s["lr_by_B"]["32768"] == [4.6e-5, 1e-4, 2.2e-4]
    ext = json.loads((tmp_path / "grid_M_d_ext.json").read_text())
    assert ext["lr_by_B"] == {"512": [1e-6, 4.6e-7]}  # minimum at the bottom of the B = 512 column


def test_queue_claims_are_exclusive_and_freed_when_their_sweep_stops(tmp_path):
    import os
    import subprocess
    assert sweep.claim("r1", tmp_path) and not sweep.claim("r1", tmp_path)
    stopped = subprocess.Popen([sys.executable, "-c", "pass"])
    stopped.wait()
    (tmp_path / "r2.claim").write_text(str(stopped.pid))
    assert sweep.pid_alive(os.getpid()) and not sweep.pid_alive(stopped.pid)
    sweep.clear_stale_claims(tmp_path)
    assert (tmp_path / "r1.claim").exists() and not (tmp_path / "r2.claim").exists()
    assert sweep.claim("r2", tmp_path)


def test_sweep_list_file_skips_blanks_and_comments(tmp_path):
    q = tmp_path / "queue.txt"
    q.write_text("configs/a.json\n\n# not yet: configs/b.json\nconfigs/c.json  # extension\n", encoding="utf-8")
    assert sweep.expand(["configs/x.json", str(q)]) == ["configs/x.json", "configs/a.json", "configs/c.json"]
