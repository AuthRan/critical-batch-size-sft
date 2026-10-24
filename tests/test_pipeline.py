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
from noise_scale import fit_rows  # noqa: E402
from analyze import cell_fit, group_fit  # noqa: E402


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


def test_fixed_steps_control_truncates(tmp_path):
    args = train_mod.parse_args([
        "--model", "tiny", "--dataset", "synthetic", "--tokens_per_step", "128", "--lr", "1e-3",
        "--max_steps", "7", "--max_len", "128", "--max_response", "64", "--micro_token_budget", "512",
        "--n_holdout", "20", "--ledger", str(tmp_path / "l.jsonl"), "--device", "cpu", "--precision", "fp32",
        "--log_every", "0",
    ])
    rec = train_mod.run(args, model=tiny_model(), tok=ByteTok(), pairs=synthetic_pairs(200))
    assert rec["n_steps"] == 7 and "steps7" in rec["run_id"]
