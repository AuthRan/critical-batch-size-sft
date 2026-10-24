"""One supervised fine-tuning run. Appends one JSON line to the ledger when done.

Example:
  python src/train.py --model HuggingFaceTB/SmolLM2-135M --dataset dolly \
      --tokens_per_step 4096 --lr 1e-4 --seed 1 --noise_scale_at 0,0.5,1
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import math
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import HF_SOURCES, collate, encode_pairs, load_pairs, plan_steps, split_holdout  # noqa: E402
from noise_scale import collect_rows, fit_rows  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True, help=f"one of {sorted(HF_SOURCES)} (or 'synthetic' in tests)")
    p.add_argument("--tokens_per_step", type=int, required=True, help="supervised tokens per optimizer step")
    p.add_argument("--lr", type=float, required=True)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--epochs", type=float, default=1.0)
    p.add_argument("--max_steps", type=int, default=0, help="if >0 truncate the plan (fixed-steps control)")
    p.add_argument("--micro_token_budget", type=int, default=4096, help="max padded tokens per micro-batch")
    p.add_argument("--max_len", type=int, default=1024)
    p.add_argument("--max_response", type=int, default=512)
    p.add_argument("--n_holdout", type=int, default=1000)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--beta2_ref_tokens", type=int, default=0,
                   help="if >0, beta2 = beta2 ** (tokens_per_step / ref): fixed half-life in tokens (Marek et al. 2025)")
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_frac", type=float, default=0.05)
    p.add_argument("--decay", choices=["linear", "cosine", "constant"], default="linear")
    p.add_argument("--clip", type=float, default=1.0, help="grad-norm clip; 0 disables")
    p.add_argument("--optim", choices=["adamw", "adamw8bit"], default="adamw")
    p.add_argument("--precision", choices=["auto", "fp32", "bf16", "fp16"], default="auto")
    p.add_argument("--grad_checkpoint", action="store_true")
    p.add_argument("--cross_eval", default="", help="comma-separated other corpora to score the final model on")
    p.add_argument("--noise_scale_at", default="", help="comma-separated fractions of training, e.g. 0,0.1,0.25,0.5,0.75,1")
    p.add_argument("--noise_scale_tokens", type=int, default=8192, help="supervised tokens per big batch in the estimator")
    p.add_argument("--noise_scale_big_batches", type=int, default=8)
    p.add_argument("--noise_scale_micro_per_big", type=int, default=8)
    p.add_argument("--ledger", default="runs/ledger.jsonl")
    p.add_argument("--run_id", default="")
    p.add_argument("--save_dir", default="", help="if set, save the final model here (for branched CBS)")
    p.add_argument("--save_at", default="",
                   help="with --save_dir: comma-separated fractions of the run at which to save a branch checkpoint "
                        "(weights, optimizer, scaler, data cursor) instead of the final model")
    p.add_argument("--init_from", default="", help="branch: start from this --save_at checkpoint (Merrill et al. 2025)")
    p.add_argument("--delta_frac", type=float, default=0.25,
                   help="branch: train on the next delta_frac of the base run's supervised tokens, in base order")
    p.add_argument("--wait_for_init_h", type=float, default=0.0,
                   help="branch: wait up to this many hours for --init_from to appear (the base run may be on the other GPU)")
    p.add_argument("--device", default="")
    p.add_argument("--log_every", type=int, default=50)
    return p.parse_args(argv)


def make_run_id(model: str, dataset: str, tokens_per_step: int, lr: float, seed: int, extra: dict | None = None) -> str:
    short = model.split("/")[-1]
    rid = f"{short}__{dataset}__B{tokens_per_step}__lr{lr:.2e}__s{seed}"
    extra = extra or {}
    if extra.get("max_steps"):
        rid += f"__steps{extra['max_steps']}"
    if extra.get("init_from"):
        rid += f"__br{Path(extra['init_from']).stem}__d{extra.get('delta_frac', 0.25):g}"
    if extra.get("beta2_ref_tokens"):
        rid += f"__b2ref{extra['beta2_ref_tokens']}"
    if extra.get("clip", 1.0) == 0:
        rid += "__noclip"
    if extra.get("optim", "adamw") != "adamw":
        rid += f"__{extra['optim']}"
    return rid


def lr_at(step: int, total: int, warmup: int, peak: float, decay: str) -> float:
    if step < warmup:
        return peak * (step + 1) / warmup
    prog = (step - warmup) / max(1, total - warmup)
    if decay == "linear":
        return peak * max(0.0, 1.0 - prog)
    if decay == "cosine":
        return peak * 0.5 * (1.0 + math.cos(math.pi * prog))
    return peak


def sum_loss(model, batch) -> torch.Tensor:
    """Sum of token cross-entropies over supervised positions, computed in fp32."""
    out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
    logits = out.logits[:, :-1, :].float()
    labels = batch["labels"][:, 1:]
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)), labels.reshape(-1), ignore_index=-100, reduction="sum"
    )


@torch.no_grad()
def evaluate(model, enc, idxs, pad_id, device, micro_token_budget, autocast_ctx) -> float:
    """Token-mean held-out loss over the given examples."""
    model.eval()
    idxs = sorted(idxs, key=lambda i: enc[i].length)
    total, n = 0.0, 0
    cur: list[int] = []
    cur_max = 0

    def flush():
        nonlocal total, n, cur, cur_max
        if not cur:
            return
        batch = collate(enc, cur, pad_id, device)
        with autocast_ctx():
            total += float(sum_loss(model, batch))
        n += sum(enc[i].sup_tokens for i in cur)
        cur, cur_max = [], 0

    for i in idxs:
        L = enc[i].length
        if cur and max(cur_max, L) * (len(cur) + 1) > micro_token_budget:
            flush()
        cur.append(i)
        cur_max = max(cur_max, L)
    flush()
    model.train()
    return total / max(1, n)


def load_model_and_tok(name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    try:
        model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
    except TypeError:  # older transformers
        model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float32)
    return model, tok


def run(args, model=None, tok=None, pairs=None) -> dict:
    t0 = time.time()
    torch.manual_seed(args.seed)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    prec = args.precision
    if prec == "auto":
        native_bf16 = device.type == "cuda" and torch.cuda.get_device_capability(device)[0] >= 8
        prec = "fp32" if device.type == "cpu" else ("bf16" if native_bf16 else "fp16")
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(prec)

    def autocast_ctx():
        if amp_dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(device_type=device.type, dtype=amp_dtype)

    scaler = torch.amp.GradScaler(device.type, enabled=(prec == "fp16"))

    if model is None or tok is None:
        model, tok = load_model_and_tok(args.model)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    pad_id = tok.pad_token_id
    model.to(device)
    if args.grad_checkpoint:
        model.gradient_checkpointing_enable()
        model.config.use_cache = False
    model.train()

    if pairs is None:
        pairs = load_pairs(args.dataset)
    enc = encode_pairs(pairs, tok, args.max_len, args.max_response)
    train_idx, hold_idx = split_holdout(len(enc), args.n_holdout, seed=0)
    train_enc = [enc[i] for i in train_idx]
    hold_enc = [enc[i] for i in hold_idx]

    # Branch: same weights, optimizer state and data order as the base run at its checkpoint; the next
    # delta_frac of the base's tokens, re-planned at this run's B with the trailing partial step kept.
    branch, plan_enc = None, train_enc
    if args.init_from:
        deadline = time.time() + 3600 * args.wait_for_init_h
        while not Path(args.init_from).exists() and time.time() < deadline:
            time.sleep(30)
        branch = torch.load(args.init_from, map_location="cpu", weights_only=False)
        meta = branch["meta"]
        if meta["n_train"] != len(train_enc):
            raise RuntimeError(f"checkpoint was made on {meta['n_train']} training examples, this run has {len(train_enc)}")
        model.load_state_dict(branch["model"])
        budget, window, window_tokens = args.delta_frac * meta["tokens_total"], [], 0
        for i in meta["remaining"]:
            if window_tokens >= budget:
                break
            window.append(i)
            window_tokens += train_enc[i].sup_tokens
        plan_enc = [train_enc[i] for i in window]
        steps = plan_steps(plan_enc, args.tokens_per_step, args.micro_token_budget, args.seed, 1.0, keep_partial=True)
    else:
        steps = plan_steps(train_enc, args.tokens_per_step, args.micro_token_budget, args.seed, args.epochs)
    if args.max_steps > 0:
        steps = steps[: args.max_steps]
    total = len(steps)
    if total == 0:
        raise RuntimeError("the plan has zero steps: tokens_per_step is larger than the data budget")
    warmup = max(1, int(round(args.warmup_frac * total)))
    beta2 = args.beta2 if args.beta2_ref_tokens <= 0 else args.beta2 ** (args.tokens_per_step / args.beta2_ref_tokens)

    params = [p for p in model.parameters() if p.requires_grad]
    if args.optim == "adamw8bit":
        import bitsandbytes as bnb

        opt = bnb.optim.AdamW8bit(params, lr=args.lr, betas=(args.beta1, beta2), weight_decay=args.weight_decay)
    else:
        opt = torch.optim.AdamW(params, lr=args.lr, betas=(args.beta1, beta2), weight_decay=args.weight_decay,
                                fused=(device.type == "cuda"))
    if branch is not None:
        opt.load_state_dict(branch["opt"])  # Merrill et al. carry the optimizer state into the branch
        if branch["scaler"]:
            scaler.load_state_dict(branch["scaler"])
        branch = {"meta": branch["meta"]}  # drop the host copy of the weights and optimizer state (~4 GB at 360M)

    save_steps: dict[int, float] = {}
    if args.save_at:
        if not args.save_dir:
            raise RuntimeError("--save_at needs --save_dir")
        save_steps = {int(round(f * total)): f for f in (float(x) for x in args.save_at.split(",") if x.strip())}

    def save_checkpoint(s: int):
        path = Path(args.save_dir) / f"f{save_steps[s]:.2f}.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"base_run_id": run_id, "step": s, "total": total, "warmup": warmup, "decay": args.decay, "lr": args.lr,
                "tokens_per_step": args.tokens_per_step, "seed": args.seed, "n_train": len(train_enc),
                "tokens_done": sum(st.sup_tokens for st in steps[:s]), "tokens_total": sum(st.sup_tokens for st in steps),
                "remaining": [i for st in steps[s:] for mb in st.micro for i in mb]}
        tmp = path.with_suffix(".tmp")  # write then rename, so a branch never reads a half-written file
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(), "meta": meta}, tmp)
        tmp.replace(path)
        saved.append(str(path))
        print(f"  saved branch checkpoint at step {s}/{total} -> {path}", flush=True)

    # Noise-scale measurement: fixed-size big batches drawn with a separate seed, independent of the run's B.
    ns_steps: set[int] = set()
    ns_batches = []
    if args.noise_scale_at:
        fracs = [float(x) for x in args.noise_scale_at.split(",") if x.strip()]
        ns_steps = {min(total, int(round(f * total))) for f in fracs}
        cap = max(1, args.noise_scale_tokens // args.noise_scale_micro_per_big)
        plan = plan_steps(train_enc, args.noise_scale_tokens, args.micro_token_budget, seed=args.seed + 1000,
                          epochs=1.0, micro_sup_cap=cap)
        for st in plan[: args.noise_scale_big_batches]:
            ns_batches.append([(collate(train_enc, mb, pad_id, device), sum(train_enc[i].sup_tokens for i in mb))
                               for mb in st.micro])

    def measure_ns(step: int) -> dict:
        def loss_fn(m, batch):
            with autocast_ctx():
                return sum_loss(m, batch)

        rows = collect_rows(model, ns_batches, loss_fn, params=params, loss_scale=(1024.0 if prec == "fp16" else 1.0))
        r = fit_rows(rows)
        r["step"] = step
        r["frac"] = step / max(1, total)
        print(f"  noise scale at step {step}/{total}: B_simple={r['B_simple']:.0f} tokens [{r['flag']}] "
              f"(|G|^2={r['G2']:.3e}, trS={r['trSigma']:.3e}, b_max={r['b_max']:.0f})", flush=True)
        return r

    run_id = args.run_id or make_run_id(args.model, args.dataset, args.tokens_per_step, args.lr, args.seed, vars(args))
    print(f"run {run_id}: {total} steps, mean {sum(s.sup_tokens for s in steps) / total:.0f} sup tokens/step, "
          f"{len(train_enc)} train / {len(hold_enc)} held-out examples, {prec} on {device}", flush=True)

    curve_step, curve_loss, curve_gnorm, curve_lr = [], [], [], []
    ns_results: list[dict] = []
    saved: list[str] = []
    clipped = skipped = 0
    done_tokens = 0
    for s, st in enumerate(steps):
        if s in ns_steps:
            ns_results.append(measure_ns(s))
        if s in save_steps:
            save_checkpoint(s)
        if branch is not None:  # the base schedule's shape at this token position, times this branch's peak lr
            m = branch["meta"]
            pos = (m["tokens_done"] + done_tokens) / m["tokens_total"] * m["total"]
            lr = lr_at(pos, m["total"], m["warmup"], args.lr, m["decay"])
        else:
            lr = lr_at(s, total, warmup, args.lr, args.decay)
        done_tokens += st.sup_tokens
        for g in opt.param_groups:
            g["lr"] = lr
        step_loss = 0.0
        for mb in st.micro:
            batch = collate(plan_enc, mb, pad_id, device)
            with autocast_ctx():
                loss = sum_loss(model, batch) / st.sup_tokens
            scaler.scale(loss).backward()
            step_loss += float(loss.detach())
        scaler.unscale_(opt)
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, args.clip if args.clip > 0 else float("inf")))
        if args.clip > 0 and gnorm > args.clip:
            clipped += 1
        scale_before = scaler.get_scale() if prec == "fp16" else None
        scaler.step(opt)
        scaler.update()
        if prec == "fp16" and scaler.get_scale() < scale_before:
            skipped += 1
        opt.zero_grad(set_to_none=True)
        curve_step.append(s)
        curve_loss.append(step_loss)
        curve_gnorm.append(gnorm)
        curve_lr.append(lr)
        if args.log_every and (s % args.log_every == 0 or s == total - 1):
            print(f"  step {s}/{total} loss {step_loss:.4f} gnorm {gnorm:.3f} lr {lr:.2e}", flush=True)
    if total in ns_steps:
        ns_results.append(measure_ns(total))

    hold_loss = evaluate(model, hold_enc, list(range(len(hold_enc))), pad_id, device, args.micro_token_budget, autocast_ctx)
    print(f"  held-out loss ({args.dataset}): {hold_loss:.4f}", flush=True)
    cross: dict[str, float] = {}
    for name in [x.strip() for x in args.cross_eval.split(",") if x.strip()]:
        if name == args.dataset:
            continue
        c_enc = encode_pairs(load_pairs(name), tok, args.max_len, args.max_response)
        _, c_hold = split_holdout(len(c_enc), args.n_holdout, seed=0)
        cross[name] = evaluate(model, c_enc, c_hold, pad_id, device, args.micro_token_budget, autocast_ctx)
        print(f"  held-out loss ({name}): {cross[name]:.4f}", flush=True)

    if args.save_dir and not args.save_at:
        Path(args.save_dir).mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.save_dir)
        if hasattr(tok, "save_pretrained"):
            tok.save_pretrained(args.save_dir)

    k = max(1, min(5, total // 10))
    thin = max(1, total // 500)
    rec = {
        "run_id": run_id,
        "args": vars(args),
        "device": str(device),
        "precision": prec,
        "beta2_used": beta2,
        "n_train": len(train_enc),
        "n_holdout": len(hold_enc),
        "n_steps": total,
        "warmup_steps": warmup,
        "mean_tokens_per_step": sum(s.sup_tokens for s in steps) / total,
        "holdout_loss": hold_loss,
        "cross": cross,
        "train_loss_first": sum(curve_loss[:k]) / k,
        "train_loss_last": sum(curve_loss[-k:]) / k,
        "clipped_frac": clipped / total,
        "skipped_steps": skipped,
        "noise_scale": ns_results,
        "saved": saved,
        "branch": None if branch is None else {
            "base_run_id": branch["meta"]["base_run_id"], "base_frac": branch["meta"]["step"] / branch["meta"]["total"],
            "base_B": branch["meta"]["tokens_per_step"], "base_lr": branch["meta"]["lr"], "delta_frac": args.delta_frac,
            "window_examples": len(plan_enc), "window_tokens": sum(e.sup_tokens for e in plan_enc)},
        "curve": {"step": curve_step[::thin], "loss": curve_loss[::thin], "gnorm": curve_gnorm[::thin], "lr": curve_lr[::thin]},
        "wall_s": time.time() - t0,
        "peak_mem_gb": (torch.cuda.max_memory_allocated(device) / 2**30) if device.type == "cuda" else None,
        "finished": dt.datetime.now().isoformat(timespec="seconds"),
    }
    Path(args.ledger).parent.mkdir(parents=True, exist_ok=True)
    with open(args.ledger, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"  done in {rec['wall_s']:.0f}s -> {args.ledger}", flush=True)
    return rec


if __name__ == "__main__":
    run(parse_args())
