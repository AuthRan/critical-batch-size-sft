"""Corpus loading, formatting, tokenisation and step planning.

Batch size is defined as SUPERVISED TOKENS PER OPTIMIZER STEP, exactly.
No packing. Examples are padded within a micro-batch, micro-batches are
length-bucketed to limit padding, and an optimizer step is a pre-planned
list of micro-batches whose supervised-token count reaches the target.
The loss of a step is the token-mean over the whole step.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import torch

PROMPT = "### Instruction:\n{instruction}\n\n### Response:\n"

HF_SOURCES = {
    "dolly": ("databricks/databricks-dolly-15k", None, "train"),
    "alpaca": ("yahma/alpaca-cleaned", None, "train"),
    "gsm8k": ("openai/gsm8k", "main", "train"),
    "oasst1": ("OpenAssistant/oasst1", None, "train"),
}


def load_pairs(name: str) -> list[dict]:
    """Return a list of {"prompt": str, "response": str} for a named corpus."""
    from datasets import load_dataset

    repo, config, split = HF_SOURCES[name]
    ds = load_dataset(repo, config, split=split) if config else load_dataset(repo, split=split)
    pairs: list[dict] = []
    if name == "dolly":
        for ex in ds:
            instr = ex["instruction"]
            if ex.get("context"):
                instr += "\n\n" + ex["context"]
            pairs.append({"prompt": PROMPT.format(instruction=instr), "response": ex["response"]})
    elif name == "alpaca":
        for ex in ds:
            instr = ex["instruction"]
            if ex.get("input"):
                instr += "\n\n" + ex["input"]
            pairs.append({"prompt": PROMPT.format(instruction=instr), "response": ex["output"]})
    elif name == "gsm8k":
        for ex in ds:
            pairs.append({"prompt": PROMPT.format(instruction=ex["question"]), "response": ex["answer"]})
    elif name == "oasst1":
        # English, first turn only, best-ranked assistant reply to a root prompt.
        by_id = {ex["message_id"]: ex for ex in ds}
        for ex in ds:
            if ex["role"] != "assistant" or ex["lang"] != "en" or ex.get("rank") != 0:
                continue
            parent = by_id.get(ex["parent_id"])
            if parent is None or parent["role"] != "prompter" or parent["parent_id"] is not None:
                continue
            if parent["lang"] != "en":
                continue
            pairs.append({"prompt": PROMPT.format(instruction=parent["text"]), "response": ex["text"]})
    else:
        raise ValueError(f"unknown corpus {name}")
    return [p for p in pairs if p["prompt"].strip() and p["response"].strip()]


@dataclass
class Encoded:
    ids: list[int]
    labels: list[int]  # -100 on prompt tokens; token id on response tokens (+ eos)

    @property
    def length(self) -> int:
        return len(self.ids)

    @property
    def sup_tokens(self) -> int:
        return sum(1 for l in self.labels if l != -100)


def encode_pairs(pairs, tok, max_len: int = 1024, max_response: int = 512) -> list[Encoded]:
    """Tokenise prompt and response separately; supervise response tokens and the eos only."""
    out: list[Encoded] = []
    eos = tok.eos_token_id
    for p in pairs:
        p_ids = tok(p["prompt"], add_special_tokens=False)["input_ids"]
        r_ids = tok(p["response"], add_special_tokens=False)["input_ids"][: max_response - 1] + [eos]
        if len(p_ids) > max_len - 8:
            continue  # the prompt alone fills the window; drop rather than keep a response-free example
        ids = (p_ids + r_ids)[:max_len]
        labels = ([-100] * len(p_ids) + r_ids)[:max_len]
        out.append(Encoded(ids, labels))
    return out


def split_holdout(n: int, n_holdout: int, seed: int = 0) -> tuple[list[int], list[int]]:
    """Fixed held-out split, independent of the run seed. Returns (train_idx, holdout_idx)."""
    rng = random.Random(seed)
    idx = list(range(n))
    rng.shuffle(idx)
    n_holdout = min(n_holdout, n // 5)
    return sorted(idx[n_holdout:]), sorted(idx[:n_holdout])


@dataclass
class Step:
    micro: list[list[int]]  # micro-batches, each a list of example indices
    sup_tokens: int  # realised supervised tokens in this step


def plan_steps(
    enc: list[Encoded],
    tokens_per_step: int,
    micro_token_budget: int,
    seed: int,
    epochs: float = 1.0,
    micro_sup_cap: int | None = None,
) -> list[Step]:
    """Plan every optimizer step in advance.

    Examples are accumulated in shuffled order until a step holds >= tokens_per_step supervised
    tokens, so the overshoot is at most one example's response. Each step is then sorted by length
    and split into micro-batches whose padded size (rows x max_len) stays within micro_token_budget
    and, if micro_sup_cap is given, whose supervised tokens stay within that cap (used by the
    noise-scale estimator to get several micro-batches per big batch).
    The trailing partial step is dropped, so every step carries >= tokens_per_step.
    """
    rng = random.Random(seed)
    n = len(enc)
    order: list[int] = []
    whole, frac = int(epochs), epochs - int(epochs)
    for _ in range(whole):
        idx = list(range(n))
        rng.shuffle(idx)
        order.extend(idx)
    if frac > 0:
        idx = list(range(n))
        rng.shuffle(idx)
        order.extend(idx[: int(n * frac)])

    steps: list[Step] = []
    cur: list[int] = []
    cur_sup = 0
    for i in order:
        cur.append(i)
        cur_sup += enc[i].sup_tokens
        if cur_sup >= tokens_per_step:
            steps.append(Step(_split_micro(enc, cur, micro_token_budget, micro_sup_cap), cur_sup))
            cur, cur_sup = [], 0
    return steps


def _split_micro(enc: list[Encoded], idxs: list[int], micro_token_budget: int, micro_sup_cap: int | None) -> list[list[int]]:
    """Sort a step's examples by length and cut into micro-batches within the padded-token budget."""
    out: list[list[int]] = []
    cur: list[int] = []
    cur_max = cur_sup = 0
    for i in sorted(idxs, key=lambda j: enc[j].length):
        L, s = enc[i].length, enc[i].sup_tokens
        too_wide = bool(cur) and max(cur_max, L) * (len(cur) + 1) > micro_token_budget
        too_many = bool(cur) and micro_sup_cap is not None and cur_sup + s > micro_sup_cap
        if too_wide or too_many:
            out.append(cur)
            cur, cur_max, cur_sup = [], 0, 0
        cur.append(i)
        cur_max = max(cur_max, L)
        cur_sup += s
    if cur:
        out.append(cur)
    return out


def collate(enc: list[Encoded], idxs: list[int], pad_id: int, device) -> dict:
    L = max(enc[i].length for i in idxs)
    ids = torch.full((len(idxs), L), pad_id, dtype=torch.long)
    labels = torch.full((len(idxs), L), -100, dtype=torch.long)
    attn = torch.zeros((len(idxs), L), dtype=torch.long)
    for r, i in enumerate(idxs):
        e = enc[i]
        ids[r, : e.length] = torch.tensor(e.ids)
        labels[r, : e.length] = torch.tensor(e.labels)
        attn[r, : e.length] = 1
    return {"input_ids": ids.to(device), "attention_mask": attn.to(device), "labels": labels.to(device)}
