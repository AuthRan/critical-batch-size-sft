# Lee & Lee (2026). Beware of the Batch Size: Hyperparameter Bias in Evaluating LoRA. arXiv:2602.09492v2 (1 Jun 2026)

Read in full 2026-10-24 (v2, 4 pages + appendix A to J). This was the PLAN section 7 risk: "v2 may contain more than its abstract". It does not overlap with us.

## Setup
- **LoRA only**: LLaMA-2-7B (13B for the size check), ranks 32 to 256; Qwen3-0.6B and Gemma3-1B in App. G. MetaMathQA (first 100k) -> GSM8K accuracy; CodeFeedback -> HumanEval in App. G.
- Fixed-sample protocol, 1 epoch. Batch measured in **samples**, 2^1 to 2^9. lr swept per B over [1e-5, 3e-3] on a {1, 2, 5} x 10^n grid. 3 seeds (lr dynamics in App. E: 1 seed). AdamW, cosine, **no warmup** (they say warmup barely matters in SFT, citing Pareja et al. 2025; the schedule matters about 5 points of accuracy).
- Cost about 1,058 GPU-h per curve.

## Findings
- B alone moves GSM8K accuracy by more than 10 points. The PiSSA vs MiLoRA contradiction is a batch-size crossover: each paper ran the variant at the batch size that favoured it. Tuned vanilla LoRA matches or beats both.
- Accuracy vs B is non-monotone, with an interior optimal B (not "smaller is better").
- lr*(B) for LoRA **rises, then falls** past a threshold (App. E, Fig. 5, one seed): the Surge pattern of Li et al. 2024.
- Optimal B is **invariant to rank and to model size (7B vs 13B)** but grows with dataset size (25k to 200k samples).
- Fixed-steps control (App. B, 200 steps): larger B wins until a threshold. "The degradation in the fixed-sample protocol is partially attributable to insufficient updates."
- Toy theory (Lemma 1): for a rank-r linear regression with random labels at init, E[B_simple] = rN/(r - 2), so the noise scale grows with N, not with r.

## What they do NOT do (the gap holds)
No CBS measurement (direct or branched), no empirical noise scale, no full fine-tuning, no held-out loss (accuracy only), no beta2 or clipping control, no trajectory over training.

## What this changes for Paper 2
1. **Add Pareja et al. (2025)**, "Unveiling the Secret Recipe: A Guide for Supervised Fine-Tuning Small LLMs", to section 2 and section 9. It is the closest prior study of batch size in *full* SFT (large batch with a lower lr helps, by their account). It must be read before the freeze, and section 2's "SFT has almost nothing" needs softening to "SFT has optima and heuristics, not a CBS".
2. **H4 runs against all three sources.** Lee & Lee (LoRA 7B vs 13B), Zhang et al. 2025 and Merrill et al. 2025 (pretraining) all find the batch-size optimum or CBS roughly size-invariant and driven by data. H4 predicts CBS grows 135M < 360M < 0.6B. Re-state it before the freeze as the literature prior (size-invariant within 2x), with the growing ordering as the alternative.
3. Their units are samples. MetaMathQA responses run to a few hundred tokens, so their 2^9 samples is about 100k+ tokens, above our 32k ceiling. Their surge could sit beyond our range, which matters for how we phrase "no knee" from the pilot.
4. They drop warmup because its length in steps changes with B. We keep warmup at 5% of steps, which is 1.6 steps at B = 32k. Name the warmup-in-steps confound in Threats; the beta2 control covers the Adam-state half of it.
5. Their data-size result supports H2b's direction (fixed-data CBS tied to steps per epoch, so smaller corpora give a smaller CBS) and makes gsm8k (7.5k) the setting to watch.
