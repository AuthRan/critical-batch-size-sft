# Zhang, Morwani, Vyas, Wu, Zou, Ghai, Foster, Kakade (2024/25). How Does Critical Batch Size Scale in Pre-training? ICLR 2025. arXiv:2410.21676v4

Read 2026-10-24: abstract, introduction and contributions, theorem statements; the experimental appendix was not read.

## Result
- CBS is defined as the batch size where the steps-to-target-loss overhead relative to linear scaling crosses a threshold (doubling B no longer halves the steps). Models from 85M to 1.2B on C4, with full sweeps over B, momentum, lr and schedule.
- **CBS scales with data size D, and is nearly invariant in model size N at fixed D** (Fig. 1 right: fixed D = 3.07B, N varied). In the Chinchilla setting it rises only because D rises with N.
- Theory: in the infinite-width (muP) limit, dynamics stop depending on width, so CBS saturates in N. For least squares with power-law source/capacity, CBS grows with D.
- They stress the hyperparameter choices that matter beyond fixed training durations (e.g. EWA, schedule-free comparisons).

## What this changes for Paper 2
1. **H4 prior.** At fixed data the pretraining CBS is model-size invariant. The same holds for LoRA SFT (Lee & Lee 2026, 7B vs 13B) and OLMo 1B vs 7B (Merrill et al. 2025). Our corpora are fixed in D across the size ladder, so the literature predicts flat CBS across 135M / 360M / 0.6B. H4 v0 predicted growth. Re-state H4 before the freeze with invariance as the prediction.
2. **Data size is the lever.** dolly (15k), oasst1 (about 10k) and gsm8k (7.5k) differ in D by about 2x in examples and more in supervised tokens. Zhang predicts the fixed-data CBS orders by D. That is a cheap secondary test (3 corpora x 3 models, ordering only) and matches H2b's "smaller corpora, bigger gap".
3. Their CBS is steps-to-target-loss; ours is loss-at-fixed-tokens within eps. Merrill notes the two are dual views. State this in the paper.
