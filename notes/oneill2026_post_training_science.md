# O'Neill, Jayasekara, Partridge (2026). Post-Training Science for Supervised Fine-Tuning. arXiv:2609.01244v1 (1 Sep 2026, Baseten)

Read 2026-10-24: section 1, all of section 2 (lr and batch, the part that overlaps us), appendix A (batch cost) and B (data stats); sections 3 to 7 (rank/alpha, loss vs downstream, size/data scaling, Muon, epochs) skimmed for batch content and found none beyond section 2.

## Batch-size content (section 2)
- 9 dense models (Qwen3 0.6B to 32B, Llama 3.2-1B/3B, 3.1-8B) plus MoEs, 4 proprietary customer SFT datasets, **5,000 examples, 1 epoch**, cosine decay, assistant-only loss, selection on validation NLL.
- LoRA (r64, alpha 32) is the primary arm: global batch {8, 16, 32, 64} **examples**, lr 3e-5 to 3e-3. **Full FT is a reference arm with only two batch sizes, {16, 64}**, lr 3e-6 to 1e-4.
- Findings: the LoRA lr* is flat at 1e-3 across 0.6B to 32B; FullFT lr* is about 3e-5 (33x lower); "global batch trades loss against cost and has no single best value"; at a fixed one-epoch budget a smaller batch takes more steps and reaches lower loss.
- lr* vs batch: "nearly batch-invariant, drifting up only weakly with batch in Qwen (+0.07 dex per doubling for the smaller models) and not at all in Llama". That is **alpha of about 0.23 for LoRA in Qwen and 0 in Llama**. No exponent for FullFT, which has only 2 batch points.
- Variance decomposition: dataset (token load) explains 56 to 87% of final NLL; batch and lr together under 0.07 each. The "basin" is the lrs within 0.01 nats of the optimum.
- Tokens per example are large: 10^3.6 to 10^4.4, about 4k to 25k tokens, so their batch range of 8 to 64 examples is roughly **30k to 1.6M tokens**, mostly above our 512 to 32k range.
- lr* drifts up with token load (+0.41 dex per decade of tokens per example for FullFT).
- Instability: every unstable run is at the top lr; larger models destabilise first.
- App. A: one-epoch wall-clock is nearly batch-invariant (within 10%) up to the memory cap, so the cost of a small batch is optimiser steps, not time, on an H200.
- They state: "The literature on both is almost entirely about pretraining" (section 2, para 1). PLAN section 2's quote is verified.

## What they do NOT do (the gap holds)
No CBS of any kind (no noise scale, no branching, no "largest B within eps" definition), no lr*(B) exponent for full FT (2 batch points), no trajectory over training, no beta2 or clipping control, proprietary data (not reproducible), and batch counted in examples, not tokens.

## What this changes for Paper 2
1. Section 2 stays: the claim is accurate. Cite them for "SFT optima are reported, not the CBS" and for the LoRA alpha of about 0 to 0.23.
2. **Contrast to make in the paper:** our full-FT pilot gives alpha = 0.52 [0.48, 0.55] on public data, against their LoRA drift of 0 to 0.23. Either full FT and LoRA scale differently with B (consistent with LoRA Without Regret: LoRA is less tolerant of large batches) or the batch ranges differ (ours 0.5k to 32k tokens, theirs about 30k to 1.6M). The Qwen3-1.7B LoRA stretch point could test this directly; consider promoting it.
3. Our tokens-per-step unit is the stronger choice. They show the dataset token load confounds absolute loss and shifts lr*, which supports PLAN's "exact supervised tokens per step" and "compare within corpus".
4. Their 0.01-nat basin vs our eps = 0.005: report CBS at both (Merrill also used 0.01).
5. The wall-clock-is-batch-invariant result does not hold on our 2080 Tis (small-B cells are slower in wall-clock at equal FLOPs, PLAN section 5), and that is worth one sentence: the practical question "should a single-GPU user accumulate?" differs from an H200 cluster's.
