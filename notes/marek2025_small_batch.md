# Marek, Lotfi, Somasundaram, Wilson, Goldblum (2025). Small Batch Size Training for Language Models. NeurIPS 2025. arXiv:2507.07101v4

Read 2026-10-24: sections 1 to 2 (half-life definition), 4.3 (Adam scaling), 4.5 (fine-tuning), 5 (recommendations); 4.1, 4.2 and 4.4 (pretraining sweeps, SGD at 1.3B) skimmed.

## Key definitions and results
- Moment half-life in tokens: beta^(t_half / (B*T)) = 1/2. Scaling rule: holding t2 fixed gives **beta2' = beta2^(B'/B)**, which is what our train.py does (`beta2 ** (tokens_per_step / beta2_ref_tokens)`).
- With t2 held fixed (and beta1 = 0.9 kept), batch size 1 matches or beats large batches in pretraining (30M grid; GPT-2 124M; GPT-3 1.3B), and small batches are far more robust to lr, beta1 and beta2 misspecification.
- **lr vs B with t2 fixed: lr* grows "more slowly than a square root"; from B = 1 to 1024 the factor is about 3, not 32.** That is alpha of about 0.16 (30M model, pretraining, B in sequences of 512 tokens).
- Fine-tuning (Gemma 3 4B on MATH, 5 epochs, full FT): Adam at B = 1 with beta2 = 0.997 against B = 16 with beta2 = 0.95 shows no statistically significant difference; Adafactor at B = 1 gives the best memory/performance trade-off. Constant lr in fine-tuning.
- Recommendation: the smallest batch that maximises throughput; avoid gradient accumulation on a single device.

## What this changes for Paper 2
1. **H5 has a literature prior that points the other way.** If alpha falls from our 0.52 (beta2 fixed) to about 0.16 (t2 fixed), then about B_ref = 4,096 lr* at B = 512 and at 32,768 would move by 8^0.36, about **2.1x**. That is above H5's falsification line (2x). H5 as written predicts under 1.3x. Re-state H5 before the freeze with Marek's implied 2x as the competing prediction; control_beta2.json (running now) is the test.
2. Their fine-tuning result is one task, one batch pair, 5 epochs, scored on MATH. It does not measure lr*(B) or a CBS in SFT, so the PLAN section 2 gap ("nobody has checked") stands.
3. Small-B wall-clock: they assume a throughput-saturating micro-batch. On our 2080 Ti a 1,024-token micro-batch is memory-capped (logits), and that is the regime where accumulation is forced. Worth one line in the discussion.
4. At B = 32k our beta2 control uses 0.95^8 = 0.66 (half-life 1.7 steps): Adam becomes close to sign-like there. If the high-lr control runs diverge, that is expected and goes in the report, not a pipeline fault.
