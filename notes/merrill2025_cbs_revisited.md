# Merrill, Arora, Groeneveld, Hajishirzi (2025). Critical Batch Size Revisited. NeurIPS 2025. arXiv:2505.23971v3

Read in full 2026-10-24 (10 pages + appendix A/B).

## What they do
- Measure the CBS directly by **local branched training**: from a checkpoint trained at base batch B with schedule eta(t), launch branches at k*B with lr f(k)*eta, f(k) = sqrt(k) for Adam (Malladi et al. 2022), linear for SGD. Train each branch for a fixed token window Delta, compare the (EMA-smoothed, alpha = 0.5) training loss L_k.
- CBS definition: k* = max k such that for all k' < k, L_k <= L_k' + eps; B* = k* * B. Interval: [B*, next k above k*], plotted at the geometric mean.
- eps = 0.01 nats ("arbitrarily"; they suggest a statistical test as future work). Delta = 2B tokens against a 600B budget (0.33%).
- Only weak assumption ("local recovery"): if the large-batch branch catches up within Delta, it stays caught up.
- **Optimizer state is carried into the branch**: they expect an immediate loss spike after the batch size changes, which Delta has to absorb.
- No lr sweep per branch: the lr is set by the sqrt rule.
- k grids are chosen per checkpoint (step 0: k = 1/16 to 1/4; later 1 to 8). They say they filtered launched runs to make the k choice systematic (some forking-paths risk, admitted).

## Findings
- OLMo 1B and 7B: CBS about 0 at init, grows fast in the first ~50B tokens, plateaus near 4096 documents (x 4096 tokens = about 16.8M tokens). Similar at both sizes, so CBS looks model-size independent (consistent with Zhang et al. 2024, Bergsma et al. 2025).
- Noise scale (McCandlish App. A, B_big = 64 docs, B_small = 1 doc, ratio of means over 4096 batches, unseen data) **underestimates the CBS by orders of magnitude** and does not track its shape for 7B.
- Application: batch-size warmup 1024 -> 2048 (168B tok) -> 4096 (503B tok) with sqrt lr scaling matches or beats the small-batch control with 43% fewer steps. The thresholds were picked from preliminary CBS measurements (footnote 1).

## Critique / limits relevant to us
- Train-loss based, not held-out. One seed. eps not tied to noise.
- Delta fixed in tokens, so the large-k branches get few steps: the measurement mixes "per-token efficiency" with "steps to absorb the optimizer-state spike".
- Pretraining only, a single data distribution, lr not re-tuned per branch.

## What this changes for Paper 2
1. **Branch protocol.** Follow theirs (carry optimizer state, sqrt-scaled lr, fixed Delta) so the comparison is like for like. Keep PLAN's 3-point lr mini-sweep (x2.15^-1, 1, x2.15 around the sqrt-scaled lr) as a robustness layer. Our pilot alpha = 0.52 [0.48, 0.55] makes sqrt the natural centre.
2. **Held-out loss** instead of smoothed train loss. Our seeds agree to 4e-4 nats, so eps = 0.005 is about 10x the seed spread. Merrill's 0.01 is looser; report both.
3. **Delta in SFT** is the hard part. One dolly epoch is about 1.07M supervised tokens, so a 0.33% window would be under one step at B = 32k. Use Delta = 25% of an epoch (about 270k tokens, 8 steps at 32k, 260 at 1k) and report a Delta-sensitivity point (12.5%). Name this in Threats: the large-B branches are step-starved by construction.
4. **Direction of the proxy error.** In pretraining B_simple << CBS. Our smoke run (B_simple about 99k tokens at step 0) against a pilot fixed-data CBS of about 16k suggests that in SFT the proxy may *over*estimate. If the noise pilot confirms it, that is a headline result: same proxy, opposite sign of error, explained by SFT starting at a low loss (small |G|).
5. **H2b wording.** "degrades because the run has too few steps, not because of gradient noise" is a false dichotomy in McCandlish's own model, where being step-limited *is* the above-CBS regime. The real distinction a branched measurement makes is **startup cost**: warmup measured in steps (5% of 32 steps = 1.6 steps at B = 32k) and Adam moment equilibration (beta2 = 0.95 has a 13.5-step half-life, comparable to the 32-step run). A warm checkpoint does not pay either. Reword H2b before the freeze.
6. Checkpoints for branching must hold the optimizer state: 360M fp32 is about 1.45 GB of weights plus 2.9 GB of Adam state, about 4.3 GB each. 4 checkpoints is 17 GB against about 34 GB free on the GPU box, so branch and then delete, one setting at a time.
