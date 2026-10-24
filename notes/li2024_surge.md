# Li et al. (2024). Surge Phenomenon in Optimal Learning Rate and Batch Size Scaling. NeurIPS 2024. arXiv:2405.14578v5

Read 2026-10-24: abstract, section 1, section 2 theorems (statements; proofs skimmed), section 3.2 (estimation), conclusions.

## Result
- Treating Adam as a sign-of-gradient update, eps_opt(B) = eps_max / (0.5 * (sqrt(B_noise / B) + sqrt(B / B_noise))).
  - B << B_noise: eps_opt is about 2 * eps_max * sqrt(B / B_noise), i.e. the **square-root rule** (SGD gives linear).
  - **Peak at B_peak = B_noise**, then lr* falls as B grows, and for very large B it rises again to an asymptote.
  - **B_noise (and the peak) moves to larger B as training progresses** and the loss falls.
- B_noise is the Hessian-weighted noise scale (McCandlish B_noise, not B_simple). They estimate it by curve fitting the grid results, not from gradient statistics.
- Verified on CNN/FashionMNIST, ResNet18/TinyImageNet, DistilGPT2/ELI5, a MoE on RedPajama.

## What this changes for Paper 2
1. **Cross-hypothesis prediction (H1 and H2).** Our pilot gives alpha = 0.52 with no knee up to B = 32k, which under Surge means B_noise > 32k there; the smoke run's step-0 B_simple of about 99k agrees. Pre-register: in each setting, a knee in lr*(B) should appear inside the grid only where the run-averaged B_simple from the noise runs is below 32k, near B_simple (within 2x, allowing for B_noise vs B_simple). Where B_simple > 32k, lr* keeps rising with alpha near 0.5. This tests both the proxy and the Surge theory in SFT, which no one has done.
2. Lee & Lee (2026) see the surge for LoRA (one seed). If full FT keeps rising while LoRA turns over at a comparable B, that is consistent with LoRA's smaller effective B_noise and with LoRA Without Regret's "LoRA tolerates large batches less".
3. Because the peak drifts right during training, a fixed-data 1-epoch lr*(B) is an average over the run. The branched measurement at 25/50/75% can show the drift directly (lr* per branch from the 3-point mini-sweep).
