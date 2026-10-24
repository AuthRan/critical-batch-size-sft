# Pareja et al. (2024). Unveiling the Secret Recipe: A Guide for Supervised Fine-Tuning Small LLMs. arXiv:2412.13337v1 (Red Hat / MIT-IBM)

Added to the reading list 2026-10-24 (cited by Lee & Lee 2026 as the full-FT batch-size reference). Read: abstract, section 3 (stacked vs phased, batch size, schedules, TULU cross-check, lr, warmup), appendix A.3 (optimizer, accumulation).

## Setup
- Full fine-tuning of Granite 3B/7B, Llama 3.2 3B, Mistral 7B on large instruction/knowledge/skills mixtures (about 230k to 310k samples per dataset), multi-epoch.
- Effective batch sizes of **128, 3,840 and 7,680 samples** (millions of tokens). Comparisons are made at **about the same number of gradient steps**, so the larger batch also sees more data.
- lr swept 1e-6 to 1e-4 at one batch configuration; warmup {0, 25, 100} steps; constant vs cosine. Adam beta2 = 0.95. Downstream benchmarks only (MMLU, MTBench, Open LLM Leaderboard).

## Findings
- "Larger batch sizes paired with lower learning rates" improve benchmark scores. Small batches reach good performance sooner per sample but plateau earlier.
- Early dynamics (lower gradient norm, higher loss early on) predict a better final model.
- Warmup and cosine decay barely matter; a constant lr is fine (one model, large batch).

## Relation to Paper 2
- The batch-size claim is made under a **fixed-step** protocol, so it measures "more data per step helps", not per-token efficiency, and says nothing about a CBS. Their batches sit 2 to 3 orders of magnitude above our 32k-token ceiling. No lr*(B) relation is fitted (lr is swept at one B), so "larger B, lower lr" is a joint recipe, not a scaling rule, and it is not in tension with our alpha = 0.52 at fixed data.
- Cite it in section 2 as the main counterpoint to "smaller is better" (Marek; LoRA Without Regret). That replaces the sentence "SFT has almost nothing" with a more accurate one: SFT has recipes and optima at a handful of batch sizes, but no CBS, no exponent and no trajectory.
- Their "early gradient norm predicts the outcome" is something we log anyway (pre-clip grad norm per step). A cheap secondary analysis: does the step-0 to 5% gnorm rank lr within a B column?
