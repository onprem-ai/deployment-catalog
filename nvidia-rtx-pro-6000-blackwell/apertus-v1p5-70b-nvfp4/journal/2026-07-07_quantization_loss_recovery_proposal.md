# 2026-07-07 -- Proposal: recovering quantization loss (ranked by effort)

**Goal:** Options for closing the quality gap between the three-tier checkpoint
and the FP8/BF16 baseline, ranked by effort. The 8B QAD feasibility journal
(apertus-v15-8b-prerelease/journal/2026-06-19_qad_feasibility_and_proposal.md)
mapped this space; this entry translates it to the 70B.

## 0. First, measure whether recovery is even needed (free, on-node, ~2-4 h)

We know the three-tier tracks FP8 at 94.2% top-1 / median top-10 KL 0.042
(teacher-forced), but not what that costs on real tasks. An lm-eval run
(gsm8k, mmlu-subset) on three-tier vs FP8 quantifies the actual gap. If it is
<= 1 point, the options below may be moot. This is the 8B journal's
"Option C" logic -- ship PTQ if acceptable.

## 1. Calibrated PTQ -- GPTQ on the NVFP4 tier (half a day)

The cheapest real lever not yet pulled: the MLP tier is data-free RTN.
GPTQ-style error-compensated quantization with ~512 calibration samples
typically recovers a meaningful chunk of RTN loss with no training at all.
Requires the calibration-forward pathway (oneshot + sequential onloading), so
a cloud box with ~150+ GB RAM -- a GH200 (480 GB LPDDR +
96 GB HBM) is almost purpose-built for it. Few hours of runtime. Effort:
small -- the existing recipe with GPTQModifier swapped in. Note the oneshot
gate_proj/fused-mapping caveats from
2026-07-07_fused_mapping_crash_and_patch.md may resurface on the oneshot
pathway; verify against the installed llm-compressor version.

## 2. Full QAD (the proper fix) -- cloud only, 2-4 days

The 8B ran QAD on-node because teacher (18 GB) + student + optimizer fit in
96 GB. For the 70B that is impossible locally -- the BF16 teacher alone is
146 GB. It needs an 8x H100/H200 node (teacher sharded + fake-quant student
under FSDP). The encouraging data point from the 8B journal: NVIDIA's
70B-class reference (Llama Nemotron Super) converged with only ~0.3B tokens
-- roughly 150-350 GPU-hours, a weekend of one node. The known hazards carry
over: must run to convergence (Baseten's undercooked run landed 14.5 points
BELOW PTQ), and the mixed NVFP4+FP8 config needs expressing in ModelOpt's QAD
trainer (the 8B three-tier config is the template) -- a day or two of setup.

## 3. On-node-only options: weak

Without room for a teacher (even the FP8 one: 72 + 49 > 96 GB), on-node
recovery is limited to:

- Teacher-free LoRA fine-tuning on task data -- partial recovery, risks
  drifting the model.
- Offline top-k logit distillation -- the full-vocab-logits problem from the
  8B journal (266752-wide logits = ~20 TB), dodged by storing only top-100
  per position (~16 GB of cached logits). Feasible but experimental; no
  reference recipe.

## Recommendation

Step 0 now, then step 1 if the gap is 1-3 points, step 2 only if GPTQ is not
enough. Steps 0+1 together are under a day of effort; they are also
prerequisites for knowing whether the full QAD is justified.

## Outcome

Proposal documented. Decision pending -- step 0 (lm-eval comparison) can run
on-node overnight (requires undeploying qwen36 for the duration, same
GPU-juggling workflow as the quantization runs).
