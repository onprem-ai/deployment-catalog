# 2026-11-28 Cyber vs physical refusal directions ("Cybliterated" feasibility)

**Question:** can refusal be removed for cyber requests only, while the model keeps refusing physical harm
(bio/chem, weapons and explosives, violence)? A card with that edit would be called "Cybliterated" (slug suffix
`-cybl`). This entry measures whether the refusal directions of the categories are far enough apart.

**Verdict: not feasible with direction ablation on the attention outputs.** The refusal directions of all
categories are largely the same direction, and every cyber-specific component that is separable either carries
no refusal or lowers physical refusal as much as cyber refusal. The fully abliterated cards stay as they are; a
category-selective product needs another mechanism (see "What would work instead").

## Measurement design (and why)

- **Original model, not abliterated:** DeepSeek V4.1 Flash 1017, official weights, 4 GPUs, the capture hook of
  `dev/abliteration/deepseek-v4-1-flash-1017/` (`abl41.py`), the same `wo_b` last-token capture as the published
  abliteration, so the directions are comparable with the published one. Rank digests identical on all 4 ranks.
- **Prompts only as inputs:** each prompt went through one prefill with `max_new_tokens` 1; the returned token was
  discarded. Chat and thinking (high) modes, merged with equal weight as in the published abliteration.
- **Categorized harmful prompts** from public sets: HarmBench (MIT, `8e1604d`; standard and contextual
  behaviors), StrongREJECT (MIT, `f7cad6c`), CategoricalHarmfulQA (Apache-2.0, `321e068`). Categories and sizes
  (capture / held-out): cyber 103/24, bio-chem 50/16, weapons 28/16, violence 44/16, fraud 73/24, harassment
  51/24. Weapons, violence and bio-chem are small (public sets have few such prompts); their numbers are less
  certain.
- **Benign sets:** general (Alpaca instructions, as for the published direction; 256/32) and **security work**
  (security.stackexchange questions from the cyber corpus; 160/32), so the cyber direction can be measured both
  against general text and against legitimate security work (topic held constant).
- **Per category:** difference-of-means direction per layer; cosine similarity between categories (median over
  the middle layers 15-33, where the directions are cleanest); the share of the cyber direction perpendicular to
  the physical directions; held-out AUC of each direction (harmful category vs benign).
- **Behavior check (runtime ablation sweep):** the direction is projected out of every `wo_b` output at runtime
  (`y <- y - λ r rᵀy`), held-out prompts, chat mode, temperature 0. **Only refusal labels are kept; no generated
  text was stored, printed or logged.** Physical categories were generated for 24 tokens only (enough to see an
  opening refusal), the others for 128. This misses late refusals, so the physical rates are lower bounds of what
  a long answer would show; the base row is measured the same way and is the comparison.

## Results

**Cosine similarity of the refusal directions** (median over layers 15-33; `general` = the published direction):

| | cyber | chembio | weapons | violence | fraud | harassment | general |
|---|---|---|---|---|---|---|---|
| cyber | 1.00 | 0.89 | 0.86 | 0.87 | 0.90 | 0.83 | 0.87 |
| chembio | 0.89 | 1.00 | 0.88 | 0.87 | 0.84 | 0.81 | 0.83 |
| weapons | 0.86 | 0.88 | 1.00 | 0.96 | 0.95 | 0.85 | 0.86 |
| violence | 0.87 | 0.87 | 0.96 | 1.00 | 0.95 | 0.91 | 0.86 |
| fraud | 0.90 | 0.84 | 0.95 | 0.95 | 1.00 | 0.87 | 0.87 |
| harassment | 0.83 | 0.81 | 0.85 | 0.91 | 0.87 | 1.00 | 0.88 |

Cyber against bio-chem, weapons and violence: 0.86-0.89. The directions are close to one shared refusal
direction, as Arditi et al. (2024) report for other models.

**Perpendicular cyber components:** the cyber direction keeps 38% of its norm
perpendicular to the three physical directions (32% perpendicular to all other harmful
categories). The harmful-cyber minus benign-security contrast keeps 80%.
Held-out AUC (chat, median over layers 15-33):

| Direction | cyber vs benign | cyber vs benign security | bio-chem | weapons | violence | benign security vs benign |
|---|---|---|---|---|---|---|
| cyber | 0.996 | 0.909 | 1.0 | 1.0 | 1.0 | 0.983 |
| cyber, perpendicular to physical | 0.98 | 0.837 | 0.561 | 0.672 | 0.643 | 0.955 |
| cyber, perpendicular to all other | 0.949 | 0.53 | 0.502 | 0.65 | 0.461 | 0.973 |
| harmful-cyber minus benign-security, perpendicular to physical | 0.822 | 1.0 | 0.535 | 0.678 | 0.684 | 0.0 |

The plain perpendicular cyber component mostly encodes the **security topic** (it separates benign security
work from general text as well as harmful cyber), not harmful intent. The harmful-vs-benign-security contrast
separates harmful cyber from security work perfectly and the physical categories only weakly (0.5-0.7), so it was
the best candidate.

**Refusal after removing a direction at runtime** (held-out prompts, refusal labels only):

| Removed direction | λ | cyber | chembio | weapons | violence | fraud | harassment | benign | benign security |
|---|---|---|---|---|---|---|---|---|---|
| none | 0 | 67% | 94% | 100% | 94% | 100% | 96% | 0% | 0% |
| published general direction | 2 | 0% | 0% | 0% | 0% | 0% | 4% | 0% | 0% |
| cyber, perpendicular to physical | 2 | 83% | 94% | 100% | 94% | 100% | 100% | 0% | 3% |
| cyber, perpendicular to physical | 4 | 88% | 100% | 100% | 94% | 100% | 100% | 3% | 9% |
| harmful-cyber minus benign-security, perpendicular to physical | 2 | 42% | 62% | 88% | 56% | 79% | 75% | 0% | 0% |
| harmful-cyber minus benign-security, perpendicular to physical | 4 | 17% | 19% | 6% | 31% | 58% | 25% | 3% | 12% |

- The published general direction removes refusal in every category, as expected.
- Removing the perpendicular cyber component **raises** cyber refusal (67% to 83-88%): without the topic
  information the requests read as plainly harmful.
- Removing the harmful-vs-security component lowers cyber refusal (67% to 42%, 17%), but physical refusal falls
  as much or more (weapons 100% to 6% at λ 4), and false refusals on benign security questions rise to 12%.
  It is not selective.

## What would work instead

- **A guard in front of the model:** keep the fully abliterated model and let a small classifier in front of it
  refuse the physical-harm categories. Independent of the weights, auditable, and the categories can be tuned per
  customer.
- **Conditional steering** (a detector decides per request whether the refusal direction is removed) or
  **fine-tuning** with category-specific data. Both are larger projects than an overlay.

## Limits

Small held-out sets for weapons, violence and bio-chem (16 each); chat mode only for the sweep; refusal detected
from the first 24 or 128 tokens; attention outputs (`wo_b`) only, as in the published abliteration (an edit in
the MoE experts or the Engram path was not tested).
