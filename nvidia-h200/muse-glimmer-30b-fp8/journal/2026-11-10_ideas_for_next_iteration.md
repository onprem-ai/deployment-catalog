# 2026-11-10 - Ideas for the Next Iteration

Open points after the first bring-up, benchmark and memory measurement of this card. None is
done yet; each needs the operator's go.

- **Start-up time.** About 7 minutes against 2.5 for BF16 without DFlash: torch.compile 63 s,
  DeepGEMM warm-up and graph capture about 2 minutes, FlashInfer autotune about 1.5 minutes
  (journal 2026-11-10, full benchmark). The caches live in the container. Options: an image
  built with the caches for this model and GPU (quick, tied to one model); a writable cache
  volume the inference node keeps per profile (the general fix, a format and node change, so a
  design first); fewer DeepGEMM warm-up shapes with vLLM's own options (small gain).
- **Draft tokens.** Only 4 and 7 were measured; 7 is the fastest for one user and was chosen.
  The drafter is trained for blocks of 16: try 10 and 16 for one user, and check how far 128
  concurrent requests fall.
- **A throughput alternative.** 4 draft tokens kept 128-request throughput near no DFlash
  (4,943 against 5,141 tokens/s on BF16). A `compose.throughput.yaml` alternative with 4 tokens
  would offer both on one card.
- **Acceptance on real traffic.** The probe's prompts are random word lists. Measure the
  acceptance length on chat, reasoning and code prompts.
- **Parallel tool calls.** The model made one tool call where two were asked for, on every
  setting tried (as on the RTX PRO 6000 card).
- **The BF16 card.** `muse-glimmer-30b-bf16` stays in the draft as the comparison: decide
  whether to publish it beside this card or remove it.
