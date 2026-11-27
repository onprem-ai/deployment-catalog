# 2026-11-26 Initial catalog entry (an extrapolation, nothing run)

**Hardware:** none used. The profile targets four RTX PRO 6000 Blackwell (96 GB, SM120); the
authoring host has two.

## What this is

The two-GPU profile
([`../../ernie-5-5-flash-preview-2x`](../../ernie-5-5-flash-preview-2x/journal/2026-11-26-initial-catalog-entry.md),
which lists the sources) with `--tensor-parallel-size 4`, `--kv-cache-memory-bytes 34359738368`
and `--max-model-len 1048576`. No source reports this model on four SM120 GPUs.

## How the numbers were chosen

- **Weights per GPU, about 40 GiB:** half of the 80.45 GiB the SM120 port reports per GPU at TP2.
  Tensors that tensor parallelism replicates instead of splitting would make it more.
- **Overhead per GPU, about 5 GiB:** the port's TP2 peak (94.75 GiB) minus its weights (80.45)
  and KV cache (9.02).
- **KV cache, 32 GiB per GPU:** below the roughly 47 GiB the two lines above leave, to keep room
  for what they miss. At the two-GPU pool's cost (9.02 GiB per GPU for 1,213,175 tokens) that is
  about 4.3M tokens. This assumes the KV cache is replicated per rank.
- **Booking 94 GiB per GPU:** the whole card. The estimate (about 78 GiB) is not a measurement,
  and a booking below the real peak would let the planner place another replica beside it.
- **Prefill chunk, sequences, DSpark, all-reduce:** left as in the two-GPU profile. With this
  much memory a larger prefill chunk (2,048 or 8,192 tokens) may prefill faster; try it after
  the first start.

## Open questions for the first start

1. Do the port's SM120 kernels and the FlashInfer PCIe all-reduce run with four ranks?
2. What KV pool does the log report for 32 GiB per GPU?
3. Does a 1M-token prefill complete, and how long does it take?
4. Is TP4 faster or slower per request than TP2 (all-reduce is 9–21% of a TP2 step)?
5. Does Baidu's data-parallel layout with expert parallelism start on SM120?

## Checker

Run with the control plane's `check_profile` (platform-alpha `b3643190`): no errors, 24 rules
passed, 4 warnings and 1 hint, all named in the README.
