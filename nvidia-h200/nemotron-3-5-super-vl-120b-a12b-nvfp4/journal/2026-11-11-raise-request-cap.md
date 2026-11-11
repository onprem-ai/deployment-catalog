# 2026-11-11 - Idea: Raise the 126-Request Limit

**Goal:** Let one H200 replica run more than 126 requests at once. The tuned card
(journal/2026-11-10-h200-tuning.md) reaches 3,729 decode tokens/s at 128 requests, and
the engine queues every request above 126.

## What limits it

At 128 requests and more, vLLM's scheduler shows `Running: 126 reqs, Waiting: N reqs, KV
cache usage: 99.6%`, while the KV cache holds 8,053,536 tokens and the requests use about
0.1 million of them. The pool runs out of blocks, not of tokens:

- vLLM sets the attention block size to 4,352 tokens so that an attention page is as large
  as a Mamba state page (`Setting attention block size to 4352 tokens to ensure that
  attention page size is >= mamba page size`).
- The KV cache has 6 groups of 4,352-token blocks
  (`kv cache group sizes (4352, 4352, 4352, 4352, 4352, 4352)`), and every running
  request holds at least one block per group for its Mamba state, however short it is.
- 38.17 GiB of KV cache in pages of this size gives room for about 126 requests.

## Tried on 2026-11-10 (vLLM v0.32.0)

| Change | Limit | Effect |
|---|---|---|
| `--max-num-seqs` 256 | 126 | none |
| prefix caching off | 126 | group sizes change, limit does not |
| `--mamba-block-size` 1024 | 126 | the attention block stays 4,352 tokens |
| engine booking 124 GiB (from 118) | 146 | 3,402 tokens/s at 128 (4,073 at 118 GiB in the same probe); TTFT p95 15.6 s at 160 |

More memory is the only change that moved the limit, by about 3.3 requests per GiB.

## Ideas, in order

1. **A newer vLLM.** Check the release notes after v0.32.0 for hybrid (Mamba) models: a
   separate pool for Mamba state, or state pages no longer forced to the attention page
   size. This raises the limit without memory or a profile change other than the image.
2. **Smaller Mamba state per request.** `--mamba-cache-dtype float16` (the conv state
   follows the model's BF16 today; the SSM state is already float16) and check that the
   answers stay the same. A smaller state page means smaller blocks and more of them.
3. **More engine memory, measured properly.** Repeat the 124 GiB engine booking with the
   full replica benchmark instead of the probe, to tell whether the drop at 128 requests
   was noise. The memory peak then needs a hardware booking of about 137 GiB, close to the
   GPU's 140.4 GiB, so the margin is the risk.
4. **Shorter maximum context.** `--max-model-len` 262144 instead of the native 1M changes
   how vLLM sizes the groups; check the `kv cache group sizes` line and the limit.

## How to measure each idea

One change per round in the draft, as in the tuning session: redeploy, read `Running: N
reqs` in the engine log at 192 requests, and probe 1, 128, 192 and 256 requests at once.
Keep a change only when the limit rises and the throughput at 128 does not fall. Finish
with the full replica benchmark and a new memory measurement.
