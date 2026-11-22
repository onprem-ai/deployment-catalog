# 2026-11-14 - vLLM Nightly Runs It: Images, DSpark and CUDA Graphs on Two RTX PRO 6000

**Goal:** Try the vLLM nightly of 2026-11-13, whose SM120 sparse MLA prefill was rewritten,
after the text-only start on `deepseekv4-flash-vision` crashed under load (journal
2026-11-14, first start on SM120).

## Setup

- Inference node `ain-xtg6o`, both RTX PRO 6000 Blackwell (SM120), tensor and expert
  parallel 2.
- Image `vllm/vllm-openai:nightly-153dc046462c4724a9a4eb393cadd60169ed943c`
  (`sha256:cfa6d27b...`; vLLM 0.34.0rc1.dev398, FlashInfer 0.7.4).
- Back to the recipe's model: no `--language-model-only` (images on), no
  `--enforce-eager` (CUDA graphs on), DSpark with 3 draft tokens and probabilistic
  sampling, no start-up patch, 32 sequences. Kept: `--disable-custom-all-reduce`,
  `enable_adaptive_verification: false`.
- Served as `deepseek-v4-flash-vision-tuning`.

## Start

Ready on the first attempt: `Model loading took 79.44 GiB` per GPU (vision tower and
DSpark drafter included), `DSpark draft model loaded: 102 params`,
`Available KV cache memory: 10.48 GiB`. The vision tower runs on FlashAttention.

## Answers

27 of 27 through the API gateway: four known-answer questions three times without
reasoning and once at `high` and `max` effort; two parallel tool calls and an answer from
their results; an image (two colors named on their sides); a hidden number at 6,745,
28,480, 115,443 and 216,229 prompt tokens.

## Probe (no tuning yet)

| Requests | Decode tokens/s | Per user | ITL p50 | DSpark acceptance |
|---|---|---|---|---|
| 1 | 156 | 156 | 6.4 ms | 2.72 |
| 8 | 496 | 64 | 15.7 ms | 2.66 |
| 16 | 796 | 51 | 19.7 ms | 2.62 |
| 32 | 1,185 | 40 | 25.3 ms | 2.64 |

No error and no restart at 32 requests, where the old image crashed with two. One request
with a 5,144-token prompt: 0.57 s to the first token (9,031 tokens/s). Peak VRAM 188.36 GiB
over both GPUs; container RAM 7.5 GiB.

Against the 0731 card (its own image, tuned): one user 156 against about 130 tokens/s, 32
requests 1,185 against 1,048 tokens/s, with images on top.

## Rounds

| Round | Change | c=1 | c=32 | c=64 | c=96 |
|---|---|---|---|---|---|
| 0 | recipe: DSpark 3 tokens, 32 sequences | 156 | 1,185 | | |
| 1 | DSpark 5 draft tokens | 146-150 | 1,182 | | |
| 2 | DSpark 3, 64 sequences | 154 | 1,169 | 1,613 | |
| 3 | 96 sequences | 155 | | 1,486 | 2,226 |

Five draft tokens accepted 2.7 to 2.85 per step but cost one-user speed: 3 stays. The
throughput grew with the sequences (KV cache 45 percent used at 96 requests); 96 is chosen,
with one-user speed unchanged. Answers again 27 of 27 on the chosen settings.

## The full benchmark

Job `4b33c069-a1e9-4177-bb97-91a351ad2a58` on replica
`7185d841-18b6-4861-a51d-a374e5c40475`, all three workloads, 05:16:26 to 06:36:18Z, on the
committed settings (`compose_sha256` `2e53b4a9...`). It succeeded: every workload ended at
its plateau and no request failed.

| Workload | Plateau | Figures |
|---|---|---|
| `max-prefill/1.0.0` | 12 | 6,623 prefill tokens/s at 1 (TTFT p50 0.74 s); 10,316 at 32 |
| `max-decode/1.1.0` | 96 | 164 decode tokens/s at 1 (ITL p50 5.9 ms); 1,387 at 32; 2,118 at 64; 2,613 at 96 |
| `mixed-load/1.1.0` | 96 | 135 decode tokens/s at 1; at 96: 6,679 prefill and 1,168 decode tokens/s |

Against the tuned 0731 card on the same GPUs (its own image): 164 against 199 decode
tokens/s for one request in `max-decode`, 2,613 at 96 against 1,418 at its plateau of 32,
and image input on top.

## GPU memory

The highest `vram_used_bytes` of the replica, read every 30 s during the benchmark:
101,147,738,112 bytes (94.20 GiB) on each GPU, within the 95 GiB hardware booking (0.8 GiB
headroom); engine booking 93 GiB (0.9792).

| Part | Bytes | GiB | Source (TP0's start-up lines) |
|---|---|---|---|
| `model_weights` | 85,341,000,171 | 79.48 | `Model loading took 79.48 GiB` (vision tower and DSpark drafter included) |
| `cuda_graphs` | 2,297,807,503 | 2.14 | `Graph capturing finished ... took 2.14 GiB` |
| `kv_cache` | 11,166,914,969 | 10.40 | `Available KV cache memory: 10.4 GiB` |
| `runtime_overhead` | 2,342,015,469 | 2.18 | the peak minus every other part |

`GPU KV cache size: 265,715 tokens` (1.01 requests of 262,144 tokens).

## Next

- The image is a nightly. A vLLM release with the same SM120 kernels replaces it once one
  exists; until then the card stays `experimental`.
- The card on the development control plane (`ain-4euk3`) still holds the text-only
  settings; it takes these settings when this copy is final.
