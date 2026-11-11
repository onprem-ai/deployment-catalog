# 2026-11-10 - First Deployment and Concurrency Tuning on H200

**Goal:** Bring up Kolibri 1 FP8 with the full 1,048,576-token context on one H200, check
its capabilities and long context, and choose the settings for the full benchmark.

## Download

The node's disk had 68 GB free for the 79 GB download; the node said so in the transfer
error (code `download-failed`, retryable, with an error report). The weights of the
Muse Glimmer BF16 card, deployed nowhere, were removed from the node
(`deploy_profiles_remove`, both catalogs that requested them), which left 126 GB, and the
download continued on its own.

## First deployment (128 sequences, 8,192 batched tokens)

Ready 6 minutes after the deploy, no restart. From the deployment log:

- `Using Transformers modeling backend.`
- `Using TRITON Fp8 MoE backend` and `Selected FlashInferFp8DeepGEMMDynamicBlockScaledKernel for
  Fp8LinearMethod`. vLLM has no tuned Triton MoE configuration for this expert shape on H200
  (`E=384,N=512,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json`) and uses its
  default. DeepGEMM's grouped MoE kernel is not used: vLLM skips it for N <= 512, and Kolibri's
  experts have N = 512.
- `Model loading took 73.45 GiB`, graph capture 0.47 GiB, `Available KV cache memory: 44.6 GiB`,
  `GPU KV cache size: 4,393,516 tokens` (4.19 requests of 1,048,576 tokens), peak activation
  1.31 GiB, engine budget 119.99 GiB (`--gpu-memory-utilization 0.8583`).
- The FP8 KV cache runs with scale 1.0: the checkpoint has no KV scales (vLLM warns). Aleph
  Alpha evaluated the model with an FP8 KV cache.

## Capabilities (through the API gateway)

- Reasoning efforts: `none` answers with no reasoning (4 tokens), `low` 48 characters of
  reasoning, `high` 1,028 characters; all answer 391 to 17 x 23. Set with
  `chat_template_kwargs: {"reasoning_effort": ...}`.
- Tool calls: two parallel calls (Zuerich and Oslo) in one answer, and a correct German
  answer from a tool result. `/v1/messages` answers.

## Long context

A code hidden in German and English filler, asked for with reasoning off, through the API
gateway:

| Prompt tokens | Position | Time to first token | Found |
|---|---|---|---|
| 31,503 | middle | 2.2 s | yes |
| 126,055 | middle | 6.2 s | yes |
| 252,542 | middle | 18.5 s | yes |
| 512,469 | 30 % | 64.3 s | yes |
| 1,012,962 | 70 % | 229.7 s | yes |

## Concurrency probes

Same probe as for Muse Glimmer (about 325-token prompts and 512-token answers, reasoning off;
for prefill 6,670-token prompts and 64-token answers).

| Setting | c=1 decode | c=64 | c=128 | c=256 | Prefill c=1 | Prefill c=16 |
|---|---|---|---|---|---|---|
| 128 seqs, 8,192 batched | 151 (6.2 ms) | 4,331 | 5,482 | - | 10,008 | 41,032 |
| 256 seqs, 16,384 batched | 146 (6.2 ms) | 3,941 / 4,317 | 5,689 / 5,590 | 6,687 (32.8 ms) | 10,228 | 40,705 |

Two runs at 64 and 128 for the second setting show the probe's noise (about 5 percent at
64). The second setting serves 256 concurrent requests (6,687 decode tokens/s, TTFT p50
1.16 s) at the same speed elsewhere; a 252,542-token prompt took 18.2 s to its first token
(18.5 s before). Its larger batch costs KV cache: 43.16 GiB, 4,015,963 tokens (3.83 requests
of 1M), graphs 0.86 GiB, peak activation 2.21 GiB. It is chosen.

The replica held 125.38 GiB after these probes: 5.4 GiB above the 120 GiB engine booking,
and 2.6 GiB below the 128 GiB hardware booking. The hardware booking is raised to 132 GiB
for margin until the full benchmark measures the peak.

## Not pursued now

- **A tuned MoE configuration.** vLLM's own tuner (`benchmarks/kernels/benchmark_moe.py`, with
  Kolibri's expert shape) ran on the node next to the live replica: one batch size took more
  than 35 minutes of 1,920 candidate configurations and slowed down sharply when the replica
  served. A complete tuning (17 batch sizes) needs the GPU alone for hours. Its result would
  ship as a container file with `VLLM_TUNED_CONFIG_FOLDER`. A follow-up.
- **Speculative decoding.** Kolibri ships no MTP head. The only drafter found is a community
  EAGLE-3 head (`josch15366/Kolibri-1-EAGLE3-drafter`, 42 downloads) measured at 1.2x for one
  stream on a DGX Spark and neutral at four streams; not from a trusted publisher, and the
  Transformers modeling backend does not expose the hidden states EAGLE-3 reads. Not used.
