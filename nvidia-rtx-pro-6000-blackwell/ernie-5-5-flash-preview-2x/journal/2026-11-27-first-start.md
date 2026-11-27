# 2026-11-27 First start on two RTX PRO 6000 Blackwell: it serves

**Hardware:** host `ain-xtg6o`, 2 × NVIDIA RTX PRO 6000 Blackwell Server Edition (97,887 MiB
each), 30 GB of RAM, driver 595.112.07. Deployed through the control plane (development stack)
from the `drafts` catalog at revision `7e2a3aef`, inference service
`24d0763a-0649-418b-9b2a-ff92eecee613`, both GPUs.

## What was changed to get here

- **Image:** pinned to the local build's digest (`sha256:f6977c10…`). The node adopted the image
  from its Docker daemon; nothing was pulled. The image is still in no registry.
- **Weights:** not downloaded by the node. The 138 selected files of an earlier `hf download`
  were placed as reflink copies in
  `cached-file-sets/huggingface/baidu--ERNIE-5.5-Flash-Preview--commit-ace1100--sel-fe27803`;
  the node verified all 165,934,846,423 bytes in about 90 seconds and marked them ready.
- **Booking:** 95 -> 94 GiB per GPU. The deploy's dry run warned "Not enough VRAM" on an empty
  GPU (95.0 GiB free) with 95.
- **Host memory:** container limit 32G -> 20G for the 30 GB host.

## Start, as the engine logged it (UTC 2026-11-26)

| Step | Result |
|---|---|
| Architecture | `Ernie5_5_MoeForCausalLM` resolved; max model len 300,000 |
| All-reduce | `FLASHINFER_PCIE_IPC`, then `PYNCCL` |
| Loading weights | 42.0 s (draft head 1.4 s) |
| Model memory | 79.66 GiB per GPU (the port reports 80.45 GiB) |
| Attention backends | `FLASHINFER_MLA_SPARSE_SM120` among them; KV block size 256 |
| KV cache | 9.02 GiB per GPU, **1,213,175 tokens**, 4.04 requests of 300,000 tokens: the port's figure exactly |
| JIT kernel warm-up | 85.6 s |
| CUDA graph capture | 18 s, 1.55 and 1.46 GiB |
| Engine init in total | 261.5 s |
| Deploy to Ready | about 6.5 minutes |

No error or traceback in the engine log. GPU memory in use when ready: 95,369 and 95,323 MiB
(93.1 GiB), below the 94 GiB booking. Container memory during warm-up: 12.8 of 20 GiB.

## First requests (straight to the replica's port, temperature 0)

| Request | Result |
|---|---|
| `GET /v1/models` | `ernie-5.5-flash-preview`, max model len 300,000 |
| Arithmetic question | correct (126); 102 reasoning tokens in `reasoning`, the answer separate; 1.9 s |
| Python function, code only | correct; 270 completion tokens in 1.4 s (about 190 tokens per second, prefill included) |
| Tool call (`get_weather`) | `finish_reason: tool_calls`, arguments `{"city": "Zurich"}`; 0.3 s |

Three requests, one at a time: a smoke test, not a benchmark.

## Still open

- No replica benchmark, no measured GPU memory block, no long prompt. The port's peak of
  94.75 GiB under a 498K-token prefill is above the 94 GiB booking and untested here.
- Not sent through the API gateway.
- The image must be in a registry before any other node can run this profile.
- Host memory limit 20G held through the start; unmeasured under load.
