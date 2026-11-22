# 2026-11-22 The PLE table in shared host RAM on two GPUs: 40% more KV cache for about 42 GiB of host RAM

This profile is `qwen38-flash-next-nvidia-fp4-1m-tp2` with the PLE n-gram table in shared host memory
(`VLLM_PLE_CPU_OFFLOAD: "1"`, `VLLM_PLE_TABLE_MEMORY: "shared"`) instead of in GPU memory. Everything else is the same: image
(`20261121-2gpu-v2`), command, weights, `--max-num-seqs 64`, `--max-model-len 1048576`, the engine's GPU memory booking.

## What the table is, and where it goes

The table is 26.8 GiB. It is split across the two tensor-parallel workers: each logs its own 13.41 GiB table (`Populating shared
PLE table ... (13.41 GiB) in /dev/shm/vllm-ple`, two different table ids), so the two workers share nothing with each other. In
shared mode the table lives in `/dev/shm`; with the table in GPU memory the weights are 51.16 GiB per GPU, with it in host memory
37.74 GiB (`Model loading took 37.74 GiB memory`).

## Measured

Two RTX PRO 6000 Blackwell Server Edition (GPUs 0 and 1 of a four-GPU machine with 61.6 GiB of RAM, 10 GiB of swap and the control
plane on the same machine; driver 595.71.05, `iommu=pt`). The container was started by hand with the platform's exact spec
(image, command, environment, weights mount, GPUs, memory limit) and `--shm-size 32g`, because the node does not apply the plan's
`shm_size` (platform-alpha issue 277): through the platform the container gets a 15.4 GiB `/dev/shm` and both workers fail with
`No space left on device`.

| | table in GPU memory (`...-tp2`) | table in shared host RAM (this profile) |
|---|---|---|
| Weights per GPU | 51.16 GiB | 37.74 GiB |
| KV cache per GPU | 33.54 GiB | 46.94 GiB |
| KV cache tokens | 4,390,469 (4.19x one 1M-token request) | 6,144,230 (5.86x) |
| Container start to ready | 7 min 9 s | 7 min 40 s |
| Output tokens/s, 3 rounds of 16 concurrent requests (15 prompts of about 6k tokens, one of about 100k) | 707 to 725 | 685 to 711 |
| GPU memory after the load | 95,335 MiB per GPU | 95,301 MiB per GPU |
| Host RAM | the whole machine used 22 GiB, with the control plane (the replica alone was not measured) | the replica holds about 42 GiB that cannot be reclaimed (anonymous 12.3 + `/dev/shm` 29.6); the machine used 47 to 49 GiB |

Text, image and tool-call requests answer. The speed is the same within the noise of one run each: the earlier benchmark files of
this profile, with the table in host RAM (2026-11-15) and in GPU memory (2026-11-20), agree at every concurrency level within
about 3 % (decode tokens per second, time to first token, latency), measured on different machines.

Host memory over the whole run, sampled every 5 s: the container reached its 48G limit (page cache, reclaimable), the host's lowest
available memory was 15.1 GiB and the highest swap use 6.5 GiB; nothing was killed. This leaves little room on a 61 GiB machine that
also runs the control plane, and a second replica does not fit. The booking is the not-reclaimable peak, 42.4 GiB, plus about
3.6 GiB: 46 GiB. The container limit is 52G, above the booking, for the page cache.

The GPU memory card is one reading: the peak is `nvidia-smi` after the heavy load (95,301 MiB, the higher GPU), the parts are the
engine's log lines (`Model loading took 37.74 GiB`, `CUDA graph pool memory: 0.74 GiB`, `Available KV cache memory: 46.94 GiB`)
and the overhead is the remainder; it was not taken by the platform's benchmark.

## Status

Experimental until the node applies `shm_size` (issue 277) and the profile has been run through the platform, with a benchmark of
its own. The profile does not block anything: the deploy goes ahead, and until the node is fixed it fails at start with the error above.
