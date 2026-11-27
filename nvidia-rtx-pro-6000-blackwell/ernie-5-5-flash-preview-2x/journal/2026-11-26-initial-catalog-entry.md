# 2026-11-26 Initial catalog entry (written from sources, nothing run)

**Hardware:** none used. The profile targets two RTX PRO 6000 Blackwell (96 GB, SM120). The
authoring host has two (Server Edition, 97,887 MiB each), but GPU 0 held 68,995 MiB of another
workload, so nothing was deployed.

## Sources read

- `baidu/ERNIE-5.5-Flash-Preview` at `ace1100dcde7bf4eb06add6ae11ce4f43a11ddfe`: model card,
  `config.json`, `tokenizer_config.json`, file sizes from the Hugging Face API. Safetensors
  165.9 GB (154.5 GiB); 289.9B of the 299.6B stored parameters are 8-bit integers (the packed
  MXFP4 experts), 7.5B FP8, 2.1B BF16. The `-FP8` repository is 301.9 GB, `-BF16` 599.6 GB.
- Baidu's deployment section: SGLang and vLLM commands for four GPUs, 131,072-token context,
  cookbook link for B300. No memory requirement is published.
- `baidu/vLLM-ERNIE` at `ba322a5`: Apache-2.0, runs on the unmodified vLLM 0.32.0 wheel, chat
  models need `--trust-remote-code`.
- `hyudryu/ERNIE-5.5-Flash-Preview-2-RTXPro6000` at `8c5b856` (created 2026-11-24, no stars, no
  issues, **no license file**): the README, `recipe/Dockerfile`, `recipe/run.sh`. Source of the
  command and of every performance and memory figure in this profile's README.

## Decisions

- **The default repository.** It is the only one of the three that fits two cards.
- **The port's V2, as reported.** 300,000 tokens per request with a 1.21M-token pool. V1's
  524,288 was shown with one request; no cap above 300,000 was tested on V2's memory budget, so
  this profile does not claim one.
- **KV cache in bytes, no `--gpu-memory-utilization`.** The port reports that a share
  over-allocates the KV cache for this model and faults in a long prefill. No engine booking is
  written, and the checker warns (`engine.kv-cache-size-fixed`, `engine.gpu-memory-limit`).
- **Booking 95 GiB per GPU:** the reported peak is 94.75 GiB.
- **`--enable-prefix-caching` written out;** the port relies on vLLM 0.32's default.
- **The port's files are not copied into the catalog** (no license). `runtime/Dockerfile` fetches
  them at a pinned commit.
- **`tokenization_ernie.py`, `encoding_ernie.py`, `decoding_ernie.py` stay in the weights' files:**
  the tokenizer's `auto_map` loads them. Read at the pinned revision: imports are `json`, `re`,
  `uuid`, `copy`, `itertools`, `dataclasses`, `typing`, `tokenizers`, `transformers`.

## Later the same day: image built, weights downloaded

- `docker build` of `runtime/Dockerfile` on the authoring host: exit 0, image
  `onpremai/vllm-ernie55:0.32.0-ba322a5-sm120v2`, local id `f6977c105730`, 33.1 GB. Not pushed,
  so there is no registry digest to pin yet. Checked in a container without a GPU: `vllm`
  0.32.0 and `vllm_ernie` import, `/opt/vLLM-ERNIE` is at `ba322a5` with `attention.py` and
  `moe.py` modified, and the replaced `vllm.v1.core.kv_cache_utils` and
  `...fused_moe.experts.deep_gemm_moe` import. The Dockerfile's paths are therefore right for
  the pinned base image.
- Weights downloaded to `/home/ops/models/baidu--ERNIE-5.5-Flash-Preview` at the pinned
  revision: 142 files, 165,935,900,630 bytes, every size as the Hugging Face API lists it
  (sizes compared, checksums not).

## Not done

- No start with the model, no request, no benchmark, no measured GPU memory.
- Host memory limits (32G, `shm_size` 16gb) are guesses.

## Checker

Run with the control plane's `check_profile` (platform-alpha `b3643190`): no errors, 24 rules
passed, 4 warnings and 1 hint, all named in the README.

## Next

Build and push the image (after clearing the port's license), pin its digest, deploy on two
empty GPUs, benchmark, measure.
