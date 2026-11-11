# 2026-11-10 - Ideas for the Next Iteration

Open points after the first bring-up, benchmark and memory measurement of this card. None is
done yet; each needs the operator's go.

## Benchmark

- **A clean benchmark file.** Both full runs lost one request to `connection reset by peer`
  while the engine kept running (`max-decode` stopped at 96 in the committed file, so the card
  shows no plateau concurrency). The likely cause is the load generator reusing a kept-alive
  connection in the instant vLLM closes it (vLLM's keep-alive is 5 s, Go's idle connections
  live 90 s): issue onprem-ai/platform-alpha#330. Run the benchmark again once it is fixed, and
  replace nothing: the new file is added beside the old one.
- **Above 128.** `max-decode` and `mixed-load` still grew at 128 (the tuning probe measured
  6,687 decode tokens/s at 256), and the load generator stops at 128. The replica's limit is
  not measured.

## Performance

- **A tuned MoE kernel configuration.** vLLM has none for this expert shape on H200
  (`E=384,N=512,device_name=NVIDIA_H200,dtype=fp8_w8a8,block_shape=[128,128].json`) and runs
  the Triton MoE on its default. DeepGEMM does not take it (N <= 512). Tune it with vLLM's
  `benchmarks/kernels/benchmark_moe.py` with the GPU alone (the run next to the live replica
  took more than 35 minutes for one of 17 batch sizes), ship the result as a container file and
  point `VLLM_TUNED_CONFIG_FOLDER` at it. Measure decode at 1, 64 and 128 before and after.
- **Speculative decoding.** Kolibri ships no MTP head. Watch for a drafter from Aleph Alpha or
  another trusted publisher; the community EAGLE-3 head (`josch15366/Kolibri-1-EAGLE3-drafter`)
  measured about 1.2x for one stream on a DGX Spark, and the Transformers modeling backend does
  not expose the hidden states it reads.
- **Start-up time.** About 6 minutes: torch.compile 45 s, then DeepGEMM warm-up and graph
  capture. The caches live in the container and are lost with it (see the Muse Glimmer FP8
  card's journal for the options: an image with the caches, a cache volume the node keeps, or
  fewer warm-up shapes).

## Memory and context

- **Overshoot.** The peak is 5.38 GiB above the 120 GiB engine booking (Muse Glimmer: 2.47
  GiB), with 6.62 GiB left to the 132 GiB hardware booking. Find which part grows (the peak
  was already reached after the tuning probes) before lowering the hardware booking.
- **KV cache scales.** The checkpoint has no FP8 KV scales; vLLM uses 1.0 and warns. Aleph Alpha
  evaluated with an FP8 KV cache, and the long-context checks passed up to 1,012,962 tokens;
  an answer-quality check against a BF16 KV cache would confirm it.
- **A 256K alternative.** Aleph Alpha recommends at most 262,144 tokens for serving. A
  `compose.context-256k.yaml` alternative would give no throughput gain by itself (the KV cache
  holds 4.0M tokens either way) but would refuse requests the model handles slowly; decide
  whether the card should offer it.
