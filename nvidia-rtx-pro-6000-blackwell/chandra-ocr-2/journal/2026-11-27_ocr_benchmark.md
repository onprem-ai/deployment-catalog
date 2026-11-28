# 2026-11-27 OCR benchmark

**Hardware:** 1 × NVIDIA RTX PRO 6000 Blackwell Server Edition (97,887 MiB), nothing else on
the host. The profile as it is in the catalog (compose `8bf3e2199d78…`), no setting changed.

The platform's OCR benchmark, workload `ocr-pages/1.0.0` (the pages of olmOCR-Bench as page
images, one page per request, at least 100 pages per level), started with
`replicas_benchmark_start`. The result is `benchmarks/2026-11-27-2142Z-ocrperfbench.yaml`, the run's
`benchmark_file` as the control plane answered it.

| Pages at once | Pages per minute | Seconds per page, p50 | p95 | Pages cut at the output limit (of 100) | Failed |
|---|---|---|---|---|---|
| 1 | 5.2 | 11.28 | 21.71 | 6 | 0 |
| 2 | 8.6 | 14.23 | 23.76 | 11 | 0 |
| 4 | 17.1 | 14.24 | 24.41 | 5 | 0 |
| 8 | 32.0 | 14.75 | 26.69 | 7 | 0 |
| 12 | 41.5 | 16.54 | 27.98 | 9 | 0 |
| 16 | 56.5 | 16.70 | 29.81 | 9 | 0 |
| 20 | 63.1 | 18.59 | 32.06 | 9 | 0 |
| 24 | 75.5 | 18.17 | 33.75 | 9 | 0 |
| 32 | 90.1 | 20.75 | 37.19 | 7 | 0 |
| 40 | 93.6 | 24.15 | 42.10 | 7 | 0 |
| 48 | 92.6 | 32.77 | 47.78 | 12 | 0 |

- Stopped at a plateau; plateau concurrency 32. No failed request. The run took 63.8 minutes.
- Mean answer length 1,624 to 1,878 tokens per page, from
  1,682 to 1,723 prompt tokens.
- **GPU memory went above the profile's booking:** peak 16,351 MiB (`nvidia-smi`, every
  30 s) against `hardware.gpu.memory_bytes_per_device` of 15,360 MiB, 991 MiB more.
  The booking is raised to 16,896 MiB (17,716,740,096 bytes) with this entry, 545 MiB above
  the peak. The compose file is unchanged, so the engine's own share
  (`--gpu-memory-utilization`) and the benchmark stay as they are.
- Host memory at the end of the run: 7.45 GiB of the 16G limit.
- The run was started with `measure_while_serving_clients`: the service had no other caller,
  but the control plane counts every ready replica as serving clients.
