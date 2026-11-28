# 2026-11-27 OCR benchmark

**Hardware:** 1 × NVIDIA RTX PRO 6000 Blackwell Server Edition (97,887 MiB), nothing else on
the host. The profile as it is in the catalog (compose `e2a904da8887…`), no setting changed.

The platform's OCR benchmark, workload `ocr-pages/1.0.0` (the pages of olmOCR-Bench as page
images, one page per request, at least 100 pages per level), started with
`replicas_benchmark_start`. The result is `benchmarks/2026-11-27-2131Z-ocrperfbench.yaml`, the run's
`benchmark_file` as the control plane answered it.

| Pages at once | Pages per minute | Seconds per page, p50 | p95 | Pages cut at the output limit (of 100) | Failed |
|---|---|---|---|---|---|
| 1 | 114.4 | 0.14 | 2.49 | 2 | 0 |
| 2 | 247.6 | 0.15 | 2.42 | 0 | 0 |
| 4 | 318.6 | 0.19 | 4.60 | 2 | 0 |
| 8 | 478.8 | 0.20 | 4.34 | 0 | 0 |
| 12 | 452.7 | 0.25 | 7.28 | 4 | 0 |
| 16 | 510.6 | 0.30 | 8.22 | 0 | 0 |
| 20 | 522.6 | 0.29 | 13.22 | 3 | 0 |
| 24 | 582.2 | 0.31 | 14.16 | 1 | 0 |
| 32 | 592.5 | 1.09 | 12.07 | 0 | 0 |
| 40 | 588.2 | 2.51 | 15.18 | 1 | 0 |

- Stopped at a plateau; plateau concurrency 24. No failed request. The run took 8.5 minutes.
- Mean answer length 151 to 320 tokens per page, from
  1,722 to 1,775 prompt tokens.
- **GPU memory went above the profile's booking:** peak 9,645 MiB (`nvidia-smi`, every
  30 s) against `hardware.gpu.memory_bytes_per_device` of 8,192 MiB, 1,453 MiB more.
  The booking is raised to 10,240 MiB (10,737,418,240 bytes) with this entry, 595 MiB above
  the peak. The compose file is unchanged, so the engine's own share
  (`--gpu-memory-utilization`) and the benchmark stay as they are.
- Host memory at the end of the run: 4.97 GiB of the 8G limit.
- The run was started with `measure_while_serving_clients`: the service had no other caller,
  but the control plane counts every ready replica as serving clients.
