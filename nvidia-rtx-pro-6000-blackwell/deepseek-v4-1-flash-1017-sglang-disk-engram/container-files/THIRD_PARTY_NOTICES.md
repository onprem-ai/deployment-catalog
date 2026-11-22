# Third-party code in container-files/

| File | Origin | License | Changes by onprem.ai |
|---|---|---|---|
| `engram_backend.py`, `sitecustomize.py`, `row_store.cpp` | [sybil-solutions/deepseek-v4.1-flash-4x-rtx-pro-6000](https://github.com/sybil-solutions/deepseek-v4.1-flash-4x-rtx-pro-6000), as in the fork soloptimizer/deepseek-v4.1-flash-4x-rtx-pro-6000 at commit 34047e42 (sybil-solutions PR #3, native AIO) | MIT, `LICENSE-MIT-sybil-solutions.txt` | none |
| `flash_mla_sm120.py` | SGLang's `python/sglang/kernels/ops/attention/flash_mla_sm120.py` ([sgl-project/sglang](https://github.com/sgl-project/sglang), Apache-2.0), as changed in ryder777777/deepseek-v4.1-flash-4x-rtx-pro-6000 at commit 1b1353f9 (sybil-solutions PR #4, MIT) | Apache-2.0, `LICENSE-Apache-2.0-sglang.txt`; the PR #4 changes MIT, `LICENSE-MIT-sybil-solutions.txt` | Marked "onprem.ai profile addition" in the file: `DSV41_SM120_PREFILL_BACKEND` (triton, dec64, flashinfer_sync), the debug switches `DSV41_SM120_PREFILL_CHECK` and `DSV41_SM120_PREFILL_MODE_FILE`, and optional fresh or per-page-size split buffers for `_split_kv_pages_to_64`. See `journal/2026-11-22_decode-speed-analysis.md`. |

`make_view.py`, `selftest.py` and `tcp_forward.py` are written by onprem.ai for this profile.
