# 2026-05-06 — Deployment Testing

**Goal:** Test speculative decoding with Gemma 4 31B IT Turbo NVFP4.

## Draft Model Approach (FAILED)

Attempted to use `google/gemma-4-31B-it-assistant` as draft model:

```bash
--speculative-config '{"method": "draft_model", "model": "google/gemma-4-31B-it-assistant", "num_speculative_tokens": 5}'
```

### Errors Encountered

1. **Transformers version too old** - vLLM image has transformers 5.6.0, but `gemma4_assistant` architecture requires 5.8.0+
   - Fixed by adding `pip install --upgrade transformers` at startup

2. **Engine core initialization failed** - After transformers upgrade, the V1 engine fails to initialize with draft model:
   ```
   RuntimeError: Engine core initialization failed. See root cause above. Failed core proc(s): {}
   ```
   - Root cause: NVFP4 quantized target model is incompatible with BF16 draft model
   - The NVFP4 quantization strips MTP weights during compression

### Research Finding

From vLLM docs: "The NVFP4 quantized checkpoint does not include MTP weights — they are stripped during quantization. To enable MTP speculative decoding, BF16 MTP weights were extracted from the original model..."

**Conclusion:** Draft model speculative decoding requires matching quantization or special weight extraction. Not viable without modifying the checkpoint.

## N-gram Speculative Decoding (SUCCESS)

N-gram spec decode uses prompt patterns to predict tokens - no separate model needed.

```bash
--speculative-config '{"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 4}'
```

### Working Configuration

```yaml
args:
  - |
    export VLLM_WORKER_MULTIPROC_METHOD=spawn
    python3 -m vllm.entrypoints.openai.api_server \
      --model /model \
      --served-model-name gemma-4-31b-it-turbo-spec \
      --gpu-memory-utilization 0.35 \
      --max-model-len 16384 --kv-cache-dtype fp8_e4m3 --max-num-seqs 64 \
      --quantization modelopt \
      --max-num-batched-tokens 4096 \
      --enable-auto-tool-choice \
      --tool-call-parser gemma4 \
      --reasoning-parser gemma4 \
      --enforce-eager \
      --speculative-config '{"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 4}'
```

### Critical Flags

| Flag | Why Required |
|------|--------------|
| `--enforce-eager` | V1 engine with spec decode crashes during CUDA graph compilation |
| `VLLM_WORKER_MULTIPROC_METHOD=spawn` | Fixes multiprocessing issues in container |
| `--max-num-batched-tokens 4096` | vLLM warns about token scheduling with spec decode |

### Test Results

```json
{
  "model": "gemma-4-31b-it-turbo-spec",
  "choices": [{"message": {"content": "Hello! How can I help you today?"}}],
  "usage": {"prompt_tokens": 15, "completion_tokens": 10}
}
```

### VRAM Usage

- Model loading: 18.55 GiB
- GPU utilization setting: 0.35 (33.6 GiB allocated)
- Same as turbo baseline - n-gram has no weight overhead

## Outcome

- Updated catalog entry to use n-gram speculative decoding
- Removed draft model approach (incompatible with NVFP4)
- Model deploys and serves requests successfully

**Next steps:**
1. Benchmark n-gram spec decode vs baseline on structured output
2. Test with longer prompts to maximize n-gram matching
3. Consider suffix-based spec decode as alternative

## Sources

- [vLLM Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode/)
- [NVFP4 + MTP Compatibility](https://github.com/bjk110/SPARK_Qwen3.5-122B-A10B-NVFP4)
