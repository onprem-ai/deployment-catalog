# Session: MiniMax M2.5 REAP 139B A10B NVFP4 - vLLM Deployment

**Date**: 2026-03-23
**Hardware**: NVIDIA RTX PRO 6000 Blackwell (96GB VRAM)
**Model**: MiniMax-M2.5-REAP-139B-A10B-NVFP4-GB10
**Source**: `saricles/MiniMax-M2.5-REAP-139B-A10B-NVFP4-GB10`
**Architecture**: `MiniMaxM2ForCausalLM` (custom HuggingFace architecture)
**Quantization**: compressed-tensors (nvfp4-pack-quantized format)

---

## Final Working Configuration

```bash
python3 -m vllm.entrypoints.openai.api_server \
  --model /model \
  --served-model-name minimax-m25-reap-139b-nvfp4 \
  --host 0.0.0.0 --port 8080 \
  --trust-remote-code \
  --gpu-memory-utilization 0.95 \
  --max-model-len 8192 \
  --quantization compressed-tensors \
  --enforce-eager \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

**Key settings that made it work**:
- `--gpu-memory-utilization 0.95` - Model needs almost all VRAM
- `--max-model-len 8192` - Limited context due to KV cache memory constraints
- `--enforce-eager` - Disabled CUDA graphs for stability
- `--quantization compressed-tensors` - Correct quant method from model config (NOT modelopt_fp4)

---

## What We Tried (Chronological)

### Attempt 1: Initial deployment with modelopt_fp4
**Status**: FAILED

Based on other NVFP4 models (like Nemotron), used `--quantization modelopt_fp4`.

**Error**:
```
ValidationError: Quantization method specified in the model config (compressed-tensors)
does not match the quantization method specified in the `quantization` argument (modelopt_fp4).
```

**Lesson**: Check model's `config.json` for `quant_method` field before assuming quantization type.

---

### Attempt 2: Missing config files
**Status**: FAILED

Model was downloaded with only safetensors weights.

**Error**:
```
Invalid repository ID or local directory specified: '/model'.
Please verify... ensure the presence of a 'config.json'.
```

**Fix Applied**: Downloaded all config files:
```bash
cd /persistent/models/saricles--MiniMax-M2.5-REAP-139B-A10B-NVFP4-GB10--model
for f in config.json tokenizer.json tokenizer_config.json generation_config.json \
         model.safetensors.index.json configuration_minimax_m2.py modeling_minimax_m2.py \
         chat_template.jinja special_tokens_map.json vocab.json merges.txt added_tokens.json; do
  curl -sLO "https://huggingface.co/saricles/MiniMax-M2.5-REAP-139B-A10B-NVFP4-GB10/resolve/main/$f"
done
```

---

### Attempt 3: 65k context with FP8 KV cache
**Status**: FAILED

Used `--max-model-len 65536 --kv-cache-dtype fp8`.

**Error**: Silent crash with "Engine core initialization failed" - no detailed error in logs.

**Root cause** (found later): Subprocess crashed but didn't log the error to main stdout.

---

### Attempt 4: Eager mode + 32k context
**Status**: FAILED

Added `--enforce-eager --max-model-len 32768`.

**Error**: Still silent crash.

---

### Attempt 5: 16k context + 0.85 utilization
**Status**: FAILED

Used `--max-model-len 16384 --gpu-memory-utilization 0.85`.

**Error** (finally visible in subprocess logs):
```
ValueError: To serve at least one request with the models's max seq len (16384),
3.88 GiB KV cache is needed, which is larger than the available KV cache memory (2.87 GiB).
Based on the available memory, the estimated maximum model length is 12128.
```

**Lesson**: The ~75GB model leaves very little room for KV cache on a 96GB GPU.

---

### Attempt 6: 8k context + 0.95 utilization
**Status**: SUCCESS

Used `--max-model-len 8192 --gpu-memory-utilization 0.95 --enforce-eager`.

Model loaded successfully and is serving requests.

---

## What Worked

1. **Config file download**: Model now has all required files
2. **Architecture detection**: vLLM recognizes `MiniMaxM2ForCausalLM`
3. **NVFP4 backend**: FlashInfer CUTLASS backend works correctly
4. **Eager mode**: Stable loading without CUDA graph compilation
5. **High memory utilization**: 0.95 gives enough headroom for weights + small KV cache

---

## Constraints

- **Max context**: ~8-12k tokens (limited by KV cache memory)
- **GPU utilization**: Must be 0.95+ to fit model
- **CUDA graphs**: Disabled (enforce_eager) for stability
