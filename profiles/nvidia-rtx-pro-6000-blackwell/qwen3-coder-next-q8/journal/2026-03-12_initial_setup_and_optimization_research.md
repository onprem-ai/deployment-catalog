# Session: Initial Setup and Optimization Research

**Date**: 2026-03-12
**Status**: WORKING
**Catalog ID**: `rtx6000bw/qwen3-coder-next-q8`

## Summary

Qwen3-Coder-Next Q8_0 quantization running on RTX 6000 Blackwell with llama.cpp CUDA backend. Higher quality than Q6_K_XL at the cost of ~14GB more storage.

## Performance

| Metric | Value |
|--------|-------|
| Generation TPS | ~141 tok/s (single user) |
| Prompt Throughput | ~3,025 tok/s |
| TTFT | ~130ms |
| Context Length | 256k tokens (native max) |
| Model Size | ~79 GB (Q8_0) |
| VRAM Usage | ~83 GiB at 256k context |

## Configuration

**Image**: `llama-cpp-blackwell:latest` (custom SM120 build)

**Current Parameters**:
```
-ngl 999                    # All layers on GPU
--flash-attn on             # Flash attention enabled
--no-mmap                   # Load to GPU memory directly
-c 262144                   # 256k context (native max)
--cache-type-k q8_0         # KV cache quantization
--cache-type-v q8_0
-b 4096 -ub 4096           # Batch sizes
--jinja                     # Jinja templating for chat
```

## Qwen's Recommended Sampling Parameters

From Qwen team and community testing, these sampling params are recommended for Qwen3-Coder models:

```
--temp 0.7
--top-k 20
--top-p 0.8
--repeat_penalty 1.05
```

**Consideration**: Adding these params would enforce consistent model behavior regardless of what clients send. Could be combined with request filtering in the AI Gateway to ignore client-provided sampling params.

**Trade-off**: Removes client control over sampling, but ensures "known good" settings for this specific model.

## Chat Template for Tool Calling

**Key finding**: Qwen3-Coder uses a **non-standard XML format** for tool calling instead of JSON. This can cause issues with Claude Code and similar tools.

**Unsloth's fix**: The Unsloth GGUF uploads (which we use: `unsloth/Qwen3-Coder-Next-GGUF`) include a fixed Jinja template that properly handles:
- Tool definitions in XML format
- Tool calls with `<tool_call><function=name><parameter=x>value</parameter></function></tool_call>` structure
- Proper formatting for llama.cpp's `--jinja` flag

**Custom template file**: A community-maintained template exists for Claude Code compatibility:
- Gist: https://gist.github.com/mostlygeek/6fe263bad8026dca73cb6f5470dfdb0d
- Usage: `--chat-template-file /path/to/template.jinja`

**Our situation**: Since we use Unsloth GGUFs, the template fix is already embedded. The `--jinja` flag alone should work correctly. A custom template file is only needed if:
1. The embedded template doesn't work for our use case
2. We need specific modifications for Claude Code compatibility

## Context Size Considerations

Our config uses 256k (native max). Some recommendations suggest 96k for:
- Faster prefill times
- Lower memory pressure
- Better stability

**Recommendation**: Keep 256k unless stability issues arise. The RTX 6000 Blackwell has 96GB VRAM, and Qwen3-Coder-Next has a tiny KV cache (2 heads/layer), so 256k is viable.

## RAM Usage Investigation

**Issue**: llama-server was hitting OOM kills at ~16.5GB RAM despite full GPU offload (`-ngl 999`).

**Root cause identified**: Prompt cache + CUDA pinned memory buffers.

**Memory breakdown from llama-server logs:**

| Allocation | Size | Location | Notes |
|------------|------|----------|-------|
| Model buffer | 80.5 GiB | VRAM | Weights |
| KV buffer | 3.3 GiB | VRAM | Context cache |
| Compute buffer | 6.4 GiB | VRAM | GPU compute |
| RS buffer | 0.3 GiB | VRAM | Recurrent state |
| **CUDA_Host compute buffer** | **4.2 GiB** | **RAM** | Pinned memory for GPU transfers |
| CUDA_Host model buffer | 315 MiB | RAM | Host-side model data |
| CUDA shared memory | ~4.5 GiB | RAM | GPU IPC (shmem) |
| **Prompt cache** | **up to 8 GiB** | **RAM** | Enabled by default, grows during use |

**Expected RAM usage:**
- Baseline (idle): ~9 GiB (4.5 GiB shmem + 4.2 GiB pinned + 0.5 GiB process)
- Peak (with prompt cache): ~17 GiB (baseline + 8 GiB prompt cache)

**This explains the ~16.5 GiB peak** that caused OOM with the 16Gi limit.

**Fix applied**: Increased memory limit from 16Gi to 20Gi.

**Potential optimizations** (if RAM becomes constrained):
- Reduce prompt cache: `--cache-prompt-size 4096` (default 8192 MiB)
- Reduce batch size: `-b 2048 -ub 2048` (smaller compute buffers)
- Reduce context: `-c 131072` (smaller KV cache, less processing)

**References:**
- [llama.cpp RAM usage discussion #9784](https://github.com/ggml-org/llama.cpp/discussions/9784)
- [Batch size memory discussion #10068](https://github.com/ggml-org/llama.cpp/discussions/10068)
- [MoE offload guide](https://huggingface.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide)

## Model Aliases for Claude Code

Some setups add aliases to route Claude model requests to this endpoint:
```yaml
aliases:
  - claude-sonnet-4-5
  - claude-haiku-4-5
  - claude-opus-4-5
```

**Trade-off**: Enables drop-in replacement for Claude requests, but could confuse tooling expecting real Claude behavior.

## Sources

- [Unsloth Qwen3-Coder-30B GGUF - Chat Template Fixes](https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/discussions/10)
- [llama.cpp Tool Call Parser Issue #15012](https://github.com/ggml-org/llama.cpp/issues/15012)
- [Qwen llama.cpp Documentation](https://qwen.readthedocs.io/en/latest/run_locally/llama.cpp.html)
- [Claude Code can now connect directly to llama.cpp (Reddit)](https://www.reddit.com/r/LocalLLaMA/comments/1p9bk2b/claude_code_can_now_connect_directly_to_llamacpp/)
- [Qwen3-Coder-Next: Complete 2026 Guide to Running AI Coding Agents Locally](https://dev.to/sienna/qwen3-coder-next-the-complete-2026-guide-to-running-powerful-ai-coding-agents-locally-1k95)
