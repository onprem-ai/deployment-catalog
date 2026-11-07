# Session: Speculative Decoding Investigation - UNSUPPORTED

**Date**: 2026-03-02
**Status**: UNSUPPORTED - Do not retry

## DO NOT RETRY THESE APPROACHES

The following approaches have been exhaustively tested and **do not work**:

| Approach | Why It Fails |
|----------|--------------|
| Draft model + flash-attn ON | "partial sequence removal not supported" - spec decoding silently disabled |
| Draft model + flash-attn OFF | Model fails to create context - qwen3next requires flash-attn |
| Draft model + flash-attn OFF + reduced context | Same context creation failure |
| Draft model + flash-attn OFF + `-fit off` | Same context creation failure |
| N-gram (`--spec-type ngram-simple`) + flash-attn ON | Same "partial sequence removal" error |
| N-gram (`--spec-type ngram-mod`) + flash-attn ON | Same limitation |
| Any `--spec-type` variant | ALL require partial sequence removal |

## Root Cause

**Fundamental incompatibility in llama.cpp:**

1. **ALL speculative decoding** (draft model AND ngram-based) requires "partial sequence removal" from the KV cache
2. **Flash attention does NOT support** partial sequence removal
3. **Qwen3-Coder-Next (qwen3next architecture) REQUIRES flash attention** to create context

This creates an impossible situation:
- Enable flash-attn → speculative decoding disabled
- Disable flash-attn → model crashes on context creation

## Error Messages

With flash attention ON (speculative decoding disabled):
```
common_speculative_is_compat: the target context does not support partial sequence removal
srv    load_model: speculative decoding not supported by this context
main: server is listening on http://0.0.0.0:8080  # Server works, but no spec decoding
```

With flash attention OFF (model crashes):
```
common_init_result: failed to create context with model '/model/Qwen3-Coder-Next-...'
```

## Draft Model Tested

The draft model was correctly configured and tokenizer-compatible:

- **Model**: `jukofyork/Qwen3-Coder-Instruct-DRAFT-0.75B-GGUF`
- **File**: `Qwen3-Coder-Instruct-DRAFT-0.75B-256k-Q4_0.gguf`
- **Size**: 448 MB (tiny overhead)
- **Tokenizer**: Verified identical to main model (same vocab.json MD5 hash: `613b8e4a622c4a2c90e9e1245fc540d6`)

## References

- [llama.cpp speculative.md](https://github.com/ggml-org/llama.cpp/blob/master/docs/speculative.md)
- [Flash attention compatibility - Discussion #15650](https://github.com/ggml-org/llama.cpp/discussions/15650)
- [Speculative decoding discussion #10466](https://github.com/ggml-org/llama.cpp/discussions/10466)
