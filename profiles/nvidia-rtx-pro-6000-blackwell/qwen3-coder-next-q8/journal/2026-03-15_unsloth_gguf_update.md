# Session: Unsloth GGUF Update

**Date**: 2026-03-15

## Summary

On February 19, 2026, Unsloth released an updated GGUF of Qwen3-Coder-Next with significant improvements. This session documents the update and its impact on our deployment.

## Key Changes

- **Much better outputs** due to llama.cpp fixing a bug in vectorized `key_gdiff` calculation
- **Fixed tool-calling** for Qwen3-Coder models
- **Dynamic 2.0 quantization** with better accuracy

## Details

This update coincided with llama.cpp merging the vectorized key_gdiff fix, which improved output quality for models using this architecture. The old GGUFs (prefixed with `__` in our models directory) have been superseded by this new version.

## Reference

- [Unsloth Qwen3-Coder-Next-GGUF Releases](https://huggingface.co/unsloth/Qwen3-Coder-Next-GGUF)
