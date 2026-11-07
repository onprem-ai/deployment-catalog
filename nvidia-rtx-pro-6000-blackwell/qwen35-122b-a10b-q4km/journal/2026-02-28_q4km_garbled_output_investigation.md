# Session: Q4_K_M Garbled Output Investigation on Blackwell SM120

**Date**: 2026-02-28
**Goal**: Test Q4_K_M quantization with custom llama-cpp-blackwell image and debug garbled output

## Background

Previous attempts with Q4_K_M using stock llama.cpp produced garbled output due to cuBLAS FP16 overflow on Blackwell (SM120). The `llama-cpp-blackwell:latest` image should be built without `-DGGML_CUDA_FORCE_CUBLAS=ON`.

## Progress

### Step 1: Verify llama-cpp-blackwell image build flags
- [x] Check if image was built without FORCE_CUBLAS
- Status: COMPLETE

**Verified in the image's Dockerfile:**
```dockerfile
# Key: NOT using GGML_CUDA_FORCE_CUBLAS (causes FP16 overflow on Blackwell)
RUN cmake -B build \
    -DGGML_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="120" \
    -DGGML_CUDA_FA_ALL_QUANTS=ON \
```

### Step 2: Deploy Q4_K_M inference service
- [x] Apply InferenceService manifest
- [x] Verify pod starts
- [x] Check model loading logs
- Status: COMPLETE

### Step 3: Verify GPU offloading
- [x] Check GPU memory usage (should be ~70GB for full model)
- [x] Verify minimal CPU_Mapped buffers
- Status: COMPLETE

**Buffer allocation:**
```
CPU_Mapped model buffer size =   409.22 MiB   # Minimal!
CUDA0 model buffer size      = 70475.10 MiB   # 70GB on GPU!
```

This is correct - Q4_K_M fully offloads to GPU unlike MXFP4.

### Step 4: Test inference quality
- [x] Send test prompt
- [ ] Verify output is coherent (not garbled) - **FAILED**
- [x] Measure tokens/second: ~100 tok/s
- Status: **FAILED - GARBLED OUTPUT**

**Test prompt:** "What is 2+2? Answer in one word."

**Output:**
```
"reasoning_content": " nutriceryttytskyylistielt几只 preference preference preferred Preferred preference preference..."
```

**Conclusion:** Custom build without FORCE_CUBLAS still produces garbled output. The issue is NOT just cuBLAS - something else is broken with Qwen3.5 + Q4_K_M on SM120.

## Recovery Instructions

If node freezes or SSH locks out:
1. Hard reboot the node
2. k3s will restart with memory protection enabled
3. Resume from last completed step above
4. Check `/var/log/syslog` or `journalctl -k` for OOM events

## Current State

- MXFP4 inference service: DELETED
- Q4_K_M inference service: DELETED (garbled output)
- Node memory: Available
- GPU memory: Free

---

## Review Feedback (2026-02-28)

### Debugging Plan

#### 1. Rebuild llama.cpp fresh
- [ ] Update to latest master
- [ ] Clean build with CUDA 12.8+ and `CMAKE_CUDA_ARCHITECTURES=120`

#### 2. Sanity check Q4_K_M on CPU
- [ ] Run one short prompt CPU-only (`-ngl 0`)
- [ ] Verify output quality is correct
- [ ] This confirms the GGUF file itself is valid

#### 3. Try env var combinations to find working kernel path
```bash
# Try each of these:
GGML_CUDA_FORCE_CUBLAS=0
GGML_CUDA_DISABLE_CUBLAS=1
GGML_CUDA_FORCE_MMQ=1
GGML_CUDA_FORCE_CUTLASS=1
```
Also try flags like `--no-kq-attention` or `GGML_CUDA_NO_*FUSED*` if available.

#### 4. Bisect GPU layer count
- [x] Start with low `-ngl 8`
- [x] Output still garbled with only 8 GPU layers
- Status: **FOUND ISSUE**

**Key finding:** Output layer is ALWAYS offloaded to GPU regardless of `-ngl`:
```
load_tensors: offloading output layer to GPU
load_tensors: offloading 7 repeating layers to GPU
```

The output layer computes logits. If SM120 has numerical issues in the output projection, it corrupts ALL generations regardless of where other layers run.

**Tested env vars (all still garbled):**
- `GGML_CUDA_FORCE_MMQ=1` - garbled
- `GGML_CUDA_DISABLE_CUBLAS=1` - garbled
- `-ot output.*=CPU` - didn't move output layer, still on GPU, still garbled

**Conclusion:** Current llama.cpp build has SM120 kernel bugs that affect ALL GPU compute paths. No env var workaround found.

### Step 5: Rebuild from latest llama.cpp master
- [x] Rebuilt with --no-cache
- [x] Confirmed latest commit: `ecbcb7e` (2026-02-28)
- [x] Tested Q4_K_M with fresh build
- Status: **STILL GARBLED**

The latest llama.cpp master does NOT fix SM120 garbled output. This is an unresolved upstream issue.

#### 5. Try different quantizations
- [ ] NVFP4/NVFP8 if available
- [ ] IQ-style quants
- [ ] See if they produce correct output at full GPU offload

#### 6. MXFP4_MOE fresh test
- [ ] After fresh llama.cpp build
- [ ] Run with `-ngl 999` and check logs
- [ ] Verify VRAM shows ~60-70GB allocated (not 6GB)
- [ ] If still CPU-mapped, MXFP4 offload on Blackwell not ready yet

### Short-term Workarounds
1. Node with ≥64GB RAM + MXFP4 `--no-mmap` (CPU-bound but correct)
2. Switch to vLLM backend with FP4/FP8 support
3. Pin to CPU-heavy MXFP4 until upstream fixes land

## Commands Reference

```bash
# Deploy Q4_K_M
kubectl apply -f <manifest>

# Check pod status
kubectl get pods -n <namespace>

# Check logs
kubectl logs -n <namespace> <pod-name>

# Check GPU memory
nvidia-smi

# Check system memory
free -h

# Delete if needed
kubectl delete inferenceservice <inference-service> -n <namespace>
```
