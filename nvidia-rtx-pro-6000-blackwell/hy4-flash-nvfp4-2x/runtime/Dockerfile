# SPDX-License-Identifier: Apache-2.0
#
# vLLM for Tencent Hy4 flash (NVFP4) on RTX PRO 6000 Blackwell (SM120), with the Engram
# tables read from the checkpoint files instead of held in GPU or host memory.
#
# Three layers of source on top of a pinned vLLM image, and nothing compiled:
# 1. vLLM main at 77766db (2026-11-17), the image the SM120 work below was reviewed against.
# 2. The seven overlay files of high-tech-system/Hy4-flash-NVFP4-vLLM-SM120 at b0067d3f
#    (Apache-2.0 and MIT, derived from vLLM and atakann/dgx-spark-recipes): ModelOpt NVFP4
#    expert weights, FlashInfer's SM120 sparse MLA backend for this model's attention, the
#    sliding-window index trim. Each file is fetched by its pinned revision and checked
#    against its SHA-256.
# 3. engram.py: this image's own hy_v4_flash/nvidia/engram.py with the disk patch of
#    ZackSample/hy4-flash-dgx-sparks (Apache-2.0) applied and extended from BF16 tables to
#    FP8 tables with a scale per row. Off unless HY4_ENGRAM_DISK=1.
#
# No kernel cache is baked in. Triton, FlashInfer and the CUDA compute cache compile
# host-side code for the CPU they run on; a cache warmed on one CPU and shipped in the image
# stops with SIGILL on a CPU without those instructions. Every node compiles its own at the
# first start. (The upstream image's own 20 MB /root/.cache/flashinfer is left as shipped.)
#
# Build: docker build -t onpremai/vllm-hy4-flash:77766db-sm120-diskengram1 runtime/
FROM vllm/vllm-openai@sha256:f4c3ca45840d5612d673b9723f8a82740707a2e5ff06c02df03246f4276a7c69

ARG VLLM_ROOT=/usr/local/lib/python3.12/dist-packages/vllm
ARG HTS=https://huggingface.co/high-tech-system/Hy4-flash-NVFP4-vLLM-SM120/resolve/b0067d3fae55d6a858843abfa64d460fa69c1bf2/patches

ADD --chmod=644 --checksum=sha256:b67afe0ff16b384261dec87338c559ae089843ad8bcc094f26671661c119f8b2 \
    ${HTS}/attention.py ${VLLM_ROOT}/models/hy_v4_flash/nvidia/attention.py
ADD --chmod=644 --checksum=sha256:69865a097d847d446fdcdfb81672249a426a8abfea4bc8da278efb2e79fa2716 \
    ${HTS}/model.py ${VLLM_ROOT}/models/hy_v4_flash/nvidia/model.py
ADD --chmod=644 --checksum=sha256:d666af60212ccf6ae2a20b60c766588a947b725fa168050cb1c19664ea65a221 \
    ${HTS}/sm120.py ${VLLM_ROOT}/models/hy_v4_flash/nvidia/sm120.py
ADD --chmod=644 --checksum=sha256:ad2adb4680eb722af574e1c73e629cc8e8df2bd39601d1639f6892853f7849f3 \
    ${HTS}/swa_index_trim.py ${VLLM_ROOT}/models/hy_v4_flash/nvidia/swa_index_trim.py
ADD --chmod=644 --checksum=sha256:269d390eeaec8a6cf5f85c014542ce4426f4d62e904c82499d6d34633c1938dd \
    ${HTS}/hy_v4_attention.py ${VLLM_ROOT}/models/hy_v4/nvidia/attention.py
ADD --chmod=644 --checksum=sha256:9513efde31a50279fe892ba01d27a546dc15b494fcd7b628639e8f82b26dafe8 \
    ${HTS}/dfly_config.py ${VLLM_ROOT}/models/hy_v4/dfly_config.py
ADD --chmod=644 --checksum=sha256:9540ec2427fe3d747944165a2d416812f2a8e64c74c062e24c9e1b44e96aec67 \
    ${HTS}/mla_attention.py ${VLLM_ROOT}/model_executor/layers/attention/mla_attention.py

# The patched Engram module replaces exactly the file it was written against.
COPY --chmod=644 engram.py /tmp/engram.py
RUN set -eu; \
    echo "c3a6cd3d8810b4a31b6c075a7438e084424a421f5746d443765f4ba73f2b26b3  ${VLLM_ROOT}/models/hy_v4_flash/nvidia/engram.py" | sha256sum -c; \
    install -m 0644 /tmp/engram.py "${VLLM_ROOT}/models/hy_v4_flash/nvidia/engram.py"; \
    rm /tmp/engram.py; \
    python3 -m py_compile \
      "${VLLM_ROOT}/models/hy_v4_flash/nvidia/attention.py" \
      "${VLLM_ROOT}/models/hy_v4_flash/nvidia/model.py" \
      "${VLLM_ROOT}/models/hy_v4_flash/nvidia/sm120.py" \
      "${VLLM_ROOT}/models/hy_v4_flash/nvidia/swa_index_trim.py" \
      "${VLLM_ROOT}/models/hy_v4_flash/nvidia/engram.py" \
      "${VLLM_ROOT}/models/hy_v4/nvidia/attention.py" \
      "${VLLM_ROOT}/models/hy_v4/dfly_config.py" \
      "${VLLM_ROOT}/model_executor/layers/attention/mla_attention.py"; \
    find "${VLLM_ROOT}/models/hy_v4_flash" "${VLLM_ROOT}/models/hy_v4" -name '__pycache__' -prune -exec rm -rf {} +; \
    test ! -e /root/.triton && test ! -e /root/.nv

ENV CUTE_DSL_ARCH=sm_120a \
    VLLM_WORKER_MULTIPROC_METHOD=spawn

LABEL org.opencontainers.image.title="vLLM for Hy4 flash NVFP4 on SM120, Engram tables on disk" \
      org.opencontainers.image.licenses="Apache-2.0 AND MIT"
