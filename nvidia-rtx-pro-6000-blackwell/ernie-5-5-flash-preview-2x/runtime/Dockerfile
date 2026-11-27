# Runtime image of the ERNIE 5.5 Flash Preview profiles on RTX PRO 6000 Blackwell (SM120).
#
#   docker build -t onpremai/vllm-ernie55:0.32.0-ba322a5-sm120v2 runtime/
#
# Built on 2026-11-26 (33.1 GB) and pushed on 2026-11-27 as
# onpremai/vllm-ernie55:0.32.0-ba322a5-sm120v2@sha256:f6977c10573049b28505adbb2542dbcd33c99ad5fb595bbbc1459bcf8f308370,
# the image compose.yaml pins. Three pinned parts:
# 1. vLLM 0.32.0, the unmodified release image (linux/amd64 manifest).
# 2. Baidu's vLLM-ERNIE plugin (Apache-2.0): the model class, the ERNIE chat renderer,
#    the ernie5 reasoning and tool-call parsers, DSpark.
# 3. The SM120 port "V2" of github.com/hyudryu/ERNIE-5.5-Flash-Preview-2-RTXPro6000: four
#    files laid over the plugin and vLLM. The stock plugin runs its attention on
#    FlashMLA, which has SM90/SM100 kernels only, and refuses to start on these GPUs.
#    That repository carries no license file: the files are fetched here at build time
#    and are not part of this catalog. The pushed image contains them.
FROM vllm/vllm-openai:v0.32.0@sha256:6ee133356d49ed24cee03dd0cda3d3980279a11c107ce3e752c89e76540f41c9

ARG VLLM_ERNIE_REF=ba322a5634f91d3e6f96ae711df72ddaf3c38b6e
ARG SM120_PORT_REF=8c5b856b6f43d521845759b9346f3c8575900cf4

RUN apt-get update \
 && apt-get install -y --no-install-recommends git \
 && rm -rf /var/lib/apt/lists/* \
 && git clone https://github.com/baidu/vLLM-ERNIE.git /opt/vLLM-ERNIE \
 && git -C /opt/vLLM-ERNIE checkout --detach "${VLLM_ERNIE_REF}" \
 && cd /opt/vLLM-ERNIE \
 && (uv pip install --system --no-deps -e . || pip install --no-deps -e .)

RUN set -eux; \
    site=/usr/local/lib/python3.12/dist-packages; \
    plugin=/opt/vLLM-ERNIE/src; \
    git clone https://github.com/hyudryu/ERNIE-5.5-Flash-Preview-2-RTXPro6000.git /tmp/sm120; \
    git -C /tmp/sm120 checkout --detach "${SM120_PORT_REF}"; \
    for f in vllm_ernie/models/ernie5_5/nvidia/attention.py vllm_ernie/models/ernie5_5/nvidia/moe.py; do \
      test -f "$plugin/$f"; install -m 0644 "/tmp/sm120/patch/v2/$f" "$plugin/$f"; \
    done; \
    for f in vllm/model_executor/layers/fused_moe/experts/deep_gemm_moe.py vllm/v1/core/kv_cache_utils.py; do \
      test -f "$site/$f"; install -m 0644 "/tmp/sm120/patch/v2/$f" "$site/$f"; \
    done; \
    rm -rf /tmp/sm120; \
    python3 -c "import vllm_ernie"
