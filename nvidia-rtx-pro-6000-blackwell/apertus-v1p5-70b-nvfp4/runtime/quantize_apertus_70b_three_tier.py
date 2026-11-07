"""Three-tier NVFP4+FP8 quantization of Apertus v1.5 70B (data-free, two-pass).

Tier 1: NVFP4A16 (weight-only) on MLP up_proj/down_proj
Tier 2: FP8_DYNAMIC on attention q/k/v/o_proj
Tier 3: BF16 (untouched) on lm_head, embed_tokens, norms, xIELU params

Why two passes: the 146 GB BF16 model does not fit this node's 64 GB RAM, so the
oneshot recipe pathway (which loads the model as a module) is unusable here --
accelerate disk offload crashes oneshot's from_accelerate converter (see journal
2026-07-07_oneshot_disk_offload_failure.md). model_free_ptq streams safetensors
without loading the model but accepts one scheme per invocation. Its
create_or_update_quant_config merges each pass into the checkpoint's
quantization_config, yielding a compressed-tensors mixed checkpoint.

vLLM-only at serving time: SGLang does not support compressed-tensors
mixed-precision checkpoints (sgl-project/sglang#16276).

GPU note: shard validation loads a full ~50 GB shard per worker onto the GPU,
so max_workers=1 and the GPU must be mostly free (undeploy inference services).

Run inside a memory-capped container (never bare on the host):
    docker run --rm --gpus all --memory=30g ...
"""

import logging
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("three_tier")

MODEL_INPUT_PATH = "/work/bf16-assembled"
PASS1_OUTPUT_PATH = "/work/output/apertus-v15-70b-nvfp4-mlp-only"
PASS2_OUTPUT_PATH = "/work/output/apertus-v15-70b-three-tier"
MAX_WORKERS = 1

ATTENTION_MODULE_PATTERNS = [
    "re:.*self_attn\\.q_proj$",
    "re:.*self_attn\\.k_proj$",
    "re:.*self_attn\\.v_proj$",
    "re:.*self_attn\\.o_proj$",
]
MLP_MODULE_PATTERNS = [
    "re:.*mlp\\.up_proj$",
    "re:.*mlp\\.down_proj$",
]
ALWAYS_IGNORED = ["lm_head", "model.embed_tokens"]


def disable_fused_global_scale_mappings() -> None:
    """Clear llm-compressor's microscale fused-set mappings for Apertus.

    llm-compressor 0.12.0 assumes LLaMA-style fused module sets -- (q,k,v) and
    (gate_proj, up_proj) -- and its eager set matcher raises ValueError on this
    checkpoint twice over: q/k/v tensors are split across shard files, and
    Apertus has NO gate_proj (xIELU MLP), so up_proj opens (gate,up) sets that
    can never complete. Same root cause as the 8B-era gate_proj crash, moved
    into the model_free pathway.

    Clearing the mappings is semantically correct for Apertus: global-scale
    fusion exists so vLLM's merged qkv_proj / gate_up_proj GEMMs share one
    global scale. Here the microscale tier covers only MLP up/down (unfused,
    no gate partner) and attention is FP8 (not microscale), so there is
    nothing to fuse. Worth an upstream issue.
    """
    import llmcompressor.entrypoints.model_free.microscale as microscale

    microscale.DEFAULT_FUSED_MAPPINGS.clear()
    microscale._DEFAULT_FUSED_MAPPINGS_LIST.clear()
    log.info("Cleared microscale fused mappings (Apertus has no fused module pairs)")


def run_pass(pass_name: str, input_path: str, output_path: str,
             scheme: str, ignore: list[str]) -> None:
    from llmcompressor import model_free_ptq

    log.info("=" * 60)
    log.info("%s: scheme=%s", pass_name, scheme)
    log.info("  input : %s", input_path)
    log.info("  output: %s", output_path)
    log.info("  ignore: %s", ignore)
    start = time.monotonic()
    model_free_ptq(
        model_stub=input_path,
        save_directory=output_path,
        scheme=scheme,
        ignore=ignore,
        max_workers=MAX_WORKERS,
        device="cuda:0",
    )
    log.info("%s finished in %.1f minutes", pass_name, (time.monotonic() - start) / 60)


def report_output_size(path: str) -> None:
    import os

    total_bytes = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            total_bytes += os.path.getsize(os.path.join(root, name))
    log.info("Checkpoint size: %.1f GiB at %s", total_bytes / (1024**3), path)


# The v1.5 prerelease upstream generation_config.json declares eos_token_id 68
# only. The 2509 Instruct release ships [2, 68, 72], where 72 is
# <|tools_suffix|> -- without it vLLM never registers the tool-call suffix as
# a stop token and generation runs past tool calls (journal
# 2026-07-08_agentic_capability_finding.md). vLLM reads eos_token_id from the
# model dir's generation_config.json only; --override-generation-config
# silently drops the key (vllm-project/vllm#45591).
TOOL_CALL_AWARE_EOS_TOKEN_IDS = [2, 68, 72]


def write_tool_call_eos_token_ids(output_path: str) -> None:
    """Set eos_token_id [2, 68, 72] in the checkpoint's generation_config.json.

    model_free_ptq copies the source generation_config.json verbatim, so the
    prerelease's missing <|tools_suffix|> (72) stop token must be added after
    each pass.
    """
    import json
    import os
    import tempfile
    from pathlib import Path

    config_path = Path(output_path) / "generation_config.json"
    generation_config = json.loads(config_path.read_text())
    generation_config["eos_token_id"] = TOOL_CALL_AWARE_EOS_TOKEN_IDS
    fd, tmp_name = tempfile.mkstemp(dir=output_path)
    with os.fdopen(fd, "w") as tmp_file:
        json.dump(generation_config, tmp_file, indent=2)
        tmp_file.write("\n")
        tmp_file.flush()
        os.fsync(tmp_file.fileno())
    Path(tmp_name).rename(config_path)
    log.info("Set eos_token_id=%s in %s", TOOL_CALL_AWARE_EOS_TOKEN_IDS, config_path)


def main() -> None:
    import llmcompressor

    log.info("llm-compressor version: %s", llmcompressor.__version__)
    disable_fused_global_scale_mappings()
    total_start = time.monotonic()

    run_pass(
        "PASS 1 (NVFP4A16 on MLP)",
        MODEL_INPUT_PATH,
        PASS1_OUTPUT_PATH,
        scheme="NVFP4A16",
        ignore=ALWAYS_IGNORED + ATTENTION_MODULE_PATTERNS,
    )
    write_tool_call_eos_token_ids(PASS1_OUTPUT_PATH)
    report_output_size(PASS1_OUTPUT_PATH)

    run_pass(
        "PASS 2 (FP8_DYNAMIC on attention)",
        PASS1_OUTPUT_PATH,
        PASS2_OUTPUT_PATH,
        scheme="FP8_DYNAMIC",
        ignore=ALWAYS_IGNORED + MLP_MODULE_PATTERNS,
    )
    write_tool_call_eos_token_ids(PASS2_OUTPUT_PATH)
    report_output_size(PASS2_OUTPUT_PATH)

    log.info("ALL DONE in %.1f minutes", (time.monotonic() - total_start) / 60)


if __name__ == "__main__":
    main()
