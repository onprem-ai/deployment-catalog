"""Two-pass NVFP4+FP8 quantization of swiss-ai/Apertus-v1.5-70B (data-free).

Two-pass model_free_ptq via llm-compressor:
  Pass 1: NVFP4A16 (weight-only) on MLP up_proj/down_proj
  Pass 2: FP8_DYNAMIC on attention q/k/v/o_proj
  Untouched: lm_head, embed_tokens, norms, xIELU params

Compared to the RC-based script (July 2026), this targets the final v1.5
release (Apertus1p5ForConditionalGeneration architecture, official vocab).

Run inside a memory-capped container (never bare on the host):
    systemd-run --scope -p MemoryMax=30G python3 this_script.py
"""

import logging
import os
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("three_tier")

# The official model downloaded by our download manager
MODEL_INPUT_PATH = "/persistent/models/swiss-ai/Apertus-v1.5-70B"
# Output goes next to the input for disk symmetry
OUTPUT_BASE = "/persistent/models/swiss-ai--Apertus-v1.5-70B--model-apertus-model"

PASS1_OUTPUT_PATH = OUTPUT_BASE + "-pass1-mlp-only"
PASS2_OUTPUT_PATH = OUTPUT_BASE

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
ALWAYS_IGNORED = [
    "lm_head",
    # official v1.5 wraps transformer in model.language_model.*
    "model.language_model.embed_tokens",
    # legacy RC path (ApertusForCausalLM without language_model prefix)
    "model.embed_tokens",
    # Vision tokenizer (Emu3.5): conv layers are 4D, not Linear.
    "re:.*vision_tokenizer.*",
    # Audio tokenizer (WavTokenizer): conv, LSTM, codebook -- not 2D Linear.
    "re:.*audio_tokenizer.*",
]


def disable_fused_global_scale_mappings() -> None:
    """Clear llm-compressor's microscale fused-set mappings for Apertus.

    llm-compressor 0.12.0 assumes LLaMA-style fused module sets -- (q,k,v) and
    (gate_proj, up_proj) -- and its eager set matcher raises ValueError on this
    checkpoint twice over: q/k/v tensors are split across shard files, and
    Apertus has NO gate_proj (xIELU MLP), so up_proj opens (gate,up) sets that
    can never complete.

    Clearing the mappings is semantically correct for Apertus: global-scale
    fusion exists so vLLM's merged qkv_proj / gate_up_proj GEMMs share one
    global scale. Here the microscale tier covers only MLP up/down (unfused,
    no gate partner) and attention is FP8 (not microscale), so there is
    nothing to fuse.
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
    total_bytes = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            fp = os.path.join(root, name)
            if os.path.isfile(fp):
                total_bytes += os.path.getsize(fp)
    log.info("Checkpoint size: %.1f GiB at %s", total_bytes / (1024**3), path)


def main() -> None:
    import llmcompressor

    log.info("llm-compressor version: %s", llmcompressor.__version__)
    log.info("Model input: %s", MODEL_INPUT_PATH)
    log.info("Final output: %s", PASS2_OUTPUT_PATH)

    # Verify input exists
    config_path = os.path.join(MODEL_INPUT_PATH, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Model input not found: {MODEL_INPUT_PATH}")
    import json
    with open(config_path) as f:
        cfg = json.load(f)
    text_cfg = cfg.get("text_config", cfg)
    log.info("Architecture: %s", cfg.get("architectures"))
    log.info("Hidden size: %s, layers: %s, vocab: %s",
             text_cfg.get("hidden_size"), text_cfg.get("num_hidden_layers"),
             text_cfg.get("vocab_size"))

    disable_fused_global_scale_mappings()
    total_start = time.monotonic()

    # Pass 1: NVFP4A16 on MLP only
    run_pass(
        "PASS 1 (NVFP4A16 on MLP)",
        MODEL_INPUT_PATH,
        PASS1_OUTPUT_PATH,
        scheme="NVFP4A16",
        ignore=ALWAYS_IGNORED + ATTENTION_MODULE_PATTERNS,
    )
    report_output_size(PASS1_OUTPUT_PATH)

    # Pass 2: FP8_DYNAMIC on attention only
    run_pass(
        "PASS 2 (FP8_DYNAMIC on attention)",
        PASS1_OUTPUT_PATH,
        PASS2_OUTPUT_PATH,
        scheme="FP8_DYNAMIC",
        ignore=ALWAYS_IGNORED + MLP_MODULE_PATTERNS,
    )
    report_output_size(PASS2_OUTPUT_PATH)

    log.info("ALL DONE in %.1f minutes", (time.monotonic() - total_start) / 60)
    log.info("Output: %s", PASS2_OUTPUT_PATH)


if __name__ == "__main__":
    main()
