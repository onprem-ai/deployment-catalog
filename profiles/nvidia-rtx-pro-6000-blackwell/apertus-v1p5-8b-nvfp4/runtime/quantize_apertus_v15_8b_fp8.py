"""Single-pass FP8_DYNAMIC quantization of swiss-ai/Apertus-v1.5-8B.

Produces an FP8_DYNAMIC checkpoint on all Linear layers as a baseline
for quality comparison against the two-pass NVFP4+FP8 scheme.

Run inside a memory-capped container (never bare on the host):
    systemd-run --scope -p MemoryMax=16G python3 this_script.py
"""

import logging
import os
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("fp8_baseline")

MODEL_INPUT_PATH = "/persistent/models/swiss-ai/Apertus-v1.5-8B"
OUTPUT_PATH = "/persistent/models/onprem-ai--Apertus-v1.5-8B-FP8--model-apertus-model"

MAX_WORKERS = 1

ALWAYS_IGNORED = [
    "lm_head",
    "model.language_model.embed_tokens",
    "model.embed_tokens",
    "re:.*vision_tokenizer.*",
    "re:.*audio_tokenizer.*",
]


def disable_fused_global_scale_mappings() -> None:
    """Clear llm-compressor's microscale fused-set mappings for Apertus."""
    import llmcompressor.entrypoints.model_free.microscale as microscale
    microscale.DEFAULT_FUSED_MAPPINGS.clear()
    microscale._DEFAULT_FUSED_MAPPINGS_LIST.clear()
    log.info("Cleared microscale fused mappings (Apertus has no fused module pairs)")


def main() -> None:
    from llmcompressor import model_free_ptq
    import llmcompressor

    log.info("llm-compressor version: %s", llmcompressor.__version__)
    log.info("Model input: %s", MODEL_INPUT_PATH)
    log.info("Output: %s", OUTPUT_PATH)

    # Verify input
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
    log.info("=" * 60)
    log.info("Single-pass FP8_DYNAMIC on all Linear")

    model_free_ptq(
        model_stub=MODEL_INPUT_PATH,
        save_directory=OUTPUT_PATH,
        scheme="FP8_DYNAMIC",
        ignore=ALWAYS_IGNORED,
        max_workers=MAX_WORKERS,
        device="cuda:0",
    )

    elapsed = (time.monotonic() - total_start) / 60
    log.info("=" * 60)
    log.info("DONE in %.1f minutes", elapsed)

    # Report size
    total_bytes = 0
    for root, _dirs, files in os.walk(OUTPUT_PATH):
        for name in files:
            fp = os.path.join(root, name)
            if os.path.isfile(fp):
                total_bytes += os.path.getsize(fp)
    log.info("Checkpoint size: %.1f GiB at %s", total_bytes / (1024**3), OUTPUT_PATH)


if __name__ == "__main__":
    main()