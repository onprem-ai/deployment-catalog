#!/usr/bin/env python3
"""Post-surgery for the two-pass three-tier checkpoint's config.json.

llm-compressor 0.12.0's create_or_update_quant_config merges the two
model_free_ptq passes into format=mixed-precision but writes BOTH
config_groups with targets=["Linear"] and leaves pass-1's attention
patterns in the global ignore list. vLLM would then resolve the FP8
attention tensors as unquantized and fail to load (or corrupt).
Reference: catalog journal 2026-07-07_two_pass_success_and_benchmarks.md.

Rewrites (matching the promoted Prerelease ThreeTier reference config):
  - nvfp4-pack-quantized group -> targets = MLP up/down regex
  - float-quantized group      -> targets = attention q/k/v/o regex
  - global ignore              -> ["lm_head", "model.embed_tokens"]
Backs up the original as config.json.as-generated. Fails loudly on any
unexpected structure -- no silent defaults.
"""

import json
import shutil
import sys
from pathlib import Path

MLP_TARGETS = ["re:.*mlp\\.up_proj$", "re:.*mlp\\.down_proj$"]
ATTENTION_TARGETS = [
    "re:.*self_attn\\.q_proj$",
    "re:.*self_attn\\.k_proj$",
    "re:.*self_attn\\.v_proj$",
    "re:.*self_attn\\.o_proj$",
]
EXPECTED_IGNORE = ["lm_head", "model.embed_tokens"]

FORMAT_TO_TARGETS = {
    "nvfp4-pack-quantized": MLP_TARGETS,
    "float-quantized": ATTENTION_TARGETS,
}


def rewrite_mixed_precision_quant_config(config_path: str) -> None:
    with Path(config_path).open() as fh:
        config = json.load(fh)

    quant_config = config.get("quantization_config")
    if quant_config is None:
        raise ValueError(f"{config_path} has no quantization_config")

    groups = quant_config.get("config_groups")
    if not groups:
        raise ValueError("quantization_config has no config_groups")

    seen_formats = []
    for group_name, group in groups.items():
        group_format = group.get("format")
        if group_format not in FORMAT_TO_TARGETS:
            raise ValueError(
                f"group {group_name} has unexpected format {group_format!r}; "
                "expected nvfp4-pack-quantized or float-quantized"
            )
        seen_formats.append(group_format)
        group["targets"] = FORMAT_TO_TARGETS[group_format]
        print(f"  {group_name}: format={group_format} -> targets={group['targets']}")

    if sorted(seen_formats) != sorted(FORMAT_TO_TARGETS):
        raise ValueError(f"expected exactly one group per format, got {seen_formats}")

    quant_config["ignore"] = EXPECTED_IGNORE
    print(f"  ignore -> {EXPECTED_IGNORE}")

    backup_path = config_path + ".as-generated"
    shutil.copy2(config_path, backup_path)
    with Path(config_path).open("w") as fh:
        json.dump(config, fh, indent=2)
        fh.write("\n")
    print(f"  rewritten; original preserved at {backup_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: fix_three_tier_quant_config.py <path-to-config.json>")
    rewrite_mixed_precision_quant_config(sys.argv[1])
