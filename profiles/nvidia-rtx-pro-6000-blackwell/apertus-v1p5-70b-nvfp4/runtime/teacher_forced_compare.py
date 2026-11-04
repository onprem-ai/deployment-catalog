#!/usr/bin/env python3
"""Teacher-forced logit comparison between two served checkpoints.

Method from catalog journal 2026-07-07_two_pass_success_and_benchmarks.md:
fixed mixed-domain texts, per-position top-k prompt logprobs via the vLLM
completions API, then top-1 agreement and top-10 KL divergence.

Usage:
  teacher_forced_compare.py collect <server_url> <out.json>
  teacher_forced_compare.py compare <a.json> <b.json>
"""

import json
import math
import sys
import urllib.request
from pathlib import Path

FIXED_TEXTS = [
    "The mitochondria is the powerhouse of the cell. During oxidative "
    "phosphorylation, electrons are transferred through the electron "
    "transport chain, creating a proton gradient across the inner membrane.",
    "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n"
    "    pivot = arr[len(arr) // 2]\n    left = [x for x in arr if x < pivot]",
    "La Suisse est un pays multilingue situe au coeur de l'Europe. Ses "
    "quatre langues nationales sont l'allemand, le francais, l'italien et "
    "le romanche.",
    "The EU AI Act establishes a risk-based regulatory framework for "
    "artificial intelligence systems, with obligations scaling from minimal "
    "to unacceptable risk categories.",
    "Q: A train travels 120 km in 1.5 hours. What is its average speed?\n"
    "A: Average speed equals distance divided by time, so 120 / 1.5 = 80 km/h.",
]

TOP_K = 10


def collect_prompt_logprobs(server_url: str, output_path: str) -> None:
    results = []
    for text in FIXED_TEXTS:
        body = json.dumps({
            "model": "probe",
            "prompt": text,
            "max_tokens": 1,
            "temperature": 0,
            "prompt_logprobs": TOP_K,
        }).encode()
        request = urllib.request.Request(
            f"{server_url}/v1/completions", data=body,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            payload = json.load(response)
        results.append(payload["choices"][0]["prompt_logprobs"])
    with Path(output_path).open("w") as fh:
        json.dump(results, fh)
    positions = sum(len(r) - 1 for r in results)  # first position has no logprobs
    print(f"collected {positions} positions -> {output_path}")


def compare_logprob_files(path_a: str, path_b: str) -> None:
    with Path(path_a).open() as fh:
        collected_a = json.load(fh)
    with Path(path_b).open() as fh:
        collected_b = json.load(fh)

    top1_matches = 0
    total_positions = 0
    kl_values = []
    for text_a, text_b in zip(collected_a, collected_b, strict=True):
        for pos_a, pos_b in zip(text_a, text_b, strict=True):
            if pos_a is None or pos_b is None:
                continue
            total_positions += 1
            top1_a = max(pos_a, key=lambda t: pos_a[t]["logprob"])
            top1_b = max(pos_b, key=lambda t: pos_b[t]["logprob"])
            if top1_a == top1_b:
                top1_matches += 1
            # KL(a||b) over a's top-k support; missing tokens in b get a floor
            floor = min(entry["logprob"] for entry in pos_b.values()) - 5.0
            kl = 0.0
            norm_a = math.log(sum(math.exp(e["logprob"]) for e in pos_a.values()))
            norm_b = math.log(sum(math.exp(e["logprob"]) for e in pos_b.values()))
            for token_id, entry in pos_a.items():
                lp_a = entry["logprob"] - norm_a
                lp_b = (pos_b[token_id]["logprob"] if token_id in pos_b else floor) - norm_b
                kl += math.exp(lp_a) * (lp_a - lp_b)
            kl_values.append(kl)

    kl_values.sort()
    print(f"positions compared: {total_positions}")
    print(f"top-1 agreement:    {top1_matches / total_positions:.3f}")
    print(f"top-{TOP_K} KL mean:     {sum(kl_values) / len(kl_values):.4f}")
    print(f"top-{TOP_K} KL median:   {kl_values[len(kl_values) // 2]:.4f}")
    print(f"top-{TOP_K} KL max:      {kl_values[-1]:.4f}")


if __name__ == "__main__":
    if len(sys.argv) != 4 or sys.argv[1] not in ("collect", "compare"):
        sys.exit(__doc__)
    if sys.argv[1] == "collect":
        collect_prompt_logprobs(sys.argv[2], sys.argv[3])
    else:
        compare_logprob_files(sys.argv[2], sys.argv[3])
