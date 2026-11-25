# 2026-11-25 Without remote code: what was checked, and what changed in the card

**Goal:** make the abliterated card safe to run: no `--trust-remote-code`, no executable file from the uploader in the
weights, and know exactly what differs from DeepSeek's own checkpoint.

## What changed in the card

- `compose.yaml`: `--trust-remote-code` removed; the superseded `--max-num-seqs 128` and the repeated `--kv-cache-dtype fp8`
  removed (the last `--max-num-seqs 8` was the one in effect, so the engine settings are unchanged).
- `ai-deploy.yaml`: `encoding/encoding_dsv4.py` removed from the weights files (the checker's
  `model.weights-executable-files` warning is gone; the official card does not list it either);
  `weights_disk_size_bytes` 163,208,757,248 -> 168,442,117,640 (the official card's figure was copied; the listed files add up to
  168,442,117,640, which is what the node downloaded); the description rewritten (typo, GPU name, a machine-specific phrase).
- `README.md` added.

## Why the flag and the file were not needed

`encoding/encoding_dsv4.py` is DeepSeek's own file: the uploader's copy has the same SHA-256 as the one in
`deepseek-ai/DeepSeek-V4-Flash-0731`. `config.json`, `tokenizer.json`, `tokenizer_config.json` and `generation_config.json` are
byte-identical to the official ones too, and the official card runs the same image (`vllm-ds4f:0731-prebaked-v2`) with the same
tokenizer mode and parsers without the flag. The card was started without it and without the file on two GPUs: ready in
3 min 51 s, weights loaded in 108 s, 616,830 KV cache tokens.

## What was verified

See the README ("Weights"): 72 of 74 files byte-identical to the official repository; the stored copy equal to the uploader's
repository by SHA-256; the overlay holds only `wo_b` weights and scales of the 43 layers and 3 MTP blocks; three layers audited as
rank-1 directional ablation (lambda 2.43 to 2.47). Validation against the official model on the same harness: README ("Validation").

## Not done

- Not every one of the 46 edited matrices was audited, and the uploader's pipeline was not rerun here.
- Thinking modes were not measured by us.
- The benchmark file of 2026-11-14 predates these compose changes (its settings are the ones in effect, but its compose hash
  differs); it was not rerun.
- The hardware booking (95 GiB) is above what a GPU of this product shows as free (94.97 GiB); see the separate commit.

## Hardware booking: 95 GiB -> 94.75 GiB (separate commit)

The measured parts of the card add up to a peak of 94.43 GiB per GPU (weights 79.36, CUDA graphs 0.14, KV cache 10.55, overhead 4.38).
An idle RTX PRO 6000 Blackwell Server Edition shows 94.97 GiB free (97,887 MiB total, 636 MiB reserved by the driver), so the
previous booking of 95 GiB was 28 MiB more than a GPU can show as free and the placement proposal marked every GPU as not eligible
(platform-alpha #384; the deploy itself is never blocked). 94.75 GiB is above the peak by 0.32 GiB and below the free memory by 0.22 GiB.
It is above the catalog's target of the product's total minus 1 GiB (94.59 GiB): the peak itself is above the target, so this card
keeps no margin for a driver that reserves more. Drop this commit if the booking should stay at 95 GiB.
