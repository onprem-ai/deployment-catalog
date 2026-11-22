# 2026-11-13 - Initial Catalog Entry

**Goal:** A first card for `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` on two RTX PRO 6000
Blackwell GPUs, next to the `deepseek-v4-flash-dspark-0731` card it succeeds. Written from
the model card, vLLM's recipe and the 0731 card's measurements; not deployed yet.

## The model

- `deepseek-ai/DeepSeek-V4-Flash-Vision-Exp` at `6821d6ad3681a4b137b066b76094fa82ebd0a380`,
  released 2026-08-31, MIT. DeepSeek's first multimodal V4 model: the V4 Flash language
  model (`DeepseekV4ForCausalLM`, 43 layers, 256 routed experts with 6 active, 1,048,576
  tokens) with a 32-layer, 1024-dimension vision encoder and a two-layer aligner (about
  0.5B parameters), and continued training. DeepSeek reports it beats V4 Flash 0731 on
  multimodal agent tasks (ApexBench 36.5 against 26.2) and holds or improves the text agent
  scores (Terminal Bench 2.1 83.9 against 82.7, DeepSWE 59.3 against 54.4).
- Weights as 0731: FP4 experts (`expert_dtype: fp4`) and FP8 block weights elsewhere, the
  vision tower in BF16, 48 safetensors files of 156.3 GiB (0731: 152.0 GiB). The DSpark
  drafter is fused into the checkpoint (`dspark_block_size: 5`); `num_nextn_predict_layers`
  is 3 (0731: 1).
- `config.json` has no `auto_map`. The repository's Python files (`encoding/`,
  `inference/`) are DeepSeek's reference code; vLLM has its own encoder
  (`--tokenizer-mode deepseek_v4`), so the file list names none of them and the command has
  no `--trust-remote-code`.
- Images: each costs at most 387 prompt tokens (`vision_max_n_token: 384`); images below
  147,456 pixels are upscaled; aspect ratios beyond 8:1 are cropped.

## The engine

vLLM support is in vllm-project/vllm#54566. The model card and vLLM's recipe name the image
`vllm/vllm-openai:deepseekv4-flash-vision` (2026-09-01), pinned here by its multi-arch
digest. Without it, the release wheels of that time routed the checkpoint to the text-only
class and failed on the vision tensors. vLLM v0.33.0 registers
`DeepseekV4ForConditionalGeneration` too, but nothing routes this checkpoint's
`DeepseekV4ForCausalLM` to it there; the first deployment shows which image to keep.

vLLM's recipe verified the model only on GB200 and MI350X (TP4, 32K context); RTX PRO 6000
(SM120) has no published run. The 0731 card needed its own prebuilt image with SM120 kernel
fixes and FlashInfer settings (`onpremai/vllm-ds4f:0731-prebaked-v2`); this card starts from
vLLM's image and its defaults, so the first deployment may need the same fixes.

## Starting settings

- Two GPUs, tensor parallel 2 with expert parallel, as vLLM's recipe.
- 95 GiB hardware booking and 93 GiB engine booking per GPU: the 0731 card's measured
  figures (94.67 GiB peak), as an estimate. The weights are about 2 GiB larger (the vision
  tower), so the KV cache is smaller than 0731's 539,477 tokens.
- 262,144 tokens of context (as 0731; 1M is advertised but not measured anywhere), 32
  sequences (0731's tuned value), FP8 KV cache, block size 256, prefix caching.
- DSpark with 3 draft tokens, probabilistic sampling and adaptive verification: the value
  vLLM's recipe measured (2.99 mean acceptance on OCRBench) and recommends until the
  acceptance is verified; the 0731 card found 5 refused below the drafter's block size on
  its own image, which this image's recipe does not mention.
- Reasoning and tool-call parsers `deepseek_v4`, with the reasoning config of the recipe.

## Next

- Deploy on two RTX PRO 6000; if vLLM's image fails on SM120, carry over the 0731 card's
  fixes (its image or its environment settings).
- Answer checks with images, measure the memory, tune (draft length, sequences, CUDA graph
  sizes as on 0731) and run the full benchmark.

## Sources

- https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp
- https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash-Vision-Exp
- https://github.com/vllm-project/vllm/pull/54566
