# DeepSeek V4.1 Flash 1017 Abliterated, cyber (pruned 210/384 experts, 2x GPU)

> **Uncensored model.** Safety refusals have been substantially removed; it will attempt almost any
> request the original refuses. Deploy it only where that is intended.

DeepSeek V4.1 Flash 1017 for security work on two RTX PRO 6000 Blackwell GPUs:

- **Abliterated** with the overlay `onprem-ai--DeepSeek-V4.1-Flash-1017-abl` (the same 86 attention
  output tensors as the four-GPU card `deepseek-v4-1-flash-1017-abl` on this branch; see its README for
  the edit). `container-files/abl41.py` replaces those tensors by name while DeepSeek's checkpoint loads.
- **Pruned at load time** to 210 of 384 routed experts per layer (`container-files/keep210-cyber.json`,
  applied by `expert_keep.py`): the experts that carry the most routing mass on public security work
  (advisories and their fix commits, Metasploit modules, Sigma rules, ATT&CK, CVEs, vulnerable-code
  review, practitioner questions, CTF write-ups, security-relevant coding), counted on prompt and
  generated tokens, with the experts the model reasons with protected (reasoning routing mass at least
  95 % of the best possible per layer). The DSpark drafter keeps all 128 experts.
- About 502B parameters, 16B active; 8 parallel requests; 200k context; a KV pool of 2,082,304 tokens.

## Weights

1. **`main`**: DeepSeek's [`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
   at `1bae0e1088993b9b82de08f59de5be5a34e42170`, unchanged (the same files as the official card, so a
   cached copy is reused). The routed experts outside the keep set are not loaded.
2. **`overlay`**: OnPrem AI model storage `onprem-ai--DeepSeek-V4.1-Flash-1017-abl` (1.8 GB, tensors only).

## Validation (two GPUs; reference: the unpruned abliterated card on four GPUs)

| | This card | Unpruned abliterated |
|---|---|---|
| CyberMetric-500 / CTIBench cti-mcq (300) / SecQA test | 94.8 / 78.0 / 100 % | 95.6 / 75.0 / 100 % |
| Harmful refusal, chat / thinking (AdvBench held-out, 64 each) | 0 % / 4.7 % | 6.3 % / 3.1 % |
| False refusals on 64 legitimate security questions | 0 % | 0 % |
| GSM8K, first 250, thinking | 96.4 % | 98.0 % |
| HumanEval pass@1 | 95.1 % | 97.6 % |
| Tool calls | 7/8 | 8/8 |
| Needle, fresh prompts up to 170k tokens | 4/4 | - |
| Decode, one request | 121 tok/s, DSpark 4.42 tokens per step | - |
| Platform benchmark: decode 1 request / plateau; prefill plateau | 91 / 360 tok/s at 12; 2,107 tok/s | - |

Refusals are marker-matched (reasoning excluded); sample sizes are small (0 of 64 has a 95 % upper
bound of about 6 %). How the keep set was built and measured, including what went wrong first, is in
`journal/`; the measurement design and the context limit are in the journal of the coder variant
`deepseek-v4-1-flash-1017-prune210-coder` on `main` of this repository.

## Licenses

MIT (DeepSeek, `licenses/source-0-LICENSE.txt`). The ranking corpora are not part of the card.
