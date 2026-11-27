# 2026-11-27 08:45 Security corpus and the first cyber keep set (prompt routing)

**Goal:** a 2-GPU abliterated DeepSeek V4.1 Flash 1017 for security work: the abliteration overlay
`onprem-ai--DeepSeek-V4.1-Flash-1017-abl` (unchanged) plus 210 of 384 routed experts per layer ranked on
security work, all 128 DSpark drafter experts, 8 requests, about 2M tokens of KV pool. The coder
variant (`deepseek-v4-1-flash-1017-prune210-coder`) was built the same way; the routing counter is
described in its journal `2026-11-27-0830-prompt-routing-ranking.md`. Routing was measured on the
unpruned abliterated card (overlay active, four GPUs).


Every corpus is user prompts only, built by `build_corpora.py` (seed 1) from public sources at pinned
revisions. The prompts are model inputs for the routing statistics; no answer was stored, and no corpus
text is part of this card. Token counts are DeepSeek V4.1's own tokenizer.

**Coder:** 8,000 of the 10,000 prompts of OnPrem AI storage `onprem-ai--coding-prompts--v1`
(`SHA256SUMS` sha256 `820d91ee...`; CommitPackFT edit-file, WildChat write-code and discuss-code; 80/20
per kind, the other 2,000 held out), 3,336,608 prompt tokens. No other corpus: the coding prompts carry
the surrounding natural language (tickets, commit messages, discussions) themselves.

**Cyber:** security work only, 7,407 prompts, 4,157,474 tokens:

| Source | Prompts | Tokens | Share | Revision | License |
|---|---|---|---|---|---|
| GitHub Security Advisories (reviewed, newest 1,000): explain the vulnerable pattern and the patch | 900 | 967,546 | 23.3 % | API snapshot 2026-11-27 | CC-BY-4.0 |
| Security fix commits linked from those advisories and from the CVEs below (`.patch`): exploitability and completeness of the fix | 423 | 892,297 | 21.5 % | commits as linked | each repository's license |
| Metasploit module documentation (`rapid7/metasploit-framework`): the module in an authorized engagement, its output, detection | 600 | 876,896 | 21.1 % | `251027d26210` | BSD-3-Clause |
| Sigma rules (`SigmaHQ/sigma`): explain, convert to SPL/KQL, triage | 900 | 369,598 | 8.9 % | `876eb7b8cfbc` | DRL 1.1 |
| Security-relevant coding prompts of coding-rank (auth, crypto, input handling, security tooling) | 362 | 236,498 | 5.7 % | as above | as above |
| MITRE ATT&CK techniques (`mitre/cti`): detection, hunting, incident response | 700 | 232,863 | 5.6 % | `e79c7c53d13c` | ATT&CK terms of use |
| NVD CVE API 2.0, most recent 6,000 CVEs: analysis, triage, advisory | 1,197 | 197,230 | 4.7 % | API snapshot 2026-11-27 | public |
| `CyberNative/Code_Vulnerability_Security_DPO`: security code review of the vulnerable code | 1,287 | 162,190 | 3.9 % | `bb962a250ec4` | Apache-2.0 |
| security.stackexchange questions naming tools, attacks or artifacts (`flax-sentence-embeddings/stackexchange_titlebody_best_voted_answer_jsonl`) | 625 | 136,981 | 3.3 % | `5ce5373dcaed` | CC-BY-NC-SA-4.0 |
| CTF write-ups (`welchbj/ctf`) | 63 | 34,633 | 0.8 % | `8050dac9256c` | MIT |
| MITRE ATT&CK malware, groups, tools: threat-intelligence briefs | 350 | 50,742 | 1.2 % | `e79c7c53d13c` | ATT&CK terms of use |

The security-relevant coding share is 5.7 %, not 30 %: only 362 of the 8,000 coding prompts are security
work (a keyword filter; a few are ordinary authentication or permission code), and the corpus is not
padded with other coding. CTF write-ups are thin: the large public write-up archives carry no license.
Left out on purpose: generic instruction and chat data, general knowledge (the CWE dictionary), and
multiple-choice security quizzes (used only for evaluation).

Evaluation sets are disjoint from the ranking data: HumanEval (`openai/openai_humaneval` `7dce6050a7d6`,
MIT), CyberMetric-500 (`tihanyin/CyberMetric` `32759c2fa902`, Apache-2.0), CTIBench cti-mcq
(`AI4Sec/cti-bench` `9237e1636ee3`, CC-BY-NC-SA-4.0), SecQA v1 test (`zefang-liu/secqa` `d00a07484283`,
CC-BY-NC-SA-4.0), GSM8K test, the AdvBench held-out prompts of the abliteration work, and 64
security.stackexchange questions that are not in the cyber corpus.

## First keep set and results

7,407 prompts, 4,187,102 prompt tokens, 26 minutes. Per layer the 210 experts with the most gate-weight
mass of the prompt tokens:

| Keep set | Cyber routing mass kept, mean / worst | Coding routing mass kept, mean / worst |
|---|---|---|
| keep210-cyber (prompt) | 92.1 % / 85.2 % | 82.7 % / 77.8 % |
| keep210-coder (prompt) | 85.3 % / 75.8 % | 90.4 % / 83.4 % |
| keep224r2 (prune224) | 74.8 % / 58.7 % | 68.9 % / 60.9 % |

The cyber and coder sets share 168 of 210 experts per layer on average.

| (2 GPUs; reference: the unpruned abliterated card, 4 GPUs) | prune210-cyber-abl (prompt ranking) | unpruned abliterated |
|---|---|---|
| CyberMetric-500 | 94.0 % | 95.6 % |
| CTIBench cti-mcq, first 300 | 76.3 % | 75.0 % |
| SecQA v1 test (110) | 100 % | 100 % |
| HumanEval pass@1 | 92.1 % | 97.6 % |
| **GSM8K, first 250, thinking** | **32.8 %** (59 cut off) | 98.0 % |
| Harmful refusal, chat / think-high (AdvBench held-out, 64 each) | 3.1 % / 1.6 % | 6.3 % / 3.1 % |
| False refusals, 64 legitimate security questions | 1.6 % | 0 % |
| Thinking answers to harmful prompts cut off at 4,096 tokens | 51.6 % | 12.5 % |
| Tool calls | 8/8 | 8/8 |
| Needle, 8k-120k | 11/12 | 12/12 |
| Decode, one request | 111 tok/s, DSpark 4.49 | – |

The abliteration survives the pruning (refusal stays low, no second overlay needed). Security knowledge
held, reasoning did not: GSM8K collapsed and thinking ran to the token limit. The cause and the fix are
in `2026-11-27-1130-decode-routing.md` (this card) and the coder card's journal.
