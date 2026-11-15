# 2026-11-15 - Cause Found: llama-server Closes the Connection After Every Streamed Answer

**Goal:** Find why the full replica benchmark failed on the first request of every workload
(journal 2026-11-14, moved out of the catalog; issue #17), and record what fixes it, so the
fix can be finished later.

## Where it was tested

Inference node `ain-4euk3` (two RTX PRO 6000 Blackwell, 62 GiB RAM), development control
plane. The engine was also run directly with Docker on the same weights to read its output
and to compare builds.

## The cause

The image's llama-server (`onpremai/llama-cpp-turboquant@sha256:ecc8ac1f...`, built
2026-05-08 from the TurboQuant fork) closes the TCP connection right after the final chunk
of every streamed answer (`"stream": true`), although the answer's headers say
`Keep-Alive: timeout=5, max=100` and carry no `Connection: close`.

The load generator streams every request and keeps connections open, as most HTTP clients
do. Its next request goes into the socket the server has already closed and fails at once
with `EOF` or `write: broken pipe`. The server never sees that request. Go does not retry a
POST on a reused connection.

Evidence, all on 2026-11-15:

| Test | Result |
|---|---|
| One raw TCP connection, streamed requests one after another | the server closes the socket after the first stream ends |
| The same with `"stream": false` | the connection stays open for all 12 requests |
| Go client like the load generator, 40 streamed requests of about 7,400 tokens | 8 of 40 failed, each on a reused connection with zero idle time |
| The same, not streamed | 0 of 40 failed |
| The load generator's own `max-prefill/1.0.0` against the engine | 1 of 8 requests failed at concurrency 1 (`Post ...: EOF`) |
| Upstream `ghcr.io/ggml-org/llama.cpp@sha256:e8d36f4d...`, same weights, 40 streamed requests | 0 of 40 failed; the connection stays open |

An idle timeout is not the cause: requests 4.8 to 6 seconds apart did not fail, and the
failing ones had zero idle time.

## Known upstream

- ggml-org/llama.cpp#29680 (opened 2026-09-29): in `tools/server/server-http.cpp` the
  chunked content provider returns `has_next` (false) after `sink.done()`, and cpp-httplib
  treats a provider returning false as a cancelled write and drops the connection.
- Fixed by ggml-org/llama.cpp#30492, "Fix connection reuse after streamed responses",
  merged 2026-11-14 (commit `cf9a9957da`).
- The TurboQuant fork (TheTom/llama-cpp-turboquant, default branch
  `feature/turboquant-kv-cache`) had not taken the fix by 2026-11-15; its newest commits
  are from 2026-11-11.
- Other clients hit the same bug: llama-swap (mostlygeek/llama-swap#1205, fixed on the
  client side by #1211 with `disableKeepAlives`), .NET and Python clients
  (ggml-org/llama.cpp#31863).

## A second fault: the host memory booking

On `ain-4euk3` the replica did not start: the container was killed for memory
(`OOMKilled=true`, exit 137) under the profile's 8 GiB limit. With 1,048,576 tokens of
context llama.cpp allocates `CUDA_Host compute buffer size = 33232.34 MiB` in host memory
(pinned), next to `CUDA0 compute buffer size = 50056.31 MiB` on the GPU. A run without the
limit peaked at 35.6 GiB of container memory at start. The profile's `host_bytes` (16 GiB)
and its 8 GiB container limit are both too small; about 40 GiB is needed. (On `ain-xtg6o`
on 2026-11-14 it started; why the limit did not apply there is not known.)

## Long context

The model's trained context is 262,144 tokens. This profile runs 1,048,576 with no rope
scaling; llama.cpp warns `n_ctx_seq (1048576) > n_ctx_train (262144) -- possible training
context overflow`. Answers beyond 262,144 tokens are not tested.

## To finish

1. Rebuild the TurboQuant image with the upstream fix (#30492, one line: return `true`
   after `sink.done()`), or with a TurboQuant fork release that includes it. Pin it by
   digest.
2. Raise the host memory: `host_bytes` and the compose memory limit to about 40 GiB.
3. Before the benchmark, test a hidden fact at about 200K, 500K and 900K tokens; if the
   model fails beyond 262,144 tokens, publish a 256K profile instead of 1M.
4. Run the full replica benchmark, then bring the profile back to `main` by pull request.
