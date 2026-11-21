# Repository Instructions

The deployment catalog: one directory per GPU product, one directory per deploy profile
(`ai-deploy.yaml`, `compose.yaml`, `journal/`, `benchmarks/`).

## Non-negotiable: warn and hint, never block

Warnings and hints are welcome; they never stop a deploy or an edit. A check, a figure that
looks too high, or a profile that does not fit a GPU today is a warning that says the real
numbers and what to do about it. The only exception is a consequence that can brick the
machine or destroy data that cannot be recovered, and a block needs the developer's proof of
that consequence in the change that adds it. The rule and its reasons are in the platform
repository (`onprem-ai/platform-alpha`): the "warn and hint, never block" section of its
`CLAUDE.md` and `docs/architecture/gpu-memory-allocation.md`.

## Hardware booking target

Keep a profile's hardware booking (`hardware.gpu.memory_bytes_per_device`) at or below the
product's total GPU memory minus 1 GiB reserved for the driver (RTX PRO 6000 Blackwell Server
Edition: 97,887 MiB total, 94.59 GiB bookable, 94 GiB in whole GiB). It is a target, not a
limit: a profile whose measured peak is over it keeps a booking above it, and its journal says
so and says it has no margin for a driver that reserves more. Never write a booking above the
card's total: no GPU of the product can ever show that much as free.
