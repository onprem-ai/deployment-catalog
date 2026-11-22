# 2026-11-21 Hardware booking: 95 GiB to 94 GiB, so that a GPU can show it as free

`hardware.gpu.memory_bytes_per_device` (the hardware booking) goes from 95 GiB to 94 GiB. Nothing else changes: the engine's
own memory budget (`engine_memory_booking_bytes_per_device`), the command, the environment and the measured card stay.

## Why

An idle RTX PRO 6000 Blackwell Server Edition reports 97,887 MiB in total (ECC enabled, driver 595.71.05) and the driver reserves
636 MiB of it (`nvidia-smi --query-gpu=memory.reserved`; `memory.used` is 0 MiB). The free memory is therefore 94.97 GiB, 28 MiB
less than a booking of 95 GiB, so the placement proposal marked every GPU as not eligible for this profile and showed
"95 GiB free, this model allocates 95 GiB". The deploy itself is never blocked (the booking is advice), but the proposal was
wrong and the message confusing.

## The figure

| | GiB |
|---|---|
| Measured peak per GPU (the card's parts added up) | 93.11 |
| Booking before | 95.00 |
| Booking now | 94.00 |
| Free on an idle GPU of this product (driver 595.71.05, ECC on) | 94.97 |

94 GiB is above the measured peak by 0.89 GiB and below the free memory by 0.97 GiB, which covers a driver that reserves
somewhat more (NVIDIA documents that the driver reserves memory for its own use; how much depends on the driver and its GPU
firmware, and the ECC setting changes the total).
A GPU that has more reserved or used by something else than that is still only warned, not refused.
