# Valen-0.8B on M4 Pro (20-core GPU, 24 GB, macOS 27)

Revision `c2c481f`. Request: one 256x144 frame + 4 questions (steer, throttle, on_road,
risk) in shared state = **302 tokens** (`input_tokens` = `compute_tokens`).

| date | device | runtime | p50 ms | p99 ms | notes |
| --- | --- | --- | --- | --- | --- |
| 2026-10-20 | CPU, 4 threads | PyTorch 2.14, transformers 5.4.0 | 5618 | 5664 | no flash-linear-attention, so DeltaNet runs the torch fallback; correctness reference only |
| pending | MPS | same | | | baseline to beat, run when no other GPU job |

Command: `OMP_NUM_THREADS=4 .venv/bin/python valen_ref.py --device cpu --warmup 1 --repeats 1 --out ref_cpu.jsonl`

Zero-shot on synthetic frames: steer says "straight" on 7 of 8 frames with low
confidence, including frames where the road is visibly offset. Expected: Valen was never
trained on driving. That's the case for fine-tuning on sim-labeled frames (DESIGN.md L2).
