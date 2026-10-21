# Valen-0.8B on M4 Pro (20-core GPU, 24 GB, macOS 27)

Revision `c2c481f`. Request: one 256x144 frame + 4 questions (steer, throttle, on_road,
risk) in shared state = **302 tokens** (`input_tokens` = `compute_tokens`).

| date | device | runtime | p50 ms | p99 ms | notes |
| --- | --- | --- | --- | --- | --- |
| 2026-10-20 | CPU, 4 threads | PyTorch 2.14, transformers 5.4.0 | 5618 | 5664 | no flash-linear-attention, so DeltaNet runs the torch fallback; correctness reference only |
| 2026-10-20 | MPS (20-core GPU) | same, bf16 autocast backbone, fp32 head | **421** | 542 | 40 samples (8 frames x 5), 2 warm-up. **Baseline to beat.** 5x slower than the ~80 ms arithmetic estimate, so the PyTorch MPS path is far from compute-bound (DeltaNet torch fallback, per-op dispatch) |

MPS vs CPU agreement: 30/32 answers (93.8%). All choices and scores agree; the 2
misses are `on_road` noul deltas of 0.044 and 0.040 (gate is 0.02), from bf16 autocast
on MPS vs fp32 on CPU. Max probability delta 0.044. Reference for the gate: `ref_cpu.jsonl`
(fp32). Command: `.venv/bin/python valen_ref.py --device mps --warmup 2 --repeats 5 --out ref_mps.jsonl`

Command: `OMP_NUM_THREADS=4 .venv/bin/python valen_ref.py --device cpu --warmup 1 --repeats 1 --out ref_cpu.jsonl`

Zero-shot on synthetic frames: steer says "straight" on 7 of 8 frames with low
confidence, including frames where the road is visibly offset. Expected: Valen was never
trained on driving. That's the case for fine-tuning on sim-labeled frames (DESIGN.md L2).
