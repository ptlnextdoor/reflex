# Valen-0.8B on M4 Pro (20-core GPU, 24 GB, macOS 27)

Revision `c2c481f`. Request: one 256x144 frame + 4 questions (steer, throttle, on_road,
risk) in shared state = **302 tokens** (`input_tokens` = `compute_tokens`).

| date | device | runtime | p50 ms | p99 ms | notes |
| --- | --- | --- | --- | --- | --- |
| 2026-10-20 | CPU, 4 threads | PyTorch 2.14, transformers 5.4.0 | 5618 | 5664 | no flash-linear-attention, so DeltaNet runs the torch fallback; correctness reference only |
| 2026-10-20 | MPS (20-core GPU) | same, bf16 autocast backbone, fp32 head | **421** | 542 | 40 samples (8 frames x 5), 2 warm-up. **Baseline to beat.** 5x slower than the ~80 ms arithmetic estimate, so the PyTorch MPS path is far from compute-bound (DeltaNet torch fallback, per-op dispatch) |
| 2026-10-20 | MLX (20-core GPU) | MLX 0.32.4, mlx-vlm 0.7.9, fp16 backbone, fp32 CPU head | **86.7** | 93.5 | end to end, 40 samples, 3 warm-up, 31/32 answers (96.9%); forward p50 82.1 ms; current transformers 5.20.2 |
| 2026-10-20 | MLX (20-core GPU) | same, bf16 backbone, fp32 CPU head | 87.8 | 94.6 | end to end, 40 samples, 3 warm-up, 29/32 answers (90.6%); forward p50 81.3 ms; no material latency gain over fp16 |

MPS vs CPU agreement: 30/32 answers (93.8%). All choices and scores agree; the 2
misses are `on_road` noul deltas of 0.044 and 0.040 (gate is 0.02), from bf16 autocast
on MPS vs fp32 on CPU. Max probability delta 0.044. Reference for the gate: `ref_cpu.jsonl`
(fp32). Command: `.venv/bin/python valen_ref.py --device mps --warmup 2 --repeats 5 --out ref_mps.jsonl`

Command: `OMP_NUM_THREADS=4 .venv/bin/python valen_ref.py --device cpu --warmup 1 --repeats 1 --out ref_cpu.jsonl`

Zero-shot on synthetic frames: steer says "straight" on 7 of 8 frames with low
confidence, including frames where the road is visibly offset. Expected: Valen was never
trained on driving. That's the case for fine-tuning on sim-labeled frames (DESIGN.md L2).

## MLX port, 2026-10-20 PDT

`valen_mlx.py` reuses the pinned snapshot's processor, Compiler, role pooling,
MixerHead and response decoder. Only ViT and the text backbone move to MLX.
The 1,319,721-parameter head remains torch CPU fp32. All requests are 302 tokens,
with unchanged shared-state spans. No LM logits, generation, KV reuse, quantization,
or patched library code. `A_log` stays fp32 per mlx-vlm's cast convention.

### Correctness before timing

Against unchanged `ref_cpu.jsonl`, fp16 passes **31/32 (96.875%)**, bf16 **29/32
(90.625%)**. Both pass this bake-off's 90% minimum, neither meets the 99% engine gate.
All 16 choices and all 8 scores pass in both rows. Misses are exclusively `on_road`:
fp16 on frame 5 (absolute error 0.02648), bf16 on frames 0, 1, 5 (maximum 0.05312).
Maximum probability delta across all answers is 0.02905 for fp16 and 0.05312 for bf16.
Answers are `ref_mlx.jsonl` and `ref_mlx_bf16.jsonl`.

Last hidden state on `synth_00.png`, shape `[1, 302, 1024]`, compared with a fresh
PyTorch CPU forward in the current environment:

| MLX weights | vs true fp32 CPU max abs | mean abs | relative L2 | vs default CPU bf16 autocast max abs |
| --- | --- | --- | --- | --- |
| fp16 | 7.58861 | 0.031962 | 0.038547 | 13.89754 |
| bf16 | 10.38353 | 0.130043 | 0.093285 | 9.09802 |
| fp32 diagnostic only, not timed | 0.042353 | 0.000132 | 0.000213 | 14.82886 |

The reference hidden maximum absolute value is 42.53351. Reduced-precision hidden
errors are not near zero, despite passing the answer gate. mRoPE position IDs match
PyTorch exactly. Full HF Qwen norms are zero-centered: existing sanitize must add
`+1` to 61 vectors, before casting. Numeric ablation with fp32 weights gives hidden
max error **0.04235 with +1**, **47.16056 without +1**. Conv1d changes from
`[out, 1, kernel]` to `[out, kernel, 1]`, patch Conv3d from NCDHW to NDHWC.
Passing full HF `model.language_model.*` / `model.visual.*` names through existing
sanitize handles the remap, offsets and layouts without surgery. Evidence:
`mlx_diagnostics.json`.

**Environment drift:** installed versions are torch **2.14.1**, transformers
**5.20.2**, not the documented baseline's transformers 5.4.0. No dependencies were
changed for this port. The current snapshot also defaults to bf16 CPU autocast even
with fp32 stored parameters, so the hidden exporter explicitly captures both modes.
The stored reference was not regenerated or relabeled. On failing frame 5, fresh
PyTorch CPU fp32 gives `on_road=0.71738476`, MLX fp32 `0.71739680`, MLX fp16
`0.71910683`, versus stored `0.74558830`. Even MLX fp32 passes only 31/32 against the
stored file. Version drift is confirmed, but its causal contribution has not been
isolated. These are not clean same-environment speedup ratios versus the old MPS row.

### Synchronized stage split

Each cell is **p50 / p90 / p99 ms**, 40 samples per stage (8 frames x 5).
Three additional end-to-end warm-ups follow the untimed eight-frame correctness pass.
FP16 and BF16 were run serially. `mx.eval` and `mx.synchronize` finish work inside
each timer. Inputs and embeddings needed by isolated stages are materialized first.
With only 40 observations, nearest-rank p99 here is the observed maximum.

| weights | end to end | backbone forward | ViT only | text tower only | agreement |
| --- | --- | --- | --- | --- | --- |
| fp16 | **86.69 / 89.16 / 93.51** | **82.06 / 85.40 / 88.23** | 16.77 / 17.63 / 19.85 | 63.15 / 66.94 / 68.65 | 31/32, 96.875% |
| bf16 | 87.78 / 89.41 / 94.56 | 81.33 / 84.94 / 86.45 | 16.82 / 18.81 / 19.90 | 63.13 / 64.93 / 67.01 | 29/32, 90.625% |

End to end includes frame read/hash, image processing, tokenization, input conversion,
backbone, CPU head and answer decoding. Backbone forward excludes preprocessing and
head, includes ViT, embedding lookup, image merge, mRoPE indices and text tower.
ViT includes patch embedding through 2x2 merger. Text-only includes all 24 decoder
layers and final norm, excludes embedding/merge/mRoPE setup. The isolated stage
medians need not sum to the full forward median.

**Biggest remaining cost: text tower, about 63 ms, 77% of fp16 forward p50.**
ViT is about 17 ms. BF16 buys no material speed and loses agreement on this workload.
MLX allocator peak was 5,117,917,512 bytes (4.77 GiB) in either run, including load-time
conversion, not steady-state residency. Raw samples and definitions: `mlx_fp16.json`,
`mlx_bf16.json`. No claim of 99% correctness or production readiness.

Exact commands, from `bench/infer/`, run serially:

```bash
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 .venv/bin/python valen_mlx.py --cpu-reference .venv/valen_cpu_hidden.npz
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 .venv/bin/python valen_mlx.py --dtype float32 --out .venv/ref_mlx_fp32.jsonl --summary .venv/check_fp32.json
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 .venv/bin/python valen_mlx.py --dtype float16 --bench --warmup 3 --repeats 5 --out ref_mlx.jsonl --summary mlx_fp16.json
OMP_NUM_THREADS=4 HF_HUB_OFFLINE=1 .venv/bin/python valen_mlx.py --dtype bfloat16 --bench --warmup 3 --repeats 5 --out ref_mlx_bf16.jsonl --summary mlx_bf16.json
HF_HUB_OFFLINE=1 .venv/bin/python valen_mlx.py --selftest
```

## Reference fixed: true fp32 (coordinator, 2026-10-20)

The original `ref_cpu.jsonl` was **not fp32**: Valen's snapshot runs the backbone under
bf16 autocast even on CPU. `valen_ref.py --fp32` now disables that
(`backbone_autocast_dtype = None`). New gate reference: `ref_cpu_fp32.jsonl`.

| runtime | agreement vs true fp32 | max prob/score delta | misses |
| --- | --- | --- | --- |
| **MLX fp16** (`ref_mlx.jsonl`) | **32/32, 100%** | 0.0065 | none |
| old CPU bf16-autocast ref (`ref_cpu.jsonl`) | 31/32, 96.9% | 0.0282 | 05 on_road |
| MPS bf16 autocast (`ref_mps.jsonl`) | 30/32, 93.8% | 0.0456 | 00, 04 on_road |
| MLX bf16 (`ref_mlx_bf16.jsonl`) | 29/32, 90.6% | 0.0329 | 00, 04, 05 on_road |

So the MLX fp16 port **passes the 99% gate**; the earlier misses came from the bf16
reference, not the port. Every bf16 path loses the noul gate; fp16 is the default.
Same environment for all rows here (torch 2.14.1, transformers 5.20.2).
Command: `OMP_NUM_THREADS=8 .venv/bin/python valen_ref.py --device cpu --fp32 --warmup 0 --repeats 1 --out ref_cpu_fp32.jsonl`

**Current best: MLX fp16, 86.7 ms p50 end to end (82.1 ms forward), 100% agreement.
4.9x faster than PyTorch MPS (421 ms). Text tower is 63 ms of it.**

## Text-tower token scaling (coordinator, 2026-10-20)

`scaling.py`: Valen text tower in MLX fp16, real weights, random embeddings (timing
only), 3 warm-up + 15 timed per size. Command: `HF_HUB_OFFLINE=1 .venv/bin/python scaling.py`

| tokens | 8 | 16 | 32 | 64 | 96 | 128 | 192 | 302 | 512 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| p50 ms | 9.0 | 14.9 | 17.6 | 18.0 | 28.6 | 32.3 | 43.4 | 67.8 | 106.6 |

Linear fit: **9.1 ms fixed + 0.19 ms/token.** At 302 tokens the per-token term is ~85%
of the time, so cutting tokens pays off nearly proportionally. Below ~64 tokens it
flattens at ~18 ms (launch/dispatch bound, 24 layers). Implications:
- Question-first streaming (only ~64 visual + ~10 readout tokens per frame) predicts
  the text tower at **~18 to 20 ms** (from 63), so ~37 ms total with the 17 ms ViT.
- The ~9 to 18 ms floor is the 24-layer launch cost. Below it needs fused kernels
  (sys1rust's lever) or a smaller backbone.
