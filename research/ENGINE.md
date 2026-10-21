# Engine design research (2026-10-20, M4 Pro 20-core GPU, 24 GB, macOS 27)

Sources: `bench/infer/RESULTS.md` (all measured ms below, commits 85e8c75 to 573b565),
transformers 5.x `models/qwen3_5/modeling_qwen3_5.py`, mlx-vlm 0.7.9
`models/qwen3_5/{gated_delta,language}.py`, Valen snapshot c2c481f (`config.json`,
`heads.py`, `compiler.py`), sys1rust `results/{SPEED,REPORT}.md`, Jev docs (Box, images),
Visual Jev arXiv 2609.25845. CPU op counts below were taken with `torch.profiler` on
this Mac (no GPU). The MPS number was not re-run.

## Verdict (10 lines)

1. **Architecture: MLX fp16 Valen-0.8B backbone in Rust (sys1rust layout),
   question-first streaming after a LoRA fine-tune of the backbone in that order (option
   a), Box as a 4-edge distribution head tapped on the ViT token grid.** Python mlx-vlm
   stays as the reference and training harness. Option (b), candidates after the image,
   is rejected: only 22 tokens are cacheable in trained order, ~4 ms saved (82 vs 86.7).
   Prompt trimming measured: 1.5 ms saved, agreement 17/32. Not a lever.
2. **Expected p50 per frame: 86.7 ms today (exact), ~55 ms after the question-first
   fine-tune (measured cache path), ~30 ms with a token diet, floor ~15 to 18 ms on this
   GPU with Valen's shape.** 120 Hz (8.3 ms) is unreachable with a 24-layer tower
   (9 ms launch floor measured); it needs the distilled student (lever 4), ~3 to 5 ms.
3. **Experiments, in order:** (a) LoRA fine-tune Valen on sim-labeled frames in
   question-first order, gate = sim-label accuracy vs the same LoRA trained state-first;
   (b) token diet (terse questions, 1 readout token each, min_pixels 128 patches),
   measure text + ViT; (c) ViT-S student on sim boxes, measure IoU and ms.
4. Most wrong DESIGN.md claim: "~80 ms compute-bound". Measured text tower is
   **9.1 ms fixed + 0.19 ms/token**; at 302 tokens matmul FLOPs explain < 1/3 of it.
5. Second: "question-first streaming needs fine-tune" is right but for the wrong
   reason: the trained head reads the frame 99.2% through candidate tokens, so
   reordering freezes the answer (15/32), and head-only retraining tops out at 64%
   (frozen question-first context rows hold only R^2 ~0.7 of the signal). Backbone LoRA.
6. Box edge-distribution head is right, but tap the **pre-merge ViT grid (12x22)**, not
   the 6x11 merged tokens: 2x2 merge = 32 px per bin on a 352x192 resample, ~23 px on
   the 256x144 source, coarser than the car (~20 px wide in the synth frames).
7. Skip int8: sys1rust measured no speedup from MLX quantized matmuls (dequant to fp16)
   and bf16 already fails the noul gate on Valen. fp16 only.
8. Straight-to-Rust is premature: the levers are token count and head retrain, both
   Python-measurable in a day each. Rust buys the last ~10 to 15% plus no-Python.
9. Keep Valen as the per-frame model only until the student exists. Expected student
   path: ViT-S/16 at 144 patches = 6.5 GFLOP, ~3 to 5 ms; Valen becomes teacher + fallback.
10. Top risk: the question-first LoRA lands below the state-first LoRA (P ~0.3).
    Settles in one paired training run on ~5k sim-labeled frames.

## 1. Where do the 421 ms (PyTorch MPS) go, and what MLX gets

**Op count, measured on CPU with torch.profiler** (same graph as MPS; MPS adds a launch
per op): one 302-token `forward_state` dispatches **13,123 non-view aten ops**
(view/reshape/select excluded). Breakdown: backbone fp32 6,742; **under bf16 autocast
12,674** (autocast adds ~5,300 `_to_copy`/`to`/`copy_` casts, every linear casts its
weight each call); head 444 (196 feature pooling + 248 Mixer). Per layer: DeltaNet 276
ops, attention 142, ViT 748 for 12 blocks.

The DeltaNet fallback (`torch_chunk_gated_delta_rule`, line 287) is **chunked, not
per-token**: chunk 64, 5 chunks for 302 tokens, two `linalg_solve_triangular` calls
(MPS has a kernel), then a Python loop of 5 chunk steps with 4 small matmuls each.
That core is 169 ops. The recurrent per-token loop (line 424) is only used for
`seq_len == 1` decode. So the fallback is not pathologically slow; it is just ~170
tiny ops x 18 layers = ~3,000 launches with fp32 up-casts of q/k/v every layer.

MPS cost model: PyTorch MPS pays ~20 to 40 us per op for encoder/launch plus Python
dispatch; 12,674 ops x 30 us = **~380 ms**, which is the 421 ms. The GPU is mostly idle.
Rough stage split by op share: DeltaNet 18 x 276 x 2 (autocast) ~ 9,900 ops ~ 300 ms;
attention 6 x 142 x 2 ~ 1,700 ops ~ 50 ms; ViT ~1,500 ops ~ 45 ms; head + processor
~ 15 ms (fp32 CPU head is 444 ops, no autocast). Evidence that it is launch-bound and
not FLOP-bound: CPU fp32 at 5.6 s has `aten::mm` + `bmm` at 53% of self time and
`_slow_conv2d_forward` x110,592 (the patch-embed conv3d unrolled per patch) at 26%.

**Why the 80 ms estimate was wrong:** it assumed 4.5 TFLOP/s achieved on
2 x 0.5B x 302 = 301 GFLOP of text matmuls + 47 GFLOP ViT (M=302 rows; weight bytes
1.18 GB fp16 = 4.3 ms at 273 GB/s, so not bandwidth-bound either). PyTorch MPS never
gets near that: it runs one small kernel per op with no fusion.

**MLX (measured, herb):** end to end **86.7 ms p50 / 93.5 p99**, backbone 82.1, ViT
16.8, text tower 63.2, 32/32 vs true fp32. Why 5x better: mlx-vlm's DeltaNet is one
fused `mx.fast.metal_kernel` per layer (`gated_delta.py` line 17; packed Dk=128 variant
line 122 for prefill), one thread-group per (head, value row), the whole T loop inside
the kernel. Lazy graph, no autocast casts, fp16 weights. Why not 3x better still:
text tower scaling (`scaling.py`) is **9.1 ms fixed + 0.19 ms/token**, so at 302 tokens
57 ms is per-token. 0.19 ms/token x 302 = 57 ms for 301 GFLOP = **5.3 TFLOP/s**
on the matmul share, but the DeltaNet kernel serializes over T inside one SIMD group
per row (T=302 sequential steps x 18 layers), so the per-token slope is mostly the
scan, not the matmuls. The flat ~18 ms from 32 to 64 tokens says the kernel floor
(24 layers x ~8 launches x ~40 us + scan latency) dominates below 64 tokens.

## 2. Floor, raw and streaming

Raw, one frame, 302 tokens, fp16 MLX: **82 ms forward** is the honest number.
Lower bound at the measured slope with no architecture change: ViT 17 + text
9.1 + 0.19 x 302 = 67 ms -> 84 ms. Matches.

Question-first streaming (measured, fa62c27): 184 cached + **118 per-frame tokens**
(66 visual + 25 state text + 27 readouts), text-from-cache **29.7 ms**, ViT 17.3,
end to end **54.7 ms p50**. Fit check: 9.1 + 0.19 x 118 = 31.5 ms. Good.

Token diet from here (fit predictions):
- Drop "A frame from the racing game..." state text (25 -> ~4 tokens) and shrink readouts
  to one `Decision:` token per question (27 -> ~8): 118 -> ~78 tokens -> 9.1 + 14.8 =
  **~24 ms** text, ~41 ms e2e.
- min_pixels to 128 patches (e.g. 256x128 resample, 8x16 grid, 32 visual tokens):
  ViT 47 -> 22 GFLOP, ~17 -> ~9 ms; text ~44 tokens -> ~17.5 ms (flat region).
  **~27 ms e2e.** Quality must be re-gated; the car is ~8% of frame width.
- Below that is the 9 to 18 ms launch floor: 24 layers x (norm, 4 projections, conv,
  scan, gated norm, out, MLP) ~ 10 kernels = 240 launches. Fusing DeltaNet projection +
  conv + scan into one kernel per layer and norm into matmul (sys1rust's lever) could
  take the fixed term to ~5 ms. Expected **~15 to 18 ms e2e floor** for Valen's shape.

**DeltaNet recurrent state vs KV cache.** Measured prefix cache: 20.8 MiB total, of
which DeltaNet state is 18 x 16 heads x 128 x 128 x 4 B = 18.9 MB fp32, conv windows
0.66 MB, KV for 184 tokens 6 x 184 x 2 x 2 x 256 x 2 B = 2.3 MB. So DeltaNet state is
8x *larger* than the KV it replaces at this prefix length, but it is constant in
prefix length, and per-frame cost does not grow with the question count: the scan reads
a fixed 128x128 matrix per head regardless of how many question tokens came before.
Attention layers still read the full KV (184 + 118 keys), but that is 6 layers x
tiny. Net: streaming is cheaper than full prefill by exactly the cached tokens
(184 x 0.19 = 35 ms, measured 63 -> 30), not cheaper than KV caching per se. The
advantage of DeltaNet is that restore is a 21 MB memcpy with no position-dependent
re-indexing, which is why the cache gate hit 32/32 with max delta 0.0007.

**Does reordering break Valen's distribution?** Yes, badly, and the audit says why:
in the trained order, swapping only the candidate role vectors between frames recovers
99.2% of the answer change; context 0.7%, decision 0.1%. Question-first puts candidates
before the image, so under causal attention plus recurrence their features change by
0.000 across frames. The backbone still carries frame information into the context
(0.096) and decision (0.063) roles; the *head* ignores it.

**Head-only retrain was tried and is not enough** (`distill_head.py`, 400 frames, 80
held out, backbone frozen): original head 40.6%, retrained Mixer 50.0%, Mixer + frame x
candidate bilinear 63.8% (peak 65.9%). on_road MAE 0.087 against a 0.02 gate. The
ridge probe explains it: question-first context/decision rows predict the teacher at
R^2 0.62 to 0.81, instruction/candidate rows at R^2 < 0. ~70% of the signal survives
the reorder in a frozen backbone, and no head recovers the rest.

**Option (b), candidates after the image, is a non-option.** Trained layout is 22
static + 66 image + 3 + 211 question/candidate/readout tokens. Keeping candidates after
the image leaves only the 22-token prefix cacheable: 22 x 0.19 = ~4 ms, 82 instead of
86.7 ms. **Prompt trimming is not a lever either (measured):** 302 -> 263 tokens saved
1.5 ms (85.6 -> 84.1) and dropped agreement to 17/32 (risk 0.46 -> 0.92). Every
exact-accuracy path below 86.7 ms needs training.

**Recommendation: option (a), LoRA the backbone in question-first order.** Reasons:
(1) Valen must be fine-tuned anyway (zero-shot steer is "straight" on 7/8 frames,
RESULTS.md), so the fine-tune is not extra cost, only its token order is a choice;
(2) the correctness gate for a *model* change is sim-label accuracy against the same
LoRA trained state-first, not 99% agreement with the zero-shot reference, which is an
*engine* gate; (3) the fine-tuned question-first checkpoint becomes the new PyTorch
reference for the Rust port's 99% gate. Cost: ~5k sim frames, LoRA on 24 layers,
hours on this GPU (one job at a time). P(question-first LoRA within 2 points of
state-first LoRA on held-out sim labels) ~0.7; the information is present (R^2 0.7
before any adaptation) and DeltaNet layers can learn to write candidate-relevant state
before the image arrives. Most likely failure: noul/score calibration drifts because
the recurrent state must now carry the question set through the image tokens.
Interim while it trains: state-first at 86.7 ms, which the sim-planned feedforward
can tolerate (DESIGN.md: the model only corrects).

## 3. Box head

Candidates compared, for a ~20 px car in a 256x144 frame, ~66 visual tokens:

| head | cost per frame | precision ceiling | confidence | verdict |
| --- | --- | --- | --- | --- |
| edge distributions over rows/cols (DESIGN.md) | 4 softmaxes over <= 22 bins, ~0 ms | sub-bin via expectation | entropy, calibrated by training | **yes** |
| Jev-style regression (4 floats from decision vector) | ~0 ms | unbounded but mean-seeking under multimodality | needs a separate head | fallback only |
| Qwen-VL coordinate tokens | ~15 decode steps x 9 ms floor = **~135 ms** | 1/1000 grid | none | no |
| DFL (YOLOv8) | 4 x 16-bin softmax on a dense feature map | sub-bin | implicit | same idea as row 1 on a grid, keep its loss |

DFL *is* the edge-distribution head with the bins tied to the feature grid, so the
DESIGN.md head is sound. Two corrections:

1. **2x2 merge is too coarse.** Processor output is a 12x22 patch grid at 352x192
   (merged 6x11, 66 tokens). One merged token = 32 x 32 resampled px = 23 x 24 source
   px. The car is ~20 px wide, so the right-edge and left-edge distributions would put
   most mass in the same bin; expectation gives the centre but the width is lost.
   At the pre-merge grid, bins are 16 px resampled = ~12 px source, with DFL-style
   expectation giving ~3 px. **Tap the ViT block-12 output before the merger**
   (12x22x768 in mlx-vlm `VisionModel`, trivially exposed), pool rows and columns to
   get 12 + 22 candidate vectors, score each against the decision vector exactly like
   choice candidates. Cost: 34 x 768 dot products, nothing.
2. The language-tower decision vector sees the image only through 66 merged tokens, so
   the *query* is coarse but the *keys* are fine; that is fine for one dominant object.
   If the hitbox needs < 3 px, go to the student (item 5), which runs the ViT at native
   256x144 = 16x9 patches of 16 px with no resample.

Metric stays mean IoU and edge px vs sim ground truth; add a calibration plot
(entropy vs error) because that is the System One contract.

## 4. Shortest path to a number this week

Python mlx-vlm first, Rust only after the shape is fixed. Reasons: every lever that
moves the number this week (head retrain, token diet, ViT tap) is a Python change on
a harness that already passes the gate and times stages; sys1rust's own history shows
Rust-vs-Python-MLX was 1.12 to 1.25x once both call the same kernels. Rust pays off
for no-Python-in-loop, p99 (sys1rust: no request > 2x median vs 9% in Python), and
fused kernels.

| # | experiment | expected | measured by |
| --- | --- | --- | --- |
| 1 | LoRA fine-tune Valen on ~5k sim-labeled frames, two runs: state-first and question-first order, same data and steps | question-first within 2 points of state-first on held-out sim labels; e2e stays 54.7 ms | `streaming.py` sim-label accuracy; cache gate vs uncached (already 32/32) |
| 2 | Token diet on the question-first checkpoint: terse questions, 1 readout token each, min_pixels 128 patches | text 30 -> ~18 ms, ViT 17 -> ~9 ms, **e2e ~27 to 30 ms**; re-gate accuracy | `streaming.py --bench`, same 40-sample protocol |
| 3 | Box head on pre-merge ViT grid, trained on sim hitboxes | IoU > 0.7, edge err < 5 px source, +0 ms | IoU/edge px vs sim Verify boxes |

Then Rust: port the measured graph to `engine/` (valen-mlx backend), gate 99%, fuse
DeltaNet (projection + conv + scan) as the first custom kernel. Target 15 to 18 ms.

## 5. Should the per-frame model be Valen-0.8B?

No, not at 120 Hz. Arithmetic: the 24-layer tower has a measured 9.1 ms fixed cost
before any token; add ViT (17 ms, 47 GFLOP at 264 patches) and the fp32 CPU head and
the best Valen can do on this GPU is ~15 to 18 ms with custom kernels, i.e. 55 to 65 Hz.

Student: ViT-S/16 (22M params, 384 wide, 12 blocks) at native 256x144 = 144 patches
= 6.5 GFLOP; ViT-Ti 1.7 GFLOP. With one fused-ish MLX graph of 12 blocks (~100
launches) expect **3 to 5 ms** for S, ~2 ms for Ti, plus a 4-edge DFL head and 2
choice heads reading the CLS token. Data: sim gives exact box + plan action per frame
for free; Valen (fine-tuned, state-first, 82 ms) supplies soft labels for the
judgement questions (off_line, risk) offline. That is DESIGN.md lever 4 made the
default rather than the fallback.

| model | per frame | role |
| --- | --- | --- |
| Valen state-first, fp16 MLX | 82 ms (measured) | teacher, offline labels, System 2 queries |
| Valen question-first LoRA, cache path as measured | 55 ms (measured mechanics) | first driver, ~18 Hz |
| same + token diet | ~27 to 37 ms (fit) | L1 cerebellum at 30 Hz |
| Valen in Rust with fused DeltaNet | ~15 to 18 ms (estimate) | ceiling for this backbone |
| ViT-S student + DFL box + choice heads | ~3 to 5 ms (estimate) | per-frame System One at 120 Hz, confidence-gated |

The plan already carries 1 ms timing; the model corrects. A 5 ms student at 120 Hz
corrects 4x more often than Valen at 30 Hz, and its box is at native resolution.

## 6. Top 3 risks

| risk | P(bites) | cheapest settling experiment |
| --- | --- | --- |
| Question-first LoRA lands more than 2 points below state-first LoRA on sim labels (recurrent state cannot carry the question set through the image well enough) | 0.3 | Experiment 1: paired LoRA runs, same data and steps, held-out sim-label accuracy. A few GPU hours. If it fails, stay state-first at 86.7 ms and move all speed work to the student. |
| Token diet to 128 patches loses the car (20 px -> ~10 px at the resample) and steer/box accuracy drops | 0.4 | Experiment 2 with the gate on sim labels at 264 vs 128 patches; if it fails, keep 264 and take ViT to the student instead. |
| The student generalizes to sim-labeled frames but not to real captures (render, HUD, compression differences) | 0.5 | Train on sim frames, test on 200 real `eye/` captures with hindsight sim labels (DESIGN.md L2). Shows up as IoU gap in one eval. |

Not in the top 3 but noted: MLX p99 spikes on varying shapes (sys1rust REPORT item 2).
Streaming fixes shapes per frame, so spikes should not appear; measured p99 54.7 -> 74.4
ms on 40 samples says watch it. Mitigation is fixed-shape padding, already in sys1rust.
