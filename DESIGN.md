# reflex: a visual System One engine that plays Polytrack

Goal: a local, minimum-latency **visual System One model runtime** in the sys1rust
lineage (Rust, MLX, custom Metal kernels, same `/v1/systemone` API), with Jev 1.14-style
vision: images in `state`, typed answers (choice, score, noul) and **Box** (hitbox)
answers with calibrated confidence. The demo and benchmark: it watches a live video
stream of the desktop, presses real keys, and drives Polytrack (kodub.com) faster than
any human. Pixels in, keys out, no hooks into the game at runtime.

Focus, in order: (1) inference engineering, (2) the harness interface to the desktop,
the game and the controls, (3) self-correction, so it improves every run.

## What exists (2026-10-20)

| model | modality | open | box | latency note |
| --- | --- | --- | --- | --- |
| Jev 1.14 (TypeSafe) | text + images | no, hosted API | yes (preview) | network round trip |
| Laya (Convai) | text | yes, ModernBERT-large | no | 33 ms stock, 13 to 15 ms in sys1rust on M5 |
| Valen 0.8B/2B/4B | text, image, video | yes, Apache-2.0 | no | Qwen3.5 + ViT, PyTorch reference, ONNX/GGUF exports |
| Visual Jev (paper) | image | 4B LoRA | no | 5.7 ms/question amortized at 32 questions on an RTX 5090 (throughput, not single-request latency) |
| OpenAI Decisions, Liquid d1, Solar Decide | text (OpenAI: + image) | no, hosted | no | network round trip |

Valen-0.8B: Qwen3.5-0.8B text tower (24 layers, 3 of every 4 are Gated DeltaNet linear
attention, hidden 1024), 12-layer ViT (768 wide, patch 16, 2x2 merge), and a 2-layer
MLP-Mixer decision head over 4 role vectors (context, instruction, candidate, decision).
Shared-state execution encodes the state once for all questions. It's the starting checkpoint.

Gap we fill: no open model has Box, and nothing runs a visual System One model fast on a Mac.

## Polytrack facts (from the 0.6.3 bundle)

- Physics is WASM in a Web Worker at a fixed **1 ms tick**, 5 boolean controls
  (up, right, down, left, reset), **deterministic** (the game ships `testDeterminism`
  and a frame-exact `Verify` replay). Key events are stamped to 1 ms.
- Render is `requestAnimationFrame`; this display is 120 Hz (8.3 ms/frame).
- The ghost feature serves full input recordings over a read-only GET. Summer 1 world
  record: **22.123 s** (22123 frames, 2026-10-09), top 5 within 147 ms, 7.0M entries.
- Consequence: the sim gives exact ground truth for every frame (car pose, hitbox, the
  action the plan took). That's a free, unlimited labeled dataset for a visual model.

## The engine (`engine/`, fork of sys1rust)

Method is sys1rust's: **bake-off first, then Rust, then levers, each one measured.**

1. **Bake-off.** Valen-0.8B on this M4 Pro: PyTorch MPS (reference answers), the ONNX
   export (CoreML EP), the GGUF export, mlx-vlm. Same frames, same questions. Report
   p50/p99 per request and agreement with the reference. That's the number to beat.
2. **Rust port.** sys1rust's crate layout: `laya-core` gains Jev's `assets` format and
   the Box type, a new `valen-mlx` backend runs ViT + Qwen3.5 + Mixer head. Gate:
   >= 99% agreement with the reference on a fixed workload, like sys1rust's.
3. **Levers**, one at a time, kept only if p50 drops and agreement holds:

| # | lever | why | needs |
| --- | --- | --- | --- |
| 1 | small frames: a 256x144 crop. Valen's processor upscales to >= 65,536 px (min_pixels), so ~256 patches, ~64 visual tokens after the 2x2 merge, unless min_pixels is lowered and quality re-checked | prefill is linear in tokens | config + agreement check |
| 2 | **question-first streaming**: questions go before the image, so their KV and DeltaNet state are computed once per session and reused every frame | in a control loop only the frame changes | fine-tune (Valen trained state-first) |
| 3 | int8 weights, sys1rust-style tuned matmuls, head only at readout rows | weight bytes per forward: ~0.55B text (the 254M-param embedding table is only gathered) + ~85M ViT = ~1.3 GB fp16, >= 4.7 ms at 273 GB/s | kernels |
| 4 | confidence cascade: distilled tiny student answers when its `answer_confidence` clears a threshold, else the full model | most frames are easy; System One confidence exists for exactly this routing | likely needed, see estimate below |

Estimate (arithmetic, to be replaced by step 1's measurement and `research/ENGINE.md`):
the M4 Pro 20-core GPU is ~9 TFLOPS fp32/fp16, so ~4 to 5 TFLOPS achieved. One frame with
4 questions in shared state is ~64 visual + ~250 text tokens, ~2 x 0.55B x 314 + ViT
= ~390 GFLOP, so **~80 ms unoptimized, compute-bound**. Question-first streaming cuts
per-frame tokens to ~64 visual + ~10 readout, ~130 GFLOP, **~25 to 30 ms**. So Valen-0.8B
lands near 30 to 40 Hz at best, and 120 Hz needs lever 4 (a distilled small backbone) or
a split where the ViT runs on the ANE. The plan carries 1 ms timing either way, so the model
only needs to correct at 30+ Hz. Most likely way this is wrong: MLX's DeltaNet and ViT
kernels are far from peak, making it slower still.

### Box head (the architecture change)

Box as **distributions, not regression**: for each edge (left, right, top, bottom),
a probability over patch columns or rows. Each row/column of visual tokens is pooled
into a candidate vector and scored against the decision vector, the same way Valen
scores choice candidates. Expected value gives sub-patch precision, and entropy gives a
calibrated confidence, so Box keeps System One's typed answer + probability contract.
YOLOv8's distribution focal loss is the precedent: bins beat raw regression on box edges.
Trained on sim-labeled Polytrack frames (exact projected car hitbox per frame), then
general data if needed. Metric: mean IoU and edge error in pixels vs sim ground truth.

## Driving with System One questions

Every frame, one request with a shared state and fixed questions:

```
state:     { frame: <image>, plan: "up+right for 34 ms", last: "up" }
questions: steer    choice {left, straight, right}
           throttle choice {gas, coast, brake}
           car      box    "the player's car in `frame`"
           off_line noul   "is the car off the planned line?"
```

The plan (from the sim optimizer, seeded with the WR) supplies 1 ms-precise timing as a
feedforward prior. The model's job is correction, the way Jev browser agents pick an action
and a target with calibrated confidence. Low confidence falls back to the plan.

## Harness interface (desktop, game, controls)

```
eye/ (Swift): ScreenCaptureKit stream of the game rect, 256x144, 120 Hz, IOSurface
   -> engine/ (Rust): /v1/systemone (Jev-compatible HTTP) + a streaming mode
      (questions once, then frames, answers back)
   -> eye/ hand: CGEvent key down/up, postToPid when the browser accepts background keys
```

- Frames go through HTTP first (sys1rust's HTTP adds 0.3 ms).
  `ponytail:` HTTP + raw BGRA body; zero-copy IOSurface handoff when HTTP shows in the profile.
- Safety: keys only reach the target app, all keys are released on any exit, and there's a
  kill switch.
- System 2 (Claude Code) reaches the same engine and controls as tools, for menus and
  recovery. Never in the per-frame loop.

Latency budget: capture <= 8.3 ms, engine ~25 to 40 ms for Valen-0.8B with streaming
(< 8 ms only with a distilled student), key to game < 2 ms, game render 1 to 2 frames.
Motion-to-photon target < 30 ms p50 without the model. Even at 60 ms with the model, the
loop reacts 3x faster than a human (~200 ms).

## Self-correction: four loops at four timescales

Cerebellum design: a feedforward motor program plus feedback error learning (Kawato 1987).

| loop | timescale | learns | signal |
| --- | --- | --- | --- |
| L0 reflex | every frame | nothing, acts | System One answers vs plan |
| L1 cerebellum | within a run | latency estimate, correction gains | predicted vs observed car box |
| L2 practice | between runs | model weights (LoRA + heads), robust plan | hindsight labels: replay sent keys in the sim, so every captured frame gets an exact label |
| L3 meta | between sessions | code and architecture | hackkit's paired held-out gate on finish times and engine p50 |

The plan is scored under injected timing jitter (sampled from L1), so it's robust, not
just fast. L3 scores with code (sim Verify, real HUD times, bench numbers), never a model.

## Generalist or specialist (decision, 2026-10-20)

**General perception, specialist control, N+1 practice.** The model is a general visual
System One (image + typed questions -> choice/score/noul/box). Nothing in it is Polytrack
code: the game only shows up as questions ("where is my car", "where is the ghost car",
"steer?") and as training data. Millisecond control comes from a per-task planner +
controller, the way a human has general vision and track-specific practice.

- Specialist-only (hand-coded pixel tracker + controller): fastest to a WR video, P ~0.6,
  generalizes to nothing.
- General-only (beat the WR as a side effect): P ~0.05. A 22 s run with zero timing
  margin isn't something general skill finds by accident.
- Hybrid (this): P ~0.35 to 0.5 live WR, and the perception model transfers.

**Data from the game's own ghosts.** Every leaderboard entry has a replayable recording
(the "ghost car"). Each one replays byte-exact in the sim, so the top-K ghosts on all 14
official tracks give unlimited exactly-labeled expert frames. The live ghost on screen is
also a general skill: "track the target object" (Box on the ghost, Box on my car, close the gap).

**N+1 loop.** Every run (1) records frames + keys, (2) re-simulates its own keys for exact
labels, (3) fine-tunes perception and updates controller gains, (4) seeds the optimizer.
Kept only if it beats the champion on held-out tracks (hackkit gate). Generalization is
measured on tracks it never trained on.

## Rules of engagement

Runtime sees only pixels and sends only OS key events to a normal browser window.
Practice happens in a headless copy of the public physics module. Kodub's files live in
gitignored `vendor/`. Bot times are **never submitted to the public leaderboard**.

## Build order (each slice ends in a number)

1. `eye/`: capture, keys, motion-to-photon. **ms.** (in progress)
2. `sim/`: headless physics, WR replay finishes at frame 22123. **parity + ticks/s.** (in progress)
3. Bake-off: Valen-0.8B on M4 Pro across runtimes. **p50/p99 ms, agreement.**
4. Data engine: real-game frames + sim labels (box, keys) from WR replays. **frames/hour.**
5. `engine/`: Rust port + Box head + levers. **ms, IoU, agreement.**
6. Closed loop on Summer 1. **finish ms, variance over 20 runs vs 22.123 s.**
7. L2 + L3 on. **finish ms per session on held-out tracks.**

P(sim line under 22.123 s) ~ 0.75. P(live from pixels under 22.123 s) ~ 0.35 to 0.5.
Most likely failure: live timing jitter on a line with almost no margin.

Skipped until a step needs it: the cascade student (lever 4), MCP tool surface,
multi-monitor, GSD `.planning/` (this file is the plan), RL from scratch.
