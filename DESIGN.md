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
Shared-state execution encodes the state once for all questions. It's the teacher and the
reference baseline (86.7 ms MLX fp16, `bench/infer/`), not the base. See "Base model" below.

## Base model: Laya-V (decision, 2026-10-22)

Two independent reviews (`research/RETHINK-fable.md`, Fable 5.1; `research/RETHINK-sol.md`,
GPT 6.1 Sol) agreed on the core, and it replaces the 2026-10-21 plan (Laya-large + fresh
projector):

- **Keep Laya's recipe, not Laya's weights.** Keep the typed head read at `[MASK]` option
  markers, the proper-scoring loss, temperature calibration, Apache-2.0 and `/v1/systemone`.
  Drop the text-only ModernBERT-large backbone. The laya-vision fork tried "SigLIP projector
  into Laya's encoder" and the model ignored the image in 5 runs (shuffled images scored the same).
- **Backbone that already sees.** ModernVBERT (ModernBERT-150M already aligned to SigLIP2,
  MIT). It's 2.9x cheaper per token than Laya-large on this M4 Pro: 7.1 vs 20.3 ms at 128
  tokens, measured on a shape probe with random weights.
  Vision: SigLIP2-B/16 at native 256x144 (6.7 ms measured) shuffled to 36 tokens.
  Bake-off alternative (Sol): MobileCLIP2-S0 (11.4M params) with late fusion into the head.
  The cheapest test below picks the winner.
- **Questions packed in one pass** with a block mask (each question sees image + state, not
  other questions), trained that way from day one. This is sys1rust's biggest win made native.
- **"More" than Jev** (all checked against docs.typesafe.ai):
  - **Box with presence:** edge distributions over the ViT grid plus a fused "present" Noul.
    Jev's Box always returns coordinates even when the target is absent.
  - **Metric Score:** levels trained on exact sim quantities, so the expectation is a
    calibrated measurement. Jev's docs say not to use Score for magnitude.
  - **Small frames:** 256x144 accepted. Jev rejects images under 280 px per side.
  - **Two-frame state** at 36 tokens per frame.
  - **Local, open, fine-tunable.** Jev is hosted only, with no fine-tuning.
- **Latency target:** about 18 ms p50 in Python MLX and about 14 ms in Rust with sys1rust's
  levers (compiled fixed shapes padded to 16, dense attention, unpad, head pruned to marker
  rows, fused qkv/RoPE). NAX gemms and int8 don't carry over (M5-only, and int8 failed the
  99% agreement gate).
  Jev's hosted round trip is reported at roughly 150 to 200 ms (third party, no official
  number). Claim no speedup until Jev is measured side by side.
- **Cascade:** the ViT-Ti student (`track/student`, IoU 0.77 held out, 2.2 to 4.3 ms median
  measured) answers the per-frame driving Scores at 120 Hz. Laya-V runs about 60 Hz as
  arbiter and does Box and general questions.
- **Valen** stays only as the measured baseline (86.7 ms) and is not a teacher: the sim gives
  exact labels and permissive VQA sets give general ones.

Plan flaw both reviews caught: the car's own Box carries almost no pose in a chase cam.
Its centre sd is 12 to 15 px over 4,635 frames, and a constant box already scores IoU 0.647.
So the controller is fed these per-frame questions instead:
- Score: lateral offset and heading error
- Noul: airborne and on_road
- Box: next gate (it moves on screen and encodes distance)
- Choice: phase

P(Laya-V p50 <= 20 ms at 5 questions on this Mac) 0.8. P(it measures lateral offset to
0.2 m RMS) ~0.65. Most likely way it's wrong: a document-trained encoder at 36 tokens
can't resolve fine road geometry, and then the student drives while Laya-V does Box and
general questions.

## Polytrack target: 21.0 s (owner, 2026-10-22)

The lap is 1,293 m, so 21.0 s needs a 221 km/h average; the current lap averages 176.
Fixing both flights and the cp2 to cp3 segment gets about 22.9 s (estimate). The last 1.9 s
needs a shorter path, so this is a route search, not refinement.

Planner v2:
- Hierarchical CEM over 6 s horizons from snapshots along the incumbent.
- A learned V(state) as terminal value.
- Go-Explore cells (checkpoint, speed, lateral) that restart jump approaches instead of refining them.
- Then the p05 finish time under jitter.

P(sim < 21.0 s) 0.35 to 0.45 (reviews' range); P(< 22.123 s) 0.65. Cheapest test first:
from the cp1 snapshot, a 6 s-horizon search on time-to-cp2 (currently 8.54 s). If it doesn't
drop under 7.0 s, the jump isn't the lever and the path is.

## Scope (2026-10-22)

In: (1) the model, Laya-V (above), MLX then Rust in sys1rust's crates with the same
`/v1/systemone` API; (2) the harness, `eye/` capture and keys; (3) the Polytrack demo, a
self-taught lap under 21.0 s driven live from pixels. `sim/`, `data/`, `learn/`, `ctrl/`
exist only to feed (3) and to label frames for (1).

Cut:
- Valen as a per-frame driver and the question-first LoRA.
- Laya-large + fresh projector.
- The car Box driving the controller.
- int8 kernels.
- Rust before the model's shape is final.
- Broad runtime bake-offs.
- World models, backbone RL, online weight updates.
- Trailer and agent-infrastructure work.

Frozen: `opt/` (WR-derived, excluded), `records/` until a sub-22 s self-taught sim lap.
Battery: one heavy job at a time, on AC only.

Milestones, each ending in a number:
1. **Cheapest tests** (about 1 day):
   - (a) Frozen ModernVBERT + MobileCLIP2-S0 + student, each with a lateral/heading Score
     head on the 4,635 frames. Gate: RMS <= 0.2 m on winter1, plus p50/p99 ms.
   - (b) The cp1 snapshot jump search. Gate: time-to-cp2 under 7.0 s.
2. **Laya-V v0** (3 to 5 days): packed block-masked questions, DFL Box with presence, sim
   question set. Gates: lateral <= 0.2 m, next-gate IoU >= 0.7, p50 <= 20 ms.
   **Planner v2:** gate <= 22.0 s, >= 0.8 finish rate at +/-2 ms jitter; stretch < 21.0 s.
3. **Closed loop from pixels**, headless or Desktop 2. Gates: HUD time over 20 runs,
   >= 10/20 finishes, motion-to-key p50/p99. Then the Rust port (99% gate) and general VQA
   for Jev parity.

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
   the Box type, `laya-mlx` gains the ViT tower and projector (base model above). Gate:
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

**Data from the game's own ghosts: rejected (owner, 2026-10-20 23:10).** Copying the #1
ghost got a human run with 1 to 2 ms nudges, not a driver. Reflex-1 learns from the track,
the physics, and its own runs only (LOOP.md "Isolation rule", `loop/isolation.py`). The
WR is a number to beat, never an input.

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
