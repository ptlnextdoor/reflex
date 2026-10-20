# reflex: a System 1 that drives Polytrack from pixels

Goal: a local program that watches a live video stream of the desktop, presses real
keyboard keys, and drives Polytrack (kodub.com) faster than any human. No hooks into
the game process at runtime. Pixels in, keys out.

Focus, in order: (1) inference engineering of the eye-brain-hand loop, (2) the
interface between the harness and the desktop, the game and the controls,
(3) self-correction, so it gets better each run like hand-eye coordination.

## Facts that shape everything (read from the 0.6.3 bundle, 2026-10-20)

- Physics is a WASM module (`lib/polytrack_physics.wasm`) run in a Web Worker at a
  fixed **1 ms tick**. Controls are **5 booleans**: up, right, down, left, reset.
- The game ships a `testDeterminism` call and a `Verify` path that replays a
  recording frame by frame and checks the finish frame. **The sim is deterministic.**
- A keydown reaches the worker by `postMessage`. The worker stamps it at
  `frames + (performance.now() - lastTick)`, so **input timing resolves to 1 ms**.
- Render runs on `requestAnimationFrame`. This Mac's display is 120 Hz (8.3 ms/frame).
- Official tracks live at `tracks/official/<name>.track` (Summer 1, Winter, Desert...).
- Finish times are integer milliseconds (`numberOfFrames`).

Consequence: the perfect run is a list of key events with 1 ms timestamps. Humans lose
to that list in two places: the **line** (which inputs) and the **execution** (when).
A bot wins on both only if its execution survives timing jitter, because a 1 ms slip
early in a run changes the whole trajectory after it. That is why vision matters:
it closes the loop.

## System 1 vs System 2, and where Laya/Jev fit

Laya, Jev, AnyJev, Xor and the rest of the "System One" models are **text** decision
models (ModernBERT-class encoders). Best local number is sys1rust at 13 to 15 ms per
decision on an M5. That is wrong modality and roughly 2 frames too slow for a 120 Hz
control loop. Use them where they fit: menu navigation (pick track, press Play,
retry), Jev-style, off the hot path.

The driving System 1 is a **sub-millisecond pixel policy**: a tiny CNN on a downscaled
crop, run in the same process as capture and key injection. System 2 (Claude/GPT) never
touches the loop. It plans offline and edits the code between sessions.

## The loop (one process, no IPC, no Python in the hot path)

```
ScreenCaptureKit stream (game rect, scaled to 256x144 by WindowServer, 120 Hz, queueDepth 3)
  -> IOSurface (zero copy)
  -> eye: state estimator (car pose + hitboxes relative to the planned line)
  -> brain: action = plan[t + latency] (feedforward) + correction(state error) (feedback)
  -> hand: CGEventPost key down/up
```

Latency budget, target vs human (~200 ms visual reaction):

| stage | target | how we know |
| --- | --- | --- |
| capture: compositor to callback | <= 1 frame (8.3 ms) | SCK frame timestamp vs mach time |
| eye + brain inference | < 1 ms p50, < 2 ms p99 | in-process timer, 10k frames |
| hand: post to browser keydown | < 2 ms | in-page `performance.now()` probe page |
| game: keydown to rendered pixel | 1 to 2 frames | motion-to-photon bench |
| **motion-to-photon, total** | **< 30 ms p50** | inject key, count until pixels change |

P(under 30 ms p50 on this M4 Pro) ~ 0.8. Most likely way it's wrong: compositor and
browser frame pacing add a frame we can't remove from user space.

Inference bake-off (sys1rust style, same frames, same answers, pick by p99):
CPU (Accelerate/BNNS) vs ANE (Core ML) vs GPU (MPSGraph). For a ~100k-parameter net
the dispatch overhead decides it, not FLOPs, so measure, don't guess.

## Hand-eye coordination: four loops at four timescales

This is the self-correcting part. It's the cerebellum's design: a feedforward motor
program plus feedback error learning (Kawato 1987), which is why it fits.

| loop | timescale | learns | signal |
| --- | --- | --- | --- |
| L0 reflex | every frame | nothing, acts | pose error vs planned line |
| L1 cerebellum | within a run | latency estimate, correction gains | forward-model prediction error (predicted vs seen pose) |
| L2 practice | between runs | vision net, robust plan | hindsight labels from the sim |
| L3 meta | between sessions | code and architecture | paired held-out gate on finish times |

**L2 is the data engine.** After a real run we know every key we sent. Re-simulating
those keys in the headless sim gives the exact car state at every ms, so every real frame
gets labeled for free (pose, hitbox). Unknown per-event delivery offsets are solved by
fitting the re-sim to the observed poses (system identification). Each run makes the
eye more accurate and the latency model tighter, with no human labeling.

**The plan gets robust, not just fast.** The trajectory optimizer searches 1 ms input
sequences in the headless sim and scores each line by finish time *under injected timing
jitter* (sample offsets from the measured L1 distribution). A line that wins by 3 ms but
crashes on a 1 ms slip loses to one that is 2 ms slower and never crashes.

**L3 reuses hackkit's gate** (`hackkit/packages/gate`): an agent proposes a change, it
runs on held-out tracks, and it's kept only if a paired sign test plus bootstrap CI says
it beats the champion. The scorer is code (sim-verified and real finish times), never a
model grading itself.

## What "no special accommodations" means here

Runtime: the bot sees only pixels and sends only OS key events to a normal browser
window. Offline: it practices in a headless copy of the public physics module, the way
a human practices. The game's own files are fetched at build time into a gitignored
`vendor/`, never committed. Bot times stay local and are **not submitted to the public
leaderboard**, since that would bump real players.

## Build order (each step is a vertical slice with a number attached)

1. `eye/`: Swift: SCK capture of a rect, CGEventPost keys, motion-to-photon bench. **Number: ms.**
2. `sim/`: Node: load physics WASM, create car on Summer 1, step with controls, prove determinism. **Number: sim ticks/s.**
3. Open-loop replay: optimizer finds a line in sim, eye/ plays it on the real game. **Number: real vs sim finish ms.**
4. Eye v1: tiny pose net, hindsight labels from step 3 runs. **Number: pose error px, inference ms.**
5. Closed loop L0 + L1. **Number: real finish ms and its variance across 20 runs.**
6. L2 + L3 loops on. **Number: finish ms per session, held-out tracks.**

P(beating the top human time on one short official track by step 5) ~ 0.35 to 0.5.
Most likely failure: the optimizer can't find world-record-class lines (human records
here may already be near-TAS), not latency.

Skipped until a step needs it: Laya/Jev menu navigation (fixed key sequence works first),
GSD `.planning/` (this file is the plan), an RL policy (feedforward plan + feedback beats
learning from scratch here), multi-monitor support.
