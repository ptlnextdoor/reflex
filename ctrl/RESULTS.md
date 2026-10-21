# ctrl: closed-loop tracking controller (track E)

Closed-loop controller the live bot will run: observe the car pose with latency, follow
Reflex-1's own reference line, correct with keys. Everything here is scored by the byte-exact
sim (`ctrl/sim.mjs`, parity-checked against `sim/sim.js` by `node ctrl/check.mjs`). No human
run touches this directory: `python3 loop/isolation.py ctrl` passes (LOOP.md isolation rule).

## Files

| file | what |
| --- | --- |
| `sim.mjs` | per-ms stepper (`Car.step(mask)`), whole-heap snapshot/restore (23 MB, 0.6 ms) |
| `check.mjs` | parity vs `sim.run` on synthetic controls, snapshot/restore, throughput |
| `reference.mjs` | `loadLine()` reads `learn/best.json` ({controls} or {toggles}); `buildReference()` runs it once, keeps per-ms pose, speed, keys, steering state, contacts |
| `segment.mjs` | temporary own reference: hand-written steer-to-waypoint rule, start pose to checkpoint 1 in 4752 ms (`lines/segment.json`); `lines/segment_fast.json` is the same rule with deadband 0.08, lookahead 40 (4030 ms) |
| `controller.mjs` | v1 tracking controller (below) |
| `episode.mjs` | closed loop: delayed noisy observation, 8 ms control step, per-toggle key jitter |
| `eval.mjs` | parallel grid (worker_threads), 740 episodes in 4.5 s on 13 threads |
| `tune.mjs` | gain search, kept only on held-out seeds, writes `best_params.json` |
| `spec.mjs` | perception spec sweep (latency, noise, bias, heading noise) |

## Controller v1

Every 8 ms (one 120 Hz frame) with the state observed `L` ms ago:

1. Progress: nearest reference point within a window around the predicted index.
2. Errors in the reference frame at that index: lateral `eY` (left positive), heading `ePsi`.
3. Latency compensation: `eY += v * sin(ePsi) * L`, progress advanced by `L * v/vRef`.
4. Steering target in normalized units: `s = s_ref(t) - Ky*eY - Kpsi*ePsi`. The steering
   actuator was identified from the sim: the normalized steer slews `0.01/ms` toward the key
   target (+1 left, -1 right, 0 none), max angle `~min(0.41, 114/v_kmh^1.5)`. The controller
   keeps a belief of the current steer and, per ms, presses the key whose one-ms slew lands
   closest to `s` (PWM at 1 ms resolution). Throttle/brake are the reference's keys.
5. Airborne (from the reference's contact flags): L/R have no effect, so the line's keys are
   held unchanged.

With zero gains and zero noise the loop reproduces the reference tick-for-tick at any latency
(feedforward timing is exact); that is the correctness check for the harness.

Rejected on measurement: lateral-velocity damping `Kvy` (worse at every value tried),
speed-scaled gains (`speedExp`, no gain), deadbands above 0.01 (collapse the finish rate).

## Iteration log (reference: `ctrl/lines/segment.json`, own line, goal = checkpoint 1 at 4752 ms)

Headline cell = latency 60 ms, key jitter +/-2 ms, position noise 0.05 m (Gaussian per axis),
20 seeds. "p50" is the median time to reach the goal; negative delta = faster than the line.

| # | change | headline finish | headline p50 (delta vs ref) | full grid | command |
| --- | --- | --- | --- | --- | --- |
| 1 | v1 defaults Ky 0.6, Kpsi 8 | 20/20 | 4715 (-37) | 33/36 cells at 20/20, worst 18/20 (L60 j5 n0.2) | `node ctrl/eval.mjs --ref ctrl/lines/segment.json --seeds 20` |
| 2 | tune.mjs, 30 iters on 8 stress cells, held-out seeds: Ky 0.278, Kpsi 6.19, deadband 0.0033 | 20/20 | 4689 (-63) | 35/36 cells at 20/20, worst 19/20 (L90 j5 n0.05) | `node ctrl/tune.mjs --ref ctrl/lines/segment.json --cells stress --iters 30 --seeds 10` then `node ctrl/eval.mjs --ref ctrl/lines/segment.json --seeds 20 --params ctrl/best_params.json --out ctrl/results/grid_segment.json` |

Full grid for row 2 (finish/20, p50 ms):

| latency \ jitter, noise | 0, 0 | 0, 0.05 | 0, 0.2 | 2, 0 | 2, 0.05 | 2, 0.2 | 5, 0 | 5, 0.05 | 5, 0.2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 8 ms | 20, 4752 | 20, 4703 | 20, 4755 | 20, 4680 | 20, 4677 | 20, 4705 | 20, 4681 | 20, 4693 | 20, 4672 |
| 30 ms | 20, 4752 | 20, 4688 | 20, 4735 | 20, 4676 | 20, 4719 | 20, 4695 | 20, 4709 | 20, 4695 | 20, 4718 |
| 60 ms | 20, 4752 | 20, 4689 | 20, 4712 | 20, 4717 | 20, 4689 | 20, 4704 | 20, 4730 | 20, 4714 | 20, 4697 |
| 90 ms | 20, 4752 | 20, 4713 | 20, 4725 | 20, 4718 | 20, 4694 | 20, 4774 | 20, 4673 | 19, 4712 | 20, 4780 |

Why the loop is faster than its own open-loop line: the segment line is a cautious bang-bang
steer-to-waypoint rule that scrubs speed in the bend; the tracking controller's heading term
smooths the same path and carries more speed. This is a property of the weak temporary
reference, not something to expect against a planner-optimized line. On a faster own segment
(same rule, wider deadband and lookahead, cp1 at 4030 ms, `ctrl/lines/segment_fast.json`)
the loop tracks it to within +0 to +3 ms p50 in every cell, 36/36 cells at 20/20.

## Perception spec (what the vision model must deliver)

Sweep of one axis at a time, others at the headline values, 20 seeds, >= 90% finish kept
(`node ctrl/spec.mjs --ref ctrl/lines/segment.json --seeds 20 --params ctrl/best_params.json --out ctrl/results/spec_segment.json`):

| axis | 20/20 up to | >= 90% up to | collapses by |
| --- | --- | --- | --- |
| observation latency (ms) | 120 | 150 | 200 (13/20), 300 (3/20) |
| white position noise (m, 1 sigma per axis) | 1.2 | > 1.2 | not reached |
| constant position bias (m) | 0.25 | 1.0 (18/20) | 2.0 (11/20) |
| heading noise (rad, 1 sigma) | 0.2 | 0.2 | 0.3 (14/20) |
| latency estimate error (ms, true minus assumed) | -40 to +20 | +80 (19/20) | not reached |
| per-frame latency jitter (ms, uniform +/-) | 32 | 32 | not reached (p50 +39 ms at 32) |

Latency error is asymmetric: compensating too much (true latency shorter than assumed) is
free, compensating too little costs time (p50 +242 ms at +80). So the live loop should assume
its p90 latency, not its p50.

Heading from motion only (no orientation output from the model; heading = chord of the last
~1.5 m of noisy positions): headline cell 19/20 at noise 0.05, 20/20 at 0.1, 19/20 at 0.2.
So a box-only model is enough on this segment if the controller low-passes the heading error
(`psiAlpha` 0.3 to 0.5 gives 20/20 across noise 0.05 to 0.2).

Spec for the model team, from this segment (the full lap will tighten it):
**end-to-end latency <= 120 ms, position error <= 0.2 m RMS with bias <= 0.25 m, heading
either from the model within 0.1 rad or derived from motion.** White noise is nearly free
because the controller integrates it; bias and latency are what hurt.

## What limits it

On this segment nothing in the grid fails. The limits are latency (above 150 ms the heading
feedback overshoots: at 200 ms, 300 kph covers 17 m per latency) and bias (a constant 1 m
offset is tracked as a real lateral error and the car is steered 1 m off the line into the
curb at speed). Steering resolution is not a limit: 0.01 normalized units per ms is far below
the 0.05 to 0.1 units of correction the loop uses.

Open: the real test is Reflex-1's full lap from `learn/best.json`. The faster and more
knife-edge the planner's line, the more the finish rate will drop; that is when v2 (sim-as-model
MPC on the heap snapshot) earns its place. `eval.mjs` picks `learn/best.json` automatically
when it exists.
