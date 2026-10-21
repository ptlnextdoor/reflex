# LOOP: autonomous improving loop (started 2026-10-20 21:58 PT)

Coordinator: jcode session `herb`. Workers: claude-fable-5-1 at max effort.
Scoring is code, never a model's opinion. Every row below has a command that reproduces it.

## Objective

Beat the Summer 1 world record (22.123 s) in Polytrack, first in the byte-exact sim,
then live from pixels with a visual System One model. Never submit anything.

## Tracks

| track | goal | metric | owner |
| --- | --- | --- | --- |
| A optimizer | sim line faster than 22123 ms, robust to timing jitter | `finishFrame` from `sim.run`, plus finish rate under +/-1 to 2 ms jitter | worker A |
| B data engine | real game frames from headless Chromium (Metal) + exact sim labels | position-match proof, frames/s | worker B |
| C model | per-frame System One that drives | held-out accuracy on sim labels, ms/frame | after B |
| D live | closed loop on the real game | real finish ms | after A, B, C |

## Facts the workers get

- Headless rendering works with no visible window: Playwright `headless=True`,
  args `--use-angle=metal --enable-gpu --ignore-gpu-blocklist`. The renderer string is
  "ANGLE Metal Renderer: Apple M4 Pro". The page shows an "unofficial version" banner but loads.
- WR (recording 59372564): 22123 frames, 133 key toggles, checkpoints at ~4100, 9690,
  14930 ms. Mask histogram: up 6880, up+right 9106, up+down 3063, up+left 2236, others small.
- Sim: 114k ticks/s single thread (`node sim/check.js`), 14 cores.

## Iteration log

| # | time | track | change | result | kept |
| --- | --- | --- | --- | --- | --- |
| 0 | 21:58 | B | headless Metal render probe | renders, 7 s boot | yes |
| 1 | 22:00 | A | coordinator scorer `loop/score.mjs`, WR calibration | WR finishes 0/40 under +/-1 ms per-toggle jitter, 0/40 at +/-2 ms. Single-toggle +/-1 ms shifts: 125/266 still finish | yes |
| 2 | 22:01 | A | single-toggle +/-1 ms shift scan | best finish **22119 ms (-4 ms vs WR)** from one shift. Slack exists | yes |
| 3 | 22:03 | D | `loop/live_replay.py`: real game, headless Metal, WR keys from Python wall clock via CDP | menu nav works, car drives, **crashed off the track before checkpoint 2** (1/3 cps at 23.2 s). Open-loop wall-clock keys are too jittery for a 0%-margin line, as track A's jitter test predicted | yes (harness) |
| 4 | 22:04 | D | CDP key-delivery timing into a headless page (200 keys) | after median alignment: p50 0.1 ms, p90 0.3 ms, max 2.9 ms. Worker stamps inputs at `frames + (performance.now() - lastBatch)` and steps in rAF batches. So delivery is good, but the WR line has zero margin: **open loop can't work. Closed-loop tracking (DESIGN L0/L1) is required** | yes |
| 5 | 22:34 | A | optimizer (opt/, snapshot worker pool, 215 s search) | **22111 ms, -12 ms vs WR**. Coordinator re-check: `loop/score.mjs` finish 22111, game's own Verify(recording, 22111) = **true** (22112 = false). Jitter +/-1 ms: 0/20 finish (still knife-edge) | yes |
| 6 | 23:06 | D | `loop/record_line.py`: inject a recording into the real game's CreateCar (game's own physics + renderer), headless, recorded | real game finishes **22111** (HUD 00:22.111, "NEW PERSONAL BEST"), WR in same setup **22123**. Side-by-side: `runs/videos/side_by_side.mp4`. This is the optimized input line replayed by the game engine, not yet the live bot driving from pixels | yes |
