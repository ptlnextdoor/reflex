# LOOP: autonomous improving loop (started 2026-10-20 21:58 PT)

Coordinator: jcode session `herb`. Workers: claude-fable-5-1 at max effort.
Scoring is code, never a model's opinion. Every row below has a command that reproduces it.

## Objective (revised 2026-10-20 23:10 PT, owner)

**Reflex-1 must learn to drive on its own and beat 22.123 s without any human run.**
The 22.111 s result (rows 2 and 5) was the #1 player's exact key presses with 15
of them moved by 1 to 2 ms, including his off-track moment at the start. That's a
human run, tweaked. It is **not** a model result and is moved to "Excluded" below.

## Isolation rule (enforced, checked by the coordinator)

No human recording, ghost, or leaderboard data may reach anything that produces Reflex-1's
driving: not as a seed, warm start, reference line, training data, reward target, or
initial controls. The agent gets only what any new player gets:

- the track (geometry from the track file the game loads, checkpoint order, start pose),
- the physics (byte-exact sim) to practice in,
- its own past runs (N+1).

The human WR (22123) is used **only** as the finish-time number to beat, read after
the fact. `loop/isolation.py` scans the agent's code dirs and fails if they reference
`recording-59372564`, `vps.kodub.com`, `leaderboard`, `ghost`, or `decodeRecording` of
anything not produced by the agent itself. Run it before logging any result.

## Tracks

| track | goal | metric | owner |
| --- | --- | --- | --- |
| R planner | learn a Summer 1 line from scratch (no human input), then beat 22123 | sim finishFrame of its own best line, learning curve over iterations | worker R |
| E controller | closed-loop tracking of R's own line under latency/jitter/noise | finish rate + time in the headline cell | worker E (reference must come from R, not the WR) |
| B data engine | real game frames + exact sim labels, from Reflex-1's own runs | position-match proof | worker B |
| C perception | visual System One model trained on B's frames | held-out accuracy, ms/frame | later |
| D live | closed loop on the real game from pixels | real finish ms + video | later |

## Excluded results (not Reflex-1)

| # | what | result | why excluded |
| --- | --- | --- | --- |
| 2 | WR with one key press moved 1 ms | 22119 | human run, tweaked |
| 5 | WR with 15 key presses moved 1 to 2 ms | 22111, Verify true, `runs/videos/side_by_side.mp4` | human run, tweaked, including his off-track at the start |

## Facts the workers get

- Headless rendering works with no visible window: Playwright `headless=True`,
  args `--use-angle=metal --enable-gpu --ignore-gpu-blocklist`.
- Sim: 114k ticks/s single thread, 14 cores. Per-ms stepper with heap snapshot/restore:
  `ctrl/sim.mjs` (Car.step(mask), 23 MB snapshot in 0.6 ms).
- Controls: 5-bit mask per ms, up=1 right=2 down=4 left=8 reset=16.

## Iteration log

| # | time | track | change | result | kept |
| --- | --- | --- | --- | --- | --- |
| 0 | 21:58 | B | headless Metal render probe | renders, 7 s boot | yes |
| 1 | 22:00 | infra | coordinator scorer `loop/score.mjs` | knife-edge finding: the human WR finishes 0/40 under +/-1 ms jitter | yes |
| 3 | 22:03 | D | `loop/live_replay.py` headless live harness + video | open-loop replay crashes; closed loop required | yes (harness) |
| 4 | 22:04 | D | CDP key timing | p50 0.1 ms, max 2.9 ms | yes |
| 6 | 23:06 | D | `loop/record_line.py`: play any line through the real game's engine and record it | works; used for the excluded row 5 video | yes (harness) |
| 7 | 23:10 | all | **isolation rule** (owner): restart planning from scratch, no human data | | yes |
| 8 | 23:20 | E | WR removed from ctrl/; parity on synthetic controls; reference = learn/best.json when present | coordinator re-check: `loop/isolation.py ctrl data learn` **ok**; `node ctrl/check.mjs` 3/3 parity, snapshot/restore exact, 115k ticks/s | yes |
| 9 | 23:23 | B | `data/hook.js`: feed byte-exact sim states into the real game renderer; `data/drive.mjs` random drives (no human data) | 10 frames, page car position == sim position, error **0.0**; HUD timer == frame index. Isolation ok (coordinator re-check) | yes |
| 10 | 23:23 | infra | killed a leftover WR-derived search (`opt/search.mjs --start opt/best.json`, pid 63029, 30 min, ~3 cores) | CPU freed for the from-scratch learner. opt/ is excluded under the isolation rule | yes |
| 11 | 23:27 | B | `data/render.py` batch renderer (live 120 Hz cadence, 512x288 + 256x144), `data/check.py` proof | `python3 data/check.py runs/drives/summer1_s1.json`: 7 frames, page pos/quat/chassis-mesh == sim, worst error **0.0**, screenshots deterministic. `python3 data/render.py runs/drives/summer1_s{1,2}.json`: 161 frames, 18.5 frames/s (0.7 to 1.35x realtime) | yes |
