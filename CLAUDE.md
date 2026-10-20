# reflex

A System 1 that drives Polytrack from a live desktop video stream with real key
presses. Read `DESIGN.md` first. It is the plan.

## Rules

- **Hot path is one Swift process**: capture, inference, key injection. No Python, no
  IPC, no allocation per frame. Python is for training and analysis only.
- **Every hot-path change posts a number.** Report p50/p99 in ms from a bench run on this
  Mac (M4 Pro, 120 Hz, macOS 27). No number = not done. Include the command that made it.
- **Don't commit the game.** Kodub's files go in gitignored `vendor/` via `sim/fetch.sh`.
- **Never submit bot times to the Polytrack leaderboard.**
- **Root of trust is code**: finish times come from the sim's Verify path or the game's
  HUD, never from a model's opinion.
- Ponytail: stdlib and platform first (ScreenCaptureKit, CGEvent, Accelerate, Core ML),
  fewest files, mark corner-cuts with `ponytail:` comments naming the ceiling.
- No em dashes in prose or commit messages.
- Commit only your own paths: `git add <paths> && git commit -m ...`. Never `git add -A`.
  On `index.lock`, wait 2 s and retry.

## Layout

- `eye/`: Swift package. Capture (SCK), hand (CGEvent), benches.
- `sim/`: Node. Headless Polytrack physics for planning and hindsight labels.
- `.claude/`: curated everything-claude-code subset (agents, skills, rules, commands).

## Agent routing (ECC agents in `.claude/agents/`)

- Latency work: `performance-optimizer`, skill `latency-critical-systems`, `benchmark-methodology`.
- Swift: `swift-reviewer`, `swift-build-resolver`. Training: `mle-reviewer`, `pytorch-patterns`.
- Loops: `loop-operator`, skills `autonomous-loops`, `eval-harness`, `verification-loop`.
- Before claiming done: `/quality-gate`, superpowers `verification-before-completion`.
