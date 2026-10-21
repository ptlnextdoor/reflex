# reflex

A visual System One engine (sys1rust lineage, Jev-style vision and Box answers) that
drives Polytrack from a live desktop video stream with real key presses. Read
`DESIGN.md` first. It is the plan.

## Rules

- **Hot path is Swift `eye/` (capture, keys) + Rust `engine/` (inference).** No Python
  in the loop, no allocation per frame. Python is for bake-offs, training and analysis.
- **Every hot-path change posts a number.** Report p50/p99 in ms from a bench run on this
  Mac (M4 Pro, 120 Hz, macOS 27). No number = not done. Include the command that made it.
- **Correctness gate for engine changes:** >= 99% answer agreement with the PyTorch
  reference on the fixed workload, as sys1rust did. Faster but different = rejected.
- **One GPU job at a time.** A bench and a training run at once corrupt both numbers.
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
- `bench/`: runtime bake-offs (Python allowed here).
- `engine/`: Rust, fork of github.com/krishhgg/sys1rust (local clone at
  `~/Workspace/developer/sys1rust`), adds images, Box, streaming.
- `.claude/`: curated everything-claude-code subset (agents, skills, rules, commands).

## Agent routing (ECC agents in `.claude/agents/`)

- Latency work: `performance-optimizer`, skill `latency-critical-systems`, `benchmark-methodology`.
- Swift: `swift-reviewer`, `swift-build-resolver`. Training: `mle-reviewer`, `pytorch-patterns`.
- Loops: `loop-operator`, skills `autonomous-loops`, `eval-harness`, `verification-loop`.
- Before claiming done: `/quality-gate`, superpowers `verification-before-completion`.
- Numbers: project skill `reflex-bench`, command `/bench`. Guard hook
  `.claude/hooks/leaderboard_guard.py` blocks any leaderboard submit (exit 2).

## Method stack

- everything-claude-code (this `.claude/`): agents, skills, rules, the guard hook.
- superpowers (user plugin): `systematic-debugging`, `verification-before-completion`,
  `writing-plans` for multi-step work.
- elon-musk-skill (user skill): question each requirement, delete before optimizing. Every
  recommendation states P(success) and its most likely failure.
- GSD: not used. `DESIGN.md` is the plan and `bench/*/RESULTS.md` is the state.
- gstack (user skill): `/browse` for checking the live game page by hand.

## Git workflow

- Remote: private GitHub `origin`. `main` is protected by convention: agents work on a
  branch `track/<letter>-<topic>`, commit small, open a PR with `gh pr create`, and the
  coordinator merges after re-running the PR's check (`loop/isolation.py` + the track's
  own check). Commit messages: imperative, one line, why in the body.
