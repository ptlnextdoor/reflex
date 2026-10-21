---
description: Run reflex's checks (selftests, sim parity, Valen reference) and report numbers
---

Follow the `reflex-bench` skill. Run, in order, and stop at the first failure:

1. `python3 .claude/hooks/leaderboard_guard.py --selftest`
2. `cd bench/infer && .venv/bin/python valen_ref.py --selftest`
3. `node sim/check.js` (if `sim/` exists)
4. `eye/.build/release/eye selftest` (if built)

Then, only if no other GPU job is running: `$ARGUMENTS` if given, else the Valen MPS
reference bench. Report a table of results and the exact commands.
