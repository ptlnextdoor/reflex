---
name: reflex-bench
description: Measure a latency or accuracy change in reflex the way this repo accepts it. Use before claiming any hot-path change (eye/, engine/, bench/) is faster, or when asked for p50/p99, motion-to-photon, agreement, or IoU numbers.
---

# reflex-bench

A change is done when it has a number from this Mac, the command that made it, and
no correctness regression. sys1rust's method, applied here.

## Steps

1. **One GPU job at a time.** Check `ps aux | grep -E 'valen|sys1rust|eye bench|train'`.
   If another GPU job runs, wait. Two at once corrupt both numbers.
2. **Warm up, then measure.** Discard the first runs (kernel compile, caches). At least
   200 samples for latency, report p50, p90, p99, max. Never a mean alone.
3. **Same inputs as the baseline.** Fixed frames and questions (`bench/infer/valen_ref.py`
   QUESTIONS), fixed revision. Different inputs = no comparison.
4. **Correctness gate.** Engine changes: >= 99% answer agreement with
   `bench/infer/ref_*.jsonl` (same choice; score within 0.05; noul within 0.02).
   Box changes: IoU vs sim labels does not drop.
5. **Record it.** Add a row to the matching `RESULTS.md` (date, device, runtime, p50, p99,
   notes) and commit it with the change.
6. **Say what it costs.** Memory, accuracy, warm-up time. A faster path that only wins on
   one input shape is a finding, not a default.

## Commands

- Capture: `eye/.build/release/eye bench-capture --rect x,y,w,h --seconds 10`
- Motion-to-photon: `eye/.build/release/eye bench-m2p --app com.google.Chrome --rect ... --trials 200`
- Valen reference: `cd bench/infer && .venv/bin/python valen_ref.py --device mps`
- Sim parity: `node sim/check.js`
