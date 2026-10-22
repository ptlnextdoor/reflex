"""Hand-tuned wall-slam sweep for 0.2.0 Track 1: keep prefix[:keep], then W+turn for dur ms, then W.

    python3 loop/v020_sweep.py --shard 0 --nshard 3
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from v020_plan import Runner  # noqa: E402
from v020_climb import CLIMB_JS  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="runs/top/v020_plan_1.json")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--fine", action="store_true")
    a = ap.parse_args()
    prefix = json.loads(Path(a.prefix).read_text())
    prefix = prefix["controls"] if isinstance(prefix, dict) else prefix
    # turn masks: W+D, W+A, D, A; tail after the turn: W, W+opposite
    grid = list(itertools.product(range(4500, 6750, 250), [3, 9, 2, 8], [200, 400, 700, 1000, 1500], [1, 0]))
    if a.fine:  # around the 5000/D/200 launch (y=22 at x=-99): then steer W+A or W+D after gap ms
        grid = list(itertools.product(range(4900, 5125, 50), [2], [100, 200, 300], [(g, t2, d2) for g in (300, 800, 1500) for t2 in (3, 9) for d2 in (300, 800)]))
    grid = grid[a.shard::a.nshard]
    out = Path(f"runs/top/v020_sweep{'f' if a.fine else ''}_{a.shard}.json")
    best = None
    with sync_playwright() as p:
        R = Runner(p)
        for i, (keep, turn, dur, tail) in enumerate(grid):
            if i and i % 100 == 0:
                R.g.close(); R.g = __import__("v020_search").Game(R.p)
            pg = R.g.pg
            for _ in range(5):
                pg.evaluate("() => { window.__car = undefined; window.__plan = {masks:[0], len:1}; window.dispatchEvent(new KeyboardEvent('keydown', {code: 'KeyR'})); }")
                pg.evaluate("() => window.dispatchEvent(new KeyboardEvent('keydown', {code: 'KeyW'}))")
                try:
                    pg.wait_for_function("() => window.__car !== undefined && window.__world !== undefined", timeout=5000)
                    break
                except Exception:
                    pg.evaluate("() => { if (window.__world) window.__world.__stepN(1); }")
            if isinstance(tail, tuple):
                g, t2, d2 = tail
                masks = prefix[:keep] + [turn] * dur + [1] * g + [t2] * d2 + [1] * (4500 - dur - g - d2)
            else:
                masks = prefix[:keep] + [turn] * dur + [tail] * (4500 - dur)
            r = pg.evaluate(CLIMB_JS, [masks, keep])
            print(keep, turn, dur, tail, round(r["score"]), r.get("pos"), r.get("end"), r.get("fin"), flush=True)
            if best is None or r["score"] > best[0]:
                best = (r["score"], masks, r)
                out.write_text(json.dumps({"controls": masks, "score": r["score"], "info": r, "params": [keep, turn, dur, tail]}))


if __name__ == "__main__":
    main()
