"""Focused CEM for the 0.2.0 Track 1 wall-climb shortcut (the 16 s route).

    python3 loop/v020_climb.py --prefix runs/top/v020_prefix5000.json --keep 4000 --seed 1

Keeps the first --keep ms of the prefix (car on the low road heading -x), then searches
--horizon ms of controls (--seg ms pieces) to get the car on top of the y=20 block at
x in [-150,-50], z >= 35, and as far toward +z as possible. Output feeds v020_plan.py --init.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from v020_plan import Runner, A  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

CLIMB_JS = r"""
([masks, prefixLen]) => {
  window.__plan = {masks, len: masks.length};
  const w = window.__world, c = window.__car;
  let best = -1e9, bestT = 0, pos = null;
  for (let t = 0; t < masks.length; t++) {
    w.__stepN(1);
    if (c.hasFinished()) return {fin: c.getTime(), score: 1e7};
    if (t >= prefixLen && t % 25 === 0) {
      const p = c.getPosition();
      const onTop = p.y > 18 && p.y < 26 && p.z > 35 && p.x > -155 && p.x < -45;
      // On top: reward reaching it early and pushing toward +z (the elevated road to the finish).
      const s = onTop ? 10000 + 20 * p.z - 0.5 * (t - prefixLen) / 25
                      : (p.x < -50 ? 30 * Math.min(p.y, 22) : 0) - Math.hypot(p.x + 100, p.z - 50);
      if (s > best) { best = s; bestT = t; pos = [p.x, p.y, p.z]; }
    }
  }
  const p = c.getPosition();
  // Only count "on top" if it is still up there at the end (not bouncing off).
  const stillUp = p.y > 18 && p.z > 35;
  return {fin: null, score: stillUp ? best + 5000 : best, pos, end: [p.x, p.y, p.z], bestT};
}
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", default="runs/top/v020_prefix5000.json")
    ap.add_argument("--keep", type=int, default=4000)
    ap.add_argument("--horizon", type=int, default=4000)
    ap.add_argument("--seg", type=int, default=50)
    ap.add_argument("--n", type=int, default=32)
    ap.add_argument("--minutes", type=float, default=12)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    prefix = json.loads(Path(a.prefix).read_text())
    prefix = (prefix["controls"] if isinstance(prefix, dict) else prefix)[: a.keep]
    segs = a.horizon // a.seg
    flat = [0.4 if x == 1 else 0.6 / (len(A) - 1) for x in A]
    probs = [flat[:] for _ in range(segs)]
    best = None
    out = Path(f"runs/top/v020_climb_{a.seed}.json")
    t0 = time.time()
    gen = 0
    with sync_playwright() as p:
        R = Runner(p)

        def score(masks):
            R.uses += 1
            if R.uses % 120 == 0:
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
            return pg.evaluate(CLIMB_JS, [masks, len(prefix)])

        while time.time() - t0 < a.minutes * 60:
            gen += 1
            pop = []
            for n in range(a.n):
                seq = best[1][:] if (n == 0 and best) else [rnd.choices(range(len(A)), weights=pr)[0] for pr in probs]
                r = score(prefix + [A[i] for i in seq for _ in range(a.seg)])
                pop.append((r["score"], seq, r))
            pop.sort(key=lambda x: -x[0])
            if best is None or pop[0][0] > best[0]:
                best = pop[0]
                ctrl = prefix + [A[i] for i in best[1] for _ in range(a.seg)]
                out.write_text(json.dumps({"controls": ctrl, "score": best[0], "info": best[2]}))
            el = pop[: max(4, a.n // 6)]
            for j in range(segs):
                cnt = [0.02] * len(A)
                for e in el:
                    cnt[e[1][j]] += 1
                tot = sum(cnt)
                probs[j] = [0.6 * c / tot + 0.4 * probs[j][k] for k, c in enumerate(cnt)]
            r = best[2]
            print(f"gen {gen} best {best[0]:.0f} pos {[round(v) for v in (r.get('pos') or [0,0,0])]} end {[round(v) for v in (r.get('end') or [0,0,0])]} fin {r.get('fin')} wall {time.time()-t0:.0f}s", flush=True)
            if r.get("fin"):
                break


if __name__ == "__main__":
    main()
