"""Receding-horizon planner for 0.2.0 Track 1, driven from Python via Game.run (proven restart path).

    python3 loop/v020_plan.py --seed 1 [--route normal] [--minutes 40]
Each step: sample N plans of HORIZON ms (SEG ms pieces), roll each out from frame 0 with the
committed prefix, score by waypoint progress (finish beats all), refit, commit COMMIT ms.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import v020_search as S  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

A = [1, 3, 9, 0, 5, 7, 13, 4]

SCORE_JS = r"""
([masks, prefixLen, W]) => {
  window.__plan = {masks, len: masks.length};
  const w = window.__world, c = window.__car;
  let wi = 0, best = -1e9;
  const prog = (p) => {
    while (wi < W.length - 1 && Math.hypot(p.x - W[wi][0], p.y - W[wi][1], p.z - W[wi][2]) < W[wi][3]) wi++;
    return wi * 1000 - Math.hypot(p.x - W[wi][0], p.y - W[wi][1], p.z - W[wi][2]);
  };
  for (let t = 0; t < masks.length; t++) {
    w.__stepN(1);
    if (c.hasFinished()) return {fin: c.getTime(), score: 1e7 - c.getTime() * 1000};
    if (t % 50 === 0) { const s = prog(c.getPosition()); if (t >= prefixLen && s > best) best = s; }
  }
  const p = c.getPosition();
  return {fin: null, score: best, pos: [p.x, p.y, p.z], wi};
}
"""


class Runner:
    def __init__(self, p):
        self.p = p
        self.g = S.Game(p)
        self.uses = 0

    def score(self, masks, prefix_len, W):
        # Ammo leaks per restart (abort(OOM) after a few hundred); recycle the browser.
        self.uses += 1
        if self.uses % 120 == 0:
            self.g.close(); self.g = S.Game(self.p)
        pg = self.g.pg
        for attempt in range(5):
            pg.evaluate("() => { window.__car = undefined; window.__plan = {masks:[0], len:1}; window.dispatchEvent(new KeyboardEvent('keydown', {code: 'KeyR'})); }")
            pg.evaluate("() => window.dispatchEvent(new KeyboardEvent('keydown', {code: 'KeyW'}))")
            try:
                pg.wait_for_function("() => window.__car !== undefined", timeout=3000)
                break
            except Exception:
                # The restart lands while a frame is mid-flight; step once and retry.
                pg.evaluate("() => { if (window.__world) window.__world.__stepN(1); }")
        else:
            raise RuntimeError("restart failed 5 times")
        return pg.evaluate(SCORE_JS, [masks, prefix_len, W])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--minutes", type=float, default=40)
    ap.add_argument("--horizon", type=int, default=2500)
    ap.add_argument("--seg", type=int, default=100)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--commit", type=int, default=500)
    ap.add_argument("--route", default="normal")
    ap.add_argument("--init", help="json whose controls[:--init-keep] seed `committed` (e.g. a climb result)")
    ap.add_argument("--init-keep", type=int, default=0)
    a = ap.parse_args()
    W = S.ROUTE if a.route == "normal" else json.loads(Path(a.route).read_text())
    rnd = random.Random(a.seed)
    segs, cs = a.horizon // a.seg, a.commit // a.seg
    flat = lambda: [0.45 if x == 1 else 0.55 / (len(A) - 1) for x in A]
    probs = [flat() for _ in range(segs)]
    committed, best_seq = [], None
    if a.init:
        c = json.loads(Path(a.init).read_text())
        committed = (c["controls"] if isinstance(c, dict) else c)[: a.init_keep or None]
    t0 = time.time()
    out = Path(f"runs/top/v020_plan_{a.seed}.json")
    with sync_playwright() as p:
        R = Runner(p)
        stall, last = 0, -1e9
        while time.time() - t0 < a.minutes * 60 and len(committed) < 40000:
            top = None
            for _ in range(a.iters):
                pop = []
                for n in range(a.n):
                    seq = best_seq[:] if (n == 0 and best_seq) else [rnd.choices(range(len(A)), weights=pr)[0] for pr in probs]
                    m = committed + [A[i] for i in seq for _ in range(a.seg)]
                    r = R.score(m, len(committed), W)
                    pop.append((r["score"], seq, r))
                pop.sort(key=lambda x: -x[0])
                if top is None or pop[0][0] > top[0]:
                    top = pop[0]
                el = pop[: max(3, a.n // 6)]
                for j in range(segs):
                    cnt = [0.02] * len(A)
                    for e in el:
                        cnt[e[1][j]] += 1
                    tot = sum(cnt)
                    probs[j] = [0.7 * c / tot + 0.3 * probs[j][k] for k, c in enumerate(cnt)]
            r = top[2]
            if r["fin"]:
                ctrl = committed + [A[i] for i in top[1] for _ in range(a.seg)]
                out.write_text(json.dumps({"controls": ctrl, "fin": r["fin"]}))
                print(f"FINISH {r['fin']:.3f} s  wall {time.time() - t0:.0f}s -> {out}", flush=True)
                return
            committed += [A[i] for i in top[1][:cs] for _ in range(a.seg)]
            best_seq = top[1][cs:] + [0] * cs
            probs = probs[cs:] + [flat() for _ in range(cs)]
            print(f"t {len(committed)} wp {r.get('wi')} score {top[0]:.0f} pos {[round(v) for v in r.get('pos', [0, 0, 0])]} wall {time.time() - t0:.0f}s", flush=True)
            # stall: back up 1.5 s with a fresh distribution
            if top[0] <= last + 2:
                stall += 1
            else:
                stall, last = 0, top[0]
            if stall >= 4:
                committed = committed[: max(0, len(committed) - 1500)]
                probs = [flat() for _ in range(segs)]; best_seq = None; stall = 0; last = -1e9
                print("  stall: backed up 1.5 s", flush=True)
            out.write_text(json.dumps({"controls": committed, "fin": None}))


if __name__ == "__main__":
    main()
