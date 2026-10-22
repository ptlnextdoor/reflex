"""In-page search on PolyTrack 0.2.0 Track 1 (Summer 1). One browser, many restarts.

    python3 loop/v020_search.py check                     # restart determinism
    python3 loop/v020_search.py replay plan.json          # finish time
    python3 loop/v020_search.py cem --minutes 15 --seed 1 # search for fastest finish (any route)
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import v020  # noqa: E402
from playwright.sync_api import sync_playwright  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SIMJS = (Path(__file__).parent / "v020_sim.js").read_text()
FINISH = [-300.0, 22.2, 280.0]  # Track 1 finish detector centre (part -15,4,14 x20,x5)


class Game:
    def __init__(self, p):
        self.server, self.b, self.ctx, self.pg = v020.boot(p)
        pg = self.pg
        pg.add_script_tag(content=SIMJS)
        v020.click_text(pg, "Play"); pg.wait_for_timeout(800)
        pg.evaluate("() => [...document.querySelectorAll('button')].find(e=>e.offsetParent && e.textContent.trim().startsWith('Track 1')).click()")
        pg.wait_for_timeout(1500)
        pg.evaluate("() => { window.__manual = true; window.__plan = {masks:[0], len:1}; }")

    def run(self, masks, maxms=30000, target=FINISH):
        pg = self.pg
        # Fresh car via the game's restart, then start the race on a manual step.
        pg.evaluate("() => { window.__car = undefined; window.dispatchEvent(new KeyboardEvent('keydown', {code: 'KeyR'})); }")
        pg.evaluate("() => window.dispatchEvent(new KeyboardEvent('keydown', {code: 'KeyW'}))")
        pg.wait_for_function("() => window.__car !== undefined", timeout=10000)
        return pg.evaluate("([m, n, tg]) => __sim.run(m, n, tg)", [masks, maxms, target])

    def close(self):
        self.b.close(); self.server.shutdown()


def masks_from(seq, seg):
    out = []
    for a in seq:
        out += [a] * seg
    return out


def cem(minutes, seed, seg=100, horizon=24000):
    rnd = random.Random(seed)
    A = [1, 3, 9, 0, 5, 7, 13, 4, 11]
    S = horizon // seg
    probs = [[1 / len(A)] * len(A) for _ in range(S)]
    probs = [[0.6 if a == 1 else 0.4 / (len(A) - 1) for a in A] for _ in range(S)]  # bias to throttle
    best = None
    t0 = time.time()
    with sync_playwright() as p:
        g = Game(p)
        it = 0
        while time.time() - t0 < minutes * 60:
            pop = []
            for _ in range(48):
                seq = [rnd.choices(range(len(A)), weights=pr)[0] for pr in probs]
                r = g.run(masks_from([A[i] for i in seq], seg), horizon)
                score = (1e6 - r["fin"] * 1000) if r["fin"] else -r["minD"]
                pop.append((score, seq, r))
            pop.sort(key=lambda x: -x[0])
            if best is None or pop[0][0] > best[0]:
                best = pop[0]
                print(f"it {it} {time.time() - t0:.0f}s best", best[2].get("fin") or f"minD {best[2]['minD']:.1f} m at {best[2].get('minAt')}", flush=True)
                Path("runs/top/v020_best.json").write_text(json.dumps({"controls": masks_from([A[i] for i in best[1]], seg), "res": best[2]}))
            el = pop[:8]
            for j in range(S):
                cnt = [0.02] * len(A)
                for e in el:
                    cnt[e[1][j]] += 1
                tot = sum(cnt)
                probs[j] = [0.7 * c / tot + 0.3 * probs[j][k] for k, c in enumerate(cnt)]
            it += 1
        g.close()


RHPJS = (Path(__file__).parent / "v020_rhp.js").read_text()
# Normal Track 1 route in 0.2.0 coordinates (from the track's own parts), radius per waypoint.
ROUTE = [
    [200, 56, 15, 14], [170, 50, 0, 14], [120, 35, 0, 14], [60, 10, 0, 14], [0, 1, -10, 14],
    [-100, 1, -10, 16], [-180, 1, -10, 16], [-215, 10, -40, 14], [-210, 11, -100, 14],
    [-150, 11, -100, 14], [-100, 11, -90, 14], [-100, 11, -40, 14], [-120, 18, 20, 14],
    [-140, 21, 80, 16], [-140, 21, 160, 16], [-170, 21, 240, 16], [-240, 21, 280, 16], [-300, 22, 280, 8],
]


def rhp(minutes, seed, route=None, horizon=3000, samples=24, iters=3, commit=400):
    t0 = time.time()
    with sync_playwright() as p:
        g = Game(p)
        g.pg.add_script_tag(content=RHPJS)
        g.run([1] * 10, 10)  # first run sets window.__world
        g.pg.evaluate("() => { window.__W = window.__world; }")
        g.pg.evaluate("""() => { window.__restart = () => {
            window.__car = undefined; window.__plan = {masks:[0], len:1};
            window.dispatchEvent(new KeyboardEvent('keydown', {code: 'KeyR'}));
            window.dispatchEvent(new KeyboardEvent('keydown', {code: 'KeyW'})); }; }""")
        res = g.pg.evaluate("""async (o) => {
            const r = await __rhp.plan({...o, restart: () => __restart(), onStep: (t, s) => { window.__progress = [t, s]; }});
            return r; }""", {"waypoints": route or ROUTE, "horizon": horizon, "seg": 100, "samples": samples,
                             "iters": iters, "commit": commit, "maxms": 40000, "seed": seed})
        print("\n".join(res["log"][-6:]))
        print(json.dumps({"fin": res["fin"], "ms": len(res["controls"]), "wall_s": round(time.time() - t0)}))
        out = Path(f"runs/top/v020_rhp_{seed}.json")
        out.write_text(json.dumps({"controls": res["controls"], "fin": res["fin"]}))
        g.close()
        return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("plan", nargs="?")
    ap.add_argument("--minutes", type=float, default=15)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    if a.cmd == "check":
        with sync_playwright() as p:
            g = Game(p)
            for i in range(3):
                t = time.time(); r = g.run([1] * 12000, 12000); print(json.dumps(r), f"{time.time() - t:.1f}s")
            g.close()
    elif a.cmd == "replay":
        with sync_playwright() as p:
            g = Game(p); print(json.dumps(g.run(v020.controls_of(json.loads(Path(a.plan).read_text())), 40000))); g.close()
    elif a.cmd == "cem":
        cem(a.minutes, a.seed)
    elif a.cmd == "rhp":
        rhp(a.minutes, a.seed)
