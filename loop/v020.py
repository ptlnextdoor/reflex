"""Headless driver for PolyTrack 0.2.0 (Ammo.js physics, 1 kHz fixed step).

Serves vendor/polytrack/0.2.0 with a patched bundle (bundle.patched.js) in which the keyboard
controller's getControls(frame) reads window.__plan.masks[frame], so the car is driven by
physics frame number (no OS keys, no timing jitter). window.__car is the player car.

    python3 loop/v020.py probe                         # menu buttons + screenshot
    python3 loop/v020.py run plan.json [--video out.webm] [--max-s 40]
plan.json: {"toggles": [[frame, mask], ...], "frames": N} or {"controls": [...]}.
mask: up=1 right=2 down=4 left=8. Prints finish time (s) or the last position.
"""
from __future__ import annotations

import argparse
import functools
import http.server
import json
import threading
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "vendor/polytrack/0.2.0"


class Static(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


def controls_of(plan):
    if "controls" in plan:
        return list(plan["controls"])
    n = plan.get("frames") or (plan["toggles"][-1][0] + 1)
    out, cur, i = [], 0, 0
    tg = plan["toggles"]
    for t in range(n):
        while i < len(tg) and tg[i][0] <= t:
            cur = tg[i][1]; i += 1
        out.append(cur)
    return out


def boot(p, video=None):
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Static, directory=str(ROOT)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    origin = f"http://127.0.0.1:{server.server_port}"
    b = p.chromium.launch(headless=True, args=["--use-angle=metal", "--enable-gpu", "--ignore-gpu-blocklist", "--autoplay-policy=no-user-gesture-required"])
    kw = {"viewport": {"width": 1280, "height": 720}, "service_workers": "block"}
    if video:
        kw["record_video_dir"] = str(Path(video).parent); kw["record_video_size"] = {"width": 1280, "height": 720}
    ctx = b.new_context(**kw)
    ctx.route("**/*", lambda r: r.continue_() if r.request.url.startswith(origin + "/") else r.abort())
    pg = ctx.new_page()
    pg.goto(origin + "/index.patched.html", wait_until="domcontentloaded")
    pg.wait_for_function("() => [...document.querySelectorAll('button')].some(b => b.textContent.trim()==='Play' && b.offsetParent)", timeout=60000)
    return server, b, ctx, pg


def click_text(pg, txt, timeout=15000):
    pg.wait_for_function("(t) => [...document.querySelectorAll('button, div, p, span')].some(e => e.textContent.trim()===t && e.offsetParent)", arg=txt, timeout=timeout)
    pg.evaluate("(t) => { const e=[...document.querySelectorAll('button, div, p, span')].filter(e => e.textContent.trim()===t && e.offsetParent).pop(); (e.closest('button')||e).click(); }", txt)


def probe():
    with sync_playwright() as p:
        server, b, ctx, pg = boot(p)
        click_text(pg, "Play")
        pg.wait_for_timeout(1500)
        print(pg.evaluate("() => [...document.querySelectorAll('button')].filter(e=>e.offsetParent).map(e=>e.textContent.trim()).slice(0,40)"))
        pg.screenshot(path=str(REPO / "runs/top/v020_play.png"))
        b.close(); server.shutdown()


def run(plan_path, video=None, max_s=40):
    plan = json.loads(Path(plan_path).read_text())
    masks = controls_of(plan)
    with sync_playwright() as p:
        server, b, ctx, pg = boot(p, video)
        pg.evaluate("(m) => { window.__plan = {masks: m, len: m.length}; }", masks)
        click_text(pg, "Play")
        pg.wait_for_timeout(1000)
        names = pg.evaluate("() => [...document.querySelectorAll('button')].filter(e=>e.offsetParent).map(e=>e.textContent.trim())")
        # Track 1 tile: first track button in the picker
        pg.evaluate("() => [...document.querySelectorAll('button')].find(e=>e.offsetParent && e.textContent.trim().startsWith('Track 1')).click()")
        pg.wait_for_timeout(1500)
        names2 = pg.evaluate("() => [...document.querySelectorAll('button')].filter(e=>e.offsetParent).map(e=>e.textContent.trim())")
        if "Play" in names2:
            click_text(pg, "Play")
        pg.wait_for_timeout(1500)
        pg.evaluate("() => window.dispatchEvent(new KeyboardEvent('keydown', {code: 'KeyW'}))")
        pg.wait_for_function("() => window.__car !== undefined", timeout=20000)
        res = pg.wait_for_function(
            "(maxs) => { const c=window.__car; if(!c) return null; if (c.hasFinished()) return {fin: c.getTime()}; if (c.getTime() > maxs) { const p=c.getPosition(); return {fin: null, t: c.getTime(), pos: [p.x,p.y,p.z]}; } return null; }",
            arg=max_s, timeout=int(max_s * 1000 * 4) + 30000, polling=100).json_value()
        print(json.dumps(res))
        if video:
            pg.wait_for_timeout(1500)
        ctx.close(); b.close(); server.shutdown()
        if video:
            vids = sorted(Path(video).parent.glob("*.webm"), key=lambda q: q.stat().st_mtime)
            if vids:
                vids[-1].rename(video)
        return res


FAST_JS = r"""
async ({masks, maxms}) => {
  // Manual 1 ms stepping: stop the rAF-driven update, run fixed steps synchronously.
  window.__plan = {masks, len: masks.length};
  window.__manual = true;
  const w = window.__world, c = window.__car;
  const out = {fin: null, trace: []};
  for (let t = 0; t < maxms; t++) {
    w.__stepN(1);
    if (t % 250 === 0) { const p = c.getPosition(); out.trace.push([t, +p.x.toFixed(1), +p.y.toFixed(1), +p.z.toFixed(1), +c.getSpeedKmh().toFixed(0)]); }
    if (c.hasFinished()) { out.fin = c.getTime(); out.ticks = t + 1; break; }
  }
  if (out.fin === null) { const p = c.getPosition(); out.end = [p.x, p.y, p.z]; }
  return out;
}
"""


def fast(plan_path, max_s=40):
    """Physics-only run at full CPU speed (no rendering wait). Returns finish seconds or None."""
    plan = json.loads(Path(plan_path).read_text())
    masks = controls_of(plan)
    with sync_playwright() as p:
        server, b, ctx, pg = boot(p)
        pg.evaluate("(m) => { window.__plan = {masks: m, len: m.length}; }", masks)
        click_text(pg, "Play")
        pg.wait_for_timeout(800)
        pg.evaluate("() => [...document.querySelectorAll('button')].find(e=>e.offsetParent && e.textContent.trim().startsWith('Track 1')).click()")
        pg.wait_for_timeout(1500)
        # Freeze realtime stepping before the start key, so the race begins on a manual step.
        pg.evaluate("() => { window.__manual = true; }")
        pg.evaluate("() => window.dispatchEvent(new KeyboardEvent('keydown', {code: 'KeyW'}))")
        pg.wait_for_function("() => window.__car !== undefined && window.__world !== undefined", timeout=20000)
        res = pg.evaluate(FAST_JS, {"masks": masks, "maxms": int(max_s * 1000)})
        print(json.dumps({k: v for k, v in res.items() if k != "trace"}))
        b.close(); server.shutdown()
        return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("plan", nargs="?")
    ap.add_argument("--video")
    ap.add_argument("--max-s", type=float, default=40)
    a = ap.parse_args()
    if a.cmd == "probe":
        probe()
    elif a.cmd == "fast":
        fast(a.plan, a.max_s)
    else:
        run(a.plan, a.video, a.max_s)
