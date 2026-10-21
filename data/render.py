#!/usr/bin/env python3
"""Render labeled real-game frames headlessly at exact sim frame indices.

For each controls file ({track, frames, toggles|controls}, e.g. from data/drive.mjs):
  1. node data/labels.mjs writes states.bin (every 227-byte sim state) + labels.jsonl.
  2. Headless Chromium (Metal) boots the real game into the track, data/hook.js takes over
     requestAnimationFrame and feeds the player car the sim states through the game's own
     UpdateResult callback, so the renderer draws exactly sim frame k.
  3. The loop runs at the live cadence (120 Hz renders, ticks fed between renders) and a
     PNG is saved whenever the render lands on a sampled frame (every --step ms).

Usage: python3 data/render.py runs/drives/summer1_s1.json [more.json ...] [--out runs/frames]
       [--step 50] [--width 512 --height 288] [--small 256x144]
Network: localhost only, every other request is aborted. Never submits anything.
"""
import argparse
import functools
import http.server
import io
import json
import re
import subprocess
import threading
import time
from pathlib import Path

from PIL import Image, ImageDraw
from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "vendor/polytrack/0.6.3"
HOOK = REPO / "data/hook.js"
STATE_BYTES = 227
RENDER_HZ = 120
LAUNCH_ARGS = ["--use-angle=metal", "--enable-gpu", "--ignore-gpu-blocklist"]


def track_title(slug):
    m = re.fullmatch(r"([a-z]+)(\d+)", slug)
    if not m:
        raise ValueError(f"not an official track slug: {slug}")
    return f"{m.group(1).capitalize()} {m.group(2)}"


def make_labels(controls_file, out_dir, step):
    subprocess.run(
        ["node", str(REPO / "data/labels.mjs"), str(controls_file), str(out_dir), str(step)],
        check=True, cwd=REPO,
    )
    return json.loads((out_dir / "meta.json").read_text())


class Session:
    """One headless browser serving the local game; frames fed from a states.bin."""

    def __init__(self, width=512, height=288):
        self.width, self.height = width, height
        self.states_path = None
        self.track = None
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), self._handler())
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.origin = f"http://127.0.0.1:{self.server.server_port}"
        self._pw = sync_playwright().start()
        self.browser = self._pw.chromium.launch(headless=True, args=LAUNCH_ARGS)
        self.context = self.browser.new_context(
            viewport={"width": width, "height": height}, service_workers="block"
        )
        origin = self.origin
        self.context.route("**/*", lambda r: r.continue_() if r.request.url.startswith(origin + "/") else r.abort())
        self.page = self.context.new_page()
        self.errors = []
        self.page.on("pageerror", lambda e: self.errors.append(str(e)))
        self.page.add_init_script(path=str(HOOK))
        self.page.goto(self.origin + "/", wait_until="domcontentloaded")
        assert self.page.evaluate("() => __rf.setup()") == "ok"
        self.page.wait_for_function("!document.body.innerText.includes('Loading')", timeout=60000)
        self.in_game = False

    def _handler(self):
        session = self

        class Handler(http.server.SimpleHTTPRequestHandler):
            def log_message(self, *_):
                pass

            def translate_path(self, path):
                if path == "/__states.bin" and session.states_path:
                    return str(session.states_path)
                return super().translate_path(path)

        return functools.partial(Handler, directory=str(ROOT))

    def _click(self, js):
        assert self.page.evaluate(js) is True, js

    def enter_track(self, slug):
        """Main menu -> Play -> track tile -> Play. Leaves the game running in manual mode."""
        if self.track == slug:
            return
        page = self.page
        if self.in_game:
            page.evaluate("() => { __rf.manual = false; }")
            self._click("() => { const b=[...document.querySelectorAll('#ui button')].find(b => b.textContent.trim()==='Exit' && b.offsetParent!==null); if(!b) return false; b.click(); return true; }")
            page.wait_for_function("() => [...document.querySelectorAll('#ui button')].some(b => b.querySelector('p')?.textContent.trim()==='Play' && b.offsetParent!==null)", timeout=20000)
            self.in_game = False
        title = track_title(slug)
        self._click("() => { const b=[...document.querySelectorAll('#ui button')].find(b => b.querySelector('p')?.textContent.trim()==='Play' && b.offsetParent!==null); if(!b) return false; b.click(); return true; }")
        page.wait_for_function("() => [...document.querySelectorAll('#ui .track')].some(t => t.offsetParent !== null)", timeout=10000)
        self._click(f"() => {{ const title={json.dumps(title)}; const t=[...document.querySelectorAll('#ui .track')].find(t => t.querySelector('.track-title p')?.textContent.trim()===title); if(!t) return false; t.querySelector('button').click(); return true; }}")
        page.wait_for_function("() => [...document.querySelectorAll('#ui button.play')].some(b => b.offsetParent !== null)", timeout=10000)
        before = page.evaluate("() => __rf.cars.length")
        self._click("() => { const b=[...document.querySelectorAll('#ui button.play')].find(b => b.offsetParent!==null); if(!b) return false; b.click(); return true; }")
        page.wait_for_function(f"() => __rf.cars.length > {before} && document.getElementById('transition-layer').style.opacity === '0'", timeout=20000)
        time.sleep(0.3)
        page.evaluate("() => { __rf.manual = true; }")
        page.wait_for_function("() => __rf.pending.length > 0", timeout=5000, polling=20)
        self.in_game = True
        self.track = slug
        self.episodes = 0

    def load_states(self, states_path):
        self.states_path = Path(states_path)
        n = self.page.evaluate("() => __rf.load('/__states.bin')")
        assert n == self.states_path.stat().st_size / STATE_BYTES
        return int(n)

    def new_car(self):
        """Fresh player car at frame 0 (first episode uses the car the track started with)."""
        page = self.page
        if self.episodes > 0:
            before = page.evaluate("() => __rf.cars.length")
            page.evaluate("() => __rf.restart()")
            for _ in range(50):
                page.evaluate(f"(dt) => __rf.step(dt)", 1000 / RENDER_HZ)
                if page.evaluate("() => __rf.cars.length") > before:
                    break
            else:
                raise RuntimeError("restart did not create a new car")
            page.evaluate(f"(dt) => __rf.step(dt)", 1000 / RENDER_HZ)
        self.episodes += 1
        return page.evaluate("() => __rf.cars.filter(c => !c.recording && !c.deleted).at(-1).id")

    def run(self, frames, step, on_sample):
        """Feed ticks at the live cadence; call on_sample(k, car_id) when a render lands on k % step == 0."""
        page = self.page
        car_id = self.new_car()
        fed = 0
        i = 0
        while fed < frames:
            i += 1
            target = min(frames, round(i * 1000 / RENDER_HZ))
            if target > fed:
                page.evaluate("([id, a, b]) => __rf.feed(id, a, b)", [car_id, fed, target])
                fed = target
            renders = page.evaluate("(dt) => __rf.step(dt)", 1000 / RENDER_HZ)
            assert renders == 1, f"expected one render per step, got {renders}"
            if fed % step == 0 and fed > 0 and target == round(i * 1000 / RENDER_HZ):
                on_sample(fed, car_id)
        return car_id

    def probe(self, car_id):
        return self.page.evaluate("(id) => __rf.probe(id)", car_id)

    def box(self):
        return self.page.evaluate("([w, h]) => __rf.box(w, h)", [self.width, self.height])

    def screenshot(self):
        return self.page.screenshot(type="png")

    def canvas(self):
        """The WebGL canvas alone (no DOM HUD), same size as the viewport."""
        return self.page.locator("#screen").screenshot(type="png")

    def close(self):
        self.browser.close()
        self._pw.stop()
        self.server.shutdown()


def render_episode(session, controls_file, out_root, step, small, debug_dir=None):
    controls_file = Path(controls_file)
    out_dir = Path(out_root) / controls_file.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    meta = make_labels(controls_file, out_dir, step)
    session.enter_track(meta["track"])
    n = session.load_states(out_dir / "states.bin")
    assert n == meta["frames"]
    small_dir = out_dir / "small"
    if small:
        small_dir.mkdir(exist_ok=True)
    saved = []
    boxes = {}
    t0 = time.perf_counter()

    def on_sample(k, car_id):
        png = session.screenshot()
        (out_dir / f"{k:06d}.png").write_bytes(png)
        if small:
            img = Image.open(io.BytesIO(png)).convert("RGB").resize(small, Image.LANCZOS)
            img.save(small_dir / f"{k:06d}.png", optimize=False)
        b = session.box()
        boxes[k] = b
        saved.append(k)
        if debug_dir and len(saved) in (1, 2, 3) and b:
            img = Image.open(io.BytesIO(png)).convert("RGB")
            ImageDraw.Draw(img).rectangle([b["x0"], b["y0"], b["x1"], b["y1"]], outline=(255, 0, 255), width=2)
            img.save(Path(debug_dir) / f"{controls_file.stem}_{k:06d}.png")

    session.run(meta["frames"], step, on_sample)
    dt = time.perf_counter() - t0
    rows = [json.loads(l) for l in (out_dir / "labels.jsonl").read_text().splitlines()]
    assert [r["k"] for r in rows] == saved, "labels.jsonl rows and saved frames differ"
    assert not session.errors, session.errors
    # Hitbox in pixels of the full frame (x0,y0,x1,y1, origin top-left); scale by small/full for the small copy.
    with (out_dir / "labels.jsonl").open("w") as f:
        for r in rows:
            b = boxes[r["k"]]
            r["box"] = None if b is None else [round(b["x0"], 2), round(b["y0"], 2), round(b["x1"], 2), round(b["y1"], 2)]
            r["box_inside"] = None if b is None else round(b["inside"], 4)
            r["image"] = f"{r['k']:06d}.png"
            r["size"] = [session.width, session.height]
            f.write(json.dumps(r) + "\n")
    print(f"{out_dir}: {len(saved)} frames ({meta['track']}, {meta['frames']} ms), {dt:.1f} s, {len(saved) / dt:.1f} frames/s, {meta['frames'] / dt / 1000:.2f}x realtime", flush=True)
    return len(saved), dt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("controls", nargs="+", help="controls json files (data/drive.mjs output)")
    ap.add_argument("--out", default=str(REPO / "runs/frames"))
    ap.add_argument("--step", type=int, default=50, help="ms between saved frames")
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=288)
    ap.add_argument("--small", default="256x144", help="extra downscaled copy WxH, or 'none'")
    ap.add_argument("--debug", default=None, help="dir for hitbox overlays of the first 3 frames per episode")
    args = ap.parse_args()
    small = None if args.small == "none" else tuple(int(v) for v in args.small.split("x"))
    if args.debug:
        Path(args.debug).mkdir(parents=True, exist_ok=True)
    session = Session(args.width, args.height)
    total, seconds = 0, 0.0
    try:
        for f in args.controls:
            n, dt = render_episode(session, f, args.out, args.step, small, args.debug)
            total += n
            seconds += dt
    finally:
        session.close()
    print(f"total {total} frames in {seconds:.1f} s render time: {total / seconds:.1f} frames/s (boot excluded)")


if __name__ == "__main__":
    main()
