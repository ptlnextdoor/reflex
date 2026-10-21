"""Record a sim-verified line playing in the REAL game, headless, frame-exact.

The player's car is created by the game with `carRecording: null` (live keys). This hooks
Worker.postMessage and swaps in a recording for the player's CreateCar message, so the
game's own physics worker plays the line tick-exactly (the same path the game uses for
ghosts and Verify), and the game's own renderer, HUD and timer show it. No key timing, no
jitter: what you see is exactly the byte-exact sim run.

    python3 loop/record_line.py opt/best.json            # our 22.111 s line
    python3 loop/record_line.py --wr                     # the human world record, same setup

ponytail: injects the recording instead of pressing keys. It's the honest way to SHOW a
TAS line; the live closed-loop bot (track D) is what presses keys from pixels.
"""
from __future__ import annotations

import argparse
import functools
import http.server
import json
import subprocess
import threading
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor/polytrack/0.6.3"


class Static(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


def load(candidate: str) -> tuple[str, int, str]:
    if candidate == "--wr":
        rec = json.loads((VENDOR / "recording-59372564.json").read_text())[0]
        return rec["recording"], rec["frames"], "wr_22123"
    j = json.loads(Path(candidate).read_text())
    name = f"{Path(candidate).stem}_{j.get('finishFrame')}"
    if "recording" in j:
        return j["recording"], j.get("frames") or j["finishFrame"], name
    # {controls} or {toggles}: encode to the game's recording format.
    out = subprocess.run(["node", "--input-type=module", "-e",
        "import fs from 'node:fs'; import {encodeRecording, controlsFromToggles} from './loop/recording.mjs';"
        "const j=JSON.parse(fs.readFileSync(process.argv[1]));"
        "const n=j.finishFrame??j.frames??(j.controls?j.controls.length:30000);"
        "const fromMasks=(m,n)=>{const c=new Uint8Array(n);let k=0,v=0;for(let t=0;t<n;t++){while(k<m.length&&m[k][0]<=t)v=m[k++][1];c[t]=v;}return c;};"
        # masks: [[tick, full 5-bit mask], ...] change points (learn/plan.mjs); toggles: [[tick, bit], ...] XOR flips.
        "const c=j.controls?Uint8Array.from(j.controls):j.masks?fromMasks(j.masks,n):controlsFromToggles(j.toggles,n);"
        "console.log(JSON.stringify({rec:encodeRecording(c),n:c.length}))", candidate],
        cwd=ROOT, capture_output=True, text=True, check=True).stdout
    r = json.loads(out)
    return r["rec"], r["n"], name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate", nargs="?", default="opt/best.json")
    ap.add_argument("--wr", action="store_true")
    ap.add_argument("--max-s", type=float, default=None, help="stop recording after this many seconds (for non-finishing attempts)")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    recording, frames, name = load("--wr" if a.wr else a.candidate)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Static, directory=str(VENDOR)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    origin = f"http://127.0.0.1:{server.server_port}"
    out_dir = ROOT / "runs/videos"
    out_dir.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True, args=["--use-angle=metal", "--enable-gpu", "--ignore-gpu-blocklist",
                                                    "--disable-background-timer-throttling",
                                                    "--disable-renderer-backgrounding"])
        ctx = b.new_context(viewport={"width": 1280, "height": 720}, service_workers="block",
                            record_video_dir=str(out_dir / "raw"), record_video_size={"width": 1280, "height": 720})
        ctx.route("**/*", lambda r: r.abort() if not r.request.url.startswith(origin + "/") else r.continue_())
        pg = ctx.new_page()
        pg.add_init_script(
            "globalThis.polytrackModConfiguration = {unblocked: true};"
            "try { localStorage.setItem('polytrack_v5_prod_startup_info', JSON.stringify({lastVersion: '0.6.3', isTutorialCompleted: true})); } catch {}"
            f"globalThis.__rec = {json.dumps(recording)}; globalThis.__finish = null; globalThis.__swapped = 0;"
            """
            const post = Worker.prototype.postMessage;
            Worker.prototype.postMessage = function (m, ...rest) {
              if (m && m.messageType === 3 && m.carRecording == null) { m = {...m, carRecording: globalThis.__rec}; globalThis.__swapped++; }
              if (m && m.messageType === 6) return;  // ignore live key control for the injected car
              return post.call(this, m, ...rest);
            };
            const W = globalThis.Worker;
            globalThis.Worker = class extends W { constructor(...a) { super(...a);
              this.addEventListener('message', e => { const d = e.data;
                if (d && d.messageType === 10 && d.carStateBuffers) for (const buf of d.carStateBuffers) {
                  const u = new Uint8Array(buf); if ((u[11] & 2) && globalThis.__finish === null) globalThis.__finish = u[12] | u[13] << 8 | u[14] << 16; } }); } };
            """)
        pg.goto(origin + "/", wait_until="domcontentloaded")
        # Headless audio decode fails sometimes; the overlay is cosmetic (physics unaffected), hide it for video.
        pg.add_style_tag(content="#error-screen{display:none!important}")
        tiles = "() => [...document.querySelectorAll('#ui .track')].some(t => t.offsetParent !== null)"
        play_main = ("() => { const b=[...document.querySelectorAll('#ui button')].find(b => b.querySelector('p')?.textContent.trim()==='Play' && b.offsetParent!==null); if(!b) return false; b.click(); return true; }")
        pg.wait_for_function(play_main, timeout=30000)  # clicks Play once it exists
        pg.wait_for_function(tiles, timeout=15000)
        pg.evaluate("() => { const t=[...document.querySelectorAll('#ui .track')].find(t => t.querySelector('.track-title p')?.textContent.trim()==='Summer 1'); t.querySelector('button').click(); }")
        pg.wait_for_function("() => [...document.querySelectorAll('#ui button.play')].some(b => b.offsetParent !== null)", timeout=15000)
        pg.evaluate("() => [...document.querySelectorAll('#ui button.play')].find(b => b.offsetParent!==null).click()")
        pg.wait_for_timeout(2000)
        swapped = pg.evaluate("globalThis.__swapped")
        pg.keyboard.down("ArrowUp")  # the game starts the clock on the first input
        pg.wait_for_timeout(50)
        pg.keyboard.up("ArrowUp")
        t0 = time.time()
        limit = frames / 1000 + (15 if a.max_s is None else 0)
        if a.max_s is not None:
            limit = min(limit, a.max_s)
        while pg.evaluate("globalThis.__finish") is None and time.time() - t0 < limit:
            pg.wait_for_timeout(250)
        pg.wait_for_timeout(3500)  # hold on the finish screen
        fin = pg.evaluate("globalThis.__finish")
        pg.screenshot(path=str(out_dir / f"{name}_finish.png"))
        video = Path(pg.video.path())
        ctx.close()
        b.close()
    server.shutdown()
    mp4 = a.out or out_dir / f"{name}.mp4"
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(video), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                    "-movflags", "+faststart", str(mp4)], check=True)
    print(json.dumps({"swapped_create_car": swapped, "game_finish_frames": fin, "expected": frames, "video": str(mp4)}))


if __name__ == "__main__":
    main()
