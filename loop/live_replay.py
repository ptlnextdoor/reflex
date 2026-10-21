"""Track D v0: open-loop live replay in the REAL game, headless, no visible window.

Boots the real Polytrack (Metal ANGLE, headless) on a local server, enters Summer 1,
and plays a candidate input line by sending keydown/keyup to the page at wall-clock
times. Then reads the game's own finish time from the page. This measures how much
real-time key delivery jitter costs vs the sim (the gap track D must close).

    python3 loop/live_replay.py --wr --runs 3
    python3 loop/live_replay.py opt/best.json --runs 5

ponytail: open loop, key timing from Python wall clock via CDP. Ceiling: CDP dispatch
jitter (~1 ms). Upgrade: inject inputs through the worker's ControlCar message with exact
frame stamps, or close the loop on pixels (track C/D).
"""
from __future__ import annotations

import argparse
import functools
import http.server
import json
import re
import subprocess
import threading
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "vendor/polytrack/0.6.3"
KEYS = [(1, "ArrowUp"), (2, "ArrowRight"), (4, "ArrowDown"), (8, "ArrowLeft"), (16, "KeyR")]


class Static(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


def toggles_for(candidate: str) -> list[tuple[int, int]]:
    out = subprocess.run(["node", "loop/toggles.mjs", candidate], cwd=ROOT,
                         capture_output=True, text=True, check=True).stdout
    return [tuple(x) for x in json.loads(out)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate", nargs="?", default="--wr")
    ap.add_argument("--wr", action="store_true")
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--shot", action="store_true", help="save a screenshot after each run to runs/live/")
    a = ap.parse_args()
    cand = "--wr" if a.wr else a.candidate
    toggles = toggles_for(cand)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Static, directory=str(VENDOR)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    origin = f"http://127.0.0.1:{server.server_port}"
    (ROOT / "runs/live").mkdir(parents=True, exist_ok=True)
    results = []
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True, args=["--use-angle=metal", "--enable-gpu", "--ignore-gpu-blocklist",
                                                    "--disable-background-timer-throttling",
                                                    "--disable-renderer-backgrounding"])
        ctx = b.new_context(viewport={"width": 960, "height": 540}, service_workers="block")
        ctx.route("**/*", lambda r: r.abort() if not r.request.url.startswith(origin + "/") else r.continue_())
        pg = ctx.new_page()
        pg.add_init_script("""
            globalThis.polytrackModConfiguration = {unblocked: true};
            globalThis.__finish = null;
            const W = globalThis.Worker;
            globalThis.Worker = class extends W { constructor(...a) { super(...a);
              this.addEventListener('message', e => { const d = e.data;
                if (d && d.messageType === 10 && d.carStateBuffers) for (const buf of d.carStateBuffers) {
                  const u = new Uint8Array(buf); if (u[11] & 2) globalThis.__finish = u[12] | u[13] << 8 | u[14] << 16; } }); } };
        """)
        pg.goto(origin + "/", wait_until="domcontentloaded")
        pg.wait_for_timeout(6000)
        # ponytail: menu navigation by visible text; adjust if the UI changes.
        for label in ["Play", "Official tracks", "Summer 1"]:
            loc = pg.get_by_text(label, exact=True)
            if loc.count():
                loc.first.click()
                pg.wait_for_timeout(1500)
        pg.screenshot(path=str(ROOT / "runs/live/menu_state.png"))
        for run in range(a.runs):
            pg.evaluate("globalThis.__finish = null")
            held = 0
            t0 = time.perf_counter()
            for tick, mask in toggles:
                target = t0 + tick / 1000.0
                while (d := target - time.perf_counter()) > 0.002:
                    time.sleep(d - 0.0015)
                while time.perf_counter() < target:
                    pass
                for bit, key in KEYS:
                    if (mask & bit) and not (held & bit):
                        pg.keyboard.down(key)
                    elif not (mask & bit) and (held & bit):
                        pg.keyboard.up(key)
                held = mask
            pg.wait_for_timeout(1500)
            for bit, key in KEYS:
                if held & bit:
                    pg.keyboard.up(key)
            fin = pg.evaluate("globalThis.__finish")
            if a.shot:
                pg.screenshot(path=str(ROOT / f"runs/live/run{run}.png"))
            results.append(fin)
            print(json.dumps({"run": run, "finish_frames": fin}), flush=True)
            pg.keyboard.press("KeyT")  # VehicleStartReset (R = checkpoint reset)
            pg.wait_for_timeout(1500)
        b.close()
    server.shutdown()
    print(json.dumps({"candidate": cand, "toggles": len(toggles), "finishes": results}))


if __name__ == "__main__":
    main()
