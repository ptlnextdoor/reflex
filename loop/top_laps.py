"""Read-only leaderboard probe: boot the real game headless, open Summer 1's leaderboard, log
the GET requests the game itself makes to vps.kodub.com, and fetch the top-N recordings.

    python3 loop/top_laps.py [--n 10] [--track summer1]
Writes vendor/polytrack/0.6.3/top/<track>_leaderboard.json and recording-<id>.json per entry.

Safety: every request that is not a GET to /v6/leaderboard or /v6/recordings on vps.kodub.com,
or to the local static server, is aborted. Nothing is ever submitted (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import functools
import http.server
import json
import threading
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "vendor/polytrack/0.6.3"
OUT = ROOT / "top"
API = "https://vps.kodub.com/v6/"


class Static(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


def allowed(req, origin):
    u = req.url
    if u.startswith(origin + "/"):
        return True
    return req.method == "GET" and (u.startswith(API + "leaderboard?") or u.startswith(API + "recordings?"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--track", default="Summer 1")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Static, directory=str(ROOT)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    origin = f"http://127.0.0.1:{server.server_port}"
    seen = []
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True, args=["--use-angle=metal", "--enable-gpu", "--ignore-gpu-blocklist"])
        ctx = b.new_context(viewport={"width": 1280, "height": 900}, service_workers="block")

        def route(r):
            if not allowed(r.request, origin):
                return r.abort()
            if r.request.url.startswith(API):
                seen.append(r.request.url)
            return r.continue_()

        ctx.route("**/*", route)
        pg = ctx.new_page()
        pg.add_init_script("globalThis.polytrackModConfiguration = {unblocked: true};")
        pg.goto(origin + "/", wait_until="domcontentloaded")
        pg.wait_for_function("!document.body.innerText.includes('Loading')", timeout=60000)
        ok = pg.get_by_text("Ok", exact=True)
        if ok.count() and ok.first.is_visible():
            ok.first.click()
        pg.wait_for_function("() => [...document.querySelectorAll('#ui button')].some(b => b.querySelector('p')?.textContent.trim()==='Play' && b.offsetParent!==null)", timeout=30000)
        pg.evaluate("() => [...document.querySelectorAll('#ui button')].find(b => b.querySelector('p')?.textContent.trim()==='Play' && b.offsetParent!==null).click()")
        pg.wait_for_function("() => [...document.querySelectorAll('#ui .track')].some(t => t.offsetParent !== null)", timeout=15000)
        pg.evaluate("""(title) => { const t=[...document.querySelectorAll('#ui .track')].find(t => t.querySelector('.track-title p')?.textContent.trim()===title); t.querySelector('button').click(); }""", a.track)
        # The track page shows its leaderboard; give the request time to go out.
        for _ in range(40):
            if any("leaderboard?" in u for u in seen):
                break
            pg.wait_for_timeout(500)
        if not any("leaderboard?" in u for u in seen):
            pg.screenshot(path=str(OUT / "probe.png"))
            names = pg.evaluate("() => [...document.querySelectorAll('#ui button')].filter(b=>b.offsetParent).map(b=>b.textContent.trim()).slice(0,40)")
            print("visible buttons:", names)
            for label in ("Leaderboard", "Leaderboards"):
                el = pg.get_by_text(label, exact=True)
                if el.count():
                    el.first.click(); pg.wait_for_timeout(3000)
        b.close()
    server.shutdown()
    lb = [u for u in seen if "leaderboard?" in u]
    if not lb:
        raise SystemExit(f"game made no leaderboard request; saw {seen}")
    url = lb[0]
    base = url.split("&skip=")[0]
    tail = "&onlyVerified=" + url.split("&onlyVerified=")[1].split("&")[0]
    req = urllib.request.Request(f"{base}&skip=0&amount={a.n}{tail}", headers={"Origin": "https://app-polytrack.kodub.com"})
    data = json.loads(urllib.request.urlopen(req, timeout=20).read())
    (OUT / "summer1_leaderboard.json").write_text(json.dumps({"url": url, **data}, indent=1))
    entries = data.get("entries", [])
    print("trackId url:", base)
    for e in entries:
        print(json.dumps({k: e.get(k) for k in ("id", "name", "frames", "verifiedState", "position")}))
    ids = [str(e["id"]) for e in entries]
    req = urllib.request.Request(f"{API}recordings?version=0.6.3&ids={','.join(ids)}", headers={"Origin": "https://app-polytrack.kodub.com"})
    recs = json.loads(urllib.request.urlopen(req, timeout=30).read())
    for e, r in zip(entries, recs):
        (OUT / f"recording-{e['id']}.json").write_text(json.dumps([r]))
    print(f"saved {len(recs)} recordings to {OUT}")


if __name__ == "__main__":
    main()
