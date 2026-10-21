#!/usr/bin/env python3
"""Local-only Chromium oracle. Python stays outside the simulation hot path."""

import base64
import functools
import http.server
import json
import re
import sys
from pathlib import Path
import threading
import zlib
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1] / "vendor/polytrack/0.6.3"


class Static(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_):
        pass


def sequence():
    raw = bytearray()
    for events in ([0], [2000], [], [], []):
        raw.extend(len(events).to_bytes(3, "little"))
        previous = 0
        for event in events:
            raw.extend((event - previous).to_bytes(3, "little"))
            previous = event
    return base64.urlsafe_b64encode(zlib.compress(raw)).decode().rstrip("=")


def main():
    track = sys.argv[1] if len(sys.argv) > 1 else "summer1"
    if (
        not re.fullmatch(r"[a-z0-9_-]+", track)
        or not (ROOT / f"tracks/official/{track}.track").is_file()
    ):
        raise ValueError(
            "Expected a fetched official track name, e.g. summer1 or winter1"
        )
    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(Static, directory=str(ROOT))
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    origin = f"http://127.0.0.1:{server.server_port}"
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(
                headless=True, args=["--enable-unsafe-swiftshader"]
            )
            context = browser.new_context(
                viewport={"width": 1280, "height": 900}, service_workers="block"
            )

            def route_request(route):
                if not route.request.url.startswith(origin + "/"):
                    return route.abort()
                # ponytail: metadata still says Summer 1. Only its local track slot changes.
                # The original main thread derives terrain and start pose from requested file.
                if route.request.url.endswith("/tracks/official/summer1.track"):
                    return route.fulfill(
                        path=str(ROOT / f"tracks/official/{track}.track")
                    )
                return route.continue_()

            context.route("**/*", route_request)
            page = context.new_page()
            page.on("pageerror", lambda e: print("PAGE ERROR:", e, flush=True))
            page.add_init_script("""
                globalThis.polytrackModConfiguration = {unblocked: true};
                globalThis.captured = [];
                const original = Worker.prototype.postMessage;
                Worker.prototype.postMessage = function(message, ...args) {
                    if (message.messageType === 0 || message.messageType === 3)
                        captured.push(structuredClone(message));
                    return original.call(this, message, ...args);
                };
            """)
            page.goto(origin + "/", wait_until="domcontentloaded")
            page.wait_for_function(
                "captured.some(x => x.messageType === 3) || [...document.querySelectorAll('button')].some(x => x.textContent === 'Ok') || document.body.innerText.includes('Hardware acceleration is disabled')"
            )
            warning = page.get_by_text("Ok", exact=True)
            if warning.is_visible():
                warning.click()
            page.wait_for_function("captured.some(x => x.messageType === 3)")
            fixture = page.evaluate("""JSON.stringify({
                init: captured.find(x => x.messageType === 0),
                create: captured.find(x => x.messageType === 3)
            }, (_, v) => ArrayBuffer.isView(v) ? {float32: Array.from(v)} : v)""")
            (ROOT / f"{track}.fixture.json").write_text(fixture)
            recordings = [("parity", sequence(), 5000)]
            if track == "summer1":
                recordings.append(
                    (
                        "recording",
                        json.loads((ROOT / "recording-59372564.json").read_text())[0][
                            "recording"
                        ],
                        22123,
                    )
                )
            for name, recording, frames in recordings:
                encoded = page.evaluate(
                    """async ({recording, frames}) => {
                    const init = captured.find(x => x.messageType === 0);
                    const create = captured.find(x => x.messageType === 3);
                    const worker = new Worker('simulation_worker.bundle.js');
                    const result = await new Promise((resolve, reject) => {
                        const timeout = setTimeout(() => reject(Error('Oracle timeout')), 30000);
                        let count = 0;
                        const bytes = new Uint8Array(frames * 227);
                        worker.onerror = e => reject(Error(e.message));
                        worker.onmessage = ({data}) => {
                            if (data.messageType !== 10) return;
                            for (const state of data.carStateBuffers) bytes.set(new Uint8Array(state), count++ * 227);
                            if (count === frames) {
                                clearTimeout(timeout);
                                let binary = '';
                                for (let offset = 0; offset < bytes.length; offset += 8192)
                                    binary += String.fromCharCode(...bytes.subarray(offset, offset + 8192));
                                resolve(btoa(binary));
                            }
                        };
                        worker.postMessage({...init, isRealtime: false});
                        worker.postMessage({...create, carRecording: recording});
                        worker.postMessage({messageType: 5, carId: create.carId, targetSimulationTimeFrames: frames});
                    });
                    worker.terminate();
                    return result;
                }""",
                    {"recording": recording, "frames": frames},
                )
                prefix = "" if track == "summer1" else f"{track}."
                (ROOT / f"{prefix}{name}.browser.bin").write_bytes(
                    base64.b64decode(encoded)
                )
                print(f"Chromium oracle: {name} {frames} states", flush=True)
            print(
                f"Captured original Init/CreateCar for {track}, external network blocked",
                flush=True,
            )
            browser.close()
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
