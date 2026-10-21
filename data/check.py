#!/usr/bin/env python3
"""Proof that rendered frames are exact: what the game holds at render time equals the sim.

Boots the same headless pipeline as data/render.py on one controls file, feeds the sim
states tick by tick at the live cadence, and at every sampled frame k asserts, against
the live page (not a file written earlier):
  1. the player car's frame counter == k,
  2. car.getPosition() == decodeState(k-1).position, per axis, within --tol (default 1e-3),
  3. car.getQuaternion() == the sim quaternion within --tol,
  4. the drawn chassis mesh (three.js "Body" matrixWorld) sits at position + massOffset
     (0.6 m along the car's up axis, exactly as the game's Car.update composes it),
  5. the rendered PNG for frame k is pixel-identical to a second screenshot of the same
     render (no in-between rendering), and differs from the previous sampled frame,
  6. the projected hitbox (every car mesh vertex through the render's view/projection
     matrices) encloses, within 1 px, every pixel of the car's silhouette (the diff between
     re-renders of the same state with the car drawn on top, no depth test, and hidden),
     and is tight (silhouette reaches each box edge within 2 px). Independent of the random
     paint. Silhouettes come from the WebGL canvas alone (the DOM HUD can cover the car).
     The car is then restored and the frame re-rendered to the identical PNG.
Prints every checked frame. Exit 0 only when every assert holds.

Usage: python3 data/check.py [runs/drives/summer1_s1.json] [--frames 50,500,1000,1500,2000,2500,3000]
"""
import argparse
import io
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render import REPO, STATE_BYTES, Session, make_labels  # noqa: E402

MASS_OFFSET = 0.6


def car_pixels(png_with, png_without):
    """Bounding box and count of pixels that differ between the car-on-top (no depth test, no
    cast shadow) and car-hidden canvas renders."""
    a = np.asarray(Image.open(io.BytesIO(png_with)).convert("RGB")).astype(int)
    b = np.asarray(Image.open(io.BytesIO(png_without)).convert("RGB")).astype(int)
    diff = np.abs(a - b).max(axis=2) > 8
    if diff.sum() == 0:
        return None
    ys, xs = np.nonzero(diff)
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()), int(diff.sum())


def decode(raw, k):
    """Position and quaternion of sim frame k (state after k ticks), same layout as sim.js decodeState."""
    base = (k - 1) * STATE_BYTES
    flags = raw[base + 11]
    off = (15 if flags & 2 else 12) + 2
    frames = raw[base + 4] | raw[base + 5] << 8 | raw[base + 6] << 16
    pos = struct.unpack_from("<3f", raw, base + off)
    quat = struct.unpack_from("<4f", raw, base + off + 12)
    return frames, pos, quat


def rotate_up(q):
    """Car up axis (0,1,0) rotated by quaternion (x,y,z,w)."""
    x, y, z, w = q
    return (2 * (x * y - w * z), 1 - 2 * (x * x + z * z), 2 * (y * z + w * x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("controls", nargs="?", default=str(REPO / "runs/drives/summer1_s1.json"))
    ap.add_argument("--frames", default="50,500,1000,1500,2000,2500,3000")
    ap.add_argument("--tol", type=float, default=1e-3)
    args = ap.parse_args()
    controls = Path(args.controls)
    if not controls.exists():
        subprocess.run(["node", str(REPO / "data/drive.mjs"), str(controls.parent), "summer1", "1-1", "20000"], check=True, cwd=REPO)
    wanted = [int(v) for v in args.frames.split(",")]
    out = Path(tempfile.mkdtemp(prefix="reflex-check-"))
    meta = make_labels(controls, out, 50)
    raw = (out / "states.bin").read_bytes()
    wanted = [k for k in wanted if k <= meta["frames"]]
    assert len(wanted) >= 5, f"need >= 5 sampled frames inside the episode ({meta['frames']} ms)"
    labels = {r["k"]: r for r in map(json.loads, (out / "labels.jsonl").read_text().splitlines())}

    session = Session()
    checked = 0
    painted = 0
    worst = 0.0
    prev_png = None
    try:
        session.enter_track(meta["track"])
        assert session.load_states(out / "states.bin") == meta["frames"]
        step = 50

        def on_sample(k, car_id):
            nonlocal checked, worst, prev_png, painted
            if k not in wanted:
                return
            frames, pos, quat = decode(raw, k)
            assert frames == k, (frames, k)
            p = session.probe(car_id)
            assert p["frames"] == k, f"page frame {p['frames']} != {k}"
            page_pos = (p["position"]["x"], p["position"]["y"], p["position"]["z"])
            page_quat = (p["quaternion"]["x"], p["quaternion"]["y"], p["quaternion"]["z"], p["quaternion"]["w"])
            err = max(abs(a - b) for a, b in zip(page_pos, pos))
            qerr = max(abs(a - b) for a, b in zip(page_quat, quat))
            assert err < args.tol, f"k={k} position error {err}"
            assert qerr < args.tol, f"k={k} quaternion error {qerr}"
            lab = labels[k]
            assert abs(lab["position"]["x"] - pos[0]) < 1e-9 and lab["speedKmh"] == struct.unpack_from("<f", raw, (k - 1) * STATE_BYTES + 7)[0]
            assert len(p["bodies"]) == 1, f"expected one drawn chassis, got {len(p['bodies'])}"
            m = p["bodies"][0]
            up = rotate_up(quat)
            expect = tuple(pos[i] + MASS_OFFSET * up[i] for i in range(3))
            merr = max(abs(m[12 + i] - expect[i]) for i in range(3))
            assert merr < args.tol, f"k={k} chassis mesh error {merr}"
            png = session.screenshot()
            again = session.screenshot()
            assert png == again, f"k={k} two screenshots of the same render differ"
            assert png != prev_png, f"k={k} frame identical to the previous sample"
            prev_png = png
            box = session.box()
            assert box and box["vertices"] == 2374 and box["meshes"] == 7, box
            # Silhouette from the WebGL canvas alone (the DOM HUD can cover the car's bottom edge).
            session.page.evaluate("() => __rf.rerender('hidden')")
            hidden = session.canvas()
            session.page.evaluate("() => __rf.rerender('ontop')")
            ontop = session.canvas()
            session.page.evaluate("() => __rf.rerender('normal')")
            shown = session.screenshot()
            assert shown == png, f"k={k} re-render with the car shown again differs from the frame"
            paint = car_pixels(ontop, hidden)
            assert paint, f"k={k} hiding the car changed no pixels"
            x0, y0, x1, y1, npx = paint
            assert box["x0"] - 1 <= x0 and box["y0"] - 1 <= y0 and x1 <= box["x1"] + 1 and y1 <= box["y1"] + 1, \
                f"k={k} car pixels {paint} escape projected box {box}"
            # And the box is tight: the silhouette reaches each edge within 2 px (vertex vs raster).
            assert x0 - box["x0"] < 2 and y0 - box["y0"] < 2 and box["x1"] - x1 < 2 and box["y1"] - y1 < 2, \
                f"k={k} projected box {box} is loose around car pixels {paint}"
            painted += 1
            worst = max(worst, err, qerr, merr)
            checked += 1
            print(f"k={k:5d} page=({page_pos[0]:.4f},{page_pos[1]:.4f},{page_pos[2]:.4f}) sim=({pos[0]:.4f},{pos[1]:.4f},{pos[2]:.4f}) "
                  f"pos_err={err:.1e} quat_err={qerr:.1e} mesh_err={merr:.1e} speed={p['speedKmh']:.1f} km/h "
                  f"held={lab['mask_held']} next={lab['steer']}/{lab['throttle']} "
                  f"box=({box['x0']:.0f},{box['y0']:.0f},{box['x1']:.0f},{box['y1']:.0f}) car_px={paint[:4]} n={paint[4]}", flush=True)

        session.run(max(wanted), step, on_sample)
    finally:
        session.close()
    assert checked == len(wanted), (checked, wanted)
    assert painted == checked
    print(f"check PASS: {checked} frames, page car state == Node sim state, worst error {worst:.2e} (tol {args.tol}), "
          f"chassis mesh at position + {MASS_OFFSET} m up, screenshots deterministic, hitbox encloses every car pixel; "
          f"track={meta['track']} source={controls.name}")


if __name__ == "__main__":
    main()
