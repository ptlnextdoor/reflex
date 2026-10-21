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
     render (no in-between rendering), and differs from the previous sampled frame.
Prints every checked frame. Exit 0 only when every assert holds.

Usage: python3 data/check.py [runs/drives/summer1_s1.json] [--frames 50,500,1000,1500,2000,2500,3000]
"""
import argparse
import json
import struct
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from render import REPO, STATE_BYTES, Session, make_labels  # noqa: E402

MASS_OFFSET = 0.6


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
    worst = 0.0
    prev_png = None
    try:
        session.enter_track(meta["track"])
        assert session.load_states(out / "states.bin") == meta["frames"]
        step = 50

        def on_sample(k, car_id):
            nonlocal checked, worst, prev_png
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
            worst = max(worst, err, qerr, merr)
            checked += 1
            print(f"k={k:5d} page=({page_pos[0]:.4f},{page_pos[1]:.4f},{page_pos[2]:.4f}) sim=({pos[0]:.4f},{pos[1]:.4f},{pos[2]:.4f}) "
                  f"pos_err={err:.1e} quat_err={qerr:.1e} mesh_err={merr:.1e} speed={p['speedKmh']:.1f} km/h "
                  f"held={lab['mask_held']} next={lab['steer']}/{lab['throttle']}", flush=True)

        session.run(max(wanted), step, on_sample)
    finally:
        session.close()
    assert checked == len(wanted), (checked, wanted)
    print(f"check PASS: {checked} frames, page car state == Node sim state, worst error {worst:.2e} (tol {args.tol}), "
          f"chassis mesh at position + {MASS_OFFSET} m up, screenshots deterministic; track={meta['track']} source={controls.name}")


if __name__ == "__main__":
    main()
