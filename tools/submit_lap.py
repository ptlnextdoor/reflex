#!/usr/bin/env python3
"""Submit one verified lap to the PolyTrack Summer 1 leaderboard, exactly as the game client does.

Owner-authorized (2026-10-23, nickname "aayushya.net"). Separate from the project's never-submit
guard on purpose: this script refuses to run unless --i-am-the-owner-and-authorize-this is given,
and it re-verifies the lap with the game's own Verify path before uploading anything.

Request shape copied from main.bundle.js submitLeaderboard() (0.6.3):
  POST https://vps.kodub.com/v6/leaderboard  (application/x-www-form-urlencoded)
  version=0.6.3&userToken=<64 hex>&nickname=..&carStyle=..&trackId=..&frames=N&recording=..
userToken: a new random 64-hex token, saved to ~/.config/polytrack/aayushya.json so later
submissions update the same leaderboard identity.

  python3 tools/submit_lap.py runs/server_asb_opt_final.json --nickname aayushya.net --dry-run
  python3 tools/submit_lap.py runs/server_asb_opt_final.json --nickname aayushya.net --i-am-the-owner-and-authorize-this
"""
import argparse
import hashlib
import json
import os
import secrets
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
API = "https://vps.kodub.com/v6/"
TRACK = "5803f9e963625804e3de3246d043dc7dde847aa32e991f7f7326b0453f1fa038"  # Summer 1, loop/track_id.mjs
HEADERS = {"Origin": "https://app-polytrack.kodub.com", "Referer": "https://app-polytrack.kodub.com/",
           "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/141.0 Safari/537.36"}
# Black body, white rims/accents: pattern 0, rims 0, exhaust 0, primary 0x111111, then game defaults.
CAR_STYLE = "AAAAABEREf___xMTE2ZmZg"
PROFILE = Path.home() / ".config/polytrack/aayushya.json"


def verify(rec: dict) -> bool:
    js = ("import fs from 'node:fs'; import {verifyRecording} from './sim/sim.js';"
          f"const j=JSON.parse(fs.readFileSync({json.dumps(str(rec['_path']))}));"
          "process.stdout.write(String(verifyRecording('summer1', j.recording, j.frames)));")
    out = subprocess.run(["node", "--input-type=module", "-e", js], cwd=REPO, capture_output=True, text=True, timeout=300)
    return out.stdout.strip() == "true"


def token() -> str:
    PROFILE.parent.mkdir(parents=True, exist_ok=True)
    if PROFILE.exists():
        return json.loads(PROFILE.read_text())["token"]
    t = hashlib.sha256(secrets.token_bytes(32)).hexdigest()
    PROFILE.write_text(json.dumps({"token": t}))
    os.chmod(PROFILE, 0o600)
    return t


def leaderboard_top(n=5):
    q = f"{API}leaderboard?version=0.6.3&trackId={TRACK}&skip=0&amount={n}&onlyVerified=false"
    with urllib.request.urlopen(urllib.request.Request(q, headers=HEADERS), timeout=30) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("lap", type=Path)
    ap.add_argument("--nickname", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--i-am-the-owner-and-authorize-this", dest="ok", action="store_true")
    a = ap.parse_args()
    rec = json.loads(a.lap.read_text())
    rec["_path"] = a.lap.resolve()
    frames = int(rec["frames"])
    assert len(rec["recording"]) < 10_000, "recording too large for the server"
    if not verify(rec):
        sys.exit("refusing: the game's Verify path rejects this lap")
    print(f"verified: {frames} frames ({frames / 1000:.3f} s) with the game's own Verify")
    body = urllib.parse.urlencode({"version": "0.6.3", "userToken": token(), "nickname": a.nickname,
                                   "carStyle": CAR_STYLE, "trackId": TRACK, "frames": str(frames),
                                   "recording": rec["recording"],
                                   # The game always sends this for official/community tracks (main.bundle.js setRecord:
                                   # n = "official"==category||"community"==category). Without it the upload is not queued
                                   # for verification.
                                   "onlyVerified": "true"})
    if a.dry_run or not a.ok:
        print("dry run: would POST", len(body), "bytes as", a.nickname, "(pass --i-am-the-owner-and-authorize-this to send)")
        return
    req = urllib.request.Request(API + "leaderboard", data=body.encode(), method="POST",
                                 headers={**HEADERS, "Content-Type": "application/x-www-form-urlencoded"})
    with urllib.request.urlopen(req, timeout=60) as r:
        print("server:", r.status, r.read().decode()[:300])
    top = leaderboard_top()
    for i, e in enumerate(top["entries"]):
        print(f"{i + 1}. {e['nickname']:20s} {e['frames'] / 1000:.3f} verified={e['verifiedState']}")


if __name__ == "__main__":
    main()
