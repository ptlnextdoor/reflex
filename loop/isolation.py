#!/usr/bin/env python3
"""Isolation check: Reflex-1's code may not touch human runs or leaderboard data.

Scans the agent-owned dirs (learn/, ctrl/, data/, student/ by default). Fails (exit 1) on any
reference to the human WR recording, the leaderboard API, ghosts, or the opt/ line that
was derived from the WR. Run before logging any Reflex-1 result.

    python3 loop/isolation.py            # default dirs
    python3 loop/isolation.py learn ctrl
    python3 loop/isolation.py --selftest
"""
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT = ["learn", "ctrl", "data", "student"]
BANNED = [
    (r"recording-\d+", "human recording file"),
    (r"59372564", "WR recording id"),
    (r"vps\.kodub\.com", "leaderboard API"),
    (r"\bleaderboard\b", "leaderboard"),
    (r"\bghost", "ghost car data"),
    (r"opt/(best|robust)\.json", "WR-derived line"),
    (r"\.\./opt/|from ['\"]\.\./opt", "WR-derived optimizer output"),
]
EXT = {".mjs", ".js", ".py", ".json", ".sh", ".ts"}


def scan(dirs):
    hits = []
    for d in dirs:
        for p in (ROOT / d).rglob("*"):
            if p.suffix not in EXT or not p.is_file() or "scratch" in p.parts or "node_modules" in p.parts:
                continue
            text = p.read_text(errors="ignore")
            for pat, why in BANNED:
                for m in re.finditer(pat, text):
                    line = text.count("\n", 0, m.start()) + 1
                    hits.append(f"{p.relative_to(ROOT)}:{line}: {why} ({m.group(0)})")
    return hits


def main():
    if sys.argv[1:] == ["--selftest"]:
        assert any(re.search(p, "load recording-59372564.json") for p, _ in BANNED)
        assert any(re.search(p, "import x from '../opt/best.json'") for p, _ in BANNED)
        assert not any(re.search(p, "const lap = ownBest.controls") for p, _ in BANNED)
        print("isolation selftest ok")
        return 0
    dirs = [a for a in sys.argv[1:] if not a.startswith("-")] or DEFAULT
    hits = scan([d for d in dirs if (ROOT / d).exists()])
    if hits:
        print("ISOLATION FAIL:\n  " + "\n  ".join(hits))
        return 1
    print(f"isolation ok: {', '.join(dirs)} reference no human runs or leaderboard data")
    return 0


if __name__ == "__main__":
    sys.exit(main())
