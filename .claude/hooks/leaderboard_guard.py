#!/usr/bin/env python3
"""PreToolUse guard: block anything that could submit to the Polytrack leaderboard.

Reads the hook JSON on stdin. Exit 2 blocks the tool call. Read-only GETs to
/v6/leaderboard and /v6/recordings stay allowed (the sim parity test uses them).
Self-check: python3 .claude/hooks/leaderboard_guard.py --selftest
"""
import json
import re
import sys

KODUB = re.compile(r"vps\.kodub\.com", re.I)
WRITE = re.compile(r"(-X\s*POST|--request\s+POST|-d\s|--data|-F\s|--form|method\s*[:=]\s*['\"]?POST|submitLeaderboard|\.post\()", re.I)


def blocked(text: str) -> bool:
    return bool(KODUB.search(text) and WRITE.search(text)) or "submitLeaderboard(" in text


def main() -> int:
    if sys.argv[1:] == ["--selftest"]:
        assert blocked('curl -X POST https://vps.kodub.com/v6/leaderboard -d "frames=1"')
        assert blocked("requests.post('https://vps.kodub.com/v6/leaderboard', data=d)")
        assert blocked("game.submitLeaderboard(token, nick)")
        assert not blocked('curl -s -H "Origin: x" "https://vps.kodub.com/v6/recordings?ids=1"')
        assert not blocked("curl -s https://vps.kodub.com/v6/leaderboard?trackId=a&amount=10")
        assert not blocked("curl -X POST localhost:8000/v1/systemone -d '{}'")
        print("leaderboard_guard selftest ok")
        return 0
    try:
        event = json.load(sys.stdin)
    except json.JSONDecodeError:
        return 0
    text = json.dumps(event.get("tool_input", {}))
    if blocked(text):
        print("Blocked: this project never submits to the Polytrack leaderboard (CLAUDE.md). "
              "Read-only GETs are fine.", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
