#!/bin/sh
# Open the reflex test browser (Chrome for Testing, separate from your Chrome)
# with the probe page. Run it, then drag the window to Desktop 2 once.
# eye captures it there by window id and sends keys by pid, so it never takes focus.
CFT="$HOME/Library/Caches/ms-playwright/chromium-1248/chrome-mac-arm64/Google Chrome for Testing.app"
PROF="$HOME/.cache/reflex/cft-profile"
PAGE="${1:-file://$(cd "$(dirname "$0")" && pwd)/probe/index.html}"
mkdir -p "$PROF"
open -n -a "$CFT" --args --user-data-dir="$PROF" --no-first-run --no-default-browser-check \
  --disable-background-timer-throttling --disable-renderer-backgrounding \
  --disable-backgrounding-occluded-windows --window-size=900,560 "$PAGE"
