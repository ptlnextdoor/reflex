#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
root=vendor/polytrack/0.6.3
base=https://app-polytrack.kodub.com/0.6.3
mkdir -p "$root"
fetch() {
  mkdir -p "$root/$(dirname "$1")"
  curl --fail --silent --show-error --location --retry 2 "$base/$1" -o "$root/$1"
}
for file in index.html main.bundle.js simulation_worker.bundle.js lib/polytrack_physics.js polytrack_physics.wasm error_screen.bundle.js forced_square.woff2 lib/draco/draco_wasm_wrapper.js lib/draco/draco_decoder.wasm; do
  fetch "$file"
done
# Only literal static assets on the versioned origin. No API discovery or calls.
node --input-type=module <<'JS' | while IFS= read -r file; do fetch "$file"; done
import fs from 'node:fs';
const bundle = fs.readFileSync('vendor/polytrack/0.6.3/main.bundle.js', 'utf8');
const assets = new Set();
for (const [, path] of bundle.matchAll(/"((?:models|images|audio|tracks\/official)\/[\w/.+-]+\.(?:glb|png|jpg|svg|ogg|mp3|track))"/g)) assets.add(path);
for (const [, path] of bundle.matchAll(/"\.\/([\w/-]+\.(?:svg|jpg|png))"/g)) assets.add(`images/${path}`);
console.log([...assets].sort().join('\n'));
JS
printf 'Fetched static Polytrack 0.6.3 assets into %s\n' "$root"
