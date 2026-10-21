# Headless Polytrack 0.6.3
1. Requires Node 20+, Python Playwright and Chromium (`python3 -m playwright install chromium`).
2. Fetch static game assets: `sim/fetch.sh`. All game files and captures stay ignored in `vendor/`.
3. Cache authorized read-only seed once: `curl -fsS -H 'Origin: https://app-polytrack.kodub.com' 'https://vps.kodub.com/v6/recordings?version=0.6.3&ids=59372564' -o vendor/polytrack/0.6.3/recording-59372564.json`.
4. Run `node sim/check.js`: fresh local Chromium capture, all-byte parity at 5 s and 22.123 s, determinism, Verify, edges, throughput.
5. `import {run, decodeRecording} from './sim/sim.js'; const {finishFrame, states} = run('summer1', new Uint8Array(5000).fill(1));`
6. `states.at(999)` decodes tick 1000. `states.raw` holds 227 bytes/tick. `for (const state of states)` decodes lazily, outside allocation-free stepping.
7. Masks: up=1, right=2, down=4, left=8, reset=16. Tick zero is first physics update, matching original Verify.
8. `decodeRecording(recording, metadata.frames)` extends final held inputs. Without duration, output ends after last toggle, not finish.
9. Other official tracks: `python3 sim/browser.py winter1`, then `run('winter1', controls)`. One synchronous caller/engine, reuse across runs, worker_threads for parallel search. Never submit times.
