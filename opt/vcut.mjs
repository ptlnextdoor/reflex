// Owner-described U-turn cuts (summer1, 0.6.3): at the U-turn entry after cp2 steer right over the
// right rail (toward -z) for a sharp V instead of the wide banked loop, then cut through before the
// straight into cp3 instead of the wide turn. Pure-pursuit through a grid of apex waypoints at full
// throttle, starting from the polished WR line at tick T0; reports the cp3 tick vs the WR (14936).
//   node opt/vcut.mjs --shard k/n            -> one JSON row per candidate that reaches cp3
//   node opt/vcut.mjs --one '<json cand>' --out f.json   -> controls for that candidate
import fs from 'node:fs';
const {Car, decodeState} = await import('../ctrl/sim.mjs');
const {decodeRecording} = await import('../sim/sim.js');
const arg = (n, d) => { const i = process.argv.indexOf(n); return i < 0 ? d : process.argv[i + 1]; };
const j = JSON.parse(fs.readFileSync(arg('--from', 'runs/server/wr_opt.json'), 'utf8'));
const line = Array.from(decodeRecording(j.recording, j.frames));
const CP3 = [-100, -80], AFTER = [-100, -30];

function attempt(c, keep = false) {
  const car = new Car('summer1');
  for (let t = 0; t < c.t0; t++) car.step(line[t]);
  const wps = [...c.wps, CP3, AFTER];
  const ctrl = keep ? line.slice(0, c.t0) : null;
  return run(car, c, wps, ctrl);
}

function run(car, c, wps, ctrl) {
  let wi = 0, mask = 1, s = null;
  for (let t = c.t0; t < c.t0 + 9000; t++) {
    if (s && (t - c.t0) % c.D === 0) {
      const p = s.position, q = s.quaternion;
      while (wi < wps.length - 1 && Math.hypot(wps[wi][0] - p.x, wps[wi][1] - p.z) < c.r) wi++;
      const fx = 2 * (q.x * q.z + q.w * q.y), fz = 1 - 2 * (q.x * q.x + q.y * q.y);
      const dx = wps[wi][0] - p.x, dz = wps[wi][1] - p.z;
      const ang = Math.atan2(fx * dz - fz * dx, fx * dx + fz * dz) * 180 / Math.PI; // >0 target on the right
      const steer = ang > c.dead ? 2 : ang < -c.dead ? 8 : 0;
      const brake = c.brakeAng && Math.abs(ang) > c.brakeAng && s.speedKmh > c.brakeV;
      mask = (brake ? 4 : 1) | steer;
    }
    if (ctrl) ctrl.push(mask);
    const raw = car.step(mask);
    s = decodeState(raw);
    if (s.position.y < -10) return null;
    if (s.nextCheckpointIndex >= 3) return {tick: t + 1, ctrl, v: Math.round(s.speedKmh), y: +s.position.y.toFixed(1)};
  }
  return null;
}

if (arg('--one')) {
  const c = JSON.parse(arg('--one'));
  const r = attempt(c, true);
  console.log(r ? `cp3 at ${r.tick} (WR 14936) v ${r.v}` : 'no cp3');
  if (r && arg('--out')) fs.writeFileSync(arg('--out'), JSON.stringify({t0: c.t0, cp3: r.tick, controls: r.ctrl}));
} else {
  const [k, n] = arg('--shard', '0/1').split('/').map(Number);
  const grid = [];
  for (const t0 of [9300, 9600, 9900])
    for (const a1 of [[-200, -40], [-215, -60], [-230, -80], [-200, -70]])
      for (const a2 of [[-190, -125], [-160, -130], [-140, -120], [-120, -110], null])
        for (const r of [8, 15]) for (const D of [10, 30]) for (const dead of [2, 6])
          for (const [brakeAng, brakeV] of [[0, 0], [50, 200], [35, 150]])
            grid.push({t0, wps: a2 ? [a1, a2] : [a1], r, D, dead, brakeAng, brakeV});
  let best = Infinity;
  for (let i = k; i < grid.length; i += n) {
    const r = attempt(grid[i]);
    if (r) { if (r.tick < best) best = r.tick; console.log(JSON.stringify({...grid[i], cp3: r.tick, gain: 14936 - r.tick, v: r.v, y: r.y})); }
  }
  console.log(`DONE shard ${k}/${n} of ${grid.length} best ${best}`);
}
