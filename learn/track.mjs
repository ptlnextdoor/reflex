// Track structure from the track file the game loads: start pose, checkpoint and finish
// gates in world coordinates. This is what a new player sees on screen, nothing more.
// Runs the game's own track parser (patched in memory to export it) in a throwaway vm.
//   node learn/track.mjs   -> prints the structure and writes learn/track.json
import fs from 'node:fs';
import vm from 'node:vm';
import {fileURLToPath} from 'node:url';

const vendor = fileURLToPath(new URL('../vendor/polytrack/0.6.3/', import.meta.url));
const PART_SIZE = 5; // No.partSize=5 in the bundle

export function loadTrack(trackName = 'summer1') {
  const fixture = JSON.parse(fs.readFileSync(`${vendor}${trackName}.fixture.json`, 'utf8'));
  let source = fs.readFileSync(`${vendor}simulation_worker.bundle.js`, 'utf8');
  const patch = (from, to) => { if (!source.includes(from)) throw Error(`patch point missing: ${from}`); source = source.replace(from, to); };
  // Export the track class (Ko), the part-config lookup (Ao) and the rotation table (io).
  patch('const Ko=class{', 'const Ko=self.__Track=class{');
  patch('function Ao(t){const e=co.get(t);', 'self.__part=Ao;function Ao(t){const e=co.get(t);');
  patch('function io(t,e){return eo[e][t].clone()}', 'function io(t,e){return eo[e][t].clone()}self.__rot=io;');
  // Stop before the worker boots the physics: only the parser is needed here.
  patch('importScripts("lib/polytrack_physics.js")', '(()=>{throw new Error("__stop__")})()');
  const context = vm.createContext({
    console, performance, TextDecoder, TextEncoder, atob, btoa, Uint8Array, Float32Array, ArrayBuffer, DataView,
    setInterval() {}, clearInterval() {}, setTimeout() {}, clearTimeout() {}, postMessage() {}, addEventListener() {},
    onmessage: null, location: {href: `file://${vendor}simulation_worker.bundle.js`},
  });
  context.self = context;
  try { vm.runInContext(source, context, {filename: 'simulation_worker.bundle.js'}); } catch (e) { if (!/__stop__/.test(String(e))) throw e; }
  const track = context.__Track.fromSaveString(fixture.create.trackData);
  if (!track) throw Error('track parse failed');
  const start = track.getStartTransform();
  const parts = [], checkpoints = [], finishes = [];
  track.forEachPart((x, y, z, id, rotation, axis, color, checkpointOrder) => {
    const cfg = context.__part(id);
    const q = context.__rot(rotation, axis);
    parts.push({x, y, z, id, rotation, axis, q: [q._x, q._y, q._z, q._w]});
    if (!cfg.detector) return;
    const rot = v => rotate(v, q);
    const g = {
      center: add([x * PART_SIZE, y * PART_SIZE, z * PART_SIZE], rot(cfg.detector.center)),
      normal: rot([0, 0, 1]), across: rot([1, 0, 0]), up: rot([0, 1, 0]), size: cfg.detector.size,
      order: checkpointOrder ?? null, partId: id,
    };
    (cfg.detector.type === 0 ? checkpoints : finishes).push(g);
  });
  checkpoints.sort((a, b) => a.order - b.order);
  return {
    start: {position: [start.position.x, start.position.y, start.position.z], quaternion: [start.quaternion._x, start.quaternion._y, start.quaternion._z, start.quaternion._w]},
    checkpoints, finishes, parts, partSize: PART_SIZE,
  };
}

function rotate([x, y, z], q) {
  const qx = q._x, qy = q._y, qz = q._z, qw = q._w;
  const ix = qw * x + qy * z - qz * y, iy = qw * y + qz * x - qx * z, iz = qw * z + qx * y - qy * x, iw = -qx * x - qy * y - qz * z;
  return [ix * qw + iw * -qx + iy * -qz - iz * -qy, iy * qw + iw * -qy + iz * -qx - ix * -qz, iz * qw + iw * -qz + ix * -qy - iy * -qx];
}
const add = (a, b) => [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
const f2 = v => v.map(x => x.toFixed(2)).join(',');

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  const t = loadTrack(process.argv[2] ?? 'summer1');
  console.log('start', JSON.stringify(t.start));
  for (const c of t.checkpoints) console.log('checkpoint', c.order, 'part', c.partId, 'center', f2(c.center), 'normal', f2(c.normal), 'across', f2(c.across), 'size', c.size);
  for (const f of t.finishes) console.log('finish part', f.partId, 'center', f2(f.center), 'normal', f2(f.normal), 'across', f2(f.across), 'size', f.size);
  console.log('parts', t.parts.length);
  fs.writeFileSync(new URL('./track.json', import.meta.url), JSON.stringify(t));
}
