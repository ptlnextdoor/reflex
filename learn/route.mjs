// Drivable-tile map and per-gate distance fields from the track file alone (what a new
// player sees). Tiles are 5 m cubes; parts are 4x4 tiles placed on a 4-tile grid. The game's
// own part table (tile footprints per part) is read through learn/track.mjs's parser patch.
//
// progress(gate, x, y, z): road distance in metres from the car to the given gate's tiles
// along the drivable graph (4-neighbour steps, slopes via tile levels, plus jump edges across
// gaps of up to JUMP cells that land at most 2 levels higher). Off-road cells pay a penalty.
//   node learn/route.mjs    -> prints field sizes and a sanity check along the gates
import fs from 'node:fs';
import vm from 'node:vm';
import {fileURLToPath} from 'node:url';

const TILE = 5, JUMP = 16;
const vendor = fileURLToPath(new URL('../vendor/polytrack/0.6.3/', import.meta.url));
const track = JSON.parse(fs.readFileSync(new URL('./track.json', import.meta.url), 'utf8'));
const DRIVABLE = new Set([0, 1, 2, 3, 4]); // Special, Road, RoadTurns, RoadWide, Plane
const WALLS = new Set([26, 27]);           // PlaneWall, PlaneWallCorner: thin walls, not floor

function partTable() {
  let source = fs.readFileSync(`${vendor}simulation_worker.bundle.js`, 'utf8');
  const patch = (from, to) => { if (!source.includes(from)) throw Error(`patch point missing: ${from}`); source = source.replace(from, to); };
  patch('function Ao(t){const e=co.get(t);', 'self.__part=Ao;function Ao(t){const e=co.get(t);');
  patch('importScripts("lib/polytrack_physics.js")', '(()=>{throw new Error("__stop__")})()');
  const context = vm.createContext({
    console, performance, TextDecoder, TextEncoder, atob, btoa, Uint8Array, Float32Array, ArrayBuffer, DataView,
    setInterval() {}, clearInterval() {}, setTimeout() {}, clearTimeout() {}, postMessage() {}, addEventListener() {},
    onmessage: null, location: {href: `file://${vendor}simulation_worker.bundle.js`},
  });
  context.self = context;
  try { vm.runInContext(source, context, {filename: 'simulation_worker.bundle.js'}); } catch (e) { if (!/__stop__/.test(String(e))) throw e; }
  return context.__part;
}

export function buildMap() {
  const part = partTable();
  const key = (x, z) => `${x},${z}`;
  const cells = new Map(); // "x,z" -> sorted array of levels
  const gateTiles = [];     // per gate: list of [x,y,z]
  const gates = [...track.checkpoints, track.finishes[0]];
  for (const p of track.parts) {
    const cfg = part(p.id);
    if (!DRIVABLE.has(cfg.category) || WALLS.has(p.id)) continue;
    const tiles = [];
    cfg.tiles.rotated(p.rotation, p.axis).forEach((x, y, z) => tiles.push([p.x + x, p.y + y, p.z + z]));
    for (const [x, y, z] of tiles) {
      const k = key(x, z);
      if (!cells.has(k)) cells.set(k, []);
      if (!cells.get(k).includes(y)) cells.get(k).push(y);
    }
    const gi = gates.findIndex(g => g.partId === p.id && Math.abs(g.center[0] - p.x * TILE) < 15 && Math.abs(g.center[2] - p.z * TILE) < 15);
    if (gi >= 0) gateTiles[gi] = tiles;
  }
  for (const v of cells.values()) v.sort((a, b) => a - b);
  if (gateTiles.length !== gates.length || gateTiles.some(g => !g)) throw Error('gate parts not found among tiles');
  return {cells, gateTiles, gates};
}

// Node id for (x, y, z); BFS over nodes with step cost TILE, jump cost TILE * k * 1.2.
export function neighbours(map, x, y, z) {
  const out = [];
  for (const [dx, dz] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
    let direct = false;
    const nb = map.cells.get(`${x + dx},${z + dz}`);
    if (nb) for (const ny of nb) if (Math.abs(ny - y) <= 1) { out.push([x + dx, ny, z + dz, TILE]); direct = true; }
    if (direct) continue;
    // Flight: straight ahead over empty or lower cells. Landing at the same level or one level
    // up needs a launch ramp (the cell behind is one level lower, i.e. the road rises in the
    // flight direction); off a flat edge the car can only land lower. Higher road ahead blocks.
    const behind = map.cells.get(`${x - dx},${z - dz}`) ?? [];
    const ramp = behind.includes(y - 1) && !behind.includes(y);
    for (let k = 2; k <= JUMP; k++) {
      const c = map.cells.get(`${x + dx * k},${z + dz * k}`);
      if (!c) continue;
      if (c.some(ny => ny > y + 1)) break;
      for (const ny of c) if (ny >= y - 8 && (ramp ? ny <= y + 1 : ny < y || (ny === y && k <= 3))) out.push([x + dx * k, ny, z + dz * k, TILE * k * 1.2]);
    }
  }
  return out;
}

export function buildFields(map = buildMap()) {
  // Forward adjacency (legal moves), then reverse it: the field is the cost-to-go to the gate,
  // so Dijkstra from the gate must walk predecessors (a jump down is not a jump up).
  const nodes = [];
  for (const [k, levels] of map.cells) { const [x, z] = k.split(',').map(Number); for (const y of levels) nodes.push([x, y, z]); }
  const pred = new Map();
  for (const [x, y, z] of nodes) for (const [nx, ny, nz, c] of neighbours(map, x, y, z)) {
    const k = `${nx},${ny},${nz}`;
    if (!pred.has(k)) pred.set(k, []);
    pred.get(k).push([x, y, z, c]);
  }
  const fields = map.gateTiles.map(tiles => {
    const dist = new Map();
    const heap = tiles.map(([x, y, z]) => [0, x, y, z]);
    for (const [, x, y, z] of heap) dist.set(`${x},${y},${z}`, 0);
    while (heap.length) {
      heap.sort((a, b) => b[0] - a[0]);
      const [d, x, y, z] = heap.pop();
      if (d > dist.get(`${x},${y},${z}`)) continue;
      for (const [nx, ny, nz, c] of pred.get(`${x},${y},${z}`) ?? []) {
        const k = `${nx},${ny},${nz}`, nd = d + c;
        if (nd < (dist.get(k) ?? Infinity)) { dist.set(k, nd); heap.push([nd, nx, ny, nz]); }
      }
    }
    return dist;
  });
  // Progress in metres: field value at the car's cell (nearest level), refined within the cell
  // by the direction toward the best neighbour. Off-road: nearest field cell plus a penalty.
  const cellList = [...map.cells.keys()].map(k => k.split(',').map(Number));
  const airCache = new Map();
  function progress(g, px, py, pz) {
    const f = fields[Math.min(g, fields.length - 1)];
    const x = Math.floor(px / TILE), z = Math.floor(pz / TILE), ty = py / TILE;
    const levels = map.cells.get(`${x},${z}`);
    let best = null;
    if (levels) for (const y of levels) { const d = f.get(`${x},${y},${z}`); if (d !== undefined && Math.abs(y - ty) < 1.5 && (best === null || Math.abs(y - ty) < Math.abs(best[0] - ty))) best = [y, d]; }
    if (best === null) {
      // Not on a road cell. Airborne or beside the road: the best cell the car could still land
      // on (at most 1.5 levels above it), by field value plus distance, so a flight over lower
      // road is scored by where it lands, not by what it flies over. Under the road: trapped.
      // ponytail: brute force over all cells, cached per metre cube (a rollout samples ~40 points).
      const ck = `${g},${Math.round(px)},${Math.round(py)},${Math.round(pz)}`;
      const hit = airCache.get(ck);
      if (hit !== undefined) return hit;
      let bd = Infinity, by = 0, ud = Infinity, uv = 0;
      for (const [cx, cz] of cellList) {
        const dh = Math.hypot((cx + 0.5) * TILE - px, (cz + 0.5) * TILE - pz);
        if (dh >= ud && 2 * dh >= bd) continue;
        for (const y of map.cells.get(`${cx},${cz}`)) {
          const v = f.get(`${cx},${y},${cz}`);
          if (v === undefined) continue;
          const dy = y * TILE - py, dd = Math.hypot(dh, dy);
          if (dy <= 1.5 * TILE) { if (v + 2 * dd < bd) { bd = v + 2 * dd; by = y; } }
          else if (dd < ud) { ud = dd; uv = v; }
        }
      }
      // On the ground plane (y ~ 0) beside a raised road is a trap: no way back up without a
      // ramp. Airborne above or beside the road is normal flight.
      const r = bd < Infinity ? bd + (py < 1 && by > 0 ? 300 : 0) : uv + 2 * ud + 300;
      if (airCache.size > 200000) airCache.clear();
      airCache.set(ck, r);
      return r;
    }
    const [y, d] = best;
    let bestN = null;
    for (const [nx, ny, nz] of neighbours(map, x, y, z)) {
      const nd = f.get(`${nx},${ny},${nz}`);
      if (nd !== undefined && nd < d && (bestN === null || nd < bestN[0])) bestN = [nd, nx, nz];
    }
    if (!bestN) return d;
    const dx = bestN[1] - x, dz = bestN[2] - z, len = Math.hypot(dx, dz);
    const proj = ((px - (x + 0.5) * TILE) * dx + (pz - (z + 0.5) * TILE) * dz) / len;
    return d - Math.max(-TILE / 2, Math.min(TILE / 2, proj));
  }
  return {map, fields, progress};
}

if (process.argv[1] && fileURLToPath(import.meta.url) === process.argv[1]) {
  const {map, fields, progress} = buildFields();
  console.log('cells', map.cells.size, 'field sizes', fields.map(f => f.size));
  const s = track.start.position;
  console.log('start -> cp0', progress(0, s[0], s[1], s[2]).toFixed(0), 'm');
  for (let g = 0; g + 1 < map.gates.length; g++) { const c = map.gates[g].center; console.log(`gate ${g} -> gate ${g + 1}`, progress(g + 1, c[0], c[1], c[2]).toFixed(0), 'm'); }
  console.log('stuck spot (-98,1.8,-80) -> cp2', progress(2, -98, 1.8, -80).toFixed(0), 'm; cp2 road (-100,10,-80) -> cp2', progress(2, -100, 10.3, -80).toFixed(0));
  console.log('y=0 road (-100,0.3,-10) -> finish', progress(3, -100, 0.3, -10).toFixed(0), 'm; ramp top (-100,15,-35) -> finish', progress(3, -100, 15, -35).toFixed(0), 'm');
}
