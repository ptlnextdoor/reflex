// Road-frame labels for every rendered frame: signed lateral offset from the road centre (m)
// and heading error vs the route direction (rad), from the sim pose and the track file only.
//   node data/roadframe.mjs            -> runs/frames/<ep>/road.jsonl (one row per labels.jsonl row)
// Route direction: downhill of the cost-to-go field to the car's next gate (learn/route.mjs),
// sampled 2 m either side. Road extent: scan across that direction for drivable cells at the
// car's level; centre = midpoint of the contiguous run under the car.
// ponytail: tile-quantized (5 m cells), so heading on curves snaps to the tile axis and the
// centre on wide turn parts is the footprint middle, not the racing line. Upgrade path: the
// game's road mesh centreline.
import fs from 'node:fs';
import {loadTrack} from '../learn/track.mjs';
import {buildMap, buildFields, partTable} from '../learn/route.mjs';

// Exact frame: on a narrow straight part (Start/Checkpoint/Finish/Straight/Slope, categories
// 0 and 1), the drivable surface is the middle two tile columns, so the centreline is the
// part's own local z axis through (5*x, 5*z). Lateral and heading are exact there; `exact`
// marks those rows. Everything else keeps the tile-quantized estimate (valid, not exact).
const part = partTable();
function rot(v, q) { // rotate vector v by quaternion [x,y,z,w]
  const [qx, qy, qz, qw] = q, [x, y, z] = v;
  const ix = qw * x + qy * z - qz * y, iy = qw * y + qz * x - qx * z, iz = qw * z + qx * y - qy * x, iw = -qx * x - qy * y - qz * z;
  return [ix * qw + iw * -qx + iy * -qz - iz * -qy, iy * qw + iw * -qy + iz * -qx - ix * -qz, iz * qw + iw * -qz + ix * -qy - iy * -qx];
}

const TILE = 5;
const root = new URL('../runs/frames/', import.meta.url);
const eps = fs.readdirSync(root).filter(e => fs.existsSync(new URL(`${e}/labels.jsonl`, root)));
const byTrack = {};
for (const e of eps) (byTrack[e.startsWith('own_') ? 'summer1' : e.split('_')[0]] ??= []).push(e);

function forwardOf(q) { // rotate local (0,0,1) by quaternion [x,y,z,w]
  const [x, y, z, w] = q;
  return [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)];
}

let stats = {rows: 0, valid: 0};
for (const [name, list] of Object.entries(byTrack)) {
  const track = loadTrack(name);
  const map = buildMap(track);
  const {progress} = buildFields(map);
  const straights = track.parts.filter(p => [0, 1].includes(part(p.id).category))
    .map(p => ({c: [5 * p.x, 5 * p.y, 5 * p.z], ax: rot([0, 0, 1], p.q), ac: rot([1, 0, 0], p.q)}));
  const drivableAt = (px, pz, py) => {
    const lv = map.cells.get(`${Math.floor(px / TILE)},${Math.floor(pz / TILE)}`);
    return !!lv && lv.some(y => Math.abs(y - py / TILE) < 1.2);
  };
  // Car forward axis sign: pick the local axis sign that best matches travel direction.
  let agree = 0;
  for (const e of list) {
    const rows = fs.readFileSync(new URL(`${e}/labels.jsonl`, root), 'utf8').trim().split('\n').map(JSON.parse);
    for (let i = 1; i < rows.length; i++) {
      const a = rows[i - 1].position, b = rows[i].position, q = rows[i].quaternion;
      if (rows[i].speedKmh < 40) continue;
      const f = forwardOf([q.x, q.y, q.z, q.w]);
      agree += Math.sign(f[0] * (b.x - a.x) + f[2] * (b.z - a.z));
    }
  }
  const fsign = agree >= 0 ? 1 : -1;
  for (const e of list) {
    const rows = fs.readFileSync(new URL(`${e}/labels.jsonl`, root), 'utf8').trim().split('\n').map(JSON.parse);
    const out = rows.map(r => {
      const {x: px, y: py, z: pz} = r.position, q = r.quaternion;
      const g = r.nextCheckpointIndex;
      const row = {image: r.image, k: r.k, valid: false, lateral: 0, heading: 0};
      if (!drivableAt(px, pz, py)) return row; // airborne or off road: no road frame
      let dx = -(progress(g, px + 2, py, pz) - progress(g, px - 2, py, pz));
      let dz = -(progress(g, px, py, pz + 2) - progress(g, px, py, pz - 2));
      const n = Math.hypot(dx, dz);
      if (!(n > 0.5)) return row;
      dx /= n; dz /= n;
      const ux = -dz, uz = dx; // across the road (left of travel)
      let lo = 0, hi = 0;
      while (lo > -30 && drivableAt(px + ux * (lo - 0.25), pz + uz * (lo - 0.25), py)) lo -= 0.25;
      while (hi < 30 && drivableAt(px + ux * (hi + 0.25), pz + uz * (hi + 0.25), py)) hi += 0.25;
      if (hi - lo < 4 || hi - lo > 40) return row;
      const f = forwardOf([q.x, q.y, q.z, q.w]).map(v => v * fsign);
      const fh = Math.hypot(f[0], f[2]);
      row.valid = true;
      row.lateral = -(lo + hi) / 2;               // + = car left of centre
      row.width = hi - lo;
      row.heading = Math.atan2((f[0] * ux + f[2] * uz) / fh, (f[0] * dx + f[2] * dz) / fh); // + = nose to the left of the route
      // Exact override on a straight narrow part: centreline = part axis, direction sign from the field.
      for (const s of straights) {
        const rx = px - s.c[0], ry = py - s.c[1], rz = pz - s.c[2];
        const along = rx * s.ax[0] + rz * s.ax[2], across = rx * s.ac[0] + rz * s.ac[2];
        if (Math.abs(along) > 10 || Math.abs(across) > 6.5 || ry < -1 || ry > 8 || Math.abs(s.ax[1]) > 0.5) continue;
        const an = Math.hypot(s.ax[0], s.ax[2]);
        let tx = s.ax[0] / an, tz = s.ax[2] / an;
        if (tx * dx + tz * dz < 0) { tx = -tx; tz = -tz; }
        const lx = -tz, lz = tx; // left of travel
        row.exact = true;
        row.lateral = rx * lx + rz * lz;
        row.width = 10.5;
        row.heading = Math.atan2((f[0] * lx + f[2] * lz) / fh, (f[0] * tx + f[2] * tz) / fh);
        break;
      }
      return row;
    });
    fs.writeFileSync(new URL(`${e}/road.jsonl`, root), out.map(o => JSON.stringify(o)).join('\n') + '\n');
    stats.rows += out.length; stats.valid += out.filter(o => o.valid).length; stats.exact = (stats.exact ?? 0) + out.filter(o => o.exact).length;
  }
  console.log(name, 'episodes', list.length, 'forward sign', fsign);
}
console.log(stats);
