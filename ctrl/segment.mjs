// Temporary reference segment, generated from the sim alone (no human data): a hand-written
// steer-to-waypoint rule drives from the start pose to checkpoint 1. Waypoints were found by
// my own key probes (ctrl/scratch, throttle plus one steer nudge): the road bends from heading
// -90 deg to about -107 deg near x=300, then back to -90 deg by x=220, cp1 at about (182, 55, -5).
// Used only to unit-test the controller machinery until learn/best.json exists.
//   node ctrl/segment.mjs   -> writes ctrl/segment.json {controls, frames, checkpoint}
import fs from 'node:fs';
import {Car, decodeState} from './sim.mjs';
import {forwardOf} from './reference.mjs';

export const WAYPOINTS = [[300, -5, 13.5], [262, -5, 2.5], [230, -5, -5], [200, -5, -5], [150, -5, -5]];

// Full-throttle pursuit of successive waypoints with bang-bang steering and a heading deadband.
export function segmentLine({horizon = 7000, deadband = 0.03, lookahead = 12} = {}) {
  const car = new Car();
  const controls = new Uint8Array(horizon);
  let wp = 0, cp = null;
  try {
    for (let t = 0; t < horizon; t++) {
      const s = decodeState(car.state);
      const [x, z] = [s.position.x, s.position.z];
      while (wp < WAYPOINTS.length - 1 && Math.hypot(WAYPOINTS[wp][0] - x, WAYPOINTS[wp][2] - z) < lookahead) wp++;
      const f = forwardOf(s.quaternion);
      const yaw = Math.atan2(f[0], f[2]);
      const want = Math.atan2(WAYPOINTS[wp][0] - x, WAYPOINTS[wp][2] - z);
      let d = want - yaw;
      while (d > Math.PI) d -= 2 * Math.PI;
      while (d < -Math.PI) d += 2 * Math.PI;
      // yaw increases toward +x; left key (8) turns toward +left = cross(up, fwd), which for
      // fwd=-x is -z... resolve the sign empirically: right key (2) decreased yaw in the probes.
      const mask = 1 | (d > deadband ? 8 : d < -deadband ? 2 : 0);
      controls[t] = mask;
      car.step(mask);
      if (cp === null && decodeState(car.state).nextCheckpointIndex >= 1) { cp = t + 1; break; }
    }
  } finally {
    car.remove();
  }
  return {controls: controls.subarray(0, cp ?? horizon), checkpoint: cp};
}

if (process.argv[1] === new URL(import.meta.url).pathname) {
  const {controls, checkpoint} = segmentLine();
  if (!checkpoint) throw Error('segment line did not reach checkpoint 1');
  fs.writeFileSync(new URL('./segment.json', import.meta.url), JSON.stringify({controls: [...controls], frames: controls.length, checkpoint}));
  console.log(`segment line reaches checkpoint 1 at ${checkpoint} ms, ${controls.length} controls -> ctrl/segment.json`);
}
