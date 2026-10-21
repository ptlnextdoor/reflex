// Closed-loop episode: sim + delayed noisy observation + controller + jittered key delivery.
// Timeline (all in sim ms): at frame time t_k = k*dt the controller sees the state from
// t_k - latency, decides masks for ticks [t_k, t_k + dt). Each key toggle is delivered at its
// planned tick + U(-jitter, jitter) ms (per-toggle, like loop/score.mjs). Observation noise:
// Gaussian on position (per axis, m) and on heading (rad, about the car's up axis).
import {Car, decodeState} from './sim.mjs';
import {Controller} from './controller.mjs';
import {forwardOf, upOf} from './reference.mjs';

export function rng(seed) {
  let s = (seed >>> 0) || 1;
  const next = () => { s = (s + 0x6D2B79F5) >>> 0; let t = s; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
  const gauss = () => { const u = Math.max(next(), 1e-12), v = next(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); };
  return {next, gauss};
}

export function episode(ref, {latency = 60, jitter = 2, noise = 0.05, bias = 0, headingNoise = 0, speedNoise = 0, headingSource = 'exact', seed = 1, params = {}, dt = 8, trace = null, track = 'summer1', abortDistance = 15, warm = 0} = {}) {
  // warm: ticks driven open-loop with the exact reference keys (no jitter) before closing the
  // loop, to evaluate the controller on a later section of the track.
  // bias: constant per-episode position offset of this magnitude (m) in a random horizontal
  // direction, on top of the white noise (a vision model's systematic error).
  // headingSource: 'exact' uses the car's true forward axis (plus headingNoise); 'motion' derives
  // heading from the delta of the last two noisy observed positions, i.e. what a box-only
  // perception model gives (no orientation output).
  const car = new Car(track);
  const ctrl = new Controller(ref, {latency, dt, params});
  const r = rng(seed);
  const horizon = ref.n + (ref.goal.finish ? 5000 : 1500);
  // Observation history: pos(3), fwd(3), up(3), speed per executed tick.
  const hist = new Float64Array(10 * (horizon + dt));
  const planned = new Uint8Array(horizon + 2 * dt);
  const events = []; // {t, bit, on}
  let executed = 0; // ticks executed so far
  let execMask = 0;
  let finish = null, aborted = false, lastPlanned = 0;
  const obs = {pos: [0, 0, 0], fwd: [0, 0, 1], up: [0, 1, 0], speed: 0};
  const obsRing = []; // recent noisy observed positions (for motion-derived heading)
  const biasAngle = r.next() * 2 * Math.PI, bx = bias * Math.cos(biasAngle), bz = bias * Math.sin(biasAngle);
  try {
    for (let k = 0; ; k++) {
      const tk = k * dt;
      if (tk >= horizon) break;
      const tObs = tk - latency;
      let masks;
      if (tObs < 0 || tObs >= executed || tk < warm) {
        // No observation yet (or warm-up): open-loop reference keys.
        masks = ref.keys.subarray(Math.min(tk, ref.n - 1), Math.min(tk + dt, ref.n));
        if (masks.length < dt) { const m = new Uint8Array(dt).fill(ref.keys[ref.n - 1]); m.set(masks); masks = m; }
        ctrl.sModel = ref.snorm[Math.min(tk + dt - 1, ref.n - 1)];
        ctrl.iObs = tk - latency; // next observation will be about this + dt
      } else {
        const h = 10 * tObs;
        obs.pos[0] = hist[h] + bx + noise * r.gauss(); obs.pos[1] = hist[h + 1] + noise * r.gauss(); obs.pos[2] = hist[h + 2] + bz + noise * r.gauss();
        let fx = hist[h + 3], fy = hist[h + 4], fz = hist[h + 5];
        const ux = hist[h + 6], uy = hist[h + 7], uz = hist[h + 8];
        if (headingNoise) { // rotate forward about up by a Gaussian angle (Rodrigues)
          const a = headingNoise * r.gauss(), c = Math.cos(a), s = Math.sin(a);
          const cx = uy * fz - uz * fy, cy = uz * fx - ux * fz, cz = ux * fy - uy * fx;
          const d = ux * fx + uy * fy + uz * fz;
          fx = fx * c + cx * s + ux * d * (1 - c); fy = fy * c + cy * s + uy * d * (1 - c); fz = fz * c + cz * s + uz * d * (1 - c);
        }
        obs.fwd[0] = fx; obs.fwd[1] = fy; obs.fwd[2] = fz; obs.up[0] = ux; obs.up[1] = uy; obs.up[2] = uz;
        obsRing.push([obs.pos[0], obs.pos[1], obs.pos[2]]);
        if (obsRing.length > 12) obsRing.shift();
        if (headingSource === 'motion' && obsRing.length >= 2) {
          // Heading from the chord of recent noisy positions, baseline ~1.5 m (longer baseline = less
          // angle noise, more lag), projected to the plane of the true up. What a box-only model gives.
          const v = hist[h + 9];
          const K = Math.min(obsRing.length - 1, Math.max(1, Math.round(1.5 / Math.max(0.05, v * dt / 1000))));
          const a = obsRing[obsRing.length - 1 - K], b = obsRing[obsRing.length - 1];
          let dx = b[0] - a[0], dy = b[1] - a[1], dz = b[2] - a[2];
          const d = dx * ux + dy * uy + dz * uz;
          dx -= d * ux; dy -= d * uy; dz -= d * uz;
          const len = Math.hypot(dx, dy, dz);
          if (len > 0.3) { obs.fwd[0] = dx / len; obs.fwd[1] = dy / len; obs.fwd[2] = dz / len; }
        }
        obs.speed = Math.max(0, hist[h + 9] + speedNoise * r.gauss());
        // hist[tObs] is the state after tick tObs; ticks tObs+1 .. tk-1 are already committed, so the
        // first controlled tick tk is `latency` ticks after the observed state.
        masks = ctrl.decide(obs, latency);
        if (trace) trace.push({t: tk, iObs: ctrl.iObs, eY: ctrl.lastEy, sModel: ctrl.sModel, speed: obs.speed});
      }
      // Plan and schedule jittered toggles.
      for (let i = 0; i < dt; i++) {
        const t = tk + i, m = masks[i];
        planned[t] = m;
        const changed = m ^ lastPlanned;
        if (changed) for (let b = 0; b < 5; b++) if (changed & (1 << b)) {
          let at = t + (jitter && t >= warm ? Math.round((r.next() * 2 - 1) * jitter) : 0);
          // Keep toggles of the same bit ordered (clamp, as the scorer does).
          for (let e = events.length - 1; e >= 0; e--) if (events[e].bit === b) { if (at <= events[e].t) at = events[e].t + 1; break; }
          if (at < executed) at = executed;
          events.push({t: at, bit: b, on: (m >> b) & 1});
        }
        lastPlanned = m;
      }
      // Execute every tick whose inputs are final: planned through tk+dt-1, so toggles can still land at >= tk+dt-jitter.
      const upto = Math.min(horizon, tk + dt - jitter);
      while (executed < upto) {
        const t = executed;
        if (events.length) {
          let w = 0;
          for (let e = 0; e < events.length; e++) {
            const ev = events[e];
            if (ev.t <= t) execMask = ev.on ? (execMask | (1 << ev.bit)) : (execMask & ~(1 << ev.bit));
            else events[w++] = ev;
          }
          events.length = w;
        }
        const raw = car.step(execMask);
        const st = decodeState(raw);
        const f = forwardOf(st.quaternion), u = upOf(st.quaternion);
        const h = 10 * t;
        hist[h] = st.position.x; hist[h + 1] = st.position.y; hist[h + 2] = st.position.z;
        hist[h + 3] = f[0]; hist[h + 4] = f[1]; hist[h + 5] = f[2];
        hist[h + 6] = u[0]; hist[h + 7] = u[1]; hist[h + 8] = u[2];
        hist[h + 9] = st.speedKmh / 3.6;
        executed++;
        if (ref.goal.finish ? st.finished : st.nextCheckpointIndex >= ref.goal.checkpoint) { finish = ref.goal.finish ? st.finishFrames : t + 1; break; }
        // Abort when hopelessly off the line: distance from the controller's matched progress point.
        if ((t & 63) === 0 && t > latency) {
          const j = Math.min(ref.n - 1, Math.max(0, ctrl.iObs + latency));
          const d = Math.hypot(st.position.x - ref.pos[3 * j], st.position.y - ref.pos[3 * j + 1], st.position.z - ref.pos[3 * j + 2]);
          if (d > abortDistance) { aborted = true; break; }
        }
      }
      if (finish !== null || aborted || executed >= horizon) break;
    }
  } finally {
    car.remove();
  }
  return {finish, aborted, ticks: executed, diag: ctrl.diag};
}
