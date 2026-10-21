// v1 reference-tracking controller. Decides per-ms key masks for the next control interval
// from a delayed, noisy observation. Pure JS, no sim access: the only model it carries is
// the measured steering actuator (normalized steer slews STEER_SLEW/ms toward the key target).
//
// Frame: ref.left/ref.fwd/ref.up at the matched progress index. Lateral error e_y > 0 means
// the car is left of the line; heading error e_psi > 0 means it points left of the line.
// Steering law (normalized units, left positive): s_des = s_ref - Ky*e_y - Kpsi*e_psi, then
// the keys drive the steering belief s_model toward s_des one ms at a time (PWM inside the
// control interval, 0.01 resolution).
import {steerTarget, slew, cross} from './reference.mjs';

export const DEFAULT_PARAMS = {
  Ky: 0.6,        // normalized steer per m of lateral error
  Kpsi: 8,        // normalized steer per rad of heading error
  Kvy: 0.0,       // normalized steer per (m/s) of lateral velocity error (damping)
  speedExp: 0.0,  // gains scale by (vref/60)^speedExp
  deadband: 0.0,  // |s_des - s_model| below this: no steer key
  window: 40,     // progress search half-window (ms) around the predicted index
  progressAlpha: 1.0, // blend between predicted progress (0) and nearest point (1)
  dvBand: 2.0,    // m/s speed error tolerance before throttle/brake correction
  brakeGain: 0.0, // extra brake duty per m/s over reference speed (0 = feedforward only)
  airD: 1,        // 1 = feedforward D in the air as the reference does
  psiAlpha: 1.0,  // heading-error low-pass: 1 = no filtering, 0.3 = heavy (for motion-derived heading)
};

export class Controller {
  constructor(ref, {latency = 60, dt = 8, params = {}} = {}) {
    this.ref = ref;
    this.L = latency;      // total loop latency the controller compensates (ms)
    this.dt = dt;
    this.p = {...DEFAULT_PARAMS, ...params};
    this.iObs = 0;         // progress index of the last observation
    this.sModel = 0;       // belief of normalized steering now
    this.lastEy = 0;
    this.psiFilt = 0;      // low-pass heading error (first-order, alpha = p.psiAlpha per step)
    this.out = new Uint8Array(dt);
    this.diag = {maxEy: 0, maxEpsi: 0, steerMs: 0};
  }

  // obs: {pos:[x,y,z], fwd:[x,y,z], up:[x,y,z], speed (m/s)} observed L_obs ms ago.
  // sent: masks already committed for the ms between the observation and the first tick this
  // decision controls (the controller's own recent commands), used to advance the steer belief.
  // Returns Uint8Array(dt) of masks for ticks [t+L_act, t+L_act+dt).
  decide(obs, ageMs) {
    const r = this.ref, p = this.p, n = r.n;
    // 1. Progress: nearest reference point to the observed position, within a window around
    //    the predicted advance since the last observation.
    const pred = Math.min(n - 1, this.iObs + this.dt);
    let best = pred, bestD = Infinity;
    const lo = Math.max(0, pred - p.window), hi = Math.min(n - 1, pred + p.window);
    for (let i = lo; i <= hi; i++) {
      const dx = obs.pos[0] - r.pos[3 * i], dy = obs.pos[1] - r.pos[3 * i + 1], dz = obs.pos[2] - r.pos[3 * i + 2];
      const d = dx * dx + dy * dy + dz * dz;
      // Ties (car standing still at the start) resolve toward the predicted index.
      if (d < bestD - 1e-12 || (Math.abs(d - bestD) <= 1e-12 && Math.abs(i - pred) < Math.abs(best - pred))) { bestD = d; best = i; }
    }
    const iObs = Math.round(pred + p.progressAlpha * (best - pred));
    this.iObs = iObs;
    // 2. Errors at the observation, in the reference frame at iObs.
    const ex = obs.pos[0] - r.pos[3 * iObs], ey3 = obs.pos[1] - r.pos[3 * iObs + 1], ez = obs.pos[2] - r.pos[3 * iObs + 2];
    const lx = r.left[3 * iObs], ly = r.left[3 * iObs + 1], lz = r.left[3 * iObs + 2];
    const fx = r.fwd[3 * iObs], fy = r.fwd[3 * iObs + 1], fz = r.fwd[3 * iObs + 2];
    const eY = ex * lx + ey3 * ly + ez * lz;
    const fl = obs.fwd[0] * lx + obs.fwd[1] * ly + obs.fwd[2] * lz;
    const ff = obs.fwd[0] * fx + obs.fwd[1] * fy + obs.fwd[2] * fz;
    const ePsiRaw = Math.atan2(fl, ff);
    this.psiFilt += p.psiAlpha * (ePsiRaw - this.psiFilt);
    const ePsi = this.psiFilt;
    // 3. Predict to the first controlled tick: the observation is ageMs old. Lateral error grows
    //    with the heading error; progress advances with the observed speed ratio.
    const vRef = r.speed[iObs];
    const ratio = vRef > 3 ? Math.min(2, Math.max(0.2, obs.speed / vRef)) : 1;
    const eYNow = eY + obs.speed * Math.sin(ePsi) * ageMs / 1000;
    const iNow = Math.min(n - 1, Math.round(iObs + ageMs * ratio));
    const eYdot = (eYNow - this.lastEy) / (this.dt / 1000);
    this.lastEy = eYNow;
    if (Math.abs(eYNow) > this.diag.maxEy) this.diag.maxEy = Math.abs(eYNow);
    if (Math.abs(ePsi) > this.diag.maxEpsi) this.diag.maxEpsi = Math.abs(ePsi);
    // 4. Steering correction (normalized units) added to the reference steering trajectory, then
    //    the keys drive the steering belief toward the corrected reference one ms at a time.
    const scale = p.speedExp ? Math.pow(Math.max(vRef, 1) / 60, p.speedExp) : 1;
    const corr = scale * (p.Ky * eYNow + p.Kpsi * ePsi + p.Kvy * eYdot);
    const airborne = r.airborne[iNow];
    for (let k = 0; k < this.dt; k++) {
      const i = Math.min(n - 1, iNow + k);
      let mask = r.keys[i] & 5; // throttle and brake feedforward
      if (p.brakeGain > 0 && !airborne && obs.speed > vRef + p.dvBand) {
        const duty = Math.min(1, p.brakeGain * (obs.speed - vRef - p.dvBand));
        if (k < Math.round(duty * this.dt)) mask |= 4;
      }
      if (airborne && !p.airD) mask &= ~4;
      let target = 0;
      if (airborne) target = steerTarget(r.keys[i]); // L/R have no effect airborne; keep the line's keys
      else {
        // Pick the key whose one-ms slew lands closest to the corrected reference steering.
        // (Holding a saturated key must keep the key down, so compare end-of-ms outcomes.)
        const sDes = Math.max(-1, Math.min(1, r.snorm[i] - corr));
        const e0 = Math.abs(slew(this.sModel, 0) - sDes);
        const eL = Math.abs(slew(this.sModel, 1) - sDes), eR = Math.abs(slew(this.sModel, -1) - sDes);
        if (eL < e0 - p.deadband && eL <= eR) target = 1;
        else if (eR < e0 - p.deadband) target = -1;
      }
      if (target > 0) mask |= 8; else if (target < 0) mask |= 2;
      this.sModel = slew(this.sModel, target);
      if (target) this.diag.steerMs++;
      this.out[k] = mask;
    }
    return this.out;
  }
}
