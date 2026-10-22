// PolyTrack 0.2.0 driver: in-page search against the real game physics (Ammo.js, 1 kHz).
// Loaded into the page by loop/v020_search.py. Provides:
//   __sim.reset()        -> restart race from frame 0 (game's own restart: KeyR)
//   __sim.run(masks, maxms) -> {fin, ticks, end, minD} driving masks[t] per ms from race start
// Determinism note: we restart via the game's reset path, then step manually.
window.__sim = {
  finish: null,
  run(masks, maxms, target) {
    window.__plan = {masks, len: masks.length};
    const w = window.__world, c = window.__car;
    let minD = 1e9, best = null;
    for (let t = 0; t < maxms; t++) {
      w.__stepN(1);
      if (target && t % 20 === 0) {
        const p = c.getPosition();
        const d = Math.hypot(p.x - target[0], p.y - target[1], p.z - target[2]);
        if (d < minD) { minD = d; best = t; }
      }
      if (c.hasFinished()) { return {fin: c.getTime(), ticks: t + 1, minD: 0}; }
    }
    const p = c.getPosition();
    return {fin: null, end: [p.x, p.y, p.z], minD, minAt: best};
  },
};
