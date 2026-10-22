// Receding-horizon planner for PolyTrack 0.2.0, run inside the page against the real physics.
// Ammo has no cheap heap snapshot, so each candidate replays the committed prefix from a fresh
// restart (deterministic, ~25 us/ms of sim). Score = progress toward the finish along a list of
// waypoints, finish beats everything.
//   __rhp.plan({waypoints, horizon, seg, samples, iters, commit, maxms, restart})
window.__rhp = {
  async plan(o) {
    const W = o.waypoints, FIN = W[W.length - 1];
    const A = [1, 3, 9, 0, 5, 7, 13, 4, 11];
    const segs = Math.round(o.horizon / o.seg), cs = Math.round(o.commit / o.seg);
    let committed = [];
    let rs = o.seed >>> 0; const rnd = () => ((rs = (rs * 1664525 + 1013904223) >>> 0) / 2 ** 32);
    let probs = Array.from({length: segs}, () => A.map(a => a === 1 ? 0.5 : 0.5 / (A.length - 1)));
    const progressOf = (p, wi) => {
      // waypoint index reached + closeness to the next one
      let k = wi;
      while (k < W.length - 1 && Math.hypot(p.x - W[k][0], p.y - W[k][1], p.z - W[k][2]) < (W[k][3] || 12)) k++;
      const d = Math.hypot(p.x - W[k][0], p.y - W[k][1], p.z - W[k][2]);
      return {k, score: k * 1000 - d};
    };
    const rollout = (masks) => {
      o.restart();
      window.__plan = {masks, len: masks.length};
      const w = window.__W, c = window.__car;
      let wi = 0, best = -1e9, fin = null;
      for (let t = 0; t < masks.length; t++) {
        w.__stepN(1);
        if (c.hasFinished()) { fin = c.getTime(); break; }
        if (t >= committed.length && t % 50 === 0) {
          const pr = progressOf(c.getPosition(), wi); wi = pr.k;
          if (pr.score > best) best = pr.score;
        } else if (t % 50 === 0) { wi = progressOf(c.getPosition(), wi).k; }
      }
      return fin !== null ? {score: 1e7 - fin * 1000, fin} : {score: best, fin: null};
    };
    const expand = seq => { const m = committed.slice(); for (const a of seq) for (let i = 0; i < o.seg; i++) m.push(A[a]); return m; };
    let bestSeq = null, log = [];
    while (committed.length < o.maxms) {
      let top = null;
      for (let it = 0; it < o.iters; it++) {
        const pop = [];
        for (let n = 0; n < o.samples; n++) {
          const seq = probs.map(p => { let r = rnd(), a = 0; while (a < p.length - 1 && (r -= p[a]) > 0) a++; return a; });
          if (n === 0 && bestSeq) seq.splice(0, seq.length, ...bestSeq);
          pop.push({seq, ...rollout(expand(seq))});
        }
        pop.sort((a, b) => b.score - a.score);
        if (!top || pop[0].score > top.score) top = pop[0];
        const el = pop.slice(0, Math.max(3, Math.round(o.samples * 0.15)));
        probs = probs.map((p, j) => p.map((_, a) => 0.7 * (0.02 + el.filter(e => e.seq[j] === a).length) / (el.length + 0.02 * A.length) + 0.3 * p[a]));
      }
      if (top.fin !== null) {
        const m = expand(top.seq);
        log.push(`FINISH ${top.fin.toFixed(3)} s`);
        return {fin: top.fin, controls: m, log};
      }
      for (let j = 0; j < cs; j++) for (let i = 0; i < o.seg; i++) committed.push(A[top.seq[j]]);
      bestSeq = top.seq.slice(cs).concat(Array(cs).fill(0));
      probs = probs.slice(cs).concat(Array.from({length: cs}, () => A.map(a => a === 1 ? 0.5 : 0.5 / (A.length - 1))));
      log.push(`t ${committed.length} score ${top.score.toFixed(0)}`);
      if (o.onStep) o.onStep(committed.length, top.score);
    }
    return {fin: null, controls: committed, log};
  },
};
