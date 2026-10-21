// Page hook for data/render.py. Injected before main.bundle.js runs (Playwright init script).
// Takes over requestAnimationFrame so the game loop is stepped by hand with exact dt, and
// feeds the player car byte-exact 227-byte states from the Node sim through the same
// callback the game's simulation worker uses (UpdateResult -> VO decode -> setCarState).
(() => {
  // First session without the tutorial overlay: the game then boots to the main menu.
  try {
    localStorage.setItem('polytrack_v5_prod_startup_info',
      JSON.stringify({lastVersion: '0.6.3', isTutorialCompleted: true}));
  } catch {}
  globalThis.polytrackModConfiguration = {unblocked: true};

  const rf = globalThis.__rf = {
    manual: false, pending: [], ts: 0, renders: 0, cars: [], states: null,
    scene: null, camera: null, req: null, carOf: new Map(), feeding: null,
  };
  const nativeRAF = window.requestAnimationFrame.bind(window);
  const nativeCAF = window.cancelAnimationFrame.bind(window);
  // In manual mode every rAF callback (the game loop, and any poller) waits for rf.step.
  window.requestAnimationFrame = cb => {
    if (!rf.manual) return nativeRAF(cb);
    rf.pending.push(cb);
    return -rf.pending.length;
  };
  window.cancelAnimationFrame = id => {
    if (id < 0) rf.pending[-id - 1] = null; else nativeCAF(id);
  };

  // Pull webpack's require out of the chunk loader and wrap the game's prototypes.
  rf.setup = () => {
    if (rf.req) return 'already';
    self.webpackChunk.push([[987654321], {}, r => { rf.req = r; }]);
    if (!rf.req) throw Error('webpack require not captured');
    const Sim = rf.req(5220).A, Car = rf.req(641).A, Renderer = rf.req(1507).A;
    rf.VO = rf.req(3899).VO;
    const createCar = Sim.prototype.createCar;
    Sim.prototype.createCar = function (start, mv, mo, track, recording, cb) {
      const entry = {id: null, recording: recording != null, calls: 0, cb: null, deleted: false};
      const wrapped = state => { entry.calls++; return cb(state); };
      const result = createCar.call(this, start, mv, mo, track, recording, wrapped);
      entry.id = result.id;
      entry.cb = wrapped;
      rf.cars.push(entry);
      return result;
    };
    const deleteCar = Sim.prototype.deleteCar;
    Sim.prototype.deleteCar = function (id) {
      for (const c of rf.cars) if (c.id === id) c.deleted = true;
      return deleteCar.call(this, id);
    };
    const setCarState = Car.prototype.setCarState;
    Car.prototype.setCarState = function (state, reset) {
      if (rf.feeding !== null) rf.carOf.set(rf.feeding, this);
      return setCarState.call(this, state, reset);
    };
    const update = Renderer.prototype.update;
    Renderer.prototype.update = function (...args) {
      const result = update.apply(this, args);
      rf.renders++;
      rf.scene = this.scene;
      rf.camera = this.camera;
      rf.rendererObj = this;
      rf.lastSun = args[0];
      return result;
    };
    return 'ok';
  };

  rf.load = async url => {
    rf.states = new Uint8Array(await (await fetch(url)).arrayBuffer());
    return rf.states.length / 227;
  };

  // Feed raw states for ticks (from, to]: raw index t holds the state with frames == t + 1.
  rf.feed = (id, from, to) => {
    const car = rf.cars.find(c => c.id === id);
    if (!car || car.deleted) throw Error('unknown or deleted car ' + id);
    const view = new DataView(new ArrayBuffer(4));
    view.setUint32(0, id, true);
    const idBytes = new Uint8Array(view.buffer);
    rf.feeding = id;
    try {
      for (let t = from; t < to; t++) {
        const bytes = rf.states.slice(t * 227, (t + 1) * 227);
        if (bytes.length !== 227) throw Error('state ' + t + ' out of range');
        if (bytes[11] & 2) throw Error('refusing to feed a finished state (tick ' + t + ')');
        bytes.set(idBytes, 0);
        const {carState} = rf.VO(bytes.subarray(4));
        if (carState.frames !== t + 1) throw Error('frames mismatch at tick ' + t);
        car.cb(carState);
      }
    } finally {
      rf.feeding = null;
    }
    return to;
  };

  rf.step = dtMs => {
    const cbs = rf.pending.filter(Boolean);
    if (cbs.length === 0) throw Error('no pending animation frame');
    rf.pending = [];
    if (rf.ts === 0) rf.ts = performance.now();
    rf.ts += dtMs;
    const before = rf.renders;
    for (const cb of cbs) cb(rf.ts);
    return rf.renders - before;
  };

  // Fresh player car through the game's own start-reset key (default binding KeyT), so a
  // new episode starts at frame 0 with a new car id and a reset camera.
  rf.restart = () => {
    const before = rf.cars.length;
    window.dispatchEvent(new KeyboardEvent('keydown', {code: 'KeyT', key: 't', bubbles: true}));
    window.dispatchEvent(new KeyboardEvent('keyup', {code: 'KeyT', key: 't', bubbles: true}));
    return before;
  };

  // Re-render the current state with the player car's meshes changed, then restore:
  // mode 'hidden' = car meshes invisible, 'noshadow' = car drawn but casting no shadow,
  // 'ontop' = car drawn last without depth test (full silhouette, nothing occludes it),
  // 'normal' = as the game draws it. data/check.py diffs 'ontop' against 'hidden':
  // those pixels are exactly the car's own geometry, independent of its random paint.
  rf.rerender = mode => {
    const scene = rf.scene;
    const meshes = [];
    scene.traverse(o => { if (o.isMesh && /^(Body|Exhaust|Suspension|Wheel|Rims)/.test(o.name)) meshes.push(o); });
    const mats = m => Array.isArray(m.material) ? m.material : [m.material];
    const was = meshes.map(m => [m.visible, m.castShadow, m.renderOrder, mats(m).map(x => [x.depthTest, x.depthWrite])]);
    for (const m of meshes) {
      if (mode === 'hidden') m.visible = false;
      if (mode !== 'normal') m.castShadow = false;
      if (mode === 'ontop') { m.renderOrder = 1e6; for (const x of mats(m)) { x.depthTest = false; x.depthWrite = false; } }
    }
    rf.rendererObj.update(rf.lastSun);
    meshes.forEach((m, i) => {
      m.visible = was[i][0]; m.castShadow = was[i][1]; m.renderOrder = was[i][2];
      mats(m).forEach((x, j) => { x.depthTest = was[i][3][j][0]; x.depthWrite = was[i][3][j][1]; });
    });
    return meshes.length;
  };

  // Exact 2D hitbox of the player car as drawn: every vertex of the car's meshes (Body +
  // Exhaust child, Suspension with morph targets applied, 4 wheels) through the last
  // render's view and projection matrices, into pixel coordinates (origin top-left, y down).
  // Returns null when no vertex lands in front of the camera. `inside` is the fraction of
  // vertices that fall inside the viewport, so a cropped box can be told apart.
  rf.box = (width, height) => {
    const cam = rf.camera, scene = rf.scene;
    if (!cam || !scene) return null;
    const names = new Set(['Body', 'Exhaust0', 'Exhaust1', 'Exhaust2', 'Suspension', 'Wheel0']);
    const meshes = [];
    scene.traverse(o => { if (o.isMesh && (names.has(o.name) || /^(Exhaust|Wheel|Rims)/.test(o.name))) meshes.push(o); });
    const THREE = rf.req(4922);
    const v = new THREE.Pq0();
    const mvp = new THREE.kn4().multiplyMatrices(cam.projectionMatrix, cam.matrixWorldInverse);
    let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity, n = 0, inside = 0;
    for (const mesh of meshes) {
      const pos = mesh.geometry.attributes.position;
      const infl = mesh.morphTargetInfluences, morphs = mesh.geometry.morphAttributes?.position;
      const m = new THREE.kn4().multiplyMatrices(mvp, mesh.matrixWorld);
      for (let i = 0; i < pos.count; i++) {
        v.fromBufferAttribute(pos, i);
        if (infl && morphs) for (let j = 0; j < morphs.length; j++) {
          const w = infl[j]; if (!w) continue;
          v.x += w * morphs[j].getX(i); v.y += w * morphs[j].getY(i); v.z += w * morphs[j].getZ(i);
        }
        // Clip-space w must be positive (in front of the near plane).
        const e = m.elements;
        const x = e[0] * v.x + e[4] * v.y + e[8] * v.z + e[12];
        const y = e[1] * v.x + e[5] * v.y + e[9] * v.z + e[13];
        const w = e[3] * v.x + e[7] * v.y + e[11] * v.z + e[15];
        if (w <= 0) continue;
        const px = (x / w + 1) * 0.5 * width, py = (1 - y / w) * 0.5 * height;
        n++;
        if (px >= 0 && px < width && py >= 0 && py < height) inside++;
        if (px < minX) minX = px; if (px > maxX) maxX = px;
        if (py < minY) minY = py; if (py > maxY) maxY = py;
      }
    }
    if (n === 0) return null;
    return {x0: minX, y0: minY, x1: maxX, y1: maxY, meshes: meshes.length, vertices: n, inside: inside / n};
  };

  // What the page holds at render time: the car object's public getters, the drawn
  // chassis meshes ("Body") and the camera matrices of the last render.
  rf.probe = id => {
    const car = rf.carOf.get(id);
    const bodies = [];
    rf.scene?.traverse(o => { if (o.name === 'Body') bodies.push(Array.from(o.matrixWorld.elements)); });
    const cam = rf.camera;
    const q = car ? car.getQuaternion() : null;
    return {
      frames: car?.getTime().numberOfFrames ?? null,
      position: car ? car.getPosition() : null,
      quaternion: q ? {x: q.x, y: q.y, z: q.z, w: q.w} : null,
      speedKmh: car?.getSpeedKmh() ?? null,
      finished: car?.hasFinished() ?? null,
      primaryHex: car?.getCarStyle().primaryHex ?? null,
      bodies, renders: rf.renders,
      camera: cam ? {
        view: Array.from(cam.matrixWorldInverse.elements),
        proj: Array.from(cam.projectionMatrix.elements),
        position: {x: cam.position.x, y: cam.position.y, z: cam.position.z},
        fov: cam.fov,
      } : null,
    };
  };
})();
