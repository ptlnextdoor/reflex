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

  // What the page holds at render time: the car object's public getters, the drawn
  // chassis meshes ("Body") and the camera matrices of the last render.
  rf.probe = id => {
    const car = rf.carOf.get(id);
    const bodies = [];
    rf.scene?.traverse(o => { if (o.name === 'Body') bodies.push(Array.from(o.matrixWorld.elements)); });
    const cam = rf.camera;
    return {
      frames: car?.getTime().numberOfFrames ?? null,
      position: car ? car.getPosition() : null,
      quaternion: car ? car.getQuaternion() : null,
      speedKmh: car?.getSpeedKmh() ?? null,
      finished: car?.hasFinished() ?? null,
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
