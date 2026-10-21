import {Worker} from 'node:worker_threads';
import os from 'node:os';

export class Pool {
  constructor(size = Math.max(1, Math.min(12, os.cpus().length - 2)), track = 'summer1') {
    this.workers = [];
    this.idle = [];
    this.pending = new Map();
    this.queue = [];
    this.nextId = 1;
    this.size = size;
    this.ready = Promise.all(Array.from({length: size}, () => new Promise((resolve, reject) => {
      const w = new Worker(new URL('./worker.mjs', import.meta.url), {workerData: {track}});
      w.on('message', msg => {
        if (msg.type === 'ready') { this.idle.push(w); resolve(); this.#drain(); return; }
        const p = this.pending.get(msg.id);
        if (!p) return;
        this.pending.delete(msg.id);
        this.idle.push(w);
        this.#drain();
        if (msg.type === 'error') p.reject(Error(msg.error)); else p.resolve(msg);
      });
      w.on('error', reject);
      this.workers.push(w);
    })));
  }
  #drain() {
    while (this.idle.length && this.queue.length) {
      const w = this.idle.pop();
      const {msg, resolve, reject} = this.queue.shift();
      this.pending.set(msg.id, {resolve, reject});
      w.postMessage(msg);
    }
  }
  #send(msg) {
    msg.id = this.nextId++;
    return new Promise((resolve, reject) => { this.queue.push({msg, resolve, reject}); this.#drain(); });
  }
  // Broadcast a new base line to every worker (each needs its own snapshots).
  async setBase(controls, every = 250) {
    await this.ready;
    // wait for all in-flight evals so every worker becomes idle, then send one base per worker
    while (this.pending.size || this.queue.length) await new Promise(r => setTimeout(r, 1));
    const results = await Promise.all(this.workers.map(w => new Promise((resolve, reject) => {
      const id = this.nextId++;
      this.pending.set(id, {resolve, reject});
      this.idle.splice(this.idle.indexOf(w), 1);
      w.postMessage({type: 'base', id, controls, every});
    })));
    return results[0];
  }
  evaluate(controls, horizon) { return this.#send({type: 'eval', controls, horizon}); }
  evaluateAll(candidates, horizon) { return Promise.all(candidates.map(c => this.evaluate(c, horizon))); }
  async close() { await Promise.all(this.workers.map(w => w.terminate())); }
}
