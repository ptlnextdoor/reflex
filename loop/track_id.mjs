// Summer 1 (or any official track) leaderboard trackId = sha256(track's own binary serialization),
// computed with the game's parser in a vm. Prints the hex id.
//   node loop/track_id.mjs [summer1]
import fs from 'node:fs';
import vm from 'node:vm';
import crypto from 'node:crypto';

const name = process.argv[2] ?? 'summer1';
const vendor = new URL('../vendor/polytrack/0.6.3/', import.meta.url).pathname;
let src = fs.readFileSync(vendor + 'simulation_worker.bundle.js', 'utf8');
const patch = (a, b) => { if (!src.includes(a)) throw Error('patch point missing: ' + a.slice(0, 60)); src = src.replace(a, b); };
patch('const Ko=class{', 'const Ko=self.__Track=class{');
patch('getId(){return Xi(this,Vo,', '__bytes(){return ji(this,Oo,"m",Yo).call(this)}getId(){return Xi(this,Vo,');
patch('importScripts("lib/polytrack_physics.js")', '(()=>{throw new Error("__stop__")})()');
// The bundle's math helpers wait for a WASM module; the serializer only needs plain Math.
patch('function Gi(t){throw new Error(t+": Waiting for WASM to compile")}', 'function Gi(t){throw new Error(t+": no wasm")}');
patch('log2:()=>Gi("log2")', 'log2:__log2');
const ctx = vm.createContext({
  console: {log() {}, warn() {}, error() {}, info() {}}, performance, TextDecoder, TextEncoder, atob, btoa,
  Uint8Array, Float32Array, ArrayBuffer, DataView, setInterval() {}, clearInterval() {}, setTimeout() {},
  clearTimeout() {}, postMessage() {}, addEventListener() {}, onmessage: null, location: {href: "file://x"}, __log2: Math.log2,
});
ctx.self = ctx;
try { vm.runInContext(src, ctx); } catch (e) { if (!/__stop__/.test(String(e))) throw Error('bundle: ' + String(e).slice(0, 200)); }
const fixture = JSON.parse(fs.readFileSync(`${vendor}${name}.fixture.json`, 'utf8'));
const track = ctx.__Track.fromSaveString(fixture.create.trackData);
const bytes = track.__bytes();
console.log(crypto.createHash('sha256').update(Buffer.from(bytes)).digest('hex'));
