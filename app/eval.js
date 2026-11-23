"use strict";
/*
 * Blind multi-arm pairwise evaluation of generated images (CONTEXT §6g decision 10).
 * Each question shows two arms (base SD, best-of-N, RL actors...) for the same held-out prompt and seed; arm names are
 * hidden, sides are random. Arm pairs are cycled in a shuffled balanced order; the (prompt, seed) slot is random but a
 * given (arm pair, slot) is never repeated. Results are POSTed to /api/eval -> outputs/rl/human_eval/<id>.json
 * (checkpoint every 10 answers, final record at the end). Manifest: data/eval_manifest.json (written by rl_final.py).
 * URL: /eval.html?n=120&tag=me
 */
const $ = (id) => document.getElementById(id);
const q = new URLSearchParams(location.search);
const NQ = Math.max(4, Math.min(600, parseInt(q.get("n"), 10) || 120));
const TAG = (q.get("tag") || "").replace(/[^\w-]/g, "").slice(0, 40);
const MIN_VIEW_MS = 450;
let M, Q = [], ans = [], cur = null, shownAt = 0, busy = false;
const id = new Date().toISOString().replace(/[:.]/g, "-") + "_" + Math.random().toString(36).slice(2, 8);

const rnd = (n) => Math.floor(Math.random() * n);
function shuffle(a) { for (let i = a.length - 1; i > 0; i--) { const j = rnd(i + 1); [a[i], a[j]] = [a[j], a[i]]; } return a; }
const src = (arm, p, s) => `${M.imgBase}${arm}/p${String(p).padStart(2, "0")}_s${String(s).padStart(2, "0")}.jpg`;

function plan() {
  const A = M.arms.length, pairs = [];
  for (let i = 0; i < A; i++) for (let j = i + 1; j < A; j++) pairs.push([i, j]);
  const used = new Set(), out = [];
  let order = [];
  while (out.length < NQ) {
    if (!order.length) order = shuffle(pairs.slice());
    const [i, j] = order.pop();
    for (let tries = 0; tries < 50; tries++) {
      const p = rnd(M.prompts.length), s = M.seeds[rnd(M.seeds.length)], key = `${i},${j},${p},${s}`;
      if (used.has(key)) continue;
      used.add(key);
      const flip = Math.random() < 0.5;
      out.push({ prompt_idx: p, seed: s, top: M.arms[flip ? j : i], bottom: M.arms[flip ? i : j] });
      break;
    }
  }
  return out;
}

function load(url) {
  return new Promise((res, rej) => { const im = new Image(); im.onload = () => im.decode().then(() => res(url), () => res(url)); im.onerror = rej; im.src = url; });
}
async function show(k) {
  cur = Q[k];
  const pair = $("pair");
  pair.classList.remove("shown");
  document.querySelectorAll(".slot").forEach((s) => s.classList.remove("picked"));
  const a = src(cur.top, cur.prompt_idx, cur.seed), b = src(cur.bottom, cur.prompt_idx, cur.seed);
  await Promise.all([load(a), load(b)]);
  $("slot0").querySelector("img").src = a;
  $("slot1").querySelector("img").src = b;
  $("prompt").textContent = `“${M.prompts[cur.prompt_idx]}”`;
  $("count").textContent = `${k + 1} / ${Q.length}`;
  $("progress-fill").style.width = `${(100 * k) / Q.length}%`;
  pair.classList.add("shown");
  shownAt = performance.now();
  busy = false;
  if (k + 1 < Q.length) { const n = Q[k + 1]; load(src(n.top, n.prompt_idx, n.seed)); load(src(n.bottom, n.prompt_idx, n.seed)); }
}
function record(partial) {
  const rec = { v: 1, id, tag: TAG, at: new Date().toISOString(), partial, manifest: M.source, n: Q.length, answers: ans };
  return fetch("/api/eval", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(rec) })
    .then((r) => r.ok).catch(() => false);
}
async function pick(k) {
  const dt = performance.now() - shownAt;
  if (busy || !cur || dt < MIN_VIEW_MS) return;
  busy = true;
  $(`slot${k}`).classList.add("picked");
  ans.push({ ...cur, choice: k, ms: Math.round(dt) });
  if (ans.length % 10 === 0 && ans.length < Q.length) record(true);
  await new Promise((r) => setTimeout(r, 180));
  if (ans.length < Q.length) return show(ans.length);
  $("progress-fill").style.width = "100%";
  $("play").hidden = true;
  $("done").hidden = false;
  $("saved").textContent = "Saving…";
  const ok = await record(false);
  $("saved").textContent = ok ? `Saved (${ans.length} answers).` : "Could not save. Keep this tab open and check the server.";
}

fetch("data/eval_manifest.json", { cache: "no-store" }).then((r) => r.json()).then((m) => {
  M = m;
  Q = plan();
  $("start").disabled = false;
  $("start").textContent = `Start · ${Q.length} picks`;
});
$("start").onclick = () => { $("intro").hidden = true; $("play").hidden = false; show(0); };
$("slot0").onclick = () => pick(0);
$("slot1").onclick = () => pick(1);
