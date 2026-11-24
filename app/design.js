"use strict";
/*
 * Design-transfer test (src/design_transfer.py). Fixed 100 pairs of Crello designs from data/design_manifest.json
 * (sides already randomised there; order shuffled here). The frozen LAPIS model's predictions were locked and hashed
 * before this page existed, so nothing here can leak into them. Answers are POSTed to /api/design ->
 * outputs/design_transfer/answers/<id>.json (checkpoint every 10, final at the end).  URL: /design.html?tag=me
 * ?set=products: ABO home products (data/products_manifest.json, POST /api/products), with the category shown.
 */
const $ = (id) => document.getElementById(id);
const q = new URLSearchParams(location.search);
const TAG = (q.get("tag") || "").replace(/[^\w-]/g, "").slice(0, 40);
const SET = q.get("set") === "products" ? "products" : "design";
const MIN_VIEW_MS = 450;
let M, Q = [], ans = [], cur = null, shownAt = 0, busy = false;
const id = new Date().toISOString().replace(/[:.]/g, "-") + "_" + Math.random().toString(36).slice(2, 8);
function shuffle(a) { for (let i = a.length - 1; i > 0; i--) { const j = Math.floor(Math.random() * (i + 1)); [a[i], a[j]] = [a[j], a[i]]; } return a; }
const src = (d) => `${M.imgBase}${d}.jpg`;
function load(url) {
  return new Promise((res, rej) => { const im = new Image(); im.onload = () => im.decode().then(() => res(url), () => res(url)); im.onerror = rej; im.src = url; });
}
async function show(k) {
  cur = Q[k];
  const pair = $("pair");
  pair.classList.remove("shown");
  document.querySelectorAll(".slot").forEach((s) => s.classList.remove("picked"));
  const a = src(cur.top), b = src(cur.bottom);
  await Promise.all([load(a), load(b)]);
  $("slot0").querySelector("img").src = a;
  $("slot1").querySelector("img").src = b;
  $("count").textContent = `${k + 1} / ${Q.length}`;
  if (cur.category) $("phase").textContent = cur.category[0].toUpperCase() + cur.category.slice(1) + "s";
  $("progress-fill").style.width = `${(100 * k) / Q.length}%`;
  pair.classList.add("shown");
  shownAt = performance.now();
  busy = false;
  if (k + 1 < Q.length) { load(src(Q[k + 1].top)); load(src(Q[k + 1].bottom)); }
}
function record(partial) {
  const rec = { v: 1, id, tag: TAG, at: new Date().toISOString(), partial, manifest: M.source, n: Q.length, answers: ans };
  return fetch(`/api/${SET}`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(rec) })
    .then((r) => r.ok).catch(() => false);
}
async function pick(k) {
  const dt = performance.now() - shownAt;
  if (busy || !cur || dt < MIN_VIEW_MS) return;
  busy = true;
  $(`slot${k}`).classList.add("picked");
  ans.push({ k: cur.k, top: cur.top, bottom: cur.bottom, choice: k, ms: Math.round(dt) });
  if (ans.length % 10 === 0 && ans.length < Q.length) record(true);
  await new Promise((r) => setTimeout(r, 180));
  if (ans.length < Q.length) return show(ans.length);
  $("progress-fill").style.width = "100%";
  $("play").hidden = true;
  $("done").hidden = false;
  $("saved").textContent = "Saving…";
  const ok = await record(false);
  $("saved").textContent = ok ? `Saved (${ans.length} answers). Tell Claude you're done.` : "Could not save. Keep this tab open and check the server.";
}
fetch(`data/${SET}_manifest.json`, { cache: "no-store" }).then((r) => r.json()).then((m) => {
  M = m;
  Q = shuffle(m.pairs.slice());
  if (SET === "products") {
    document.querySelector(".lead").textContent = "100 pairs of home products: chairs, sofas, tables, rugs, lamps.";
    document.querySelector(".hint").textContent = "Tap the one you'd rather have in your home. Judge the look, not the price or the photo. The model's guesses were saved before you started.";
    document.title = "This or That · products";
  }
  $("start").disabled = false;
  $("start").textContent = `Start · ${Q.length} picks`;
});
$("start").onclick = () => { $("intro").hidden = true; $("play").hidden = false; show(0); };
$("slot0").onclick = () => pick(0);
$("slot1").onclick = () => pick(1);
