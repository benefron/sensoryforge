"use strict";
// The evaluation page's shell (roadmap A5b): a draw list and a canvas that plays the
// stimulus frames of one draw as they stream in. Richer inspection arrives with A6.
(function () {
  const $ = (id) => document.getElementById(id);
  const CHUNK = 32;
  const PAGE = 100;
  const state = { offset: 0, total: 0, entry: null, token: 0, chunks: new Map(),
    meta: null, low: 0, high: 1, frame: 0, playing: false, timer: null };

  async function getJSON(url) {
    const r = await fetch(url);
    const body = await r.json();
    if (!r.ok) throw new Error(body.error || r.statusText);
    return body;
  }

  function query() {
    const q = new URLSearchParams();
    if ($("f-split").value) q.set("split", $("f-split").value);
    if ($("f-class").value) q.set("class", $("f-class").value);
    if ($("f-q").value) q.set("q", $("f-q").value);
    return q;
  }

  async function loadSummary() {
    const s = await getJSON("/api/dataset");
    const bits = [s.name || s.folder, s.n_draws + " draws"];
    if (s.sensors) bits.push(s.sensors.n_sensors + " sensors");
    if (s.dt_ms) bits.push("dt " + s.dt_ms + " ms");
    $("summary").textContent = bits.join(" · ");
    for (const k of Object.keys(s.counts.split))
      $("f-split").add(new Option(k + " (" + s.counts.split[k] + ")", k));
    for (const k of Object.keys(s.counts.class))
      $("f-class").add(new Option((k === s.no_class_token ? "no class" : k) + " (" + s.counts.class[k] + ")", k));
  }

  async function loadList(append) {
    if (!append) { state.offset = 0; $("draws").replaceChildren(); }
    const q = query();
    q.set("offset", state.offset);
    q.set("limit", PAGE);
    const page = await getJSON("/api/draws?" + q);
    state.total = page.total;
    for (const row of page.items) {
      const li = document.createElement("li");
      li.textContent = row.entry;
      li.dataset.entry = row.entry;
      const small = document.createElement("small");
      small.textContent = [row.split, row.class || "no class", row.status || ""].filter(Boolean).join(" · ");
      li.append(small);
      li.addEventListener("click", () => openDraw(row.entry));
      $("draws").append(li);
    }
    state.offset += page.items.length;
    $("count").textContent = state.total + " matching";
    $("more").hidden = state.offset >= state.total;
  }

  function mark(entry) {
    for (const li of $("draws").children)
      li.setAttribute("aria-current", li.dataset.entry === entry ? "true" : "false");
  }

  async function fetchChunk(token, start) {
    const q = new URLSearchParams({ entry: state.entry, start, count: CHUNK });
    const r = await fetch("/api/draw/frames?" + q);
    if (!r.ok) throw new Error((await r.json()).error || r.statusText);
    const buf = await r.arrayBuffer();
    if (token !== state.token) return null;
    const h = (n) => r.headers.get(n);
    const chunk = { start: +h("X-Frame-Start"), count: +h("X-Frame-Count"),
      height: +h("X-Frame-Height"), width: +h("X-Frame-Width"), total: +h("X-Frame-Total"),
      dt: +h("X-Frame-Dt-Ms"), low: +h("X-Frame-Min"), high: +h("X-Frame-Max"),
      data: new Float32Array(buf) };
    state.chunks.set(chunk.start, chunk);
    state.low = Math.min(state.low, chunk.low);
    state.high = Math.max(state.high, chunk.high);
    return chunk;
  }

  function chunkFor(frame) {
    const start = Math.floor(frame / CHUNK) * CHUNK;
    return state.chunks.get(start);
  }

  function paint(frame) {
    const chunk = chunkFor(frame);
    if (!chunk) return false;
    const { height, width } = chunk;
    const canvas = $("movie");
    if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
    const ctx = canvas.getContext("2d");
    const img = ctx.createImageData(width, height);
    const off = (frame - chunk.start) * width * height;
    const span = state.high > state.low ? state.high - state.low : 1;
    for (let i = 0; i < width * height; i++) {
      const v = Math.max(0, Math.min(1, (chunk.data[off + i] - state.low) / span));
      const g = Math.round(255 * v);
      img.data[4 * i] = g; img.data[4 * i + 1] = g; img.data[4 * i + 2] = g; img.data[4 * i + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
    $("scrub").value = frame;
    $("clock").textContent = "frame " + frame + " / " + (chunk.total - 1) + " · " + (frame * chunk.dt).toFixed(1) + " ms";
    return true;
  }

  async function show(frame) {
    state.frame = frame;
    const token = state.token;
    if (!chunkFor(frame)) await fetchChunk(token, Math.floor(frame / CHUNK) * CHUNK);
    if (token !== state.token) return;
    paint(frame);
    const next = (Math.floor(frame / CHUNK) + 1) * CHUNK;
    if (next < state.meta.total && !state.chunks.has(next))
      fetchChunk(token, next).catch(() => {});
  }

  function stop() {
    state.playing = false;
    clearTimeout(state.timer);
    $("play").textContent = "Play";
  }

  function tick() {
    if (!state.playing) return;
    const next = state.frame + 1 >= state.meta.total ? 0 : state.frame + 1;
    const t0 = performance.now();
    show(next).then(() => {
      const wait = Math.max(0, Math.min(state.meta.dt, 50) - (performance.now() - t0));
      state.timer = setTimeout(tick, wait);
    });
  }

  async function openDraw(entry) {
    stop();
    state.entry = entry;
    state.token += 1;
    state.chunks = new Map();
    state.low = Infinity; state.high = -Infinity;
    mark(entry);
    $("draw-title").textContent = entry;
    $("info").textContent = "";
    $("timing").textContent = "opening…";
    const t0 = performance.now();
    try {
      const first = await fetchChunk(state.token, 0);
      if (!first) return;
      state.meta = { total: first.total, dt: first.dt };
      paint(0);
      $("timing").textContent = "first frame after " + Math.round(performance.now() - t0) + " ms";
      const scrub = $("scrub");
      scrub.max = first.total - 1; scrub.value = 0; scrub.disabled = false;
      $("play").disabled = false;
      if (first.total > CHUNK) fetchChunk(state.token, CHUNK).catch(() => {});
      const d = await getJSON("/api/draw?entry=" + encodeURIComponent(entry));
      $("info").textContent = JSON.stringify(d.record, null, 1);
    } catch (err) {
      $("timing").textContent = String(err.message || err);
    }
  }

  $("play").addEventListener("click", () => {
    if (state.playing) { stop(); return; }
    state.playing = true;
    $("play").textContent = "Pause";
    tick();
  });
  $("scrub").addEventListener("input", (e) => { stop(); show(+e.target.value); });
  $("more").addEventListener("click", () => loadList(true));
  $("filters").addEventListener("change", () => loadList(false));
  $("f-q").addEventListener("input", () => loadList(false));
  $("filters").addEventListener("submit", (e) => e.preventDefault());
  $("random").addEventListener("click", async () => {
    try {
      const pick = await getJSON("/api/draws/random?" + query());
      openDraw(pick.draw.entry);
    } catch (err) { $("timing").textContent = String(err.message || err); }
  });

  loadSummary().then(() => loadList(false)).catch((err) => { $("summary").textContent = String(err.message || err); });
})();
