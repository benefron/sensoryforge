"use strict";
// The evaluation page (roadmap A5b, A6a): a draw list with pickers (filters, draw seed,
// random with a repeatable pick seed) and a movie of the stimulus frames of one draw that
// streams in as it plays. Click-to-pin traces arrive with A6b, the declared-fields panel
// and two-draw comparison with A6c. The frame geometry is in geometry.js.
(function () {
  const G = window.EvalGeometry;
  const $ = (id) => document.getElementById(id);
  const CHUNK = 32;          // frames per request; chunk starts are multiples of this
  const PAGE = 100;
  const MAX_BYTES = 128 << 20;   // decoded chunks kept in the browser
  const MAX_IN_FLIGHT = 4;
  const MAX_STEP_MS = 100;   // a hidden tab's long gap must not make the movie jump
  const state = { offset: 0, total: 0, entry: null, token: 0, seq: 0, tick: 0,
    chunks: new Map(), pending: new Map(), bytes: 0, meta: null, draw: null,
    low: Infinity, high: -Infinity, scaleText: "", frame: 0, cursor: 0, last: 0,
    playing: false, raf: 0, speed: 1, image: null, channel: 0 };

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
    if ($("f-seed").value.trim()) q.set("seed", $("f-seed").value.trim());
    return q;
  }

  function say(text) { $("timing").textContent = text; }

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
    mark(state.entry);
  }

  function mark(entry) {
    for (const li of $("draws").children)
      li.setAttribute("aria-current", li.dataset.entry === entry ? "true" : "false");
  }

  // ---- chunks: one promise per chunk start, so a chunk is never fetched twice ----

  const chunkStart = (frame) => Math.floor(frame / CHUNK) * CHUNK;

  async function fetchChunk(token, start) {
    const q = new URLSearchParams({ entry: state.entry, start, count: CHUNK, channel: state.channel });
    try {
      const r = await fetch("/api/draw/frames?" + q);
      if (!r.ok) throw new Error((await r.json()).error || r.statusText);
      const buf = await r.arrayBuffer();
      if (token !== state.token) return null;
      const h = (n) => r.headers.get(n);
      const chunk = { start: +h("X-Frame-Start"), count: +h("X-Frame-Count"),
        height: +h("X-Frame-Height"), width: +h("X-Frame-Width"), total: +h("X-Frame-Total"),
        dt: +h("X-Frame-Dt-Ms"), low: +h("X-Frame-Min"), high: +h("X-Frame-Max"),
        data: new Float32Array(buf), used: ++state.tick };
      state.chunks.set(chunk.start, chunk);
      state.bytes += buf.byteLength;
      state.low = Math.min(state.low, chunk.low);
      state.high = Math.max(state.high, chunk.high);
      evict();
      return chunk;
    } catch (err) {
      if (token !== state.token) return null;   // the draw changed; nobody is waiting
      throw err;
    }
  }

  function ensureChunk(start) {
    const have = state.chunks.get(start);
    if (have) { have.used = ++state.tick; return Promise.resolve(have); }
    let p = state.pending.get(start);
    if (!p) {
      const token = state.token;
      p = fetchChunk(token, start).finally(() => {
        if (token === state.token) state.pending.delete(start);
      });
      state.pending.set(start, p);
    }
    return p;
  }

  // Drop the least recently used chunks beyond the byte limit, never the one on screen.
  function evict() {
    if (state.bytes <= MAX_BYTES) return;
    const keep = chunkStart(state.frame);
    const order = [...state.chunks.values()].filter((c) => c.start !== keep).sort((a, b) => a.used - b.used);
    for (const c of order) {
      if (state.bytes <= MAX_BYTES) break;
      state.chunks.delete(c.start);
      state.bytes -= c.data.byteLength;
    }
  }

  // The first moment anything fails ends playback and says why; nothing fails silently.
  function fail(err) {
    stop();
    say("error: " + String(err && err.message || err));
  }

  function prefetch(frame) {
    if (!state.meta) return;
    const fps = (1000 / state.meta.dt) * state.speed;
    const ahead = Math.max(2 * CHUNK, Math.min(16 * CHUNK, fps));
    for (let s = chunkStart(frame); s <= frame + ahead && s < state.meta.total; s += CHUNK) {
      if (state.pending.size >= MAX_IN_FLIGHT) break;
      if (!state.chunks.has(s) && !state.pending.has(s)) ensureChunk(s).catch(fail);
    }
  }

  // ---- painting ----

  function scaleFor() {
    const peak = G.peakFor(state.draw && state.draw.frame_peak, state.channel);
    return G.colourScale(peak, state.low, state.high);
  }

  function paint(frame) {
    const chunk = state.chunks.get(chunkStart(frame));
    if (!chunk) return false;
    chunk.used = ++state.tick;
    const nx = chunk.height, ny = chunk.width;   // frame axis 0 is x and axis 1 is y
    const canvas = $("movie");
    if (canvas.width !== nx || canvas.height !== ny) {
      canvas.width = nx; canvas.height = ny; state.image = null;
    }
    const ctx = canvas.getContext("2d");
    if (!state.image) state.image = ctx.createImageData(nx, ny);
    const sc = scaleFor();
    G.toRGBA(chunk.data, (frame - chunk.start) * nx * ny, nx, ny, sc.low, sc.high, state.image.data);
    ctx.putImageData(state.image, 0, 0);
    state.frame = frame;
    $("scrub").value = frame;
    $("clock").textContent = "frame " + frame + " / " + (chunk.total - 1) + " · " + (frame * chunk.dt).toFixed(1) + " ms";
    const text = G.fmt(sc.low) + " … " + G.fmt(sc.high) + " (" + sc.source + ")";
    if (text !== state.scaleText) { state.scaleText = text; $("scale-text").textContent = text; }
    return true;
  }

  function drawAxes() {
    const b = state.draw && state.draw.bundle;
    const t = b && b.layout && b.layout.bounds && b.layout.bounds.length >= 2
      ? G.axisText(b.layout.bounds, b.units && b.units.space) : null;
    for (const id of ["xmin", "xmax", "xlabel", "ymin", "ymax", "ylabel"])
      $(id).textContent = t ? t[id] : "";
  }

  function drawScaleBar() {
    const ctx = $("bar").getContext("2d");
    const img = ctx.createImageData(96, 1);
    for (let i = 0; i < 96; i++) {
      const c = G.ramp(i / 95);
      img.data.set([c[0], c[1], c[2], 255], 4 * i);
    }
    ctx.putImageData(img, 0, 0);
  }

  // ---- playback by wall clock ----

  function stop() {
    state.playing = false;
    cancelAnimationFrame(state.raf);
    $("play").textContent = "Play";
  }

  // Advance the cursor by the wall time since the last refresh, at `speed` times real
  // time; frames the display cannot show are skipped. When the frame the cursor reaches
  // has not arrived, the cursor waits (the buffering time is not counted).
  function step(now) {
    if (!state.playing) return;
    const gap = Math.max(0, Math.min(now - state.last, MAX_STEP_MS));  // rAF's clock can lead performance.now()
    state.last = now;
    const total = state.meta.total;
    const next = G.frameAt(state.cursor, gap, state.speed, state.meta.dt, total);
    const idx = Math.floor(next);
    prefetch(idx);
    if (paint(idx)) { state.cursor = next; $("buffering").hidden = true; }
    else $("buffering").hidden = false;
    state.raf = requestAnimationFrame(step);
  }

  function play() {
    if (!state.meta) return;
    state.playing = true;
    state.cursor = state.frame;
    state.last = performance.now();
    $("play").textContent = "Pause";
    state.raf = requestAnimationFrame(step);
  }

  async function show(frame) {
    const seq = ++state.seq;
    state.frame = frame;
    state.cursor = frame;
    try {
      const chunk = await ensureChunk(chunkStart(frame));
      if (!chunk || seq !== state.seq) return;
      paint(frame);
      prefetch(frame);
    } catch (err) { fail(err); }
  }

  // ---- opening a draw ----

  async function openDraw(entry) {
    stop();
    state.entry = entry;
    state.token += 1;
    state.seq += 1;
    state.chunks = new Map(); state.pending = new Map(); state.bytes = 0;
    state.low = Infinity; state.high = -Infinity;
    state.meta = null; state.draw = null; state.frame = 0; state.cursor = 0;
    state.scaleText = ""; state.image = null;
    mark(entry);
    try { history.replaceState(null, "", "#entry=" + encodeURIComponent(entry)); } catch (e) { /* file:// or sandbox */ }
    $("draw-title").textContent = entry;
    $("info").textContent = "";
    $("scale-text").textContent = "";
    $("buffering").hidden = true;
    say("opening…");
    const token = state.token;
    const t0 = performance.now();
    try {
      const [first, draw] = await Promise.all([
        ensureChunk(0),
        getJSON("/api/draw?entry=" + encodeURIComponent(entry)),
      ]);
      if (!first || token !== state.token) return;
      state.draw = draw;
      state.meta = { total: first.total, dt: first.dt };
      drawAxes();
      paint(0);
      say("first frame after " + Math.round(performance.now() - t0) + " ms");
      const scrub = $("scrub");
      scrub.max = first.total - 1; scrub.value = 0; scrub.disabled = false;
      $("play").disabled = false;
      prefetch(0);
      $("info").textContent = JSON.stringify(draw.record, null, 1);
    } catch (err) {
      if (token === state.token) say(String(err.message || err));
    }
  }

  function openFromHash() {
    const m = /^#entry=(.+)$/.exec(location.hash);
    if (!m) return;
    const entry = decodeURIComponent(m[1]);
    if (entry !== state.entry) openDraw(entry);
  }

  $("play").addEventListener("click", () => { if (state.playing) stop(); else play(); });
  $("scrub").addEventListener("input", (e) => { stop(); show(+e.target.value); });
  $("speed").addEventListener("change", (e) => { state.speed = +e.target.value; prefetch(state.frame); });
  $("more").addEventListener("click", () => loadList(true));
  $("filters").addEventListener("change", () => loadList(false));
  $("f-q").addEventListener("input", () => loadList(false));
  $("f-seed").addEventListener("input", () => loadList(false));
  $("filters").addEventListener("submit", (e) => e.preventDefault());
  $("random").addEventListener("click", async () => {
    const q = query();
    const seed = $("pick-seed").value.trim();
    if (seed) q.set("pick_seed", seed);
    try {
      const pick = await getJSON("/api/draws/random?" + q);
      $("pick-note").textContent = "picked 1 of " + pick.total + " with pick seed " + pick.pick_seed +
        (seed ? "" : " (type it in to repeat)");
      openDraw(pick.draw.entry);
    } catch (err) { say(String(err.message || err)); }
  });
  window.addEventListener("hashchange", openFromHash);

  drawScaleBar();
  loadSummary().then(() => loadList(false)).then(openFromHash)
    .catch((err) => { $("summary").textContent = String(err.message || err); });
})();
