"use strict";
// The evaluation page (roadmap A5b, A6a): a draw list with pickers (filters, draw seed,
// random with a repeatable pick seed) and a movie of the stimulus frames of one draw that
// streams in as it plays. Clicking the movie pins a sensor or a neuron (A6b) and the traces
// under it share one time cursor with the movie; a neuron's membrane voltage is re-simulated
// by the server and shown with its comparison against the stored run. The declared fields of
// the open draw sit under the traces (A6c); "Compare with…" opens a second draw B beside it,
// on one clock and one colour scale, lays B's traces over the pinned point's and sets the two
// declarations side by side. Frame geometry is in geometry.js, trace geometry in traces.js,
// the declared-fields rows in record.js.
(function () {
  const G = window.EvalGeometry;
  const T = window.EvalTraces;
  const R = window.EvalRecord;
  const $ = (id) => document.getElementById(id);
  const CHUNK = 32;          // frames per request; chunk starts are multiples of this
  const PAGE = 100;
  const MAX_BYTES = 128 << 20;   // decoded chunks kept in the browser
  const MAX_IN_FLIGHT = 4;
  const MAX_STEP_MS = 100;   // a hidden tab's long gap must not make the movie jump
  // A store holds the decoded chunks of one draw's movie (draw A's, or draw B's when comparing).
  const newStore = (entry) => ({ entry, chunks: new Map(), pending: new Map(), bytes: 0,
    low: Infinity, high: -Infinity, image: null });
  const state = { offset: 0, total: 0, entry: null, token: 0, seq: 0, tick: 0,
    store: newStore(null), meta: null, draw: null, cmp: null, picking: false,
    scaleText: "", frame: 0, cursor: 0, last: 0,
    playing: false, raf: 0, speed: 1, channel: 0,
    dims: null, view: { kind: "sensors", name: null }, geom: null, field: null, pin: null,
    traceToken: 0, rows: [] };
  const COLOUR_B = "#d9822b";   // draw B's traces and marks
  const PLOT = { left: 62, right: 8, height: 84, pad: 5 };   // trace canvas margins, CSS px

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
      li.addEventListener("click", () => (state.picking ? setCompare(row.entry) : openDraw(row.entry)));
      $("draws").append(li);
    }
    state.offset += page.items.length;
    $("count").textContent = state.total + " matching";
    $("more").hidden = state.offset >= state.total;
    mark(state.entry);
  }

  function mark(entry) {
    const b = state.cmp ? state.cmp.store.entry : null;
    for (const li of $("draws").children) {
      li.setAttribute("aria-current", li.dataset.entry === entry ? "true" : "false");
      li.dataset.b = String(li.dataset.entry === b);
    }
  }

  // ---- chunks: one promise per chunk start, so a chunk is never fetched twice ----

  const chunkStart = (frame) => Math.floor(frame / CHUNK) * CHUNK;

  // A store is current while it is the open draw's or the comparison's; a chunk that arrives
  // for a draw that was closed meanwhile is dropped.
  const current = (store) => state.store === store || (state.cmp !== null && state.cmp.store === store);

  async function fetchChunk(store, start) {
    const q = new URLSearchParams({ entry: store.entry, start, count: CHUNK, channel: state.channel });
    try {
      const r = await fetch("/api/draw/frames?" + q);
      if (!r.ok) throw new Error((await r.json()).error || r.statusText);
      const buf = await r.arrayBuffer();
      if (!current(store)) return null;
      const h = (n) => r.headers.get(n);
      const chunk = { start: +h("X-Frame-Start"), count: +h("X-Frame-Count"),
        height: +h("X-Frame-Height"), width: +h("X-Frame-Width"), total: +h("X-Frame-Total"),
        dt: +h("X-Frame-Dt-Ms"), low: +h("X-Frame-Min"), high: +h("X-Frame-Max"),
        data: new Float32Array(buf), used: ++state.tick };
      store.chunks.set(chunk.start, chunk);
      store.bytes += buf.byteLength;
      store.low = Math.min(store.low, chunk.low);
      store.high = Math.max(store.high, chunk.high);
      evict(store, store === state.store ? state.frame : frameOfB(state.frame));
      return chunk;
    } catch (err) {
      if (!current(store)) return null;   // the draw changed; nobody is waiting
      throw err;
    }
  }

  function ensureChunk(store, start) {
    const have = store.chunks.get(start);
    if (have) { have.used = ++state.tick; return Promise.resolve(have); }
    let p = store.pending.get(start);
    if (!p) {
      p = fetchChunk(store, start).finally(() => {
        if (store.pending.get(start) === p) store.pending.delete(start);
      });
      store.pending.set(start, p);
    }
    return p;
  }

  // Drop the least recently used chunks beyond the byte limit, never the one on screen.
  function evict(store, frame) {
    if (store.bytes <= MAX_BYTES) return;
    const keep = chunkStart(frame);
    const order = [...store.chunks.values()].filter((c) => c.start !== keep).sort((a, b) => a.used - b.used);
    for (const c of order) {
      if (store.bytes <= MAX_BYTES) break;
      store.chunks.delete(c.start);
      store.bytes -= c.data.byteLength;
    }
  }

  // The first moment anything fails ends playback and says why; nothing fails silently.
  function fail(err) {
    stop();
    say("error: " + String(err && err.message || err));
  }

  function prefetchStore(store, frame, total, dt) {
    const fps = (1000 / dt) * state.speed;
    const ahead = Math.max(2 * CHUNK, Math.min(16 * CHUNK, fps));
    for (let s = chunkStart(frame); s <= frame + ahead && s < total; s += CHUNK) {
      if (store.pending.size >= MAX_IN_FLIGHT) break;
      if (!store.chunks.has(s) && !store.pending.has(s)) ensureChunk(store, s).catch(fail);
    }
  }

  function prefetch(frame) {
    if (!state.meta) return;
    prefetchStore(state.store, frame, state.meta.total, state.meta.dt);
    const c = state.cmp;
    if (c && c.meta) prefetchStore(c.store, frameOfB(frame), c.meta.total, c.meta.dt);
  }

  // ---- painting ----

  // The frame of draw B at the time of frame `frame` of draw A (a shorter draw holds its last).
  const frameOfB = (frame) => state.cmp && state.cmp.meta && state.meta
    ? G.matchFrame(frame, state.meta.dt, state.cmp.meta.dt, state.cmp.meta.total) : 0;

  const scaleOf = (store, draw) => G.colourScale(G.peakFor(draw && draw.frame_peak, state.channel), store.low, store.high);

  // One colour scale for both movies while comparing, so equal colours mean equal values.
  function scaleFor() {
    const a = scaleOf(state.store, state.draw);
    const c = state.cmp;
    if (!c || !c.draw) return a;
    const b = scaleOf(c.store, c.draw);
    return { low: Math.min(a.low, b.low), high: Math.max(a.high, b.high),
      source: (a.source === b.source ? a.source : a.source + " / " + b.source) + ", shared by both draws" };
  }

  function paintChunk(store, chunk, frame, canvasId, sc) {
    chunk.used = ++state.tick;
    const nx = chunk.height, ny = chunk.width;   // frame axis 0 is x and axis 1 is y
    const canvas = $(canvasId);
    if (canvas.width !== nx || canvas.height !== ny) {
      canvas.width = nx; canvas.height = ny; store.image = null;
    }
    const ctx = canvas.getContext("2d");
    if (!store.image) store.image = ctx.createImageData(nx, ny);
    G.toRGBA(chunk.data, (frame - chunk.start) * nx * ny, nx, ny, sc.low, sc.high, store.image.data);
    ctx.putImageData(store.image, 0, 0);
    return { nx, ny };
  }

  function paint(frame) {
    const chunk = state.store.chunks.get(chunkStart(frame));
    if (!chunk) return false;
    let other = null, fb = 0;
    if (state.cmp && state.cmp.meta) {
      fb = frameOfB(frame);
      other = state.cmp.store.chunks.get(chunkStart(fb));
      if (!other) return false;   // both movies wait for the slower one
    }
    const sc = scaleFor();
    state.dims = paintChunk(state.store, chunk, frame, "movie", sc);
    if (other) {
      paintChunk(state.cmp.store, other, fb, "movie-b", sc);
      drawOverlay();
    }
    state.frame = frame;
    $("scrub").value = frame;
    moveCursor();
    const bText = other ? " · B frame " + fb + " / " + (other.total - 1) : "";
    $("clock").textContent = "frame " + frame + " / " + (chunk.total - 1) + " · " + (frame * chunk.dt).toFixed(1) + " ms" + bText;
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
      const need = [ensureChunk(state.store, chunkStart(frame))];
      if (state.cmp && state.cmp.meta) need.push(ensureChunk(state.cmp.store, chunkStart(frameOfB(frame))));
      const [chunk] = await Promise.all(need);
      if (!chunk || seq !== state.seq) return;
      paint(frame);
      prefetch(frame);
    } catch (err) { fail(err); }
  }

  // ---- pinning: the overlay on the movie ----

  const bounds = () => {
    const b = state.draw && state.draw.bundle;
    return b && b.layout && b.layout.bounds && b.layout.bounds.length >= 2 ? b.layout.bounds : null;
  };

  function fitOverlay(oid, mid) {
    const o = $(oid), r = $(mid).getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
    if (o.width !== w || o.height !== h) { o.width = w; o.height = h; }
  }

  function drawOverlay() {
    fitOverlay("overlay", "movie");
    const o = $("overlay"), ctx = o.getContext("2d");
    ctx.clearRect(0, 0, o.width, o.height);
    const d = state.dims, b = bounds();
    if (!d || !b) return;
    const sx = o.width / d.nx, sy = o.height / d.ny, dpr = window.devicePixelRatio || 1;
    const at = (x, y) => { const p = G.mmToPixel(x, y, b, d.nx, d.ny); return [p.cx * sx, p.cy * sy]; };
    const pin = state.pin;
    if (state.view.kind === "population" && state.geom) {
      ctx.lineWidth = dpr;
      ctx.strokeStyle = "rgba(255,255,255,0.85)";
      ctx.fillStyle = "rgba(0,0,0,0.55)";
      for (const [x, y] of state.geom.centers) {
        const [px, py] = at(x, y);
        ctx.beginPath(); ctx.arc(px, py, 2.2 * dpr, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
      }
    }
    if (pin && pin.kind === "unit" && state.field) {
      let top = 0;
      for (const blk of state.field.inputs) for (const w of blk.weights) top = Math.max(top, w);
      ctx.fillStyle = "rgba(255,255,255,0.35)";
      const coords = state.draw.bundle.layout.coords;
      for (const blk of state.field.inputs) {
        blk.sensors.forEach((k, n) => {
          const [px, py] = at(coords[k][0], coords[k][1]);
          ctx.globalAlpha = 0.15 + 0.6 * (top > 0 ? blk.weights[n] / top : 1);
          ctx.fillRect(px - sx / 2, py - sy / 2, sx, sy);
        });
      }
      ctx.globalAlpha = 1;
    }
    if (pin) {
      let xy = null;
      if (pin.kind === "sensor") xy = [(pin.i + 0.5) * sx, (G.pixelOf(pin.i, pin.j, d.nx, d.ny).cy + 0.5) * sy];
      else if (state.geom) { const c = state.geom.centers[pin.index]; xy = at(c[0], c[1]); }
      if (xy) {
        ctx.lineWidth = 2 * dpr;
        ctx.strokeStyle = "#ff3b30";
        ctx.beginPath(); ctx.arc(xy[0], xy[1], 6 * dpr, 0, 2 * Math.PI); ctx.stroke();
      }
    }
    drawOverlayB();
  }

  // The pinned point on draw B's movie, in B's colour, when the pin means the same in B.
  function drawOverlayB() {
    fitOverlay("overlay-b", "movie-b");
    const o = $("overlay-b"), ctx = o.getContext("2d");
    ctx.clearRect(0, 0, o.width, o.height);
    const c = state.cmp, pin = state.pin, dpr = window.devicePixelRatio || 1;
    if (!c || !c.draw || !c.draw.bundle || !pin || c.problem) return;
    const nx = $("movie-b").width, ny = $("movie-b").height, sx = o.width / nx, sy = o.height / ny;
    let xy = null;
    if (pin.kind === "sensor") xy = [(pin.i + 0.5) * sx, (G.pixelOf(pin.i, pin.j, nx, ny).cy + 0.5) * sy];
    else if (state.geom) {
      const bb = c.draw.bundle.layout.bounds, m = state.geom.centers[pin.index];
      if (bb && bb.length >= 2 && m) { const p = G.mmToPixel(m[0], m[1], bb, nx, ny); xy = [p.cx * sx, p.cy * sy]; }
    }
    if (!xy) return;
    ctx.lineWidth = 2 * dpr;
    ctx.strokeStyle = COLOUR_B;
    ctx.beginPath(); ctx.arc(xy[0], xy[1], 6 * dpr, 0, 2 * Math.PI); ctx.stroke();
  }

  function pinText() {
    const p = state.pin;
    if (!p) return "Click the movie to pin a " + (state.view.kind === "sensors" ? "sensor" : "neuron") + ".";
    if (p.kind === "sensor") return "Pinned sensor " + p.k + " (x index " + p.i + ", y index " + p.j + ")";
    const c = state.geom && state.geom.centers[p.index];
    const u = state.draw.bundle.units && state.draw.bundle.units.space;
    return "Pinned " + state.view.name + " neuron " + p.index +
      (c ? " at (" + G.fmt(c[0]) + ", " + G.fmt(c[1]) + ")" + (u ? " " + u : "") : "");
  }

  function setPin(pin) {
    state.pin = pin;
    state.field = null;
    $("pin-text").textContent = pinText();
    updateProblem();
    drawOverlay();
    loadTraces();
  }

  function onMovieClick(e) {
    const d = state.dims, b = bounds();
    if (!d || !state.draw || !b) return;
    const r = $("movie").getBoundingClientRect();
    const cx = ((e.clientX - r.left) / r.width) * d.nx, cy = ((e.clientY - r.top) / r.height) * d.ny;
    if (state.view.kind === "sensors") {
      const s = G.sensorAt(cx, cy, d.nx, d.ny);
      if (s) setPin({ kind: "sensor", i: s.i, j: s.j, k: s.k });
    } else if (state.geom) {
      const p = G.pixelToMm(cx, cy, b, d.nx, d.ny);
      const n = G.nearestUnit(p.x, p.y, state.geom.centers);
      if (n >= 0) setPin({ kind: "unit", index: n });
    }
  }

  async function setView(value) {
    const token = state.token;
    state.pin = null; state.field = null; state.geom = null;
    if (value === "sensors") state.view = { kind: "sensors", name: null };
    else {
      state.view = { kind: "population", name: value };
      try {
        const g = await getJSON("/api/draw/population?entry=" + encodeURIComponent(state.entry) +
          "&population=" + encodeURIComponent(value));
        if (token !== state.token) return;
        state.geom = g;
      } catch (err) { say("error: " + err.message); state.view = { kind: "sensors", name: null }; $("view").value = "sensors"; }
    }
    $("pin-text").textContent = pinText();
    updateProblem();
    drawOverlay();
    loadTraces();
  }

  function fillViews() {
    const sel = $("view");
    const want = state.view.kind === "population" ? state.view.name : "sensors";
    sel.replaceChildren(new Option("sensor", "sensors"));
    const pops = (state.draw.bundle && state.draw.bundle.populations) || [];
    for (const p of pops) sel.add(new Option(p.name + " neuron (" + p.n_units + ")", p.name));
    sel.disabled = !state.draw.bundle;
    sel.value = pops.some((p) => p.name === want) ? want : "sensors";
    return sel.value;
  }

  // ---- traces under the movie, with one shared time cursor ----

  function plotWidth() { return Math.max(200, $("tracebox").clientWidth); }

  function newRow(title, kind) {
    const wrap = document.createElement("div");
    wrap.className = "trace";
    const h = document.createElement("h3");
    h.textContent = title;
    const canvas = document.createElement("canvas");
    const note = document.createElement("p");
    note.className = "note";
    note.textContent = "loading…";
    wrap.append(h, canvas, note);
    $("tracebox").insertBefore(wrap, $("cursor"));
    const row = { title, kind, canvas, note, series: null, other: null, unit: "" };
    canvas.addEventListener("click", (e) => {
      const r = canvas.getBoundingClientRect();
      if (!state.meta) return;
      stop();
      show(T.frameOfX(e.clientX - r.left, r.width, PLOT.left, PLOT.right, state.meta.dt, state.meta.total));
    });
    state.rows.push(row);
    return row;
  }

  function clearTraces() {
    for (const r of state.rows) r.canvas.parentNode.remove();
    state.rows = [];
    moveCursor();
  }

  function drawRow(row) {
    const w = plotWidth(), h = PLOT.height, dpr = window.devicePixelRatio || 1;
    const c = row.canvas;
    c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
    c.style.height = h + "px";
    const ctx = c.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const s = row.series, o = row.other;
    if (!s || !s.length) return;
    const css = getComputedStyle(document.documentElement);
    const fg = css.getPropertyValue("--fg").trim() || "#222", muted = css.getPropertyValue("--muted").trim() || "#888";
    const t1 = (state.meta.total - 1) * state.meta.dt;
    ctx.font = "11px system-ui, sans-serif";
    ctx.fillStyle = muted;
    ctx.textAlign = "right";
    const lines = [[s[0], fg]];
    if (o && o.length) lines.push([o[0], COLOUR_B]);
    if (row.kind === "ticks") {
      // with two draws, A's ticks fill the upper half and B's the lower
      ctx.fillText(row.unit, PLOT.left - 6, h / 2 + 4);
      ctx.lineWidth = 1;
      const lane = (h - 2 * PLOT.pad) / lines.length;
      lines.forEach(([one, colour], n) => {
        ctx.strokeStyle = colour;
        const hits = T.nonzero(one.time_ms, one.values);
        const top = PLOT.pad + n * lane + 3, bottom = PLOT.pad + (n + 1) * lane - 3;
        ctx.beginPath();
        for (let m = 0; m < hits.times.length; m++) {
          const x = T.xOfTime(hits.times[m], 0, t1, w, PLOT.left, PLOT.right);
          ctx.moveTo(x, top); ctx.lineTo(x, bottom);
        }
        ctx.stroke();
      });
      return;
    }
    const all = [];
    for (const [one] of lines) for (const v of one.values) all.push(v);
    const rg = T.range(all);
    const view = { t0: 0, t1, lo: rg.lo, hi: rg.hi, w, h, left: PLOT.left, right: PLOT.right, pad: PLOT.pad };
    ctx.fillText(G.fmt(rg.hi), PLOT.left - 6, PLOT.pad + 9);
    ctx.fillText(G.fmt(rg.lo), PLOT.left - 6, h - PLOT.pad);
    if (row.unit) ctx.fillText(row.unit, PLOT.left - 6, h / 2 + 4);
    ctx.lineWidth = 1.2; ctx.lineJoin = "round";
    for (const [one, colour] of lines) {
      ctx.strokeStyle = colour;
      const pts = T.project(one.time_ms, one.values, view);
      ctx.beginPath();
      let pen = false;
      for (const p of pts) {
        if (!p) { pen = false; continue; }
        if (pen) ctx.lineTo(p[0], p[1]); else { ctx.moveTo(p[0], p[1]); pen = true; }
      }
      ctx.stroke();
    }
  }

  function redrawTraces() { for (const r of state.rows) drawRow(r); drawOverlay(); moveCursor(); }

  function moveCursor() {
    const cur = $("cursor");
    if (!state.meta || !state.rows.length) { cur.hidden = true; return; }
    cur.hidden = false;
    const w = plotWidth(), t1 = (state.meta.total - 1) * state.meta.dt;
    cur.style.left = T.xOfTime(state.frame * state.meta.dt, 0, t1, w, PLOT.left, PLOT.right) + "px";
  }

  function traceURL(source, extra, entry) {
    const q = new URLSearchParams({ entry: entry || state.entry, source, ...extra });
    return "/api/draw/trace?" + q;
  }

  // Draw B's trace for the same point, laid over the row; a failure is said under the row.
  async function fillOther(row, url, alive) {
    try {
      const body = await getJSON(url);
      if (!alive()) return null;
      row.other = body.series;
      drawRow(row);
      return body;
    } catch (err) {
      if (alive()) { row.note.textContent += (row.note.textContent ? " · " : "") + "B: " + String(err.message || err); row.note.className = "note bad"; }
      return null;
    }
  }

  function addLegend(row) {
    const h = row.canvas.parentNode.querySelector("h3");
    h.append(" · ");
    const a = document.createElement("span"); a.className = "legend-a"; a.textContent = "A";
    const b = document.createElement("span"); b.className = "legend-b"; b.textContent = "B";
    h.append(a, " and ", b);
  }

  async function fill(row, url, unit, note) {
    try {
      const body = await getJSON(url);
      row.series = body.series; row.unit = unit || "";
      row.note.textContent = note || "";
      row.note.className = "note";
      drawRow(row);
      return body;
    } catch (err) {
      row.note.textContent = String(err.message || err);
      row.note.className = "note bad";
      return null;
    }
  }

  function describeCheck(c) {
    const shas = "bundle code " + String(c.bundle_sha || "unknown").slice(0, 8) + ", running code " +
      String(c.code_sha || "unknown").slice(0, 8) + (c.bundle_sha && c.code_sha && c.bundle_sha !== c.code_sha ? " (different code)" : "");
    if (c.matches)
      return { text: "matches the stored run: the re-run gives the same " + c.readout + " for all " + c.n_units +
        " neurons · " + shas + " · re-run " + c.seconds + " s", cls: "note good" };
    const first = c.first_differences.length ? " (first: " + c.first_differences.join(", ") + ")" : "";
    return { text: "differs from the stored run (" + c.n_differing + " of " + c.n_units + " neurons" + first +
      "; largest difference " + G.fmt(c.max_abs_difference) + ") · " + shas + " · re-run " + c.seconds + " s" +
      (c.notes && c.notes.length ? " · " + c.notes.join("; ") : ""), cls: "note bad" };
  }

  async function loadTraces() {
    const token = ++state.traceToken;
    clearTraces();
    const pin = state.pin, b = state.draw && state.draw.bundle;
    if (!pin || !b) return;
    const units = b.units || {};
    const alive = () => token === state.traceToken;
    const c = state.cmp;
    const bEntry = c && c.draw && c.draw.bundle && !c.problem ? c.store.entry : null;
    if (c && c.problem) say("B's traces are not shown: " + c.problem);
    const mk = (title, kind) => { const r = newRow(title, kind); if (bEntry) addLegend(r); return r; };
    if (pin.kind === "sensor") {
      const row = mk("Stimulus at sensor " + pin.k + " (channel " + state.channel + ")", "line");
      const q = { index: pin.k, channel: state.channel };
      await Promise.all([fill(row, traceURL("stimulus", q), units.frame || ""),
        bEntry ? fillOther(row, traceURL("stimulus", q, bEntry), alive) : null]);
      return;
    }
    const pop = state.view.name, idx = pin.index;
    const readout = b.populations.find((p) => p.name === pop).readout;
    const drive = mk("Drive (neuron " + idx + ")", "line");
    const filt = mk("Filtered current (neuron " + idx + ")", "line");
    const out = mk(readout[0].toUpperCase() + readout.slice(1) + " (neuron " + idx + ")", readout === "state" ? "line" : "ticks");
    const volt = mk("Membrane voltage (neuron " + idx + "), re-simulated", "line");
    volt.note.textContent = "re-simulating the population from its stored filtered current…";
    const q = { population: pop, index: idx };
    fieldFor(token, pop, idx);
    const outUnit = readout === "spikes" ? "spikes" : readout === "events" ? "events" : (units.state || "state");
    await Promise.all([
      fill(drive, traceURL("drive", q), units.drive || ""),
      fill(filt, traceURL("filtered", q), units.filtered || ""),
      fill(out, traceURL("readout", q), outUnit),
      bEntry ? fillOther(drive, traceURL("drive", q, bEntry), alive) : null,
      bEntry ? fillOther(filt, traceURL("filtered", q, bEntry), alive) : null,
      bEntry ? fillOther(out, traceURL("readout", q, bEntry), alive) : null,
    ]);
    if (!alive()) return;
    const voltage = async (entry) => getJSON("/api/draw/voltage?" + new URLSearchParams({ entry, ...q }));
    const [ra, rb] = await Promise.allSettled([voltage(state.entry), bEntry ? voltage(bEntry) : Promise.resolve(null)]);
    if (!alive()) return;
    const notes = [];
    let cls = "note good";
    if (ra.status === "fulfilled") {
      volt.series = ra.value.series; volt.unit = "mV";
      const d = describeCheck(ra.value.verification);
      notes.push((bEntry ? "A " : "") + d.text);
      if (d.cls.includes("bad")) cls = "note bad";
    } else { notes.push(String(ra.reason.message || ra.reason)); cls = "note bad"; }
    if (rb.status === "fulfilled" && rb.value) {
      volt.other = rb.value.series;
      const d = describeCheck(rb.value.verification);
      notes.push("B " + d.text);
      if (d.cls.includes("bad")) cls = "note bad";
    } else if (rb.status === "rejected") { notes.push("B: " + String(rb.reason.message || rb.reason)); cls = "note bad"; }
    volt.note.textContent = notes.join(" · "); volt.note.className = cls;
    drawRow(volt);
    moveCursor();
  }

  async function fieldFor(token, pop, idx) {
    try {
      const g = await getJSON("/api/draw/population?" + new URLSearchParams({ entry: state.entry, population: pop, index: idx }));
      if (token !== state.traceToken) return;
      state.field = g.field;
      drawOverlay();
    } catch (err) { /* the traces report their own errors; the field is a bonus */ }
  }

  // ---- opening a draw ----

  async function openDraw(entry) {
    stop();
    state.entry = entry;
    state.token += 1;
    state.seq += 1;
    if (state.cmp && state.cmp.store.entry === entry) resetCompare();   // A and B cannot be one draw
    state.store = newStore(entry);
    state.meta = null; state.draw = null; state.frame = 0; state.cursor = 0;
    state.scaleText = ""; state.dims = null;
    state.geom = null; state.field = null;
    clearTraces();
    mark(entry);
    writeHash();
    $("draw-title").textContent = entry;
    $("fields-body").replaceChildren();
    $("scale-text").textContent = "";
    $("buffering").hidden = true;
    say("opening…");
    const token = state.token;
    const t0 = performance.now();
    try {
      const [first, draw] = await Promise.all([
        ensureChunk(state.store, 0),
        getJSON("/api/draw?entry=" + encodeURIComponent(entry)),
      ]);
      if (!first || token !== state.token) return;
      state.draw = draw;
      state.meta = { total: first.total, dt: first.dt };
      drawAxes();
      if (state.cmp && state.cmp.meta) await ensureChunk(state.cmp.store, 0);   // paint() waits for both movies
      if (token !== state.token) return;
      paint(0);
      const want = fillViews();
      if (want === "sensors") {
        state.view = { kind: "sensors", name: null };
        if (state.pin && state.pin.kind !== "sensor") state.pin = null;
      } else await setViewKeepPin(want);
      if (token !== state.token) return;
      $("pin-text").textContent = pinText();
      drawOverlay();
      if (state.pin) loadTraces();
      say("first frame after " + Math.round(performance.now() - t0) + " ms");
      const scrub = $("scrub");
      scrub.max = first.total - 1; scrub.value = 0; scrub.disabled = false;
      $("play").disabled = false;
      prefetch(0);
      $("compare-pick").disabled = false;
      renderFields();
      if (state.cmp) refreshCompare();
    } catch (err) {
      if (token === state.token) say(String(err.message || err));
    }
  }

  // ---- the declared fields, and a second draw to compare with ----

  function cell(tag, text, cls) {
    const el = document.createElement(tag);
    el.textContent = text;
    if (cls) el.className = cls;
    return el;
  }

  function renderFields() {
    const body = $("fields-body");
    body.replaceChildren();
    const d = state.draw;
    if (!d) return;
    const c = state.cmp && state.cmp.draw;
    $("fields-tools").hidden = !c;
    if (d.error) body.append(cell("p", "The bundle could not be read: " + d.error, "note bad"));
    const a = R.rows(d.record, d.seeds);
    let list;
    if (c) {
      list = R.compare(a, R.rows(c.record, c.seeds));
      if ($("only-diff").checked) list = list.filter((r) => !r.same);
      if (!list.length) { body.append(cell("p", "Every declared field is the same in both draws.", "muted")); return; }
    } else list = a;
    for (const g of R.grouped(list)) {
      const table = document.createElement("table");
      const head = document.createElement("tr");
      head.append(cell("th", g.group));
      if (c) head.append(cell("th", "A"), cell("th", "B"));
      else head.append(cell("th", ""));
      table.append(head);
      for (const r of g.rows) {
        const tr = document.createElement("tr");
        tr.append(cell("td", r.path.replace(/^draw\.values\./, "").replace(/^draw\./, ""), "path"));
        if (c) {
          tr.append(cell("td", r.a === null ? "not in this draw" : r.a), cell("td", r.b === null ? "not in this draw" : r.b, "b"));
          if (!r.same) tr.className = "differs";
        } else tr.append(cell("td", r.text));
        table.append(tr);
      }
      body.append(table);
    }
  }

  function writeHash() {
    const q = new URLSearchParams();
    if (state.entry) q.set("entry", state.entry);
    if (state.cmp) q.set("vs", state.cmp.store.entry);
    try { history.replaceState(null, "", "#" + q); } catch (e) { /* file:// or sandbox */ }
  }

  // Whether the pinned point means the same in B; traces of B are laid over A's only if so.
  function updateProblem() {
    const c = state.cmp;
    if (!c || !c.draw) return;
    c.problem = R.pinProblem(state.pin, state.view.name, state.draw.bundle, c.draw.bundle);
  }

  function refreshCompare() {
    const c = state.cmp;
    if (!c || !c.draw || !state.draw) return;
    updateProblem();
    const shorter = c.meta.total * c.meta.dt < state.meta.total * state.meta.dt
      ? " B is shorter and holds its last frame." : "";
    $("b-caption").replaceChildren(cell("span", "B", "legend-b"), " · " + c.store.entry);
    $("compare-clear").hidden = false;
    $("compare-text").textContent = "B (orange) is " + c.store.entry + ": one clock, one colour scale." + shorter;
    mark(state.entry);
    renderFields();
    writeHash();
    drawOverlay();
    loadTraces();
  }

  async function setCompare(entry) {
    state.picking = false;
    $("draws").classList.remove("picking");
    if (!state.draw) return;
    if (entry === state.entry) { $("compare-text").textContent = "B must be a different draw than A."; return; }
    stop();
    const c = { store: newStore(entry), draw: null, meta: null, problem: null };
    state.cmp = c;
    $("compare-text").textContent = "loading " + entry + "…";
    try {
      const [first, draw] = await Promise.all([
        ensureChunk(c.store, 0),
        getJSON("/api/draw?entry=" + encodeURIComponent(entry)),
      ]);
      if (state.cmp !== c || !first) return;
      c.draw = draw;
      c.meta = { total: first.total, dt: first.dt };
      $("stage-b").hidden = false;
      $("stage-row").classList.add("two");
      state.scaleText = "";
      refreshCompare();
      show(state.frame);
    } catch (err) {
      if (state.cmp !== c) return;
      clearCompare();
      say("error: " + String(err.message || err));
    }
  }

  function resetCompare() {
    state.picking = false;
    $("draws").classList.remove("picking");
    state.cmp = null;
    $("stage-b").hidden = true;
    $("stage-row").classList.remove("two");
    $("compare-clear").hidden = true;
    $("compare-text").textContent = "";
    state.scaleText = "";
  }

  function clearCompare() {
    resetCompare();
    mark(state.entry);
    renderFields();
    writeHash();
    if (state.meta) show(state.frame);
    drawOverlay();
    loadTraces();
  }

  // Re-open the population view of the previous draw on a new draw, keeping the pinned neuron.
  async function setViewKeepPin(name) {
    const keep = state.pin && state.pin.kind === "unit" ? state.pin : null;
    await setView(name);
    if (keep && state.geom && keep.index < state.geom.centers.length) state.pin = keep;
  }

  function openFromHash() {
    const q = new URLSearchParams(location.hash.slice(1));
    const entry = q.get("entry"), vs = q.get("vs");
    if (!entry) return;
    const wantB = vs && vs !== entry ? vs : null;
    const haveB = state.cmp ? state.cmp.store.entry : null;
    if (entry !== state.entry) {
      // a draw opened from a link brings its comparison along, once A has loaded
      openDraw(entry).then(() => { if (wantB) return setCompare(wantB); });
    } else if (wantB !== haveB) { if (wantB) setCompare(wantB); else clearCompare(); }
  }

  $("compare-pick").addEventListener("click", () => {
    state.picking = !state.picking;
    $("draws").classList.toggle("picking", state.picking);
    $("compare-text").textContent = state.picking ? "Click a draw in the list to compare with, or click here again to cancel." : "";
  });
  $("compare-clear").addEventListener("click", clearCompare);
  $("only-diff").addEventListener("change", renderFields);
  $("play").addEventListener("click", () => { if (state.playing) stop(); else play(); });
  $("scrub").addEventListener("input", (e) => { stop(); show(+e.target.value); });
  $("speed").addEventListener("change", (e) => { state.speed = +e.target.value; prefetch(state.frame); });
  $("movie").addEventListener("click", onMovieClick);
  $("view").addEventListener("change", (e) => setView(e.target.value));
  window.addEventListener("resize", redrawTraces);
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
