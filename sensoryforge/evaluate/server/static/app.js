"use strict";
// The evaluation page (roadmap A5b, A6a): a draw list with pickers (filters, draw seed,
// random with a repeatable pick seed) and a movie of the stimulus frames of one draw that
// streams in as it plays. Clicking the movie pins a sensor or a neuron (A6b) and the traces
// under it share one time cursor with the movie; a neuron's membrane voltage is re-simulated
// by the server and shown with its comparison against the stored run. The declared-fields
// panel and two-draw comparison arrive with A6c. Frame geometry is in geometry.js, trace
// geometry in traces.js.
(function () {
  const G = window.EvalGeometry;
  const T = window.EvalTraces;
  const $ = (id) => document.getElementById(id);
  const CHUNK = 32;          // frames per request; chunk starts are multiples of this
  const PAGE = 100;
  const MAX_BYTES = 128 << 20;   // decoded chunks kept in the browser
  const MAX_IN_FLIGHT = 4;
  const MAX_STEP_MS = 100;   // a hidden tab's long gap must not make the movie jump
  const state = { offset: 0, total: 0, entry: null, token: 0, seq: 0, tick: 0,
    chunks: new Map(), pending: new Map(), bytes: 0, meta: null, draw: null,
    low: Infinity, high: -Infinity, scaleText: "", frame: 0, cursor: 0, last: 0,
    playing: false, raf: 0, speed: 1, image: null, channel: 0,
    dims: null, view: { kind: "sensors", name: null }, geom: null, field: null, pin: null,
    traceToken: 0, rows: [] };
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
    state.dims = { nx, ny };
    state.frame = frame;
    $("scrub").value = frame;
    moveCursor();
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

  // ---- pinning: the overlay on the movie ----

  const bounds = () => {
    const b = state.draw && state.draw.bundle;
    return b && b.layout && b.layout.bounds && b.layout.bounds.length >= 2 ? b.layout.bounds : null;
  };

  function fitOverlay() {
    const o = $("overlay"), r = $("movie").getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
    if (o.width !== w || o.height !== h) { o.width = w; o.height = h; }
  }

  function drawOverlay() {
    fitOverlay();
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
    const row = { title, kind, canvas, note, series: null, unit: "" };
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
    const s = row.series;
    if (!s || !s.length) return;
    const css = getComputedStyle(document.documentElement);
    const fg = css.getPropertyValue("--fg").trim() || "#222", muted = css.getPropertyValue("--muted").trim() || "#888";
    const t1 = (state.meta.total - 1) * state.meta.dt;
    ctx.font = "11px system-ui, sans-serif";
    ctx.fillStyle = muted;
    ctx.textAlign = "right";
    if (row.kind === "ticks") {
      const view = { t0: 0, t1, lo: 0, hi: 1, w, h, left: PLOT.left, right: PLOT.right, pad: PLOT.pad };
      ctx.fillText(row.unit, PLOT.left - 6, h / 2 + 4);
      ctx.strokeStyle = fg; ctx.lineWidth = 1;
      const hits = T.nonzero(s[0].time_ms, s[0].values);
      ctx.beginPath();
      for (let n = 0; n < hits.times.length; n++) {
        const x = T.xOfTime(hits.times[n], 0, t1, w, PLOT.left, PLOT.right);
        ctx.moveTo(x, view.pad + 8); ctx.lineTo(x, h - view.pad - 8);
      }
      ctx.stroke();
      return;
    }
    const all = [];
    for (const one of s) for (const v of one.values) all.push(v);
    const rg = T.range(all);
    const view = { t0: 0, t1, lo: rg.lo, hi: rg.hi, w, h, left: PLOT.left, right: PLOT.right, pad: PLOT.pad };
    ctx.fillText(G.fmt(rg.hi), PLOT.left - 6, PLOT.pad + 9);
    ctx.fillText(G.fmt(rg.lo), PLOT.left - 6, h - PLOT.pad);
    if (row.unit) ctx.fillText(row.unit, PLOT.left - 6, h / 2 + 4);
    ctx.strokeStyle = fg; ctx.lineWidth = 1.2; ctx.lineJoin = "round";
    const pts = T.project(s[0].time_ms, s[0].values, view);
    ctx.beginPath();
    let pen = false;
    for (const p of pts) {
      if (!p) { pen = false; continue; }
      if (pen) ctx.lineTo(p[0], p[1]); else { ctx.moveTo(p[0], p[1]); pen = true; }
    }
    ctx.stroke();
  }

  function redrawTraces() { for (const r of state.rows) drawRow(r); drawOverlay(); moveCursor(); }

  function moveCursor() {
    const cur = $("cursor");
    if (!state.meta || !state.rows.length) { cur.hidden = true; return; }
    cur.hidden = false;
    const w = plotWidth(), t1 = (state.meta.total - 1) * state.meta.dt;
    cur.style.left = T.xOfTime(state.frame * state.meta.dt, 0, t1, w, PLOT.left, PLOT.right) + "px";
  }

  function traceURL(source, extra) {
    const q = new URLSearchParams({ entry: state.entry, source, ...extra });
    return "/api/draw/trace?" + q;
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
    if (pin.kind === "sensor") {
      const row = newRow("Stimulus at sensor " + pin.k + " (channel " + state.channel + ")", "line");
      await fill(row, traceURL("stimulus", { index: pin.k, channel: state.channel }), units.frame || "");
      return;
    }
    const pop = state.view.name, idx = pin.index;
    const readout = b.populations.find((p) => p.name === pop).readout;
    const mk = (source, title, kind) => newRow(title, kind);
    const drive = mk("drive", "Drive (neuron " + idx + ")", "line");
    const filt = mk("filtered", "Filtered current (neuron " + idx + ")", "line");
    const out = mk("readout", readout[0].toUpperCase() + readout.slice(1) + " (neuron " + idx + ")", readout === "state" ? "line" : "ticks");
    const volt = mk("voltage", "Membrane voltage (neuron " + idx + "), re-simulated", "line");
    volt.note.textContent = "re-simulating the population from its stored filtered current…";
    const q = { population: pop, index: idx };
    fieldFor(token, pop, idx);
    await Promise.all([
      fill(drive, traceURL("drive", q), units.drive || ""),
      fill(filt, traceURL("filtered", q), units.filtered || ""),
      fill(out, traceURL("readout", q), readout === "spikes" ? "spikes" : readout === "events" ? "events" : (units.state || "state")),
    ]);
    if (!alive()) return;
    try {
      const body = await getJSON("/api/draw/voltage?" + new URLSearchParams({ entry: state.entry, ...q }));
      if (!alive()) return;
      volt.series = body.series; volt.unit = "mV";
      const d = describeCheck(body.verification);
      volt.note.textContent = d.text; volt.note.className = d.cls;
      drawRow(volt);
    } catch (err) {
      if (!alive()) return;
      volt.note.textContent = String(err.message || err); volt.note.className = "note bad";
    }
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
    state.chunks = new Map(); state.pending = new Map(); state.bytes = 0;
    state.low = Infinity; state.high = -Infinity;
    state.meta = null; state.draw = null; state.frame = 0; state.cursor = 0;
    state.scaleText = ""; state.image = null; state.dims = null;
    state.geom = null; state.field = null;
    clearTraces();
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
      $("info").textContent = JSON.stringify(draw.record, null, 1);
    } catch (err) {
      if (token === state.token) say(String(err.message || err));
    }
  }

  // Re-open the population view of the previous draw on a new draw, keeping the pinned neuron.
  async function setViewKeepPin(name) {
    const keep = state.pin && state.pin.kind === "unit" ? state.pin : null;
    await setView(name);
    if (keep && state.geom && keep.index < state.geom.centers.length) state.pin = keep;
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
