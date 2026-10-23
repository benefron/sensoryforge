"use strict";
// Pure helpers for the linked trace panel (roadmap A6b): no DOM, so the node test can run
// them. A trace is drawn over the whole draw, time on x, with margins `left` and `right`
// pixels inside a canvas `w` pixels wide; one shared time cursor crosses every trace.
(function (root) {
  // Range of the finite values (null and NaN are gaps), padded when flat, or [0, 1].
  function range(values) {
    let lo = Infinity, hi = -Infinity;
    for (const v of values) {
      if (v === null || v === undefined || !isFinite(v)) continue;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
    if (!(hi >= lo)) return { lo: 0, hi: 1 };
    if (hi === lo) { const p = Math.abs(lo) > 0 ? Math.abs(lo) * 0.1 : 0.5; return { lo: lo - p, hi: hi + p }; }
    return { lo, hi };
  }

  // Pixel x of time t on a plot spanning [t0, t1] inside margins.
  function xOfTime(t, t0, t1, w, left, right) {
    const span = w - left - right;
    return left + (t1 > t0 ? (t - t0) / (t1 - t0) : 0) * span;
  }

  // Time under pixel x, clamped to [t0, t1].
  function timeOfX(x, t0, t1, w, left, right) {
    const span = w - left - right;
    const f = span > 0 ? Math.max(0, Math.min(1, (x - left) / span)) : 0;
    return t0 + f * (t1 - t0);
  }

  // The frame under pixel x for `total` frames `dt` apart starting at 0.
  function frameOfX(x, w, left, right, dt, total) {
    const t = timeOfX(x, 0, (total - 1) * dt, w, left, right);
    return Math.max(0, Math.min(total - 1, Math.round(t / dt)));
  }

  // Points [x, y] (pixels, y downwards) of a series in a plot `h` pixels tall with `pad`
  // pixels above and below; a gap is a null entry, so the line breaks there.
  function project(times, values, view) {
    const { t0, t1, lo, hi, w, h, left, right, pad } = view;
    const out = new Array(values.length);
    const inner = h - 2 * pad;
    for (let n = 0; n < values.length; n++) {
      const v = values[n];
      if (v === null || v === undefined || !isFinite(v)) { out[n] = null; continue; }
      const f = hi > lo ? (v - lo) / (hi - lo) : 0.5;
      out[n] = [xOfTime(times[n], t0, t1, w, left, right), pad + (1 - f) * inner];
    }
    return out;
  }

  // The times and values where a count series is non-zero (a spike or event ticks).
  function nonzero(times, values) {
    const t = [], v = [];
    for (let n = 0; n < values.length; n++) if (values[n]) { t.push(times[n]); v.push(values[n]); }
    return { times: t, values: v };
  }

  const api = { range, xOfTime, timeOfX, frameOfX, project, nonzero };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.EvalTraces = api;
})(typeof window !== "undefined" ? window : globalThis);
