"use strict";
// Frame geometry for the movie (roadmap A6a): pure functions with no DOM, so the node
// test can run them. A stored frame is [nx, ny] row-major: axis 0 is x and axis 1 is y
// ("ij" indexing, sensor k = i * ny + j, the order of /api/draw's layout coordinates).
// The canvas puts x to the right and y upwards, so canvas column cx = i and canvas row
// cy = ny - 1 - j. A6b maps clicks back to sensors with sensorAt().
(function (root) {
  const STOPS = [[0, 68, 1, 84], [0.25, 59, 82, 139], [0.5, 33, 145, 140],
    [0.75, 94, 201, 98], [1, 253, 231, 37]];
  const LUT = new Uint8ClampedArray(256 * 3);
  for (let n = 0; n < 256; n++) {
    const t = n / 255;
    let k = 0;
    while (k < STOPS.length - 2 && t > STOPS[k + 1][0]) k++;
    const a = STOPS[k], b = STOPS[k + 1], u = (t - a[0]) / (b[0] - a[0]);
    for (let c = 0; c < 3; c++) LUT[3 * n + c] = Math.round(a[c + 1] + u * (b[c + 1] - a[c + 1]));
  }

  // Canvas pixel of the frame value (i, j) of a frame with nx x-samples and ny y-samples.
  function pixelOf(i, j, nx, ny) { return { cx: i, cy: ny - 1 - j }; }

  // The frame value (i, j) and the sensor index under a canvas pixel, or null outside.
  function sensorAt(cx, cy, nx, ny) {
    if (!(cx >= 0 && cx < nx && cy >= 0 && cy < ny)) return null;
    const i = Math.floor(cx), j = ny - 1 - Math.floor(cy);
    return { i, j, k: i * ny + j };
  }

  // Fill `out` (RGBA, nx * ny pixels) from the frame at data[offset ...]; returns it.
  function toRGBA(data, offset, nx, ny, low, high, out) {
    const px = out || new Uint8ClampedArray(nx * ny * 4);
    const span = high > low ? high - low : 1;
    for (let i = 0; i < nx; i++) {
      for (let j = 0; j < ny; j++) {
        const v = data[offset + i * ny + j];
        const t = v === v ? Math.max(0, Math.min(1, (v - low) / span)) : 0;
        const l = 3 * Math.round(255 * t);
        const p = 4 * ((ny - 1 - j) * nx + i);
        px[p] = LUT[l]; px[p + 1] = LUT[l + 1]; px[p + 2] = LUT[l + 2]; px[p + 3] = 255;
      }
    }
    return px;
  }

  // The colour of a value in [0, 1], as [r, g, b] (for the scale bar).
  function ramp(t) {
    const l = 3 * Math.round(255 * Math.max(0, Math.min(1, t)));
    return [LUT[l], LUT[l + 1], LUT[l + 2]];
  }

  // The colour range of the movie. `peak` is the audit's core.frames.peak of the draw
  // (or null); `runLow` and `runHigh` are the range of the chunks fetched so far.
  function colourScale(peak, runLow, runHigh) {
    if (peak !== null && peak !== undefined && isFinite(peak)) {
      const low = Math.min(0, runLow);
      if (peak > low) return { low, high: peak, source: "audit peak" };
    }
    if (!(runHigh >= runLow)) return { low: 0, high: 1, source: "running range" };
    return { low: runLow, high: runHigh, source: "running range" };
  }

  // The audit's peak for a channel from /api/draw's frame_peak, or null.
  function peakFor(framePeak, channel) {
    if (!framePeak) return null;
    for (const v of framePeak.values) if (v.channel === channel) return v.value;
    for (const v of framePeak.values) if (v.channel === null || v.channel === undefined) return v.value;
    return null;
  }

  function fmt(v) {
    const a = Math.abs(v);
    return a !== 0 && (a < 1e-3 || a >= 1e5) ? v.toExponential(2) : String(+v.toPrecision(4));
  }

  // The axis texts from the layout bounds [[xlo, xhi], [ylo, yhi]] and the space unit.
  function axisText(bounds, unit) {
    const u = unit ? " (" + unit + ")" : "";
    return { xlabel: "x" + u, ylabel: "y" + u, xmin: fmt(bounds[0][0]), xmax: fmt(bounds[0][1]),
      ymin: fmt(bounds[1][0]), ymax: fmt(bounds[1][1]) };
  }

  // The frame to show at `elapsedMs` of wall time since playing began at `startFrame`,
  // at `speed` times real time, for frames `dtMs` apart: a fractional frame, so a display
  // refresh that spans several frames skips them. Wraps to 0 at the end of a draw.
  function frameAt(startFrame, elapsedMs, speed, dtMs, total) {
    const f = startFrame + (Math.max(0, elapsedMs) * speed) / dtMs;
    return f >= total ? f % total : f;
  }

  const api = { pixelOf, sensorAt, toRGBA, ramp, colourScale, peakFor, axisText, frameAt, fmt };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.EvalGeometry = api;
})(typeof window !== "undefined" ? window : globalThis);
