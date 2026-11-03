"use strict";
// The declared-fields panel and two-draw comparison (roadmap A6c): pure helpers with no DOM,
// so the node test can run them. A draw's record is the manifest line as stored (its
// declaration); `rows()` flattens it to dotted paths in display order, `compare()` lines two
// draws' rows up path by path, and `grouped()` splits rows into the panel's sections.
(function (root) {
  const ORDER = ["Entry", "Draw", "Values drawn", "Seeds"];

  function number(v) {
    if (Number.isInteger(v)) return Number.isSafeInteger(v) ? String(v) : String(v) + " (rounded)";
    return String(+v.toPrecision(6));
  }

  // A [name, start, end] list is a timeline: "hold 0–30 · slide 30–90".
  function isTimeline(v) {
    return Array.isArray(v) && v.length > 0 && v.every((s) =>
      Array.isArray(s) && s.length === 3 && typeof s[0] === "string" && typeof s[1] === "number" && typeof s[2] === "number");
  }

  function leaf(v) {
    if (v === null || v === undefined) return "—";
    if (typeof v === "number") return number(v);
    if (typeof v === "string") return v === "" ? "(empty)" : v;
    if (typeof v === "boolean") return v ? "yes" : "no";
    if (Array.isArray(v) && v.length === 0) return "(none)";
    if (isTimeline(v)) return v.map((s) => s[0] + " " + number(s[1]) + "–" + number(s[2])).join(" · ");
    return JSON.stringify(v);
  }

  function walk(value, prefix, out) {
    if (value && typeof value === "object" && !Array.isArray(value)) {
      const keys = Object.keys(value);
      if (keys.length === 0 && prefix) out.push([prefix, "(none)"]);
      for (const k of keys) walk(value[k], prefix ? prefix + "." + k : k, out);
    } else out.push([prefix, leaf(value)]);
  }

  function groupOf(path) {
    if (path.startsWith("draw.values.")) return "Values drawn";
    if (path.startsWith("draw.")) return "Draw";
    if (path.startsWith("seeds.")) return "Seeds";
    if (path.indexOf(".") < 0) return "Entry";
    const head = path.split(".")[0];
    return head[0].toUpperCase() + head.slice(1);
  }

  // Rows {path, group, text} of a record; `seeds` (path-free name -> decimal string, from
  // /api/draw) replaces the record's seeds, which JSON parsing may have rounded.
  function rows(record, seeds) {
    const flat = [];
    walk(record || {}, "", flat);
    const out = flat.map(([path, text]) => ({ path, group: groupOf(path), text }));
    if (seeds) for (const r of out) {
      const m = /^seeds\.(.+)$/.exec(r.path);
      if (m && seeds[m[1]] !== undefined) r.text = String(seeds[m[1]]);
    }
    return out;
  }

  // The rows of both draws by path: A's order, then paths only B has.
  function compare(a, b) {
    const text = (list) => new Map(list.map((r) => [r.path, r.text]));
    const ma = text(a), mb = text(b), seen = new Set(), out = [];
    for (const r of a.concat(b)) {
      if (seen.has(r.path)) continue;
      seen.add(r.path);
      const x = ma.has(r.path) ? ma.get(r.path) : null, y = mb.has(r.path) ? mb.get(r.path) : null;
      out.push({ path: r.path, group: r.group, a: x, b: y, same: x === y });
    }
    return out;
  }

  // Rows split into sections: the fixed ones first, others in the order met.
  function grouped(list) {
    const by = new Map();
    for (const r of list) {
      if (!by.has(r.group)) by.set(r.group, []);
      by.get(r.group).push(r);
    }
    const names = ORDER.filter((n) => by.has(n)).concat([...by.keys()].filter((n) => ORDER.indexOf(n) < 0));
    return names.map((group) => ({ group, rows: by.get(group) }));
  }

  // Why traces of draw B cannot be laid over a pin made in draw A, or null when they can.
  // `pin` is {kind: "sensor", k} or {kind: "unit", index}; the bundles are /api/draw's
  // `bundle` bodies.
  function pinProblem(pin, view, a, b) {
    if (!pin) return null;
    if (!b) return "its bundle could not be read";
    if (pin.kind === "sensor") {
      const na = a.layout.coords.length, nb = b.layout.coords.length;
      return na === nb ? null : "the sensor arrays differ (" + na + " and " + nb + " sensors)";
    }
    const pb = (b.populations || []).find((p) => p.name === view);
    if (!pb) return "it has no population " + view;
    return pin.index < pb.n_units ? null : "population " + view + " has only " + pb.n_units + " neurons";
  }

  const api = { rows, compare, grouped, pinProblem };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.EvalRecord = api;
})(typeof window !== "undefined" ? window : globalThis);
