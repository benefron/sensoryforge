"use strict";
// Pure helpers for the checks page (roadmap A8): no DOM, so the node test can run them.
(function (root) {
  const ORDER = ["fail", "error", "pass", "skipped"];

  // The draw browser's address for one failing entry.
  function entryHref(entry) {
    return "/#" + new URLSearchParams({ entry }).toString();
  }

  // "3 of 14 entries failed" / "all 12 passed" / "not run: <reason>".
  function verdict(check) {
    if (check.status === "skipped" || check.status === "error") return (check.status === "error" ? "error: " : "not run: ") + (check.note || "");
    if (check.failed) return check.failed + " of " + check.tested + " failed";
    return "all " + check.tested + " passed";
  }

  // Checks in the order they ran, grouped by pack, each group with its own counts.
  function groups(checks) {
    const out = [];
    for (const c of checks) {
      let g = out.find((x) => x.pack === c.pack);
      if (!g) { g = { pack: c.pack, checks: [], counts: { pass: 0, fail: 0, skipped: 0, error: 0 } }; out.push(g); }
      g.checks.push(c);
      g.counts[c.status] += 1;
    }
    return out;
  }

  // One line for the top of the page.
  function headline(counts) {
    const bits = ORDER.filter((s) => counts[s]).map((s) => counts[s] + " " + s);
    return bits.length ? bits.join(" · ") : "no checks";
  }

  // The failures of a check that name an entry, without repeats, in order.
  function failedEntries(check) {
    const seen = new Set(), out = [];
    for (const f of check.failures) if (f.entry && !seen.has(f.entry)) { seen.add(f.entry); out.push(f.entry); }
    return out;
  }

  // A number from `numbers` as text: integers as they are, others to 4 significant digits.
  function number(v) {
    if (v === null || v === undefined) return "none";
    if (Array.isArray(v)) return "[" + v.map(number).join(", ") + "]";
    if (typeof v !== "number") return String(v);
    return Number.isInteger(v) ? String(v) : Number(v.toPrecision(4)).toString();
  }

  const api = { entryHref, verdict, groups, headline, failedEntries, number };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.EvalChecks = api;
})(typeof window !== "undefined" ? window : globalThis);
