"use strict";
// Pure helpers for the data-set dashboard (roadmap A7): no DOM, so the node test can run
// them. A bar or cell is a link to the draw browser filtered to the draws behind it.
(function (root) {
  // "[0.5, 0.6)" for a bin of a histogram; the last bin is closed at the top.
  function rangeLabel(edges, i, fmt) {
    const last = i === edges.length - 2;
    if (edges[0] === edges[edges.length - 1]) return fmt(edges[0]);
    return "[" + fmt(edges[i]) + ", " + fmt(edges[i + 1]) + (last ? "]" : ")");
  }

  // Each count as a fraction of the largest, 0 for an empty histogram.
  function fractions(counts) {
    const top = Math.max(0, ...counts);
    return counts.map((n) => (top ? n / top : 0));
  }

  // The draw browser's address for a dashboard filter: `query` is an /api/draws query
  // string, `label` says what is being shown.
  function listHref(query, label) {
    return "/#" + new URLSearchParams({ list: query, label }).toString();
  }

  // The label of a coverage bin: a category as it is, a numeric bin as a range.
  function coverageLabel(cov, i, fmt) {
    return cov.labels ? cov.labels[i] : rangeLabel(cov.edges, i, fmt);
  }

  // One line on how much of an axis the draws reached.
  function coverageText(cov) {
    const n = cov.counts.length, reached = n - cov.n_empty;
    let text = reached + " of " + n + (cov.labels ? " values" : " bins") + " reached by " + cov.n + " draws";
    if (cov.n_outside) text += " · " + cov.n_outside + " outside the declared domain";
    return text;
  }

  // The filter the browser should apply for a hash like `#list=<query>&label=<text>`;
  // null when the hash carries none.
  function listFromHash(hash) {
    const q = new URLSearchParams(hash.replace(/^#/, ""));
    const list = q.get("list");
    if (!list) return null;
    return { query: list, label: q.get("label") || list };
  }

  const api = { rangeLabel, fractions, listHref, coverageLabel, coverageText, listFromHash };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.EvalDashboard = api;
})(typeof window !== "undefined" ? window : globalThis);
