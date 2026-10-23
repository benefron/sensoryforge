"use strict";
// Pure helpers for the anomaly gallery (roadmap A9): no DOM, so the node test can run them.
(function (root) {
  // The /api/gallery query for the chosen signals and limit.
  function query(chosen, all, limit) {
    const q = new URLSearchParams({ limit: String(limit) });
    if (chosen.length && chosen.length < all.length) q.set("signal", chosen.join(","));
    return q.toString();
  }

  // The draw browser's address for an entry.
  function entryHref(entry) {
    return "/#" + new URLSearchParams({ entry }).toString();
  }

  // "high", "medium" or "low", for colouring a card by its score.
  function tier(score) {
    return score >= 0.95 ? "high" : score >= 0.6 ? "medium" : "low";
  }

  // The class (or split) line under an entry id.
  function where(item) {
    return (item.class || "no class") + " · " + item.split;
  }

  // One line per reason: its title, the population, the detail.
  function reasonLine(reason, titles) {
    return (titles[reason.signal] || reason.signal) + (reason.population ? " (" + reason.population + ")" : "") + ": " + reason.detail;
  }

  // "3 of 12 entries flagged" for the header.
  function shownText(payload) {
    return payload.total + " flagged of " + payload.n_entries + " entries" +
      (payload.items.length < payload.total ? " · showing the worst " + payload.items.length : "");
  }

  // One line for a population silent for most of a class.
  function systemicLine(g) {
    return g.population + " is silent for " + g.n_silent + " of " + g.n_active + " active entries of " + g.group.replace(/^split:/, "split ");
  }

  const api = { systemicLine, query, entryHref, tier, where, reasonLine, shownText };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  else root.EvalGallery = api;
})(typeof window !== "undefined" ? window : globalThis);
