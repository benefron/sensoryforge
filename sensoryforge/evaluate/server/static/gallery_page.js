"use strict";
// The anomaly gallery page (roadmap A9): the worst draws, each one click from the inspector.
(function () {
  const G = window.EvalGallery;
  const $ = (id) => document.getElementById(id);
  let signals = [];

  function el(tag, text, cls) {
    const e = document.createElement(tag);
    if (text !== undefined) e.textContent = text;
    if (cls) e.className = cls;
    return e;
  }

  function chosen() {
    return Array.from($("signals").querySelectorAll("input:checked")).map((i) => i.value);
  }

  function chips(list) {
    if (signals.length) return;
    signals = list.map((s) => s.id);
    for (const s of list) {
      const label = el("label", undefined, "chip");
      const box = el("input");
      box.type = "checkbox"; box.value = s.id; box.checked = true; box.disabled = !s.n_entries;
      label.title = s.what;
      label.append(box, document.createTextNode(" " + s.title + " (" + s.n_entries + ")"));
      $("signals").append(label);
    }
  }

  function render(payload) {
    const titles = Object.fromEntries(payload.signals.map((s) => [s.id, s.title]));
    chips(payload.signals);
    $("shown").textContent = G.shownText(payload);
    $("cards").replaceChildren();
    if (!payload.items.length) $("cards").append(el("p", "Nothing is flagged by the chosen signals.", "muted"));
    for (const item of payload.items) {
      const card = el("article", undefined, "card " + G.tier(item.score));
      const a = el("a", item.entry);
      a.href = G.entryHref(item.entry);
      a.title = "open this draw in the inspector";
      const head = el("h3");
      head.append(a, el("span", " " + item.score.toFixed(2), "muted"));
      const ul = el("ul");
      for (const r of item.reasons) ul.append(el("li", G.reasonLine(r, titles)));
      card.append(head, el("p", G.where(item), "muted"), ul);
      $("cards").append(card);
    }
    $("systemic").replaceChildren();
    for (const g of payload.systemic) $("systemic").append(el("li", G.systemicLine(g)));
    $("systemic-box").hidden = !payload.systemic.length;
    const p = payload.params;
    $("notes").textContent = payload.unavailable.map((u) => "Not available: " + u.signal + " (" + u.why + "). ").join("") +
      "Outliers: robust z at least " + p.z + " among at least " + p.min_peers + " entries of a class · saturated at " +
      p.saturation + " spikes per bin per unit · class mismatch at " + p.mismatch_ratio + " times the distance.";
  }

  async function load() {
    const q = G.query(signals.length ? chosen() : [], signals, $("limit").value);
    const r = await fetch("/api/gallery?" + q);
    const body = await r.json();
    if (!r.ok) { $("summary").textContent = body.error; return; }
    $("summary").textContent = "Draws ranked by what looks wrong, worst first. Every card opens its draw.";
    render(body);
  }

  $("signals").addEventListener("change", () => load().catch(fail));
  $("limit").addEventListener("change", () => load().catch(fail));
  $("signals").addEventListener("submit", (e) => e.preventDefault());
  function fail(e) { $("summary").textContent = String(e.message || e); }
  load().catch(fail);
})();
