"use strict";
// The dashboard page (roadmap A7): counts, histograms and the coverage map. Every bar,
// count and cell is a link to the draw browser showing the draws behind it.
(function () {
  const D = window.EvalDashboard, G = window.EvalGeometry;
  const $ = (id) => document.getElementById(id);

  function el(tag, text, cls) {
    const e = document.createElement(tag);
    if (text !== undefined) e.textContent = text;
    if (cls) e.className = cls;
    return e;
  }

  function link(query, label, children, cls) {
    const a = el("a", undefined, cls);
    a.href = D.listHref(query, label);
    a.title = "list these draws";
    a.append(...children);
    return a;
  }

  // A row of bars, one per label: the label, a bar scaled to the largest, the count.
  function bars(items, labelOf, heading) {
    const box = el("div", undefined, "bars");
    const frac = D.fractions(items.map((i) => i.n));
    items.forEach((item, k) => {
      const fill = el("span", undefined, "fill");
      fill.style.width = (100 * frac[k]).toFixed(1) + "%";
      const track = el("span", undefined, "track");
      track.append(fill);
      box.append(link(item.query, heading + ": " + labelOf(item, k) + " (" + item.n + ")",
        [el("span", labelOf(item, k), "lab"), track, el("span", String(item.n), "n")], "bar"));
    });
    return box;
  }

  function counts(dash) {
    const root = $("counts");
    const titles = { split: "Split", class: "Class", kind: "Kind", session_type: "Session type", status: "Status" };
    for (const key of Object.keys(titles)) {
      const items = dash.counts[key];
      if (!items || !items.length) continue;
      const block = el("div", undefined, "block");
      block.append(el("h3", titles[key]), bars(items, (i) => i.label, titles[key]));
      root.append(block);
    }
  }

  function matrix(dash) {
    const m = dash.matrix, table = el("table", undefined, "matrix");
    const head = el("tr");
    head.append(el("th", ""), ...m.splits.map((s) => el("th", s)));
    table.append(head);
    const top = Math.max(0, ...m.cells.flat().map((c) => c.n));
    m.classes.forEach((cls, r) => {
      const tr = el("tr");
      tr.append(el("th", cls));
      m.cells[r].forEach((cell, c) => {
        const td = el("td");
        td.style.setProperty("--shade", top ? (cell.n / top).toFixed(2) : "0");
        td.append(cell.n ? link(cell.query, cls + " in " + m.splits[c] + " (" + cell.n + ")", [document.createTextNode(String(cell.n))], "cell") : document.createTextNode("0"));
        tr.append(td);
      });
      table.append(tr);
    });
    $("matrix").append(table);
  }

  function histograms(dash) {
    const root = $("histograms");
    for (const h of dash.histograms) {
      const block = el("div", undefined, "block");
      const unit = h.unit ? " (" + h.unit + ")" : "";
      block.append(el("h3", h.title + unit));
      if (!h.counts.length) {
        block.append(el("p", "no value to show (" + h.n_missing + " draws have none)", "muted"));
      } else {
        const items = h.counts.map((n, b) => ({ n, query: h.queries[b] }));
        block.append(bars(items, (_, b) => D.rangeLabel(h.edges, b, G.fmt), h.title));
        block.append(el("p", "from the " + h.source + (h.n_missing ? " · " + h.n_missing + " draws without a value" : ""), "muted"));
      }
      root.append(block);
    }
  }

  function coverage(dash) {
    const root = $("coverage");
    if (!dash.coverage.length) root.append(el("p", "no declared axis to cover", "muted"));
    let last = null;
    for (const cov of dash.coverage) {
      if (cov.class !== last) { root.append(el("h3", cov.class)); last = cov.class; }
      const row = el("div", undefined, "covrow");
      const lab = el("span", cov.axis + (cov.circular ? " (circular)" : cov.scale === "log" ? " (log)" : ""), "axis");
      const strip = el("span", undefined, "strip");
      const top = Math.max(0, ...cov.counts);
      cov.counts.forEach((n, b) => {
        const name = D.coverageLabel(cov, b, G.fmt);
        const cell = n ? link(cov.queries[b], cov.class + " " + cov.axis + " " + name + " (" + n + ")", [el("span", String(n))], "cvc") : el("span", "0", "cvc empty");
        cell.title = name + ": " + n;
        cell.style.setProperty("--shade", top ? (n / top).toFixed(2) : "0");
        strip.append(cell);
      });
      row.append(lab, strip);
      const range = cov.lo !== undefined ? G.fmt(cov.lo) + " to " + G.fmt(cov.hi) : (cov.labels || []).join(", ");
      row.append(el("span", range + " · " + D.coverageText(cov), "muted covtext"));
      root.append(row);
    }
    if (dash.coverage_skipped.length)
      root.append(el("p", "not binned: " + dash.coverage_skipped.map((s) => s.class + "." + s.axis).join(", ") + " (" + dash.coverage_skipped[0].why + ")", "muted"));
  }

  async function main() {
    const [dash, summary] = await Promise.all([
      fetch("/api/dashboard").then(async (r) => { const b = await r.json(); if (!r.ok) throw new Error(b.error || r.statusText); return b; }),
      fetch("/api/dataset").then((r) => r.json()),
    ]);
    const bits = [summary.name || summary.folder, dash.n_draws + " draws"];
    if (dash.n_sessions) bits.push(dash.n_sessions + " sessions");
    $("summary").textContent = bits.join(" · ");
    counts(dash); matrix(dash); histograms(dash); coverage(dash);
    $("notes").textContent = dash.unavailable.map((u) => "Not shown: " + u.what + " (" + u.why + ").").join(" ");
  }

  main().catch((err) => { $("summary").textContent = String(err.message || err); });
})();
