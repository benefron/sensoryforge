"use strict";
// The checks page (roadmap A8): each reference-free check with its verdict, its raw
// numbers and the draws that failed, each a link to the draw browser.
(function () {
  const C = window.EvalChecks;
  const $ = (id) => document.getElementById(id);

  function el(tag, text, cls) {
    const e = document.createElement(tag);
    if (text !== undefined) e.textContent = text;
    if (cls) e.className = cls;
    return e;
  }

  function failures(check) {
    const box = el("ul", undefined, "failures");
    for (const f of check.failures) {
      const li = el("li");
      if (f.entry) {
        const a = el("a", f.where.split(" / ")[0]);
        a.href = C.entryHref(f.entry);
        a.title = "open this draw";
        li.append(a, document.createTextNode((f.where.includes(" / ") ? " / " + f.where.split(" / ").slice(1).join(" / ") : "") + ": " + f.detail));
      } else {
        li.append(el("strong", f.where), document.createTextNode(": " + f.detail));
      }
      box.append(li);
    }
    if (check.failed > check.failures.length)
      box.append(el("li", "… " + (check.failed - check.failures.length) + " more not listed", "muted"));
    return box;
  }

  function numbers(check) {
    const keys = Object.keys(check.numbers || {});
    if (!keys.length) return null;
    const dl = el("dl", undefined, "numbers");
    for (const k of keys) dl.append(el("dt", k), el("dd", C.number(check.numbers[k])));
    return dl;
  }

  function card(check) {
    const box = el("details", undefined, "check " + check.status);
    box.open = check.status === "fail" || check.status === "error";
    const head = el("summary");
    head.append(el("span", check.status, "badge"), el("strong", check.title), el("span", " " + C.verdict(check), "muted"));
    box.append(head, el("p", check.what, "what"), el("p", check.id, "muted id"));
    if (check.note && check.status !== "skipped" && check.status !== "error") box.append(el("p", check.note, "muted"));
    if (check.failures.length) box.append(failures(check));
    const n = numbers(check);
    if (n) box.append(n);
    return box;
  }

  function render(doc) {
    $("summary").textContent = C.headline(doc.counts) + " · " + doc.created_at + " · packs " + doc.inputs.packs.join(", ");
    $("packs").value = doc.inputs.packs.join(",");
    $("list").replaceChildren();
    for (const g of C.groups(doc.checks)) {
      $("list").append(el("h2", g.pack + " checks · " + C.headline(g.counts)));
      for (const c of g.checks) $("list").append(card(c));
    }
    const p = doc.inputs.params;
    $("notes").textContent = "sample " + p.sample + " entries · seed " + p.seed +
      (p.bounds_mv ? " · voltage limits " + p.bounds_mv.join(" to ") + " mV" : " · no voltage limits") +
      (p.limit ? " · first " + p.limit + " entries" : "");
  }

  async function poll(id) {
    for (;;) {
      const job = await (await fetch("/api/jobs/" + id)).json();
      if (job.state !== "running") return job;
      $("job").textContent = "running…";
      await new Promise((r) => setTimeout(r, 500));
    }
  }

  async function run() {
    $("run").disabled = true;
    const packs = $("packs").value.trim() || "core";
    const r = await fetch("/api/stages/checks/run?packs=" + encodeURIComponent(packs), { method: "POST" });
    const job = await r.json();
    if (!r.ok) { $("job").textContent = job.error; $("run").disabled = false; return; }
    const done = await poll(job.id);
    $("run").disabled = false;
    if (done.state === "failed") { $("job").textContent = "failed: " + done.error; return; }
    $("job").textContent = "";
    await load();
  }

  async function load() {
    const r = await fetch("/api/stage/checks");
    if (!r.ok) {
      $("summary").textContent = "checks.json has not been computed for this data set.";
      $("offer").hidden = false;
      return;
    }
    $("offer").hidden = false;
    render(await r.json());
  }

  $("run").addEventListener("click", () => run().catch((e) => { $("job").textContent = String(e.message || e); $("run").disabled = false; }));
  load().catch((e) => { $("summary").textContent = String(e.message || e); });
})();
