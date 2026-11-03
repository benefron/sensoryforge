"use strict";
// Answers the evaluation pages' requests from a snapshot (roadmap A12), so they open from
// disk with no server. `snapshot/index.js` defines window.EvalSnapshot: `entries` maps a
// request key to the recorded response, `files` maps a key to the script that holds it (a
// chosen draw's frames and traces, loaded when first asked for). The key is the path plus
// the query as a sorted JSON list, the same as export/snapshot.py's request_key.
(function () {
  const snap = window.EvalSnapshot;
  if (!snap) return;
  const loading = new Map();

  function keyOf(url) {
    const u = new URL(url, "http://snapshot.invalid/");
    const q = new Map();
    for (const [k, v] of u.searchParams) q.set(k, v);
    const pairs = Array.from(q.entries()).sort((a, b) => (a[0] < b[0] ? -1 : a[0] > b[0] ? 1 : 0));
    return u.pathname + "?" + JSON.stringify(pairs.map(([k, v]) => [k, v]));
  }

  function load(name) {
    if (!loading.has(name)) {
      loading.set(name, new Promise((resolve, reject) => {
        const s = document.createElement("script");
        s.src = name;
        s.onload = () => resolve((window.EvalSnapshotParts || {})[name] || {});
        s.onerror = () => reject(new Error("the snapshot file " + name + " is missing"));
        document.head.append(s);
      }));
    }
    return loading.get(name);
  }

  function reply(status, body, type, headers) {
    return new Response(body, { status, headers: Object.assign({ "Content-Type": type }, headers || {}) });
  }

  function failure(status, message) {
    return reply(status, JSON.stringify({ error: message }), "application/json");
  }

  function decode(entry) {
    if (!entry.x) return reply(entry.s, entry.b, entry.t, entry.h);
    const raw = atob(entry.b), bytes = new Uint8Array(raw.length);
    for (let i = 0; i < raw.length; i++) bytes[i] = raw.charCodeAt(i);
    return reply(entry.s, bytes, entry.t, entry.h);
  }

  async function answer(url, init) {
    const method = ((init && init.method) || "GET").toUpperCase();
    if (method !== "GET") return failure(405, "this is a read-only snapshot taken " + snap.made_at + "; nothing can be run here");
    const key = keyOf(typeof url === "string" ? url : url.url);
    let entry = snap.entries[key];
    if (!entry && snap.files[key]) entry = (await load(snap.files[key]))[key];
    if (entry) return decode(entry);
    return failure(404, "not in this snapshot (taken " + snap.made_at + "); it holds " +
      (snap.draws.length ? snap.draws.length + " draws in full: " + snap.draws.join(", ") : "no draws in full") +
      ", and for each only the points that were pinned when it was written. Serve the data set to see everything.");
  }

  window.fetch = answer;

  // The pages build links for the server ("/#entry=...", "/dashboard"); on disk they are files.
  const PAGES = { "/": "index.html", "/dashboard": "dashboard.html", "/gallery": "gallery.html", "/checks": "checks.html" };
  document.addEventListener("click", (event) => {
    const a = event.target.closest && event.target.closest("a[href^='/']");
    if (!a || event.defaultPrevented) return;
    const raw = a.getAttribute("href"), cut = raw.indexOf("#");
    const path = cut < 0 ? raw : raw.slice(0, cut), page = PAGES[path];
    if (!page) return;
    event.preventDefault();
    window.location.href = page + (cut < 0 ? "" : raw.slice(cut));
  }, true);

  document.addEventListener("DOMContentLoaded", () => {
    const note = document.createElement("p");
    note.className = "muted";
    note.textContent = "Snapshot of " + (snap.name || "a data set") + " taken " + snap.made_at +
      ". Read-only; it holds " + snap.draws.length + " draws in full.";
    const header = document.querySelector("header");
    if (header) header.append(note);
  });
})();
