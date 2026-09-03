/* GraphRAG UI logic.
 *
 * Everything the page shows comes from the same FastAPI process that serves it,
 * so there is no CORS setup and no second server to run.
 *
 * The interface deliberately surfaces the routing decision, the generated
 * Cypher and the subgraph rather than only the answer -- the project's claim is
 * that routing to a graph beats plain retrieval, and hiding the routing would
 * hide the evidence for it.
 */

const $ = (id) => document.getElementById(id);

const EXAMPLES = [
  ["Explanation", "How does dense passage retrieval encode questions and passages?"],
  ["Relational", "Which datasets does DPR evaluate on?"],
  ["Counting", "How many distinct methods are evaluated on Natural Questions?"],
  ["Authors", "Who are the authors of the dense passage retrieval paper?"],
  ["Corpus-wide", "What are the main research themes across these papers?"],
];

/* ---------- small helpers ---------- */

const esc = (s) =>
  String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c])
  );

async function getJSON(url, opts) {
  const res = await fetch(url, opts);
  if (!res.ok) {
    // FastAPI puts the useful part in `detail`; surface it rather than a bare status.
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (body.detail) detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail);
    } catch { /* non-JSON error body */ }
    throw new Error(detail);
  }
  return res.json();
}

/* ---------- sidebar ---------- */

async function loadStats() {
  try {
    const s = await getJSON("/stats");
    const g = s.graph || {};
    const nodes = Object.entries(g).filter(([k]) => !k.startsWith("-[")).reduce((a, [, v]) => a + v, 0);
    const rels = Object.entries(g).filter(([k]) => k.startsWith("-[")).reduce((a, [, v]) => a + v, 0);
    $("s-papers").textContent = s.index.papers ?? 0;
    $("s-chunks").textContent = s.index.chunks ?? 0;
    $("s-ents").textContent = nodes.toLocaleString();
    $("s-rels").textContent = rels.toLocaleString();
  } catch (e) {
    console.error("stats failed", e);
  }
  try {
    const h = await getJSON("/health");
    $("m-backend").textContent = h.backend;
  } catch { /* leave the placeholder */ }
}

async function loadCommunities() {
  const box = $("communities");
  try {
    const data = await getJSON("/communities");
    $("m-comms").textContent = data.count;
    if (!data.count) {
      box.innerHTML = '<span class="empty">No communities yet. Run <code>graphrag communities --build</code>.</span>';
      return;
    }
    box.innerHTML = data.communities
      .map(
        (c) =>
          `<div class="comm"><div class="comm-title">${esc(c.title || c.id)}</div>` +
          `<div class="comm-body">${esc(c.summary || "")}</div></div>`
      )
      .join("");
  } catch (e) {
    box.innerHTML = `<span class="empty">Could not load themes: ${esc(e.message)}</span>`;
  }
}

/* ---------- rendering ---------- */

function renderRows(rows) {
  const card = $("rows-card");
  if (!rows || !rows.length) { card.style.display = "none"; return; }

  const cols = [...new Set(rows.flatMap((r) => Object.keys(r)))];
  const head = `<tr>${cols.map((c) => `<th>${esc(c)}</th>`).join("")}</tr>`;
  const body = rows
    .slice(0, 60)
    .map((r) => `<tr>${cols.map((c) => `<td>${esc(r[c] ?? "")}</td>`).join("")}</tr>`)
    .join("");

  $("rows").innerHTML = head + body;
  card.querySelector("h3").textContent = `Graph rows (${rows.length})`;
  card.style.display = "";
}

function renderSources(cits) {
  const card = $("src-card");
  if (!cits || !cits.length) { card.style.display = "none"; return; }

  $("sources").innerHTML = cits
    .map((c) => {
      const where =
        c.source === "graph" || c.source === "global"
          ? "knowledge graph"
          : `${esc(c.paper_id)} · chars ${c.char_start}–${c.char_end}`;
      const sec = c.section ? `<div class="src-sec">${esc(c.section)}</div>` : "";
      return `<div class="src"><div class="src-head">[${c.index}] ${where}</div>${sec}` +
             `<div class="src-body">${esc(c.preview)}</div></div>`;
    })
    .join("");
  card.querySelector("h3").textContent = `Sources (${cits.length})`;
  card.style.display = "";
}

let network = null;

async function renderGraph(result) {
  const card = $("graph-card");

  // Seed from the entities the graph query actually returned; fall back to the
  // question's longer words so vector-mode answers still show a neighbourhood.
  let seeds = (result.graph_rows || [])
    .flatMap((r) => Object.values(r))
    .filter((v) => typeof v === "string" && v.length > 2);

  if (!seeds.length) {
    seeds = result.question.split(/\s+/).filter((w) => w.length > 5).slice(0, 2);
  }
  seeds = [...new Set(seeds)].slice(0, 3);
  if (!seeds.length) { card.style.display = "none"; return; }

  const edges = [];
  const nodeIds = new Set();
  for (const seed of seeds) {
    try {
      const sg = await getJSON(`/graph/subgraph?entity=${encodeURIComponent(seed)}&limit=40`);
      for (const e of sg.edges) {
        edges.push({ from: e.source, to: e.target });
        nodeIds.add(e.source);
        nodeIds.add(e.target);
      }
    } catch { /* a seed with no neighbourhood is not an error */ }
  }

  if (!nodeIds.size) { card.style.display = "none"; return; }
  card.style.display = "";

  const nodes = [...nodeIds].slice(0, 90).map((id) => ({
    id,
    label: id.length > 26 ? id.slice(0, 26) + "…" : id,
    title: id,
  }));
  const keep = new Set(nodes.map((n) => n.id));

  if (network) network.destroy();
  network = new vis.Network(
    $("graph-panel"),
    { nodes, edges: edges.filter((e) => keep.has(e.from) && keep.has(e.to)) },
    {
      nodes: {
        shape: "dot", size: 11,
        color: { background: "#1f6feb", border: "#58a6ff", highlight: { background: "#58a6ff", border: "#a5d6ff" } },
        font: { color: "#8b949e", size: 12, face: "system-ui" },
      },
      edges: { color: { color: "#30363d", highlight: "#58a6ff" }, width: 1, smooth: { type: "continuous" } },
      physics: { barnesHut: { gravitationalConstant: -9000, springLength: 150 }, stabilization: { iterations: 180 } },
      interaction: { hover: true, tooltipDelay: 150 },
    }
  );
}

/* ---------- ask ---------- */

async function ask() {
  const question = $("q").value.trim();
  if (!question) return;

  const btn = $("go");
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span>Thinking…';
  $("error").innerHTML = "";
  $("results").classList.remove("visible");

  const mode = $("mode").value;
  const payload = { question, k: Number($("k").value) };
  if (mode) payload.mode = mode;

  try {
    const r = await getJSON("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    $("badge").textContent = r.mode.toUpperCase();
    $("badge").dataset.mode = r.mode;
    $("reason").textContent = r.route_reason || "";
    $("tokens").textContent = `${r.tokens.toLocaleString()} tokens`;
    $("calls").textContent = `${r.calls} call${r.calls === 1 ? "" : "s"}`;

    $("fallback-note").innerHTML = r.fell_back
      ? '<div class="warn">The graph returned nothing for this question, so it fell back to passage retrieval.</div>'
      : "";

    $("answer").textContent = r.answer;

    if (r.dropped_citations && r.dropped_citations.length) {
      $("fallback-note").innerHTML +=
        `<div class="warn">Dropped invented citations: ${r.dropped_citations.join(", ")}. ` +
        `The model cited sources that were never supplied.</div>`;
    }

    if (r.cypher) {
      $("cypher").textContent = r.cypher;
      $("cypher-card").style.display = "";
    } else {
      $("cypher-card").style.display = "none";
    }

    renderRows(r.graph_rows);
    renderSources(r.citations);
    $("results").classList.add("visible");
    renderGraph(r);
  } catch (e) {
    $("error").innerHTML = `<div class="err"><strong>Request failed.</strong> ${esc(e.message)}</div>`;
  } finally {
    btn.disabled = false;
    btn.textContent = "Ask";
  }
}

/* ---------- wiring ---------- */

$("examples").innerHTML = EXAMPLES.map(
  ([tag, q], i) => `<button class="chip" data-i="${i}"><span class="chip-tag">${esc(tag)}</span> ${esc(q)}</button>`
).join("");

$("examples").addEventListener("click", (e) => {
  const chip = e.target.closest(".chip");
  if (!chip) return;
  $("q").value = EXAMPLES[chip.dataset.i][1];
  ask();
});

$("go").addEventListener("click", ask);
$("q").addEventListener("keydown", (e) => { if (e.key === "Enter") ask(); });
$("k").addEventListener("input", (e) => { $("k-val").textContent = e.target.value; });

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
    tab.classList.add("active");
    $(`panel-${tab.dataset.panel}`).classList.add("active");
  });
});

loadStats();
loadCommunities();
