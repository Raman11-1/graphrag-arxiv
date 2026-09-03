"""Streamlit UI.

Run with:  streamlit run src/graphrag/ui/app.py

The interface deliberately exposes *how* an answer was produced, not just the
answer: which route was chosen and why, the generated Cypher, the graph rows,
and the subgraph behind the result. For a project whose whole claim is that
routing to a graph beats plain retrieval, hiding the routing would hide the
point.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `streamlit run src/graphrag/ui/app.py` without an editable install.
_SRC = Path(__file__).resolve().parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import streamlit as st  # noqa: E402

from graphrag.config import settings  # noqa: E402

st.set_page_config(page_title="GraphRAG", page_icon="🔎", layout="wide")

MODE_HELP = {
    "vector": "Semantic passage retrieval — good for explanations.",
    "graph": "Cypher over the knowledge graph — exact, for relations and counts.",
    "global": "Community summaries — for corpus-wide questions.",
    "hybrid": "Graph facts plus passages, fused.",
}


@st.cache_resource(show_spinner="Loading models and indexes…")
def _backend():
    from graphrag.llm.factory import get_backend

    return get_backend()


@st.cache_resource(show_spinner=False)
def _graph_store():
    from graphrag.graph.store import GraphStore

    return GraphStore(read_only=True)


@st.cache_data(ttl=30, show_spinner=False)
def _stats() -> dict:
    from graphrag.graph.store import GraphStore
    from graphrag.ingest.pipeline import stats as index_stats

    out = {"index": index_stats(), "graph": {}}
    try:
        out["graph"] = GraphStore(read_only=True).counts()
    except Exception:  # noqa: BLE001 - graph may not exist yet
        pass
    return out


def render_subgraph(entities: list[str], depth: int = 1, limit: int = 40) -> None:
    """Draw the neighbourhood of the answer's entities with pyvis."""
    from pyvis.network import Network

    store = _graph_store()
    edges: list[tuple[str, str, str]] = []

    for entity in entities[:4]:
        for rel in ("EVALUATES_ON", "USES", "PROPOSES", "COMPARES_TO", "BUILDS_ON"):
            try:
                rows = store.query(
                    f"MATCH (a)-[:{rel}]->(b) "
                    f"WHERE toLower(a.name) CONTAINS toLower($e) "
                    f"   OR toLower(b.name) CONTAINS toLower($e) "
                    f"RETURN a.name AS a, b.name AS b LIMIT $lim",
                    {"e": entity, "lim": limit},
                )
            except Exception:  # noqa: BLE001 - empty rel table
                continue
            edges.extend((r["a"], r["b"], rel) for r in rows if r.get("a") and r.get("b"))

    if not edges:
        st.info("No graph neighbourhood found for this answer.")
        return

    net = Network(height="480px", width="100%", bgcolor="#ffffff", font_color="#222")
    net.barnes_hut(spring_length=180)

    seen: set[str] = set()
    for a, b, _rel in edges[:120]:
        for node in (a, b):
            if node not in seen:
                net.add_node(node, label=node[:34], title=node, size=16)
                seen.add(node)
    for a, b, rel in edges[:120]:
        net.add_edge(a, b, title=rel, label=rel.lower().replace("_", " "))

    st.components.v1.html(net.generate_html(notebook=False), height=500, scrolling=False)


# --- sidebar ----------------------------------------------------------

with st.sidebar:
    st.header("Corpus")
    stats = _stats()
    index = stats["index"]
    c1, c2 = st.columns(2)
    c1.metric("Papers", index.get("papers", 0))
    c2.metric("Chunks", index.get("chunks", 0))

    graph_counts = stats.get("graph") or {}
    if graph_counts:
        nodes = sum(v for k, v in graph_counts.items() if not k.startswith("-["))
        rels = sum(v for k, v in graph_counts.items() if k.startswith("-["))
        c3, c4 = st.columns(2)
        c3.metric("Entities", nodes)
        c4.metric("Relations", rels)
    else:
        st.warning("No graph yet. Run `graphrag extract` then `graphrag graph --rebuild`.")

    st.divider()
    st.caption(f"Backend: `{settings.llm_backend}`")
    st.caption(f"Answers: `{settings.answer_model}`")

    forced = st.selectbox(
        "Retrieval mode",
        ["auto (route)", "vector", "graph", "global", "hybrid"],
        help="Force a mode to compare routes on the same question.",
    )
    top_k = st.slider("Passages retrieved", 3, 20, 8)

# --- main -------------------------------------------------------------

st.title("GraphRAG")
st.caption("Hybrid knowledge-graph + vector retrieval over research papers.")

examples = {
    "Explanation (vector)": "How does dense passage retrieval encode questions and passages?",
    "Relational (graph)": "Which datasets does DPR evaluate on?",
    "Counting (graph)": "How many distinct methods are evaluated on Natural Questions?",
    "Corpus-wide (global)": "What are the main research themes across these papers?",
}
chosen_example = st.radio(
    "Try an example", list(examples), horizontal=True, label_visibility="collapsed"
)

question = st.text_input("Question", value=examples[chosen_example])

if st.button("Ask", type="primary") and question.strip():
    from graphrag.llm.meter import scoped as scoped_meter
    from graphrag.retrieve import ask as run_ask

    try:
        with scoped_meter() as request_meter, st.spinner("Routing, retrieving, answering…"):
            result = run_ask(
                question,
                backend=_backend(),
                k=top_k,
                force_mode=None if forced.startswith("auto") else forced,
            )
    except Exception as exc:  # noqa: BLE001 - surface failures in the UI
        st.error(f"{type(exc).__name__}: {exc}")
        st.stop()

    left, right = st.columns([3, 2])

    with left:
        badge = {"vector": "blue", "graph": "green", "global": "violet", "hybrid": "orange"}
        st.markdown(
            f":{badge.get(result.mode, 'grey')}[**{result.mode.upper()}**] "
            f"— {result.route_reason}"
        )
        st.caption(MODE_HELP.get(result.mode, ""))
        if result.fell_back:
            st.warning("The graph returned nothing; fell back to passage retrieval.")

        st.markdown("### Answer")
        st.markdown(result.answer.text)

        if result.answer.dropped_citations:
            st.warning(
                f"Dropped invented citations: {result.answer.dropped_citations}. "
                "The model cited sources that were not provided."
            )

    with right:
        st.markdown("### Provenance")
        st.metric("Tokens", f"{request_meter.total_tokens:,}")

        if result.cypher:
            with st.expander("Generated Cypher", expanded=True):
                st.code(result.cypher, language="cypher")

        if result.graph_rows:
            with st.expander(f"Graph rows ({len(result.graph_rows)})", expanded=True):
                st.dataframe(result.graph_rows, use_container_width=True)

        if result.answer.citations:
            with st.expander(f"Sources ({len(result.answer.citations)})"):
                for i, hit in enumerate(result.answer.citations, start=1):
                    where = (
                        "knowledge graph"
                        if hit.source in ("graph", "global")
                        else f"{hit.paper_id} · chars {hit.char_start}–{hit.char_end}"
                    )
                    st.markdown(f"**[{i}]** {where}")
                    if hit.section:
                        st.caption(hit.section)
                    st.caption(" ".join(hit.text.split())[:320] + "…")

    if graph_counts:
        st.markdown("### Knowledge graph neighbourhood")
        seeds = [r.get("dataset") or r.get("method") or r.get("author") for r in result.graph_rows]
        seeds = [s for s in seeds if isinstance(s, str)] or [
            w for w in question.split() if len(w) > 4
        ]
        render_subgraph(seeds)
