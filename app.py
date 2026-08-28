"""Streamlit front end for PitchLens.

This is a thin client. It owns no models, no index and no pipeline code — every
question and every upload is an HTTP call to the FastAPI service, which keeps the
expensive objects resident. Point it somewhere else with API_URL.
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000").rstrip("/")

# Generation can involve several LLM round trips; ingestion is a vision pass over
# every page of a deck. Both need far longer than the requests default.
QUERY_TIMEOUT = 300
INGEST_TIMEOUT = 1800

SUGGESTIONS = (
    "What problem does this company solve?",
    "What is the total addressable market?",
    "Who are the founders?",
    "What are the revenue streams?",
    "How much funding is being raised?",
)

KIND_ICONS = {"transcript": "\U0001f4c4", "summary": "✨"}
FALLBACK_ICON = "\U0001f4c4"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp { background: #0d1117; }
.main .block-container { background: #0d1117; padding-top: 2rem; }
h1, h2, h3 { color: #e2e8f0 !important; }
p, li, .stMarkdown { color: #94a3b8; }

[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f0f1a 0%, #1a1a2e 50%, #16213e 100%);
    border-right: 1px solid rgba(99, 102, 241, 0.2);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

.hero-title {
    font-size: 2.4rem;
    font-weight: 700;
    background: linear-gradient(135deg, #818cf8, #c084fc, #fb7185);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 0.25rem;
}
.hero-title.compact { font-size: 1.6rem; }
.hero-sub { color: #64748b; font-size: 1rem; margin-bottom: 2rem; }
.hero-sub.compact { font-size: 0.85rem; margin-bottom: 1rem; }

.user-msg {
    background: linear-gradient(135deg, #312e81 0%, #4338ca 100%);
    border-radius: 16px 16px 4px 16px;
    padding: 1rem 1.25rem;
    margin: 0.5rem 0;
    color: #e0e7ff;
    box-shadow: 0 4px 16px rgba(99, 102, 241, 0.25);
}
.ai-msg {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 16px 16px 16px 4px;
    padding: 1rem 1.25rem;
    margin: 0.5rem 0 1.25rem 0;
    color: #e2e8f0;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.3);
    white-space: pre-wrap;
}
.msg-footer {
    margin-top: 0.75rem;
    border-top: 1px solid rgba(99, 102, 241, 0.15);
    padding-top: 0.6rem;
}
.citation-tag {
    display: inline-block;
    background: rgba(99, 102, 241, 0.15);
    border: 1px solid rgba(99, 102, 241, 0.4);
    border-radius: 20px;
    padding: 2px 10px;
    font-size: 0.78rem;
    color: #a5b4fc;
    margin: 2px;
}
.trace {
    font-size: 0.72rem;
    color: #64748b;
    letter-spacing: 0.03em;
    margin-top: 0.5rem;
}
.warn-badge {
    background: linear-gradient(135deg, #7c2d12 0%, #9a3412 100%);
    border: 1px solid #f97316;
    border-radius: 8px;
    padding: 0.5rem 1rem;
    color: #fed7aa;
    font-size: 0.85rem;
    margin-bottom: 0.5rem;
}
.offline-badge {
    background: linear-gradient(135deg, #450a0a 0%, #7f1d1d 100%);
    border: 1px solid #ef4444;
    border-radius: 10px;
    padding: 1rem 1.25rem;
    color: #fecaca;
    margin-bottom: 1rem;
}

.metric-card {
    background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    text-align: center;
}
.metric-val { font-size: 1.6rem; font-weight: 700; color: #818cf8; line-height: 2.1rem; }
.metric-val.small { font-size: 1rem; }
.metric-label {
    font-size: 0.8rem;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.empty-state { text-align: center; padding: 4rem 2rem; color: #64748b; }
.empty-state .glyph { font-size: 4rem; margin-bottom: 1rem; }
.empty-state h3 { color: #475569 !important; }

[data-testid="stFileUploader"] {
    background: rgba(99, 102, 241, 0.05);
    border: 2px dashed rgba(99, 102, 241, 0.3);
    border-radius: 12px;
    padding: 1rem;
}

.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    transition: all 0.2s ease !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.4) !important;
}

[data-baseweb="tab-list"] { background: #1e293b !important; border-radius: 10px; gap: 4px; }
[data-baseweb="tab"] { color: #64748b !important; border-radius: 8px !important; }
[aria-selected="true"] {
    background: linear-gradient(135deg, #4f46e5, #7c3aed) !important;
    color: white !important;
}
</style>
"""


class PitchLensClient:
    """Typed wrapper over the FastAPI service.

    Every network concern — base URL, timeouts, error shape — is settled here so
    the view code below only ever deals with plain dicts.
    """

    def __init__(self, base_url: str):
        self.base_url = base_url
        self._session = requests.Session()

    def _call(self, method: str, path: str, *, timeout: int, **kwargs) -> dict:
        response = self._session.request(method, f"{self.base_url}{path}", timeout=timeout, **kwargs)
        if response.status_code >= 400:
            raise RuntimeError(response.json().get("detail", response.text))
        return response.json()

    def health(self) -> dict:
        return self._call("GET", "/health", timeout=10)

    def decks(self) -> list[dict]:
        return self._call("GET", "/decks", timeout=10)

    def query(self, question: str, mode: str, agentic: bool) -> dict:
        return self._call(
            "POST",
            "/query",
            timeout=QUERY_TIMEOUT,
            json={"question": question, "mode": mode, "agentic": agentic},
        )

    def ingest(self, filename: str, payload: bytes) -> dict:
        return self._call(
            "POST",
            "/ingest",
            timeout=INGEST_TIMEOUT,
            files={"file": (filename, payload, "application/pdf")},
        )


def html(markup: str) -> None:
    st.markdown(markup, unsafe_allow_html=True)


def metric(label: str, value: str, *, small: bool = False) -> str:
    cls = "metric-val small" if small else "metric-val"
    return f'<div class="metric-card"><div class="{cls}">{value}</div><div class="metric-label">{label}</div></div>'


def render_citations(citations: list[dict]) -> str:
    tags = []
    for citation in citations:
        icon = KIND_ICONS.get(citation["kind"], FALLBACK_ICON)
        tags.append(
            f'<span class="citation-tag">{icon} {citation["deck"]}'
            f' — slide {citation["slide"]}</span>'
        )
    return " ".join(tags)


def render_turn(turn: dict) -> None:
    result = turn["result"]
    html(f'<div class="user-msg">\U0001f9d1 {turn["question"]}</div>')

    if not result["grounded"]:
        html(
            '<div class="warn-badge">⚠️ <strong>Low confidence</strong> — the retrieved slides '
            "did not clearly support this answer. Verify it against the deck.</div>"
        )

    agent_label = "agentic" if turn["agentic"] else "single-pass"
    trace = (
        f"{turn['mode']} · {agent_label} · {result['retrieval_rounds']} retrieval round(s)"
        f" · {result['latency_ms']} ms"
    )
    if result.get("rewritten_query"):
        trace += f" · rewritten: “{result['rewritten_query']}”"

    html(
        f'<div class="ai-msg">\U0001f916 {result["answer"]}'
        f'<div class="msg-footer">\U0001f4cc {render_citations(result["citations"])}'
        f'<div class="trace">{trace}</div></div></div>'
    )


def ask(client: PitchLensClient, question: str, mode: str, agentic: bool) -> None:
    with st.spinner("Retrieving and reasoning…"):
        try:
            result = client.query(question, mode, agentic)
        except (requests.RequestException, RuntimeError) as exc:
            st.error(f"Query failed: {exc}")
            return
    st.session_state.chat.append(
        {"question": question, "mode": mode, "agentic": agentic, "result": result}
    )
    st.rerun()


st.set_page_config(
    page_title="PitchLens",
    page_icon="\U0001f4ca",
    layout="wide",
    initial_sidebar_state="expanded",
)
html(CSS)

st.session_state.setdefault("chat", [])
client = PitchLensClient(API_URL)

try:
    health = client.health()
except requests.RequestException as exc:
    html(
        f'<div class="offline-badge"><strong>Cannot reach the PitchLens API</strong> at '
        f"<code>{API_URL}</code>.<br>Start it with "
        f"<code>uvicorn api.main:app --reload</code>, or set <code>API_URL</code>.<br>"
        f"<small>{exc}</small></div>"
    )
    st.stop()


with st.sidebar:
    html('<div class="hero-title compact">\U0001f4ca PitchLens</div>')
    html('<div class="hero-sub compact">FastAPI · FAISS · BM25 · Gemini</div>')
    st.divider()

    st.markdown("**⚙️ Retrieval**")
    mode = st.selectbox(
        "Strategy",
        health["modes"],
        index=health["modes"].index(health["default_mode"]),
        help="The ablation axis: dense only, sparse only, RRF-fused hybrid, or hybrid with cross-encoder re-ranking.",
    )
    agentic = st.toggle(
        "Corrective agent",
        value=True,
        help="Grades retrieved context, rewrites the query and retrieves again when the evidence is weak.",
    )

    st.divider()
    st.markdown("**\U0001f5c2️ Corpus**")
    st.markdown(f"`{health['decks']}` decks · `{health['chunks']}` chunks")
    for deck in client.decks():
        st.caption(f"\U0001f4c1 {deck['name']} — {deck['slides']} slides, {deck['chunks']} chunks")

    st.divider()
    if st.session_state.chat and st.button("\U0001f5d1️ Clear chat", use_container_width=True):
        st.session_state.chat = []
        st.rerun()


html('<div class="hero-title">\U0001f4ca PitchLens</div>')
html('<div class="hero-sub">Upload a pitch deck, then ask anything about it — with slide-level citations.</div>')

tab_chat, tab_upload = st.tabs(["\U0001f4ac Chat", "\U0001f4e4 Upload"])


with tab_chat:
    if not health["ready"]:
        html(
            '<div class="empty-state"><div class="glyph">\U0001f4e4</div>'
            "<h3>No decks indexed yet</h3>"
            "<p>Upload a pitch deck in the <strong>Upload</strong> tab to get started.</p></div>"
        )
    else:
        c1, c2, c3 = st.columns(3)
        c1.markdown(metric("Questions asked", str(len(st.session_state.chat))), unsafe_allow_html=True)
        c2.markdown(metric("Retrieval mode", mode, small=True), unsafe_allow_html=True)
        c3.markdown(
            metric("Agent", "corrective" if agentic else "off", small=True), unsafe_allow_html=True
        )
        html("<br>")

        for turn in st.session_state.chat:
            render_turn(turn)

        if not st.session_state.chat:
            st.markdown("**\U0001f4a1 Try one of these:**")
            for column, suggestion in zip(st.columns(len(SUGGESTIONS)), SUGGESTIONS):
                if column.button(suggestion, use_container_width=True):
                    ask(client, suggestion, mode, agentic)

        question = st.chat_input("Ask anything about the indexed decks…")
        if question:
            ask(client, question, mode, agentic)


with tab_upload:
    st.markdown("### Upload a pitch deck")
    st.markdown(
        "The API extracts each slide's text layer, routes image-only slides through a vision "
        "model, writes structured summaries, and rebuilds the index."
    )

    uploaded = st.file_uploader("Choose a PDF", type=["pdf"], label_visibility="collapsed")
    if st.button("\U0001f680 Ingest", disabled=uploaded is None):
        with st.spinner("Ingesting — this runs one vision pass per image-only slide…"):
            try:
                result = client.ingest(uploaded.name, uploaded.getvalue())
            except (requests.RequestException, RuntimeError) as exc:
                st.error(f"Ingestion failed: {exc}")
            else:
                deck = result["deck"]
                st.success(
                    f"**{deck['name']}** indexed — {deck['slides']} slides "
                    f"({deck['vision_slides']} via vision), {deck['chunks']} chunks "
                    f"in {result['seconds']}s. Switch to the Chat tab."
                )
                st.rerun()

    decks = client.decks()
    if decks:
        st.divider()
        st.markdown("#### Indexed decks")
        for deck in decks:
            with st.expander(f"\U0001f4ca {deck['name']}"):
                d1, d2, d3 = st.columns(3)
                d1.metric("Slides", deck["slides"])
                d2.metric("Vision slides", deck["vision_slides"])
                d3.metric("Chunks", deck["chunks"])
