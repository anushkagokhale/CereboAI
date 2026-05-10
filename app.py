"""
app.py - PDF RAG Chatbot (Professional UI)
"""

import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from ingest import ingest_pdfs, load_vectorstore, vectorstore_exists
from chatbot import build_qa_chain, ask

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CereboAI – PDF Chatbot",
    page_icon="🧠",
    layout="wide",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }

    /* Title area */
    .hero {
        text-align: center;
        padding: 2.5rem 1rem 1rem 1rem;
    }
    .hero h1 {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(90deg, #58a6ff, #a371f7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.3rem;
    }
    .hero p {
        color: #8b949e;
        font-size: 1rem;
    }

    /* Chat bubbles */
    [data-testid="stChatMessage"] {
        border-radius: 12px;
        margin-bottom: 0.5rem;
    }

    /* Input box */
    [data-testid="stChatInput"] textarea {
        background-color: #161b22 !important;
        border: 1px solid #30363d !important;
        border-radius: 10px !important;
        color: #e6edf3 !important;
    }

    /* Buttons */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
        border: 1px solid #30363d;
        background-color: #21262d;
        color: #e6edf3;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        background-color: #388bfd20;
        border-color: #58a6ff;
        color: #58a6ff;
    }

    /* Hide Streamlit branding */
    #MainMenu, footer, header { visibility: hidden; }

    /* Expander */
    [data-testid="stExpander"] {
        border: 1px solid #30363d !important;
        border-radius: 8px !important;
        background-color: #161b22 !important;
    }
</style>
""", unsafe_allow_html=True)

# ── API key (silent, from .env) ───────────────────────────────────────────────
api_key = os.environ.get("GROQ_API_KEY", "")

# ── Session state ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# ── Auto-load existing vector store ──────────────────────────────────────────
if st.session_state.vectorstore is None and vectorstore_exists() and api_key:
    with st.spinner("Loading previous document index..."):
        st.session_state.vectorstore = load_vectorstore()
        st.session_state.qa_chain = build_qa_chain(
            st.session_state.vectorstore, api_key
        )

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 CereboAI")
    st.markdown("<p style='color:#8b949e; font-size:0.85rem;'>Powered by Groq · LangChain · FAISS</p>", unsafe_allow_html=True)
    st.divider()

    st.markdown("### 📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Drag and drop PDF files here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    process_btn = st.button("⚡  Process Documents", use_container_width=True)

    if process_btn:
        if not uploaded_files:
            st.warning("Please upload at least one PDF.")
        elif not api_key:
            st.error("GROQ_API_KEY missing from .env file.")
        else:
            with st.spinner("Processing your documents..."):
                try:
                    tmp_paths = []
                    with tempfile.TemporaryDirectory() as tmpdir:
                        for uf in uploaded_files:
                            tmp_path = os.path.join(tmpdir, uf.name)
                            with open(tmp_path, "wb") as f:
                                f.write(uf.read())
                            tmp_paths.append(tmp_path)
                        vs, n_docs, n_chunks = ingest_pdfs(tmp_paths)

                    st.session_state.vectorstore = load_vectorstore()
                    st.session_state.qa_chain = build_qa_chain(
                        st.session_state.vectorstore, api_key
                    )
                    st.session_state.chat_history = []
                    st.success(f"✅ {n_docs} pages · {n_chunks} chunks indexed")
                except Exception as e:
                    st.error(f"Error: {e}")

    if st.session_state.qa_chain:
        st.divider()
        st.markdown("<p style='color:#3fb950; font-size:0.9rem;'>● Documents ready</p>", unsafe_allow_html=True)
        uploaded_names = [uf.name for uf in uploaded_files] if uploaded_files else []
        if uploaded_names:
            for name in uploaded_names:
                st.markdown(f"<p style='color:#8b949e; font-size:0.8rem;'>📄 {name}</p>", unsafe_allow_html=True)

    st.divider()
    if st.button("🗑️  Clear Chat", use_container_width=True):
        st.session_state.chat_history = []
        if st.session_state.vectorstore and api_key:
            st.session_state.qa_chain = build_qa_chain(
                st.session_state.vectorstore, api_key
            )
        st.rerun()

# ── Hero header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🧠 CereboAI</h1>
    <p>Ask questions — answers come <strong>only</strong> from your uploaded documents.</p>
</div>
""", unsafe_allow_html=True)

# ── Main chat area ────────────────────────────────────────────────────────────
if not st.session_state.qa_chain:
    st.markdown("""
    <div style='text-align:center; margin-top:4rem; color:#8b949e;'>
        <p style='font-size:3rem;'>📄</p>
        <p style='font-size:1.1rem;'>Upload a PDF in the sidebar to get started.</p>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg and msg["sources"]:
                with st.expander("📎 View sources"):
                    for i, doc in enumerate(msg["sources"], 1):
                        page = doc.metadata.get("page", "?")
                        src  = doc.metadata.get("source", "document")
                        st.markdown(f"**[{i}] {os.path.basename(src)} — page {int(page)+1}**")
                        st.caption(doc.page_content[:300] + "…")

    user_input = st.chat_input("Ask anything about your documents…")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                try:
                    response = ask(st.session_state.qa_chain, user_input)
                    answer   = response["answer"]
                    sources  = response["sources"]

                    st.markdown(answer)

                    if sources:
                        with st.expander("📎 View sources"):
                            for i, doc in enumerate(sources, 1):
                                page = doc.metadata.get("page", "?")
                                src  = doc.metadata.get("source", "document")
                                st.markdown(f"**[{i}] {os.path.basename(src)} — page {int(page)+1}**")
                                st.caption(doc.page_content[:300] + "…")

                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })
                except Exception as e:
                    err = f"⚠️ {e}"
                    st.error(err)
                    st.session_state.chat_history.append({"role": "assistant", "content": err})