"""
app.py
------
Streamlit UI for the RAG PDF Chatbot.

Layout
  Sidebar → upload PDFs + Process button + Groq API key input
  Main    → chat interface with session history
"""

import os
import tempfile
import streamlit as st

from ingest import ingest_pdfs, load_vectorstore
from chatbot import build_qa_chain, ask


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PDF RAG Chatbot",
    page_icon="📄",
    layout="wide",
)

st.title("📄 PDF RAG Chatbot")
st.caption("Ask questions — answers come **only** from the documents you upload.")


# ── Session state init ────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Configuration")

    api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get your free key at https://console.groq.com",
    )
    if not api_key:
        api_key = os.environ.get("GROQ_API_KEY", "")

    st.divider()

    st.header("📁 Upload PDFs")
    uploaded_files = st.file_uploader(
        "Choose one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
    )

    process_btn = st.button("⚡ Process Documents", use_container_width=True)

    if process_btn:
        if not uploaded_files:
            st.warning("Please upload at least one PDF before processing.")
        elif not api_key:
            st.error("Please enter your Groq API key.")
        else:
            with st.spinner("Loading, chunking, and embedding your PDFs …"):
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

                    st.success(
                        f"✅ Processed **{n_docs}** page(s) → **{n_chunks}** chunks across "
                        f"**{len(uploaded_files)}** file(s). Ready to chat!"
                    )
                except Exception as e:
                    st.error(f"Error during processing: {e}")

    if st.session_state.qa_chain:
        st.divider()
        st.success("✅ Documents loaded. Ask away!")

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()


# ── Main chat area ────────────────────────────────────────────────────────────
if not st.session_state.qa_chain:
    st.info("👈 Upload your PDFs in the sidebar and click **Process Documents** to begin.")
else:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask a question about your documents …")

    if user_input:
        with st.chat_message("user"):
            st.markdown(user_input)
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.chat_message("assistant"):
            with st.spinner("Thinking …"):
                try:
                    response = ask(st.session_state.qa_chain, user_input)
                    answer   = response["answer"]
                    sources  = response["sources"]

                    st.markdown(answer)

                    if sources:
                        with st.expander("📌 Source chunks used"):
                            for i, doc in enumerate(sources, 1):
                                page = doc.metadata.get("page", "?")
                                src  = doc.metadata.get("source", "uploaded PDF")
                                st.markdown(
                                    f"**[{i}] {os.path.basename(src)} — page {page + 1}**\n\n"
                                    f"{doc.page_content[:300]}…"
                                )

                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": answer}
                    )

                except Exception as e:
                    err_msg = f"⚠️ Error generating answer: {e}"
                    st.error(err_msg)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": err_msg}
                    )