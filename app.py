import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from ingest import ingest_pdfs, load_vectorstore, vectorstore_exists
from chatbot import build_qa_chain, ask

st.set_page_config(
    page_title="CereboAI – Document Intelligence",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="expanded" 
)

st.markdown("""
<style>
    /* Force sidebar visibility */
    [data-testid="stSidebar"] {
        background-color: #111316 !important;
        border-right: 1px solid #252930 !important;
        display: block !important;
        visibility: visible !important;
    }
    
    /* Make the expansion arrow visible even in dark mode */
    [data-testid="collapsedControl"] {
        color: #d4a853 !important;
        display: block !important;
    }

    .stApp { background-color: #0c0d0f; color: #e8e9ea; }
    
    /* Action Button (Gold) */
    div[data-testid="stSidebar"] .stButton > button {
        background: #d4a853 !important;
        color: #0c0d0f !important;
        border: none !important;
        font-weight: 600 !important;
    }
</style>
""", unsafe_allow_html=True)

api_key = os.environ.get("GROQ_API_KEY", "")

#session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

#sidebar
with st.sidebar:
    st.markdown("<h2 style='color:#f0eddf;'>CereboAI</h2>", unsafe_allow_html=True)
    st.divider()
    
    #PDF Loading 
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)
    
    if st.button("Process Documents"):
        if uploaded_files and api_key:
            with st.spinner("Processing..."):
                tmp_paths = []
                with tempfile.TemporaryDirectory() as tmpdir:
                    for uf in uploaded_files:
                        tmp_path = os.path.join(tmpdir, uf.name)
                        with open(tmp_path, "wb") as f:
                            f.write(uf.read())
                        tmp_paths.append(tmp_path)
                    
                    #chunking and vector store
                    vs, n_docs, n_chunks = ingest_pdfs(tmp_paths)
                    st.session_state.qa_chain = build_qa_chain(vs, api_key)
                    st.success(f"Indexed {n_chunks} chunks!")

#main chat logic
st.markdown("<h1 style='text-align:center;'>CereboAI</h1>", unsafe_allow_html=True)

if st.session_state.qa_chain:
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if prompt := st.chat_input("Ask a question about your documents..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        #QA Chain & Strict Retrieval
        response = ask(st.session_state.qa_chain, prompt)
        with st.chat_message("assistant"):
            st.markdown(response["answer"])
        st.session_state.chat_history.append({"role": "assistant", "content": response["answer"]})
else:
    st.info("Upload documents in the sidebar to begin.")