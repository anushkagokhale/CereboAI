"""
ingest.py
---------
Handles PDF loading, text chunking, embedding generation, and FAISS vector store creation.
"""

import os
import pickle
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ── Chunking strategy ─────────────────────────────────────────────────────────
# chunk_size = 1000 characters  → large enough to preserve full sentences and
#   surrounding context (ideal for factual Q&A over technical / research docs).
# chunk_overlap = 200 characters → 20 % overlap prevents answers from being cut
#   at chunk boundaries; keeps cross-chunk context intact without doubling data.
# Interview note: smaller chunks (≤500) suit keyword search but lose context;
#   larger (≥2000) reduce retrieval precision. 1000/200 is the sweet spot for
#   most general-purpose RAG pipelines.
CHUNK_SIZE    = 1000
CHUNK_OVERLAP = 200

EMBED_MODEL   = "all-MiniLM-L6-v2"   # fast, lightweight, 384-dim sentence embeddings
VECTORSTORE_PATH = "vectorstore/faiss_index"


def load_pdfs(pdf_paths: list[str]):
    """Load one or more PDFs and return a flat list of LangChain Documents."""
    all_docs = []
    for path in pdf_paths:
        loader = PyPDFLoader(path)
        all_docs.extend(loader.load())
    return all_docs


def chunk_documents(docs):
    """Split documents into overlapping chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],   # tries to split on paragraph > line > word
    )
    return splitter.split_documents(docs)


def build_vectorstore(chunks):
    """
    Create a FAISS vector store from document chunks.
    Returns the FAISS retriever object.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def save_vectorstore(vectorstore, path: str = VECTORSTORE_PATH):
    """Persist the FAISS index to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    vectorstore.save_local(path)


def load_vectorstore(path: str = VECTORSTORE_PATH):
    """Load a previously saved FAISS index from disk."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)


def ingest_pdfs(pdf_paths: list[str]):
    """
    Full pipeline: load → chunk → embed → store.
    Returns the FAISS vectorstore object (also saves to disk).
    """
    docs   = load_pdfs(pdf_paths)
    chunks = chunk_documents(docs)
    vs     = build_vectorstore(chunks)
    save_vectorstore(vs)
    return vs, len(docs), len(chunks)