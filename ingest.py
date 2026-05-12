#Handles PDF loading, chunking, embedding, and FAISS.
"""
Chunking strategy:
chunk_size=1000, overlap=200 — preserves sentence context while keeping
retrieval precision high. 20% overlap prevents answers from being cut at
chunk boundaries.
"""

import os
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CHUNK_SIZE       = 1000
CHUNK_OVERLAP    = 200
EMBED_MODEL      = "all-MiniLM-L6-v2"
VECTORSTORE_PATH = "faiss_index"


def get_embeddings():
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL)


def load_pdfs(pdf_paths: list[str]):
    all_docs = []
    for path in pdf_paths:
        logger.info(f"Loading PDF: {path}")
        loader = PyPDFLoader(path)
        all_docs.extend(loader.load())
    logger.info(f"Loaded {len(all_docs)} pages total.")
    return all_docs


def chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(docs)
    logger.info(f"Split into {len(chunks)} chunks.")
    return chunks


def build_vectorstore(chunks):
    logger.info("Building FAISS vector store...")
    embeddings = get_embeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore


def save_vectorstore(vectorstore, path: str = VECTORSTORE_PATH):
    os.makedirs(path, exist_ok=True)
    vectorstore.save_local(path)
    logger.info(f"Vector store saved to '{path}'.")


def load_vectorstore(path: str = VECTORSTORE_PATH):
    logger.info(f"Loading existing vector store from '{path}'...")
    embeddings = get_embeddings()
    vs = FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
    logger.info("Vector store loaded successfully.")
    return vs


def vectorstore_exists(path: str = VECTORSTORE_PATH) -> bool:
    return os.path.exists(path) and os.path.exists(os.path.join(path, "index.faiss"))


def ingest_pdfs(pdf_paths: list[str]):
    """Full pipeline: load → chunk → embed → save. Returns vectorstore."""
    docs   = load_pdfs(pdf_paths)
    chunks = chunk_documents(docs)
    vs     = build_vectorstore(chunks)
    save_vectorstore(vs)
    return vs, len(docs), len(chunks)