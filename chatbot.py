"""
chatbot.py
----------
Builds a conversational RAG chain with manual memory (no langchain.memory import).
"""

import logging
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_groq import ChatGroq

logger = logging.getLogger(__name__)


# ── Manual conversation memory (no external dependency) ───────────────────────
class ConversationBufferMemory:
    def __init__(self):
        self.history = []

    def load_history(self) -> str:
        if not self.history:
            return ""
        lines = []
        for turn in self.history:
            lines.append(f"Human: {turn['input']}")
            lines.append(f"Assistant: {turn['answer']}")
        return "\n".join(lines)

    def save(self, question: str, answer: str):
        self.history.append({"input": question, "answer": answer})

    def clear(self):
        self.history = []


# ── Prompt ────────────────────────────────────────────────────────────────────
STRICT_PROMPT_TEMPLATE = """You are a precise document assistant.
Answer the user's question using ONLY the information provided in the context below.
Do NOT use any prior knowledge or information outside of this context.
If the answer is not contained in the provided context, say:
"I don't have enough information in the uploaded documents to answer this question."

Conversation so far:
{chat_history}

Context from documents:
{context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(
    input_variables=["chat_history", "context", "question"],
    template=STRICT_PROMPT_TEMPLATE,
)


def build_qa_chain(vectorstore, api_key: str, top_k: int = 4):
    """Returns (chain, retriever, memory)."""
    logger.info("Building QA chain...")

    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0.2,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    memory = ConversationBufferMemory()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context":      RunnableLambda(lambda x: format_docs(retriever.invoke(x["question"]))),
            "question":     RunnableLambda(lambda x: x["question"]),
            "chat_history": RunnableLambda(lambda x: memory.load_history()),
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )

    logger.info("QA chain ready.")
    return chain, retriever, memory


def ask(chain_tuple, question: str) -> dict:
    chain, retriever, memory = chain_tuple
    source_docs = retriever.invoke(question)
    answer = chain.invoke({"question": question})
    memory.save(question, answer)
    return {
        "answer":  answer,
        "sources": source_docs,
    }