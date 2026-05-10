from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq

STRICT_PROMPT_TEMPLATE = """You are a precise document assistant.
Answer the user's question using ONLY the information provided in the context below.
Do NOT use any prior knowledge or information outside of this context.
If the answer is not contained in the provided context, say:
"I don't have enough information in the uploaded documents to answer this question."

Context:
{context}

Question: {question}

Answer:"""

PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template=STRICT_PROMPT_TEMPLATE,
)

def build_qa_chain(vectorstore, api_key: str, top_k: int = 4):
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
        temperature=0.2,
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k},
    )

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
        }
        | PROMPT
        | llm
        | StrOutputParser()
    )

    return chain, retriever


def ask(qa_chain_tuple, question: str) -> dict:
    chain, retriever = qa_chain_tuple
    answer = chain.invoke(question)
    sources = retriever.invoke(question)
    return {
        "answer": answer,
        "sources": sources,
    }