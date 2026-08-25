# rag_pipeline/rag_pipeline.py
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

def build_rag_pipeline(vectorstore, api_key: str):
    """Hybrid RAG: Hugging Face embeddings + Gemini reasoning LLM."""
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # ✅ Fix: unwrap dict before sending to retriever
    def get_context(input_dict):
        question_text = input_dict["question"]
        docs = retriever.invoke(question_text)
        return "\n\n".join([d.page_content for d in docs])

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",
        google_api_key=api_key,
        temperature=0.3,
    )

    prompt = ChatPromptTemplate.from_template("""
        You are an assistant that answers questions based on a startup pitch deck.
        Use the provided context carefully to craft an insightful and accurate answer.

        Context:
        {context}

        Question:
        {question}

        Helpful, concise answer:
    """)

    rag_chain = (
        {
            "context": get_context,   # ✅ uses text only
            "question": RunnablePassthrough(),
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain
