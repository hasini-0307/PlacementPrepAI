from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.retriever import get_retriever


def create_chain(vectorstore):

    retriever = get_retriever(vectorstore)

    llm = ChatGroq(
        model="llama-3.3-70b-versatile"
    )

    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful assistant.

Answer ONLY from the provided context.

If the answer is not available in the context, say:

"I couldn't find that information in the uploaded documents."

Context:
{context}

Question:
{question}

Answer:
"""
    )

    parser = StrOutputParser()

    return retriever, prompt, llm, parser