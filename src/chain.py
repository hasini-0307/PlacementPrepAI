from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.retriever import get_retriever


def create_chain(vectorstore):

    retriever = get_retriever(vectorstore)

    from src.llm import get_llm

    llm = get_llm()

    prompt = ChatPromptTemplate.from_template(
"""
You are a helpful AI assistant.

Use the provided context to answer questions.

You may infer, summarize, compare, analyze, and generate insights based on the information in the context.

If the answer is explicitly present, provide it accurately.

If the answer requires reasoning, derive it from the context and clearly indicate that it is an inference.

Only say "I couldn't find that information in the uploaded documents" when the context contains insufficient information to reasonably answer the question.

Conversation History:
{history}

Context:
{context}

Question:
{question}

Answer:
"""
)


    parser = StrOutputParser()

    return retriever, prompt, llm, parser