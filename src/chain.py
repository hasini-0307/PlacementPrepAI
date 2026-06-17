from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.retriever import get_retriever


def create_chain(vectorstore):

    retriever = get_retriever(vectorstore)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash"
    )

    prompt = ChatPromptTemplate.from_template(
      """
You are a helpful assistant.

Answer only from the provided context.

If the answer is not available in the context,
say:

"I couldn't find that information in the uploaded document."

Context:
{context}

Question:
{question}

Answer:
"""
    )

    parser = StrOutputParser()

    return retriever, prompt, llm, parser