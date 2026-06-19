from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from src.retriever import get_retriever


def create_chain(vectorstore,chunks):

    retriever = get_retriever(vectorstore,chunks)

    from src.llm import get_llm

    llm = get_llm()

    prompt = ChatPromptTemplate.from_template(
"""
You are PlacementPrep AI, an expert resume reviewer and career coach.

Use the provided context to answer questions.

You may:

- infer strengths and weaknesses
- summarize the candidate profile
- analyze projects and skills
- compare with average students
- recommend improvements
- suggest career paths
- identify gaps
- provide constructive feedback

Base your reasoning on the context.

If information is missing, make reasonable inferences and explicitly mention that they are inferred conclusions.

Only respond with:

"I couldn't find enough information in the uploaded documents."

when the context truly lacks sufficient information.

Conversation History:
{history}

Document Context:
{context}

Question:
{question}

Answer:
"""
)


    parser = StrOutputParser()

    return retriever, prompt, llm, parser