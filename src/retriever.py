from langchain_classic.retrievers.multi_query import MultiQueryRetriever

from src.llm import get_llm


def get_retriever(vectorstore):

    # Base retriever
    base_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 3
        }
    )

    # LLM for generating multiple queries
    llm = get_llm()

    # MultiQuery Retriever
    retriever = MultiQueryRetriever.from_llm(
        retriever=base_retriever,
        llm=llm
    )

    return retriever