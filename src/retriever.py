from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_classic.retrievers import EnsembleRetriever

from src.llm import get_llm
from src.bm25_retriever import get_bm25_retriever


def get_retriever(vectorstore, chunks):

    # Vector retriever
    vector_retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 20
        }
    )

    # BM25 retriever
    bm25_retriever = get_bm25_retriever(chunks)

    # Hybrid retriever
    hybrid_retriever = EnsembleRetriever(
        retrievers=[
            vector_retriever,
            bm25_retriever
        ],
        weights=[
            0.7,
            0.3
        ]
    )

    # MultiQuery on top of hybrid retrieval
    llm = get_llm()

    retriever = MultiQueryRetriever.from_llm(
        retriever=hybrid_retriever,
        llm=llm
    )

    return retriever