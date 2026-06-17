from langchain_chroma import Chroma
from src.embeddings import get_embedding_model


def create_vectorstore(chunks):

    embedding_model = get_embedding_model()

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="chroma_db"
    )

    return vectorstore


def load_vectorstore():

    embedding_model = get_embedding_model()

    vectorstore = Chroma(
        persist_directory="chroma_db",
        embedding_function=embedding_model
    )

    return vectorstore