from langchain_chroma import Chroma

from src.embeddings import get_embedding_model


def create_vectorstore(chunks):

    embeddings = get_embedding_model()

    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)

    return vectorstore
