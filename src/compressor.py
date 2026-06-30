from langchain_classic.retrievers.document_compressors import EmbeddingsFilter

from src.embeddings import get_embedding_model


def get_compressor():

    embeddings = get_embedding_model()

    compressor = EmbeddingsFilter(embeddings=embeddings, similarity_threshold=0.5)

    return compressor
