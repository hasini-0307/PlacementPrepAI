from src.loader import load_pdf
from src.splitter import split_documents
from src.vectorstore import create_vectorstore


def process_documents(pdf_paths):

    all_docs = []

    for pdf_path in pdf_paths:

        docs = load_pdf(pdf_path)

        all_docs.extend(docs)

    chunks = split_documents(all_docs)

    vectorstore = create_vectorstore(chunks)

    return vectorstore, chunks