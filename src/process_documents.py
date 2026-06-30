from src.loader import load_pdf
from src.splitter import split_documents
from src.vectorstore import create_vectorstore
from src.logger import logger


def process_documents(pdf_paths):
    try:

        all_docs = []

        for pdf_path in pdf_paths:

            docs = load_pdf(pdf_path)

            all_docs.extend(docs)
            logger.info("Processing uploaded documents")

        chunks = split_documents(all_docs)
        logger.info("Created %d chunks", len(chunks))

        logger.info("Generating embeddings...")

        vectorstore = create_vectorstore(chunks)
        logger.info("Embeddings generated successfully")
        logger.info("Vector database created successfully")

        logger.info("Document processing completed successfully")
        logger.info("=" * 60)
        return vectorstore, chunks

    except Exception:

        logger.exception("Document processing failed")

        raise
