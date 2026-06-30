def has_sufficient_context(docs):

    if len(docs) == 0:
        return False

    non_empty_docs = [doc for doc in docs if len(doc.page_content.strip()) > 50]

    return len(non_empty_docs) >= 2
