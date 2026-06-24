from sentence_transformers import CrossEncoder

class Reranker:

    def __init__(self):

        self.model = CrossEncoder(
            "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )


    def rerank(
        self,
        query,
        documents,
        top_k=5
    ):

        pairs = [
            (query, doc.page_content)
            for doc in documents
        ]

        scores = self.model.predict(pairs)

        scored_docs = list(
            zip(documents, scores)
        )

        scored_docs.sort(
            key=lambda x: x[1],
            reverse=True
        )

        top_docs = [
    doc
    for doc, score in scored_docs[:top_k]
]

        top_scores = [
    float(score)
    for doc, score in scored_docs[:top_k]
]

        return top_docs, top_scores