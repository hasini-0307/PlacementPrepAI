from src.chain import create_chain
from src.process_documents import process_documents


class RAGPipeline:

    def __init__(self):

        self.vectorstore = None
        self.retriever = None
        self.prompt = None
        self.llm = None
        self.parser = None


    def load_documents(self, pdf_paths):

        # Release previous objects
        self.vectorstore = None
        self.retriever = None

        # Build new vectorstore
        self.vectorstore = process_documents(pdf_paths)

        self.retriever, self.prompt, self.llm, self.parser = create_chain(
            self.vectorstore
        )


    def ask(self, question):

        if self.vectorstore is None:

            return (
                "Please upload and process PDF documents first.",
                []
            )

        docs = self.retriever.invoke(question)

        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        messages = self.prompt.invoke(
            {
                "context": context,
                "question": question
            }
        )

        try:

            response = self.llm.invoke(messages)

            answer = self.parser.invoke(response)

        except Exception:

            answer = (
                "⚠️ LLM unavailable or rate limit exceeded.\n\n"
                "Please try again later."
            )

            return answer, []

        pages = sorted(
            set(
                doc.metadata["page"] + 1
                for doc in docs
            )
        )

        return answer, pages