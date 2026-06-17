from src.vectorstore import load_vectorstore
from src.chain import create_chain


class RAGPipeline:

    def __init__(self):

        self.vectorstore = load_vectorstore()

        self.retriever, self.prompt, self.llm, self.parser = create_chain(
            self.vectorstore
        )

    def ask(self, question):

        docs = self.retriever.invoke(question)

        context = "\n\n".join(
            doc.page_content for doc in docs
        )

        messages = self.prompt.invoke(
            {
                "context": context,
                "question": question
            }
        )

        response = self.llm.invoke(messages)

        answer = self.parser.invoke(response)

        pages = sorted(
            set(doc.metadata["page"] + 1 for doc in docs)
        )

        return answer, pages