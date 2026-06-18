from src.chain import create_chain
from src.process_documents import process_documents
from src.memory import get_chat_history
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage


class RAGPipeline:

    def __init__(self):

        self.vectorstore = None
        self.retriever = None
        self.prompt = None
        self.llm = None
        self.parser = None
        self.chat_history = get_chat_history()


    def load_documents(self, pdf_paths):

        self.vectorstore = None
        self.retriever = None

        self.vectorstore = process_documents(pdf_paths)

        self.retriever, self.prompt, self.llm, self.parser = create_chain(
            self.vectorstore
        )


    def ask(self, question):

        if self.vectorstore is None:

            return iter([
                "Please upload and process PDF documents first."
            ]), []

        docs = self.retriever.invoke(question)

        context = "\n\n".join(
            doc.page_content
            for doc in docs
        )

        history = "\n".join(
         f"{msg.type}: {msg.content}"
         for msg in self.chat_history.messages
)

        messages = self.prompt.invoke(
            {
                "history": history,
                "context": context,
                "question": question
            }
        )

        try:

            response = self.llm.stream(messages)

        except Exception:

            return iter([
                "⚠️ LLM unavailable or rate limit exceeded. Please try again later."
            ]), []

        pages = sorted(
            set(
                doc.metadata["page"] + 1
                for doc in docs
            )
        )

        return response, pages