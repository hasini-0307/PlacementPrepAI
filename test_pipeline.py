from dotenv import load_dotenv

from src.loader import load_pdf
from src.splitter import split_documents
from src.vectorstore import create_vectorstore
from src.chain import create_chain

load_dotenv()

pdf_path = "uploads/Hasini_Kolasani_Resume.pdf"

# Load PDF
documents = load_pdf(pdf_path)

# Split into chunks
chunks = split_documents(documents)

# Create vector database
vectorstore = create_vectorstore(chunks)

# Create components
retriever, prompt, llm, parser = create_chain(vectorstore)

while True:

    query = input("\nAsk a question (type 'exit' to quit): ")

    if query.lower() == "exit":
        break

    # Retrieve documents
    docs = retriever.invoke(query)

    # Combine retrieved chunks
    context = "\n\n".join(doc.page_content for doc in docs)

    # Build prompt
    messages = prompt.invoke(
        {
            "context": context,
            "question": query
        }
    )

    # Gemini
    response = llm.invoke(messages)

    # Parse output
    answer = parser.invoke(response)

    print("\nAnswer:")
    print(answer)