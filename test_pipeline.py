from dotenv import load_dotenv

from src.vectorstore import load_vectorstore
from src.chain import create_chain

from src.vectorstore import load_vectorstore

load_dotenv()

pdf_path = "uploads/Hasini_Kolasani_Resume.pdf"


vectorstore = load_vectorstore()

# Create components
retriever, prompt, llm, parser = create_chain(vectorstore)

while True:

    query = input("\nAsk a question (type 'exit' to quit): ")

    if query.lower() == "exit":
        break

    # Retrieve documents
    docs = retriever.invoke(query)

    # Create context
    context = "\n\n".join(doc.page_content for doc in docs)

    # Build prompt
    messages = prompt.invoke(
        {
            "context": context,
            "question": query
        }
    )

    # Gemini response
    response = llm.invoke(messages)

    # Extract answer
    answer = parser.invoke(response)

    print("\nAnswer:")
    print(answer)

    # Show source pages
    pages = sorted(
        set(doc.metadata["page"] + 1 for doc in docs)
    )

    print("\nSources:")
    for page in pages:
        print(f"Page {page}")