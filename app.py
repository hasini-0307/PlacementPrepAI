import streamlit as st
from dotenv import load_dotenv

from src.vectorstore import load_vectorstore
from src.chain import create_chain

# Load environment variables
load_dotenv()

# Load backend components
vectorstore = load_vectorstore()
retriever, prompt, llm, parser = create_chain(vectorstore)

# Streamlit page settings
st.set_page_config(
    page_title="PlacementPrep AI",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 PlacementPrep AI")

# User input
user_input = st.text_input(
    "Ask a question"
)

# Button click
if st.button("Send"):

    # Retrieve relevant documents
    docs = retriever.invoke(user_input)

    # Build context
    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    # Create prompt
    messages = prompt.invoke(
        {
            "context": context,
            "question": user_input
        }
    )

    # Generate response
    with st.spinner("Thinking..."):
        response = llm.invoke(messages)

    answer = parser.invoke(response)

    # Display answer
    st.subheader("Answer")
    st.write(answer)

    # Display source pages
    pages = sorted(
        set(doc.metadata["page"] + 1 for doc in docs)
    )

    st.subheader("Sources")

    for page in pages:
        st.write(f"📄 Page {page}")