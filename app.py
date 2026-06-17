import streamlit as st
from dotenv import load_dotenv

from src.vectorstore import load_vectorstore
from src.chain import create_chain

# Load environment variables
load_dotenv()

# Load backend components
vectorstore = load_vectorstore()
retriever, prompt, llm, parser = create_chain(vectorstore)

# Page configuration
st.set_page_config(
    page_title="PlacementPrep AI",
    page_icon="🤖",
    layout="wide"
)

# Title
st.title("🤖 PlacementPrep AI")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input(
    "Ask anything about your documents..."
)

if user_input:

    # Save and display user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    with st.chat_message("user"):
        st.markdown(user_input)

    # Retrieve relevant chunks
    docs = retriever.invoke(user_input)

    # Build context
    context = "\n\n".join(
        doc.page_content for doc in docs
    )

    # Build prompt
    messages = prompt.invoke(
        {
            "context": context,
            "question": user_input
        }
    )

    # Generate response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            response = llm.invoke(messages)

            answer = parser.invoke(response)

            # Source pages
            pages = sorted(
                set(doc.metadata["page"] + 1 for doc in docs)
            )

            source_text = "\n\n📚 Sources:\n"

            for page in pages:
                source_text += f"- Page {page}\n"

            full_response = answer + source_text

            st.markdown(full_response)

    # Save assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_response
        }
    )