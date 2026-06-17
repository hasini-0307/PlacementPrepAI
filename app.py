import streamlit as st
from dotenv import load_dotenv
from src.rag_pipeline import RAGPipeline

# Load environment variables
load_dotenv()

# Initialize pipeline
pipeline = RAGPipeline()

# Page configuration
st.set_page_config(
    page_title="PlacementPrep AI",
    page_icon="🤖",
    layout="wide"
)

# Sidebar
with st.sidebar:

    st.title("🤖 PlacementPrep AI")

    st.markdown("---")

    st.markdown("""
### About

Ask questions about your documents using:

- 📄 Semantic Search
- 🧠 Gemini 2.5 Flash
- 🔍 ChromaDB
- 🤗 HuggingFace Embeddings
""")

    st.markdown("---")

    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# Main title
st.title("🤖 PlacementPrep AI")

st.caption(
    "Chat with your documents using RAG + Gemini"
)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input(
    "Ask anything about your documents..."
)

if user_input:

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_input
        }
    )

    # Display user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate response
    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            answer, pages = pipeline.ask(user_input)

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