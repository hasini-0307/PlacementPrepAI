import os
import streamlit as st
from dotenv import load_dotenv

from src.rag_pipeline import RAGPipeline

# Load environment variables
load_dotenv()

# Initialize pipeline once
if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline()

pipeline = st.session_state.pipeline

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Page config
st.set_page_config(
    page_title="PlacementPrep AI",
    page_icon="🤖",
    layout="wide"
)

# Sidebar
with st.sidebar:

    st.title("🤖 PlacementPrep AI")
    with st.expander("⚙️ Tech Stack"):

     st.markdown("""
    **LLM**
    - Groq
    - Llama 3.3 70B

    **Embeddings**
    - sentence-transformers/all-MiniLM-L6-v2

    **Vector Database**
    - ChromaDB (In-Memory)

    **Framework**
    - LangChain

    **Frontend**
    - Streamlit

    **Retrieval**
    - MMR Search
    """)

    st.markdown("---")

    st.markdown("""
### About

Ask questions about your uploaded documents using:

- 📄 Semantic Search
- 🧠 Llama 3.3 70B (Groq)
- 🔍 In-Memory ChromaDB
- 🤗 HuggingFace Embeddings
- 🔎 MMR Retrieval
""")

    st.markdown("---")

    # Upload PDFs
    uploaded_files = st.file_uploader(
        "📂 Upload PDF(s)",
        type=["pdf"],
        accept_multiple_files=True
    )

    # Process PDFs
    if st.button("⚙️ Process PDFs"):

        if uploaded_files:

            with st.spinner("Processing PDFs..."):

                os.makedirs("uploads", exist_ok=True)

                pdf_paths = []

                for file in uploaded_files:

                    save_path = os.path.join(
                        "uploads",
                        file.name
                    )

                    with open(save_path, "wb") as f:
                        f.write(file.getbuffer())

                    pdf_paths.append(save_path)

                pipeline.load_documents(pdf_paths)

            st.success("Documents processed successfully!")

        else:
            st.warning("Please upload at least one PDF.")

    st.markdown("---")

    # Clear chat
    if st.button("🗑 Clear Chat"):
        st.session_state.messages = []
        st.rerun()

    # Reset documents
    if st.button("🔄 Reset Documents"):

        st.session_state.pipeline = RAGPipeline()
        st.session_state.messages = []

        st.rerun()

# Main title
st.title("🤖 PlacementPrep AI")

st.caption(
     "Chat with your documents using RAG + Groq + Llama 3.3 70B"
)

# Display previous messages
for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_input = st.chat_input(
    "Ask anything about your documents..."
)

if user_input:

    # Check if PDFs are processed
    if pipeline.vectorstore is None:

        st.warning(
            "Please upload and process PDF documents first."
        )

        st.stop()

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

    # Generate answer
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