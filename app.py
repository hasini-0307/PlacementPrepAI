
import os
import shutil
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

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
- Hybrid Search + MultiQuery
""")

    st.markdown("---")

    st.markdown("""
### About

Ask questions about your uploaded documents using:

- 📄 Semantic Search
- 🧠 Llama 3.3 70B (Groq)
- 🔍 In-Memory ChromaDB
- 🤗 HuggingFace Embeddings
- 🔎 MultiQuery Retrieval
- 🔀 Hybrid Search
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


    #ats score analyzer
    if st.button("📊 ATS Analysis"):

     if pipeline.vectorstore is None:

        st.warning(
            "Please upload documents first."
        )

     else:

        with st.spinner("Analyzing resume..."):

            report = pipeline.ats_analysis()

        st.subheader("📊 ATS Report")

        st.markdown(report)


     # skill gap analyzer 
    if st.button("🎯 Skill Gap Analysis"):

     if pipeline.vectorstore is None:

        st.warning(
            "Please upload documents first."
        )

     else:

        with st.spinner(
            "Analyzing candidate and job requirements..."
        ):

            report = pipeline.skill_gap_analysis()

        st.subheader(
            "🎯 Skill Gap Report"
        )

        st.markdown(report) 


        #roadmap generator 
          
    roadmap_goal = st.text_input(
    "🎯 Career Goal",
    placeholder="Amazon SDE / ML Engineer / Google SWE..."
)

    if st.button("🛣 Generate Roadmap"):

     if pipeline.vectorstore is None:

        st.warning(
            "Please upload documents first."
        )

     elif roadmap_goal == "":

        st.warning(
            "Enter a career goal."
        )

     else:

        with st.spinner(
            "Generating roadmap..."
        ):

            report = pipeline.roadmap(
                roadmap_goal
            )

        st.subheader(
            "🛣 Personalized Roadmap"
        )

        st.markdown(report)

        #interviw questions generator

    st.markdown("---")

    st.subheader("🎤 Mock Interview")

    interview_role = st.text_input(
    "Target Role",
    placeholder="Amazon SDE / ML Engineer / Google SWE"
)

    if st.button("🎤 Generate Interview"):

     if pipeline.vectorstore is None:

        st.warning(
            "Please upload documents first."
        )

     elif interview_role == "":

        st.warning(
            "Please enter a role."
        )

     else:

        with st.spinner(
            "Generating interview questions..."
        ):

            report = pipeline.interview_questions(
                interview_role
            )

        st.subheader(
            "🎤 Mock Interview"
        )

        st.markdown(report)    


    # Clear Chat
    if st.button("🗑 Clear Chat"):

        st.session_state.messages = []

        pipeline.chat_history.clear()

        st.rerun()

    # Reset Session
    if st.button("🔄 Reset Session"):

        st.session_state.pipeline = RAGPipeline()

        st.session_state.messages = []

        st.rerun()

    # Delete Everything
    if st.button("❌ Delete Everything"):

        st.session_state.pipeline = RAGPipeline()

        st.session_state.messages = []

        if os.path.exists("uploads"):
            shutil.rmtree("uploads")

        st.success("All documents and chat history deleted.")

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

    # Check if documents are loaded
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

    # Assistant response
    with st.chat_message("assistant"):

        response, pages = pipeline.ask(user_input)

        answer = st.write_stream(
            chunk.content
            for chunk in response
        )

        # Store memory
        pipeline.chat_history.add_message(
            HumanMessage(content=user_input)
        )

        pipeline.chat_history.add_message(
            AIMessage(content=answer)
        )

        # Show sources
        if pages:

            with st.expander("📚 Sources"):

                for page in pages:

                    st.write(f"Page {page}")

    # Save assistant response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

