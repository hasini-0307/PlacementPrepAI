import os
import shutil
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from src.logger import logger
from src.interview_state import get_interview_state
from src.pdf_generator import create_pdf

if "interview_state" not in st.session_state:
    st.session_state.interview_state = get_interview_state()


# Load environment variables
load_dotenv()


logger.info("=" * 60)
logger.info("PlacementPrepAI Started")
logger.info("=" * 60)

from src.rag_pipeline import RAGPipeline

# Initialize pipeline once
if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline()


pipeline = st.session_state["pipeline"]
logger.info("RAG Pipeline initialized")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Page config
st.set_page_config(page_title="PlacementPrep AI", page_icon="🤖", layout="wide")

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
        "📂 Upload PDF(s)", type=["pdf"], accept_multiple_files=True
    )

    # Process PDFs
    if st.button("⚙️ Process PDFs"):

        if uploaded_files:

            with st.spinner("Processing PDFs..."):

                os.makedirs("uploads", exist_ok=True)

                pdf_paths = []

                for file in uploaded_files:

                    save_path = os.path.join("uploads", file.name)

                    with open(save_path, "wb") as f:
                        f.write(file.getbuffer())

                    pdf_paths.append(save_path)

                pipeline.load_documents(pdf_paths)
                logger.info("Processed %d uploaded PDF(s)", len(uploaded_files))

            st.success("Documents processed successfully!")

        else:
            st.warning("Please upload at least one PDF.")

    st.markdown("---")

    # ats score analyzer
    if st.button("📊 ATS Analysis"):

        if pipeline.vectorstore is None:

            st.warning("Please upload documents first.")

        else:

            with st.spinner("Analyzing resume..."):

                report = pipeline.ats_analysis()

            st.session_state["ats_report"] = report

        # skill gap analyzer
    if st.button("🎯 Skill Gap Analysis"):

        if pipeline.vectorstore is None:

            st.warning("Please upload documents first.")

        else:

            with st.spinner("Analyzing candidate and job requirements..."):

                report = pipeline.skill_gap_analysis()

            st.session_state["skill_gap_report"] = report

            # roadmap generator

    roadmap_goal = st.text_input(
        "🎯 Career Goal", placeholder="Amazon SDE / ML Engineer / Google SWE..."
    )

    if st.button("🛣 Generate Roadmap"):

        if pipeline.vectorstore is None:

            st.warning("Please upload documents first.")

        elif roadmap_goal == "":

            st.warning("Enter a career goal.")

        else:

            with st.spinner("Generating roadmap..."):

                report = pipeline.roadmap(roadmap_goal)

            st.session_state["roadmap_report"] = report

    # Dynamic Interview
    st.markdown("---")
    st.subheader("🎤 Dynamic Interview")

    interview_role = st.text_input("Target Role", placeholder="Amazon SDE")

    if st.button("🎤 Start Interview"):

        if pipeline.vectorstore is None:

            st.warning("Please upload documents first.")

        elif interview_role == "":

            st.warning("Please enter a role.")

        else:

            question = pipeline.start_interview(interview_role)

            st.session_state.interview_state["active"] = True
            st.session_state.interview_state["role"] = interview_role

            st.session_state.interview_state["current_question"] = question
            st.session_state.interview_state["history"] = []

            st.rerun()

    st.markdown("---")

    # Clear chat
    if st.button("🗑 Clear Chat"):

        st.session_state.messages = []

        pipeline.chat_history.clear()

        st.rerun()

    # Reset session
    if st.button("🔄 Reset Session"):

        st.session_state.pipeline = RAGPipeline()
        st.session_state.messages = []
        st.session_state.interview_state = get_interview_state()

        st.session_state.pop("ats_report", None)
        st.session_state.pop("skill_gap_report", None)
        st.session_state.pop("roadmap_report", None)

        st.rerun()

    # Delete everything
    if st.button("❌ Delete Everything"):

        st.session_state.pipeline = RAGPipeline()
        st.session_state.messages = []
        st.session_state.interview_state = get_interview_state()

        if os.path.exists("uploads"):
            shutil.rmtree("uploads")

        if os.path.exists("chroma_db"):
            shutil.rmtree("chroma_db")

        st.success("All documents deleted.")

        st.rerun()


# Main title
st.title("🤖 PlacementPrep AI")

st.caption("Chat with your documents using RAG + Groq + Llama 3.3 70B")

tab1, tab2, tab3, tab4, tab5 = st.tabs(
    ["💬 Chat", "🎤 Interview", "📊 ATS", "🎯 Skill Gap", "🛣 Roadmap"]
)

with tab3:

    st.subheader("📊 ATS Report")
    st.caption("Analyze resume quality and ATS compatibility")

    if "ats_report" in st.session_state:

        st.markdown(st.session_state["ats_report"])

        pdf = create_pdf("ATS Report", st.session_state["ats_report"])

        st.download_button(
            "⬇ Download PDF",
            data=pdf,
            file_name="ATS_Report.pdf",
            mime="application/pdf",
            key="ats_pdf",
        )

    else:

        st.info("Run ATS Analysis from sidebar.")


with tab4:

    st.subheader("🎯 Skill Gap Report")

    if "skill_gap_report" in st.session_state:

        st.markdown(st.session_state["skill_gap_report"])

        pdf = create_pdf("Skill Gap Report", st.session_state["skill_gap_report"])

        st.download_button(
            "⬇ Download PDF",
            data=pdf,
            file_name="Skill_Gap_Report.pdf",
            mime="application/pdf",
            key="skill_gap_pdf",
        )

    else:

        st.info("Run Skill Gap Analysis from sidebar.")

with tab5:

    st.subheader("🛣 Personalized Roadmap")

    if "roadmap_report" in st.session_state:

        st.markdown(st.session_state["roadmap_report"])

        pdf = create_pdf("Roadmap", st.session_state["roadmap_report"])

        st.download_button(
            "⬇ Download PDF",
            data=pdf,
            file_name="Roadmap.pdf",
            mime="application/pdf",
            key="roadmap_pdf",
        )

    else:

        st.info("Generate roadmap from sidebar.")


with tab2:

    if st.session_state.interview_state["active"]:

        st.markdown("---")
        st.subheader("🎤 Live Interview")

        # Previous rounds
        for item in st.session_state.interview_state["history"]:

            st.markdown("### Question")
            st.info(item["question"])

            st.markdown("### Your Answer")
            st.write(item["answer"])

            st.markdown("### Feedback")
            st.success(item["feedback"])

            st.markdown("---")

        # Current question
        current_question = st.session_state.interview_state["current_question"]

        st.markdown("### Current Question")
        st.info(current_question)
        if st.button("🛑 End Interview"):

            st.session_state.interview_state = get_interview_state()

            st.rerun()

        interview_answer = st.text_area(
            "Your Answer",
            key=f"interview_answer_{len(st.session_state.interview_state['history'])}",
        )

        if st.button("Submit Answer"):

            result = pipeline.continue_interview(
                st.session_state.interview_state["role"],
                current_question,
                interview_answer,
                st.session_state.interview_state["history"],
            )

            st.session_state.interview_state["history"].append(
                {
                    "question": current_question,
                    "answer": interview_answer,
                    "feedback": result["feedback"],
                }
            )

            st.session_state.interview_state["current_question"] = result[
                "next_question"
            ]

            st.rerun()
    else:
        st.info("Start an interview from the sidebar.")


with tab1:
    st.subheader("💬 Chat Assistant")
    # Display previous messages
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):

            st.markdown(message["content"])
    # Chat input
    user_input = st.chat_input("Ask anything about your documents...")

    if user_input:

        # Check if documents are loaded
        if pipeline.vectorstore is None:

            st.warning("Please upload and process PDF documents first.")

            st.stop()

        # Save user message
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Display user message
        with st.chat_message("user"):

            st.markdown(user_input)

        # Assistant response
        with st.chat_message("assistant"):

            logger.info("New Question: %s", user_input)

            response, pages, metadata = pipeline.ask(user_input)
            logger.info("Response generated successfully")

        if isinstance(response, str):

            answer = response
            st.markdown(answer)

        else:

            if isinstance(response, str):
                answer = response
                st.write(answer)
            else:
                answer = st.write_stream(chunk.content for chunk in response)
            score = metadata["max_score"]

            if score > -6.5:
                confidence = "🟢 High"

            elif score > -11:
                confidence = "🟡 Medium"

            else:
                confidence = "🔴 Low"

            # Store memory
            pipeline.chat_history.add_message(HumanMessage(content=user_input))

            pipeline.chat_history.add_message(AIMessage(content=answer))

            # Show sources
            with st.expander("📚 Sources & Retrieval Info"):

                st.markdown(f"**Confidence:** {confidence}")

                st.markdown(f"**Max Score:** {metadata['max_score']:.2f}")
                st.markdown(f"**Chunks Retrieved:** {metadata['num_chunks']}")

                st.markdown(f"**Pages Used:** {', '.join(map(str, metadata['pages']))}")

                st.markdown(f"**Retrieval:** {metadata['retrieval']}")

                st.markdown(f"**Re-ranking:** {metadata['reranker']}")

                st.markdown(f"**Average Relevance Score:** {metadata['avg_score']:.2f}")
                # Save assistant response
        st.session_state.messages.append({"role": "assistant", "content": answer})
