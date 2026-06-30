import streamlit as st
from langchain_groq import ChatGroq


@st.cache_resource
def get_llm():
    llm = ChatGroq(
        model="llama-3.3-70b-versatile", tags=["PlacementPrepAI"], temperature=0
    )

    return llm
