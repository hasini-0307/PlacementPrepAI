from src.llm import get_llm


def generate_interview(context, role):

    llm = get_llm()

    prompt = f"""
You are an experienced technical interviewer.

Candidate information:

{context}

Target role:

{role}

Generate:

1. 3 Behavioral questions
2. 5 Technical questions
3. 2 Project-based questions

Return only the questions in numbered format with headings of which category they belong to.
"""

    response = llm.invoke(prompt)

    return response.content
