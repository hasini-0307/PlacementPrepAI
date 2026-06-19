from src.llm import get_llm


def analyze_resume(context):

    llm = get_llm()

    prompt = f"""
You are an experienced technical recruiter and ATS evaluator.

Analyze the candidate based on the resume context below.

Provide:

1. Technical Skills score (out of 10)
2. Projects score
3. Experience score
4. Leadership score
5. Academics score
6. Overall ATS score

Then provide:

Strengths
Weaknesses
Recommendations

Resume Context:

{context}
"""

    response = llm.invoke(prompt)

    return response.content