from src.llm import get_llm


def analyze_skill_gap(context):

    llm = get_llm()

    prompt = f"""
You are an expert technical recruiter.

The uploaded documents contain a candidate's resume and one or more job descriptions.

Analyze them and provide:

1. Match Score (out of 100)

2. Strong Skills

3. Missing Skills

4. Skills that should be prioritized

5. Recommendations

6. Overall Verdict

Document Context:

{context}
"""

    response = llm.invoke(prompt)

    return response.content