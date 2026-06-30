from src.llm import get_llm


def generate_roadmap(context, goal):

    llm = get_llm()

    prompt = f"""
You are an expert career mentor and placement coach.

Using the candidate information below, create a detailed roadmap.

Goal:
{goal}

Provide:

1. Current strengths
2. Weaknesses
3. Skills to learn
4. Monthly roadmap
5. Recommended projects
6. Resources and preparation strategy

Candidate Information:

{context}

Return the roadmap in a structured format.
"""

    response = llm.invoke(prompt)

    return response.content
