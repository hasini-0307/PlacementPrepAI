# src/interview_agent.py

import re
from src.llm import get_llm


def start_interview(context, role):

    llm = get_llm()

    prompt = f"""
You are an experienced technical interviewer.

Candidate information:
{context}

Target role:
{role}

Ask ONE interview question suitable for this role.

Rules:
- Ask only one question.
- Do not provide explanations.
- Do not provide answers.
- Output only the question.
"""

    response = llm.invoke(prompt)

    return response.content.strip()


def evaluate_and_continue(
        role,
        previous_question,
        answer,
        history
):

    llm = get_llm()

    prompt = f"""
You are conducting a technical interview for the role:

{role}

Previous Interview History:

{history}

Previous Question:

{previous_question}

Candidate Answer:

{answer}

Evaluate the answer carefully.

Provide:

1. Communication Score (/10)
2. Technical Depth Score (/10)
3. Strengths
4. Weaknesses
5. Suggestions

Then ask ONE NEW follow-up question.

IMPORTANT RULES:
- NEVER repeat any previous question.
- Use the candidate's answer to decide the next question.
- Increase difficulty gradually.
- Ask only one question.
- Do not ask generic questions unless necessary.

IMPORTANT:
- NEVER repeat any question from the interview history.
- Use the candidate's answer to determine the next question.
- If the answer mentions a concept, ask deeper about that concept.
- Increase difficulty gradually.
- Ask only one question.

Format EXACTLY:

Communication Score:
...

Technical Depth Score:
...

Strengths:
...

Weaknesses:
...

Suggestions:
...

Next Question:
...
"""

    response = llm.invoke(prompt)

    text = response.content.strip()

    # -------- Extract Next Question --------

    match = re.search(
        r"Next Question:\s*(.*)",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )

    if match:

        next_question = match.group(1).strip()

        feedback = re.sub(
            r"Next Question:.*",
            "",
            text,
            flags=re.DOTALL | re.IGNORECASE
        ).strip()

    else:

        feedback = text

        next_question = (
            "Can you describe a challenging project you worked on and explain how you solved the problems you faced?"
        )

    # Prevent empty next question
    if len(next_question) < 5:

        next_question = (
            "Can you explain one technical challenge you faced in a project and how you solved it?"
        )

    return {
        "feedback": feedback,
        "next_question": next_question
    }   