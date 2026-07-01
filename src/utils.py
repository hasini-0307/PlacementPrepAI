import re


def normalize_question(question: str) -> str:
    question = question.lower()
    question = question.strip()
    question = re.sub(r"[^\w\s]", "", question)
    return question