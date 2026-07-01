from src.utils import normalize_question


def test_normalize_question():

    assert normalize_question("Who is the Candidate?") == "who is the candidate"


def test_normalize_spaces():

    assert normalize_question("   Hello World   ") == "hello world"


def test_remove_punctuation():

    assert normalize_question("Hello!!!") == "hello"
