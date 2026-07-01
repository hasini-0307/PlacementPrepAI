from types import SimpleNamespace
from src.context_guard import has_sufficient_context


def make_doc(text):
    return SimpleNamespace(page_content=text)


def test_empty_docs():
    assert has_sufficient_context([]) is False


def test_one_valid_doc():
    docs = [make_doc("A" * 100)]
    assert has_sufficient_context(docs) is False


def test_two_valid_docs():
    docs = [make_doc("A" * 100), make_doc("B" * 100)]
    assert has_sufficient_context(docs) is True


def test_one_valid_one_invalid():
    docs = [make_doc("A" * 100), make_doc("short")]
    assert has_sufficient_context(docs) is False


def test_many_docs_two_valid():
    docs = [
        make_doc("A" * 100),
        make_doc("B" * 100),
        make_doc("small"),
        make_doc(""),
        make_doc("abc"),
    ]
    assert has_sufficient_context(docs) is True


def test_whitespace_docs():
    docs = [make_doc(" " * 100), make_doc("A" * 100)]
    assert has_sufficient_context(docs) is False
