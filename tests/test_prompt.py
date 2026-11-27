"""Tests for RAG prompt construction (app/llm.py)."""

from app.llm import build_chat_messages


def test_prompt_includes_context_and_question():
    chunks = [{"filename": "space.txt", "text": "Mars is a red planet."}]
    messages = build_chat_messages("What color is Mars?", chunks)

    user_content = messages[-1]["content"]
    assert "Mars is a red planet." in user_content  # the context
    assert "space.txt" in user_content              # the source name
    assert "What color is Mars?" in user_content    # the question


def test_prompt_instructs_to_answer_only_from_context():
    messages = build_chat_messages("anything", [])
    system_content = messages[0]["content"]
    assert messages[0]["role"] == "system"
    assert "only" in system_content.lower()
    assert "context" in system_content.lower()


def test_prompt_instructs_to_say_it_does_not_know():
    messages = build_chat_messages("anything", [])
    system_content = messages[0]["content"].lower()
    assert "don't know" in system_content


def test_prompt_keeps_all_chunks():
    chunks = [
        {"filename": "a.txt", "text": "first fact"},
        {"filename": "b.txt", "text": "second fact"},
    ]
    messages = build_chat_messages("question?", chunks)
    user_content = messages[-1]["content"]
    assert "first fact" in user_content
    assert "second fact" in user_content
