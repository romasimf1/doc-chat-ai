"""Tests for text chunking (app/chunking.py)."""

from app.chunking import chunk_text


def test_empty_input_returns_empty_list():
    assert chunk_text("") == []
    assert chunk_text("   \n\t  ") == []


def test_short_text_returns_single_chunk():
    assert chunk_text("hello world") == ["hello world"]


def test_chunks_do_not_exceed_chunk_size():
    text = " ".join(f"word{i}" for i in range(200))
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk) <= 100


def test_chunks_overlap():
    """The start of one chunk must repeat the end of the previous one."""
    text = " ".join(f"word{i}" for i in range(200))
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    # Chunk 1 starts at most `overlap` chars before chunk 0 ended, so
    # its first characters must appear inside chunk 0.
    assert chunks[1][:10] in chunks[0]


def test_no_text_is_lost():
    """Every word of the input must appear in at least one chunk."""
    words = [f"word{i}" for i in range(200)]
    text = " ".join(words)
    chunks = chunk_text(text, chunk_size=100, overlap=20)
    joined = " ".join(chunks)
    for word in words:
        assert word in joined


def test_chunks_never_split_words():
    text = " ".join(["abcdefghij"] * 50)  # 10-letter words, chunk of 25
    chunks = chunk_text(text, chunk_size=25, overlap=5)
    for chunk in chunks:
        # No partial word: every piece is a whole word (or words).
        for piece in chunk.split(" "):
            assert piece == "abcdefghij"
