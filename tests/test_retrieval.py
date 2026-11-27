"""Tests for retrieval ranking (app/store.py).

A fake embedding function replaces the real OpenAI embeddings:
each text becomes a 3-number vector counting the words
"cat", "dog" and "mars". That makes ranking fully deterministic.
"""

import pytest

from app.store import VectorStore


def fake_embed(texts):
    """Deterministic keyword-count embeddings for tests only."""
    vectors = []
    for text in texts:
        words = text.lower().split()
        vectors.append([
            float(sum(1 for w in words if "cat" in w)),
            float(sum(1 for w in words if "dog" in w)),
            float(sum(1 for w in words if "mars" in w)),
        ])
    return vectors


@pytest.fixture
def store(tmp_path):
    return VectorStore(path=str(tmp_path / "store.json"))


def test_search_ranks_most_similar_chunk_first(store):
    chunks = [
        "cats are cute animals",   # cat-heavy
        "dogs are loyal animals",  # dog-heavy
        "mars is a red planet",    # mars-heavy
    ]
    store.add_document("animals.txt", chunks, fake_embed(chunks))

    results = store.search(fake_embed(["tell me about cats"])[0], top_k=3)

    assert results[0]["text"] == "cats are cute animals"
    assert results[0]["score"] > results[1]["score"]
    assert results[0]["filename"] == "animals.txt"


def test_search_returns_top_k_only(store):
    chunks = ["cat"] * 5
    store.add_document("a.txt", chunks, fake_embed(chunks))
    results = store.search(fake_embed(["cat"])[0], top_k=2)
    assert len(results) == 2


def test_search_on_empty_store_returns_empty_list(store):
    assert store.search(fake_embed(["cat"])[0]) == []


def test_cosine_similarity_identical_vectors():
    assert VectorStore.cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)


def test_cosine_similarity_orthogonal_vectors():
    assert VectorStore.cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)


def test_cosine_similarity_zero_vector():
    assert VectorStore.cosine_similarity([0, 0], [1, 2]) == 0.0


def test_delete_document_removes_its_chunks(store):
    chunks = ["cats are cute", "dogs are loyal"]
    doc_id = store.add_document("a.txt", chunks, fake_embed(chunks))
    assert store.delete_document(doc_id) is True
    assert store.list_documents() == []
    assert store.search(fake_embed(["cats"])[0]) == []
    assert store.delete_document(doc_id) is False  # already gone
