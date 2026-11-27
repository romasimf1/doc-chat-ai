"""Text chunking for the RAG pipeline.

Documents are split into overlapping chunks of roughly CHUNK_SIZE
characters. Overlap keeps context from being lost at chunk borders:
a sentence that straddles two chunks still appears whole in one of them.
"""

from app import config


def chunk_text(text: str, chunk_size: int = config.CHUNK_SIZE,
               overlap: int = config.CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks.

    - Empty or whitespace-only input returns an empty list.
    - Chunks break at word boundaries (never in the middle of a word)
      unless a single word is longer than the chunk size.
    - Each chunk (after the first) starts `overlap` characters before
      the previous chunk ended.
    """
    if not text or not text.strip():
        return []

    # Normalise whitespace: collapse runs of spaces/newlines into one space
    # so chunk sizes are predictable.
    text = " ".join(text.split())
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:])
            break
        # Move the cut back to the nearest space so we don't split a word.
        cut = text.rfind(" ", start, end)
        if cut <= start:
            cut = end  # No space found: one very long word, cut mid-word.
        chunks.append(text[start:cut])
        # Next chunk starts `overlap` chars before this one ended, so
        # context survives the border. Never move backwards (can happen
        # when overlap is almost as large as the chunk size).
        new_start = cut + 1 - overlap
        start = new_start if new_start > start else cut + 1
        # Snap forward to the next word boundary so a chunk never
        # starts in the middle of a word.
        if start < len(text) and text[start] != " " and text[start - 1] != " ":
            nxt = text.find(" ", start)
            if nxt != -1:
                start = nxt + 1
        # Skip a leading space, if any.
        while start < len(text) and text[start] == " ":
            start += 1
    return chunks
