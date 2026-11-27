"""Text extraction from uploaded files.

Supported formats: .txt, .md, .pdf, .docx.
Each extractor returns plain text (empty string when nothing readable
was found). Unknown extensions raise ValueError.
"""

import os

from pypdf import PdfReader


def extract_text(filename: str, content: bytes) -> str:
    """Extract plain text from a file's bytes, based on its extension."""
    ext = os.path.splitext(filename)[1].lower()
    if ext in (".txt", ".md"):
        return content.decode("utf-8", errors="ignore")
    if ext == ".pdf":
        return _extract_pdf(content)
    if ext == ".docx":
        return _extract_docx(content)
    raise ValueError(f"Unsupported file type: {ext or '(no extension)'}. "
                     "Upload a .pdf, .txt, .md or .docx file.")


def _extract_pdf(content: bytes) -> str:
    import io

    reader = PdfReader(io.BytesIO(content))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def _extract_docx(content: bytes) -> str:
    import io

    from docx import Document

    doc = Document(io.BytesIO(content))
    return "\n".join(paragraph.text for paragraph in doc.paragraphs)
